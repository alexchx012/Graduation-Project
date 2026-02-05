# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class PPO:
    """近端策略优化算法 (Proximal Policy Optimization, https://arxiv.org/abs/1707.06347)"""

    policy: ActorCritic | ActorCriticRecurrent
    """Actor-Critic 模块（策略网络+价值网络）"""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent,  #ActorCritic是标准前馈网络，ActorCriticRecurrent是循环神经网络。
        #policy对象本质上是一个Actor-Critic复合模块,它同时包含:Actor网络（输出动作π_θ(a|s)）和Critic网络（输出状态价值V_θ(s)）。
        #放在一起的好处是Actor和Critic可以共享前几层网络,提取相同的状态特征。使用同一个优化器更新所有参数。
        #总之policy是一个包含了两个子网络的复合模块。对外统一接口：通过act()和evaluate()方法分别访问Actor和Critic。对内结构分离：Actor和Critic有各自独立的前向计算逻辑。
        #同时还能使用同一个优化器，根据总损失函数同时更新两部分参数。

        num_learning_epochs: int = 5,          #对同一批采样数据重复训练多少轮，算法伪代码中的K
        num_mini_batches: int = 4,             #每轮训练将数据分成多少个小批量，算法伪代码中的M
        # N (并行环境数) = 4096，T (每个环境采样步数) = 24，总样本数 NT = 4096 × 24 = 98,304
        # num_learning_epochs=5：对这98,304条样本训练5轮  num_mini_batches=4： 每轮分成4个批次，每批次 M = 98,304/4 = 24,576条样本。总更新次数 = 5 × 4 = 20次梯度更新

        clip_param: float = 0.2,               #截断系数，PPO-Clip的核心，通常设置为0.2
        gamma: float = 0.99,                   #折扣因子，控制未来奖励的权重，通常设置为0.99
        lam: float = 0.95,                     #GAE的平滑参数，控制优势估计的偏差-方差权衡，通常设置为0.95
        value_loss_coef: float = 1.0,          #c1，总目标函数中L^{VF}的权重系数。
        entropy_coef: float = 0.01,            #c2，总目标函数中熵奖励S的权重系数（注意loss代码中通常是减去熵损失，即加上熵奖励）。
        learning_rate: float = 0.001,          #优化器的学习率
        max_grad_norm: float = 1.0,            #梯度裁剪阈值，防止梯度爆炸。
        use_clipped_value_loss: bool = True,   #是否使用截断的价值函数损失。标准 PPO 只有策略 Clip，但此处允许对价值函数也进行 Clip（防止V(s)更新太猛）
        schedule: str = "adaptive",            #学习率调度方式，可选 "constant"（恒定）或 "adaptive"（自适应）
        desired_kl: float = 0.01,              #d_targ，自适应KL的目标值。如果开启自适应模式，用于动态调整学习率或惩罚系数。
        device: str = "cpu",                   #模型和数据所在设备，"cpu"或"cuda"
        normalize_advantage_per_mini_batch: bool = False,    #是否对每个mini-batch单独做优势归一化
        # RND parameters
        rnd_cfg: dict | None = None,           #RND (Random Network Distillation) 配置字典，用于好奇心驱动探索：给agent内在奖励鼓励探索新状态
        # Symmetry parameters
        symmetry_cfg: dict | None = None,      #对称性数据增强配置字典，用于机器人左右对称任务（如双足行走），通过镜像数据增加样本多样性
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,     #多GPU分布式训练配置字典，包含global_rank（GPU编号）和world_size（总GPU数）
    ) -> None:
        # 设备相关参数
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # 多GPU参数
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND（随机网络蒸馏）组件，用于内在奖励/好奇心驱动探索      用于探索稀疏奖励环境（Intrinsic Rewards），会在计算奖励时给r_t加上额外的“好奇心奖励”。
        if rnd_cfg is not None:
            # 提取PPO中使用的参数
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # 创建RND模块
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # 创建RND优化器
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # 对称性组件（用于数据增强或镜像损失）         Symmetry：利用机器人的形态对称性（如左腿动作和右腿动作的镜像关系）来增强数据量或约束策略，加速训练。
        if symmetry_cfg is not None:
            # 检查是否启用对称性
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # 如果未启用对称性则打印提示
            if not use_symmetry:
                print("对称性未用于学习，仅用于日志记录。")
            # 如果函数是字符串则解析为可调用函数
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # 检查配置是否有效
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"对称性配置存在但函数不可调用: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # 检查策略是否与对称性兼容
            if isinstance(policy, ActorCriticRecurrent):
                raise ValueError("循环策略不支持对称性增强。")
            # 存储对称性配置
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO组件
        self.policy = policy
        self.policy.to(self.device)

        # 创建优化器（Adam优化器用于一阶梯度优化）
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        #这里的self.policy.parameters()包含了Actor网络的参数: θ_actor；Critic网络的参数: θ_critic
        #它们共同组成完整的参数集θ,在训练时同时更新

        # 创建轨迹存储器（用于存储采样数据）
        self.storage: RolloutStorage | None = None        #self.storage将用于存储一整个batch（如几千步）的数据
        self.transition = RolloutStorage.Transition()     #self.transition用于临时存储单步交互数据

        # PPO超参数
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        # 创建轨迹存储器
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """根据观测生成动作（对应伪代码第3行：用旧策略在环境里跑T步）"""
        # transition 对象：这就是每一步产生的各自的数据。它暂存了(s_t, a_t, V(s_t), \log \pi(a_t|s_t))。

        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        # 计算动作a_t和价值V(s_t)（Actor输出动作，Critic输出价值估计）
        self.transition.actions = self.policy.act(obs).detach()       # 使用policy.act()获取动作 (调用Actor)
        self.transition.values = self.policy.evaluate(obs).detach()   # 使用policy.evaluate()获取价值 (调用Critic)
        #.detach()：数据收集阶段不需要计算梯度，只存数据（Tensor），所以切断计算图，节省显存。

        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        #计算动作的对数概率，虽然PPO公式中用的是概率比率r_t(θ)，但实际计算时使用对数概率更稳定

        #记录动作分布的均值和标准差（用于后续的梯度计算）
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        # 在env.step()之前记录观测s_t
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict,     # 下一个状态 s_{t+1}
        rewards: torch.Tensor,     # 奖励 r_t（环境反馈的标量奖励）
        dones: torch.Tensor,       # 终止标志 d_t（布尔值，True表示回合结束）
        extras: dict[str, torch.Tensor]
    ) -> None:
        """处理环境交互结果，收集transition数据（对应数据收集流程）"""
        #此函数在环境执行 env.step() 后调用，对应笔记 Sec 2 中的"奖励"和"下一步观测"。

        # 更新归一化器
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # 记录奖励r_t和终止标志d_t
        # 注意：这里克隆是因为后续会根据超时进行bootstrap
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # 计算内在奖励并加到外在奖励上（RND好奇心驱动）
        # 如果开启 RND，r_t = r_extrinsic + r_intrinsic
        if self.rnd:
            # 计算内在奖励
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # 将内在奖励加到外在奖励上
            self.transition.rewards += self.intrinsic_rewards

        # 超时情况下的Bootstrap（用γ*V(s)补偿被截断的未来奖励）
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # 记录transition到存储器
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """计算回报和优势估计（对应伪代码第4行：计算优势Â_t）
        
        使用GAE（广义优势估计）计算优势函数：
        Â_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t+1}δ_{T-1}
        其中TD残差 δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        # 计算最后一步的价值估计（用于GAE计算）
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND损失
        mean_rnd_loss = 0 if self.rnd else None
        # 对称性损失
        mean_symmetry_loss = 0 if self.symmetry else None

        # 获取小批量生成器
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 迭代所有小批量
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1  # 每个样本的增强次数。默认1表示不做增强。
            original_batch_size = obs_batch.batch_size[0]

            # 检查是否需要对每个小批量做优势归一化
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # 执行对称性数据增强
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # 使用对称性进行增强
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # 返回形状: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # 计算每个样本的增强次数
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # 其余批量数据按增强次数重复
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # 基于当前参数重新计算动作对数概率与熵
            # 注意：因为策略参数已更新，需要重新前向计算
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # 注意：只保留第一个增强（原始样本）的熵
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # 计算KL散度并自适应调整学习率
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # 在所有GPU上汇总KL散度
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # 仅在主进程更新学习率
                    # TODO：是否需要？若KL在各GPU上相同，学习率也应一致。
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # 在所有GPU上同步学习率
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # 更新所有参数组的学习率
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # 代理目标损失（PPO-Clip）
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))   # 计算概率比率
                         #exp操作把对数概率转回比率
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 价值函数损失
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # 对称性损失
            if self.symmetry:
                # 获取对称动作
                # 注意：若前面已做增强则无需再次增强
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # 计算每个样本的增强次数
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actor在对称增强观测上的预测动作
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # 计算对称增强后的动作
                # 注意：假设第一个增强是原始样本。这里不使用之前采样的动作，
                #      因为对称损失使用的是动作分布的均值。
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # 计算损失
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # 将损失加到总损失
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND损失
            # TODO：将这部分处理移到RND模块内部。
            if self.rnd:
                # 提取rnd_state
                # TODO：确认是否仍需要torch no grad；这里只是仿射变换。
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # 预测嵌入与目标
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # 用均方误差计算损失
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # 计算PPO梯度
            self.optimizer.zero_grad()
            loss.backward()
            # 计算RND梯度
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # 汇总多GPU梯度
            if self.is_multi_gpu:
                self.reduce_parameters()

            # 应用PPO梯度
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # 应用RND梯度
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # 累计损失
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND损失
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # 对称性损失
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # 按更新次数做平均
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # 清空存储器
        self.storage.clear()

        # 构造损失字典
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def broadcast_parameters(self) -> None:
        """将模型参数广播到所有GPU。"""
        # 获取当前GPU上的模型参数
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # 广播模型参数
        torch.distributed.broadcast_object_list(model_params, src=0)
        # 从源GPU加载模型参数到所有GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """收集所有GPU的梯度并求平均。

        该函数在反向传播之后调用，用于同步所有GPU的梯度。
        """
        # 创建张量用于存放梯度
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # 在所有GPU上对梯度求平均
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # 获取所有参数
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # 用汇总后的梯度更新所有参数的梯度
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # 从共享缓冲区复制数据
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # 更新下一个参数的偏移量
                offset += numel