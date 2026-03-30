# MORL Repair Reopen Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 基于阶段三真实设计过程和后续失效证据，重开 MORL 修复方案，目标是不再把“4 个通道都亮了”误当成“策略学会了走”，而是重建一个以 baseline locomotion prior 为硬约束、以多目标 trade-off 为次层变化的 MORL v2 训练方案。

**Architecture:** 先复盘阶段三的真实设计目标与演化，再将当前问题重新定义为“reward architecture 失衡”，不是“单个 channel scale 没调对”。修复方案不再优先追加 `Pilot C`，而是先恢复一个不可削弱的 locomotion scaffold，再把 energy / smooth / stable 作为在该 scaffold 之上的受控目标，最后用固定命令 `S1` 从训练早期开始验收。

**Tech Stack:** Isaac Lab, RSL-RL, PowerShell, `env_isaaclab`, MORL reward/env config, `run_morl_eval.py`

## 1. 阶段三到底是怎么设计出来的

### 1.1 初始设计目标

[Fact] 阶段三一开始的设计并不是“只保留 4 个 reward term”，而是“4 个主目标 + baseline 辅助约束保留”。  
证据：`docs/daily_logs/2026-3/2026-03-14/phase_morl_plan.md`

[Fact] 当时的明确原则是：

- 不整体替换 baseline `rewards`
- 通过 `RewardTermCfg.weight` 直接做线性标量化
- 保留 `track_ang_vel_z_exp / lin_vel_z_l2 / ang_vel_xy_l2 / dof_acc_l2 / feet_air_time / termination`
- 把 `dof_torques_l2` 和 `action_rate_l2` 从 baseline 固定项中移出，分别由 energy 和 smooth 主目标替代

[Inference + Evidence] 阶段三设计时真正优先保证的是：

- 与 baseline 的工程连续性
- 4 个研究目标在训练中“都有通道”
- Pareto 分析可以在 4 个物理指标上开展

而不是“任何权重点下都必须保持 baseline 级前进能力”。

### 1.2 3 月 18 日和 19 日是怎么修 reward 的

[Fact] 3 月 18 日首次把问题解释为：`speed` 奖励在训练初期进入指数核死梯度高原。  
证据：`docs/daily_logs/2026-3/2026-03-18/2026-3-18.md`

[Fact] 3 月 19 日又把 `energy` 解释为同类问题：scale 相对真实功率量级过大，导致整条通道几乎为 0。  
证据：`docs/daily_logs/2026-3/2026-03-19/2026-3-19.md`

[Fact] 当时对应修复是：

- `speed scale: 5.0 -> 1.0`
- `energy scale: 1.0 -> 0.005`

[Inference + Evidence] 这两次修复都只解决了“单个 objective 通道有没有梯度”问题，没有重新检查：

- 多个正向 objective 同时存在时，总 reward 的最优行为会不会偏向“不动”
- baseline locomotion prior 是否已经被削弱到不足以支撑前进

### 1.3 3 月 20 日之后，成功判据已经偏到“通道激活”

[Fact] 3 月 20 日日志把“4 个目标通道全部激活”当成关键成功标准。  
证据：`docs/daily_logs/2026-3/2026-03-20/2026-3-20.md`

[Fact] 但同一日志里，第三轮全量重训后的 `vx_err` 仍然对所有策略接近 `1.0`：

- `P1: vx_err = 1.009`
- `P2: vx_err = 1.006`
- `P3: vx_err = 0.999`
- `P4: vx_err = 1.000`
- `P10: vx_err = 0.998`

[Inference + Evidence] 这说明阶段三从 3 月 20 日开始，事实上已经接受了一个危险判据：

- “主目标通道能分化” 被视为成功
- “策略是否真的跟随前进命令” 被降成了次要指标

[Fact] 同一天的 `P2` pilot 甚至把 `cmd=0` 下几乎不动视为能效优先的合理行为。  
证据：`docs/daily_logs/2026-3/2026-03-20/2026-3-20.md`

[Inference + Evidence] 这从设计上打开了一个口子：  
如果 energy / smooth / stable 在“少动”时都容易得高分，而 speed 没有硬性保持 dominant，那么站立或近静止就会成为很强的局部最优。

### 1.4 3 月 22 日又发生了一次任务定义变化

[Fact] 3 月 22 日为了规避 ROS2 动态重连导致的任务定义突变，训练命令源被切换成了 `UniformVelocityCommand` 随机命令。  
证据：`docs/daily_logs/2026-3/2026-03-22/2026-3-22.md`

[Fact] 此后训练命令分布不再是 ROS2 固定 `vx=1.0`，而是内部随机命令。

[Inference + Evidence] 阶段三后半段真正优化的是：

- 随机命令分布下
- 4 个目标 channel 能分化
- 代表性策略可复现

并不是“固定前进命令协议下，仍然保持 locomotion capability”。

## 2. 当前根因重述

[Fact] `Pilot A` 在修正训练命令分布后仍然 `Hard Fail`。  
证据：`logs/eval/phase_morl_repair/pilot_a_p1_cmdfix_seed42_s1.json`

[Fact] `Pilot B` 在 `repair_forward_v1 + baseline warm-start` 后仍然 `Hard Fail`。  
证据：`logs/eval/phase_morl_repair/pilot_b_p10_cmdfix_warm_seed42_s1.json`

[Fact] `Pilot B` 的训练配置里命令 profile 确实已生效：

- `lin_vel_x = (0.5, 1.5)`
- `lin_vel_y = (0.0, 0.0)`
- `ang_vel_z = (0.0, 0.0)`
- `heading_command = false`

证据：`logs/rsl_rl/unitree_go1_rough/2026-03-28_15-00-59_pilot_b_p10_cmdfix_warm_seed42/params/env.yaml`

[Fact] `Pilot B` 末段日志里：

- `track_lin_vel_xy_exp ≈ 0.08`
- `track_ang_vel_z_exp ≈ 0.735`
- `morl_energy ≈ 0.191`
- `morl_smooth ≈ 0.196`
- `morl_stable ≈ 0.367`
- `error_vel_xy ≈ 1.9`

证据：`logs/sweep/phase_morl_repair/pilot_b_p10_cmdfix_warm_seed42_attempt1_stdout.log`

[Inference + Evidence] 当前最强根因应重新表述为：

**不是单个 reward channel 没梯度，而是当前 reward architecture 把“低动作、低功耗、低角速度、低偏航误差”的静止/近静止策略做成了更容易的高分解，而速度目标被削弱到不足以维持 locomotion prior。**

### 2.1 为什么说是 architecture 问题，而不只是 scale 问题

[Fact] `speed` 和 `energy` 的 scale 已经在阶段三被单独调通了；训练日志也确实出现了通道分化。

[Fact] 但即使通道都亮了，末段 `error_vel_xy` 仍接近 `2.0`，固定命令评估仍几乎不前进。

[Inference + Evidence] 所以问题不是“看不见梯度”，而是“看见了错误方向的总 reward”。

### 2.2 为什么不是命令链路主因

[Fact] baseline 在同一 `S1 + --skip_ros2` 协议下可以正常前进。

[Fact] 子 agent 从“命令/观测/评估协议”方向做过反驳，没有找到足够证据把主因改写成命令链路失效。

[Inference + Evidence] 命令链路可能仍有次级问题，但它解释不了 baseline 正常而 MORL family 整体不走。

### 2.3 为什么 warm-start 不是主因，但确实是混杂因素

[Fact] 当前 `--init_checkpoint` 实际调用的是 `runner.load(...)`，会带优化器状态和迭代号。  
因此 `Pilot B` 更像 resume-like finetune，而不是严格 policy-only warm-start。

[Inference + Evidence] 这削弱了 `Pilot B` 作为“纯 warm-start 验证”的证据强度，但解释不了 `Pilot A` 的失败。  
所以 warm-start 语义问题不是主因，只是后续实验洁净度必须修的点。

## 3. 重开修复的原则

### 原则 1：baseline locomotion prior 必须是不可削弱的硬约束

[Inference + Evidence] 阶段三失败的根本不是“没有 4 个目标”，而是把 baseline 中真正驱动 locomotion 的核心拉弱了。

新的训练方案必须满足：

- 任意权重点下，都不能把 speed core 削弱到“站着不动也更优”
- `track_lin_vel_xy_exp` 不再允许被降到 `0.1 / 0.2 / 0.25` 这种量级

### 原则 2：secondary objective 不能在静止时给出大额正向奖励

[Inference + Evidence] energy / smooth / stable 若在“少动”时就能轻松拿大量正分，会天然奖励退缩解。

新的训练方案必须满足：

- energy / smooth / stable 不能独立把静止策略抬成高总分
- 它们必须建立在“先会走”的前提上

### 原则 3：从第一轮 smoke 开始就用 `S1` 固定命令验收

[Fact] 阶段三的问题之所以拖到阶段四才暴露，是因为训练层只看 channel activation，评估层长期用随机命令口径。

新的方案必须满足：

- 每轮修复后先跑训练短 smoke
- 再立刻用 `S1 + --skip_ros2` 验证是否仍保有前进能力

### 原则 4：warm-start 必须与 resume 解耦

[Inference + Evidence] 如果后续还需要 warm-start，它只能是 policy-init，不应继续混入 optimizer / iteration 状态。

## 4. 备选修复路线

### 路线 A：只继续调当前 4 个 objective 的 scale / weight

**做法**

- 继续增大 speed weight
- 继续压低 energy / smooth / stable 权重
- 不改 reward 结构

**优点**

- 改动最小

**缺点**

- 只是继续在同一 reward architecture 上调参
- 无法根除“静止也高分”的结构性问题

**结论**

- [Inference + Evidence] 不推荐作为主方案

### 路线 B：恢复 baseline locomotion scaffold，在其上叠加 MORL 次级目标

**做法**

- 恢复 baseline 的速度驱动与 locomotion scaffold
- energy / smooth / stable 不再直接替代 baseline locomotion 结构
- 它们只在 scaffold 之上产生受控 trade-off

**优点**

- 最符合当前证据
- 直接针对 architecture 层根因
- 保留“会走”作为前置条件

**缺点**

- 会重写阶段三的训练叙事
- 需要重新定义 MORL reward wiring

**结论**

- [Inference + Evidence] 推荐

### 路线 C：继续保留当前 reward 结构，但引入 policy-only warm-start + 更保守 PPO

**做法**

- 单独修 `--init_checkpoint`
- 调小 `clip_param`
- 减少训练步数

**优点**

- 可测试优化层是否会放大遗忘

**缺点**

- 无法解释 `Pilot A` 的失败
- 不能触及 architecture 根因

**结论**

- [Inference + Evidence] 只能作为后续二级对照，不适合作为主修复路线

## 5. 推荐方案：Locomotion Scaffold MORL v2

### 5.1 新的训练结构

推荐改为两层结构：

`R_total = R_scaffold + R_morl_secondary`

其中：

- `R_scaffold`：固定的 baseline locomotion scaffold，任何策略都一样，任何权重都不能削弱
- `R_morl_secondary`：在 scaffold 之上的多目标偏好项，只负责 trade-off，不负责决定“走不走”

### 5.2 `R_scaffold` 应包含什么

建议固定保留 baseline locomotion 核心：

- `track_lin_vel_xy_exp`：恢复 baseline 强度
- `track_ang_vel_z_exp`
- `lin_vel_z_l2`
- `ang_vel_xy_l2`
- `dof_acc_l2`
- `feet_air_time`
- `time_out / base_contact`

建议恢复 baseline 中被阶段三拿掉的两个负惩罚：

- `dof_torques_l2`
- `action_rate_l2`

[Inference + Evidence] 这两个项在 baseline 中本来就是“动起来但别太猛”的稳定器，阶段三把它们拿掉后，energy/smooth 变成了额外正向奖励，奖励几何形状已经变了。

### 5.3 `R_morl_secondary` 应怎么改

推荐方向：

- speed 不再被允许降到破坏 locomotion 的权重量级
- energy / smooth / stable 改成次级偏好，而不是“与速度并列争抢总奖励主导权”的一级正向奖金

具体建议：

1. 速度维度：
   - 恢复 baseline 速度驱动强度，作为固定 scaffold
   - 若论文仍要求 speed 进入 4 维主目标，可在评估端保留 `J_speed` 作为正式分析维度

2. energy / smooth / stable：
   - 不再直接用当前这套“站着也容易得高分”的正向 exp 奖励做一级主导
   - 只允许在 locomotion scaffold 已成立的前提下影响 trade-off

3. 若需要训练侧 gating：
   - secondary objective 只在“前进已成立”的状态下计入，避免静止获利

[Don’t know] gating 的具体阈值现在还不能拍死，需要先用短 smoke 看 `vx_meas` 分布再定。

## 6. 新的执行顺序

### Task 1：先做 reward contribution 诊断

**目标**

- 先把“各 reward term 的带权贡献”打平看清楚

**Files**

- Modify: `scripts/go1-ros2-test/train.py`
- Modify: `src/go1-ros2-test/envs/morl_env_cfg.py`
- Test: `tests/unit/test_morl_reward_contribution_logging.py`

**验收**

- 训练日志能同时输出每个 reward term 的 raw value 和 weighted contribution
- 可以直接回答“是谁在总 reward 里压过了 speed”

### Task 2：实现 MORL v2 scaffold 配置

**目标**

- 建一个新的 MORL v2 env cfg，不覆写旧阶段三配置

**Files**

- Create: `src/go1-ros2-test/envs/morl_env_cfg_v2.py`
- Create: `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/morl_env_cfg_v2.py`
- Modify: `src/go1-ros2-test/envs/__init__.py`
- Modify: `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/__init__.py`
- Create or Modify: `src/go1-ros2-test/envs/mdp/morl_rewards_v2.py`
- Mirror accordingly

**关键要求**

- 不复用当前 `morl_env_cfg.py` 的 reward wiring
- 明确把 v1 和 v2 并存，保留证据链

### Task 3：修正 warm-start 语义

**目标**

- 把 `--init_checkpoint` 变成真正的 policy-only init

**Files**

- Modify: `scripts/go1-ros2-test/train.py`
- Test: `tests/unit/test_morl_repair_cli.py`

**关键要求**

- `runner.load(..., load_optimizer=False)` 或等价实现
- 不继承 `current_learning_iteration`

### Task 4：只做两个新 pilot，不再继续旧 A/B/C

推荐新的最小验证：

1. `V2-A`
   - 冷启动
   - MORL v2 scaffold
   - `P10`
   - `600 iter`

2. `V2-B`
   - policy-only warm-start
   - MORL v2 scaffold
   - `P10`
   - `600 iter`

**统一验收**

- 训练后立即跑 `S1 + --skip_ros2`
- 只要 `mean_vx_meas < 0.3` 或 `J_speed > 0.7`，就视为 v2 方案失败，继续回到 architecture 层

### Task 5：只有 v2 通过，才考虑扩到 `P2`

[Inference + Evidence] 当前不应直接跑 `Pilot C`，因为 `A/B` 已经说明旧 architecture 自身站不住。  
新的 `P2` 验证只能建立在 `P10` 先恢复前进能力之后。

## 7. 当前最稳妥的结论

[Fact] 阶段三当时的思路是：先把每个 objective channel 调活，再做随机命令下的多目标训练与 Pareto 分析。

[Fact] 这套思路在“channel activation”“随机命令下训练可复现”上是成立的。

[Inference + Evidence] 但它没有约束“任意权重下都必须保留 baseline locomotion prior”，因此在 architecture 层允许了“少动但高分”的解。

[Inference + Evidence] 所以本轮修复不应再延续“继续调 scale / weight + 补 pilot”这条线，而应重建 reward architecture：

- 恢复 locomotion scaffold
- secondary objective 不再主导总 reward
- 从第一轮 smoke 开始就用 `S1` 固定命令验收
