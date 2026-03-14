)`

# 毕设阶段三：多目标 Reward 设计与实现 — 详细分步骤计划

> 基于 `毕设阶段.md` 阶段三定义制定。
> 原始起草日期：2026-03-14
> 本稿已按仓库现状、任务书、开题报告和阶段 0-4 实际产物完成审查后修订。

---

## 前置状态确认

| 条目                                | 状态      | 说明                                                                                       |
| ----------------------------------- | --------- | ------------------------------------------------------------------------------------------ |
| `plan.md` 阶段 0-4                | ✅ 完成   | Rough baseline + 地形评估 + PPO/DR 实验                                                    |
| `reward_engineering.md` 第 1-7 章 | ✅ 完成   | 已形成 baseline / terrain / PPO / DR 的解释文档                                            |
| Phase 4 最佳配置                    | ✅ 确定   | `PPO-Clip-High`（`clip_param=0.3`）                                                    |
| 当前 Rough baseline checkpoint      | ✅        | `logs/rsl_rl/unitree_go1_rough/2026-03-08_16-46-27_baseline_rough_ros2cmd/model_1499.pt` |
| Baseline 多种子                     | ✅        | baseline 已补齐 `seed=42,43,44`                                                          |
| 课题边界                            | ✅ 已核对 | 任务书/开题报告要求：4 个主目标 + 多指标评价体系，不要求环境只保留 4 个 reward term        |

---

## 阶段三总目标

1. 在现有 Rough ROS2Cmd baseline 上引入 4 个**主目标**：速度、能效、平滑、稳定。
2. 保留 baseline 中与 locomotion 可行性强相关的**辅助约束**，避免把阶段三变成完全不同的新任务。
3. 训练 10 组不同权重的探索性策略，并对代表性策略补多种子验证。
4. 在统一测试集上使用 **4 个物理主指标** 做 Pareto Front 和 Hypervolume 分析。

---

## 核心设计决策

### 1. 主目标与辅助约束分离

本阶段不采用“只保留 4 个 reward term”的做法。

原因：

- 任务书和开题报告要求的是“实现不少于 4 个关键奖励函数”和“建立统一的多指标评价体系”，不是删除 baseline 的全部辅助约束。
- 当前 `UnitreeGo1Ros2CmdRoughEnvCfg` 明确继承 Rough baseline 的 reward、terrain、PPO、DR 结构，直接整体替换 `rewards` 会破坏与阶段 1-4 的可比性。
- rough baseline 中若干 reward/termination 项承担训练稳定和行为边界约束，不宜与 4 个研究主目标混为一谈。

### 2. Linear Scalarization 的工程实现方式

方法层面保留线性加权：

`R_primary = w1 * r_speed + w2 * r_energy + w3 * r_smooth + w4 * r_stable`

工程实现上不再强制单独创建 `linear_scalarization.py` 类。

原因：

- Isaac Lab 当前训练链路本来就是通过 `RewardTermCfg.weight` 汇总标量 reward。
- 直接通过环境配置覆盖 4 个主目标权重即可复用现有 RSL-RL 训练路径。
- 避免维护两套 reward 汇总逻辑。

### 3. Pareto 主图的评价对象

Pareto Front / Hypervolume 主分析只使用 4 个**物理主指标**：

- `J_speed`
- `J_energy`
- `J_smooth`
- `J_stable`

不再直接使用训练期间 4 个 `morl_*` reward 的均值做主分析。

原因：

- 训练 reward 是优化代理信号。
- 论文与任务书要求的是可解释的多指标评价体系。
- 物理指标更适合作为主结果展示与后续答辩口径。

### 4. 阶段三的证据强度

阶段三采用“两层证据”：

- 探索层：10 组权重 × 1 个种子（`seed=42`）
- 确认层：对代表性策略补齐 `seed=43,44`

这样既控制训练量，又避免用单个 seed 直接下 Pareto 结论。

---

## 4 个主目标定义

### 训练主目标与评估主指标映射

| 主目标 | 训练目标（reward proxy）                            | 评估主指标（物理量）                                              | 说明                             |
| ------ | --------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------- |
| 速度   | `r_speed = exp(-k_v * ‖v_xy - v_xy_cmd‖²)`     | `J_speed = RMSE(‖v_xy - v_xy_cmd‖)`                           | 评估时是误差，越小越好           |
| 能效   | `r_energy = exp(-k_e * Σ\|τ_i * ω_i\|)`        | `J_energy = Σ_t Σ_i \|τ_i * ω_i\| * Δt / distance`                | 训练用功率 proxy，评估用单位路程能耗 |
| 平滑   | `r_smooth = exp(-k_a * Σ(a_t - a_{t-1})²)`      | `J_smooth = mean(‖a_t - a_{t-1}‖)`                            | 避免动作抖动                     |
| 稳定   | `r_stable = exp(-k_s * (ω_roll² + ω_pitch²))` | `J_stable = 0.5 * norm(RMS(姿态波动)) + 0.5 * norm(RMS(ω_xy))` | 统一跨场景口径，不用绝对水平姿态 |

### 训练主目标设计说明

#### 速度

- 直接复用 `track_lin_vel_xy_exp` 路径。
- 主目标聚焦 `xy` 线速度误差。
- `track_ang_vel_z_exp` 保留为辅助约束，不纳入 4 维 Pareto 主目标。

#### 能效

- 不再用 `joint_torques_l2` 作为主目标最终定义。
- 改用 `joint_power = Σ|τ_i * ω_i|`，更接近机械功率 / 能耗 proxy。
- 训练时对 `joint_power` 做 exp 包装；评估时统计单位路程能耗。

#### 平滑

- 主目标沿用 `action_rate_l2` 路径，但不再把 baseline 里的 `action_rate_l2` 同时作为固定辅助项重复保留。
- 训练目标和评估指标语义一致，便于解释。

#### 稳定

- 不采用 `flat_orientation_l2` 作为 rough 全场景主稳定性目标。
- 原因：rough baseline 已关闭 `flat_orientation_l2`，坡地/台阶场景下绝对水平姿态不是合理目标。
- 训练时用 `ω_xy` 做稳定性 proxy。
- 评估时用“姿态波动 + 角速度波动”的统一组合指标。

### 仍待阶段三实现中校准的常数

以下超参数在文档中先占位，不在本阶段计划中预设死值：

- `k_v`
- `k_e`
- `k_a`
- `k_s`

阶段三将通过 smoke test 和 reward 分布检查确定其数量级。

---

## 辅助约束保留清单

### 固定保留，不参与权重向量

| 项目                         | 当前 rough baseline 状态 | 在阶段三中的角色 |
| ---------------------------- | ------------------------ | ---------------- |
| `track_ang_vel_z_exp`      | active                   | 固定辅助约束     |
| `lin_vel_z_l2`             | active                   | 固定辅助约束     |
| `ang_vel_xy_l2`            | active                   | 固定辅助约束     |
| `dof_acc_l2`               | active                   | 固定辅助约束     |
| `feet_air_time`            | active                   | 固定辅助约束     |
| `time_out` termination     | active                   | 固定终止条件     |
| `base_contact` termination | active                   | 固定终止条件     |

### 不再作为固定辅助项重复保留

| 项目                    | 当前 rough baseline 状态 | 阶段三处理方式                   |
| ----------------------- | ------------------------ | -------------------------------- |
| `dof_torques_l2`      | active                   | 由能效主目标替代，不重复固定保留 |
| `action_rate_l2`      | active                   | 由平滑主目标替代，不重复固定保留 |
| `flat_orientation_l2` | weight = 0.0             | 继续不作为固定辅助项启用         |
| `undesired_contacts`  | `null`                 | 不重新引入                       |
| `dof_pos_limits`      | weight = 0.0             | 不启用                           |

---

## 分步骤执行计划

### 步骤 M1：实现 4 个主目标 reward 函数

**目标**：在 `src/go1-ros2-test/envs/mdp/` 下新增 `morl_rewards.py`，仅实现 4 个主目标函数。

**新建文件**：

- `src/go1-ros2-test/envs/mdp/morl_rewards.py`

**函数设计**：

```python
def morl_track_vel_exp(env, command_name: str = "base_velocity", scale: float = 5.0) -> torch.Tensor:
    ...

def morl_energy_power_exp(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    ...

def morl_action_smoothness_exp(env, scale: float = 0.01) -> torch.Tensor:
    ...

def morl_stability_ang_vel_exp(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    ...
```

**要求**：

1. 函数签名必须与 `RewardTermCfg.params` 一致。
2. 只实现 4 个主目标，不在这里复制辅助约束项。
3. 统一返回值域 `(0, 1]` 的 exp 形式 reward。

**测试**：

- `tests/unit/test_morl_rewards.py`
- 核查输出值域
- 核查 CPU/GPU 张量兼容

### 步骤 M2：创建 MORL 环境配置

**目标**：在 Rough ROS2Cmd baseline 上叠加 4 个主目标，并保留辅助约束。

**新建文件**：

- `src/go1-ros2-test/envs/morl_env_cfg.py`

**设计要求**：

1. 继承 `UnitreeGo1Ros2CmdRoughEnvCfg`
2. 不整体替换 `rewards`
3. 只覆盖或新增：
   - 4 个 primary objective
   - 需要关闭的重叠 baseline 项（`dof_torques_l2`、`action_rate_l2`）
4. 明确提供 `UnitreeGo1MORLEnvCfg_PLAY`

**配置思路**（示意，具体以 `configclass` 实际可用写法为准）：

```python
@configclass
class UnitreeGo1MORLEnvCfg(UnitreeGo1Ros2CmdRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # primary objectives
        self.rewards.track_lin_vel_xy_exp.func = morl_track_vel_exp
        self.rewards.track_lin_vel_xy_exp.weight = 0.25
        self.rewards.track_lin_vel_xy_exp.params = {"command_name": "base_velocity", "scale": 5.0}

        self.rewards.morl_energy = RewTerm(...)
        self.rewards.morl_smooth = RewTerm(...)
        self.rewards.morl_stable = RewTerm(...)

        # remove duplicate baseline penalties
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = 0.0

        # keep auxiliary constraints fixed
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -2.5e-07
        self.rewards.feet_air_time.weight = 0.01
```

**CLI 权重覆盖**：

- 在 `scripts/go1-ros2-test/train.py` 中新增 `--morl_weights`
- 仅覆盖 4 个 primary objective 的 weight
- 不影响辅助约束项

### 步骤 M3：注册 MORL 任务 ID

**修改文件**：

- `src/go1-ros2-test/envs/__init__.py`

**新增任务**：

| 任务 ID                                             | 配置类                        | PPO Runner                      |
| --------------------------------------------------- | ----------------------------- | ------------------------------- |
| `Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v0`      | `UnitreeGo1MORLEnvCfg`      | `UnitreeGo1RoughPPORunnerCfg` |
| `Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v0` | `UnitreeGo1MORLEnvCfg_PLAY` | `UnitreeGo1RoughPPORunnerCfg` |

**同步更新**：

- `scripts/go1-ros2-test/train.py` 的 `_ROS2_TASK_IDS`
- `scripts/go1-ros2-test/eval.py` 的 `_ROS2_TASK_IDS`

### 步骤 M4：同步到 robot_lab 镜像

按 `CLAUDE.md` File Sync Rules：

| Source（先改）                                 | Mirror（同步）                                                                                                                     | 操作   |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `src/go1-ros2-test/envs/mdp/morl_rewards.py` | `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/morl_rewards.py`                               | 新文件 |
| `src/go1-ros2-test/envs/morl_env_cfg.py`     | `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/morl_env_cfg.py` | 新文件 |
| `src/go1-ros2-test/envs/__init__.py`         | `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/__init__.py`     | 更新   |

### 步骤 M5：最小测试闭环

**新增测试**：

- `tests/sim_required/test_go1_ros2cmd_morl_registration.py`
- `tests/unit/test_morl_weight_override.py`
- `tests/unit/test_morl_metric_computation.py`

**测试目标**：

1. MORL train/play task 可被 gym 解析
2. MORL task 仍使用 `UnitreeGo1RoughPPORunnerCfg`
3. `--morl_weights` 只影响 4 个主目标
4. 4 个物理主指标计算函数可对短 rollout 输出结构化结果

### 步骤 M6：Smoke Test（1 iter）

1. WSL 端启动 ROS2 命令发布
2. Windows 端执行：

```powershell
python scripts/go1-ros2-test/train.py `
  --task Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v0 `
  --num_envs 64 `
  --max_iterations 1 `
  --headless `
  --disable_ros2_tracking_tune `
  --morl_weights "0.25,0.25,0.25,0.25"
```

**通过标准**：

- 任务可启动并产出 checkpoint
- `params/env.yaml` 中 4 个主目标权重正确写入
- 辅助约束项仍保留固定值
- `params/agent.yaml` 中 `clip_param=0.3`

### 步骤 M7：探索性训练（10 组权重 × 1 个 seed）

**权重主表**：

| 策略ID | 权重 [速度,能效,平滑,稳定] | 预期特点   |
| ------ | -------------------------- | ---------- |
| P1     | [0.7, 0.1, 0.1, 0.1]       | 速度优先   |
| P2     | [0.1, 0.7, 0.1, 0.1]       | 能效优先   |
| P3     | [0.1, 0.1, 0.7, 0.1]       | 平滑优先   |
| P4     | [0.1, 0.1, 0.1, 0.7]       | 稳定优先   |
| P5     | [0.4, 0.3, 0.2, 0.1]       | 综合均衡   |
| P6     | [0.5, 0.3, 0.1, 0.1]       | 速度+能效  |
| P7     | [0.3, 0.3, 0.2, 0.2]       | 四目标均衡 |
| P8     | [0.2, 0.4, 0.2, 0.2]       | 能效偏重   |
| P9     | [0.3, 0.2, 0.3, 0.2]       | 平滑偏重   |
| P10    | [0.2, 0.2, 0.2, 0.4]       | 稳定偏重   |

**训练协议**：

- 使用 `PPO-Clip-High`：`clip_param=0.3`
- `max_iterations=1500`
- 探索层统一使用 `seed=42`
- 本阶段主实验不默认启用 Warm-start

**批量脚本**：

- `scripts/phase_morl/run_morl_train_sweep.py`

### 步骤 M8：确认性训练（代表性策略补多种子）

探索层完成后，选择 5 个代表性策略补齐 `seed=43,44`：

- P1
- P2
- P3
- P4
- 从 P5-P10 中选择 1 个非支配折中点

**定位**：

- 这一步是阶段三正式结论的确认层
- 如果时间不足，文档中必须明确标注探索层结果为 exploratory

### 步骤 M9：创建物理指标评估脚本

**新建文件**：

- `scripts/phase_morl/run_morl_eval.py`

**输出 4 个主指标**：

| 指标         | 计算方式                                           | 优化方向 |
| ------------ | -------------------------------------------------- | -------- |
| `J_speed`  | `RMSE(‖v_xy - v_xy_cmd‖)`                      | 越小越好 |
| `J_energy` | `Σ_t Σ_i \|τ_i * ω_i\| * Δt / distance`           | 越小越好 |
| `J_smooth` | `mean(‖a_t-a_{t-1}‖)`                          | 越小越好 |
| `J_stable` | `0.5*norm(RMS(姿态波动)) + 0.5*norm(RMS(ω_xy))` | 越小越好 |

**补充结果**：

- `success_rate`
- `mean_base_contact_rate`
- `recovery_time`（S6）
- `mean_timeout_rate`

### 步骤 M10：执行统一测试集评估

对每个已训练策略在统一测试集上输出 4 个主指标和补充结果：

```powershell
python scripts/phase_morl/run_morl_eval.py `
  --task Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v0 `
  --load_run <policy_run_dir> `
  --num_envs 64 `
  --eval_steps 3000 `
  --summary_json logs/eval/phase_morl/<policy_id>.json
```

### 步骤 M11：Pareto 分析与可视化

**分析脚本**：

- `scripts/phase_morl/analyze_pareto.py`

**分析规则**：

1. 只在 4 维物理主指标空间中做 Pareto 和 HV
2. 统一按**最小化问题**处理
3. 归一化边界一旦确定即冻结，不随补 seed 或补策略重算
4. `success_rate` 等补充结果不进入主 Pareto 维度

**输出**：

- `logs/eval/phase_morl/pareto_analysis.json`
- `docs/figures/pareto_front_*.png`

### 步骤 M12：更新文档

更新 `docs/reward_engineering.md` 第 8 章，内容包括：

1. 4 个主目标与辅助约束的拆分
2. 训练主目标与评估物理指标的映射
3. 10 组探索性策略 + 代表性多种子确认
4. Pareto Front / Hypervolume 分析
5. 不同权重偏好下的行为差异

---

## 通过标准汇总

| 步骤 | 通过标准                                                  |
| ---- | --------------------------------------------------------- |
| M1   | 4 个主目标函数单元测试通过                                |
| M2   | MORL 环境配置可实例化，且未整体替换 baseline 辅助约束     |
| M3   | MORL task train/play 注册成功                             |
| M4   | `robot_lab` 镜像同步完成，无 import 错误                |
| M5   | 任务注册测试、权重覆盖测试、指标计算测试通过              |
| M6   | Smoke test 通过，checkpoint 产出，env/agent yaml 核对通过 |
| M7   | 10 组探索性策略训练完成                                   |
| M8   | 代表性策略补齐 `seed=43,44`                             |
| M9   | 评估脚本输出 4 个物理主指标和补充结果                     |
| M10  | 统一测试集 JSON 产出                                      |
| M11  | Pareto Front / Hypervolume 分析完成                       |
| M12  | 第 8 章完稿                                               |

---

## 风险与对策

| # | 风险                                   | 影响                       | 对策                                                   |
| - | -------------------------------------- | -------------------------- | ------------------------------------------------------ |
| 1 | 主目标 scale 设定不合理                | 4 个主目标分布失衡         | 先做 1 iter smoke + 短评估，检查 reward 和物理指标分布 |
| 2 | 把辅助约束删得过多                     | 训练不收敛或学出不可行步态 | 保留既有 rough baseline 中的关键辅助约束与 termination |
| 3 | 10 组探索性策略区分度不足              | Pareto Front 退化          | 在探索层结束后允许替换 1-2 个中间权重点                |
| 4 | 单种子结果偶然性较大                   | Pareto 结论不稳            | 用代表性策略补齐多种子确认                             |
| 5 | Hypervolume 口径不稳定                 | 不同批次结果不可比         | 归一化边界冻结后全阶段统一使用                         |
| 6 | `joint_power` / 稳定性指标实现有 bug | 主指标失真                 | 为指标计算单独补 unit test                             |

## 附录 A：实现模板

### A.1 主目标 reward 函数模板

```python
import torch
from isaaclab.managers import SceneEntityCfg


def morl_track_vel_exp(env, command_name: str = "base_velocity", scale: float = 5.0) -> torch.Tensor:
    asset = env.scene["robot"]
    err = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-scale * err)


def morl_energy_power_exp(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    power = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return torch.exp(-scale * power)


def morl_action_smoothness_exp(env, scale: float = 0.01) -> torch.Tensor:
    action_diff = env.action_manager.action - env.action_manager.prev_action
    return torch.exp(-scale * torch.sum(torch.square(action_diff), dim=1))


def morl_stability_ang_vel_exp(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.root_ang_vel_b[:, :2]
    return torch.exp(-scale * torch.sum(torch.square(ang_vel_xy), dim=1))
```

### A.2 训练权重主表

```yaml
P1:  {w: [0.7, 0.1, 0.1, 0.1]}
P2:  {w: [0.1, 0.7, 0.1, 0.1]}
P3:  {w: [0.1, 0.1, 0.7, 0.1]}
P4:  {w: [0.1, 0.1, 0.1, 0.7]}
P5:  {w: [0.4, 0.3, 0.2, 0.1]}
P6:  {w: [0.5, 0.3, 0.1, 0.1]}
P7:  {w: [0.3, 0.3, 0.2, 0.2]}
P8:  {w: [0.2, 0.4, 0.2, 0.2]}
P9:  {w: [0.3, 0.2, 0.3, 0.2]}
P10: {w: [0.2, 0.2, 0.2, 0.4]}
```

---

## 附录 B：Pareto / HV 分析约束

### B.1 主分析对象

仅使用下列 4 维最小化指标：

- `J_speed`
- `J_energy`
- `J_smooth`
- `J_stable`

### B.2 归一化规则

- 使用固定边界归一化到 `[0, 1]`
- 边界一旦确定即冻结
- 不使用当前候选集自适应 min-max 作为最终报告口径

### B.3 Hypervolume 规则

- 按最小化问题使用 `pymoo`
- `ref_point` 必须设为比归一化后最差点更差的点
- 不再先把目标转成“最大化得分”再做 HV

---

## 附录 C：阶段四衔接说明

阶段三产出将直接服务于阶段四：

1. 从阶段三代表性策略中选取进入正式场景评估的候选
2. 阶段四保留 `success_rate`、`recovery_time`、`base_contact_rate` 等补充指标
3. 阶段三的 4 维物理主指标口径在阶段四继续沿用，避免论文前后不一致
