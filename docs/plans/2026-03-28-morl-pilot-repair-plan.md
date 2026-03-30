# MORL Pilot Repair Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不启动全量 MORL 重训的前提下，用 3 个 pilot 快速定位“最小充分修复层”，判断阶段四正式评估前到底应先修训练命令分布、是否必须加 warm-start，以及是否已经进入 reward/PPO 层问题。

**Architecture:** 固定当前 `S4-1` 评估协议和现有 MORL reward 权重定义，不再切回 ROS2 publisher。3 个 pilot 按“从少改到中改”的顺序推进：Pilot A 只改训练命令分布；Pilot B 在同一分布修复上加入 rough baseline warm-start 并验证官方锚点 `P10`；Pilot C 复用同一修复方案验证非速度偏置策略 `P2`。只有 pilot 给出正向证据后，才进入全量重训。

**Tech Stack:** Isaac Lab, RSL-RL, PowerShell, `env_isaaclab`, `scripts/go1-ros2-test/train.py`, `scripts/phase_morl/run_morl_eval.py`

## 1. 当前事实与边界

- [Fact] `rough baseline seed42` 在 `S1 + --skip_ros2` 下可正常前进：`mean_vx_meas=0.9709`，`J_speed=0.0548`。
- [Fact] `P1/P2/P3/P4/P9/P10 seed42` 在同一 `S1` 固定命令协议下几乎都不前进：`mean_vx_meas≈0`，`J_speed≈1.0`。
- [Fact] 2026-03-22 这轮 MORL 训练使用的是 `UniformVelocityCommand` 随机命令分布，而不是阶段四固定前进命令协议；训练配置中存在 `lin_vel_x/y`、`ang_vel_z` 对称随机范围，且 `heading_command=true`。
- [Fact] 2026-03-14 的阶段三计划中写明“本阶段主实验不默认启用 Warm-start”。
- [Inference + Evidence] 该 Warm-start 结论针对的是当时正式主实验的公平性与口径统一，不等于当前 repair 阶段也必须禁用 Warm-start。
- [Inference + Evidence] 当前最上游、最值得先验证的修复层仍然是“训练命令分布与阶段四评估协议不一致”。
- [Don't know] 仅修命令分布是否就足以让 `P10` 或 `P2` 恢复有效前进。
- [Don't know] 当前训练入口里的 `resume/load` 语义是否等同于“只加载 baseline 权重、不继承优化器与旧训练状态”的严格 warm-start。

## 2. 总体原则

- 阶段四评估入口保持不变：继续使用 `scripts/phase_morl/run_morl_eval.py --scenario S1 --skip_ros2` 做 pilot 主判据。
- pilot 期间不改 reward 定义、不改 `P1/P2/P10` 的权重向量、不改 `clip_param=0.3` 这类现有 MORL sweep 主口径。
- pilot 的主目标不是直接拿到论文结果，而是判断“哪一层开始必须改”。
- 若 pilot 未恢复 `S1` 固定命令前进，则停止全量重训，先做更深层诊断。

## 3. 统一 pilot 协议

### 3.1 训练命令修复口径

训练侧统一引入一个新的 forward-only command profile，建议命名为 `repair_forward_v1`：

- `lin_vel_x = (0.5, 1.5)`
- `lin_vel_y = (0.0, 0.0)`
- `ang_vel_z = (0.0, 0.0)`
- `heading_command = False`
- `rel_heading_envs = 0.0`
- `rel_standing_envs = 0.0`

设计理由：

- [Inference + Evidence] 它覆盖阶段四 `S1/S2/S3/S4/S5/S6` 的主要前进速度量级，避免继续训练“均值接近 0 的对称随机指令”。
- [Inference + Evidence] 它仍保留一定速度随机性，不会把 pilot 直接退化为“只背固定 `vx=1.0`”。

### 3.2 统一训练预算

- `seed=42`
- `num_envs=4096`
- `max_iterations=600`
- `--headless`
- `--disable_ros2_tracking_tune`
- `--clip_param 0.3`

说明：

- [Inference + Evidence] 当前 3 个 pilot 的目的是筛选修复层，不是一次性产出正式 checkpoint；先用 `600 iter` 控制计算成本更合理。
- [Assumption] `600 iter` 足以观察速度跟踪是否从“几乎完全不前进”跃迁到“明显恢复前进”。

### 3.3 统一评估判据

所有 pilot 训练结束后，统一跑一次：

```powershell
python scripts/phase_morl/run_morl_eval.py `
  --load_run <pilot_run_dir> `
  --checkpoint model_599.pt `
  --scenario S1 `
  --skip_ros2 `
  --summary_json <output_json>
```

主判据分三档：

| 档位 | 评估条件 | 解释 |
| --- | --- | --- |
| Hard Fail | `mean_vx_meas < 0.3` 或 `J_speed > 0.7` | 基本仍未恢复前进能力 |
| Borderline | `0.3 <= mean_vx_meas < 0.6` 且 `0.4 < J_speed <= 0.7` | 有恢复迹象，但不够支撑全量重训 |
| Pass | `mean_vx_meas >= 0.6` 且 `J_speed <= 0.4` | 已证明该修复层有效，允许进入下一层或放大全量实验 |

强通过条件：

- `mean_vx_meas >= 0.8`
- `J_speed <= 0.2`

训练侧辅助判据：

- 末段 `Episode_Reward/track_lin_vel_xy_exp >= 0.6`
- 末段 `Metrics/base_velocity/error_vel_xy <= 0.5`

## 4. Task 1: 先冻结 repair 基础设施

**Files:**

- Modify: `scripts/go1-ros2-test/train.py`
- Modify: `scripts/go1-ros2-test/checkpoint_utils.py`
- Modify: `src/go1-ros2-test/envs/morl_env_cfg.py`
- Modify: `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/morl_env_cfg.py`
- Create: `scripts/phase_morl/run_morl_repair_pilots.py`
- Test: `tests/unit/test_morl_repair_cli.py`
- Test: `tests/unit/test_checkpoint_utils.py`

**Step 1: 写失败测试**

- 验证训练入口支持 `repair_forward_v1` 这类 command profile 覆盖。
- 验证 warm-start 能接收 baseline checkpoint 的直接路径或跨实验目录路径。
- 验证 warm-start 与 resume 语义区分清楚，不会复用旧 log 目录。

**Step 2: 实现最小训练修复接口**

- 在 MORL 训练入口增加 command profile override。
- 推荐增加显式 `--init_checkpoint` 或等效入口，语义定义为“新 run 中只初始化模型权重，不把旧 run 当成断点续训”。
- 如果无法做到 policy-only load，必须在日志中明确标注 warm-start 退化成“resume-like finetune”，并降低结论强度。

**Step 3: 固化 pilot runner**

- 单独创建 `scripts/phase_morl/run_morl_repair_pilots.py`。
- 不修改历史 `run_morl_train_sweep.py` 的默认行为，避免污染 2026-03-22 证据链。
- 所有 pilot 输出统一落到新目录：
  - `logs/sweep/phase_morl_repair/`
  - `logs/eval/phase_morl_repair/`

**Step 4: 通过条件**

- command profile 可被写入 `env.yaml`
- warm-start checkpoint 路径可被稳定解析
- pilot runner 能为 A/B/C 生成固定 manifest

## 5. Task 2: Pilot A

### Pilot A 定义

- Policy: `P1`
- Weights: `0.7,0.1,0.1,0.1`
- Change Layer: 只改训练命令分布
- Warm-start: 不使用
- 目的: 验证“只修 command distribution”能否先救回最容易恢复的速度偏重策略

### Pilot A 训练命令

```powershell
python scripts/phase_morl/run_morl_repair_pilots.py `
  --pilot A `
  --policy-id P1 `
  --morl_weights 0.7,0.1,0.1,0.1 `
  --command_profile repair_forward_v1 `
  --seed 42 `
  --max_iterations 600
```

### Pilot A 结果解释

- [Inference + Evidence] 如果 A 都 Hard Fail，说明“仅修命令分布”不足以恢复 locomotion prior。
- [Inference + Evidence] 如果 A Pass，说明最小修复层很可能已经锁定在训练命令分布，而 reward/PPO 还不是第一优先级。

## 6. Task 3: Pilot B

### Pilot B 定义

- Policy: `P10`
- Weights: `0.2,0.2,0.2,0.4`
- Change Layer: 训练命令分布 + rough baseline warm-start
- Warm-start Source: `logs/rsl_rl/unitree_go1_rough/2026-03-08_16-46-27_baseline_rough_ros2cmd/model_1499.pt`
- 目的: 验证官方锚点 `P10` 是否必须继承 baseline locomotion prior 才能恢复固定命令前进

### Pilot B 训练命令

```powershell
python scripts/phase_morl/run_morl_repair_pilots.py `
  --pilot B `
  --policy-id P10 `
  --morl_weights 0.2,0.2,0.2,0.4 `
  --command_profile repair_forward_v1 `
  --init_checkpoint logs/rsl_rl/unitree_go1_rough/2026-03-08_16-46-27_baseline_rough_ros2cmd/model_1499.pt `
  --seed 42 `
  --max_iterations 600
```

### Pilot B 结果解释

- [Inference + Evidence] 若 A Fail 但 B Pass，则“命令分布 + warm-start”就是当前最小充分修复层。
- [Inference + Evidence] 若 A Pass 且 B Pass，则 warm-start 可能不是硬性必须，但对官方锚点仍可能是更稳妥方案。
- [Inference + Evidence] 若 B 仍 Hard Fail，则仅靠 command/warm-start 还不够，下一轮应转向 reward scale、observation 或 PPO 层。

## 7. Task 4: Pilot C

### Pilot C 定义

- Policy: `P2`
- Weights: `0.1,0.7,0.1,0.1`
- Change Layer: 与 Pilot B 相同
- Warm-start Source: 同 Pilot B
- 目的: 验证同一修复方案能否推广到非速度偏重、且最容易压低前进意愿的能效偏重策略

### Pilot C 训练命令

```powershell
python scripts/phase_morl/run_morl_repair_pilots.py `
  --pilot C `
  --policy-id P2 `
  --morl_weights 0.1,0.7,0.1,0.1 `
  --command_profile repair_forward_v1 `
  --init_checkpoint logs/rsl_rl/unitree_go1_rough/2026-03-08_16-46-27_baseline_rough_ros2cmd/model_1499.pt `
  --seed 42 `
  --max_iterations 600
```

### Pilot C 结果解释

- [Inference + Evidence] 若 C Pass，说明 command + warm-start 修复大概率已可扩展到 confirmatory set，不必第一时间改 reward/PPO。
- [Inference + Evidence] 若 B Pass 但 C Fail，说明 locomotion prior 已恢复，但非速度偏重权重仍会把速度目标压塌；下一层应优先检查 reward floor、weight normalization 或 auxiliary constraint，不是先全量重训。

## 8. Task 5: 汇总与决策

**Files:**

- Create: `logs/eval/phase_morl_repair/pilot_summary.csv`
- Create: `logs/eval/phase_morl_repair/pilot_decision.md`
- Update later: `docs/daily_logs/2026-3/2026-03-28/2026-3-28.md`

**汇总字段：**

- `pilot_id`
- `policy_id`
- `warm_start`
- `command_profile`
- `mean_cmd_vx`
- `mean_vx_meas`
- `mean_vx_abs_err`
- `J_speed`
- `pass_tier`

**决策矩阵：**

| 结果组合 | 结论 | 下一步 |
| --- | --- | --- |
| A Pass, B Pass, C Pass | 命令分布修复已基本成立；warm-start 作为稳妥增强项 | 进入 confirmatory full retrain |
| A Fail, B Pass, C Pass | 最小充分修复层 = 命令分布 + warm-start | 以该方案做全量重训 |
| A Pass, B Pass, C Fail | 速度能力可恢复，但非速度偏重策略仍失衡 | 先补 reward-layer 小修，再决定是否全量重训 |
| A Fail, B Fail | 问题已超出 command/warm-start 层 | 暂停全量重训，转向更深层诊断 |

## 9. 风险与防误判规则

- [Fact] 2026-03-14 的“不默认 Warm-start”是旧主实验口径，不是 repair 阶段禁令。
- [Inference + Evidence] 当前 repair 的目标是找到最小修复层，因此允许把 warm-start 当成一个受控变量，而不是方法学违规。
- [Don't know] `runner.load()` 是否会带入优化器状态与旧 iteration 计数；若会，必须把 warm-start 结果标注为“baseline initialized finetune”，不能直接宣称等同于冷启动重训。
- [Assumption] `S1` 是判断“是否恢复基本前进能力”的最低成本代表场景；pilot 阶段先不以 `S4/S6` 为主判据。
- [Inference + Evidence] 只要 A/B/C 还没给出明确通过证据，就不应启动 `P1-P10` 全量重训，更不应继续跑 `23 x 6` 阶段四正式评估。

## 10. 本轮计划结论

- 先不改 reward/PPO。
- 先做 3 个 pilot，但它们不是并列盲试，而是按“命令分布 -> warm-start -> 非速度偏重泛化”这个层级推进。
- 真正的全量重训开关，不取决于主观判断，而取决于 A/B/C 在 `S1` 固定命令协议下是否恢复前进。
