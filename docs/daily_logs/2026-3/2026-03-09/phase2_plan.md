# 阶段 2 详细分步骤计划：Reward 权重调优实验

> 基于 `plan.md` 阶段 2 定义 + 当前代码/数据状态制定。
> 日期：2026-03-09

## 前置状态确认

| 条目                                | 状态             | 说明                                                                         |
| ----------------------------------- | ---------------- | ---------------------------------------------------------------------------- |
| 阶段 0（Rough 闭环）                | ✅ 完成          | Rough ROS2Cmd 任务已注册、smoke test 通过                                    |
| 阶段 1（Flat+Rough baseline）       | ✅ 完成          | 两组 baseline 训练+评估完成                                                  |
| Flat baseline checkpoint            | ✅               | `logs/rsl_rl/unitree_go1_flat/2026-03-08_15-50-01_baseline_flat_ros2cmd`   |
| Rough baseline checkpoint           | ✅               | `logs/rsl_rl/unitree_go1_rough/2026-03-08_16-46-27_baseline_rough_ros2cmd` |
| `reward_engineering.md` 第 1-4 章 | ✅ 完成          | 383 行，含 baseline 数据解读                                                 |
| `reward_engineering.md` 第 5 章   | ❌ 未写          | **本阶段核心产出**                                                     |
| `flat_env_cfg.py`                 | ✅ 干净 baseline | 无 reward 覆盖，仅替换 command source                                        |

## 阶段 2 总目标

1. **执行 reward 权重调优实验**（One-factor-at-a-time，在 Flat 环境上）
2. **产出第 5 章**（Reward 权重调优实验）+ baseline vs 变体对比图

## 实验变体设计

全部基于 Flat baseline（300 iters，网络 `[128,128,128]`），每次只改一个变量：

| 变体 ID     | 改动项                   | baseline 值 | 变体值          | 假设                                     |
| ----------- | ------------------------ | ----------- | --------------- | ---------------------------------------- |
| **A** | `track_lin_vel_xy_exp` | 1.5         | **3.5**   | 提升速度跟踪权重 → 降低稳态误差         |
| **B** | `action_rate_l2`       | -0.01       | **-0.05** | 增大动作平滑惩罚 → 更平滑但可能响应变慢 |

> 备选变体 C（`feet_air_time` 权重调整）视时间决定是否执行。

---

## 分步骤执行计划

### 步骤 2.1：创建变体环境配置文件（代码）

新建 `src/go1-ros2-test/envs/flat_env_cfg_variants.py`：

- **继承** `UnitreeGo1Ros2CmdFlatEnvCfg`（保留 ROS2 command source）
- **仅覆盖目标 reward 权重**（在 `__post_init__` 中修改 `self.rewards.xxx.weight`）
- 每个变体需要 Train + Play 两个配置类（共 4 个类）：
  - `UnitreeGo1Ros2CmdFlatEnvCfg_TrackVelHigh`（训练，track_vel=3.5）
  - `UnitreeGo1Ros2CmdFlatEnvCfg_TrackVelHigh_PLAY`（评估）
  - `UnitreeGo1Ros2CmdFlatEnvCfg_ActionRateHigh`（训练，action_rate=-0.05）
  - `UnitreeGo1Ros2CmdFlatEnvCfg_ActionRateHigh_PLAY`（评估）

**关键实现细节**：

```python
@configclass
class UnitreeGo1Ros2CmdFlatEnvCfg_TrackVelHigh(UnitreeGo1Ros2CmdFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # One-factor override: only change track_lin_vel_xy_exp weight
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
```

### 步骤 2.2：注册变体任务 ID

修改 `src/go1-ros2-test/envs/__init__.py`，新增 4 个 gym 注册：

| 任务 ID                                                            | 配置类                   | PPO Runner           |
| ------------------------------------------------------------------ | ------------------------ | -------------------- |
| `Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-TrackVelHigh-v0`        | `_TrackVelHigh`        | `FlatPPORunnerCfg` |
| `Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-TrackVelHigh-Play-v0`   | `_TrackVelHigh_PLAY`   | `FlatPPORunnerCfg` |
| `Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-ActionRateHigh-v0`      | `_ActionRateHigh`      | `FlatPPORunnerCfg` |
| `Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-ActionRateHigh-Play-v0` | `_ActionRateHigh_PLAY` | `FlatPPORunnerCfg` |

> 全部使用 `UnitreeGo1FlatPPORunnerCfg`（与 baseline 一致，确保公平对比）。

### 步骤 2.3：更新 train.py / eval.py 白名单

在 `scripts/go1-ros2-test/train.py` 和 `eval.py` 的 `_ROS2_TASK_IDS` 集合中加入上述 4 个新任务 ID。

### 步骤 2.4：同步到 robot_lab 镜像（必须在训练前完成）

按 `CLAUDE.md` File Sync Rules：

| Source（先改）                                      | Mirror（同步）                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------- |
| `src/go1-ros2-test/envs/flat_env_cfg_variants.py` | `robot_lab/.../unitree_go1_ros2/flat_env_cfg_variants.py`（**新文件**） |
| `src/go1-ros2-test/envs/__init__.py`              | `robot_lab/.../unitree_go1_ros2/__init__.py`                                  |

同步后在 `CLAUDE.md` 同步表新增 `flat_env_cfg_variants.py` 条目。

### 步骤 2.5：TDD — 集成测试桩

新增 `tests/sim_required/test_go1_ros2cmd_flat_variants_registration.py`：

- `@pytest.mark.sim_required`
- 验证 4 个变体任务 ID 可被 gymnasium 解析注册
- 验证变体配置类的 reward 权重确实与 baseline 不同

### 步骤 2.6：Smoke Test（每个变体 1 iter）

**前提**：WSL 侧 ROS2 发布命令（与 baseline 相同）。

```powershell
# WSL 侧（后台持续发布）
wsl -d Ubuntu-22.04 bash -lc "source /opt/ros/humble/setup.bash; ros2 topic pub /go1/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.8, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' --rate 20"
```

```powershell
# Windows 侧 — 变体 A smoke
conda shell.powershell hook | Out-String | Invoke-Expression
conda activate env_isaaclab
python scripts/go1-ros2-test/train.py --task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-TrackVelHigh-v0 --num_envs 64 --max_iterations 1 --headless --disable_ros2_tracking_tune

# 变体 B smoke
python scripts/go1-ros2-test/train.py --task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-ActionRateHigh-v0 --num_envs 64 --max_iterations 1 --headless --disable_ros2_tracking_tune
```

**Smoke 通过标准**：

- 训练不报错，产出 checkpoint
- `params/env.yaml` 中目标 reward 权重与预期一致（A: `track_lin_vel_xy_exp=3.5`，B: `action_rate_l2=-0.05`）
- 其余 reward 权重与 baseline 完全一致（One-factor 验证）

### 步骤 2.7：正式训练（全量 300 iters）

使用与 baseline 完全相同的训练参数，仅任务 ID 不同：

```powershell
# 变体 A：track_lin_vel_xy_exp = 3.5
python scripts/go1-ros2-test/train.py --task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-TrackVelHigh-v0 --num_envs 4096 --max_iterations 300 --headless --disable_ros2_tracking_tune --run_name variant_a_track_vel_high

# 变体 B：action_rate_l2 = -0.05
python scripts/go1-ros2-test/train.py --task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-ActionRateHigh-v0 --num_envs 4096 --max_iterations 300 --headless --disable_ros2_tracking_tune --run_name variant_b_action_rate_high
```

W&B 命名规范：

- 变体 A: `wandb_project=go1-flat-locomotion`，`run_name=variant_a_track_vel_high`
- 变体 B: `wandb_project=go1-flat-locomotion`，`run_name=variant_b_action_rate_high`

### 步骤 2.8：评估（与 baseline 使用相同标准）

```powershell
# 变体 A 评估
python scripts/go1-ros2-test/eval.py --task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-TrackVelHigh-Play-v0 --num_envs 64 --eval_steps 3000 --warmup_steps 300 --strict_pass --summary_json results/phase2/variant_a_eval.json

# 变体 B 评估
python scripts/go1-ros2-test/eval.py --task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-ActionRateHigh-Play-v0 --num_envs 64 --eval_steps 3000 --warmup_steps 300 --strict_pass --summary_json results/phase2/variant_b_eval.json
```

### 步骤 2.9：数据对比与分析

收集三组数据（baseline + 2 个变体），产出对比表：

| 指标                    | Baseline (1.5/-0.01) | 变体 A (3.5/-0.01) | 变体 B (1.5/-0.05) |
| ----------------------- | -------------------- | ------------------ | ------------------ |
| 稳态 vx_abs_err         | ~0.085               | ?                  | ?                  |
| mean_vx_abs_err         | 0.153                | ?                  | ?                  |
| stable_ratio            | 0.824                | ?                  | ?                  |
| base_contact            | ≈0                  | ?                  | ?                  |
| TensorBoard reward 曲线 | 基准                 | 对比               | 对比               |

分析维度：

1. 变体 A：速度跟踪是否明显改善？其他指标（力矩、平滑度）是否劣化？
2. 变体 B：动作平滑度是否改善？速度跟踪是否受损？

### 步骤 2.10：撰写第 5 章

更新 `docs/reward_engineering.md`，新增第 5 章内容：

**第 5 章：Reward 权重调优实验**

- 5.1 实验方法论（One-factor-at-a-time）
- 5.2 变体 A 分析：提升 `track_lin_vel_xy_exp`（1.5→3.5）
- 5.3 变体 B 分析：增强 `action_rate_l2`（-0.01→-0.05）
- 5.4 baseline vs 变体对比图（TensorBoard/W&B 截图）
- 5.5 调参直觉总结（什么时候该调哪个 reward）

---

## 执行顺序与依赖关系

```
2.1 创建变体配置 ──→ 2.2 注册任务 ID ──→ 2.3 更新白名单
                                              │
                                              ▼
                          2.4 同步 robot_lab ──→ 2.5 TDD 测试桩
                                                      │
                                                      ▼
                                              2.6 Smoke Test
                                                      │
                                                      ▼
                                              2.7 正式训练（A/B 可并行）
                                                      │
                                                      ▼
                                              2.8 评估
                                                      │
                                                      ▼
                                              2.9 数据对比分析
                                                      │
                                                      ▼
                                              2.10 撰写第 5 章

```

## 约束提醒

1. **所有训练必须加 `--disable_ros2_tracking_tune`**（避免脚本自动改 PPO 参数）
2. **robot_lab 同步必须在 smoke test 之前完成**（train.py 从 robot_lab 导入）
3. **One-factor-at-a-time**：每个变体只改一个 reward 权重，其余保持 baseline
4. **TDD**：配置类需有 `@pytest.mark.sim_required` 集成测试桩
5. **File Sync**：新增的 `flat_env_cfg_variants.py` 需在 CLAUDE.md 同步表登记
