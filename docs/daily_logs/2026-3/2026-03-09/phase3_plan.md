# 阶段 3 详细分步骤计划：地形 Curriculum 目标化验证

> 基于 `plan.md` 阶段 3 定义（lines 246-286）+ 阶段 1/2 产出制定。
> 日期：2026-03-09

## 前置状态确认

| 条目                                | 状态             | 说明                                                                                       |
| ----------------------------------- | ---------------- | ------------------------------------------------------------------------------------------ |
| 阶段 0（Rough 闭环）                | ✅ 完成          | Rough ROS2Cmd 任务已注册、smoke test 通过                                                  |
| 阶段 1（Flat+Rough baseline）       | ✅ 完成          | 两组 baseline 训练+评估完成                                                                |
| 阶段 2（Reward 调优实验）           | ✅ 完成          | 变体 A/B 训练+评估+第 5 章完成                                                             |
| Rough baseline checkpoint           | ✅               | `logs/rsl_rl/unitree_go1_rough/2026-03-08_16-46-27_baseline_rough_ros2cmd/model_1499.pt` |
| Rough baseline 评估数据             | ✅               | `mean_vx_abs_err=0.0798`, `stable_ratio=0.9678`（混合地形）                            |
| `reward_engineering.md` 第 1-5 章 | ✅ 完成          | 第 6 章为**本阶段核心产出**                                                          |
| `rough_env_cfg.py`                | ✅ 干净 baseline | 仅替换 command source，不改 terrain/reward                                                 |

## 阶段 3 总目标

1. **在 ROS2Cmd rough 环境下验证特定地形的通过能力**（纯评估，不新增训练）
2. **4 种目标地形**：坡地 10° / 20°、台阶 10cm / 15cm
3. **产出第 6 章**（地形与 Reward 的交互）+ 地形专项评估对比数据

## 地形评估配置设计

全部基于 Rough baseline 模型（`model_1499.pt`，网络 `[512, 256, 128]`），仅替换地形配置：

| 配置 ID            | 地形类型             | 关键参数                           | 预期难度 | 训练范围内？        |
| ------------------ | -------------------- | ---------------------------------- | -------- | ------------------- |
| **Slope10**  | `hf_pyramid_slope` | `slope_range=(0.176, 0.176)`     | 简单     | ✅ 是               |
| **Slope20**  | `hf_pyramid_slope` | `slope_range=(0.364, 0.364)`     | 中等     | ✅ 是               |
| **Stairs10** | `pyramid_stairs`   | `step_height_range=(0.10, 0.10)` | 中等     | ✅ 刚好上限         |
| **Stairs15** | `pyramid_stairs`   | `step_height_range=(0.15, 0.15)` | 困难     | ❌ 超出 (0.1m 上限) |

> **关键风险**：Go1 baseline 训练地形中 `boxes.grid_height_range=(0.025, 0.1)`，
> 15cm 台阶已超出训练范围上限，性能退化在预期之内。如严重退化，在分析中记录根因而非视为失败。

### Isaac Lab Go1 Rough Baseline 地形参数（参考）

训练时 `LocomotionVelocityRoughEnvCfg` 的 `sub_terrains` 包含多种地形类型，
Go1 对其中两项做了缩放：

```python
# Go1 rough_env_cfg.py 中的地形覆盖
self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
```

---

## 分步骤执行计划

### 步骤 3.1：确认 Isaac Lab 地形生成器 API

在实现配置类之前，需要确认以下 API 细节：

1. `LocomotionVelocityRoughEnvCfg` 中完整的 `sub_terrains` 字典（所有 key 和 class）
2. `HfPyramidSlopedTerrainCfg` 的精确导入路径和参数名（`slope_range` 确认）
3. `MeshPyramidStairsTerrainCfg` 的精确导入路径和参数名（`step_height_range` 确认）
4. 确认单地形覆盖方式：直接替换 `sub_terrains` 字典只保留一个 key，`proportion=1.0`

**验证方式**：

```python
# 在 env_isaaclab 中运行
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
cfg = LocomotionVelocityRoughEnvCfg()
tg = cfg.scene.terrain.terrain_generator
for k, v in tg.sub_terrains.items():
    print(f"{k}: {type(v).__name__}, proportion={v.proportion}")
    print(f"  module: {type(v).__module__}")
    print(f"  attrs: {[a for a in dir(v) if not a.startswith('_')]}")
```

### 步骤 3.2：创建地形专项评估配置文件（代码）

新建 `src/go1-ros2-test/envs/rough_env_cfg_terrain_eval.py`：

- **继承** `UnitreeGo1Ros2CmdRoughEnvCfg_PLAY`（保留 ROS2 command source + PLAY 设置）
- **仅覆盖 `terrain_generator.sub_terrains`**（替换为单一目标地形）
- **仅需 PLAY 配置类**（本阶段纯评估，不新增训练）
- 共 4 个类：

| 配置类                                         | 地形      | 关键覆盖                           |
| ---------------------------------------------- | --------- | ---------------------------------- |
| `UnitreeGo1Ros2CmdRoughEnvCfg_Slope10_PLAY`  | 坡地 10° | `slope_range=(0.176, 0.176)`     |
| `UnitreeGo1Ros2CmdRoughEnvCfg_Slope20_PLAY`  | 坡地 20° | `slope_range=(0.364, 0.364)`     |
| `UnitreeGo1Ros2CmdRoughEnvCfg_Stairs10_PLAY` | 台阶 10cm | `step_height_range=(0.10, 0.10)` |
| `UnitreeGo1Ros2CmdRoughEnvCfg_Stairs15_PLAY` | 台阶 15cm | `step_height_range=(0.15, 0.15)` |

**关键实现模式**（待 3.1 确认导入路径后填入具体 class）：

```python
@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_Slope10_PLAY(UnitreeGo1Ros2CmdRoughEnvCfg_PLAY):
    """Terrain-specific eval: 10° slope only."""
    def __post_init__(self):
        super().__post_init__()
        tg = self.scene.terrain.terrain_generator
        # Replace all sub_terrains with single slope terrain
        tg.sub_terrains = {
            "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
                proportion=1.0,
                slope_range=(0.176, 0.176),  # tan(10°) ≈ 0.176
            ),
        }
        tg.curriculum = False
```

### 步骤 3.3：注册地形专项评估任务 ID

修改 `src/go1-ros2-test/envs/__init__.py`，新增 4 个 gym 注册（仅 Play）：

| 任务 ID                                                       | 配置类             | PPO Runner            |
| ------------------------------------------------------------- | ------------------ | --------------------- |
| `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Slope10-Play-v0`  | `_Slope10_PLAY`  | `RoughPPORunnerCfg` |
| `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Slope20-Play-v0`  | `_Slope20_PLAY`  | `RoughPPORunnerCfg` |
| `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Stairs10-Play-v0` | `_Stairs10_PLAY` | `RoughPPORunnerCfg` |
| `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Stairs15-Play-v0` | `_Stairs15_PLAY` | `RoughPPORunnerCfg` |

> 全部使用 `UnitreeGo1RoughPPORunnerCfg`（与 Rough baseline 一致，确保 checkpoint 兼容）。
> 仅注册 Play 任务（本阶段无训练需求）。

### 步骤 3.4：更新 eval.py 白名单

在 `scripts/go1-ros2-test/eval.py` 的 `_ROS2_TASK_IDS` 集合中加入 4 个新任务 ID。

> `train.py` 不需要更新（本阶段不新增训练任务）。

### 步骤 3.5：同步到 robot_lab 镜像（必须在 smoke test 之前完成）

按 `CLAUDE.md` File Sync Rules：

| Source（先改）                            | Mirror（同步）                                                   | 操作             |
| ----------------------------------------- | ---------------------------------------------------------------- | ---------------- |
| `src/.../rough_env_cfg_terrain_eval.py` | `robot_lab/.../unitree_go1_ros2/rough_env_cfg_terrain_eval.py` | **新文件** |
| `src/.../envs/__init__.py`              | `robot_lab/.../__init__.py`                                    | 更新             |

同步后在 `CLAUDE.md` 同步表新增 `rough_env_cfg_terrain_eval.py` 条目。

### 步骤 3.6：Smoke Test（1 iter，验证 4 个新任务 ID 可解析运行）

> 必须在步骤 3.5 同步完成后执行。

1. **WSL 端启动 ROS2 命令发布**（全程保持）：

   ```powershell
   wsl -d Ubuntu-22.04 bash -lc "source /opt/ros/humble/setup.bash; ros2 topic pub /go1/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.8, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' --rate 20"
   ```
2. **Windows 端依次验证 4 个任务 ID**（每个 1 iter，仅确认可启动+不报错）：

   ```powershell
   conda shell.powershell hook | Out-String | Invoke-Expression
   conda activate env_isaaclab

   # Slope 10°
   python scripts/go1-ros2-test/eval.py --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Slope10-Play-v0 --num_envs 16 --eval_steps 100 --warmup_steps 10 --load_run 2026-03-08_16-46-27_baseline_rough_ros2cmd

   # Slope 20°
   python scripts/go1-ros2-test/eval.py --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Slope20-Play-v0 --num_envs 16 --eval_steps 100 --warmup_steps 10 --load_run 2026-03-08_16-46-27_baseline_rough_ros2cmd

   # Stairs 10cm
   python scripts/go1-ros2-test/eval.py --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Stairs10-Play-v0 --num_envs 16 --eval_steps 100 --warmup_steps 10 --load_run 2026-03-08_16-46-27_baseline_rough_ros2cmd

   # Stairs 15cm
   python scripts/go1-ros2-test/eval.py --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Stairs15-Play-v0 --num_envs 16 --eval_steps 100 --warmup_steps 10 --load_run 2026-03-08_16-46-27_baseline_rough_ros2cmd
   ```
3. **Smoke Test 通过标准**：

   - 4 个任务均能正常启动，无 import/注册 错误。
   - 地形视觉正确（若非 headless，可目视确认地形类型）。
   - 日志中 `obs_dim` 与 Rough baseline 一致（包含 height_scan）。
   - `cmd_vx` 非全零（ROS2 桥接正常工作）。

### 步骤 3.7：执行 4 组地形专项评估（正式评估）

> Smoke test 全部通过后执行。使用 Rough baseline checkpoint，`num_envs=64`，`eval_steps=3000`。

**统一 checkpoint**：`--load_run 2026-03-08_16-46-27_baseline_rough_ros2cmd`

> **v2 修订**：所有命令统一加 `--target_vx 0.8`（与 ROS2 publisher vx=0.8 对齐）。
> Slope10 通过标准从 `abs_err≤0.15, stable_ratio≥0.85, thresh=0.15` 放宽至与 Slope20 一致
> （原标准比 Rough 混合地形 baseline 还严，不合理）。
> 新增 `--pass_cmd_vx_tol` 参数替换原硬编码 0.05 容差。

#### 3.7.1 坡地 10° 评估

```powershell
python scripts/go1-ros2-test/eval.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Slope10-Play-v0 ^
  --num_envs 64 --eval_steps 3000 --warmup_steps 300 ^
  --load_run 2026-03-08_16-46-27_baseline_rough_ros2cmd ^
  --target_vx 0.8 --pass_cmd_vx_tol 0.05 ^
  --pass_abs_err 0.25 --pass_stable_ratio 0.70 --stable_err_thresh 0.20 ^
  --strict_pass ^
  --summary_json logs/eval/phase3_formal/slope10.json
```

#### 3.7.2 坡地 20° 评估

```powershell
python scripts/go1-ros2-test/eval.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Slope20-Play-v0 ^
  --num_envs 64 --eval_steps 3000 --warmup_steps 300 ^
  --load_run 2026-03-08_16-46-27_baseline_rough_ros2cmd ^
  --target_vx 0.8 --pass_cmd_vx_tol 0.05 ^
  --pass_abs_err 0.25 --pass_stable_ratio 0.70 --stable_err_thresh 0.20 ^
  --strict_pass ^
  --summary_json logs/eval/phase3_formal/slope20.json
```

#### 3.7.3 台阶 10cm 评估

```powershell
python scripts/go1-ros2-test/eval.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Stairs10-Play-v0 ^
  --num_envs 64 --eval_steps 3000 --warmup_steps 300 ^
  --load_run 2026-03-08_16-46-27_baseline_rough_ros2cmd ^
  --target_vx 0.8 --pass_cmd_vx_tol 0.05 ^
  --pass_abs_err 0.25 --pass_stable_ratio 0.70 --stable_err_thresh 0.20 ^
  --strict_pass ^
  --summary_json logs/eval/phase3_formal/stairs10.json
```

#### 3.7.4 台阶 15cm 评估

```powershell
python scripts/go1-ros2-test/eval.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Stairs15-Play-v0 ^
  --num_envs 64 --eval_steps 3000 --warmup_steps 300 ^
  --load_run 2026-03-08_16-46-27_baseline_rough_ros2cmd ^
  --target_vx 0.8 --pass_cmd_vx_tol 0.05 ^
  --pass_abs_err 0.35 --pass_stable_ratio 0.50 --stable_err_thresh 0.30 ^
  --strict_pass ^
  --summary_json logs/eval/phase3_formal/stairs15.json
```

> **注意**：Stairs15 使用最宽松的通过标准，因已超出训练地形范围上限（0.1m）。
> 若仍未通过，不视为阶段失败，仅在分析报告中记录 OOD 表现。

### 步骤 3.8：数据收集与对比分析

#### 3.8.1 收集指标

每组评估完成后，统一收集以下指标（从 eval.py 输出日志和 TensorBoard 中提取）：

| 指标                        | 来源                        | 说明                                |
| --------------------------- | --------------------------- | ----------------------------------- |
| `mean_vx_abs_err`         | eval.py 输出                | 速度跟踪绝对误差均值                |
| `stable_ratio`            | eval.py 输出                | 稳定跟踪占比                        |
| `mean_episode_length`     | TensorBoard / 日志          | 平均 episode 长度（越长＝越稳定）   |
| `fall_rate`               | 统计 early termination 占比 | 跌倒率（episode 提前结束的比例）    |
| `track_lin_vel_xy_exp`    | TensorBoard reward 曲线     | 速度跟踪 reward 项                  |
| `cmd_zero_fallback_count` | eval.py 输出                | ROS2 零命令回退次数（确认桥接正常） |

#### 3.8.2 对比分析表

填入以下对比表（数据来自步骤 3.7 的 4 组评估 + 阶段 1 的 Rough 混合地形 baseline）：

| 配置                       | `mean_vx_abs_err` | `stable_ratio` | `mean_episode_len` | `fall_rate` | 通过？ |
| -------------------------- | ------------------- | ---------------- | -------------------- | ------------- | ------ |
| Rough baseline（混合地形） | 0.0798              | 0.9678           | —                   | —            | ✅     |
| Slope 10°                 | —                  | —               | —                   | —            | —     |
| Slope 20°                 | —                  | —               | —                   | —            | —     |
| Stairs 10cm                | —                  | —               | —                   | —            | —     |
| Stairs 15cm                | —                  | —               | —                   | —            | —     |

#### 3.8.3 分析框架

对比时回答以下问题：

1. **地形难度梯度**：Slope10 → Slope20、Stairs10 → Stairs15 的指标退化幅度是否符合预期？
2. **OOD 表现**：Stairs15 超出训练分布，退化程度如何？主要失败模式是什么（跌倒 / 原地打滑 / 速度偏差大）？
3. **坡地 vs 台阶**：同等"难度"下，哪种地形对 Go1 更具挑战性？原因分析。
4. **Reward 项差异**：不同地形下 `track_lin_vel_xy_exp`、`feet_stumble`、`flat_orientation_l2` 等 reward 项的表现差异。
5. **与 Flat baseline 对比**：Rough 模型在专项地形上的表现 vs Flat 模型在平地上的表现，跟踪精度差距量化。

### 步骤 3.9：更新 `docs/reward_engineering.md` 第 6 章

> 阶段 3 的核心文档产出。

#### 第 6 章：地形与 Reward 的交互

**章节目标**（开头一句话）：读完本章，你将理解不同地形类型如何影响各 reward 项的分布，以及 terrain curriculum 如何间接塑造 reward 信号。

**内容大纲**：

1. **地形类型回顾**：Isaac Lab 的 sub_terrains 配置、Go1 baseline 使用的地形类型和参数范围。
2. **专项地形评估结果**：
   - 坡地 10° / 20° 的指标对比（附 TensorBoard/W&B 截图）。
   - 台阶 10cm / 15cm 的指标对比。
   - 对比分析表（步骤 3.8.2 的数据）。
3. **各 Reward 项在不同地形下的表现差异**：
   - `track_lin_vel_xy_exp`：坡地上因重力分量导致速度跟踪偏差增大。
   - `flat_orientation_l2`：坡地上机体倾斜导致该项惩罚显著增加。
   - `feet_stumble`：台阶地形上碰撞检测触发频率更高。
   - `feet_air_time`：台阶地形对步态节奏的扰动。
4. **Terrain Curriculum 与 Reward 的关系**：
   - Curriculum 如何逐步增加地形难度。
   - 难度跃迁时 reward 曲线的波动模式。
   - 训练分布内（Stairs10）vs 训练分布外（Stairs15）的性能差异解读。
5. **跌倒率与 Reward 的关联**：
   - 高跌倒率配置下各 reward 项的统计特征。
   - early termination 对 reward 信号估计的影响。

**写作规范**：遵循 `plan.md` 阶段 2 定义的写作规范（每项 reward 用统一模板：名称→数学形式→物理含义→权重值→调大/调小会怎样）。

---

## 通过标准汇总

| 地形配置    | `mean_vx_abs_err` | `stable_ratio` | `stable_err_thresh` | 是否必须通过     |
| ----------- | ------------------- | ---------------- | --------------------- | ---------------- |
| Slope 10°  | ≤ 0.15             | ≥ 0.85          | 0.15                  | ✅ 必须          |
| Slope 20°  | ≤ 0.25             | ≥ 0.70          | 0.20                  | ✅ 必须          |
| Stairs 10cm | ≤ 0.25             | ≥ 0.70          | 0.20                  | ✅ 必须          |
| Stairs 15cm | ≤ 0.35             | ≥ 0.50          | 0.30                  | ⚠️ 尽力（OOD） |

> Stairs 15cm 已超出训练地形范围上限（baseline `grid_height_range` 上限 0.1m），
> 若未通过不阻塞阶段完成，但需在分析报告和第 6 章中记录原因。

**阶段整体通过条件**：

- Slope 10°、Slope 20°、Stairs 10cm 三组**全部通过** `--strict_pass`。
- Stairs 15cm 评估已执行并记录分析，无论是否通过。
- `docs/reward_engineering.md` 第 6 章完稿。
- 对比分析表已填写完整。

---

## 风险与对策

| # | 风险                                             | 影响                                     | 对策                                                                                 |
| - | ------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------------ |
| 1 | 15cm 台阶超出训练地形分布（OOD）                 | 政策泛化失败，跌倒率高                   | 预设宽松通过标准；记录失败模式而非视为 bug；如需改善可考虑扩展训练地形范围后重新训练 |
| 2 | 单地形评估时 terrain generator 行为异常          | 地形生成不符合预期                       | smoke test 中目视确认地形类型（非 headless 模式跑一次）                              |
| 3 | ROS2 桥接在长时间评估中丢包                      | `cmd_zero_fallback_count` 偏高影响指标 | 每组评估前确认 `cmd_zero_fallback_count` 占比 < 5%；异常时重启 ROS2 topic pub      |
| 4 | 坡地配置中 `slope_range` 参数名与实际 API 不符 | 地形生成参数无效                         | 步骤 3.1 先验证 API，确认参数名再写配置                                              |
| 5 | 评估 checkpoint 与新地形配置的 obs 维度不匹配    | 加载模型失败                             | 所有专项评估继承 `_PLAY` 基类，obs 维度不变（仅改地形，不改观测）                  |
