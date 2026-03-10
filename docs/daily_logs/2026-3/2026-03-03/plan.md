# 2026-03-03 实施计划（v2 修订版）

> 基于 v1 审查发现的 14 项问题修订。变更摘要见末尾附录 A。

## 目标

在 `d:\Graduation-Project` 内直接执行，完成**主机仿真 + ROS2 高层决策**方案下的 Go1 locomotion 复现与扩展：

1. 复现 Isaac Lab locomotion baseline（在 ROS2Cmd 管线内，仅替换 command source）。
2. 实现地形 Curriculum 验证（坡地 10°/20°、台阶 10cm/15cm）。
3. 完成 PPO 参数调优与 Domain Randomization 对比（每组 3 种子）。
4. 产出《深入理解 Reward 工程》讲解文档。

## 复现口径（必须统一）

- 复现对象：Isaac Lab 的 reward / terrain / curriculum / PPO / DR 基线配置。
- ROS2 角色：仅作为高层速度指令输入（`/go1/cmd_vel`）。
- 与纯 Isaac Lab baseline 的唯一区别：command source 由内部采样改为 `Ros2VelocityCommandCfg`。
- **所有 reward 权重、terrain、PPO、DR 配置均继承 Isaac Lab 原始值，不做修改。**
  - `flat_env_cfg.py` 已恢复为纯净版本（v2 修订：删除了 v1 中遗留的 reward 权重覆盖）。
- **复现实验必须加** `--disable_ros2_tracking_tune`，避免脚本对 ROS2 任务自动改学习率和熵系数。

## 当前状态（截至 2026-03-04）

- 已有任务：
  - `Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0`
  - `Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-Play-v0`
- 已完成修复（v2）：
  - `flat_env_cfg.py` reward 权重已恢复为 Isaac Lab 原始值。
  - `eval.py` 已添加 terrain curriculum 禁用逻辑（`terrain_levels = None` + `terrain_generator.curriculum = False`）。
- 当前缺口：
  - 没有 `Rough ROS2Cmd` 任务 ID。
  - `train.py/eval.py` 的 ROS2 任务白名单仅包含 Flat。
- 当前可执行入口（本仓库）
  - 训练：`scripts/go1-ros2-test/train.py`
  - 评估：`scripts/go1-ros2-test/eval.py`

---

## 阶段 0：先打通 Rough ROS2Cmd 闭环（预计 0.5-1 天）

### 0.1 代码改动

1. 新增 `src/go1-ros2-test/envs/rough_env_cfg.py`

   - 继承 `UnitreeGo1RoughEnvCfg` 和 `UnitreeGo1RoughEnvCfg_PLAY`（Isaac Lab Go1 rough baseline）。
   - 仅替换 command term 为 `Ros2VelocityCommandCfg`。
   - 不改 reward/terrain/PPO/DR 默认值（与 Flat ROS2Cmd 保持一致的策略：只换 command source）。
   - **必须同时实现两个配置类**：
     - `UnitreeGo1Ros2CmdRoughEnvCfg`（训练用）
     - `UnitreeGo1Ros2CmdRoughEnvCfg_PLAY`（评估用，继承自 `UnitreeGo1RoughEnvCfg_PLAY`，自动禁用 curriculum 和减少 num_envs）
2. 修改 `src/go1-ros2-test/envs/__init__.py`

   - 新增任务注册：
     - `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0`
     - `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0`
   - **关键：Rough 任务必须引用 `UnitreeGo1RoughPPORunnerCfg`**，不能复用 Flat 的 `UnitreeGo1FlatPPORunnerCfg`。
     - Rough 网络：`[512, 256, 128]`，`max_iterations=1500`
     - Flat 网络：`[128, 128, 128]`，`max_iterations=300`
     - 用错 Runner 会导致网络容量不足，Rough 环境无法收敛。
   - 注册示例：
     ```python
     gym.register(
         id="Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0",
         entry_point="isaaclab.envs:ManagerBasedRLEnv",
         disable_env_checker=True,
         kwargs={
             "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo1Ros2CmdRoughEnvCfg",
             "rsl_rl_cfg_entry_point": (
                 "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents."
                 "rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg"
             ),
         },
     )
     ```
3. 修改 `scripts/go1-ros2-test/train.py`

   - 在 `_ROS2_TASK_IDS` 中加入 rough 的两个 ROS2Cmd 任务 ID。
4. 修改 `scripts/go1-ros2-test/eval.py`

   - 同步加入 rough 的两个 ROS2Cmd 任务 ID。

### 0.2 同步规则（必须，且必须在 smoke test 之前完成）

按 `AGENTS.md` 的 `src -> robot_lab` 规则，同步新增镜像文件：

- `src/go1-ros2-test/envs/rough_env_cfg.py`
  → `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/rough_env_cfg.py`

并在 `AGENTS.md` 的同步表增加该条目。

**注意**：`train.py` 中 ROS2 桥接适配器从 `robot_lab` 包导入（`from robot_lab.ros2_bridge import ...`），因此 `src -> robot_lab` 同步必须在任何训练/评估命令之前完成，否则 import 会失败。

### 0.3 验证 height_scanner 兼容性

Rough 环境比 Flat 多一个 `height_scan` 观测项（高度扫描数据）。需要在 smoke test 中确认：

- height_scanner 的 `prim_path` 为 `{ENV_REGEX_NS}/Robot/trunk`（Go1 特定，不是默认的 `base`）。
- 观测维度正确，训练日志中 `obs_dim` 包含 height_scan 的维度。
- 如果 height_scanner 报错或输出全零，需要检查 USD 资产中 trunk link 的名称。

### 0.4 TDD/验证

- 新增集成测试桩（仿真相关）：

  - `tests/sim_required/test_go1_ros2cmd_rough_registration.py`
  - 使用 `@pytest.mark.sim_required`。
  - 验证 rough ROS2Cmd 任务可被解析/注册。
- Smoke Test（1 iter）

  1. WSL 发布 ROS2 命令：
     ```powershell
     wsl -d Ubuntu-22.04 bash -lc "source /opt/ros/humble/setup.bash; ros2 topic pub /go1/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.8, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' --rate 20"
     ```
  2. Windows 训练（主机仿真）：
     ```powershell
     conda shell.powershell hook | Out-String | Invoke-Expression
     conda activate env_isaaclab
     python scripts/go1-ros2-test/train.py --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0 --num_envs 64 --max_iterations 1 --headless --disable_ros2_tracking_tune
     ```

### 0.5 通过标准

- 训练不报错，能产出 checkpoint。
- `params/ros2.yaml` 生成。
- `params/agent.yaml` 中网络维度为 `[512, 256, 128]`（确认使用了 Rough PPO Runner）。
- 日志中 `cmd_vx` 非全零，`cmd_zero_fallback_count` 非持续高占比。
- 观测维度包含 height_scan（对比 Flat 的 obs_dim 应更大）。

### 0.6 文档触发

Smoke test 通过后，立即创建 `docs/reward_engineering.md` 并完成第 1 章（Reward Shaping 入门）。
此时读者还没有训练数据，第 1 章只需要概念性介绍，为后续章节铺垫。

---

## 阶段 1：ROS2Cmd 基线复现（Flat + Rough）（预计 2-3 天）

### 1.1 实验原则

- 全部使用 ROS2Cmd 任务 ID。
- 全部加 `--disable_ros2_tracking_tune`。
- 先复现，不做自定义 reward，不做参数调优。
- Flat 和 Rough 均使用 Isaac Lab 原始 reward 权重（不做任何覆盖）。

### 1.2 训练矩阵

1. Flat baseline（ROS2Cmd）— `max_iterations=300`
2. Rough baseline（ROS2Cmd）— `max_iterations=1500`（训练时间显著更长）

统一记录：

- `params/env.yaml`
- `params/agent.yaml`
- `params/ros2.yaml`
- TensorBoard/W&B 曲线

W&B 命名规范：

- Flat: `wandb_project=go1-flat-locomotion`，`run_name=baseline_flat_ros2cmd`
- Rough: `wandb_project=go1-rough-locomotion`，`run_name=baseline_rough_ros2cmd`

### 1.3 评估命令

Flat 评估（使用默认通过标准）：

```powershell
python scripts/go1-ros2-test/eval.py --task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-Play-v0 --num_envs 64 --eval_steps 3000 --warmup_steps 300 --strict_pass
```

Rough 评估（使用放宽的通过标准）：

```powershell
python scripts/go1-ros2-test/eval.py --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0 --num_envs 64 --eval_steps 3000 --warmup_steps 300 --pass_abs_err 0.25 --pass_stable_ratio 0.7 --stable_err_thresh 0.2 --strict_pass
```

> Rough 环境的通过标准必须放宽。在粗糙地形上速度跟踪误差天然更大，
> 使用 Flat 的标准（`abs_err≤0.1, stable_ratio≥0.9`）几乎必定失败。
> 初版 Rough 标准：`abs_err≤0.25, stable_ratio≥0.7, stable_err_thresh=0.2`。
> 待首次 baseline 训练完成后，根据实际数据校准这些阈值。

### 1.4 通过标准

- Flat：`mean_vx_abs_err ≤ 0.1`，`stable_ratio ≥ 0.9`（延续已验证标准）。
- Rough：`mean_vx_abs_err ≤ 0.25`，`stable_ratio ≥ 0.7`（初版，待校准）。
- 两者均需确认 `cmd_zero_fallback_count` 占比低于 5%。

### 1.5 文档触发

- Flat 训练完成后：更新 `docs/reward_engineering.md` 第 2 章（Flat Reward 逐项解读），附 TensorBoard 曲线截图。
- Rough 训练完成后：更新第 3 章（Flat vs Rough Reward 差异）。
- 两组 baseline 对比完成后：更新第 4 章（ROS2Cmd 对 Reward 的影响）。

### 1.6 时间说明

Rough 训练 `max_iterations=1500`，按 4096 envs 估算单次训练需数小时。加上 ROS2 桥接调试，总计 2-3 天较为现实（v1 估计 1-2 天偏乐观）。

---

## 阶段 2：Reward 工程讲解文档（贯穿全流程，非独立阶段）

> **核心原则**：`docs/reward_engineering.md` 在阶段 0 smoke test 通过后立即创建，
> 随训练推进逐章更新。阅读难度由易到难，让读者能跟着实验进度自然理解 reward 工程。

### 文档路径

`docs/reward_engineering.md`

### 章节结构与写入时机

| 章节                                          | 内容                                                                                                                                     | 写入时机                    | 难度       |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | ---------- |
| 第 1 章：什么是 Reward Shaping                | RL reward 的基本概念、为什么需要多项 reward 组合、Isaac Lab reward 框架简介（`RewardTermCfg` 结构）                                    | 阶段 0 完成后               | 入门       |
| 第 2 章：Go1 Flat Baseline 的 Reward 逐项解读 | 逐条解释 Flat baseline 的每个 reward 项：数学形式、物理含义、权重值、调参直觉。附 TensorBoard 截图标注各项曲线                           | 阶段 1 Flat 训练完成后      | 入门→基础 |
| 第 3 章：Flat vs Rough 的 Reward 差异         | 对比两套配置的 reward 权重差异、Rough 新增项（如 `feet_stumble`、`flat_orientation_l2`）、为什么 Rough 需要更多正则化项              | 阶段 1 Rough 训练完成后     | 基础       |
| 第 4 章：ROS2Cmd 对 Reward 的影响             | 命令时延/丢包如何影响 `track_lin_vel_xy_exp`、零命令回退对 reward 曲线的扰动、`cmd_zero_fallback_count` 与 reward 的关联分析         | 阶段 1 两组 baseline 对比后 | 基础→进阶 |
| 第 5 章：Reward 权重调优实验                  | 单变量实验设计（One-factor-at-a-time）、`track_lin_vel_xy_exp` 权重提升实验（如 3.5）、`action_rate_l2` 微调、baseline vs 变体对比图 | 阶段 2 实验完成后           | 进阶       |
| 第 6 章：地形与 Reward 的交互                 | 不同地形（坡地/台阶）下各 reward 项的表现差异、terrain curriculum 如何间接影响 reward 分布、跌倒率与 reward 的关系                       | 阶段 3 完成后               | 进阶       |
| 第 7 章：PPO/DR 改动如何反映到 Reward 曲线    | 学习率/clip_param/entropy_coef 对 reward 收敛速度的影响、DR 扰动（摩擦/质量/外推力）对 reward 稳定性的影响、3 种子统计分析方法           | 阶段 4 完成后               | 高级       |

### 写作规范

- 每章开头用一句话总结"读完本章你会理解什么"。
- 数学公式用 LaTeX 行内格式，紧跟直觉解释（"这个公式的意思是……"）。
- 每个 reward 项的解释遵循统一模板：`名称 → 数学形式 → 物理含义 → 权重值 → 调大/调小会怎样`。
- 配图优先使用 TensorBoard/W&B 截图，标注关键拐点。
- 章节之间用"上一章我们了解了 X，本章在此基础上讨论 Y"衔接。

### 阶段 2 的定位调整

阶段 2 不再承担"从零写文档"的任务，而是聚焦于：

1. **补完第 4-5 章**（基于阶段 1 的训练数据）。
2. **执行 reward 权重调优实验**：
   - 先冻结 baseline，再做自定义项。
   - 每次只改一个变量（One-factor-at-a-time）。
   - 备选自定义项：
     - `track_lin_vel_xy_exp` 权重提升（如 3.5，复现 v1 中的调参尝试）
     - 动作平滑强化（在已有 `action_rate_l2` 基础上微调）
     - `feet_slide`/`feet_stumble` 类项
   - 之前 `flat_env_cfg.py` 中的 reward 权重调整（`track_lin_vel_xy_exp=3.5` 等）现在作为本阶段的一个正式实验变体，而非烧进环境配置。
3. **产物**：第 5 章完稿 + baseline vs 单项改动的对比图。

---

## 阶段 3：地形 Curriculum 目标化验证（预计 2-3 天）

### 3.1 目标

在 ROS2Cmd rough 环境下验证：

- 坡地 10° / 20°
- 台阶 10cm / 15cm

### 3.2 方法

1. 使用 rough baseline 先跑全地形 curriculum（阶段 1 的产物）。
2. 创建专项评估地形配置，精确控制单一地形参数：
   - Isaac Lab 的 terrain generator 使用 `sub_terrains` 字典配置。
   - 专项评估需要自定义 terrain generator 配置，只保留目标地形类型。
   - **坡地评估**：创建只包含 `hf_pyramid_slope` 的配置，设 `slope_range` 为：
     - 10°: `slope_range=(0.176, 0.176)`（`tan(10°)≈0.176`）
     - 20°: `slope_range=(0.364, 0.364)`（`tan(20°)≈0.364`）
   - **台阶评估**：创建只包含 `pyramid_stairs` 的配置，设 `step_height_range` 为：
     - 10cm: `step_height_range=(0.10, 0.10)`
     - 15cm: `step_height_range=(0.15, 0.15)`
   - 实现方式：在 `rough_env_cfg.py` 中新增专项评估配置类，或通过 Hydra override 在命令行覆盖。
3. 对比 flat vs rough 曲线差异：
   - `track_lin_vel_xy_exp`
   - `vx_abs_err`
   - episode length
   - 跌倒率/存活时间

### 3.3 通过标准（初版）

- 坡地 10°：稳定通过，长时不跌倒。
- 坡地 20°：可持续行走并保持可接受误差。
- 台阶 10cm/15cm：达到预设通过率阈值（在日志中明确阈值）。

> **v2 注意**：Go1 的 terrain 参数已针对小型机器人缩放（`boxes.grid_height_range=(0.025, 0.1)`），
> 15cm 台阶已超出 baseline 训练范围上限（0.1m），可能需要额外训练或调整预期。

### 3.4 文档触发

阶段 3 完成后，更新 `docs/reward_engineering.md` 第 6 章（地形与 Reward 的交互）。

---

## 阶段 4：PPO 调优 + DR（预计 3-4 天）

### 4.1 PPO 调参（在 ROS2Cmd rough 上）

参数组：

- `learning_rate`：baseline 1e-3，候选 5e-4 / 3e-3
- `clip_param`：baseline 0.2，候选 0.1 / 0.3
- `entropy_coef`：baseline 0.01，候选 0.005 / 0.02

约束：

- 单次仅改一个参数。
- 其余保持 baseline。
- **每组 3 个种子**（seed=42, 123, 456），取均值±标准差。
- 结论需基于统计显著性，而非单次观察。

### 4.2 DR 实验

Go1 Rough baseline 的 DR 配置现状（需准确理解）：

- `push_robot = None`（**已禁用**，不是"baseline DR 包含 push"）
- `add_base_mass`：范围 `(-1.0, 3.0)` kg（相对保守）
- `base_com = None`（**已禁用**）
- 物理材料摩擦：`static=0.8, dynamic=0.6`（固定值，未做范围随机化）
- 观测噪声：已启用（各项 ±Uniform 噪声）

实验组设计：

1. **Baseline DR**（即上述配置，作为对照组）
2. **摩擦随机化**：将 `static_friction_range` 从 `(0.8, 0.8)` 扩展为 `(0.5, 1.2)`，`dynamic_friction_range` 从 `(0.6, 0.6)` 扩展为 `(0.4, 1.0)`
3. **质量扰动扩展**：将 `mass_distribution_params` 从 `(-1.0, 3.0)` 扩展为 `(-3.0, 5.0)`
4. **外部推力启用**：将 `push_robot` 从 `None` 恢复为 `EventTerm(interval_range_s=(10.0, 15.0), velocity_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5)})`
   - 注意：这是**新增**扰动，不是"增强"，实验报告中需准确描述。

每组 3 个种子。

### 4.3 通过标准

- 收敛稳定性不劣于 baseline。
- 泛化指标（粗糙地形/扰动场景）有可解释改进。

### 4.4 时间说明

每组参数 × 3 种子 × Rough 训练时长，总训练量显著增加。PPO 3 参数 × 2 候选 = 6 组 × 3 种子 = 18 次训练；DR 3 组 × 3 种子 = 9 次训练。合计 27 次 Rough 训练，需合理安排并行。预计 3-4 天。

### 4.5 文档触发

阶段 4 完成后，更新 `docs/reward_engineering.md` 第 7 章（PPO/DR 改动如何反映到 Reward 曲线），完成全部 7 章。

---

## 阶段 5：总结与交付（预计 0.5-1 天）

### 5.1 文档

1. `docs/reward_engineering.md`（全 7 章已在各阶段逐步完成，此处做最终审校和润色）
2. `docs/daily_logs/2026-03-03/2026-3-3.md`（执行日志）
3. 最终实验对比表（baseline / terrain / PPO / DR），包含均值±标准差

### 5.2 模型与记录

- 统一存档到 `logs/rsl_rl/...`
- 标注最佳 checkpoint 及对应配置哈希（env/agent/ros2）

---

## 命令规范（本仓库直接执行）

### PowerShell 环境初始化

```powershell
conda shell.powershell hook | Out-String | Invoke-Expression
conda activate env_isaaclab
```

### 训练入口（统一）

```powershell
python scripts/go1-ros2-test/train.py --task <TASK_ID> --num_envs <N> --max_iterations <K> --headless --disable_ros2_tracking_tune
```

### 评估入口（统一）

Flat:

```powershell
python scripts/go1-ros2-test/eval.py --task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-Play-v0 --num_envs 64 --eval_steps 3000 --warmup_steps 300 --strict_pass
```

Rough（注意放宽的通过标准）:

```powershell
python scripts/go1-ros2-test/eval.py --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0 --num_envs 64 --eval_steps 3000 --warmup_steps 300 --pass_abs_err 0.25 --pass_stable_ratio 0.7 --stable_err_thresh 0.2 --strict_pass
```

---

## 风险与对策

1. 风险：ROS2 桥接消息丢失导致 `cmd_zero_fallback_count` 偏高。

   - 对策：先做 1 iter + 短程评估 smoke，先排链路问题再跑长训。
2. 风险：误把 ROS2 自动 tuning 当 baseline。

   - 对策：复现实验强制 `--disable_ros2_tracking_tune`。
3. 风险：`src -> robot_lab` 未同步导致运行配置与源码不一致。

   - 对策：每次提交前检查镜像 diff，并更新 AGENTS 同步表。
   - **同步必须在 smoke test / 训练之前完成**（train.py 从 robot_lab 包导入桥接适配器）。
4. 风险：Rough 任务使用错误的 PPO Runner（Flat 的小网络）。

   - 对策：smoke test 通过标准中检查 `params/agent.yaml` 的网络维度。
5. 风险：height_scanner 在 ROS2Cmd 环境中不工作。

   - 对策：smoke test 中验证观测维度包含 height_scan，且数值非全零。
6. 风险：15cm 台阶超出 Go1 baseline 训练的地形范围上限（0.1m）。

   - 对策：阶段 3 中如果 15cm 台阶表现差，考虑扩展训练地形范围或调整预期。

---

## 时间线（v2 修订）

- 阶段 0：0.5-1 天
- 阶段 1：2-3 天（Rough 训练时间长，v1 低估）
- 阶段 2：1-2 天
- 阶段 3：2-3 天（需自定义地形配置，v1 低估）
- 阶段 4：3-4 天（3 种子 × 多组实验，v1 低估）
- 阶段 5：0.5-1 天
- 合计：10-15 天

---

## 附录 A：v1 → v2 变更清单

| #  | 问题                                            | 修复                                                                                                |
| -- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 1  | Rough 任务注册引用了 Flat 的 PPO Runner         | 阶段 0.1 明确要求使用 `UnitreeGo1RoughPPORunnerCfg`，并给出注册示例                               |
| 2  | eval.py 未禁用 Rough 的 terrain curriculum      | **已修复代码**：eval.py 添加 `terrain_levels=None` + `terrain_generator.curriculum=False` |
| 3  | eval.py 的 Rough 通过标准过严                   | 阶段 1.3 为 Rough 提供放宽的评估参数（`abs_err≤0.25, stable_ratio≥0.7`）                        |
| 4  | flat_env_cfg.py 的 reward 权重不是原始 baseline | **已修复代码**：删除 reward 权重覆盖，恢复为纯净 baseline                                     |
| 5  | Rough ROS2Cmd 的 reward 权重策略未明确          | 阶段 0.1 明确：与 Flat 一致，只换 command source，不改 reward                                       |
| 6  | 缺少 `_PLAY` 配置类说明                       | 阶段 0.1 显式要求实现 `_PLAY` 配置类                                                              |
| 7  | 未验证 height_scanner 兼容性                    | 新增阶段 0.3 和 0.5 中的 height_scanner 验证步骤                                                    |
| 8  | 地形专项评估方案不具体                          | 阶段 3.2 给出具体的 slope_range/step_height_range 参数和实现方式                                    |
| 9  | PPO 调参只跑 1 个种子                           | 阶段 4.1 改为 3 个种子（42, 123, 456）                                                              |
| 10 | DR baseline 定义模糊                            | 阶段 4.2 详细列出 Go1 Rough baseline DR 的实际配置和各实验组设计                                    |
| 11 | W&B 命名规范缺失                                | 阶段 1.2 添加 W&B 项目和 run 命名规范                                                               |
| 12 | robot_lab 同步时序未强调                        | 阶段 0.2 添加"必须在 smoke test 之前完成"的说明                                                     |
| 13 | 时间估算偏乐观                                  | 阶段 1/3/4 时间上调，合计从 7-10 天调整为 10-15 天                                                  |
| 14 | 复现口径措辞与代码矛盾                          | 复现口径章节修正措辞，明确"所有 reward 权重继承原始值"                                              |
