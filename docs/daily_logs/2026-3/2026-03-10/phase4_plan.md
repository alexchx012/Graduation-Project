# 阶段 4 详细计划：PPO 调优 + Domain Randomization

> 基于 `docs/daily_logs/2026-3/2026-03-03/plan.md` §4.1–4.5 展开。
> 预计 **5 天**。

---

## 设计要点

| 主题 | 方案 | 理由 |
| --- | --- | --- |
| PPO 变体实现 | CLI 覆盖参数 `--learning_rate` / `--clip_param` / `--entropy_coef` | 避免过度工程化（12 个新任务 ID），且规避 `_apply_ros2_tracking_tune` 静默覆盖风险 |
| Baseline 种子 | 前置必做步骤 4.0，补齐 3 种子 | 无 3 种子 baseline 则无法做统计对比 |
| ROS2 Publisher | 批量脚本内置 watchdog | 45h 串行训练需要 publisher 保活 |
| DR 评估 | 标准环境 + 抗扰动交叉评估 | 需验证 DR 的泛化价值 |
| 组合实验 | 步骤 4.6b：best PPO + best DR | 为"最佳配置推荐"提供实验依据 |

---

## 前置条件

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| Phase 0-3 | ✅ 全部完成 | Flat/Rough baseline + OFAT + 地形验证 |
| Rough baseline checkpoint | ✅ 可用 | `logs/rsl_rl/unitree_go1_rough/2026-03-08_16-46-27_baseline_rough_ros2cmd/model_1499.pt` |
| Rough baseline 配置 | ✅ 已确认 | 4096 envs, 1500 iters, `[512,256,128]`, `height_scan(187)` |
| eval.py | ✅ 已更新 | 支持 `--target_vx` 和放宽通过标准 |
| reward_engineering.md | ✅ Ch1-6 完成 | 待本阶段完成后写 Ch7 |
| **Baseline 种子数** | ⚠️ **仅 1 个** | 必须在步骤 4.0 补训 seed=123, 456 |

---

## 步骤 4.0：补训 Baseline 3 种子

> **目标**：确保 baseline 拥有 3 种子数据，使后续所有统计对比有效。
>
> **[Fact]** 现有 baseline 仅 1 个 checkpoint（seed 未知）。没有 baseline 3 种子的均值±标准差，
> 步骤 4.8 的 t-test / 效应量分析全部无法执行。这不是"可选决策"，是**阻塞项**。

**操作**：

1. 从现有 baseline checkpoint 的 `params/agent.yaml` 确认其 seed
2. 补训剩余 2 个种子（从 `{42, 123, 456}` 中排除已有 seed）
3. 使用与 Phase 1 完全一致的训练命令：

```powershell
# 示例：补训 seed=123（另一个 seed=456 同理）
python scripts/go1-ros2-test/train.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0 ^
  --num_envs 4096 --max_iterations 1500 --headless ^
  --disable_ros2_tracking_tune --seed 123 ^
  --run_name baseline_rough_seed123
```

**时间估算**：2 × 2.5h = 5h（可在 Day 1 其他工作的间隙串行运行）

**产出**：3 个 baseline checkpoint + 对应 `params/agent.yaml`

---

## 步骤 4.1：确认 Rough baseline PPO/DR 参数

> 目标：从代码中精确提取当前 baseline 的 PPO 超参和 DR 配置，作为实验基准。

**操作**：

1. 查看 `UnitreeGo1RoughPPORunnerCfg`（位于 IsaacLab 安装包内：
   `isaaclab_tasks/manager_based/locomotion/velocity/config/go1/agents/rsl_rl_ppo_cfg.py`）
   > 注意：该配置定义在 IsaacLab 安装包内，不在本仓库 `src/` 下。
2. 记录以下 PPO 参数：
   - `learning_rate`（预期 1e-3）
   - `clip_param`（预期 0.2）
   - `entropy_coef`（预期 0.01）
   - `num_learning_epochs`、`num_mini_batches`、`gamma`、`lam`
   - `schedule`（是否有 LR 线性衰减？）、`desired_kl`（是否有自适应 LR？）
     > 如果存在 LR schedule，PPO-LR 实验只改初始值，后期差异会被衰减压缩，需在分析中说明。
3. 查看 `UnitreeGo1Ros2CmdRoughEnvCfg` 中的 DR 配置（继承自 `UnitreeGo1RoughEnvCfg`）：
   - `push_robot`（预期 `None`）
   - `add_base_mass`（预期 `(-1.0, 3.0)`）
   - `base_com`（预期 `None`）
   - 物理材料摩擦（预期 `static=0.8, dynamic=0.6` 固定值）
   - 观测噪声配置
4. 将确认结果写入工作日志

**产出**：baseline 参数确认表（写入 `2026-3-10.md`），包含 LR schedule 信息

---

## 步骤 4.2：CLI PPO 覆盖方案

> 目标：为 PPO 调参实验提供参数覆盖机制。
>
> **设计决策**：采用 CLI 参数覆盖而非为每个 PPO 变体创建独立任务 ID。原因：
>
> 1. **防止静默覆盖**：若创建新任务 ID 并加入 `_ROS2_TASK_IDS`，忘传 `--disable_ros2_tracking_tune` 时
>    `_apply_ros2_tracking_tune()` 会静默将 `learning_rate` 覆盖为 `5e-4`、`entropy_coef` 覆盖为 `0.002`，
>    导致实验数据作废且**不会报错**。
> 2. **避免过度工程化**：PPO 参数属于 Agent Config，不属于 Env Config。创建 12 个任务 ID 来改 3 个数字，
>    与已有的 `--seed`/`--max_iterations`/`--num_envs` CLI 覆盖模式不一致。
> 3. **无需同步 robot_lab**、无需更新 `__init__.py`、无需更新 `_ROS2_TASK_IDS`。

**实验矩阵（不变）**：

| 实验 ID | 改动参数 | baseline 值 | 候选值 |
| --- | --- | --- | --- |
| PPO-LR-Low | `learning_rate` | 1e-3 | **5e-4** |
| PPO-LR-High | `learning_rate` | 1e-3 | **3e-3** |
| PPO-Clip-Low | `clip_param` | 0.2 | **0.1** |
| PPO-Clip-High | `clip_param` | 0.2 | **0.3** |
| PPO-Ent-Low | `entropy_coef` | 0.01 | **0.005** |
| PPO-Ent-High | `entropy_coef` | 0.01 | **0.02** |

**操作**：

1. 修改 `scripts/go1-ros2-test/train.py`，在 argparse 中新增 3 个可选参数：

```python
parser.add_argument("--learning_rate", type=float, default=None,
                    help="Override PPO learning rate.")
parser.add_argument("--clip_param", type=float, default=None,
                    help="Override PPO clip parameter.")
parser.add_argument("--entropy_coef", type=float, default=None,
                    help="Override PPO entropy coefficient.")
```

2. 在 `main()` 中 `agent_cfg` 加载后、Runner 创建前添加覆盖逻辑：

```python
# PPO hyperparameter CLI overrides (Phase 4 experiments)
if args_cli.learning_rate is not None:
    agent_cfg.algorithm.learning_rate = args_cli.learning_rate
if args_cli.clip_param is not None:
    agent_cfg.algorithm.clip_param = args_cli.clip_param
if args_cli.entropy_coef is not None:
    agent_cfg.algorithm.entropy_coef = args_cli.entropy_coef
```

3. **关键**：PPO 覆盖必须在 `_apply_ros2_tracking_tune()` 调用**之后**，确保 CLI 显式值优先。
   代码顺序：`update_rsl_rl_cfg` → `_apply_ros2_tracking_tune` → **PPO CLI overrides**

4. 添加安全日志——覆盖生效时打印确认：

```python
if any([args_cli.learning_rate, args_cli.clip_param, args_cli.entropy_coef]):
    print(f"[PPO Override] lr={agent_cfg.algorithm.learning_rate}, "
          f"clip={agent_cfg.algorithm.clip_param}, "
          f"ent={agent_cfg.algorithm.entropy_coef}")
```

5. TDD：新增测试验证 CLI 参数能正确覆盖 agent_cfg（可为单元测试，不需仿真）

**训练命令示例**：

```powershell
# 直接复用现有 Rough ROS2Cmd 任务，无需新任务 ID
python scripts/go1-ros2-test/train.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0 ^
  --num_envs 4096 --max_iterations 1500 --headless ^
  --disable_ros2_tracking_tune --seed 42 ^
  --learning_rate 5e-4 --run_name ppo_lr_low_seed42
```

**产出**：`train.py` 新增 3 个 CLI 参数 + 覆盖逻辑 + 单元测试

---

## 步骤 4.3：创建 DR 变体配置

> 目标：为 3 组 DR 实验创建环境配置文件。
>
> 每个变体提供伪代码级实现，待步骤 4.1 确认参数后以 Isaac Lab 源码为准修正。

**实验矩阵**：

| 实验 ID | 改动项 | baseline 值 | 实验值 |
| --- | --- | --- | --- |
| DR-Friction | 摩擦随机化 | `static=0.8, dynamic=0.6`（固定） | `static=(0.5,1.2), dynamic=(0.4,1.0)` |
| DR-Mass | 质量扰动扩展 | `(-1.0, 3.0)` kg | `(-3.0, 5.0)` kg |
| DR-Push | 外部推力启用 | `push_robot=None` | `interval=(10,15)s, vel=(-0.5,0.5)` |

**操作**：

1. 创建 `src/go1-ros2-test/envs/rough_env_cfg_dr_variants.py`
2. 为每个实验定义 `EnvCfg` 子类 + `_PLAY` 子类，伪代码如下：

```python
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as base_mdp

from .rough_env_cfg import (
    UnitreeGo1Ros2CmdRoughEnvCfg,
    UnitreeGo1Ros2CmdRoughEnvCfg_PLAY,
)

# --- DR-Friction ---
@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_DRFriction(UnitreeGo1Ros2CmdRoughEnvCfg):
    """Rough ROS2Cmd + friction randomization."""
    def __post_init__(self):
        super().__post_init__()
        # 覆盖物理材料的摩擦系数为范围随机化
        # 需要确认 Isaac Lab 中的实际 API：
        # 方案 A: 通过 event term 随机化 rigid body material
        self.events.randomize_friction = EventTerm(
            func=base_mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": ...,  # 步骤 4.1 确认
                "static_friction_range": (0.5, 1.2),
                "dynamic_friction_range": (0.4, 1.0),
            },
        )

# --- DR-Mass ---
@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_DRMass(UnitreeGo1Ros2CmdRoughEnvCfg):
    """Rough ROS2Cmd + extended mass perturbation."""
    def __post_init__(self):
        super().__post_init__()
        # 扩大已有 add_base_mass 的范围
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 5.0)

# --- DR-Push ---
@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_DRPush(UnitreeGo1Ros2CmdRoughEnvCfg):
    """Rough ROS2Cmd + external push perturbation (NEW, not enhancement)."""
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = EventTerm(
            func=base_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
            },
        )
```

> **[Assumption]** 上述 API 调用基于 Isaac Lab 常见模式，步骤 4.1 确认参数后可能需要微调。
> 特别是 `randomize_rigid_body_material` 的 `asset_cfg` 参数需要指向正确的 robot asset。
> 如果 API 不匹配，需以 Isaac Lab 源码为准修正。

3. 每个 Train 配置需配对一个 `_PLAY` 子类（继承对应 DR Train 配置 + `_PLAY` 的 num_envs/curriculum 设置）
4. 在 `__init__.py` 注册 3×2=6 个新任务 ID：

| 任务 ID | 类型 |
| --- | --- |
| `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Friction-v0` | Train |
| `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Friction-Play-v0` | Play |
| `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Mass-v0` | Train |
| `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Mass-Play-v0` | Play |
| `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Push-v0` | Train |
| `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Push-Play-v0` | Play |

5. 更新 `train.py` / `eval.py` 的 `_ROS2_TASK_IDS` 白名单
6. 同步 `rough_env_cfg_dr_variants.py` → `robot_lab/` 镜像
7. 更新 `CLAUDE.md` 同步表

> **注意**：DR 变体改的是 **Env Config**（`self.events`），与 PPO 调参不同，
> 确实需要新建配置类和任务 ID。这与 Phase 2 的 `flat_env_cfg_variants.py` 模式一致。

**产出**：3 个 DR 变体配置 + 6 个任务注册

---

## 步骤 4.4：Smoke Test（PPO 6 组 + DR 3 组 = 9 组）

> 目标：验证所有变体配置的链路正确性。

**操作**：

1. 每组用 `num_envs=64, max_iterations=1` 运行 smoke test
2. PPO 变体命令示例：

```powershell
python scripts/go1-ros2-test/train.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0 ^
  --num_envs 64 --max_iterations 1 --headless ^
  --disable_ros2_tracking_tune ^
  --learning_rate 5e-4 --run_name smoke_ppo_lr_low
```

3. DR 变体命令示例：

```powershell
python scripts/go1-ros2-test/train.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Friction-v0 ^
  --num_envs 64 --max_iterations 1 --headless ^
  --disable_ros2_tracking_tune
```

4. **核验项**（防止 `_apply_ros2_tracking_tune` 静默覆盖）：

| 核验点 | 方法 | 通过条件 |
| --- | --- | --- |
| 任务注册 | checkpoint 目录创建 | ✅ 存在 |
| PPO 参数生效 | 检查 `params/agent.yaml` | `learning_rate` / `clip_param` / `entropy_coef` 与实验设计一致 |
| DR 事件配置 | 训练日志中搜索事件名 | 摩擦/质量/推力参数出现在日志中 |
| reward 项 | 日志中 reward 项名称 | 与 baseline 一致（DR 不改 reward） |

**产出**：9/9 smoke test 通过记录 + `params/agent.yaml` 核验截图

---

## 步骤 4.4b：创建批量训练/评估脚本

> 目标：自动化 27+ 次训练和 33+ 次评估的执行、ROS2 publisher 保活、失败检测。
>
> **设计原因**：27 次串行训练跨越 45+ 小时，WSL publisher 可能 OOM Kill / WSL 休眠 / DDS 断连。
> 任何一次中断导致 `cmd_zero_fallback_count` 飙升，该次训练数据作废。

**操作**：

1. 创建 `scripts/go1-ros2-test/run/phase4/run_phase4_ppo_sweep.ps1`：
   - 输入参数矩阵（6 组 PPO × 3 种子）
   - 每次训练前启动/验证 WSL ROS2 publisher（复用 Phase 1 脚本模式）
   - publisher 存活 watchdog：训练过程中每 5 分钟检查 WSL 进程
   - 训练结束后验证 checkpoint 文件存在
   - 失败时记录日志并跳过（不中断整个 sweep）
   - 自动调用 `--run_name ppo_{param}_{value}_seed{N}` 命名

2. 创建 `scripts/go1-ros2-test/run/phase4/run_phase4_dr_sweep.ps1`：
   - 同样模式，3 组 DR × 3 种子

3. 创建 `scripts/go1-ros2-test/run/phase4/run_phase4_eval_all.ps1`：
   - 遍历所有 checkpoint 目录
   - 自动执行 eval.py 并保存 JSON 到 `logs/eval/phase4_formal/`
   - 评估过程中同样需要 ROS2 publisher

4. **发散判定**：
   - 定义发散标准：训练完成后检查最终 `mean_reward`
   - 若最终 reward < baseline 收敛后 reward 的 30%，标记为发散
   - 发散的训练仍保留 checkpoint 和日志，但在汇总表中标注

**产出**：3 个 `.ps1` 批量脚本

---

## 步骤 4.5：正式训练 — PPO 实验（6 组 × 3 种子 = 18 次）

> 目标：在 Rough ROS2Cmd 环境上训练 6 组 PPO 变体，每组 3 个种子。

**训练参数**：

- `num_envs=4096`
- `max_iterations=1500`（与 baseline 一致）
- `--headless --disable_ros2_tracking_tune`
- 种子：`seed=42, 123, 456`

**命令模板**（通过 CLI 覆盖，无需新任务 ID）：

```powershell
python scripts/go1-ros2-test/train.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0 ^
  --num_envs 4096 --max_iterations 1500 --headless ^
  --disable_ros2_tracking_tune --seed 42 ^
  --learning_rate 5e-4 --run_name ppo_lr_low_seed42
```

> 使用 `--learning_rate` / `--clip_param` / `--entropy_coef` CLI 覆盖，
> 不再创建独立任务 ID。`--task` 始终为 `Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0`。

**运行日志命名规范**：`{date}_{time}_ppo_{param}_{value}_seed{N}`

**并行策略**：

- 单 GPU 环境下只能串行，每次训练约 2-3 小时
- 18 次训练 × 2.5h ≈ 45h → 约 2 天连续运行
- 建议按参数分组执行：先跑 LR（6 次），再 Clip（6 次），最后 Ent（6 次）
- 使用步骤 4.4b 的批量脚本自动执行

**产出**：18 个 checkpoint + 训练日志

---

## 步骤 4.6：正式训练 — DR 实验（3 组 × 3 种子 = 9 次）

> 目标：在 3 种 DR 变体上训练，每组 3 个种子。

**训练参数**：与 PPO 实验相同（`4096 envs, 1500 iters`）

**命令模板**：

```powershell
python scripts/go1-ros2-test/train.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Friction-v0 ^
  --num_envs 4096 --max_iterations 1500 --headless ^
  --disable_ros2_tracking_tune --seed 42
```

> DR 变体需要使用各自的任务 ID（不同于 PPO 的 CLI 覆盖方式），因为 DR 修改的是 Env Config。

**运行日志命名规范**：`{date}_{time}_dr_{type}_seed{N}`

**时间估算**：9 次 × 2.5h ≈ 22.5h → 约 1 天

**产出**：9 个 checkpoint + 训练日志

---

## 步骤 4.6b：组合验证实验

> 目标：验证 best PPO + best DR 的组合效果，为"最佳配置推荐"提供实验依据。
>
> **设计原因**：PPO 和 DR 实验完全独立，没有任何组合实验。
> 步骤 4.9 第 7 章第 5-6 节承诺分析"交互效应"和"最佳配置推荐"，
> 但没有组合实验数据，这两节只能是猜测。

**前置条件**：步骤 4.5 + 4.6 训练全部完成，且已初步分析出 best PPO 和 best DR。

**操作**：

1. 根据步骤 4.5/4.6 的训练曲线，快速判定 best PPO 和 best DR 配置
   - 判定标准：最终 100 iter 平均 episode reward 最高者
2. 训练组合配置：1 组 × 3 种子

```powershell
# 示例：best PPO 为 lr=5e-4，best DR 为 DR-Push
python scripts/go1-ros2-test/train.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-DR-Push-v0 ^
  --num_envs 4096 --max_iterations 1500 --headless ^
  --disable_ros2_tracking_tune --seed 42 ^
  --learning_rate 5e-4 --run_name combo_best_seed42
```

> CLI PPO 覆盖 + DR 任务 ID 的组合自然支持此用法，无需额外代码。

**时间估算**：3 × 2.5h = 7.5h（可与步骤 4.7 评估交错执行）

**降级方案**：若时间不足，只训练 1 个种子（seed=42）作为探索性实验，
在分析中明确标注"单种子结果，仅供参考"。

**产出**：1-3 个组合配置 checkpoint

---

## 步骤 4.7：评估

> 目标：对所有训练完成的模型进行标准化评估。

### 4.7.1 标准环境评估（公平对比）

所有模型在**相同的 baseline Rough 混合地形环境**下评估：

- `--task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0`
- `num_envs=64, eval_steps=3000, warmup_steps=300`
- ROS2 publisher `vx=0.8, 20Hz`
- `--target_vx 0.8`
- 通过标准：与 Rough baseline 评估一致（`pass_abs_err=0.25, pass_stable_ratio=0.7, stable_err_thresh=0.2`）

> **注意**：eval.py 中 cmd_vx 对 target_vx 的容差已硬编码为 0.05（第 438 行），
> 无需通过命令行传入。

**评估命令模板**：

```powershell
python scripts/go1-ros2-test/eval.py ^
  --task Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0 ^
  --num_envs 64 --eval_steps 3000 --warmup_steps 300 ^
  --target_vx 0.8 --pass_abs_err 0.25 --pass_stable_ratio 0.7 ^
  --stable_err_thresh 0.2 --seed 42 ^
  --load_run <run_dir> --checkpoint model_1499.pt ^
  --summary_json logs/eval/phase4_formal/ppo/ppo_lr_low_seed42.json
```

> 添加 `--seed 42` 保证评估可复现性。

**评估矩阵**：

| 类别 | 实验组数 | 种子数 | 评估次数 |
| --- | --- | --- | --- |
| Baseline（对照） | 1 | 3 | 3 |
| PPO 变体 | 6 | 3 | 18 |
| DR 变体 | 3 | 3 | 9 |
| 组合验证 | 1 | 1-3 | 1-3 |
| **合计** | | | **31-33** |

### 4.7.2 抗扰动交叉评估（DR 泛化验证）

> 仅在标准环境评估 DR 变体无法验证 DR 训练是否真正增强了鲁棒性。

选取 baseline + 3 个 DR 变体的 best seed checkpoint，在 DR-Push Play 环境下交叉评估：

| 被评估模型 | 评估环境 | 目的 |
| --- | --- | --- |
| Baseline | DR-Push-Play | baseline 在推力扰动下的表现（对照） |
| DR-Friction | DR-Push-Play | 摩擦 DR 对推力扰动的迁移能力 |
| DR-Mass | DR-Push-Play | 质量 DR 对推力扰动的迁移能力 |
| DR-Push | DR-Push-Play | 推力 DR 在同分布下的优势 |

> 需要为 DR-Push 创建 `_PLAY` 评估环境（步骤 4.3 已包含）。
> 此部分为 4 次评估，时间代价低（~20min），信息量高。

**产出**：31-33 个标准评估 JSON + 4 个交叉评估 JSON

---

## 步骤 4.8：数据分析

> 目标：基于 3 种子统计，分析各 PPO/DR 变体对性能的影响。

### 4.8.1 汇总表

PPO 实验汇总表（均值 ± 标准差）：

| 配置 | `mean_vx_abs_err` | `stable_ratio` | `base_contact_rate` | 收敛 iter |
| --- | --- | --- | --- | --- |
| Baseline | μ ± σ | μ ± σ | μ ± σ | — |
| PPO-LR-Low | μ ± σ | μ ± σ | μ ± σ | — |
| PPO-LR-High | μ ± σ | μ ± σ | μ ± σ | — |
| PPO-Clip-Low | μ ± σ | μ ± σ | μ ± σ | — |
| PPO-Clip-High | μ ± σ | μ ± σ | μ ± σ | — |
| PPO-Ent-Low | μ ± σ | μ ± σ | μ ± σ | — |
| PPO-Ent-High | μ ± σ | μ ± σ | μ ± σ | — |

DR 实验汇总表（同格式）+ 组合验证行。

> "收敛 iter"形式化定义：
> 连续 100 iter 的 episode_reward 滑动平均变化 < 1% 时的首个 iter。

### 4.8.2 分析框架

对比时回答以下问题：

1. **收敛速度**：哪个 PPO 参数对收敛速度影响最大？LR 还是 Clip 还是 Entropy？
2. **最终性能**：哪组配置的 `mean_vx_abs_err` 最低？改善是否统计显著？
3. **稳定性**：哪组的种子间方差最小（训练最稳定）？
4. **DR 泛化**：DR 变体在标准环境 vs 抗扰动环境的表现差异？
5. **安全性**：DR 变体的 `base_contact_rate` 是否有变化？外部推力是否增加了翻倒风险？
6. **PPO vs DR**：PPO 调参和 DR 扩展哪个对最终性能的影响更大？
7. **组合效应**：best PPO + best DR 的组合是否优于各自单独的效果？

> **统计严谨性声明**：
>
> - **统计功效局限**：n=3 的 t-test 自由度仅为 4，仅能检测极大效应量（Cohen's d > 2.0）。
>   主要依靠**效应量**（Cohen's d）和**置信区间**做判断，而非 p 值。
> - **多重比较**：9 组对比 × α=0.05 → 预期假阳性 ≈ 37%。
>   使用 Bonferroni 校正 α_adj = 0.05/9 ≈ 0.006，或在结论中标注未校正。
> - 如果关键对比不显著，根据效应量方向给出**定性结论**，明确标注 [Inference]。

### 4.8.3 可选：专项地形交叉评估

如果时间允许，选取 PPO/DR 最佳配置在 Phase 3 的 4 种专项地形上评估，
与 baseline 的专项地形数据（Phase 3 已有）对比。

**产出**：完整数据分析写入工作日志

---

## 步骤 4.9：撰写 `reward_engineering.md` 第 7 章

> 阶段 4 的核心文档产出。

### 第 7 章：PPO/DR 改动如何反映到 Reward 曲线

**章节目标**（开头一句话）：读完本章，你将理解 PPO 超参数和 Domain Randomization 如何影响 reward 的收敛行为和最终分布。

**内容大纲**：

1. **PPO 超参数回顾**：`learning_rate`、`clip_param`、`entropy_coef` 的作用和数学形式
2. **PPO 调参实验结果**：各参数对 reward 收敛曲线的影响（附 TensorBoard 截图），收敛速度 vs 最终性能的 trade-off，3 种子统计分析方法说明
3. **Domain Randomization 回顾**：摩擦/质量/推力随机化的物理含义
4. **DR 实验结果**：DR 对训练 reward 波动的影响，DR 扩展与泛化能力的关系（含交叉评估数据），各 reward 项在 DR 环境下的行为差异
5. **PPO vs DR 的交互效应**：两种调优维度是否独立？组合验证实验结论
6. **最佳配置推荐**：基于 Phase 4 全部实验数据（含组合验证），推荐最终配置组合

**写作规范**：遵循 `plan.md` 写作规范（统一模板、LaTeX 公式、TensorBoard 截图、章节衔接句）。

**产出**：`docs/reward_engineering.md` 第 7 章完稿

---

## 通过标准汇总

| 条件 | 要求 |
| --- | --- |
| Baseline 3 种子完成 | 3 个 baseline checkpoint 存在 |
| PPO 6 组训练完成 | 18 个 checkpoint 存在 |
| DR 3 组训练完成 | 9 个 checkpoint 存在 |
| 组合验证完成 | 至少 1 个组合 checkpoint 存在 |
| 标准评估全部完成 | 31-33 个 JSON 文件 |
| 交叉评估完成 | 4 个交叉评估 JSON 文件 |
| params/agent.yaml 核验 | 所有 PPO 变体的实际参数与实验设计一致 |
| 收敛稳定性 | 不劣于 baseline（均值±标准差范围内） |
| 数据分析 | 7 个问题全部回答 |
| 第 7 章完稿 | reward_engineering.md 更新 |

**阶段整体通过条件**：

- PPO 和 DR 实验全部完成训练和评估
- 至少 1 组配置在 `mean_vx_abs_err` 上统计显著优于 baseline（效应量 Cohen's d > 0.8），或至少能用数据解释为什么 baseline 已是局部最优
- `docs/reward_engineering.md` 第 7 章完稿
- 数据分析（含 3 种子统计 + 效应量）写入工作日志

---

## 风险与对策

| # | 风险 | 影响 | 对策 |
| --- | --- | --- | --- |
| 1 | 训练时间过长（30 次 × 2.5h = 75h） | 超出 5 天预算 | 优先跑 PPO-LR 和 DR-Push（信息量最大），其余视时间决定 |
| 2 | PPO-LR-High 发散 | 训练失败 | 定义发散标准：最终 reward < baseline 收敛 reward 的 30%；批量脚本自动检测并标记 |
| 3 | DR-Push 导致大量翻倒 | 训练不收敛 | 降低推力范围（如 `(-0.3, 0.3)`）重试 |
| 4 | 3 种子方差过大无法得出结论 | 统计不显著 | 依靠效应量（Cohen's d）和置信区间做定性判断，明确标注统计功效局限 |
| 5 | Baseline 种子不足 | 缺少对照基线 | 步骤 4.0 补训至 3 个种子 |
| 6 | GPU 显存不足（4096 envs + 大网络） | 训练 OOM | 降低 num_envs 至 2048 |
| 7 | `_apply_ros2_tracking_tune` 静默覆盖 PPO 参数 | 实验数据作废 | CLI 覆盖方案 + smoke test 核验 `params/agent.yaml` |
| 8 | WSL ROS2 publisher 中断（45h 串行训练） | 训练数据作废 | 批量脚本内置 publisher watchdog（每 5min 检查） |
| 9 | 磁盘空间不足 | 训练中断 | 预估 30 checkpoint × ~100MB + logs ≈ 5GB，训练前检查可用空间 |

---

## 时间线估算

| 天 | 任务 | 说明 |
| --- | --- | --- |
| Day 1 | 步骤 4.0 + 4.1 + 4.2 + 4.3 + 4.4 + 4.4b | 补训 baseline（后台）+ 参数确认 + CLI 覆盖实现 + DR 配置 + smoke test + 批量脚本 |
| Day 2 | 步骤 4.5（前半） | PPO-LR + PPO-Clip 训练（12 次，~30h，跨夜运行） |
| Day 3 | 步骤 4.5（后半） + 4.6 | PPO-Ent + DR 训练（15 次，~37h，跨夜运行） |
| Day 4 | 步骤 4.6b + 4.7 | 组合验证训练 + 全部评估（标准 + 交叉） |
| Day 5 | 步骤 4.8 + 4.9 | 数据分析 + 第 7 章撰写 |

---

## 附录：审查问题清单

| # | 问题 ID | 问题 | 修复 |
| --- | --- | --- | --- |
| 1 | C1 | `_apply_ros2_tracking_tune` 静默覆盖 PPO 变体参数 | 改用 CLI 覆盖方案（步骤 4.2）+ smoke test 核验 `params/agent.yaml`（步骤 4.4） |
| 2 | C2 | `--pass_cmd_vx_tol` 在 eval.py 中不存在 | 从评估命令中删除该参数（步骤 4.7） |
| 3 | C3 | PPO 配置文件路径不存在 + 12 任务 ID 过度工程化 | CLI 覆盖方案替代（步骤 4.2）；修正 `UnitreeGo1RoughPPORunnerCfg` 路径（步骤 4.1） |
| 4 | C4 | Baseline 只有 1 个种子，统计对比被阻塞 | 提升为前置步骤 4.0 |
| 5 | C5 | 27 次串行训练没有 ROS2 Publisher 管理方案 | 新增步骤 4.4b 批量脚本 + publisher watchdog |
| 6 | M1 | 没有 PPO+DR 组合实验 | 新增步骤 4.6b 组合验证 |
| 7 | M2 | DR 评估仅标准环境 | 步骤 4.7.2 新增抗扰动交叉评估 |
| 8 | M3 | DR 实现细节缺失 | 步骤 4.3 补充伪代码级 `__post_init__` 实现 |
| 9 | M4 | Day 4 时间预算严重不足 | 时间线从 4 天调整为 5 天 |
| 10 | M5 | 没有自动化批处理脚本 | 新增步骤 4.4b |
| 11 | M6 | 发散没有量化判定标准 | 风险表 #2 新增量化标准 + 批量脚本自动检测 |
| 12 | m1 | 3 种子 t-test 统计功效极低 | 步骤 4.8.2 新增统计功效声明 + 效应量为主 |
| 13 | m2 | 缺少多重比较校正 | 步骤 4.8.2 新增 Bonferroni 校正说明 |
| 14 | m3 | "收敛 iter"没有形式化定义 | 步骤 4.8.1 新增定义 |
| 15 | m4 | 未考虑 LR schedule 交互效应 | 步骤 4.1 新增 `schedule`/`desired_kl` 确认项 |
| 16 | s1 | CLAUDE.md 同步表新文件路径未确定 | 步骤 4.3 操作项 #7 |
| 17 | s2 | eval.py 未传 `--seed` | 步骤 4.7 评估命令新增 `--seed 42` |
| 18 | s3 | 磁盘空间未预估 | 风险表 #9 |
| 19 | s4 | target_vx 不一致（0.8 vs 1.0） | 步骤 4.7 统一为 `--target_vx 0.8`，与 ROS2 publisher 对齐 |
| 20 | — | 分析问题从 6 个增至 7 个 | 步骤 4.8.2 新增第 7 题（组合效应） |
