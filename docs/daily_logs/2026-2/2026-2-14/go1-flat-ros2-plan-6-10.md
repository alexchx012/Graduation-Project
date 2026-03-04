# Go1 Flat + ROS2 阶段 6-10 执行计划

> 前置文档：[go1-flat-ros2-plan.md](../2026-2-9/go1-flat-ros2-plan.md) | [go1-flat-ros2-plan-4-5.md](../2026-2-10/go1-flat-ros2-plan-4-5.md) | [2026-2-11.md](../2026-2-11/2026-2-11.md)
> 日期：2026-02-14

## TL;DR

阶段 1-5 已在 2-11 日完成：rclpy 桥接器 + Ros2VelocityCommand + 两个新任务 ID 注册 + 通信/超时/双端冒烟测试全部通过。

本文档覆盖剩余工作：
- **P1**：WSL 脚本节点 + 一键运行编排（阶段 6/8）
- **P2**：模型推理桩节点 + 日志增强（阶段 7/9）
- **P3**：50 iter 验收 + 回归对比测试（阶段 10 + 测试用例 4/5）
- **基础设施**：统一文件归档规则，消除双源混乱

---

## 0. 源码归档规则（本次确立）

### 0.1 权威源定义

项目中 `go1-ros2-test` 任务的所有文件归档于三个任务文件夹中：

| 层 | 权威源路径 | 职责 |
|---|---|---|
| 环境/MDP | `src/go1-ros2-test/` | Cfg、Command、Bridge 代码 |
| 脚本 | `scripts/go1-ros2-test/` | 训练入口、ROS2 节点、运行编排 |
| 配置 | `configs/go1-ros2-test/` | YAML 配置（如后续新增） |

### 0.2 运行时部署

`robot_lab/` 是唯一的 pip 安装包，`import robot_lab.*` 只找到它。修改流程：

```
编辑 src/go1-ros2-test/... (权威源)
  ↓ 同步
复制到 robot_lab/... (运行时部署)
```

### 0.3 历史副本处理

| 路径 | 处置 |
|---|---|
| `scripts/reinforcement_learning/rsl_rl_ros2/train.py` | 降级为历史副本，头注释改为 `# DEPRECATED: canonical source is scripts/go1-ros2-test/train.py` |
| `scripts/reinforcement_learning/rsl_rl_ros2/cli_args.py` | 同上 |

### 0.4 头注释规范

权威源文件头部使用：
```python
# Canonical source: <工作区相对路径>
# Deployed to: <robot_lab 内路径> (runtime)
```

副本文件头部使用：
```python
# DEPRECATED: canonical source is <权威源路径>
```

### 0.5 同步规则

- 对 `src/go1-ros2-test/` 或 `scripts/go1-ros2-test/` 做的修改，完成后必须同步到 `robot_lab/` 对应路径
- `scripts/go1-ros2-test/` 下的独立脚本（ros2_nodes/、run/）不需要部署到 robot_lab，但需在 `src/go1-ros2-test/` 中无对应物也无需创建
- 同步后应验证 `robot_lab/` 中的文件能正常 import

---

## 1. 阶段 6 — WSL ROS2 脚本节点（P1）

### 1.1 新增文件

**`scripts/go1-ros2-test/ros2_nodes/go1_cmd_script_node.py`**

| 项目 | 说明 |
|---|---|
| 运行环境 | WSL Ubuntu-22.04，系统 Python `/usr/bin/python3`，**不用 conda** |
| 依赖 | `rclpy`、`geometry_msgs`（ROS2 humble 自带） |
| 发布话题 | `geometry_msgs/msg/Twist` on `/go1/cmd_vel` |
| 频率 | 默认 20 Hz，通过 `--rate` 可调 |

**CLI 参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--topic` | `/go1/cmd_vel` | ROS2 话题名 |
| `--rate` | `20` | 发布频率 Hz |
| `--profile` | `constant` | 命令生成模式：`constant` / `sine` / `step` |
| `--vx` | `0.5` | 前进速度 m/s |
| `--vy` | `0.0` | 侧向速度 m/s |
| `--wz` | `0.0` | 偏航角速度 rad/s |
| `--duration` | `inf` | 运行时长（秒），`inf` 表示持续运行 |

**Profile 行为定义：**

| Profile | vx(t) | vy(t) | wz(t) |
|---|---|---|---|
| `constant` | `vx` | `vy` | `wz` |
| `sine` | `vx * sin(2π * 0.1 * t)` | `vy * sin(2π * 0.1 * t)` | `wz * sin(2π * 0.1 * t)` |
| `step` | 每 5s 在 `0` 与 `vx` 之间切换 | 同理 | 同理 |

**实现要点：**

- 基于 `rclpy.create_node()` + `create_timer(1.0/rate, callback)` 实现定时发布
- 每条消息打印 `[t={elapsed:.2f}s] pub Twist(vx={:.3f}, vy={:.3f}, wz={:.3f})`
- `SIGINT` 优雅退出：`destroy_node()` → `rclpy.shutdown()`
- shebang 行 `#!/usr/bin/env python3`

### 1.2 无需同步到 robot_lab

此文件是独立的 WSL 端 ROS2 节点，不属于 `robot_lab` 包。

---

## 2. 阶段 7 — 模型推理桩节点（P2，预留）

### 2.1 新增文件

**`scripts/go1-ros2-test/ros2_nodes/go1_cmd_model_node.py`**

| 项目 | 说明 |
|---|---|
| 运行环境 | WSL Ubuntu-22.04，系统 Python |
| 订阅 | `std_msgs/msg/Float32MultiArray` on `/go1/obs_flat` |
| 发布 | `geometry_msgs/msg/Twist` on `/go1/cmd_vel` |

**CLI 参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--obs_topic` | `/go1/obs_flat` | 观测订阅话题 |
| `--cmd_topic` | `/go1/cmd_vel` | 命令发布话题 |
| `--rate` | `20` | 发布频率 Hz |
| `--model_path` | `None` | 模型文件路径（预留） |

**首版实现：**

- `infer(obs)` 桩函数：忽略输入，返回固定 `Twist(vx=0.3, vy=0.0, wz=0.0)`
- 预留 `# TODO: torch.load(model_path); cmd = model.forward(obs)` 注释占位
- 收到观测时打印 `[obs] dim={len}, first_5={obs[:5]}`
- 无观测时仍按频率发布桩命令

### 2.2 无需同步到 robot_lab

同阶段 6，独立 WSL 脚本。

---

## 3. 阶段 8 — 运行编排脚本（P1）

### 3.1 WSL 端启动脚本

**`scripts/go1-ros2-test/run/run_ros2_cmd_node.sh`**

```bash
#!/bin/bash
source /opt/ros/humble/setup.bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${SCRIPT_DIR}/ros2_nodes/go1_cmd_script_node.py" "$@"
```

### 3.2 Windows 端编排脚本

**`scripts/go1-ros2-test/run/run_go1_ros2_train.ps1`**

逻辑：

1. **参数定义**：`$Profile`（默认 `constant`）、`$NumEnvs`（默认 1）、`$MaxIter`（默认 50）、`$Vx`（默认 0.5）
2. **Step 1 — 启动 WSL ROS2 节点**（后台）
3. **Step 2 — 等待 3 秒稳定**
4. **Step 3 — conda activate env_isaaclab + isaaclab.bat 训练**
5. **Step 4 — 清理 WSL 进程，打印日志路径**
6. **try/finally 确保 WSL 进程被终止**

### 3.3 无需同步到 robot_lab

独立编排脚本。

---

## 4. 阶段 9 — 日志与可观测性（P2）

### 4.1 TensorBoard 指标增强

**修改文件**：`src/go1-ros2-test/envs/mdp/commands/ros2_velocity_command.py`
**同步到**：`robot_lab/.../velocity/mdp/ros2_velocity_command.py`

变更内容：

1. 在 `__init__()` 中新增 3 个指标：`cmd_vx`、`cmd_vy`、`cmd_wz`
2. 在 `_update_command()` 末尾记录当前命令值
3. RSL-RL 自动将 `reset()` 返回的 `extras` 写入 TensorBoard，前缀 `Command/`

最终可见指标：
- `Command/cmd_vx`、`Command/cmd_vy`、`Command/cmd_wz`
- `Command/cmd_timeout_count`、`Command/cmd_zero_fallback_count`

### 4.2 ROS2 配置落盘

**修改文件**：`scripts/go1-ros2-test/train.py`

在 `dump_yaml(... "params/agent.yaml" ...)` 之后，新增 `params/ros2.yaml` 落盘逻辑。

### 4.3 CLI 参数化（可选，低优先级）

新增 `add_ros2_args(parser)` 函数，将硬编码配置暴露为命令行参数。本步骤为可选。

---

## 5. 阶段 10 — 回退路径验证

天然完成。验证方式：运行原任务训练确认无 ROS2 报错。

---

## 6. 测试用例补全

### 测试 4 — 50 iter 完整验收（P3）

| 项 | 值 |
|---|---|
| WSL 端 | `go1_cmd_script_node.py --profile constant --vx 0.5 --rate 20` |
| Windows 端 | `--task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0 --num_envs 1 --max_iterations 50` |
| 判定标准 | 训练完成无死锁；`Command/cmd_timeout_count` 接近 0；TensorBoard 中 `Command/cmd_vx` 显示非零值 |

### 测试 5 — 回归对比（P3）

| 组 | 命令 |
|---|---|
| 原任务 | `--task Isaac-Velocity-Flat-Unitree-Go1-v0 --num_envs 4 --max_iterations 20 --seed 42` |
| ROS2 任务 | `--task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0 --num_envs 1 --max_iterations 20 --seed 42` + WSL constant `vx=0.5` |
| 判定标准 | 两者均完成训练；原任务不受 ROS2 代码影响 |

---

## 文件变更总览

### 新增文件

| 文件 | 说明 | 需同步到 robot_lab？ |
|---|---|---|
| `scripts/go1-ros2-test/ros2_nodes/go1_cmd_script_node.py` | WSL 命令发布节点 | 否 |
| `scripts/go1-ros2-test/ros2_nodes/go1_cmd_model_node.py` | 模型推理桩节点 | 否 |
| `scripts/go1-ros2-test/run/run_go1_ros2_train.ps1` | Windows 编排脚本 | 否 |
| `scripts/go1-ros2-test/run/run_ros2_cmd_node.sh` | WSL 启动脚本 | 否 |

### 修改文件

| 文件（权威源） | 说明 | 同步目标 |
|---|---|---|
| `src/go1-ros2-test/envs/mdp/commands/ros2_velocity_command.py` | 新增 cmd_vx/vy/wz 指标 | `robot_lab/.../velocity/mdp/ros2_velocity_command.py` |
| `scripts/go1-ros2-test/train.py` | ros2.yaml 落盘 | 无需 |

### 头注释更新

| 文件 | 变更 |
|---|---|
| `scripts/go1-ros2-test/train.py` | `# Original path` → `# Canonical source` |
| `scripts/go1-ros2-test/cli_args.py` | 同上 |
| `scripts/reinforcement_learning/rsl_rl_ros2/train.py` | 标记 DEPRECATED |
| `scripts/reinforcement_learning/rsl_rl_ros2/cli_args.py` | 同上 |
| `src/go1-ros2-test/` 下所有文件 | `# Original path` → `# Canonical source` + `# Deployed to` |

---

## 执行顺序与预估时间

| 顺序 | 步骤 | 阶段 | 优先级 | 预估时间 |
|---|---|---|---|---|
| 1 | 头注释更新 + 归档规则落地 | 0 | P1 | 10 min |
| 2 | go1_cmd_script_node.py | 6 | P1 | 30 min |
| 3 | run_ros2_cmd_node.sh + run_go1_ros2_train.ps1 | 8 | P1 | 30 min |
| 4 | ros2_velocity_command.py 指标增强 | 9.1 | P2 | 15 min |
| 5 | train.py ros2.yaml 落盘 | 9.2 | P2 | 15 min |
| 6 | 测试 4：50 iter 验收 | 10 | P3 | 15 min |
| 7 | 测试 5：回归对比 | 10 | P3 | 15 min |
| 8 | go1_cmd_model_node.py | 7 | P2 | 15 min |
| 9 | CLI 参数化（可选） | 9.3 | P3 | 20 min |
| 10 | 回退路径确认 | 10 | P3 | 5 min |

---

## Verification

| 检查项 | 方法 | 预期 |
|---|---|---|
| 脚本节点运行 | `wsl -d Ubuntu-22.04 bash -lc "source /opt/ros/humble/setup.bash && python3 .../go1_cmd_script_node.py --profile sine"` | 20Hz Twist 日志输出 |
| 一键编排 | `.\scripts\go1-ros2-test\run\run_go1_ros2_train.ps1` | ROS2 + 训练依次启动完成 |
| ros2.yaml 落盘 | 检查 `logs/rsl_rl/unitree_go1_flat/<run>/params/ros2.yaml` | 含 topic、timeout 字段 |
| TensorBoard 指标 | `tensorboard --logdir logs/rsl_rl/unitree_go1_flat/<run>` | 可见 5 个 `Command/` 指标 |
| 50 iter 验收 | 测试 4 标准 | 无死锁，命令来源 ROS2 |
| 回归对比 | 测试 5 标准 | 两任务完成，原任务无影响 |
| 头注释一致 | grep 检查 `Canonical source` / `DEPRECATED` | 所有文件注释方向正确 |

---

## Decisions

1. **归档规则**：`go1-ros2-test` 文件夹为任务权威源，`scripts/reinforcement_learning/rsl_rl_ros2/` 降级为历史副本
2. **新脚本位置**：统一归入 `scripts/go1-ros2-test/ros2_nodes/` 和 `scripts/go1-ros2-test/run/`
3. **模型推理节点仅实现桩函数**：不引入模型加载复杂度
4. **CLI 参数化为可选**：优先保证编排可用性
5. **指标通过 RSL-RL extras 自动流入 TensorBoard**：无需修改训练循环
