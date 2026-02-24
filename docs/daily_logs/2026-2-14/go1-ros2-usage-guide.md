# Go1 Flat + ROS2 高层决策联动训练 — 使用指南

---

## 1. 概述

本任务在 Isaac Lab 的 Go1 Flat 速度跟踪训练基础上，接入 ROS2 高层速度命令。
**ROS2 负责发送 `(vx, vy, wz)` 高层速度目标，Isaac Lab 内的 PPO 依然训练 12 维关节动作。**

数据流：

```
WSL (Ubuntu 22.04)                    Windows (Isaac Lab)
┌─────────────────────┐                ┌────────────────────────────────┐
│ go1_cmd_script_node │  /go1/cmd_vel  │ Ros2TwistSubscriberAdapter     │
│   (Twist publisher) │ ─────────────> │   ↓ physics callback 每步同步  │
└─────────────────────┘   (DDS)        │ Ros2VelocityCommand (MDP)      │
                                       │   ↓ clip → broadcast all envs  │
                                       │ PPO OnPolicyRunner 训练        │
                                       └────────────────────────────────┘
```

## 2. 前置条件

| 组件    | 要求                                                      |
| ------- | --------------------------------------------------------- |
| Windows | Isaac Sim 5.1 + Isaac Lab 已安装                          |
| Conda   | `env_isaaclab` 环境可用                                 |
| WSL     | Ubuntu-22.04，已安装 ROS2 Humble（`/opt/ros/humble`）   |
| DDS     | Windows 与 WSL 同网段可互通（默认 FastDDS，无需额外配置） |

## 3. 快速开始

### 方式一：一键编排脚本（推荐）

在 **Windows PowerShell** 中：

```powershell
cd D:\Graduation-Project

# 最简运行：constant 命令，50 iter
.\scripts\go1-ros2-test\run\run_go1_ros2_train.ps1

# 完整参数示例
.\scripts\go1-ros2-test\run\run_go1_ros2_train.ps1 `
    -Profile sine `
    -MaxIter 100 `
    -Vx 0.3 `
    -Vy 0.0 `
    -Wz 0.1 `
    -Rate 20 `
    -Seed 42 `
    -NumEnvs 1
```

脚本会自动：

1. 后台启动 WSL ROS2 节点
2. 等待 3 秒稳定
3. 激活 `env_isaaclab` 并启动训练
4. 训练结束或中断时自动清理 WSL 进程

### 方式二：手动分步启动

**终端 1（WSL Ubuntu-22.04）**— 启动 ROS2 命令节点：

```bash
source /opt/ros/humble/setup.bash
python3 scripts/go1-ros2-test/ros2_nodes/go1_cmd_script_node.py \
    --profile constant --vx 0.5 --rate 20
```

**终端 2（Windows PowerShell）**— 启动训练：

```powershell
conda activate env_isaaclab
.\IsaacLab\isaaclab.bat -p scripts\go1-ros2-test\train.py `
    --task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0 `
    --num_envs 1 `
    --max_iterations 50 `
    --headless
```

## 4. 任务 ID

| 任务 ID                                             | 用途                       |
| --------------------------------------------------- | -------------------------- |
| `Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0`      | ROS2 命令驱动训练          |
| `Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-Play-v0` | ROS2 命令驱动回放/评估     |
| `Isaac-Velocity-Flat-Unitree-Go1-v0`              | 原始任务（基线，不受影响） |

## 5. 参数参考

### 一键编排脚本参数 (`run_go1_ros2_train.ps1`)

| 参数         | 类型   | 默认值            | 说明                                         |
| ------------ | ------ | ----------------- | -------------------------------------------- |
| `-Profile` | string | `constant`      | 命令模式：`constant` / `sine` / `step` |
| `-NumEnvs` | int    | `1`             | 并行环境数                                   |
| `-MaxIter` | int    | `50`            | 最大训练迭代数                               |
| `-Vx`      | float  | `0.5`           | 前进速度 m/s                                 |
| `-Vy`      | float  | `0.0`           | 侧向速度 m/s                                 |
| `-Wz`      | float  | `0.0`           | 偏航角速度 rad/s                             |
| `-Rate`    | int    | `20`            | ROS2 发布频率 Hz                             |
| `-Task`    | string | `...ROS2Cmd-v0` | 任务 ID                                      |
| `-Seed`    | int    | `-1`            | 随机种子（-1 不设置）                        |

### ROS2 脚本节点参数 (`go1_cmd_script_node.py`)

| 参数           | 默认值           | 说明             |
| -------------- | ---------------- | ---------------- |
| `--topic`    | `/go1/cmd_vel` | ROS2 话题名      |
| `--rate`     | `20`           | 发布频率 Hz      |
| `--profile`  | `constant`     | 命令模式         |
| `--vx`       | `0.5`          | 前进速度 m/s     |
| `--vy`       | `0.0`          | 侧向速度 m/s     |
| `--wz`       | `0.0`          | 偏航角速度 rad/s |
| `--duration` | `inf`          | 运行时长秒数     |

### 命令 Profile 说明

| Profile      | 行为                                   |
| ------------ | -------------------------------------- |
| `constant` | 恒定速度输出                           |
| `sine`     | 正弦波变化，频率 0.1Hz，幅值为设定速度 |
| `step`     | 每 5 秒在设定速度和 0 之间切换         |

## 6. ROS2 话题协议

| 话题              | 消息类型                       | 方向             | 说明                         |
| ----------------- | ------------------------------ | ---------------- | ---------------------------- |
| `/go1/cmd_vel`  | `geometry_msgs/Twist`        | WSL → Isaac Lab | 高层速度命令                 |
| `/go1/obs_flat` | `std_msgs/Float32MultiArray` | Isaac Lab → WSL | 观测输出（第二阶段，未实现） |

命令字段映射：

```
Twist.linear.x  → vx（前进速度）
Twist.linear.y  → vy（侧向速度）
Twist.angular.z → wz（偏航角速度）
```

## 7. 超时与回退机制

| 参数                  | 值                   | 行为                            |
| --------------------- | -------------------- | ------------------------------- |
| `startup_mode`      | `startup_blocking` | 启动时阻塞等待首条 ROS2 消息    |
| `startup_timeout_s` | `15.0`             | 等待超时后打印 WARNING 继续运行 |
| `cmd_timeout_s`     | `0.5`              | 运行期超时阈值                  |

运行期回退链路：

```
收到新命令 → 使用新命令（clip 后 broadcast）
  ↓ 超时 ≤ 0.5s
保持上一条命令（cmd_timeout_count++）
  ↓ 超时 > 0.5s
回退到零命令（cmd_zero_fallback_count++）
```

## 8. 训练输出

### 目录结构

```
logs/rsl_rl/unitree_go1_flat/<timestamp>/
├── params/
│   ├── env.yaml          # 环境配置
│   ├── agent.yaml         # PPO 超参
│   └── ros2.yaml          # ROS2 桥接配置
├── git/
│   └── Graduation-Project.diff
├── events.out.tfevents.*  # TensorBoard 日志
├── model_0.pt             # 初始模型
└── model_<N>.pt           # 最终模型
```

### TensorBoard 指标

```bash
tensorboard --logdir logs/rsl_rl/unitree_go1_flat
```

ROS2 相关指标（`Metrics/base_velocity/` 前缀）：

| 指标                        | 含义             |
| --------------------------- | ---------------- |
| `cmd_vx`                  | 当前 vx 命令值   |
| `cmd_vy`                  | 当前 vy 命令值   |
| `cmd_wz`                  | 当前 wz 命令值   |
| `cmd_timeout_count`       | 物理步间超时次数 |
| `cmd_zero_fallback_count` | 长超时零回退次数 |

## 9. 回退到基线

如遇问题，可随时切回原始任务（不依赖 ROS2）：

```powershell
conda activate env_isaaclab
.\IsaacLab\isaaclab.bat -p scripts\go1-ros2-test\train.py `
    --task Isaac-Velocity-Flat-Unitree-Go1-v0 `
    --num_envs 4 `
    --max_iterations 300 `
    --headless
```

原始任务代码和配置完全未被修改。

## 10. 关键文件清单

### 权威源码（`src/go1-ros2-test/`）

| 文件                                           | 说明                                                     |
| ---------------------------------------------- | -------------------------------------------------------- |
| `envs/__init__.py`                           | Gym 环境注册                                             |
| `envs/flat_env_cfg.py`                       | 环境配置（替换 base_velocity 为 Ros2VelocityCommandCfg） |
| `envs/mdp/commands/ros2_velocity_command.py` | MDP 命令项（从 env 属性读取 ROS2 命令）                  |
| `ros2_bridge/twist_subscriber_graph.py`      | ROS2 rclpy 订阅 → env 属性同步适配器                    |

### 运行时部署（`robot_lab/`）

上述源码同步部署到 `robot_lab/source/robot_lab/robot_lab/` 对应目录，供 Isaac Lab 运行时加载。

### 脚本

| 文件                                                        | 说明                                     |
| ----------------------------------------------------------- | ---------------------------------------- |
| `scripts/go1-ros2-test/train.py`                          | 训练入口（自动检测 ROS2 任务并启用桥接） |
| `scripts/go1-ros2-test/cli_args.py`                       | RSL-RL CLI 参数                          |
| `scripts/go1-ros2-test/ros2_nodes/go1_cmd_script_node.py` | WSL ROS2 脚本命令节点                    |
| `scripts/go1-ros2-test/ros2_nodes/go1_cmd_model_node.py`  | WSL ROS2 模型推理桩节点                  |
| `scripts/go1-ros2-test/run/run_go1_ros2_train.ps1`        | Windows 一键编排                         |
| `scripts/go1-ros2-test/run/run_ros2_cmd_node.sh`          | WSL 端启动器                             |

## 11. 常见问题

### Q: 训练时命令值一直为 0？

1. 确认 WSL ROS2 节点正在运行并发布消息
2. 检查 DDS 连通性：在 WSL 中 `ros2 topic echo /go1/cmd_vel` 确认有数据
3. 确认 Isaac Sim 的 ROS2 bridge 扩展已启用（训练脚本会自动启用）

### Q: 启动时出现 "No message received within 15.0s" WARNING？

这是正常行为。训练脚本会在等待 ROS2 首条消息超时后继续运行。确保 WSL ROS2 节点先于训练启动即可。一键编排脚本已自动处理启动顺序。

### Q: 如何用自定义高层策略替代脚本节点？

将任何能发布 `geometry_msgs/Twist` 到 `/go1/cmd_vel` 的 ROS2 节点作为命令源即可。`go1_cmd_model_node.py` 提供了模型推理节点的骨架，只需替换 `_infer()` 方法中的桩逻辑。

### Q: `num_envs > 1` 时命令如何分配？

当前实现将同一条 ROS2 命令 broadcast 到所有并行环境。多环境主要用于加速 PPO 采样，所有 env 共享相同的速度目标。
