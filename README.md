# Graduation-Project

本项目为上海电力大学-人工智能学部-智能科学与技术专业-解哲昊的毕业论文代码仓库

使用isaacsim5.1.0+isaaclab2.3.0+robot_lab结构

框架为win11+WSL2(Ubuntu22.04），Win11 上 Isaac Sim/Isaac Lab/robot_lab 负责物理仿真、环境步进、奖励与训练；WSL2 里的 ROS2 节点只订阅 Win11 发来的状态/传感器话题，做决策（策略推理或高层控制），再把控制指令话题发回 Win11 驱动仿真机器人。

### 📁 当前代码仓库结构

```
Graduation-Project/
├── README.md                       # 项目说明
├── AGENTS.md                       # 代理执行规则和文件同步配置
├── LICENSE                         # 许可证
├── .gitignore                      # Git忽略配置
│
├── configs/                        # 配置文件
│   ├── go1-ros2-test/
│   │   └── README.md               # Go1配置说明
│   └── ros2/
│       ├── fastrtps_win_to_wsl.xml # FastRTPS Win→WSL配置
│       └── fastrtps_wsl_to_win.xml # FastRTPS WSL→Win配置
│
├── src/                            # 核心业务代码
│   ├── algorithms/
│   │   └── ppo.py                  # PPO算法实现
│   ├── evaluation/                 # 评估模块（待完善）
│   ├── utils/                      # 工具模块（待完善）
|   └── go1-ros2-test/              # Go1机器人ROS2集成
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── flat_env_cfg.py     # 平地环境配置
│       │   └── mdp/
│       │       ├── __init__.py
│       │       └── commands/
│       │           ├── __init__.py
│       │           └── ros2_velocity_command.py  # ROS2速度命令
│       └── ros2_bridge/
│           ├── __init__.py
│           └── twist_subscriber_graph.py  # Twist话题订阅器
│
├── scripts/                        # 训练和评估脚本
│   ├── go1-ros2-test/
│   │   ├── cli_args.py             # 命令行参数解析
│   │   ├── train.py                # 训练脚本
│   │   ├── eval.py                 # 评估脚本
│   │   ├── ros2_nodes/
│   │   │   ├── go1_cmd_model_node.py   # 模型推理ROS2节点
│   │   │   └── go1_cmd_script_node.py  # 脚本控制ROS2节点
│   │   └── run/                    # 运行配置和日志
│   │       ├── Debug-ROS2-Test/    # 调试测试
│   │       └── VF-go1-Formal-Training/  # 正式训练
│   └── reinforcement_learning/
│       └── rsl_rl_ros2/
│           ├── cli_args.py         # 命令行参数
│           └── train.py            # RSL_RL训练脚本
│
├── docs/                           # 文档和日志
│   ├── daily_logs/                 # 日常工作日志
│   │   ├── 2026-01-25/ ~ 2026-02-24/  # 各日期日志
│   │   ├── PPO笔记 -Final.md       # PPO算法笔记
│   │   └── PPO阅读笔记-raw.md      # PPO阅读笔记
│   └── problems/
│       └── robot_lab运行失败.md    # 问题记录
│
├── logs/                           # 运行日志
│   └── rsl_rl/
│       └── unitree_go1_flat/       # Go1平地训练日志
│           └── [时间戳目录]/
│               ├── params/         # 参数快照(agent/env/ros2.yaml)
│               └── git/            # Git状态快照
│
├── outputs/                        # Hydra训练输出
│   └── [日期]/[时间]/
│
├── results/
│   └── figures/                    # 生成的图表
│
├── wandb/                          # Weights & Biases日志
│
├── IsaacLab ->  /d/IsaacLab     # 符号链接
├── isaacsim ->  /c/isaacsim     # 符号链接
└── robot_lab -> /d/isaaclab-kuozhan/robot_lab  # 符号链接
```

---

### 📁 目标代码仓库结构（规划中）

```
Graduation-Project/
├── README.md                    # 项目说明
├── REPRODUCE.md                 # 复现指南
├── requirements.txt             # Python依赖
├── configs/                     # 环境状态：数据库配置，API秘钥，环境常量
│   ├── env/
│   │   ├── go1_flat.yaml
│   │   ├── go1_slope.yaml
│   │   └── go1_stairs.yaml
│   ├── morl/
│   │   └── weight_configs.yaml  # 10组权重配置
│   └── train/
│       └── ppo_config.yaml
├── src/                         # 业务代码，算法，核心组件
│   ├── envs/
│   │   └── go1_morl_env.py      # 多目标环境
│   ├── rewards/
│   │   └── mo_rewards.py        # 4目标reward实现
│   ├── algorithms/
│   │   └── linear_scalarization.py
│   ├── evaluation/
│   │   ├── pareto_analysis.py
│   │   ├── hypervolume.py
│   │   └── metrics.py
│   └── utils/
│       └── wandb_logger.py
├── scripts/                     # 构建脚本，部署任务
│   ├── train_baseline.py        # 单目标训练
│   ├── train_morl.py            # 多目标训练
│   ├── evaluate.py              # 评估脚本
│   └── generate_figures.py      # 生成论文图表
├── checkpoints/                 # 模型存储
└── results/
    ├── raw_data/                # 原始实验数据
    └── figures/                 # 生成的图表
```
