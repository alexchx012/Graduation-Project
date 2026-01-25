# Graduation-Project

本项目为上海电力大学-人工智能学部-智能科学与技术专业-解哲昊的毕业论文代码仓库

使用isaacsim5.1.0+isaaclab2.3.0+robot_lab结构

框架为win11+WSL2(Ubuntu22.04），Win11 上 Isaac Sim/Isaac Lab/robot_lab 负责物理仿真、环境步进、奖励与训练；WSL2 里的 ROS2 节点只订阅 Win11 发来的状态/传感器话题，做决策（策略推理或高层控制），再把控制指令话题发回 Win11 驱动仿真机器人。

### 📁 代码仓库结构

```
Graduation-Project/
├── README.md                    # 项目说明
├── REPRODUCE.md                 # 复现指南
├── requirements.txt             # Python依赖
├── configs/
│   ├── env/
│   │   ├── go1_flat.yaml
│   │   ├── go1_slope.yaml
│   │   └── go1_stairs.yaml
│   ├── morl/
│   │   └── weight_configs.yaml  # 10组权重配置
│   └── train/
│       └── ppo_config.yaml
├── src/
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
├── scripts/
│   ├── train_baseline.py        # 单目标训练
│   ├── train_morl.py            # 多目标训练
│   ├── evaluate.py              # 评估脚本
│   └── generate_figures.py      # 生成论文图表
├── checkpoints/                 # 模型存储
└── results/
    ├── raw_data/                # 原始实验数据
    └── figures/                 # 生成的图表
```

### 