# 2026-02-09 Go1 Flat + ROS2 高层决策联动训练执行计划（工作空间最小链路）

## Summary

基于现状：

- 2026-01-26 已验证 ROS2 与 Win11 Isaac 双向通信。
- 2026-01-27 已跑通 Go1 Flat 训练（`Isaac-Velocity-Flat-Unitree-Go1-v0`）。
- 当前目标是让 ROS2 做高层决策并参与训练闭环。

本计划采用：**ROS2 输出高层速度命令（vx, vy, wz），Win11 侧继续 PPO 训练关节动作**。
第一阶段目标：`num_envs=1`，闭环稳定跑完 `50 iterations`。

## Public API / Interface Changes

1. 新任务 ID（不破坏原任务）

- `Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0`
- `Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-Play-v0`

2. 新训练参数

- `--ros2_cmd_topic`（默认 `/go1/cmd_vel`）
- `--ros2_mode`（`startup_blocking` / `non_blocking`，默认 `startup_blocking`）
- `--ros2_startup_timeout_s`（默认 15）
- `--ros2_cmd_timeout_s`（默认 0.5）
- `--ros2_enable_obs_pub`（默认 false）
- `--ros2_obs_topic`（默认 `/go1/obs_flat`）

3. ROS2 话题协议（第一阶段）

- 命令输入：`geometry_msgs/msg/Twist` on `/go1/cmd_vel`
- 观测输出：`std_msgs/msg/Float32MultiArray` on `/go1/obs_flat`（第二阶段开启）

## Implementation Plan

1. 复制最小可运行训练链路到当前工作空间（不改上游核心）

- `scripts/reinforcement_learning/rsl_rl_ros2/train.py`
- `scripts/reinforcement_learning/rsl_rl_ros2/cli_args.py`
- 基线来源：`robot_lab/scripts/reinforcement_learning/rsl_rl/train.py` `robot_lab/scripts/reinforcement_learning/rsl_rl/cli_args.py`

2. 在 `robot_lab` 新增 Go1 ROS2 任务注册包

- `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/__init__.py`
- `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/flat_env_cfg.py`
- 配置复用：`isaaclab_tasks.manager_based.locomotion.velocity.config.go1.flat_env_cfg:UnitreeGo1FlatEnvCfg` 和 `isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents.rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg`

3. 新增 ROS2 命令项（替代随机速度命令）

- `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/ros2_velocity_command.py`
- 实现：
  输出维度固定 **(num_envs, 3)**。
  从 ROS2 订阅得到 [linear.x, linear.y, angular.z](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "linear.x, linear.y, angular.z") 写入命令。
  按原 task ranges 裁剪（与 [velocity_env_cfg.py](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "velocity_env_cfg.py") 一致）。
  超时回退策略：回退上一次命令，超过阈值回零命令。

4. 在 Isaac 进程内创建 ROS2 Subscribe Twist 图适配器

- `robot_lab/source/robot_lab/robot_lab/ros2_bridge/twist_subscriber_graph.py`
- 实现：
  启用扩展 **isaacsim.ros2.bridge**。
  创建 **/ROS2CmdGraph**，节点 **OnPlaybackTick** + **ROS2SubscribeTwist**。
  通过 [SubscribeTwist.outputs:linearVelocity&#34;)](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "og.Controller.attribute(&quot;/ROS2CmdGraph/SubscribeTwist.outputs:linearVelocity&quot;)") 和 **...angularVelocity** 读取最新值。
  启动阶段等待首条命令（**startup_blocking**）超时则降级为非阻塞。

5. 新任务接入新 CommandTerm

- 在 [flat_env_cfg.py](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "flat_env_cfg.py") 中将 **commands.base_velocity** 改为 **Ros2VelocityCommandCfg**。
  保留奖励、终止、PPO 配置不变，避免训练数学被破坏。

6. 第一阶段 ROS2 脚本节点（WSL2）

- `scripts/ros2_nodes/go1_cmd_script_node.py`
- 行为：
  固定频率（20Hz）发布 **Twist**。
  支持 profile：constant/sine/step（命令可重复回放）。
  提供 **--topic --rate --profile** 参数。

7. 第二阶段模型推理节点（预留）

- `scripts/ros2_nodes/go1_cmd_model_node.py`
- 行为：
  订阅 **/go1/obs_flat**，发布 **/go1/cmd_vel**。
  首版只留接口与推理桩函数，不在第一阶段耦合模型复杂度。

8. 运行编排

- `scripts/run/run_go1_ros2_train.ps1`
- `scripts/run/run_ros2_cmd_node.sh`
- 目标：
  一键启动顺序固定：ROS2 节点 -> 训练 -> 自动收集日志路径与 topic 状态。

9. 日志与可观测性

- 训练日志新增：`ros2/cmd_vx`, `ros2/cmd_vy`, `ros2/cmd_wz`, `ros2/cmd_timeout_count`, `ros2/startup_wait_s`
- 落盘：`params/ros2.yaml`

10. 回退路径

- 原命令和原 task 不改。
  训练失败时可直接切回：
  `--task Isaac-Velocity-Flat-Unitree-Go1-v0`

## Test Cases

1. 通信冒烟：
   WSL 发布 **Twist**，训练侧能读到非零命令并写入 TensorBoard。
   判定：命令曲线变化与发布一致。
2. 启动阻塞模式：
   未启动 ROS2 发布者时，训练等待首条命令至超时并给出 warning，不崩溃。
   判定：进程继续运行且回退策略生效。
3. 非阻塞模式：
   中途停止 ROS2 发布者，训练不中断。
   判定：**cmd_timeout_count** 增加且奖励计算持续。
4. 第一阶段验收（你选定标准）：
   num_envs=1，**max_iterations=50** 完成，无死锁。
   判定：训练结束且日志显示命令来源为 ROS2。
5. 回归对比：
   同机同 seed 跑原任务和 ROS2 新任务各 20 iter。
   判定：两者都可完成，原任务不受影响。

## Assumptions / Defaults

1. 任务固定为 **Go1 Flat**（**Isaac-Velocity-Flat-Unitree-Go1-v0** 基线）。
2. 决策层固定为高层速度命令，不直接由 ROS2 输出 12 维关节动作。
3. 第一阶段规模固定 **num_envs=1**，目标 50 iter 稳定闭环。
4. ROS2 决策路径固定“脚本节点先行，模型推理后接”。
5. 消息协议固定“标准消息最小集”：命令 **Twist**，观测 **Float32MultiArray**。
6. “混合模式”定义固定为：启动可阻塞等首条命令，运行期非阻塞+超时回退。
7. 改造范围固定为工作空间最小链路，不修改 IsaacLab 上游核心源码。

## 关键文件绝对路径（超链接）

- [D:/Graduation-Project/scripts/reinforcement_learning/rsl_rl_ros2/train.py](D:/Graduation-Project/scripts/reinforcement_learning/rsl_rl_ros2/train.py)
- [D:/Graduation-Project/scripts/reinforcement_learning/rsl_rl_ros2/cli_args.py](D:/Graduation-Project/scripts/reinforcement_learning/rsl_rl_ros2/cli_args.py)
- [D:/Graduation-Project/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/__init__.py](D:/Graduation-Project/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/__init__.py)
- [D:/Graduation-Project/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/flat_env_cfg.py](D:/Graduation-Project/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/flat_env_cfg.py)
- [D:/Graduation-Project/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/ros2_velocity_command.py](D:/Graduation-Project/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/ros2_velocity_command.py)
- [D:/Graduation-Project/robot_lab/source/robot_lab/robot_lab/ros2_bridge/twist_subscriber_graph.py](D:/Graduation-Project/robot_lab/source/robot_lab/robot_lab/ros2_bridge/twist_subscriber_graph.py)
- [D:/Graduation-Project/scripts/ros2_nodes/go1_cmd_script_node.py](D:/Graduation-Project/scripts/ros2_nodes/go1_cmd_script_node.py)
- [D:/Graduation-Project/scripts/ros2_nodes/go1_cmd_model_node.py](D:/Graduation-Project/scripts/ros2_nodes/go1_cmd_model_node.py)
- [D:/Graduation-Project/scripts/run/run_go1_ros2_train.ps1](D:/Graduation-Project/scripts/run/run_go1_ros2_train.ps1)
- [D:/Graduation-Project/scripts/run/run_ros2_cmd_node.sh](D:/Graduation-Project/scripts/run/run_ros2_cmd_node.sh)
