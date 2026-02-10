# Go1 ROS2 第4+5步最小落地计划（含冒烟验证）

## Summary

目标是仅完成你确认的范围：

1. 在 Isaac 进程内落地 **ROS2SubscribeTwist** 图适配器，把命令写入 **env.unwrapped.ros2_latest_cmd_vel** 与 **env.unwrapped.ros2_latest_cmd_stamp_s**。
2. 将 Go1 ROS2 任务的 **commands.base_velocity** 切到 **Ros2VelocityCommandCfg**。
3. 做最小可运行冒烟验证（不扩展训练 CLI 参数）。

## Public APIs / Interfaces 变更

1. 新增内部桥接模块：[twist_subscriber_graph.py](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "robot_lab/source/robot_lab/robot_lab/ros2_bridge/twist_subscriber_graph.py")。
2. 新增桥接类接口（内部 API）：**Ros2TwistSubscriberGraphAdapter**。
3. 新增内部配置 dataclass（内部 API）：**Ros2TwistBridgeCfg**。
4. **unitree_go1_ros2** 任务配置行为变化：**commands.base_velocity** 从随机速度命令改为 ROS2 外部命令项。
5. 本轮不新增命令行参数，默认值硬编码在训练入口中（后续再做 CLI 化）。

## Implementation Plan

### 1. 新建 ROS2 图适配器模块

文件：

* 新建 [__init__.py](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "robot_lab/source/robot_lab/robot_lab/ros2_bridge/__init__.py")
* 新建 [twist_subscriber_graph.py](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "robot_lab/source/robot_lab/robot_lab/ros2_bridge/twist_subscriber_graph.py")

实现内容：

1. **Ros2TwistBridgeCfg** 默认值固定：
   * **graph_path="/ROS2CmdGraph"**
   * **topic_name="/go1/cmd_vel"**
   * **queue_size=10**
   * **startup_mode="startup_blocking"**
   * **startup_timeout_s=15.0**
   * **command_attr="ros2_latest_cmd_vel"**
   * **command_stamp_attr="ros2_latest_cmd_stamp_s"**
2. **Ros2TwistSubscriberGraphAdapter** 生命周期：
   * **setup()**：
     启用扩展 **isaacsim.ros2.bridge**，创建 execution graph，节点至少含：
     OnPlaybackTick + **ROS2SubscribeTwist**，并连接 **OnPlaybackTick.outputs:tick -> SubscribeTwist.inputs:execIn**。
   * **attach()**：
     注册 [env.sim.add_physics_callback(...)](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "env.sim.add_physics_callback(...)")，每个 physics step 读取图节点输出并同步到 **env.unwrapped**。
   * **wait_for_first_message()**：
     实现 startup blocking（超时后 warning 并降级继续）。
   * **close()**：
     移除 physics callback，清理句柄（幂等）。
3. 数据同步规则（关键）：
   * 每次“新消息”到达时写：
     * [env.unwrapped.ros2_latest_cmd_vel = [linear.x, linear.y, angular.z]](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "env.unwrapped.ros2_latest_cmd_vel = [linear.x, linear.y, angular.z]")
     * **env.unwrapped.ros2_latest_cmd_stamp_s = <单调递增时间戳>**
   * 时间戳用于 **Ros2VelocityCommand** 判新，必须在“同值重复消息”时也更新。
4. 判新策略（决策）：

* 首选：使用 **SubscribeTwist.outputs:execOut** 的变化作为新消息触发依据。
* 兜底：若 **execOut** 在运行中不可可靠读取，则回退为“速度值变化判新”并打印显式 warning（已知该兜底无法覆盖同值重复消息）。

### 2. 在训练入口接入桥接器

文件：

* 修改 [train.py](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "scripts/reinforcement_learning/rsl_rl_ros2/train.py")

实现内容：

1. 在 [gym.make(...)](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "gym.make(...)") 之后、[RslRlVecEnvWrapper(...)](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "RslRlVecEnvWrapper(...)") 之前接入桥接器。
2. 仅当任务是 **Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0** 或 **...-Play-v0** 时启用桥接。
3. 执行 **wait_for_first_message()**：
   * **startup_blocking**：等待首条命令直到 **15s**，超时打印 warning，继续训练。
4. 用 **try/finally** 保证 **adapter.close()** 与 **env.close()** 顺序可靠。

### 3. 完成第5步命令切换

文件：

* 修改 [flat_env_cfg.py](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/flat_env_cfg.py")

实现内容：

1. 在 **UnitreeGo1Ros2CmdFlatEnvCfg.__post_init__()** 中：
   * 先 **super().__post_init__()**
   * 读取原 **self.commands.base_velocity.ranges**
   * 替换为 [mdp.Ros2VelocityCommandCfg(...)](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "mdp.Ros2VelocityCommandCfg(...)")
   * 将原范围映射到 **Ros2VelocityCommandCfg.Ranges(lin_vel_x, lin_vel_y, ang_vel_z)**
   * 固定 **cmd_timeout_s=0.5**
   * 固定 **command_attr/command_stamp_attr** 为当前实现字段名
2. **PLAY** 配置继承同样行为，保持奖励/终止/PPO 配置不变。

### 4. 最小冒烟验证

验证目标：证明链路“ROS2 Twist -> env attributes -> Ros2VelocityCommand”真实生效。

场景：

1. 启动发布者（WSL ROS2，20Hz，常值 Twist）。
2. 启动训练（Windows，**num_envs=1**, **max_iterations=1**, ROS2Cmd 任务）。
3. 验收标准：

* 训练可启动，不出现命令项/图初始化异常。
* **startup_blocking** 在发布者在线时快速通过。
* 运行期间 **ros2_latest_cmd_vel** 非零且与发布值一致（**vx/vy/wz** 映射正确）。
* **ros2_latest_cmd_stamp_s** 在重复同值发布时仍持续更新（证明“同值重复消息不误超时”）。
* 中断发布后，**Ros2VelocityCommand** 在 [0.5s](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "0.5s") 后触发回零路径（不崩溃）。

## 关键边界与失败处理

1. 缺失 **isaacsim.ros2.bridge** 扩展：仅对 ROS2Cmd 任务报错，其他任务不受影响。
2. 图已存在：复用或先清理后重建，避免重复节点冲突。
3. 回调重复注册：使用唯一 callback 名称并在注册前检查/移除。
4. 任何桥接初始化失败：给出明确日志（图路径、topic、模式、超时参数）。

## Assumptions / Defaults

1. 本轮范围严格锁定“第4+5步 + 最小冒烟”，不做 CLI 参数扩展。
2. 默认 topic 固定 **/go1/cmd_vel**。
3. 默认模式固定 **startup_blocking**，超时 **15s**，命令超时 [0.5s](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/SNight/.vscode/extensions/openai.chatgpt-0.4.71-win32-x64/webview/# "0.5s")。
4. 仅改动 ROS2 新任务链路，不影响 **Isaac-Velocity-Flat-Unitree-Go1-v0** 原任务。
