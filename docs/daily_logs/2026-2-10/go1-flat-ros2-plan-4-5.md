# Go1 ROS2 第4+5步最小落地计划（含冒烟验证）

> **审查更新**：2026-02-11 已融合审查意见，补充完整代码框架和风险处理

## Summary

目标是仅完成你确认的范围：

1. 在 Isaac 进程内落地 **ROS2SubscribeTwist** 图适配器，把命令写入 **env.unwrapped.ros2_latest_cmd_vel** 与 **env.unwrapped.ros2_latest_cmd_stamp_s**。
2. 将 Go1 ROS2 任务的 **commands.base_velocity** 切到 **Ros2VelocityCommandCfg**。
3. 做最小可运行冒烟验证（不扩展训练 CLI 参数）。

## Public APIs / Interfaces 变更

1. 新增内部桥接模块：`robot_lab/source/robot_lab/robot_lab/ros2_bridge/twist_subscriber_graph.py`
2. 新增桥接类接口（内部 API）：**Ros2TwistSubscriberGraphAdapter**
3. 新增内部配置 dataclass（内部 API）：**Ros2TwistBridgeCfg**
4. **unitree_go1_ros2** 任务配置行为变化：**commands.base_velocity** 从随机速度命令改为 ROS2 外部命令项
5. 本轮不新增命令行参数，默认值硬编码在训练入口中（后续再做 CLI 化）

## 依赖关系图

```
                           ┌────────────────────────┐
                           │   isaacsim.ros2.bridge │
                           │    (Isaac Sim 扩展)     │
                           └───────────┬────────────┘
                                       │ enable_extension()
                                       ▼
┌──────────────────┐     setup()     ┌────────────────────────────┐
│   train.py       │ ───────────────>│ Ros2TwistSubscriberGraph   │
│                  │                 │   Adapter                  │
│                  │     attach()    │                            │
│  env = gym.make()│ ───────────────>│  - 创建 OmniGraph          │
│                  │                 │  - 注册 physics callback   │
└──────────────────┘                 └──────────────┬─────────────┘
         │                                          │
         │                              physics callback 每步执行
         │                                          │
         ▼                                          ▼
┌────────────────────┐              ┌─────────────────────────────┐
│ env.unwrapped      │<─────────────│  同步 linearVelocity/       │
│  .ros2_latest_cmd_vel             │  angularVelocity 到 env     │
│  .ros2_latest_cmd_stamp_s         └─────────────────────────────┘
└────────┬───────────┘
         │ command_manager.compute() 时读取
         ▼
┌─────────────────────┐
│ Ros2VelocityCommand │
│   - 判新/超时逻辑   │
│   - 输出 command    │
└─────────────────────┘
```

## Implementation Plan

### 1. 新建 ROS2 图适配器模块

**需要创建的文件结构**：

```
robot_lab/source/robot_lab/robot_lab/ros2_bridge/
├── __init__.py
└── twist_subscriber_graph.py
```

#### 1.1 `__init__.py` 完整内容

```python
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from .twist_subscriber_graph import Ros2TwistBridgeCfg, Ros2TwistSubscriberGraphAdapter

__all__ = ["Ros2TwistBridgeCfg", "Ros2TwistSubscriberGraphAdapter"]
```

#### 1.2 `twist_subscriber_graph.py` 完整实现

**OmniGraph 节点类型信息**（基于 IsaacSim 5.1）：

| 节点名称           | 完整类型路径                                |
| ------------------ | ------------------------------------------- |
| OnPlaybackTick     | `omni.graph.action.OnPlaybackTick`        |
| ROS2SubscribeTwist | `isaacsim.ros2.bridge.ROS2SubscribeTwist` |

**ROS2SubscribeTwist 端口**：

- 输入：`execIn`, `context`, `nodeNamespace`, `qosProfile`, `queueSize`, `topicName`
- 输出：`linearVelocity` (float[3]), `angularVelocity` (float[3]), `execOut`
- **关键约束**：**无 timestamp 输出端口**

```python
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import omni.graph.core as og


@dataclass
class Ros2TwistBridgeCfg:
    """ROS2 Twist 订阅桥接配置"""
  
    graph_path: str = "/ROS2CmdGraph"
    topic_name: str = "/go1/cmd_vel"
    queue_size: int = 10
    startup_mode: Literal["startup_blocking", "non_blocking"] = "startup_blocking"
    startup_timeout_s: float = 15.0
    command_attr: str = "ros2_latest_cmd_vel"
    command_stamp_attr: str = "ros2_latest_cmd_stamp_s"


class Ros2TwistSubscriberGraphAdapter:
    """在 Isaac 进程内订阅 ROS2 Twist 消息的图适配器"""
  
    def __init__(self, cfg: Ros2TwistBridgeCfg):
        self.cfg = cfg
        self._graph = None
        self._env = None
        self._callback_name = f"ros2_twist_sync_{id(self)}"
        self._last_linear = None
        self._last_angular = None
        self._local_rx_time_s = -math.inf
        self._setup_done = False
  
    def setup(self):
        """创建 OmniGraph 和节点"""
        if self._setup_done:
            return
      
        # 检查并清理已存在的图
        existing_graph = og.get_graph_by_path(self.cfg.graph_path)
        if existing_graph is not None:
            print(f"[WARNING] Graph {self.cfg.graph_path} already exists, removing...")
            og.delete_graph(self.cfg.graph_path)
      
        # 创建新图
        keys = og.Controller.Keys
        try:
            (self._graph, (tick_node, sub_node), _, _) = og.Controller.edit(
                {"graph_path": self.cfg.graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                    ],
                    keys.SET_VALUES: [
                        ("SubscribeTwist.inputs:topicName", self.cfg.topic_name),
                        ("SubscribeTwist.inputs:queueSize", self.cfg.queue_size),
                    ],
                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                    ],
                },
            )
            print(f"[INFO] Created OmniGraph: {self.cfg.graph_path}")
            print(f"[INFO] Subscribing to topic: {self.cfg.topic_name}")
            self._setup_done = True
          
        except Exception as e:
            raise RuntimeError(
                f"Failed to create ROS2 subscriber graph:\n"
                f"  graph_path: {self.cfg.graph_path}\n"
                f"  topic: {self.cfg.topic_name}\n"
                f"  error: {e}"
            )
  
    def attach(self, env):
        """附加到环境，注册 physics callback"""
        self._env = env
        unwrapped = getattr(env, "unwrapped", env)
      
        # 初始化 env 属性
        setattr(unwrapped, self.cfg.command_attr, [0.0, 0.0, 0.0])
        setattr(unwrapped, self.cfg.command_stamp_attr, -math.inf)
      
        # 获取 SimulationContext
        sim = getattr(unwrapped, "sim", None)
        if sim is None:
            # 备用路径
            try:
                from isaacsim.core.api import get_current_simulation_context
                sim = get_current_simulation_context()
            except ImportError:
                pass
      
        if sim is None or not hasattr(sim, "add_physics_callback"):
            raise RuntimeError(
                "Cannot find physics callback API. "
                "Expected env.unwrapped.sim.add_physics_callback() to be available."
            )
      
        # 先移除可能存在的同名回调
        try:
            sim.remove_physics_callback(self._callback_name)
        except Exception:
            pass  # 不存在时忽略
      
        sim.add_physics_callback(self._callback_name, self._sync_callback)
        print(f"[INFO] Registered physics callback: {self._callback_name}")
  
    def _sync_callback(self, event):
        """每个 physics step 执行，同步 ROS2 数据到 env 属性"""
        if self._env is None:
            return
      
        # 读取图节点输出
        linear = self._read_attribute(
            f"{self.cfg.graph_path}/SubscribeTwist.outputs:linearVelocity"
        )
        angular = self._read_attribute(
            f"{self.cfg.graph_path}/SubscribeTwist.outputs:angularVelocity"
        )
      
        if linear is None or angular is None:
            return
      
        # 判新：检查值是否变化（因 ROS2SubscribeTwist 无 timestamp 输出）
        is_new = self._check_is_new_message(linear, angular)
      
        if is_new:
            self._last_linear = list(linear)
            self._last_angular = list(angular)
            # 使用仿真时间作为接收时间戳
            self._local_rx_time_s = self._get_sim_time_s()
      
        # 写入 env 属性
        unwrapped = getattr(self._env, "unwrapped", self._env)
        setattr(unwrapped, self.cfg.command_attr, [linear[0], linear[1], angular[2]])
        setattr(unwrapped, self.cfg.command_stamp_attr, self._local_rx_time_s)
  
    def _check_is_new_message(self, linear, angular) -> bool:
        """检查是否为新消息（值变化判新）"""
        if self._last_linear is None:
            return True  # 首次接收
      
        tol = 1e-6
        for i in range(3):
            if abs(linear[i] - self._last_linear[i]) > tol:
                return True
            if abs(angular[i] - self._last_angular[i]) > tol:
                return True
        return False
  
    def _read_attribute(self, attr_path: str) -> list | None:
        """读取 OmniGraph 属性值"""
        try:
            attr = og.Controller.attribute(attr_path)
            if attr is None:
                return None
            value = attr.get()
            if value is None:
                return None
            return list(value)
        except Exception:
            return None
  
    def _get_sim_time_s(self) -> float:
        """获取当前仿真时间"""
        try:
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            return timeline.get_current_time()
        except Exception:
            return time.time()  # 回退到系统时间
  
    def wait_for_first_message(self, timeout_s: float = 15.0) -> bool:
        """阻塞等待首条 ROS2 消息"""
        import omni.kit.app
      
        start = time.time()
        while time.time() - start < timeout_s:
            # 更新一帧，让 OmniGraph 有机会接收消息
            omni.kit.app.get_app().update()
          
            linear = self._read_attribute(
                f"{self.cfg.graph_path}/SubscribeTwist.outputs:linearVelocity"
            )
            if linear is not None and any(abs(v) > 1e-9 for v in linear):
                print(f"[INFO] First ROS2 message received after {time.time() - start:.2f}s")
                return True
          
            time.sleep(0.05)  # 20Hz 检查频率
      
        return False
  
    def close(self):
        """清理资源"""
        # 移除 physics callback
        if self._env is not None:
            try:
                unwrapped = getattr(self._env, "unwrapped", self._env)
                sim = getattr(unwrapped, "sim", None)
                if sim is not None:
                    sim.remove_physics_callback(self._callback_name)
            except Exception as e:
                print(f"[WARNING] Failed to remove physics callback: {e}")
      
        self._env = None
        self._graph = None
        print(f"[INFO] ROS2 Twist subscriber adapter closed")
```

#### 1.3 判新策略说明

> ⚠️ **重要**：`ROS2SubscribeTwist` 节点的 `execOut` 无法在 physics callback 中可靠地监听"变化"。因此采用**值变化判新**策略。

**已知限制**：此方案无法处理"同值重复消息"。但在 20Hz 发布频率下：

- 即使连续两条消息值相同，0.05s 的间隔远小于 0.5s 超时阈值
- 只有在发布者停止后才可能触发超时，此时无新消息是预期行为

---

### 2. 在训练入口接入桥接器

**修改文件**：`scripts/reinforcement_learning/rsl_rl_ros2/train.py`

#### 2.1 扩展启用位置（AppLauncher 后，gym.make 前）

```python
# === 位置：AppLauncher 初始化后 ===
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# === 启用 ROS2 桥接扩展（仅对 ROS2Cmd 任务） ===
_ROS2_TASK_IDS = {
    "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0",
    "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-Play-v0",
}

if args_cli.task in _ROS2_TASK_IDS:
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()  # 确保扩展加载完成
        print("[INFO] isaacsim.ros2.bridge extension enabled")
    except Exception as e:
        raise RuntimeError(
            f"Failed to enable isaacsim.ros2.bridge extension: {e}\n"
            f"This extension is required for ROS2Cmd tasks."
        )
```

#### 2.2 桥接器接入点（gym.make 后，RslRlVecEnvWrapper 前）

```python
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # === ROS2 桥接器接入点（START）===
    ros2_bridge_adapter = None
    if args_cli.task in _ROS2_TASK_IDS:
        try:
            from robot_lab.ros2_bridge import (
                Ros2TwistBridgeCfg,
                Ros2TwistSubscriberGraphAdapter,
            )
          
            bridge_cfg = Ros2TwistBridgeCfg(
                graph_path="/ROS2CmdGraph",
                topic_name="/go1/cmd_vel",
                queue_size=10,
                startup_mode="startup_blocking",
                startup_timeout_s=15.0,
                command_attr="ros2_latest_cmd_vel",
                command_stamp_attr="ros2_latest_cmd_stamp_s",
            )
          
            ros2_bridge_adapter = Ros2TwistSubscriberGraphAdapter(bridge_cfg)
            ros2_bridge_adapter.setup()
            ros2_bridge_adapter.attach(env)
          
            # startup blocking：等待首条消息
            if bridge_cfg.startup_mode == "startup_blocking":
                success = ros2_bridge_adapter.wait_for_first_message(
                    timeout_s=bridge_cfg.startup_timeout_s
                )
                if not success:
                    print(f"[WARNING] ROS2 bridge: No message received within "
                          f"{bridge_cfg.startup_timeout_s}s, continuing anyway...")
                        
        except Exception as e:
            print(f"[ERROR] Failed to setup ROS2 bridge: {e}")
            print("[INFO] Training will continue without ROS2 command input")
            ros2_bridge_adapter = None
    # === ROS2 桥接器接入点（END）===

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
```

#### 2.3 关闭逻辑修改（try/finally 保证顺序）

```python
    # === 修改关闭逻辑，确保顺序 ===
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
      
        # 先关闭桥接器
        if ros2_bridge_adapter is not None:
            try:
                ros2_bridge_adapter.close()
                print("[INFO] ROS2 bridge adapter closed")
            except Exception as e:
                print(f"[WARNING] Error closing ROS2 bridge: {e}")
      
        # 再关闭环境
        env.close()
```

---

### 3. 完成第5步命令切换

**修改文件**：`robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/flat_env_cfg.py`

> ⚠️ **当前问题**：文件只有空继承，缺少 `__post_init__` 实现

**完整修改后的文件**：

```python
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.flat_env_cfg import (
    UnitreeGo1FlatEnvCfg,
    UnitreeGo1FlatEnvCfg_PLAY,
)

from robot_lab.tasks.manager_based.locomotion.velocity import mdp


@configclass
class UnitreeGo1Ros2CmdFlatEnvCfg(UnitreeGo1FlatEnvCfg):
    """Go1 flat task config for ROS2 high-level command integration."""

    def __post_init__(self):
        # 先调用父类的 __post_init__
        super().__post_init__()
      
        # 读取原有的速度范围配置
        original_ranges = self.commands.base_velocity.ranges
      
        # 替换为 ROS2 速度命令配置
        self.commands.base_velocity = mdp.Ros2VelocityCommandCfg(
            asset_name="robot",  # 必须与父类一致
            command_attr="ros2_latest_cmd_vel",
            command_stamp_attr="ros2_latest_cmd_stamp_s",
            cmd_timeout_s=0.5,
            ranges=mdp.Ros2VelocityCommandCfg.Ranges(
                lin_vel_x=original_ranges.lin_vel_x,
                lin_vel_y=original_ranges.lin_vel_y,
                ang_vel_z=original_ranges.ang_vel_z,
            ),
        )


@configclass
class UnitreeGo1Ros2CmdFlatEnvCfg_PLAY(UnitreeGo1FlatEnvCfg_PLAY):
    """Go1 flat play config for ROS2 high-level command integration."""

    def __post_init__(self):
        super().__post_init__()
      
        original_ranges = self.commands.base_velocity.ranges
      
        self.commands.base_velocity = mdp.Ros2VelocityCommandCfg(
            asset_name="robot",
            command_attr="ros2_latest_cmd_vel",
            command_stamp_attr="ros2_latest_cmd_stamp_s",
            cmd_timeout_s=0.5,
            ranges=mdp.Ros2VelocityCommandCfg.Ranges(
                lin_vel_x=original_ranges.lin_vel_x,
                lin_vel_y=original_ranges.lin_vel_y,
                ang_vel_z=original_ranges.ang_vel_z,
            ),
        )
```

---

### 4. 最小冒烟验证

#### Step 1: 启动 ROS2 发布者（WSL Ubuntu-22.04）

```bash
# 终端 1 - WSL
wsl -d Ubuntu-22.04 bash -lc "
source /opt/ros/humble/setup.bash
ros2 topic pub /go1/cmd_vel geometry_msgs/msg/Twist \
    '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}' \
    --rate 20
"
```

#### Step 2: 启动训练（Windows PowerShell）

```powershell
# 终端 2 - Windows
conda activate env_isaaclab
cd D:\Graduation-Project
python scripts/reinforcement_learning/rsl_rl_ros2/train.py `
    --task Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0 `
    --num_envs 1 `
    --max_iterations 1 `
    --headless
```

#### Step 3: 验收检查项

| 检查项           | 预期结果                                       | 验证方法                          |
| ---------------- | ---------------------------------------------- | --------------------------------- |
| 启动无异常       | 无导入错误、无图创建失败                       | 观察控制台输出                    |
| startup_blocking | "First ROS2 message received" 消息在 <15s 出现 | 观察控制台输出                    |
| 命令值正确       | `ros2_latest_cmd_vel = [0.5, 0.0, 0.3]`      | 在代码中添加打印或断点            |
| 时间戳更新       | `ros2_latest_cmd_stamp_s` 每 50ms 更新一次   | 在代码中添加打印                  |
| 超时回退         | 停止发布者后 0.5s，命令回零                    | 停止 `ros2 topic pub`，观察行为 |

#### Step 4: 调试打印（临时，验证后删除）

在 `Ros2VelocityCommand._update_command()` 中添加：

```python
def _update_command(self):
    self._elapsed_time_s += self._env.step_dt
    latest_cmd, has_new_cmd = self._read_latest_command()

    # === 调试打印（验证后删除）===
    if self._elapsed_time_s % 1.0 < self._env.step_dt:  # 每秒打印一次
        print(f"[DEBUG] cmd={latest_cmd.tolist()}, new={has_new_cmd}, "
              f"stamp={self._last_rx_time_s:.3f}, elapsed={self._elapsed_time_s:.3f}")
    # === 调试打印结束 ===

    # ... 原有逻辑 ...
```

---

## 关键边界与失败处理

| 场景                               | 处理方式                                                                |
| ---------------------------------- | ----------------------------------------------------------------------- |
| 缺失 `isaacsim.ros2.bridge` 扩展 | 仅对 ROS2Cmd 任务报错，其他任务不受影响                                 |
| 图已存在                           | 调用 `og.delete_graph()` 先清理后重建                                 |
| 回调重复注册                       | 使用唯一 callback 名称 `ros2_twist_sync_{id(self)}`，注册前检查/移除  |
| 桥接初始化失败                     | 打印错误日志（含图路径、topic、模式、超时参数），继续训练但无 ROS2 输入 |
| 无 ROS2 发布者                     | 等待 15s 后 warning，继续训练，命令为零                                 |
| 发布者中途停止                     | 0.5s 内保持最后命令，之后回零                                           |
| 发布者重新启动                     | 新命令被正确接收，恢复正常                                              |

---

## 边界测试用例

| 测试场景       | 操作                                        | 预期行为                                |
| -------------- | ------------------------------------------- | --------------------------------------- |
| 无 ROS2 发布者 | 仅启动训练                                  | 等待 15s 后 warning，继续训练，命令为零 |
| 发布者中途停止 | 训练中停止 `ros2 topic pub`               | 0.5s 内保持最后命令，之后回零           |
| 发布者重新启动 | 停止后再启动                                | 新命令被正确接收，恢复正常              |
| 非 ROS2 任务   | 使用 `Isaac-Velocity-Flat-Unitree-Go1-v0` | 不加载桥接器，行为与原任务一致          |
| 扩展缺失       | 禁用 `isaacsim.ros2.bridge`               | 清晰的错误提示，不影响其他任务          |

---

## Assumptions / Defaults

1. 本轮范围严格锁定"第4+5步 + 最小冒烟"，不做 CLI 参数扩展
2. 默认 topic 固定 **/go1/cmd_vel**
3. 默认模式固定 **startup_blocking**，超时 **15s**，命令超时 **0.5s**
4. 仅改动 ROS2 新任务链路，不影响 **Isaac-Velocity-Flat-Unitree-Go1-v0** 原任务

---

## 执行顺序建议

1. **创建目录和文件结构**

   - 创建 `robot_lab/source/robot_lab/robot_lab/ros2_bridge/`
   - 创建 `__init__.py` 和 `twist_subscriber_graph.py`
2. **实现 twist_subscriber_graph.py**

   - 按上述代码框架实现
   - 可选：单元测试（仅图创建，不涉及训练）
3. **修改 flat_env_cfg.py**

   - 添加 `__post_init__` 切换命令配置
4. **修改 train.py**

   - 添加扩展启用代码
   - 添加桥接器接入点
   - 添加 `try/finally` 清理逻辑
5. **冒烟验证**

   - 按上述步骤执行
   - 确认调试打印输出正确
   - 清理调试代码

---

## 风险备注

1. **physics callback 执行时机**：假设在 `env.step()` 物理模拟期间执行，如有差异需调整
3. **线程安全**：假设 `og.Controller` 读取和 env 属性写入在同一线程，如有多线程需加锁
4. **值变化判新限制**：无法处理同值重复消息，但在 20Hz 发布频率下影响可忽略
