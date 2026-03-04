# ROS2 DDS 通信调试详细记录（2026-02-20）

本文记录调试 Go1 训练端（Windows/IsaacLab）与命令发布端（WSL2/ROS2）之间
DDS 通信不通问题的完整排查过程。

---

## 问题现象

来自 2026-02-19 训练日志：

- `cmd_zero_fallback_count` 持续增长至 1000
- `cmd_vx/vy/wz` 始终为 0.0
- `cmd_timeout_count` 为 0（超时逻辑本身正常）

---

## 第一轮（debug_test1）：确认基线

**现象**：

```
[ROS2 Bridge] Sync #1: cmd=[0.0, 0.0, 0.0], rx_time=-inf
[WARNING] ROS2 bridge: No message received within 15.0s, continuing anyway...
```

ROS2 节点与订阅均创建成功，但 `rx_time=-inf`——没有收到任何消息。

**初步结论**：WSL2 与 Windows 之间 DDS 发现（discovery）失败。

---

## 第二轮（debug_test2）：FastRTPS XML 解析错误

添加 FastRTPS 配置和 `ROS_LOCALHOST_ONLY=1` 后：

```
[XMLPARSER Error] Invalid element found into 'discoverySettingsType'. Name: use_WriterLivelinessProtocol
[XMLPARSER Error] Error parsing 'fastrtps_win_to_wsl.xml'
```

**根因**：配置文件使用了无效元素和错误根元素 `<dds>`，与 Fast-DDS schema 不匹配。

**修复**：
- 根据 Fast-DDS 官方 schema 重写 `fastrtps_win_to_wsl.xml` 和 `fastrtps_wsl_to_win.xml`，使用
  `<profiles xmlns="...">`，改用 `<initialPeersList>` 指定单播对端。
- 在 `train.py` 和 `run_ros2_cmd_node.sh` 中同步设置 `ROS_DOMAIN_ID`、
  `RMW_IMPLEMENTATION`、`FASTRTPS_DEFAULT_PROFILES_FILE`。

---

## 第三轮（debug_test3）：loopback 地址不对称问题

XML 解析错误消失，但仍无消息（`rx_time=-inf`）。

**发现**：WSL2 mirrored networking 下，`127.0.0.1` 是隔离命名空间，两侧各自独立。

UDP 探针双向测试结论：

| 方向 | 地址 | 可达 |
|------|------|------|
| Windows → WSL2 | `192.168.0.104` | ✓ |
| Windows → WSL2 | `127.0.0.1` | ✗ |
| WSL2 → Windows | `127.0.0.1` | ✓ |
| WSL2 → Windows | `192.168.0.104` | ✗ |

**修复**：
- 在 `fastrtps_win_to_wsl.xml` 的 `initialPeersList` 中改用 `192.168.0.104`。
- 移除 `ROS_LOCALHOST_ONLY=1`（与实际 IP 冲突）。

---

## 第四轮（debug_test4）：multicast 元数据路径仍存在

仍失败。

**根因**（经查 GitHub Issue #12344）：
WSL2 mirrored networking **不支持 multicast**。即使配置了 unicast `initialPeersList`，
Fast-DDS 默认仍会用 multicast 交换元数据（metatraffic），导致 discovery 往返路径不成立。

**修复**：在 XML 中补充：
- `<transport_descriptors>`：显式定义 UDPv4 传输（含 `interfaceWhiteList`）
- `<metatrafficUnicastLocatorList>` 禁用 multicast 元数据路径
- `<useBuiltinTransports>false</useBuiltinTransports>` + `<userTransports>`

---

## 第五轮（debug_test5）：环境变量未完整透传

仍失败。检查发现 `train.py` 中缺少两个关键变量：

```python
os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
os.environ.setdefault("FASTDDS_DEFAULT_PROFILES_FILE", fastrtps_config)
```

**修复**：在 `train.py` 补全上述变量；同步更新 `run_ros2_cmd_node.sh`（两套 run 脚本）。

---

## 第六轮：正式训练链路对齐与最终验证

还发现正式训练启动脚本的额外问题：

1. `run_vf_go1_formal_training.ps1` 中 `wslpath` 路径转换方式在 PowerShell 下
   会将反斜杠吃掉，导致 `$WslProjectRoot` 为空。

   修复：
   ```powershell
   $ProjectRootForward = $ProjectRoot -replace '\\', '/'
   $WslProjectRoot = (wsl -d Ubuntu-22.04 bash -c "wslpath -u '$ProjectRootForward'").Trim()
   ```

2. `run_vf_go1_formal_training.ps1` 对应的 `run_ros2_cmd_node.sh` 缺少
   `ROS_DOMAIN_ID`、`FASTDDS_BUILTIN_TRANSPORTS` 等 DDS 环境变量。

3. 两套启动脚本均缺少发布节点存活检查（`pgrep -f go1_cmd_script_node.py`）。

所有问题修复后，最小化冒烟验证（`MaxIter=20`, `NumEnvs=64`）通过：

```
ROS2 publisher PID (WSL): 1121
[ROS2 Bridge] Received msg #1: vx=1.000, vy=0.000, wz=0.000
[ROS2 Bridge] Sync #1000: cmd=[1.0, 0.0, 0.0], rx_time=...
Metrics/base_velocity/cmd_zero_fallback_count: 0.0000
Metrics/base_velocity/cmd_vx: 1.0000
```

---

## 第七轮（run_vf_go1_two_stage_training.ps1）：Stage-2 启动失败与修复

在 two-stage（Stage-1→Stage-2）合并同一 W&B run 的场景下，出现了“看起来像通信断开，实际是训练在 Stage-2 初始化阶段退出”的问题。

**首次失败现象（log_dir 冲突）**：

文件：`wandb/run-20260220_220023-vfgo1twostage9d46e35a0f7d/files/output.log`

```text
wandb.sdk.lib.config_util.ConfigError: Attempted to change value of key "log_dir" from ..._stage1... to ..._stage2...
```

**二次失败现象（runner_cfg 冲突）**：

文件：`wandb/run-20260220_220717-vfgo1twostage8a9068c51f9a/files/output.log`

```text
wandb.sdk.lib.config_util.ConfigError: Attempted to change value of key "runner_cfg" ...
If you really want to do this, pass allow_val_change=True to config.update()
```

**排查结论**：

- 该问题不是 ROS2 DDS 链路断开；核心是 **同一 run 恢复训练时，W&B config 对同 key 改值受限**。
- Stage-2 相比 Stage-1 必然会改动 `runner_cfg`（例如 `resume/load_run/run_name`），默认 `wandb.config.update()` 会触发 `ConfigError` 并提前退出。

**修复方案**（保留旧内容，增量修复）：

1. 保留同 run 目录复用逻辑（避免 `log_dir` 漂移）：
   - 条件：`resume + WANDB_RUN_ID + WANDB_RESUME in {allow,must,auto}`
2. 仅在上述条件命中时，对 `rsl_rl` 的 `WandbSummaryWriter.store_config` 进行补丁：
   - `runner_cfg/policy_cfg/alg_cfg/env_cfg` 写入改为 `wandb.config.update(..., allow_val_change=True)`
3. 修复位置：`scripts/go1-ros2-test/train.py`

**最小闭环验证（1+1 iter）**：

- Stage-1 输出：`wandb/run-20260220_222115-vfgo1twostage8b289cfd1106/files/output.log`
- Stage-2 输出：`wandb/run-20260220_222250-vfgo1twostage8b289cfd1106/files/output.log`
- 两阶段 run id 一致：`vfgo1twostage8b289cfd1106`
- Stage-2 关键证据：

```text
[ROS2 Bridge] Received msg #2: vx=1.000, vy=0.000, wz=0.000
[ROS2 Bridge] Sync #1: cmd=[1.0, 0.0, 0.0], rx_time=...
```

说明 Stage-2 已正常进入训练循环并接收到 `vx=1.0` 指令。

---

## 修改文件汇总

| 文件 | 修改内容 |
|------|----------|
| `configs/ros2/fastrtps_win_to_wsl.xml` | 修复 schema；强制 UDPv4；非对称地址策略 |
| `configs/ros2/fastrtps_wsl_to_win.xml` | 同上（WSL2 侧视角） |
| `scripts/go1-ros2-test/train.py` | 补全 `FASTDDS_*` 环境变量；新增 two-stage 同 run 恢复时的 W&B config 改值兼容补丁 |
| `scripts/go1-ros2-test/run/Debug-ROS2-Test/run_ros2_cmd_node.sh` | 补全 DDS 环境变量；强制 `/usr/bin/python3` |
| `scripts/go1-ros2-test/run/Debug-ROS2-Test/run_debug_training.ps1` | 修复 wslpath；增加节点存活检查 |
| `scripts/go1-ros2-test/run/VF-go1-Formal-Training/run_vf_go1_formal_training.ps1` | 修复 wslpath；增加节点存活检查 |
| `scripts/go1-ros2-test/run/VF-go1-Formal-Training/run_ros2_cmd_node.sh` | 补全 DDS 环境变量；强制 `/usr/bin/python3` |
| `scripts/go1-ros2-test/run/VF-go1-Formal-Training/run_vf_go1_two_stage_training.ps1` | 共享 `WANDB_RUN_ID`；增强 Stage-1 输出目录探测（project-root 与 script-local 双路径） |
| `robot_lab/source/robot_lab/robot_lab/ros2_bridge/twist_subscriber_graph.py` | 增强 debug 日志 |

---

## 技术背景参考

- WSL2 mirrored networking 不支持 multicast：GitHub Issue #12344
- Fast-DDS Well-Known Ports: `7400 + 250×domain + 10 + 2×participantId`
  - Domain 0, Participant 0 → metatraffic unicast port = **7410**
- 官方初始 peers 示例：<https://fast-dds.docs.eprosima.com/en/2.6.x/fastdds/use_cases/wifi/initial_peers.html>
- W&B Config 设计（默认不可变，保障可复现）：<https://docs.wandb.ai/guides/track/config/>
- W&B 恢复运行（resume）行为说明：<https://docs.wandb.ai/guides/runs/resuming>
- W&B `allow_val_change` 参数说明：<https://docs.wandb.ai/models/ref/python/experiments/settings>
