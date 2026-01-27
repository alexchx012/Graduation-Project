# Isaac Lab 速度跟踪 (Go1 平地) - 运行日志

- 开始时间: 2026-01-27 17:02:44
- 目标: 使用必需的 .bat 运行器运行 Unitree Go1 (平地) 的 Isaac Lab 速度跟踪任务。
- 状态: 确认是运行 play（预训练）还是从头训练/短期训练。

## 操作记录
- 计划: 询问用户使用哪种运行模式（play 还是 train）。

- [2026-01-27 17:03:03] 命令:
  - Set-Content docs\daily_logs\2026-01-27_isaaclab_velocity_tracking.md (initialize log)
  - 输出: Log initialized at docs\daily_logs\2026-01-27_isaaclab_velocity_tracking.md
  - 备注: CondaError: Cannot write to condarc file at C:\Users\SNight\.condarc (permission denied) appeared on shell startup.
- [2026-01-27 17:03:21] 命令:
  - Add-Content docs\daily_logs\2026-01-27_isaaclab_velocity_tracking.md (append init-command result)
  - 输出: (none)
  - 备注: CondaError: Cannot write to condarc file at C:\Users\SNight\.condarc (permission denied) appeared on shell startup.
- [2026-01-27 17:03:38] 命令:
  - Add-Content docs\daily_logs\2026-01-27_isaaclab_velocity_tracking.md (append log of previous append)
  - 输出: (none)
  - 备注: CondaError: Cannot write to condarc file at C:\Users\SNight\.condarc (permission denied) appeared on shell startup.
- [2026-01-27 17:04:04] 备注:
  - 为避免无限递归，日志写入操作本身不再单独记录，只在同一条目中注明命令。

- [2026-01-27 17:04:04] 操作:
  - 计划: 询问用户是运行 Play（预训练）还是 Train（从头训练/短期训练）模式来测试 Go1 平地环境。
  - Command used to write this entry: Add-Content docs\daily_logs\2026-01-27_isaaclab_velocity_tracking.md
  - 输出: (none)
  - 备注: CondaError: Cannot write to condarc file at C:\Users\SNight\.condarc (permission denied) appears on shell startup.
- [2026-01-27 17:08:09] 命令:
  - Get-Content IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go1\flat_env_cfg.py -TotalCount 200
  - 输出（摘要）: 确认了 UnitreeGo1FlatEnvCfg 和 UnitreeGo1FlatEnvCfg_PLAY；平地模式使用平面地形并禁用高度扫描。
  - 备注: CondaError: Cannot write to condarc file at C:\Users\SNight\.condarc (permission denied) appears on shell startup.

- [2026-01-27 17:08:09] 命令:
  - Get-Content IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go1\__init__.py -TotalCount 200
  - 输出（摘要）: Found task IDs: Isaac-Velocity-Flat-Unitree-Go1-v0 and Isaac-Velocity-Flat-Unitree-Go1-Play-v0 (plus rough variants).
  - 备注: CondaError: Cannot write to condarc file at C:\Users\SNight\.condarc (permission denied) appears on shell startup.
- [2026-01-27 17:12:44] 命令:
  - rg -n "def add_app_launcher_args|headless" -S IsaacLab\\source\\isaaclab\\isaaclab\\app\\app_launcher.py
  - 输出（摘要）: Confirmed --headless CLI arg exists in AppLauncher.
  - 备注: 之前一些日志追加尝试因工具超时而失败；此条目进行了合并。CondaError: Cannot write to condarc file at C:\\Users\\SNight\\.condarc (permission denied) appears on shell startup.
- [2026-01-27 17:13:33] 命令:
  - Get-Content IsaacLab\\source\\isaaclab_tasks\\isaaclab_tasks\\manager_based\\locomotion\\velocity\\config\\go1\\agents\\rsl_rl_ppo_cfg.py -TotalCount 200
  - 输出（摘要）: UnitreeGo1FlatPPORunnerCfg defaults: max_iterations=300, experiment_name=unitree_go1_flat.
  - 备注: CondaError: Cannot write to condarc file at C:\\Users\\SNight\\.condarc (permission denied) appears on shell startup.
- [2026-01-27 17:14:11] 计划:
  - 使用 IsaacLab rsl_rl 训练脚本对 Go1 平地环境进行短期训练（冒烟测试）。
  - 要运行的命令:
    .\\isaaclab.bat -p scripts\\reinforcement_learning\\rsl_rl\\train.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 --headless --num_envs 64 --max_iterations 20 --seed 0
  - 预期结果: 训练开始，在 IsaacLab\\logs\\rsl_rl\\unitree_go1_flat\\<timestamp> 下创建日志并运行 20 次迭代。
- [2026-01-27 17:18:14] 命令:
  - .\\isaaclab.bat -p scripts\\reinforcement_learning\\rsl_rl\\train.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 --headless --num_envs 64 --max_iterations 20 --seed 0
  - 输出（摘要）: Failed early. Error: ModuleNotFoundError: No module named 'isaaclab'.
  - 原始输出（摘录）:
    [INFO] Using python from: ...\\_isaac_sim\\python.bat
    There was an error running python.
    ModuleNotFoundError: No module named 'isaaclab'
  - 备注: 可能是使用 -p 参数但未激活 conda 环境时，PYTHONPATH 未设置 IsaacLab 包。
- [2026-01-27 17:19:04] 计划:
  - 使用显式的 PYTHONPATH 重试训练，以便在使用 kit python 时可以导入 IsaacLab 包。
  - 要运行的命令 (PowerShell):
    =\"C:\\Users\\SNight\\Desktop\\GP\\Graduation-Project\\IsaacLab\\source\\isaaclab;C:\\Users\\SNight\\Desktop\\GP\\Graduation-Project\\IsaacLab\\source\\isaaclab_assets;C:\\Users\\SNight\\Desktop\\GP\\Graduation-Project\\IsaacLab\\source\\isaaclab_rl;C:\\Users\\SNight\\Desktop\\GP\\Graduation-Project\\IsaacLab\\source\\isaaclab_tasks;C:\\Users\\SNight\\Desktop\\GP\\Graduation-Project\\IsaacLab\\source\\isaaclab_contrib;C:\\Users\\SNight\\Desktop\\GP\\Graduation-Project\\IsaacLab\\source\\isaaclab_mimic;\"; \
    .\\isaaclab.bat -p scripts\\reinforcement_learning\\rsl_rl\\train.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 --headless --num_envs 64 --max_iterations 20 --seed 0
- [2026-01-27 17:23:46] 命令:
  - (PowerShell) Set PYTHONPATH to IsaacLab source packages; run isaaclab.bat -p train.py ... (same as prior)
  - 输出（摘要）: Isaac Sim launched, but failed due to missing dependencies: flatdict, gymnasium, rsl-rl-lib. Also pipapi failed to install prettytable/hidapi/gymnasium. Training did not start.
  - 关键错误:
    - ModuleNotFoundError: No module named 'flatdict'
    - ModuleNotFoundError: No module named 'gymnasium'
    - PackageNotFoundError: rsl-rl-lib
  - 备注: 需要将依赖项安装到 Isaac Sim python 中，或使用为 Isaac Lab 配置的 conda 环境。
- [2026-01-27 17:25:11] 命令:
  - cmd /c conda env list
  - 输出（摘要）: Failed due to CondaError (cannot write to C:\\Users\\SNight\\.condarc) and PermissionError (WinError 5) in conda virtual_packages cuda plugin.

- [2026-01-27 17:25:11] 命令:
  - cmd /c "set CONDA_NO_PLUGINS=true & set CONDARC=...\\configs\\condarc & set CONDA_REPORT_ERRORS=false & conda --no-plugins env list"
  - 输出（摘要）: Still failed with PermissionError [WinError 5] and KeyError('active_prefix_name'). Conda not usable in current shell.
- [2026-01-27 17:27:25] 计划:
  - 将缺失的 Python 依赖项安装到 Isaac Sim python (kit) 中，以满足 isaaclab/isaaclab_tasks 和 rsl-rl 的需求。
  - 要运行的命令（从 IsaacLab 根目录）:
    .\\isaaclab.bat -p -m pip install flatdict==4.0.1 gymnasium==1.2.1 prettytable==3.3.0 hidapi==0.14.0.post2 rsl-rl-lib==3.1.2 "onnxscript>=0.5"
  - 备注: 这会写入 Isaac Sim Python 环境（可能在工作区外），因此可能需要提升权限。
- [2026-01-27 19:08:11] 命令:
  - Get-Content log.txt
  - 输出（摘要）: 训练似乎在 conda env_isaaclab 下启动；配置已解析；日志目录已创建；环境信息已打印。关于 MaterialX、GPU、PCIe 的警告。出现了关于 .condarc 的 CondaError，但在提供的日志片段中没有致命回溯。
- [2026-01-27 19:19:57] 命令:
  - Get-Content log2.txt
  - 输出（摘要）: python process still running (CPU ~9961s). IsaacLab temp log only shows Physx warning. Kit log last lines show SimulationApp startup, then opening Go1 USD asset from omniverse S3; no further progress. No training iteration output after >10 minutes.
- [2026-01-27 19:31:33] 来自 log3.txt 的证据:
  - Kit log size/time changed from 19:02:25 to 19:24:04 (size 332,437 -> 342,242), indicating a long stall during initialization.
  - 训练产物已创建: model_0.pt, model_19.pt, events.out.tfevents；表明训练开始后 20 次迭代快速完成。
  - 未显示致命错误；关于 .condarc 的 CondaError 持续存在但未阻止运行。
- [2026-01-27 20:13:06] 命令:
  - cmd /c "call C:\\Users\\SNight\\anaconda3\\Scripts\\activate.bat env_isaaclab && D:\\IsaacLab\\isaaclab.bat -p scripts\\reinforcement_learning\\rsl_rl\\train.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 --headless --num_envs 64 --max_iterations 20 --seed 0"
  - 输出（摘要）: 训练启动并完成 20/20 次迭代。训练时间约 41.52 秒。日志位于 D:\\IsaacLab\\logs\\rsl_rl\\unitree_go1_flat\\2026-01-27_20-11-41。
  - 警告: PhysxCfg enable_external_forces_every_iteration; ground plane material; FabricManager point instancer; rsl_rl obs_groups policy/critic warnings. No fatal errors.
- [2026-01-27 20:15:30] 总结:
  - 使用管理员权限 + conda 环境激活成功运行训练。
  - 复现命令（单行）:
    cmd /c "call C:\\Users\\SNight\\anaconda3\\Scripts\\activate.bat env_isaaclab && D:\\IsaacLab\\isaaclab.bat -p scripts\\reinforcement_learning\\rsl_rl\\train.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 --headless --num_envs 64 --max_iterations 20 --seed 0"
  - 输出证据: Learning iteration 0..19/20; Training time ~41.52s; logs in D:\\IsaacLab\\logs\\rsl_rl\\unitree_go1_flat\\2026-01-27_20-11-41.
