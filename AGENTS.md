# AGENTS.md

## Startup Protocol (Mandatory)

Before starting any task in this workspace:

1. Use the default Conda environment: `env_isaaclab`.
2. Use the default WSL distro: `Ubuntu-22.04`.
3. Assume both defaults are valid without re-checking every session.
4. Re-verify only if:
   - the user explicitly asks to verify, or
   - a related command fails.
5. Privilege rule for environment commands:
   - Human user in local terminal: usually no elevation needed.
   - Agent execution in this sandbox: request elevated execution for
     `conda`/`wsl` access when sandbox restrictions block or may block it.

## Task Execution Discipline

### Think first about what you don’t know (Rule 1)

* Before you start, ask yourself: Is the information I have sufficient? What is missing?
* If it isn’t sufficient, say “I’m missing XXX.” Don’t invent a self-consistent story to fill the gaps.
* Don’t use data to make psychological attributions; behavior may have real-world causes you can’t see.

### Don’t take shortcuts (Rule 2)

* Before applying a framework, rule out simpler explanations.
* Don’t use “AI-ish” wording (prohibited: parallelism/rhetorical flourishes, opening with a rhetorical question, end-of-paragraph summaries; if deleting something doesn’t affect the whole, delete it).
* Don’t flatter: would you say the same if someone else did the same thing? If you can’t produce a comparison sample, don’t praise.
* Do only what is asked; you may suggest related work, but don’t take it upon yourself to do it.

### After fixing, verify (Rule 3)

* If you changed code, run build/test; only say “fixed” if it passes.
* If you changed A, grep all references to A and review all related code.
* Debugging: first understand why it’s wrong → locate the issue → minimal fix → verify.

### Distinguish facts from guesses (Rule 4)

* Clearly label in your output: [Fact] [Inference + Evidence] [Assumption] [Don’t know]
* If your output contains no “Don’t know,” go back and reassess.

### Trace corrections to the root cause (Rule 5)

* When corrected, answer three things: which assumption was wrong? what other conclusions are affected? how will you intercept it next time?


## Conda Usage Rules (From Workspace Logs)

After reviewing all files under `docs/daily_logs`:

1. Activate `env_isaaclab` before Windows-side Isaac Lab runtime commands, especially:
   - `isaaclab.bat ...`
   - `scripts/reinforcement_learning/.../train.py`
   - `rsl_rl` training/play launches
2. Do not use conda Python inside WSL ROS2 runtime.
   - For ROS2 (`rclpy`) in WSL, use system Python (`/usr/bin/python3`).
   - Keep conda auto-base disabled in WSL ROS2 sessions.
3. Conda activation is not required for non-runtime work:
   - Markdown/doc updates
   - Reading logs and source files
   - Git/file operations and planning
4. If Isaac Lab launch shows missing `isaaclab`/dependency import errors, switch to
   `env_isaaclab` and rerun.

## Default Environment Commands

### Conda (PowerShell)

```powershell
conda shell.powershell hook | Out-String | Invoke-Expression
conda activate env_isaaclab
```

### WSL (Ubuntu 22.04)

```powershell
wsl -d Ubuntu-22.04 bash -lc "<command>"
```

Fallback distro name (only if needed): `Ubuntu22.04`.

## File Sync Rules (src → robot_lab)

`src/go1-ros2-test/` is the **authoritative copy** for Go1 ROS2 task code.
After editing any file there, the corresponding file in
`robot_lab/source/robot_lab/robot_lab/` **must be updated in the same commit/session**.

Known sync pairs:

| Source (edit here first) | Mirror (keep in sync) |
|---|---|
| `src/go1-ros2-test/ros2_bridge/twist_subscriber_graph.py` | `robot_lab/source/robot_lab/robot_lab/ros2_bridge/twist_subscriber_graph.py` |
| `src/go1-ros2-test/ros2_bridge/__init__.py` | `robot_lab/source/robot_lab/robot_lab/ros2_bridge/__init__.py` |
| `src/go1-ros2-test/envs/mdp/commands/ros2_velocity_command.py` | `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/ros2_velocity_command.py` |
| `src/go1-ros2-test/envs/flat_env_cfg.py` | `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/flat_env_cfg.py` |
| `src/go1-ros2-test/envs/__init__.py` | `robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/__init__.py` |

Notes:
- `scripts/` and `configs/` are standalone runtime files with no counterparts in
  `robot_lab/`, `IsaacLab/`, or `Isaacsim/`; they do **not** need to be synced.
- If a new file is added under `src/go1-ros2-test/`, determine the robot_lab mirror
  path before the session ends and add it to the table above.
- After syncing, grep the mirror file to confirm the diff is intentional (no stale code).

## Last Verified

- Verified on `2026-02-10`.
- `conda activate env_isaaclab` succeeded and pointed to:
  `C:\Users\SNight\anaconda3\envs\env_isaaclab\python.exe`
- `wsl -d Ubuntu-22.04` succeeded.
