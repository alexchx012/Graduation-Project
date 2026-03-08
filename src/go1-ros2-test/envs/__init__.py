# Canonical source: src/go1-ros2-test/envs/__init__.py
# Deployed to: robot_lab/.../config/quadruped/unitree_go1_ros2/__init__.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo1Ros2CmdFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents."
            "rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo1Ros2CmdFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents."
            "rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo1Ros2CmdRoughEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents."
            "rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo1Ros2CmdRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents."
            "rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg"
        ),
    },
)

##
# Phase 2 — Reward-weight variant tasks (One-factor-at-a-time experiments).
##

# Variant A: track_lin_vel_xy_exp weight 1.5 → 3.5
gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-TrackVelHigh-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_variants:UnitreeGo1Ros2CmdFlatEnvCfg_TrackVelHigh",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents."
            "rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-TrackVelHigh-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_variants:UnitreeGo1Ros2CmdFlatEnvCfg_TrackVelHigh_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents."
            "rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg"
        ),
    },
)

# Variant B: action_rate_l2 weight -0.01 → -0.05
gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-ActionRateHigh-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_variants:UnitreeGo1Ros2CmdFlatEnvCfg_ActionRateHigh",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents."
            "rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-ActionRateHigh-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_variants:UnitreeGo1Ros2CmdFlatEnvCfg_ActionRateHigh_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents."
            "rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg"
        ),
    },
)
