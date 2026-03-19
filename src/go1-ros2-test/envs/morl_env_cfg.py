# Canonical source: src/go1-ros2-test/envs/morl_env_cfg.py
# Deployed to: robot_lab/.../config/quadruped/unitree_go1_ros2/morl_env_cfg.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MORL (Multi-Objective Reinforcement Learning) environment configuration.

This module defines the MORL environment that extends the Rough ROS2Cmd baseline
with 4 primary objectives:
- Speed (velocity tracking)
- Energy (power consumption)
- Smoothness (action rate)
- Stability (angular velocity)

Design decisions:
1. Inherit from UnitreeGo1Ros2CmdRoughEnvCfg to preserve baseline structure
2. Add 4 primary objectives as new reward terms
3. Disable overlapping baseline penalties (dof_torques_l2, action_rate_l2)
4. Keep auxiliary constraints fixed (track_ang_vel_z_exp, lin_vel_z_l2, etc.)
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo1Ros2CmdRoughEnvCfg, UnitreeGo1Ros2CmdRoughEnvCfg_PLAY

# Import MORL reward functions from local mdp module
from . import mdp as local_mdp


def _configure_morl_rewards(cfg) -> None:
    """Apply the MORL reward structure on top of the rough ROS2 baseline."""

    cfg.rewards.track_lin_vel_xy_exp.func = local_mdp.morl_track_vel_exp
    cfg.rewards.track_lin_vel_xy_exp.weight = 0.25
    cfg.rewards.track_lin_vel_xy_exp.params = {
        "command_name": "base_velocity",
        "scale": 1.0,  # was 5.0; lowered to escape dead-gradient plateau (see 2026-3-18 log)
    }

    cfg.rewards.morl_energy = RewTerm(
        func=local_mdp.morl_energy_power_exp,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale": 0.005,  # was 1.0; lowered to match Go1 power range ~5-600W (see 2026-3-19 log)
        },
    )
    cfg.rewards.morl_smooth = RewTerm(
        func=local_mdp.morl_action_smoothness_exp,
        weight=0.25,
        params={"scale": 0.01},
    )
    cfg.rewards.morl_stable = RewTerm(
        func=local_mdp.morl_stability_ang_vel_exp,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale": 1.0,
        },
    )

    cfg.rewards.track_ang_vel_z_exp.weight = 0.75
    cfg.rewards.lin_vel_z_l2.weight = -2.0
    cfg.rewards.ang_vel_xy_l2.weight = -0.05
    cfg.rewards.dof_acc_l2.weight = -2.5e-07
    cfg.rewards.feet_air_time.weight = 0.01

    cfg.rewards.dof_torques_l2.weight = 0.0
    cfg.rewards.action_rate_l2.weight = 0.0
    cfg.rewards.flat_orientation_l2.weight = 0.0
    cfg.rewards.dof_pos_limits.weight = 0.0
    cfg.rewards.undesired_contacts = None


@configclass
class UnitreeGo1MORLEnvCfg(UnitreeGo1Ros2CmdRoughEnvCfg):
    """Go1 MORL task config with 4 primary objectives.

    Primary objectives (weights configurable via CLI):
    - morl_speed: velocity tracking (default weight: 0.25)
    - morl_energy: power consumption (default weight: 0.25)
    - morl_smooth: action smoothness (default weight: 0.25)
    - morl_stable: angular velocity stability (default weight: 0.25)

    Auxiliary constraints (fixed weights, inherited from baseline):
    - track_ang_vel_z_exp: 0.75
    - lin_vel_z_l2: -2.0
    - ang_vel_xy_l2: -0.05
    - dof_acc_l2: -2.5e-07
    - feet_air_time: 0.01

    Disabled baseline terms (replaced by primary objectives):
    - dof_torques_l2: 0.0 (replaced by morl_energy)
    - action_rate_l2: 0.0 (replaced by morl_smooth)
    """

    def __post_init__(self):
        super().__post_init__()
        _configure_morl_rewards(self)


@configclass
class UnitreeGo1MORLEnvCfg_PLAY(UnitreeGo1Ros2CmdRoughEnvCfg_PLAY):
    """Go1 MORL play config with 4 primary objectives.

    Same reward structure as UnitreeGo1MORLEnvCfg but with play-specific settings:
    - Reduced num_envs (50)
    - Disabled observation corruption
    - Disabled push events
    """

    def __post_init__(self):
        super().__post_init__()
        _configure_morl_rewards(self)
