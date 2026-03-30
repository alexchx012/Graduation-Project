# Canonical source: src/go1-ros2-test/envs/morl_env_cfg_v2.py
# Deployed to: robot_lab/.../config/quadruped/unitree_go1_ros2/morl_env_cfg_v2.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MORL v2 environment configuration with a fixed locomotion scaffold."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo1Ros2CmdRoughEnvCfg, UnitreeGo1Ros2CmdRoughEnvCfg_PLAY
from . import mdp as local_mdp


def _configure_morl_v2_rewards(cfg) -> None:
    """Apply a fixed locomotion scaffold plus secondary MORL preference terms."""

    cfg.morl_secondary_scale = 0.25
    cfg.morl_diagnostic_terms = (
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "lin_vel_z_l2",
        "ang_vel_xy_l2",
        "dof_acc_l2",
        "feet_air_time",
        "dof_torques_l2",
        "action_rate_l2",
        "morl_speed",
        "morl_energy",
        "morl_smooth",
        "morl_stable",
    )

    # Fixed locomotion scaffold: keep the baseline walking prior intact.
    cfg.rewards.track_lin_vel_xy_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.weight = 0.75
    cfg.rewards.lin_vel_z_l2.weight = -2.0
    cfg.rewards.ang_vel_xy_l2.weight = -0.05
    cfg.rewards.dof_acc_l2.weight = -2.5e-07
    cfg.rewards.feet_air_time.weight = 0.01
    cfg.rewards.dof_torques_l2.weight = -0.0002
    cfg.rewards.action_rate_l2.weight = -0.01
    cfg.rewards.flat_orientation_l2.weight = 0.0
    cfg.rewards.dof_pos_limits.weight = 0.0
    cfg.rewards.undesired_contacts = None

    secondary_weight = cfg.morl_secondary_scale * 0.25
    cfg.rewards.morl_speed = RewTerm(
        func=local_mdp.morl_v2_speed_pref,
        weight=secondary_weight,
        params={
            "command_name": "base_velocity",
            "scale": 1.0,
        },
    )
    cfg.rewards.morl_energy = RewTerm(
        func=local_mdp.morl_v2_energy_pref,
        weight=secondary_weight,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale": 0.005,
        },
    )
    cfg.rewards.morl_smooth = RewTerm(
        func=local_mdp.morl_v2_smooth_pref,
        weight=secondary_weight,
        params={"scale": 0.01},
    )
    cfg.rewards.morl_stable = RewTerm(
        func=local_mdp.morl_v2_stable_pref,
        weight=secondary_weight,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale": 1.0,
        },
    )


@configclass
class UnitreeGo1MORLEnvCfg_v2(UnitreeGo1Ros2CmdRoughEnvCfg):
    """Go1 MORL v2 config with fixed locomotion scaffold and secondary trade-offs."""

    def __post_init__(self):
        super().__post_init__()
        _configure_morl_v2_rewards(self)


@configclass
class UnitreeGo1MORLEnvCfg_v2_PLAY(UnitreeGo1Ros2CmdRoughEnvCfg_PLAY):
    """Go1 MORL v2 play config with fixed locomotion scaffold and secondary trade-offs."""

    def __post_init__(self):
        super().__post_init__()
        _configure_morl_v2_rewards(self)
