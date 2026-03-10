# Canonical source: src/go1-ros2-test/envs/rough_env_cfg_dr_variants.py
# Deployed to: robot_lab/.../config/quadruped/unitree_go1_ros2/rough_env_cfg_dr_variants.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 Domain Randomization variant configs for DR ablation experiments.

Each variant inherits the Rough ROS2Cmd baseline and overrides exactly ONE
DR factor.  All other rewards, terrain, PPO, and command settings remain
identical to the baseline so that any performance delta can be attributed
to the single changed DR factor.

Variants
--------
DRFriction   physics_material   static (0.8,0.8)→(0.5,1.2), dynamic (0.6,0.6)→(0.4,1.0)
DRMass       add_base_mass      (-1.0, 3.0) → (-3.0, 5.0)
DRPush       push_robot         None → re-enabled with velocity_range ±0.5 m/s
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import isaaclab.envs.mdp as base_mdp

from .rough_env_cfg import (
    UnitreeGo1Ros2CmdRoughEnvCfg,
    UnitreeGo1Ros2CmdRoughEnvCfg_PLAY,
)


# ---------------------------------------------------------------------------
# DR-Friction: randomize ground friction (baseline uses fixed 0.8/0.6)
# ---------------------------------------------------------------------------

@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_DRFriction(UnitreeGo1Ros2CmdRoughEnvCfg):
    """Rough ROS2Cmd + friction randomization.

    Baseline uses fixed friction (static=0.8, dynamic=0.6).
    This variant randomizes static to [0.5, 1.2] and dynamic to [0.4, 1.0].
    """

    def __post_init__(self):
        super().__post_init__()
        # --- One-factor DR override ---
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)


@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_DRFriction_PLAY(UnitreeGo1Ros2CmdRoughEnvCfg_PLAY):
    """Play config for DR-Friction variant."""

    def __post_init__(self):
        super().__post_init__()
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)


# ---------------------------------------------------------------------------
# DR-Mass: expand base mass perturbation range
# ---------------------------------------------------------------------------

@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_DRMass(UnitreeGo1Ros2CmdRoughEnvCfg):
    """Rough ROS2Cmd + extended mass perturbation.

    Baseline uses add_base_mass range (-1.0, 3.0) kg.
    This variant expands to (-3.0, 5.0) kg for stronger mass randomization.
    """

    def __post_init__(self):
        super().__post_init__()
        # --- One-factor DR override ---
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 5.0)


@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_DRMass_PLAY(UnitreeGo1Ros2CmdRoughEnvCfg_PLAY):
    """Play config for DR-Mass variant."""

    def __post_init__(self):
        super().__post_init__()
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 5.0)


# ---------------------------------------------------------------------------
# DR-Push: re-enable external push perturbation (disabled in Go1 baseline)
# ---------------------------------------------------------------------------

@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_DRPush(UnitreeGo1Ros2CmdRoughEnvCfg):
    """Rough ROS2Cmd + external push perturbation.

    Go1 baseline disables push_robot (set to None).
    This variant re-enables it with the default LocomotionVelocityRoughEnvCfg
    parameters: interval 10-15s, velocity ±0.5 m/s in x and y.
    """

    def __post_init__(self):
        super().__post_init__()
        # --- One-factor DR override: re-enable push ---
        self.events.push_robot = EventTerm(
            func=base_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
            },
        )


@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_DRPush_PLAY(UnitreeGo1Ros2CmdRoughEnvCfg_PLAY):
    """Play config for DR-Push variant.

    Note: The base PLAY config disables push_robot. This variant re-enables
    it for cross-evaluation (testing DR-Push trained models under perturbation).
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = EventTerm(
            func=base_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
            },
        )

