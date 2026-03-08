# Canonical source: src/go1-ros2-test/envs/flat_env_cfg_variants.py
# Deployed to: robot_lab/.../config/quadruped/unitree_go1_ros2/flat_env_cfg_variants.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 reward-weight variant configs for One-factor-at-a-time experiments.

Each variant inherits the clean Flat ROS2Cmd baseline and overrides exactly
ONE reward weight.  All other rewards, terrain, PPO, and command settings
remain identical to the baseline so that any performance delta can be
attributed to the single changed factor.

Variants
--------
A  track_lin_vel_xy_exp  1.5 → 3.5   (stronger velocity-tracking incentive)
B  action_rate_l2       -0.01 → -0.05 (stronger action-smoothness penalty)
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .flat_env_cfg import (
    UnitreeGo1Ros2CmdFlatEnvCfg,
    UnitreeGo1Ros2CmdFlatEnvCfg_PLAY,
)


# ---------------------------------------------------------------------------
# Variant A: track_lin_vel_xy_exp  1.5 → 3.5
# ---------------------------------------------------------------------------

@configclass
class UnitreeGo1Ros2CmdFlatEnvCfg_TrackVelHigh(UnitreeGo1Ros2CmdFlatEnvCfg):
    """Flat ROS2Cmd variant A — increased velocity-tracking reward weight.

    Hypothesis: raising the weight from 1.5 to 3.5 will reduce steady-state
    velocity-tracking error, possibly at the cost of higher torques or
    less smooth actions.
    """

    def __post_init__(self):
        super().__post_init__()
        # --- One-factor override ---
        self.rewards.track_lin_vel_xy_exp.weight = 3.5


@configclass
class UnitreeGo1Ros2CmdFlatEnvCfg_TrackVelHigh_PLAY(
    UnitreeGo1Ros2CmdFlatEnvCfg_PLAY,
):
    """Play/eval counterpart for variant A."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 3.5


# ---------------------------------------------------------------------------
# Variant B: action_rate_l2  -0.01 → -0.05
# ---------------------------------------------------------------------------

@configclass
class UnitreeGo1Ros2CmdFlatEnvCfg_ActionRateHigh(UnitreeGo1Ros2CmdFlatEnvCfg):
    """Flat ROS2Cmd variant B — stronger action-smoothness penalty.

    Hypothesis: increasing the penalty from -0.01 to -0.05 will yield
    smoother joint trajectories with potentially slower velocity response.
    """

    def __post_init__(self):
        super().__post_init__()
        # --- One-factor override ---
        self.rewards.action_rate_l2.weight = -0.05


@configclass
class UnitreeGo1Ros2CmdFlatEnvCfg_ActionRateHigh_PLAY(
    UnitreeGo1Ros2CmdFlatEnvCfg_PLAY,
):
    """Play/eval counterpart for variant B."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.05

