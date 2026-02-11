# Original path: robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go1_ros2/flat_env_cfg.py
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
        super().__post_init__()

        original_ranges = self.commands.base_velocity.ranges

        self.commands.base_velocity = mdp.Ros2VelocityCommandCfg(
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
            command_attr="ros2_latest_cmd_vel",
            command_stamp_attr="ros2_latest_cmd_stamp_s",
            cmd_timeout_s=0.5,
            ranges=mdp.Ros2VelocityCommandCfg.Ranges(
                lin_vel_x=original_ranges.lin_vel_x,
                lin_vel_y=original_ranges.lin_vel_y,
                ang_vel_z=original_ranges.ang_vel_z,
            ),
        )
