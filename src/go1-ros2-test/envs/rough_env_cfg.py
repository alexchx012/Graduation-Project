# Canonical source: src/go1-ros2-test/envs/rough_env_cfg.py
# Deployed to: robot_lab/.../config/quadruped/unitree_go1_ros2/rough_env_cfg.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import (
    UnitreeGo1RoughEnvCfg,
    UnitreeGo1RoughEnvCfg_PLAY,
)

from robot_lab.tasks.manager_based.locomotion.velocity import mdp


_ROS2_CMD_TIMEOUT_S = 0.25


@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg(UnitreeGo1RoughEnvCfg):
    """Go1 rough task config for ROS2 high-level command integration.

    Only the command source is replaced (ROS2 Twist subscriber).
    All reward weights, terrain, PPO, and DR settings are inherited from the
    Isaac Lab baseline without modification.
    """

    def __post_init__(self):
        super().__post_init__()

        # [2026-03-22] ROS2 命令源已禁用，使用 Isaac Lab 标准随机命令生成器
        # 原因：ROS2 动态连接导致命令源不一致，影响训练结果可比性
        # 详见：docs/daily_logs/2026-3/2026-03-21/2026-3-21.md
        #
        # 父类 UnitreeGo1RoughEnvCfg 的默认命令源（UniformVelocityCommand）会被保留，
        # 它会在每次环境重置时生成随机速度命令，符合 MORL 多场景训练的设计初衷。

        # original_ranges = self.commands.base_velocity.ranges
        # self.commands.base_velocity = mdp.Ros2VelocityCommandCfg(
        #     command_attr="ros2_latest_cmd_vel",
        #     command_stamp_attr="ros2_latest_cmd_stamp_s",
        #     cmd_timeout_s=_ROS2_CMD_TIMEOUT_S,
        #     ranges=mdp.Ros2VelocityCommandCfg.Ranges(
        #         lin_vel_x=original_ranges.lin_vel_x,
        #         lin_vel_y=original_ranges.lin_vel_y,
        #         ang_vel_z=original_ranges.ang_vel_z,
        #     ),
        # )


@configclass
class UnitreeGo1Ros2CmdRoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg_PLAY):
    """Go1 rough play config for ROS2 high-level command integration."""

    def __post_init__(self):
        super().__post_init__()

        # [2026-03-22] ROS2 命令源已禁用（同训练配置）
        # 使用 Isaac Lab 标准随机命令生成器

        # original_ranges = self.commands.base_velocity.ranges
        # self.commands.base_velocity = mdp.Ros2VelocityCommandCfg(
        #     command_attr="ros2_latest_cmd_vel",
        #     command_stamp_attr="ros2_latest_cmd_stamp_s",
        #     cmd_timeout_s=_ROS2_CMD_TIMEOUT_S,
        #     ranges=mdp.Ros2VelocityCommandCfg.Ranges(
        #         lin_vel_x=original_ranges.lin_vel_x,
        #         lin_vel_y=original_ranges.lin_vel_y,
        #         ang_vel_z=original_ranges.ang_vel_z,
        #     ),
        # )
