# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Integration test stub for Go1 ROS2Cmd Rough task registration.

These tests require a running Isaac Sim instance and are marked with
``sim_required`` so they can be skipped in CI environments without a GPU.
"""

import pytest

pytestmark = pytest.mark.sim_required

ROUGH_TRAIN_TASK = "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-v0"
ROUGH_PLAY_TASK = "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0"


def test_rough_ros2cmd_train_task_registered():
    """Verify the Rough ROS2Cmd training task can be resolved by gymnasium."""
    import gymnasium as gym

    import robot_lab.tasks  # noqa: F401 — triggers gym.register calls

    spec = gym.spec(ROUGH_TRAIN_TASK)
    assert spec is not None
    assert "rough_env_cfg:UnitreeGo1Ros2CmdRoughEnvCfg" in spec.kwargs["env_cfg_entry_point"]
    assert "UnitreeGo1RoughPPORunnerCfg" in spec.kwargs["rsl_rl_cfg_entry_point"]


def test_rough_ros2cmd_play_task_registered():
    """Verify the Rough ROS2Cmd play task can be resolved by gymnasium."""
    import gymnasium as gym

    import robot_lab.tasks  # noqa: F401

    spec = gym.spec(ROUGH_PLAY_TASK)
    assert spec is not None
    assert "rough_env_cfg:UnitreeGo1Ros2CmdRoughEnvCfg_PLAY" in spec.kwargs["env_cfg_entry_point"]
    assert "UnitreeGo1RoughPPORunnerCfg" in spec.kwargs["rsl_rl_cfg_entry_point"]


def test_rough_ros2cmd_uses_rough_ppo_runner():
    """Confirm Rough tasks reference the Rough PPO runner, not the Flat one."""
    import gymnasium as gym

    import robot_lab.tasks  # noqa: F401

    for task_id in (ROUGH_TRAIN_TASK, ROUGH_PLAY_TASK):
        spec = gym.spec(task_id)
        runner_path = spec.kwargs["rsl_rl_cfg_entry_point"]
        assert "RoughPPORunnerCfg" in runner_path, (
            f"{task_id} should use RoughPPORunnerCfg, got: {runner_path}"
        )
        assert "FlatPPORunnerCfg" not in runner_path, (
            f"{task_id} must NOT use FlatPPORunnerCfg, got: {runner_path}"
        )
