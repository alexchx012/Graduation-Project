# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Integration test stub for Phase 2 Flat reward-weight variant task registration.

These tests require a running Isaac Sim instance and are marked with
``sim_required`` so they can be skipped in CI environments without a GPU.
"""

import pytest

pytestmark = pytest.mark.sim_required

# -- Task IDs under test --
VARIANT_A_TRAIN = "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-TrackVelHigh-v0"
VARIANT_A_PLAY = "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-TrackVelHigh-Play-v0"
VARIANT_B_TRAIN = "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-ActionRateHigh-v0"
VARIANT_B_PLAY = "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-ActionRateHigh-Play-v0"

BASELINE_TRAIN = "Isaac-Velocity-Flat-Unitree-Go1-ROS2Cmd-v0"


def _get_spec(task_id: str):
    import gymnasium as gym

    import robot_lab.tasks  # noqa: F401 — triggers gym.register calls

    return gym.spec(task_id)


# ---- Variant A: track_lin_vel_xy_exp 1.5 → 3.5 ----


def test_variant_a_train_task_registered():
    """Verify variant A training task can be resolved by gymnasium."""
    spec = _get_spec(VARIANT_A_TRAIN)
    assert spec is not None
    assert "flat_env_cfg_variants:UnitreeGo1Ros2CmdFlatEnvCfg_TrackVelHigh" in spec.kwargs["env_cfg_entry_point"]


def test_variant_a_play_task_registered():
    """Verify variant A play task can be resolved by gymnasium."""
    spec = _get_spec(VARIANT_A_PLAY)
    assert spec is not None
    assert "flat_env_cfg_variants:UnitreeGo1Ros2CmdFlatEnvCfg_TrackVelHigh_PLAY" in spec.kwargs["env_cfg_entry_point"]


def test_variant_a_uses_flat_ppo_runner():
    """Variant A must use Flat PPO runner (same as baseline for fair comparison)."""
    for task_id in (VARIANT_A_TRAIN, VARIANT_A_PLAY):
        spec = _get_spec(task_id)
        runner_path = spec.kwargs["rsl_rl_cfg_entry_point"]
        assert "FlatPPORunnerCfg" in runner_path, (
            f"{task_id} should use FlatPPORunnerCfg, got: {runner_path}"
        )


# ---- Variant B: action_rate_l2 -0.01 → -0.05 ----


def test_variant_b_train_task_registered():
    """Verify variant B training task can be resolved by gymnasium."""
    spec = _get_spec(VARIANT_B_TRAIN)
    assert spec is not None
    assert "flat_env_cfg_variants:UnitreeGo1Ros2CmdFlatEnvCfg_ActionRateHigh" in spec.kwargs["env_cfg_entry_point"]


def test_variant_b_play_task_registered():
    """Verify variant B play task can be resolved by gymnasium."""
    spec = _get_spec(VARIANT_B_PLAY)
    assert spec is not None
    assert "flat_env_cfg_variants:UnitreeGo1Ros2CmdFlatEnvCfg_ActionRateHigh_PLAY" in spec.kwargs["env_cfg_entry_point"]


def test_variant_b_uses_flat_ppo_runner():
    """Variant B must use Flat PPO runner (same as baseline for fair comparison)."""
    for task_id in (VARIANT_B_TRAIN, VARIANT_B_PLAY):
        spec = _get_spec(task_id)
        runner_path = spec.kwargs["rsl_rl_cfg_entry_point"]
        assert "FlatPPORunnerCfg" in runner_path, (
            f"{task_id} should use FlatPPORunnerCfg, got: {runner_path}"
        )

