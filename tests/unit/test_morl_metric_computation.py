# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MORL physical metric helpers."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
METRICS_MODULE_PATH = ROOT / "scripts" / "phase_morl" / "metrics.py"


def _load_metrics_module():
    module_name = "_phase_morl_metrics_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, METRICS_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_j_speed_is_zero_for_perfect_tracking():
    metrics = _load_metrics_module()

    commanded_xy = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    actual_xy = commanded_xy.clone()

    result = metrics.compute_j_speed(commanded_xy, actual_xy)

    torch.testing.assert_close(result, torch.tensor(0.0))


def test_j_energy_scales_inverse_with_distance():
    metrics = _load_metrics_module()
    joint_torque = torch.ones(5, 2)
    joint_vel = torch.ones(5, 2)

    result_1m = metrics.compute_j_energy(joint_torque, joint_vel, dt=0.5, distance=1.0)
    result_2m = metrics.compute_j_energy(joint_torque, joint_vel, dt=0.5, distance=2.0)

    torch.testing.assert_close(result_2m, result_1m / 2.0)


def test_j_smooth_matches_expected_oscillation_norm():
    metrics = _load_metrics_module()
    actions = torch.tensor(
        [
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
        ]
    )

    result = metrics.compute_j_smooth(actions)

    expected = 2.0 * math.sqrt(2.0)
    torch.testing.assert_close(result, torch.tensor(expected))


def test_j_stable_combines_pose_and_angular_velocity_rms():
    metrics = _load_metrics_module()
    pose_fluctuation = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    ang_vel_xy = torch.tensor(
        [
            [0.0, 2.0],
            [0.0, 2.0],
        ]
    )

    result = metrics.compute_j_stable(pose_fluctuation, ang_vel_xy)

    expected = 0.5 * 1.0 + 0.5 * 2.0
    torch.testing.assert_close(result, torch.tensor(expected))


def test_summarize_morl_metrics_returns_scalar_mapping():
    metrics = _load_metrics_module()

    summary = metrics.summarize_morl_metrics(
        commanded_xy=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        actual_xy=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        joint_torque=torch.ones(2, 2),
        joint_vel=torch.ones(2, 2),
        actions=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        ang_vel_xy=torch.zeros(2, 2),
        pose_fluctuation=torch.zeros(2, 2),
        dt=0.5,
        distance=2.0,
    )

    assert set(summary) == {"J_speed", "J_energy", "J_smooth", "J_stable"}
    assert all(isinstance(value, float) for value in summary.values())
