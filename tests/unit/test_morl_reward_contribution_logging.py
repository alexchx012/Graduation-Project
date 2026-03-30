# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MORL reward contribution logging helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "go1-ros2-test" / "morl_reward_logging.py"


def _load_module():
    module_name = "_morl_reward_logging_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRewardManager:
    def __init__(self):
        torch = pytest.importorskip("torch")
        self._episode_sums = {
            "track_lin_vel_xy_exp": torch.tensor([6.0, 10.0], dtype=torch.float32),
            "dof_torques_l2": torch.tensor([-2.0, -6.0], dtype=torch.float32),
        }
        self._weights = {
            "track_lin_vel_xy_exp": 1.5,
            "dof_torques_l2": -0.0002,
        }
        self.reset_calls: list[object] = []

    def get_term_cfg(self, term_name: str):
        return SimpleNamespace(weight=self._weights[term_name])

    def reset(self, env_ids=None):
        self.reset_calls.append(env_ids)
        return {"Episode_Reward/track_lin_vel_xy_exp": 0.4}


def test_build_reward_contribution_log_reports_weighted_and_raw_terms():
    module = _load_module()
    reward_manager = _FakeRewardManager()

    extras = module.build_reward_contribution_log(
        reward_manager=reward_manager,
        max_episode_length_s=20.0,
    )

    assert extras["Episode_RewardWeighted/track_lin_vel_xy_exp"] == pytest.approx(0.4)
    assert extras["Episode_RewardRaw/track_lin_vel_xy_exp"] == pytest.approx(0.4 / 1.5)
    assert extras["Episode_RewardWeighted/dof_torques_l2"] == pytest.approx(-0.2)
    assert extras["Episode_RewardRaw/dof_torques_l2"] == pytest.approx(1000.0)


def test_attach_reward_contribution_logging_wraps_reset():
    module = _load_module()
    reward_manager = _FakeRewardManager()

    module.attach_reward_contribution_logging(
        reward_manager=reward_manager,
        max_episode_length_s=20.0,
    )

    extras = reward_manager.reset(env_ids=[0, 1])

    assert reward_manager.reset_calls == [[0, 1]]
    assert "Episode_Reward/track_lin_vel_xy_exp" in extras
    assert "Episode_RewardWeighted/track_lin_vel_xy_exp" in extras
    assert "Episode_RewardRaw/dof_torques_l2" in extras
