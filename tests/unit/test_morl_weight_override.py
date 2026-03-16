# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MORL weight override behavior and config wiring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLI_MODULE_PATH = ROOT / "scripts" / "go1-ros2-test" / "morl_cli.py"
MORL_ENV_CFG_PATH = ROOT / "src" / "go1-ros2-test" / "envs" / "morl_env_cfg.py"
TRAIN_SCRIPT_PATH = ROOT / "scripts" / "go1-ros2-test" / "train.py"
EVAL_SCRIPT_PATH = ROOT / "scripts" / "go1-ros2-test" / "eval.py"


def _load_cli_module():
    module_name = "_morl_cli_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, CLI_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class DummyRewardTerm:
    def __init__(self, weight: float):
        self.weight = weight


def _make_dummy_env_cfg():
    rewards = SimpleNamespace(
        track_lin_vel_xy_exp=DummyRewardTerm(0.25),
        morl_energy=DummyRewardTerm(0.25),
        morl_smooth=DummyRewardTerm(0.25),
        morl_stable=DummyRewardTerm(0.25),
        track_ang_vel_z_exp=DummyRewardTerm(0.75),
        lin_vel_z_l2=DummyRewardTerm(-2.0),
        ang_vel_xy_l2=DummyRewardTerm(-0.05),
        dof_acc_l2=DummyRewardTerm(-2.5e-07),
        feet_air_time=DummyRewardTerm(0.01),
        dof_torques_l2=DummyRewardTerm(0.0),
        action_rate_l2=DummyRewardTerm(0.0),
    )
    return SimpleNamespace(rewards=rewards)


def test_parse_morl_weights_accepts_valid_normalized_vector():
    cli = _load_cli_module()

    weights = cli.parse_morl_weights("0.7,0.1,0.1,0.1")

    assert weights == (0.7, 0.1, 0.1, 0.1)


@pytest.mark.parametrize(
    "raw_value",
    [
        "0.5,0.5",
        "0.5,0.25,0.25,0.25",
        "0.7,-0.1,0.2,0.2",
        "0.4,0.2,0.2,0.1",
    ],
)
def test_parse_morl_weights_rejects_invalid_vectors(raw_value: str):
    cli = _load_cli_module()

    with pytest.raises(ValueError):
        cli.parse_morl_weights(raw_value)


def test_apply_morl_weight_override_changes_only_primary_terms():
    cli = _load_cli_module()
    env_cfg = _make_dummy_env_cfg()

    cli.apply_morl_weight_override(env_cfg, (0.4, 0.3, 0.2, 0.1))

    assert env_cfg.rewards.track_lin_vel_xy_exp.weight == 0.4
    assert env_cfg.rewards.morl_energy.weight == 0.3
    assert env_cfg.rewards.morl_smooth.weight == 0.2
    assert env_cfg.rewards.morl_stable.weight == 0.1
    assert env_cfg.rewards.track_ang_vel_z_exp.weight == 0.75
    assert env_cfg.rewards.lin_vel_z_l2.weight == -2.0
    assert env_cfg.rewards.ang_vel_xy_l2.weight == -0.05
    assert env_cfg.rewards.dof_acc_l2.weight == -2.5e-07
    assert env_cfg.rewards.feet_air_time.weight == 0.01


def test_morl_env_cfg_wires_speed_func_and_fixed_constraints():
    source = MORL_ENV_CFG_PATH.read_text(encoding="utf-8")

    assert "cfg.rewards.track_lin_vel_xy_exp.func = local_mdp.morl_track_vel_exp" in source
    assert '"scale": 5.0' in source
    assert 'cfg.rewards.morl_energy = RewTerm(' in source
    assert 'cfg.rewards.morl_smooth = RewTerm(' in source
    assert 'cfg.rewards.morl_stable = RewTerm(' in source
    assert "cfg.rewards.track_ang_vel_z_exp.weight = 0.75" in source
    assert "cfg.rewards.dof_torques_l2.weight = 0.0" in source
    assert "cfg.rewards.action_rate_l2.weight = 0.0" in source


def test_train_and_eval_scripts_include_morl_task_ids():
    train_source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")
    eval_source = EVAL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "MORL_TASK_IDS as _MORL_TASK_IDS" in train_source
    assert "MORL_TASK_IDS as _MORL_TASK_IDS" in eval_source
    assert "} | set(_MORL_TASK_IDS)" in train_source
    assert "} | set(_MORL_TASK_IDS)" in eval_source


def test_train_script_exposes_morl_weights_cli():
    train_source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--morl_weights" in train_source
    assert "apply_morl_weight_override" in train_source
