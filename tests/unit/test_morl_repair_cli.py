# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MORL repair-pilot CLI behavior."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CLI_MODULE_PATH = ROOT / "scripts" / "go1-ros2-test" / "morl_cli.py"
TRAIN_SCRIPT_PATH = ROOT / "scripts" / "go1-ros2-test" / "train.py"
REPAIR_PILOT_SCRIPT_PATH = ROOT / "scripts" / "phase_morl" / "run_morl_repair_pilots.py"


def _load_cli_module():
    module_name = "_morl_repair_cli_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, CLI_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_repair_runner_module():
    module_name = "_run_morl_repair_pilots_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, REPAIR_PILOT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_dummy_env_cfg():
    return SimpleNamespace(
        commands=SimpleNamespace(
            base_velocity=SimpleNamespace(
                heading_command=True,
                rel_heading_envs=1.0,
                rel_standing_envs=0.02,
                ranges=SimpleNamespace(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-1.0, 1.0),
                    ang_vel_z=(-1.0, 1.0),
                ),
            )
        )
    )


def test_apply_morl_command_profile_freezes_forward_only_distribution():
    cli = _load_cli_module()
    env_cfg = _make_dummy_env_cfg()

    cli.apply_morl_command_profile(env_cfg, "repair_forward_v1")

    base_velocity = env_cfg.commands.base_velocity
    assert base_velocity.ranges.lin_vel_x == (0.5, 1.5)
    assert base_velocity.ranges.lin_vel_y == (0.0, 0.0)
    assert base_velocity.ranges.ang_vel_z == (0.0, 0.0)
    assert base_velocity.heading_command is False
    assert base_velocity.rel_heading_envs == 0.0
    assert base_velocity.rel_standing_envs == 0.0


def test_train_script_exposes_repair_cli_flags():
    source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--morl_command_profile" in source
    assert "--init_checkpoint" in source
    assert "--log_morl_reward_contributions" in source
    assert "apply_morl_command_profile" in source
    assert "load_optimizer=False" in source
    assert "runner.current_learning_iteration = 0" in source
    assert "attach_reward_contribution_logging" in source


def test_repair_pilot_runner_builds_warm_start_command_for_pilot_b():
    module = _load_repair_runner_module()

    pilot = module._build_repair_experiment("B")
    cmd = module._build_repair_train_cmd(ROOT, pilot, seed=42, max_iterations=600)
    joined = " ".join(cmd)

    assert pilot["policy_id"] == "P10"
    assert pilot["command_profile"] == "repair_forward_v1"
    assert pilot["init_checkpoint"].endswith(
        "2026-03-08_16-46-27_baseline_rough_ros2cmd\\model_1499.pt"
    ) or pilot["init_checkpoint"].endswith(
        "2026-03-08_16-46-27_baseline_rough_ros2cmd/model_1499.pt"
    )
    assert "--morl_command_profile repair_forward_v1" in joined
    assert "--init_checkpoint" in joined
    assert "--max_iterations 600" in joined


def test_repair_pilot_runner_accepts_workspace_root_with_agents_md():
    module = _load_repair_runner_module()

    assert module._is_project_root(ROOT) is True
