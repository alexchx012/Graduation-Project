# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MORL physical evaluation script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "run_morl_eval.py"


def _load_module():
    module_name = "_run_morl_eval_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_build_parser_has_morl_eval_defaults():
    module = _load_module()

    parser = module.build_parser()
    args = parser.parse_args([])

    assert args.task == "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v0"
    assert args.num_envs == 64
    assert args.eval_steps == 3000
    assert args.warmup_steps == 300
    assert args.summary_json is None


def test_infer_policy_id_from_load_run_directory_name():
    module = _load_module()

    policy_id = module.infer_policy_id(r"D:\Graduation-Project\logs\rsl_rl\unitree_go1_rough\2026-03-16_10-16-41_morl_p10_seed42")

    assert policy_id == "morl_p10_seed42"


def test_run_morl_eval_does_not_import_eval_module_at_runtime():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "from eval import _ROS2_TASK_IDS" not in source


def test_run_morl_eval_updates_sim_after_enabling_ros2_bridge():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'enable_extension("isaacsim.ros2.bridge")' in source
    assert "simulation_app.update()" in source


def test_run_morl_eval_bootstraps_windows_ros2_environment():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'os.environ.setdefault("ROS_DOMAIN_ID", "0")' in source
    assert 'os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")' in source
    assert 'os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")' in source
    assert 'FASTRTPS_DEFAULT_PROFILES_FILE' in source


def test_run_morl_eval_has_allow_no_ros2_flag():
    module = _load_module()

    parser = module.build_parser()
    args = parser.parse_args([])
    assert hasattr(args, "allow_no_ros2")
    assert args.allow_no_ros2 is False

    args_on = parser.parse_args(["--allow_no_ros2"])
    assert args_on.allow_no_ros2 is True


def test_run_morl_eval_fails_fast_on_ros2_timeout():
    """Verify the source contains RuntimeError for ROS2 timeout (fail-fast)."""
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "raise RuntimeError" in source
    assert "No message received on" in source
    assert "Use --allow_no_ros2 to override" in source


def test_run_morl_eval_rejects_zero_cmd_vx_results():
    """Defence-in-depth: zero mean_cmd_vx must be rejected unless --allow_no_ros2."""
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert 'mean_cmd_vx' in source
    assert "mean_cmd_vx ≈ 0.0" in source or 'mean_cmd_vx' in source
