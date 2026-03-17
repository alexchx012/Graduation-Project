# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""sim_required stub for the MORL physical evaluation script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim_required

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "run_morl_eval.py"


def _load_module():
    module_name = "_run_morl_eval_script_stub"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_morl_eval_script_defaults_to_morl_play_task():
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args([])

    assert args.task == "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v0"
