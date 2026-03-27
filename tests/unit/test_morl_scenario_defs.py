# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Phase 4 MORL scenario definitions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "scenario_defs.py"


def _load_module():
    module_name = "_morl_scenario_defs_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_scenario_defs_module_exists():
    assert MODULE_PATH.exists(), f"Missing scenario definitions module: {MODULE_PATH}"


def test_list_scenarios_returns_expected_stage4_ids():
    module = _load_module()

    assert module.list_scenarios() == ["S1", "S2", "S3", "S4", "S5", "S6"]


def test_get_scenario_spec_returns_fixed_command_and_analysis_metadata():
    module = _load_module()

    spec = module.get_scenario_spec("S1")

    assert spec.scenario_id == "S1"
    assert spec.command_vx == 1.0
    assert spec.terrain_mode == "plane"
    assert spec.analysis_group == "main"
    assert spec.disturbance_mode == "none"


def test_get_scenario_spec_marks_s6_as_disturbance_case():
    module = _load_module()

    spec = module.get_scenario_spec("S6")

    assert spec.scenario_id == "S6"
    assert spec.command_vx == 0.8
    assert spec.analysis_group == "stress"
    assert spec.disturbance_mode != "none"
