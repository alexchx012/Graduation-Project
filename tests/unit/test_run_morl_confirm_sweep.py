# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MORL confirmation sweep runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "run_morl_confirm_sweep.py"


def _load_module():
    module_name = "_run_morl_confirm_sweep_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_default_confirmation_set_matches_selected_m8_policies():
    module = _load_module()

    assert module.DEFAULT_CONFIRM_POLICY_IDS == ("P1", "P2", "P3", "P4", "P10")
    assert module.DEFAULT_CONFIRM_SEEDS == [42, 43, 44]

    selected = module._select_confirmation_experiments(None)
    assert [exp["policy_id"] for exp in selected] == list(module.DEFAULT_CONFIRM_POLICY_IDS)


def test_select_confirmation_experiments_accepts_subset_override():
    module = _load_module()

    selected = module._select_confirmation_experiments("P2,P10")

    assert [exp["policy_id"] for exp in selected] == ["P2", "P10"]


def test_select_confirmation_experiments_rejects_unknown_ids():
    module = _load_module()

    try:
        module._select_confirmation_experiments("P2,P11")
    except ValueError as exc:
        assert "Unknown policy ids" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown policy id")
