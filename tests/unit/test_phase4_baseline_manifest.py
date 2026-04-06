# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for baseline normalization in the default Phase 4 manifest."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "phase4_manifest.py"
DEFAULT_MANIFEST_PATH = ROOT / "scripts" / "phase_morl" / "manifests" / "phase4_main_manifest.json"


def _load_module():
    module_name = "_phase4_manifest_baseline_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_default_phase4_manifest_normalizes_baseline_seed_entries():
    module = _load_module()

    entries = module.load_phase4_manifest(DEFAULT_MANIFEST_PATH)
    baseline_entries = [entry for entry in entries if entry.family == "baseline"]

    assert [entry.canonical_seed for entry in baseline_entries] == [42, 43, 44]
    assert [entry.output_stem for entry in baseline_entries] == [
        "baseline_seed42",
        "baseline_seed43",
        "baseline_seed44",
    ]
    assert all(entry.checkpoint == "model_1499.pt" for entry in baseline_entries)
    assert all(entry.task == "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0" for entry in baseline_entries)
