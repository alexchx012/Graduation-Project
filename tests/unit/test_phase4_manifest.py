# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Phase 4 manifest helpers."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "phase4_manifest.py"


def _load_module():
    module_name = "_phase4_manifest_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_tmp_path():
    base = ROOT / ".tmp_test_phase4_manifest"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_load_phase4_manifest_reads_morl_and_baseline_entries(local_tmp_path):
    module = _load_module()

    manifest_path = local_tmp_path / "phase4_main_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "family": "morl",
                        "policy_id": "P1",
                        "canonical_seed": 42,
                        "run_dir": "D:/Graduation-Project/logs/rsl_rl/unitree_go1_rough/2026-03-31_16-11-33_morl_p1_seed42",
                        "checkpoint": "model_899.pt",
                        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                        "output_stem": "morl_p1_seed42",
                        "evidence_layer": "A",
                        "official_hv_eligible": True,
                    },
                    {
                        "family": "baseline",
                        "policy_id": "baseline",
                        "canonical_seed": 43,
                        "run_dir": "D:/Graduation-Project/logs/rsl_rl/unitree_go1_rough/_archive_20260331_pre_v2_sweep/2026-03-10_10-59-57_baseline_rough_seed43",
                        "checkpoint": "model_1499.pt",
                        "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0",
                        "output_stem": "baseline_seed43",
                        "evidence_layer": "D",
                        "official_hv_eligible": False,
                        "source_state": "archive",
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    entries = module.load_phase4_manifest(manifest_path)

    assert [entry.policy_id for entry in entries] == ["P1", "baseline"]
    assert entries[0].family == "morl"
    assert entries[0].task == "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2"
    assert entries[1].family == "baseline"
    assert entries[1].canonical_seed == 43
    assert entries[1].checkpoint == "model_1499.pt"
    assert entries[1].source_state == "archive"


def test_load_phase4_manifest_rejects_missing_required_fields(local_tmp_path):
    module = _load_module()

    manifest_path = local_tmp_path / "phase4_main_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "family": "morl",
                        "policy_id": "P1",
                        "canonical_seed": 42,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        module.load_phase4_manifest(manifest_path)


def test_filter_phase4_manifest_entries_applies_policy_seed_and_family_filters(local_tmp_path):
    module = _load_module()

    manifest_path = local_tmp_path / "phase4_main_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "family": "morl",
                        "policy_id": "P1",
                        "canonical_seed": 42,
                        "run_dir": "D:/Graduation-Project/logs/rsl_rl/unitree_go1_rough/2026-03-31_16-11-33_morl_p1_seed42",
                        "checkpoint": "model_899.pt",
                        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                        "output_stem": "morl_p1_seed42",
                        "evidence_layer": "A",
                        "official_hv_eligible": True
                    },
                    {
                        "family": "morl",
                        "policy_id": "P1",
                        "canonical_seed": 43,
                        "run_dir": "D:/Graduation-Project/logs/rsl_rl/unitree_go1_rough/2026-04-02_14-37-09_morl_p1_seed43",
                        "checkpoint": "model_899.pt",
                        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                        "output_stem": "morl_p1_seed43",
                        "evidence_layer": "A",
                        "official_hv_eligible": True
                    },
                    {
                        "family": "baseline",
                        "policy_id": "baseline",
                        "canonical_seed": 42,
                        "run_dir": "D:/Graduation-Project/logs/rsl_rl/unitree_go1_rough/2026-03-08_16-46-27_baseline_rough_ros2cmd",
                        "checkpoint": "model_1499.pt",
                        "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0",
                        "output_stem": "baseline_seed42",
                        "evidence_layer": "D",
                        "official_hv_eligible": False
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    entries = module.load_phase4_manifest(manifest_path)

    filtered = module.filter_phase4_manifest_entries(
        entries,
        policy_ids={"P1", "BASELINE"},
        seeds={42},
        families={"baseline"},
    )

    assert [(entry.family, entry.policy_id, entry.canonical_seed) for entry in filtered] == [
        ("baseline", "baseline", 42),
    ]


def test_load_phase4_manifest_raises_file_not_found_for_missing_manifest(local_tmp_path):
    module = _load_module()

    missing_path = local_tmp_path / "missing_phase4_main_manifest.json"

    with pytest.raises(FileNotFoundError):
        module.load_phase4_manifest(missing_path)
