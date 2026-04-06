# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for manifest-driven Phase 4 evaluation matrix behavior."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "run_full_eval_matrix.py"


def _load_module():
    module_name = "_run_full_eval_matrix_under_test"
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
    base = ROOT / ".tmp_test_run_full_eval_matrix_manifest"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_parser_accepts_manifest_argument():
    module = _load_module()

    parser = module.build_parser()
    args = parser.parse_args(["--manifest", "scripts/phase_morl/manifests/phase4_main_manifest.json"])

    assert args.manifest.name == "phase4_main_manifest.json"


def test_parser_accepts_validate_argument():
    module = _load_module()

    parser = module.build_parser()
    args = parser.parse_args(["--validate"])

    assert args.validate is True


def test_load_eval_targets_from_manifest_preserves_task_checkpoint_and_output_stem(local_tmp_path):
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
                        "canonical_seed": 42,
                        "run_dir": "D:/Graduation-Project/logs/rsl_rl/unitree_go1_rough/2026-03-08_16-46-27_baseline_rough_ros2cmd",
                        "checkpoint": "model_1499.pt",
                        "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0",
                        "output_stem": "baseline_seed42",
                        "evidence_layer": "D",
                        "official_hv_eligible": False,
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    targets = module.load_eval_targets_from_manifest(manifest_path)

    assert [target.policy_id for target in targets] == ["P1", "baseline"]
    assert targets[0].task == "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2"
    assert targets[0].checkpoint == "model_899.pt"
    assert targets[0].output_stem == "morl_p1_seed42"
    assert targets[1].task == "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0"
    assert targets[1].checkpoint == "model_1499.pt"
    assert targets[1].output_stem == "baseline_seed42"


def test_validate_eval_targets_reports_missing_run_dir_and_checkpoint(local_tmp_path):
    module = _load_module()

    existing_run_dir = local_tmp_path / "2026-03-31_16-11-33_morl_p1_seed42"
    existing_run_dir.mkdir(parents=True)
    (existing_run_dir / "model_899.pt").write_text("checkpoint", encoding="utf-8")

    targets = [
        module.EvalTarget(
            run_dir_name=str(existing_run_dir),
            policy_id="P1",
            seed=42,
            task="Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
            checkpoint="model_899.pt",
            output_stem="morl_p1_seed42",
            family="morl",
        ),
        module.EvalTarget(
            run_dir_name=str(local_tmp_path / "missing_run_dir"),
            policy_id="baseline",
            seed=42,
            task="Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0",
            checkpoint="model_1499.pt",
            output_stem="baseline_seed42",
            family="baseline",
        ),
        module.EvalTarget(
            run_dir_name=str(existing_run_dir),
            policy_id="P2",
            seed=42,
            task="Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
            checkpoint="model_missing.pt",
            output_stem="morl_p2_seed42",
            family="morl",
        ),
    ]

    errors = module.validate_eval_targets(targets)

    assert len(errors) == 2
    assert any("Missing run directory" in err for err in errors)
    assert any("Missing checkpoint" in err for err in errors)


def test_validate_mode_returns_success_for_default_manifest():
    import subprocess
    import sys

    cmd = [
        sys.executable,
        str(MODULE_PATH),
        "--manifest",
        str(ROOT / "scripts" / "phase_morl" / "manifests" / "phase4_main_manifest.json"),
        "--validate",
    ]

    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0
    assert "[VALIDATE] OK:" in completed.stdout
