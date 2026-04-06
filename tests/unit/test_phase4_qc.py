# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Phase 4 QC reporting."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "check_phase4_qc.py"


def _load_module():
    module_name = "_check_phase4_qc_under_test"
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
    base = ROOT / ".tmp_test_phase4_qc"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _eval_payload(*, policy_id: str, cmd_vx: float, mean_cmd_vx: float, scenario_id: str = "S1") -> dict:
    return {
        "policy_id": policy_id,
        "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0",
        "scenario_id": scenario_id,
        "scenario_name": "flat_mid_speed",
        "terrain_mode": "plane",
        "cmd_vx": cmd_vx,
        "disturbance_mode": "none",
        "analysis_group": "main",
        "checkpoint": f"D:/fake/{policy_id}/model.pt",
        "load_run": f"D:/fake/{policy_id}",
        "eval_steps": 3000,
        "warmup_steps": 300,
        "effective_steps": 2700,
        "effective_env_steps": 172800,
        "step_dt": 0.02,
        "mean_cmd_vx": mean_cmd_vx,
        "mean_vx_meas": 0.95,
        "mean_vx_abs_err": 0.05,
        "J_speed": 0.1,
        "J_energy": 90.0,
        "J_smooth": 1.3,
        "J_stable": 0.3,
        "success_rate": 1.0,
        "mean_base_contact_rate": 0.0,
        "mean_timeout_rate": 0.001,
        "recovery_time": None,
        "elapsed_seconds": 10.0,
    }


def test_build_qc_payload_uses_mean_cmd_vx_proxy_and_notes_schema_limitation(local_tmp_path):
    module = _load_module()

    summary_dir = local_tmp_path / "eval"
    manifest_path = local_tmp_path / "phase4_main_manifest.json"
    analysis_config_path = local_tmp_path / "phase4_analysis_config.json"

    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "family": "baseline",
                        "policy_id": "baseline",
                        "canonical_seed": 42,
                        "run_dir": "D:/runs/baseline_seed42",
                        "checkpoint": "model_1499.pt",
                        "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0",
                        "output_stem": "baseline_seed42",
                        "evidence_layer": "D",
                        "official_hv_eligible": False,
                        "source_state": "active",
                    },
                    {
                        "family": "baseline",
                        "policy_id": "baseline",
                        "canonical_seed": 43,
                        "run_dir": "D:/runs/baseline_seed43",
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
    analysis_config_path.write_text(
        json.dumps(
            {
                "official_policy_set": ["P1", "P2", "P3", "P4", "P10"],
                "exploratory_policy_set": ["P5", "P6", "P7", "P8", "P9"],
                "baseline_policy_set": ["baseline"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_json(summary_dir / "baseline_seed42_S1.json", _eval_payload(policy_id="baseline_seed42", cmd_vx=1.0, mean_cmd_vx=1.0))
    _write_json(summary_dir / "baseline_seed43_S1.json", _eval_payload(policy_id="baseline_seed43", cmd_vx=1.0, mean_cmd_vx=1.0))

    payload = module.build_qc_payload(
        summary_dir=summary_dir,
        manifest_path=manifest_path,
        analysis_config_path=analysis_config_path,
        output_dir=local_tmp_path / "aggregated",
    )
    report = module.render_qc_report(payload)

    assert payload["checks"]["mean_cmd_vx_proxy"]["status"] == "PASS"
    assert "schema does not provide explicit zero-command fallback counter" in report
    assert "mean_cmd_vx proxy check" in report


def test_build_qc_payload_flags_degenerate_archived_baseline_seed(local_tmp_path):
    module = _load_module()

    summary_dir = local_tmp_path / "eval"
    manifest_path = local_tmp_path / "phase4_main_manifest.json"
    analysis_config_path = local_tmp_path / "phase4_analysis_config.json"

    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "family": "baseline",
                        "policy_id": "baseline",
                        "canonical_seed": 42,
                        "run_dir": "D:/runs/baseline_seed42",
                        "checkpoint": "model_1499.pt",
                        "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0",
                        "output_stem": "baseline_seed42",
                        "evidence_layer": "D",
                        "official_hv_eligible": False,
                        "source_state": "active",
                    },
                    {
                        "family": "baseline",
                        "policy_id": "baseline",
                        "canonical_seed": 43,
                        "run_dir": "D:/runs/baseline_seed43",
                        "checkpoint": "model_1499.pt",
                        "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0",
                        "output_stem": "baseline_seed43",
                        "evidence_layer": "D",
                        "official_hv_eligible": False,
                        "source_state": "archive",
                    },
                    {
                        "family": "baseline",
                        "policy_id": "baseline",
                        "canonical_seed": 44,
                        "run_dir": "D:/runs/baseline_seed44",
                        "checkpoint": "model_1499.pt",
                        "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0",
                        "output_stem": "baseline_seed44",
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
    analysis_config_path.write_text(
        json.dumps(
            {
                "official_policy_set": ["P1", "P2", "P3", "P4", "P10"],
                "exploratory_policy_set": ["P5", "P6", "P7", "P8", "P9"],
                "baseline_policy_set": ["baseline"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_json(summary_dir / "baseline_seed42_S1.json", _eval_payload(policy_id="baseline_seed42", cmd_vx=1.0, mean_cmd_vx=1.0))
    bad = _eval_payload(policy_id="baseline_seed43", cmd_vx=1.0, mean_cmd_vx=1.0)
    bad["mean_vx_meas"] = 0.002
    bad["mean_vx_abs_err"] = 0.998
    bad["J_speed"] = 0.999
    bad["J_energy"] = 276.0
    bad["J_smooth"] = 0.03
    _write_json(summary_dir / "baseline_seed43_S1.json", bad)
    _write_json(summary_dir / "baseline_seed44_S1.json", _eval_payload(policy_id="baseline_seed44", cmd_vx=1.0, mean_cmd_vx=1.0))

    payload = module.build_qc_payload(
        summary_dir=summary_dir,
        manifest_path=manifest_path,
        analysis_config_path=analysis_config_path,
        output_dir=local_tmp_path / "aggregated",
    )
    report = module.render_qc_report(payload)

    assert payload["checks"]["baseline_degenerate_control"]["status"] == "WARN"
    assert payload["checks"]["baseline_degenerate_control"]["degenerate_seed_ids"] == [43]
    assert "baseline seed43" in report
    assert "degenerate archived control" in report


def test_build_qc_payload_flags_seed_outlier_without_excluding_it(local_tmp_path):
    module = _load_module()

    summary_dir = local_tmp_path / "eval"
    manifest_path = local_tmp_path / "phase4_main_manifest.json"
    analysis_config_path = local_tmp_path / "phase4_analysis_config.json"

    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "family": "morl",
                        "policy_id": "P3",
                        "canonical_seed": 42,
                        "run_dir": "D:/runs/morl_p3_seed42",
                        "checkpoint": "model_899.pt",
                        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                        "output_stem": "morl_p3_seed42",
                        "evidence_layer": "A",
                        "official_hv_eligible": True,
                        "source_state": "active",
                    },
                    {
                        "family": "morl",
                        "policy_id": "P3",
                        "canonical_seed": 43,
                        "run_dir": "D:/runs/morl_p3_seed43",
                        "checkpoint": "model_899.pt",
                        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                        "output_stem": "morl_p3_seed43",
                        "evidence_layer": "A",
                        "official_hv_eligible": True,
                        "source_state": "active",
                    },
                    {
                        "family": "morl",
                        "policy_id": "P3",
                        "canonical_seed": 44,
                        "run_dir": "D:/runs/morl_p3_seed44",
                        "checkpoint": "model_899.pt",
                        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                        "output_stem": "morl_p3_seed44",
                        "evidence_layer": "A",
                        "official_hv_eligible": True,
                        "source_state": "active",
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    analysis_config_path.write_text(
        json.dumps(
            {
                "official_policy_set": ["P1", "P2", "P3", "P4", "P10"],
                "exploratory_policy_set": ["P5", "P6", "P7", "P8", "P9"],
                "baseline_policy_set": ["baseline"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    low = _eval_payload(policy_id="morl_p3_seed42", cmd_vx=0.5, mean_cmd_vx=0.5, scenario_id="S5")
    low["task"] = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2"
    low["scenario_name"] = "stairs_15cm"
    low["terrain_mode"] = "stairs_15cm"
    low["mean_vx_meas"] = 0.18
    _write_json(summary_dir / "morl_p3_seed42_S5.json", low)

    mid = _eval_payload(policy_id="morl_p3_seed43", cmd_vx=0.5, mean_cmd_vx=0.5, scenario_id="S5")
    mid["task"] = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2"
    mid["scenario_name"] = "stairs_15cm"
    mid["terrain_mode"] = "stairs_15cm"
    mid["mean_vx_meas"] = 0.488
    _write_json(summary_dir / "morl_p3_seed43_S5.json", mid)

    high = _eval_payload(policy_id="morl_p3_seed44", cmd_vx=0.5, mean_cmd_vx=0.5, scenario_id="S5")
    high["task"] = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2"
    high["scenario_name"] = "stairs_15cm"
    high["terrain_mode"] = "stairs_15cm"
    high["mean_vx_meas"] = 0.441
    _write_json(summary_dir / "morl_p3_seed44_S5.json", high)

    payload = module.build_qc_payload(
        summary_dir=summary_dir,
        manifest_path=manifest_path,
        analysis_config_path=analysis_config_path,
        output_dir=local_tmp_path / "aggregated",
    )
    report = module.render_qc_report(payload)

    assert payload["checks"]["seed_outlier"]["status"] == "WARN"
    assert payload["checks"]["seed_outlier"]["outlier_count"] == 1
    assert payload["checks"]["seed_outlier"]["rows"][0]["policy_id"] == "P3"
    assert payload["checks"]["seed_outlier"]["rows"][0]["scenario_id"] == "S5"
    assert payload["checks"]["seed_outlier"]["rows"][0]["canonical_seed"] == 42
    assert "seed outlier" in report
    assert "P3" in report
    assert "S5" in report
