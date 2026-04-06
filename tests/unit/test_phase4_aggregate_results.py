# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Phase 4 aggregation outputs."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "aggregate_phase4_results.py"


def _load_module():
    module_name = "_aggregate_phase4_results_under_test"
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
    base = ROOT / ".tmp_test_phase4_aggregate_results"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_eval_payload(*, policy_id: str, scenario_id: str, scenario_name: str, cmd_vx: float, mean_cmd_vx: float, mean_vx_meas: float, mean_vx_abs_err: float, j_speed: float, j_energy: float, j_smooth: float, j_stable: float, success_rate: float, terrain_mode: str = "plane", disturbance_mode: str = "none", analysis_group: str = "main", recovery_time: float | None = None) -> dict:
    return {
        "policy_id": policy_id,
        "task": "Isaac-Velocity-Rough-Unitree-Go1-ROS2Cmd-Play-v0" if policy_id.startswith("baseline") else "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "terrain_mode": terrain_mode,
        "cmd_vx": cmd_vx,
        "disturbance_mode": disturbance_mode,
        "analysis_group": analysis_group,
        "checkpoint": f"D:/fake/{policy_id}/model.pt",
        "load_run": f"D:/fake/{policy_id}",
        "eval_steps": 3000,
        "warmup_steps": 300,
        "effective_steps": 2700,
        "effective_env_steps": 172800,
        "step_dt": 0.02,
        "mean_cmd_vx": mean_cmd_vx,
        "mean_vx_meas": mean_vx_meas,
        "mean_vx_abs_err": mean_vx_abs_err,
        "J_speed": j_speed,
        "J_energy": j_energy,
        "J_smooth": j_smooth,
        "J_stable": j_stable,
        "success_rate": success_rate,
        "mean_base_contact_rate": 0.0,
        "mean_timeout_rate": 0.001,
        "recovery_time": recovery_time,
        "elapsed_seconds": 10.0,
    }


def test_generate_phase4_outputs_writes_split_csvs_and_baseline_annotations(local_tmp_path):
    module = _load_module()

    summary_dir = local_tmp_path / "eval"
    output_dir = local_tmp_path / "aggregated"
    manifest_path = local_tmp_path / "phase4_main_manifest.json"
    analysis_config_path = local_tmp_path / "phase4_analysis_config.json"

    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "family": "morl",
                        "policy_id": "P1",
                        "canonical_seed": 42,
                        "run_dir": "D:/runs/morl_p1_seed42",
                        "checkpoint": "model_899.pt",
                        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                        "output_stem": "morl_p1_seed42",
                        "evidence_layer": "A",
                        "official_hv_eligible": True,
                        "source_state": "active",
                    },
                    {
                        "family": "morl",
                        "policy_id": "P1",
                        "canonical_seed": 43,
                        "run_dir": "D:/runs/morl_p1_seed43",
                        "checkpoint": "model_899.pt",
                        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                        "output_stem": "morl_p1_seed43",
                        "evidence_layer": "A",
                        "official_hv_eligible": True,
                        "source_state": "active",
                    },
                    {
                        "family": "morl",
                        "policy_id": "P5",
                        "canonical_seed": 42,
                        "run_dir": "D:/runs/morl_p5_seed42",
                        "checkpoint": "model_899.pt",
                        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                        "output_stem": "morl_p5_seed42",
                        "evidence_layer": "B",
                        "official_hv_eligible": False,
                        "source_state": "active",
                    },
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

    _write_json(
        summary_dir / "morl_p1_seed42_S1.json",
        _base_eval_payload(
            policy_id="morl_p1_seed42",
            scenario_id="S1",
            scenario_name="flat_mid_speed",
            cmd_vx=1.0,
            mean_cmd_vx=1.0,
            mean_vx_meas=0.98,
            mean_vx_abs_err=0.04,
            j_speed=0.09,
            j_energy=85.0,
            j_smooth=1.3,
            j_stable=0.31,
            success_rate=1.0,
        ),
    )
    _write_json(
        summary_dir / "morl_p1_seed43_S1.json",
        _base_eval_payload(
            policy_id="morl_p1_seed43",
            scenario_id="S1",
            scenario_name="flat_mid_speed",
            cmd_vx=1.0,
            mean_cmd_vx=1.0,
            mean_vx_meas=0.95,
            mean_vx_abs_err=0.05,
            j_speed=0.10,
            j_energy=87.0,
            j_smooth=1.4,
            j_stable=0.33,
            success_rate=1.0,
        ),
    )
    _write_json(
        summary_dir / "morl_p5_seed42_S1.json",
        _base_eval_payload(
            policy_id="morl_p5_seed42",
            scenario_id="S1",
            scenario_name="flat_mid_speed",
            cmd_vx=1.0,
            mean_cmd_vx=1.0,
            mean_vx_meas=0.97,
            mean_vx_abs_err=0.06,
            j_speed=0.12,
            j_energy=83.0,
            j_smooth=1.2,
            j_stable=0.34,
            success_rate=1.0,
        ),
    )
    _write_json(
        summary_dir / "baseline_seed42_S1.json",
        _base_eval_payload(
            policy_id="baseline_seed42",
            scenario_id="S1",
            scenario_name="flat_mid_speed",
            cmd_vx=1.0,
            mean_cmd_vx=1.0,
            mean_vx_meas=0.96,
            mean_vx_abs_err=0.05,
            j_speed=0.10,
            j_energy=90.0,
            j_smooth=1.30,
            j_stable=0.35,
            success_rate=1.0,
        ),
    )
    _write_json(
        summary_dir / "baseline_seed43_S1.json",
        _base_eval_payload(
            policy_id="baseline_seed43",
            scenario_id="S1",
            scenario_name="flat_mid_speed",
            cmd_vx=1.0,
            mean_cmd_vx=1.0,
            mean_vx_meas=0.002,
            mean_vx_abs_err=0.998,
            j_speed=0.999,
            j_energy=276.0,
            j_smooth=0.03,
            j_stable=0.36,
            success_rate=1.0,
        ),
    )
    _write_json(
        summary_dir / "baseline_seed44_S1.json",
        _base_eval_payload(
            policy_id="baseline_seed44",
            scenario_id="S1",
            scenario_name="flat_mid_speed",
            cmd_vx=1.0,
            mean_cmd_vx=1.0,
            mean_vx_meas=0.97,
            mean_vx_abs_err=0.04,
            j_speed=0.11,
            j_energy=93.0,
            j_smooth=1.29,
            j_stable=0.31,
            success_rate=1.0,
        ),
    )

    outputs = module.generate_phase4_outputs(
        summary_dir=summary_dir,
        manifest_path=manifest_path,
        analysis_config_path=analysis_config_path,
        output_dir=output_dir,
        dry_run=False,
    )

    checkpoint_rows = list(csv.DictReader(outputs["checkpoint_level_csv"].read_text(encoding="utf-8").splitlines()))
    confirmatory_rows = list(csv.DictReader(outputs["policy_level_confirmatory_csv"].read_text(encoding="utf-8").splitlines()))
    exploratory_rows = list(csv.DictReader(outputs["policy_level_exploratory_csv"].read_text(encoding="utf-8").splitlines()))
    baseline_rows = list(csv.DictReader(outputs["baseline_control_csv"].read_text(encoding="utf-8").splitlines()))

    assert len(checkpoint_rows) == 6
    assert len(confirmatory_rows) == 1
    assert len(exploratory_rows) == 1
    assert len(baseline_rows) == 1

    confirmatory = confirmatory_rows[0]
    assert confirmatory["policy_id"] == "P1"
    assert confirmatory["scenario_id"] == "S1"
    assert confirmatory["num_seeds"] == "2"
    assert confirmatory["seed_list"] == "42,43"
    assert float(confirmatory["J_speed"]) == pytest.approx(0.095)

    exploratory = exploratory_rows[0]
    assert exploratory["policy_id"] == "P5"
    assert exploratory["scenario_id"] == "S1"
    assert exploratory["num_seeds"] == "1"

    baseline = baseline_rows[0]
    assert baseline["policy_id"] == "baseline"
    assert baseline["scenario_id"] == "S1"
    assert baseline["num_seeds"] == "3"
    assert baseline["has_degenerate_seed"] == "true"
    assert baseline["degenerate_seed_ids"] == "43"
    assert baseline["narrative_effective_seed_ids"] == "42,44"
    assert "degenerate archived control" in baseline["paper_note"]


def test_generate_phase4_outputs_dry_run_returns_planned_paths_without_writing(local_tmp_path):
    module = _load_module()

    summary_dir = local_tmp_path / "eval"
    output_dir = local_tmp_path / "aggregated"
    manifest_path = local_tmp_path / "phase4_main_manifest.json"
    analysis_config_path = local_tmp_path / "phase4_analysis_config.json"

    manifest_path.write_text('{"entries": []}', encoding="utf-8")
    analysis_config_path.write_text(
        '{"official_policy_set": ["P1"], "exploratory_policy_set": [], "baseline_policy_set": ["baseline"]}',
        encoding="utf-8",
    )

    outputs = module.generate_phase4_outputs(
        summary_dir=summary_dir,
        manifest_path=manifest_path,
        analysis_config_path=analysis_config_path,
        output_dir=output_dir,
        dry_run=True,
    )

    assert outputs["checkpoint_level_csv"] == output_dir / "checkpoint_level.csv"
    assert outputs["policy_level_confirmatory_csv"] == output_dir / "policy_level_confirmatory.csv"
    assert outputs["policy_level_exploratory_csv"] == output_dir / "policy_level_exploratory.csv"
    assert outputs["baseline_control_csv"] == output_dir / "baseline_control.csv"
    assert outputs["qc_report_md"] == output_dir / "qc_report.md"
    assert not output_dir.exists()
