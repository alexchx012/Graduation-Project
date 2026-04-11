# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Phase 4 ablation postprocess runner."""

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
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "run_phase4_ablation_postprocess.py"


def _load_module():
    module_name = "_run_phase4_ablation_postprocess_under_test"
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
    base = ROOT / ".tmp_test_run_phase4_ablation_postprocess"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_main_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "family": "morl",
                        "policy_id": "P10",
                        "canonical_seed": 42,
                        "run_dir": "D:/runs/2026-04-02_23-24-36_morl_p10_seed42",
                        "checkpoint": "model_899.pt",
                        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                        "output_stem": "morl_p10_seed42",
                        "evidence_layer": "A",
                        "official_hv_eligible": True,
                        "source_state": "active",
                    },
                    {
                        "family": "morl",
                        "policy_id": "P10",
                        "canonical_seed": 43,
                        "run_dir": "D:/runs/2026-04-03_00-26-57_morl_p10_seed43",
                        "checkpoint": "model_899.pt",
                        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                        "output_stem": "morl_p10_seed43",
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


def _write_ablation_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "anchor_policy_id": "P10",
                "training_protocol": {
                    "training_seeds": [42, 43],
                },
                "entries": [
                    {
                        "ablation_id": "anchor-full",
                        "name": "morl_p10_anchor_full",
                        "policy_id": "P10",
                        "role": "anchor_full",
                    },
                    {
                        "ablation_id": "anchor-no-energy",
                        "name": "morl_p10_ablation_no_energy",
                        "policy_id": "P10-no-energy",
                        "role": "ablation_variant",
                    },
                    {
                        "ablation_id": "anchor-no-smooth",
                        "name": "morl_p10_ablation_no_smooth",
                        "policy_id": "P10-no-smooth",
                        "role": "ablation_variant",
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _eval_payload(*, policy_id: str, scenario_id: str, mean_vx_meas: float, mean_vx_abs_err: float, j_speed: float, j_energy: float, j_smooth: float, j_stable: float) -> dict:
    return {
        "policy_id": policy_id,
        "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
        "scenario_id": scenario_id,
        "scenario_name": "flat_mid_speed",
        "terrain_mode": "plane",
        "cmd_vx": 1.0,
        "disturbance_mode": "none",
        "analysis_group": "main",
        "checkpoint": f"D:/fake/{policy_id}/model.pt",
        "load_run": f"D:/fake/{policy_id}",
        "eval_steps": 3000,
        "warmup_steps": 300,
        "effective_steps": 2700,
        "effective_env_steps": 172800,
        "step_dt": 0.02,
        "mean_cmd_vx": 1.0,
        "mean_vx_meas": mean_vx_meas,
        "mean_vx_abs_err": mean_vx_abs_err,
        "J_speed": j_speed,
        "J_energy": j_energy,
        "J_smooth": j_smooth,
        "J_stable": j_stable,
        "success_rate": 1.0,
        "mean_base_contact_rate": 0.0,
        "mean_timeout_rate": 0.001,
        "recovery_time": None,
        "elapsed_seconds": 10.0,
    }


def test_build_resolved_entries_combines_anchor_and_ablations(local_tmp_path):
    module = _load_module()

    main_manifest_path = local_tmp_path / "phase4_main_manifest.json"
    ablation_manifest_path = local_tmp_path / "phase4_ablation_manifest.json"
    run_root = local_tmp_path / "runs"
    run_root.mkdir(parents=True)
    _write_main_manifest(main_manifest_path)
    _write_ablation_manifest(ablation_manifest_path)

    for run_name in (
        "2026-04-06_21-05-30_morl_p10_ablation_no_energy_seed42",
        "2026-04-06_22-08-47_morl_p10_ablation_no_energy_seed43",
        "2026-04-07_00-10-24_morl_p10_ablation_no_smooth_seed42",
        "2026-04-07_01-12-52_morl_p10_ablation_no_smooth_seed43",
    ):
        (run_root / run_name).mkdir(parents=True)

    entries, metadata = module.build_resolved_ablation_entries(
        main_manifest_path=main_manifest_path,
        ablation_manifest_path=ablation_manifest_path,
        run_root=run_root,
        include_anchor_full=True,
    )

    assert [entry.policy_id for entry in entries] == [
        "P10",
        "P10",
        "P10-no-energy",
        "P10-no-energy",
        "P10-no-smooth",
        "P10-no-smooth",
    ]
    assert {entry.family for entry in entries} == {"ablation"}
    assert metadata["anchor_policy_id"] == "P10"
    assert metadata["variant_policy_ids"] == ["P10-no-energy", "P10-no-smooth"]


def test_generate_ablation_postprocess_outputs_writes_comparison_and_qc(local_tmp_path):
    module = _load_module()

    summary_dir = local_tmp_path / "eval"
    output_dir = local_tmp_path / "ablation"
    run_root = local_tmp_path / "runs"
    run_root.mkdir(parents=True)
    main_manifest_path = local_tmp_path / "phase4_main_manifest.json"
    ablation_manifest_path = local_tmp_path / "phase4_ablation_manifest.json"
    _write_main_manifest(main_manifest_path)
    _write_ablation_manifest(ablation_manifest_path)

    for run_name in (
        "2026-04-06_21-05-30_morl_p10_ablation_no_energy_seed42",
        "2026-04-06_22-08-47_morl_p10_ablation_no_energy_seed43",
        "2026-04-07_00-10-24_morl_p10_ablation_no_smooth_seed42",
        "2026-04-07_01-12-52_morl_p10_ablation_no_smooth_seed43",
    ):
        (run_root / run_name).mkdir(parents=True)

    _write_json(summary_dir / "morl_p10_seed42_S1.json", _eval_payload(policy_id="morl_p10_seed42", scenario_id="S1", mean_vx_meas=0.96, mean_vx_abs_err=0.04, j_speed=0.10, j_energy=100.0, j_smooth=1.00, j_stable=0.20))
    _write_json(summary_dir / "morl_p10_seed43_S1.json", _eval_payload(policy_id="morl_p10_seed43", scenario_id="S1", mean_vx_meas=0.98, mean_vx_abs_err=0.03, j_speed=0.11, j_energy=110.0, j_smooth=1.10, j_stable=0.22))
    _write_json(summary_dir / "morl_p10_ablation_no_energy_seed42_S1.json", _eval_payload(policy_id="morl_p10_ablation_no_energy_seed42", scenario_id="S1", mean_vx_meas=0.95, mean_vx_abs_err=0.05, j_speed=0.12, j_energy=150.0, j_smooth=1.20, j_stable=0.25))
    _write_json(summary_dir / "morl_p10_ablation_no_energy_seed43_S1.json", _eval_payload(policy_id="morl_p10_ablation_no_energy_seed43", scenario_id="S1", mean_vx_meas=0.94, mean_vx_abs_err=0.06, j_speed=0.13, j_energy=160.0, j_smooth=1.25, j_stable=0.28))
    _write_json(summary_dir / "morl_p10_ablation_no_smooth_seed42_S1.json", _eval_payload(policy_id="morl_p10_ablation_no_smooth_seed42", scenario_id="S1", mean_vx_meas=0.93, mean_vx_abs_err=0.07, j_speed=0.15, j_energy=120.0, j_smooth=1.50, j_stable=0.30))
    _write_json(summary_dir / "morl_p10_ablation_no_smooth_seed43_S1.json", _eval_payload(policy_id="morl_p10_ablation_no_smooth_seed43", scenario_id="S1", mean_vx_meas=0.92, mean_vx_abs_err=0.08, j_speed=0.16, j_energy=125.0, j_smooth=1.55, j_stable=0.33))

    outputs = module.generate_phase4_ablation_postprocess_outputs(
        summary_dir=summary_dir,
        main_manifest_path=main_manifest_path,
        ablation_manifest_path=ablation_manifest_path,
        run_root=run_root,
        output_dir=output_dir,
        include_anchor_full=True,
        dry_run=False,
    )

    policy_rows = list(csv.DictReader(outputs["policy_level_ablation_csv"].read_text(encoding="utf-8").splitlines()))
    comparison_rows = list(csv.DictReader(outputs["ablation_comparison_csv"].read_text(encoding="utf-8").splitlines()))
    qc_report = outputs["qc_report_md"].read_text(encoding="utf-8")
    resolved_manifest = json.loads(outputs["resolved_manifest_json"].read_text(encoding="utf-8"))

    assert len(resolved_manifest["entries"]) == 6
    assert len(policy_rows) == 3
    assert len(comparison_rows) == 2

    no_energy_row = next(row for row in comparison_rows if row["ablation_policy_id"] == "P10-no-energy")
    assert no_energy_row["scenario_id"] == "S1"
    assert float(no_energy_row["anchor_J_energy"]) == pytest.approx(105.0)
    assert float(no_energy_row["ablation_J_energy"]) == pytest.approx(155.0)
    assert float(no_energy_row["delta_J_energy"]) == pytest.approx(50.0)

    assert "pair coverage" in qc_report
    assert "PASS" in qc_report
