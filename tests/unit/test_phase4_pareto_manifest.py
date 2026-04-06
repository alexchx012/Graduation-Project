# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Phase 4 scenario-level Pareto/HV analysis."""

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
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "analyze_phase4_pareto.py"


def _load_module():
    module_name = "_analyze_phase4_pareto_under_test"
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
    base = ROOT / ".tmp_test_phase4_pareto_manifest"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_generate_phase4_pareto_outputs_writes_non_anchor_statistics(local_tmp_path):
    module = _load_module()

    checkpoint_level_csv = local_tmp_path / "checkpoint_level.csv"
    policy_level_confirmatory_csv = local_tmp_path / "policy_level_confirmatory.csv"
    analysis_config_path = local_tmp_path / "phase4_analysis_config.json"
    output_dir = local_tmp_path / "pareto"
    figure_dir = local_tmp_path / "figures"

    analysis_config_path.write_text(
        json.dumps(
            {
                "normalization_bounds": {
                    "J_speed": [0.0, 1.2],
                    "J_energy": [0.0, 2500.0],
                    "J_smooth": [0.0, 2.6],
                    "J_stable": [0.0, 0.7],
                },
                "ref_point": [1.1, 1.1, 1.1, 1.1],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    policy_fieldnames = [
        "family",
        "policy_id",
        "scenario_id",
        "scenario_name",
        "terrain_mode",
        "disturbance_mode",
        "analysis_group",
        "task",
        "num_seeds",
        "seed_list",
        "output_stems",
        "source_states",
        "evidence_layer",
        "official_hv_eligible",
        "cmd_vx",
        "cmd_vx_std",
        "mean_cmd_vx",
        "mean_cmd_vx_std",
        "mean_cmd_vx_abs_diff",
        "mean_cmd_vx_abs_diff_std",
        "mean_vx_meas",
        "mean_vx_meas_std",
        "mean_vx_abs_err",
        "mean_vx_abs_err_std",
        "J_speed",
        "J_speed_std",
        "J_energy",
        "J_energy_std",
        "J_smooth",
        "J_smooth_std",
        "J_stable",
        "J_stable_std",
        "success_rate",
        "success_rate_std",
        "mean_base_contact_rate",
        "mean_base_contact_rate_std",
        "mean_timeout_rate",
        "mean_timeout_rate_std",
        "recovery_time",
        "recovery_time_std",
    ]
    policy_rows = [
        {
            "family": "morl",
            "policy_id": "P1",
            "scenario_id": "S1",
            "scenario_name": "flat_mid_speed",
            "terrain_mode": "plane",
            "disturbance_mode": "none",
            "analysis_group": "main",
            "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
            "num_seeds": "3",
            "seed_list": "42,43,44",
            "output_stems": "morl_p1_seed42,morl_p1_seed43,morl_p1_seed44",
            "source_states": "active,active,active",
            "evidence_layer": "A",
            "official_hv_eligible": "true",
            "cmd_vx": 1.0,
            "cmd_vx_std": 0.0,
            "mean_cmd_vx": 1.0,
            "mean_cmd_vx_std": 0.0,
            "mean_cmd_vx_abs_diff": 0.0,
            "mean_cmd_vx_abs_diff_std": 0.0,
            "mean_vx_meas": 0.95,
            "mean_vx_meas_std": 0.01,
            "mean_vx_abs_err": 0.05,
            "mean_vx_abs_err_std": 0.01,
            "J_speed": 0.1,
            "J_speed_std": 0.01,
            "J_energy": 120.0,
            "J_energy_std": 2.0,
            "J_smooth": 0.6,
            "J_smooth_std": 0.01,
            "J_stable": 0.6,
            "J_stable_std": 0.01,
            "success_rate": 1.0,
            "success_rate_std": 0.0,
            "mean_base_contact_rate": 0.0,
            "mean_base_contact_rate_std": 0.0,
            "mean_timeout_rate": 0.001,
            "mean_timeout_rate_std": 0.0,
            "recovery_time": 0.2,
            "recovery_time_std": 0.01,
        },
        {
            "family": "morl",
            "policy_id": "P2",
            "scenario_id": "S1",
            "scenario_name": "flat_mid_speed",
            "terrain_mode": "plane",
            "disturbance_mode": "none",
            "analysis_group": "main",
            "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
            "num_seeds": "3",
            "seed_list": "42,43,44",
            "output_stems": "morl_p2_seed42,morl_p2_seed43,morl_p2_seed44",
            "source_states": "active,active,active",
            "evidence_layer": "A",
            "official_hv_eligible": "true",
            "cmd_vx": 1.0,
            "cmd_vx_std": 0.0,
            "mean_cmd_vx": 1.0,
            "mean_cmd_vx_std": 0.0,
            "mean_cmd_vx_abs_diff": 0.0,
            "mean_cmd_vx_abs_diff_std": 0.0,
            "mean_vx_meas": 0.94,
            "mean_vx_meas_std": 0.01,
            "mean_vx_abs_err": 0.06,
            "mean_vx_abs_err_std": 0.01,
            "J_speed": 0.6,
            "J_speed_std": 0.01,
            "J_energy": 50.0,
            "J_energy_std": 2.0,
            "J_smooth": 0.6,
            "J_smooth_std": 0.01,
            "J_stable": 0.6,
            "J_stable_std": 0.01,
            "success_rate": 1.0,
            "success_rate_std": 0.0,
            "mean_base_contact_rate": 0.0,
            "mean_base_contact_rate_std": 0.0,
            "mean_timeout_rate": 0.001,
            "mean_timeout_rate_std": 0.0,
            "recovery_time": 0.2,
            "recovery_time_std": 0.01,
        },
        {
            "family": "morl",
            "policy_id": "P10",
            "scenario_id": "S1",
            "scenario_name": "flat_mid_speed",
            "terrain_mode": "plane",
            "disturbance_mode": "none",
            "analysis_group": "main",
            "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
            "num_seeds": "3",
            "seed_list": "42,43,44",
            "output_stems": "morl_p10_seed42,morl_p10_seed43,morl_p10_seed44",
            "source_states": "active,active,active",
            "evidence_layer": "A",
            "official_hv_eligible": "true",
            "cmd_vx": 1.0,
            "cmd_vx_std": 0.0,
            "mean_cmd_vx": 1.0,
            "mean_cmd_vx_std": 0.0,
            "mean_cmd_vx_abs_diff": 0.0,
            "mean_cmd_vx_abs_diff_std": 0.0,
            "mean_vx_meas": 0.945,
            "mean_vx_meas_std": 0.01,
            "mean_vx_abs_err": 0.055,
            "mean_vx_abs_err_std": 0.01,
            "J_speed": 0.3,
            "J_speed_std": 0.01,
            "J_energy": 80.0,
            "J_energy_std": 2.0,
            "J_smooth": 0.3,
            "J_smooth_std": 0.01,
            "J_stable": 0.3,
            "J_stable_std": 0.01,
            "success_rate": 1.0,
            "success_rate_std": 0.0,
            "mean_base_contact_rate": 0.0,
            "mean_base_contact_rate_std": 0.0,
            "mean_timeout_rate": 0.001,
            "mean_timeout_rate_std": 0.0,
            "recovery_time": 0.2,
            "recovery_time_std": 0.01,
        },
        {
            "family": "morl",
            "policy_id": "P1",
            "scenario_id": "S2",
            "scenario_name": "flat_high_speed",
            "terrain_mode": "plane",
            "disturbance_mode": "none",
            "analysis_group": "stress",
            "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
            "num_seeds": "3",
            "seed_list": "42,43,44",
            "output_stems": "morl_p1_seed42,morl_p1_seed43,morl_p1_seed44",
            "source_states": "active,active,active",
            "evidence_layer": "A",
            "official_hv_eligible": "true",
            "cmd_vx": 1.5,
            "cmd_vx_std": 0.0,
            "mean_cmd_vx": 1.5,
            "mean_cmd_vx_std": 0.0,
            "mean_cmd_vx_abs_diff": 0.0,
            "mean_cmd_vx_abs_diff_std": 0.0,
            "mean_vx_meas": 1.45,
            "mean_vx_meas_std": 0.01,
            "mean_vx_abs_err": 0.05,
            "mean_vx_abs_err_std": 0.01,
            "J_speed": 0.1,
            "J_speed_std": 0.01,
            "J_energy": 60.0,
            "J_energy_std": 2.0,
            "J_smooth": 0.2,
            "J_smooth_std": 0.01,
            "J_stable": 0.2,
            "J_stable_std": 0.01,
            "success_rate": 1.0,
            "success_rate_std": 0.0,
            "mean_base_contact_rate": 0.0,
            "mean_base_contact_rate_std": 0.0,
            "mean_timeout_rate": 0.001,
            "mean_timeout_rate_std": 0.0,
            "recovery_time": 0.2,
            "recovery_time_std": 0.01,
        },
        {
            "family": "morl",
            "policy_id": "P2",
            "scenario_id": "S2",
            "scenario_name": "flat_high_speed",
            "terrain_mode": "plane",
            "disturbance_mode": "none",
            "analysis_group": "stress",
            "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
            "num_seeds": "3",
            "seed_list": "42,43,44",
            "output_stems": "morl_p2_seed42,morl_p2_seed43,morl_p2_seed44",
            "source_states": "active,active,active",
            "evidence_layer": "A",
            "official_hv_eligible": "true",
            "cmd_vx": 1.5,
            "cmd_vx_std": 0.0,
            "mean_cmd_vx": 1.5,
            "mean_cmd_vx_std": 0.0,
            "mean_cmd_vx_abs_diff": 0.0,
            "mean_cmd_vx_abs_diff_std": 0.0,
            "mean_vx_meas": 1.0,
            "mean_vx_meas_std": 0.01,
            "mean_vx_abs_err": 0.5,
            "mean_vx_abs_err_std": 0.01,
            "J_speed": 0.6,
            "J_speed_std": 0.01,
            "J_energy": 60.0,
            "J_energy_std": 2.0,
            "J_smooth": 0.6,
            "J_smooth_std": 0.01,
            "J_stable": 0.6,
            "J_stable_std": 0.01,
            "success_rate": 1.0,
            "success_rate_std": 0.0,
            "mean_base_contact_rate": 0.0,
            "mean_base_contact_rate_std": 0.0,
            "mean_timeout_rate": 0.001,
            "mean_timeout_rate_std": 0.0,
            "recovery_time": 0.2,
            "recovery_time_std": 0.01,
        },
        {
            "family": "morl",
            "policy_id": "P10",
            "scenario_id": "S2",
            "scenario_name": "flat_high_speed",
            "terrain_mode": "plane",
            "disturbance_mode": "none",
            "analysis_group": "stress",
            "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
            "num_seeds": "3",
            "seed_list": "42,43,44",
            "output_stems": "morl_p10_seed42,morl_p10_seed43,morl_p10_seed44",
            "source_states": "active,active,active",
            "evidence_layer": "A",
            "official_hv_eligible": "true",
            "cmd_vx": 1.5,
            "cmd_vx_std": 0.0,
            "mean_cmd_vx": 1.5,
            "mean_cmd_vx_std": 0.0,
            "mean_cmd_vx_abs_diff": 0.0,
            "mean_cmd_vx_abs_diff_std": 0.0,
            "mean_vx_meas": 1.2,
            "mean_vx_meas_std": 0.01,
            "mean_vx_abs_err": 0.3,
            "mean_vx_abs_err_std": 0.01,
            "J_speed": 0.3,
            "J_speed_std": 0.01,
            "J_energy": 80.0,
            "J_energy_std": 2.0,
            "J_smooth": 0.3,
            "J_smooth_std": 0.01,
            "J_stable": 0.3,
            "J_stable_std": 0.01,
            "success_rate": 1.0,
            "success_rate_std": 0.0,
            "mean_base_contact_rate": 0.0,
            "mean_base_contact_rate_std": 0.0,
            "mean_timeout_rate": 0.001,
            "mean_timeout_rate_std": 0.0,
            "recovery_time": 0.2,
            "recovery_time_std": 0.01,
        },
    ]
    _write_csv(policy_level_confirmatory_csv, policy_fieldnames, policy_rows)

    checkpoint_fieldnames = [
        "family",
        "policy_id",
        "canonical_seed",
        "output_stem",
        "source_state",
        "evidence_layer",
        "official_hv_eligible",
        "run_dir",
        "checkpoint",
        "summary_path",
        "scenario_id",
        "scenario_name",
        "terrain_mode",
        "disturbance_mode",
        "analysis_group",
        "task",
        "cmd_vx",
        "mean_cmd_vx",
        "mean_cmd_vx_abs_diff",
        "mean_vx_meas",
        "mean_vx_abs_err",
        "J_speed",
        "J_energy",
        "J_smooth",
        "J_stable",
        "success_rate",
        "mean_base_contact_rate",
        "mean_timeout_rate",
        "recovery_time",
        "eval_steps",
        "warmup_steps",
        "effective_steps",
        "effective_env_steps",
        "step_dt",
        "elapsed_seconds",
        "is_degenerate_archived_control",
    ]

    checkpoint_rows = []
    for policy_id, base_metrics in {
        ("P1", "S1"): (0.1, 120.0, 0.6, 0.6, 0.95, 0.05),
        ("P2", "S1"): (0.6, 50.0, 0.6, 0.6, 0.94, 0.06),
        ("P10", "S1"): (0.3, 80.0, 0.3, 0.3, 0.945, 0.055),
        ("P1", "S2"): (0.1, 60.0, 0.2, 0.2, 1.45, 0.05),
        ("P2", "S2"): (0.6, 60.0, 0.6, 0.6, 1.0, 0.5),
        ("P10", "S2"): (0.3, 80.0, 0.3, 0.3, 1.2, 0.3),
    }.items():
        policy, scenario = policy_id
        j_speed, j_energy, j_smooth, j_stable, mean_vx_meas, mean_vx_abs_err = base_metrics
        for seed in (42, 43, 44):
            checkpoint_rows.append(
                {
                    "family": "morl",
                    "policy_id": policy,
                    "canonical_seed": seed,
                    "output_stem": f"morl_{policy.lower()}_seed{seed}",
                    "source_state": "active",
                    "evidence_layer": "A",
                    "official_hv_eligible": "true",
                    "run_dir": f"D:/runs/morl_{policy.lower()}_seed{seed}",
                    "checkpoint": "model_899.pt",
                    "summary_path": f"D:/eval/morl_{policy.lower()}_seed{seed}_{scenario}.json",
                    "scenario_id": scenario,
                    "scenario_name": "flat_mid_speed" if scenario == "S1" else "flat_high_speed",
                    "terrain_mode": "plane",
                    "disturbance_mode": "none",
                    "analysis_group": "main" if scenario == "S1" else "stress",
                    "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
                    "cmd_vx": 1.0 if scenario == "S1" else 1.5,
                    "mean_cmd_vx": 1.0 if scenario == "S1" else 1.5,
                    "mean_cmd_vx_abs_diff": 0.0,
                    "mean_vx_meas": mean_vx_meas,
                    "mean_vx_abs_err": mean_vx_abs_err,
                    "J_speed": j_speed,
                    "J_energy": j_energy,
                    "J_smooth": j_smooth,
                    "J_stable": j_stable,
                    "success_rate": 1.0,
                    "mean_base_contact_rate": 0.0,
                    "mean_timeout_rate": 0.001,
                    "recovery_time": 0.2,
                    "eval_steps": 3000,
                    "warmup_steps": 300,
                    "effective_steps": 2700,
                    "effective_env_steps": 172800,
                    "step_dt": 0.02,
                    "elapsed_seconds": 10.0,
                    "is_degenerate_archived_control": "false",
                }
            )
    _write_csv(checkpoint_level_csv, checkpoint_fieldnames, checkpoint_rows)

    outputs = module.generate_phase4_pareto_outputs(
        checkpoint_level_csv=checkpoint_level_csv,
        policy_level_confirmatory_csv=policy_level_confirmatory_csv,
        analysis_config_path=analysis_config_path,
        output_dir=output_dir,
        figure_dir=figure_dir,
        num_bootstrap=64,
        random_seed=123,
        dry_run=False,
    )

    hv_payload = json.loads(outputs["confirmatory_scenario_hv_json"].read_text(encoding="utf-8"))
    hv_ci_payload = json.loads(outputs["bootstrap_hv_ci_json"].read_text(encoding="utf-8"))
    freq_rows = list(csv.DictReader(outputs["front_membership_frequency_csv"].open(encoding="utf-8", newline="")))
    bootstrap_rows = list(csv.DictReader(outputs["bootstrap_front_membership_csv"].open(encoding="utf-8", newline="")))
    robustness_rows = list(csv.DictReader(outputs["robustness_summary_csv"].open(encoding="utf-8", newline="")))

    assert sorted(hv_payload.keys()) == ["S1", "S2"]
    assert hv_payload["S1"]["pareto_policies"] == ["P1", "P2", "P10"]
    assert hv_payload["S2"]["pareto_policies"] == ["P1"]
    assert hv_ci_payload["S1"]["num_bootstrap"] == 64
    assert hv_ci_payload["S2"]["num_bootstrap"] == 64

    freq = {row["policy_id"]: row for row in freq_rows}
    assert freq["P1"]["front_count"] == "2"
    assert freq["P2"]["front_count"] == "1"
    assert freq["P10"]["front_count"] == "1"

    bootstrap = {(row["scenario_id"], row["policy_id"]): row for row in bootstrap_rows}
    assert float(bootstrap[("S2", "P1")]["p_on_front"]) == pytest.approx(1.0)
    assert float(bootstrap[("S2", "P2")]["p_on_front"]) == pytest.approx(0.0)
    assert float(bootstrap[("S2", "P10")]["p_on_front"]) == pytest.approx(0.0)

    robust = {row["policy_id"]: row for row in robustness_rows}
    assert robust["P1"]["front_count"] == "2"
    assert robust["P2"]["front_count"] == "1"
    assert robust["P10"]["front_count"] == "1"

    assert outputs["phase4_hv_bar_png"].exists()
    assert outputs["phase4_pareto_pngs"]["S1"].exists()
    assert outputs["phase4_pareto_pngs"]["S2"].exists()
