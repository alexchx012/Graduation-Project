# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MORL Pareto analysis script."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "analyze_pareto.py"


def _load_module():
    module_name = "_analyze_pareto_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_objective_rows_uses_frozen_bounds_and_clips_values():
    module = _load_module()

    rows = [
        {"policy": "P1", "J_speed": 0.6, "J_energy": 1250.0, "J_smooth": 1.0, "J_stable": 0.25},
        {"policy": "P2", "J_speed": 1.5, "J_energy": 3000.0, "J_smooth": -0.2, "J_stable": 0.6},
    ]

    normalized = module.normalize_objective_rows(rows, bounds=module.FROZEN_NORMALIZATION_BOUNDS)

    assert normalized[0]["normalized_objectives"] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert normalized[1]["normalized_objectives"] == pytest.approx([1.0, 1.0, 0.0, 1.0])


def test_compute_pareto_front_mask_treats_objectives_as_minimization():
    module = _load_module()

    objectives = [
        [0.2, 0.5, 0.5, 0.5],
        [0.3, 0.6, 0.6, 0.6],
        [0.2, 0.6, 0.4, 0.6],
        [0.7, 0.1, 0.1, 0.1],
    ]

    mask = module.compute_pareto_front_mask(objectives)

    assert mask == [True, False, True, True]


def test_compute_hypervolume_returns_exact_union_volume_for_toy_case():
    module = _load_module()

    hv = module.compute_hypervolume(
        points=[
            [0.2, 0.8],
            [0.5, 0.3],
        ],
        ref_point=[1.0, 1.0],
    )

    assert hv == pytest.approx(0.41)


def test_aggregate_policy_rows_computes_mean_std_and_seed_count():
    module = _load_module()

    run_rows = [
        {"policy": "P1", "seed": 42, "J_speed": 0.8, "J_energy": 700.0, "J_smooth": 0.3, "J_stable": 0.2, "success_rate": 0.95},
        {"policy": "P1", "seed": 43, "J_speed": 1.0, "J_energy": 900.0, "J_smooth": 0.5, "J_stable": 0.4, "success_rate": 0.85},
        {"policy": "P5", "seed": 42, "J_speed": 0.7, "J_energy": 500.0, "J_smooth": 0.2, "J_stable": 0.1, "success_rate": 1.0},
    ]

    policy_rows = module.aggregate_policy_rows(run_rows)

    assert [row["policy"] for row in policy_rows] == ["P1", "P5"]

    p1 = policy_rows[0]
    assert p1["num_seeds"] == 2
    assert p1["seeds"] == [42, 43]
    assert p1["J_speed"] == pytest.approx(0.9)
    assert p1["J_speed_std"] == pytest.approx(0.1)
    assert p1["success_rate"] == pytest.approx(0.9)

    p5 = policy_rows[1]
    assert p5["num_seeds"] == 1
    assert p5["J_energy_std"] == pytest.approx(0.0)


def test_build_analysis_payload_includes_policy_and_run_level_sections():
    module = _load_module()

    run_rows = [
        {"run": "run_p1_s42", "policy": "P1", "seed": 42, "policy_id": "morl_p1_seed42", "J_speed": 0.8, "J_energy": 700.0, "J_smooth": 0.3, "J_stable": 0.2, "success_rate": 0.95},
        {"run": "run_p1_s43", "policy": "P1", "seed": 43, "policy_id": "morl_p1_seed43", "J_speed": 1.0, "J_energy": 900.0, "J_smooth": 0.5, "J_stable": 0.4, "success_rate": 0.85},
        {"run": "run_p5_s42", "policy": "P5", "seed": 42, "policy_id": "morl_p5_seed42", "J_speed": 0.7, "J_energy": 500.0, "J_smooth": 0.2, "J_stable": 0.1, "success_rate": 1.0},
    ]

    payload = module.build_analysis_payload(run_rows)

    assert set(payload) >= {
        "generated_from_runs",
        "objective_keys",
        "normalization_bounds",
        "ref_point",
        "run_level",
        "policy_level",
        "pareto_front",
        "hypervolume",
    }
    assert payload["objective_keys"] == ["J_speed", "J_energy", "J_smooth", "J_stable"]
    assert payload["generated_from_runs"] == 3
    assert payload["generated_from_policies"] == 2
    assert payload["pareto_front"]["policy_names"] == ["P5"]
    assert math.isfinite(payload["hypervolume"])
