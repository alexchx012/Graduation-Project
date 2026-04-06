# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Phase 4 figure rendering helpers."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "analyze_phase4_pareto.py"


def _load_module():
    module_name = "_analyze_phase4_pareto_render_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_render_phase4_figures_writes_hv_bar_and_scenario_scatter():
    module = _load_module()

    base = ROOT / ".tmp_test_phase4_render"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        figure_dir = path / "figures"
        scenario_payloads = {
            "S1": {
                "scenario_id": "S1",
                "hypervolume": 0.45,
                "policy_rows": [
                    {"policy": "P1", "J_speed": 0.1, "J_energy": 120.0, "J_smooth": 0.6, "J_stable": 0.6},
                    {"policy": "P2", "J_speed": 0.6, "J_energy": 50.0, "J_smooth": 0.6, "J_stable": 0.6},
                    {"policy": "P10", "J_speed": 0.3, "J_energy": 80.0, "J_smooth": 0.3, "J_stable": 0.3},
                ],
                "pareto_policies": ["P1", "P2", "P10"],
            },
            "S2": {
                "scenario_id": "S2",
                "hypervolume": 0.25,
                "policy_rows": [
                    {"policy": "P1", "J_speed": 0.1, "J_energy": 60.0, "J_smooth": 0.2, "J_stable": 0.2},
                    {"policy": "P2", "J_speed": 0.6, "J_energy": 60.0, "J_smooth": 0.6, "J_stable": 0.6},
                    {"policy": "P10", "J_speed": 0.3, "J_energy": 80.0, "J_smooth": 0.3, "J_stable": 0.3},
                ],
                "pareto_policies": ["P1"],
            },
        }
        bounds = {
            "J_speed": (0.0, 1.2),
            "J_energy": (0.0, 2500.0),
            "J_smooth": (0.0, 2.6),
            "J_stable": (0.0, 0.7),
        }

        outputs = module.render_phase4_figures(scenario_payloads, figure_dir=figure_dir, bounds=bounds)

        assert outputs["phase4_hv_bar_png"].exists()
        assert outputs["phase4_pareto_pngs"]["S1"].exists()
        assert outputs["phase4_pareto_pngs"]["S2"].exists()
    finally:
        shutil.rmtree(path, ignore_errors=True)
