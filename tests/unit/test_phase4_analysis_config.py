# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the default Phase 4 analysis config."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "scripts" / "phase_morl" / "manifests" / "phase4_analysis_config.json"


def test_default_phase4_analysis_config_has_required_sections():
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert data["normalization_bounds"]["J_speed"] == [0.0, 1.2]
    assert data["normalization_bounds"]["J_energy"] == [0.0, 2500.0]
    assert data["normalization_bounds"]["J_smooth"] == [0.0, 2.6]
    assert data["normalization_bounds"]["J_stable"] == [0.0, 0.7]
    assert data["ref_point"] == [1.1, 1.1, 1.1, 1.1]
    assert data["official_policy_set"] == ["P1", "P2", "P3", "P4", "P10"]
    assert data["exploratory_policy_set"] == ["P5", "P6", "P7", "P8", "P9"]
    assert data["baseline_policy_set"] == ["baseline"]
    assert data["frozen_on"] == "2026-04-05"
    assert data["verified_scenarios"] == ["S2", "S3", "S4", "S5", "S6"]
