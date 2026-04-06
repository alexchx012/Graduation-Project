# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the manifest-driven ablation runner."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "run_morl_ablation.py"


def _load_module():
    module_name = "_run_morl_ablation_under_test"
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
    base = ROOT / ".tmp_test_run_morl_ablation_manifest"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "anchor_policy_id": "P10",
                "training_protocol": {
                    "task": "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v2",
                    "command_profile": "repair_forward_v2",
                    "num_envs": 4096,
                    "max_iterations": 900,
                    "clip_param": 0.2,
                    "curriculum_warmup": 300,
                    "curriculum_ramp": 300,
                    "training_seeds": [42, 43, 44],
                    "init_checkpoint": "logs/rsl_rl/unitree_go1_rough/2026-03-08_16-46-27_baseline_rough_ros2cmd/model_1499.pt",
                    "init_with_optimizer": True,
                },
                "entries": [
                    {
                        "ablation_id": "anchor-full",
                        "name": "morl_p10_anchor_full",
                        "policy_id": "P10",
                        "role": "anchor_full",
                        "morl_weights": [0.2, 0.2, 0.2, 0.4],
                    },
                    {
                        "ablation_id": "anchor-no-energy",
                        "name": "morl_p10_ablation_no_energy",
                        "policy_id": "P10-no-energy",
                        "role": "ablation_variant",
                        "morl_weights": [0.25, 0.0, 0.25, 0.5],
                    },
                    {
                        "ablation_id": "anchor-no-smooth",
                        "name": "morl_p10_ablation_no_smooth",
                        "policy_id": "P10-no-smooth",
                        "role": "ablation_variant",
                        "morl_weights": [0.25, 0.25, 0.0, 0.5],
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_manifest_defaults_select_only_ablation_variants(local_tmp_path):
    module = _load_module()

    manifest_path = local_tmp_path / "phase4_ablation_manifest.json"
    _write_manifest(manifest_path)

    manifest = module.load_phase4_ablation_manifest(manifest_path, project_root=ROOT)
    experiments, protocol = module.select_ablation_experiments(
        manifest,
        project_root=ROOT,
        entry_ids=None,
        include_anchor_full=False,
    )

    assert [exp["ablation_id"] for exp in experiments] == ["anchor-no-energy", "anchor-no-smooth"]
    assert [exp["policy_id"] for exp in experiments] == ["P10-no-energy", "P10-no-smooth"]
    assert protocol["training_seeds"] == [42, 43, 44]
    assert protocol["max_iterations"] == 900


def test_manifest_can_include_anchor_full_when_requested(local_tmp_path):
    module = _load_module()

    manifest_path = local_tmp_path / "phase4_ablation_manifest.json"
    _write_manifest(manifest_path)

    manifest = module.load_phase4_ablation_manifest(manifest_path, project_root=ROOT)
    experiments, _ = module.select_ablation_experiments(
        manifest,
        project_root=ROOT,
        entry_ids=None,
        include_anchor_full=True,
    )

    assert [exp["ablation_id"] for exp in experiments] == [
        "anchor-full",
        "anchor-no-energy",
        "anchor-no-smooth",
    ]

