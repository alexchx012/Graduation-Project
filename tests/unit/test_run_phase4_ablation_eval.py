# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Phase 4 ablation evaluation runner."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "run_phase4_ablation_eval.py"


def _load_module():
    module_name = "_run_phase4_ablation_eval_under_test"
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
    base = ROOT / ".tmp_test_run_phase4_ablation_eval"
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


def test_parser_accepts_include_anchor_full():
    module = _load_module()

    parser = module.build_parser()
    args = parser.parse_args(["--include-anchor-full"])

    assert args.include_anchor_full is True


def test_manifest_defaults_expand_only_ablation_variants(local_tmp_path):
    module = _load_module()

    manifest_path = local_tmp_path / "phase4_ablation_manifest.json"
    run_root = local_tmp_path / "runs"
    run_root.mkdir(parents=True)
    _write_manifest(manifest_path)

    for run_name in (
        "2026-04-06_21-05-30_morl_p10_ablation_no_energy_seed42",
        "2026-04-06_22-08-47_morl_p10_ablation_no_energy_seed43",
        "2026-04-07_00-10-24_morl_p10_ablation_no_smooth_seed42",
        "2026-04-07_01-12-52_morl_p10_ablation_no_smooth_seed43",
    ):
        run_dir = run_root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "model_899.pt").write_text("checkpoint", encoding="utf-8")

    targets = module.load_eval_targets_from_ablation_manifest(
        manifest_path,
        project_root=ROOT,
        run_root=run_root,
    )

    assert [(target.policy_id, target.seed) for target in targets] == [
        ("P10-no-energy", 42),
        ("P10-no-energy", 43),
        ("P10-no-smooth", 42),
        ("P10-no-smooth", 43),
    ]
    assert all(target.output_stem.startswith("morl_p10_ablation_") for target in targets)
    assert all(target.task == module.DEFAULT_EVAL_TASK for target in targets)


def test_manifest_can_include_anchor_full(local_tmp_path):
    module = _load_module()

    manifest_path = local_tmp_path / "phase4_ablation_manifest.json"
    run_root = local_tmp_path / "runs"
    run_root.mkdir(parents=True)
    _write_manifest(manifest_path)

    for run_name in (
        "2026-04-06_20-00-00_morl_p10_anchor_full_seed42",
        "2026-04-06_21-05-30_morl_p10_ablation_no_energy_seed42",
        "2026-04-07_00-10-24_morl_p10_ablation_no_smooth_seed42",
    ):
        run_dir = run_root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "model_899.pt").write_text("checkpoint", encoding="utf-8")

    targets = module.load_eval_targets_from_ablation_manifest(
        manifest_path,
        project_root=ROOT,
        run_root=run_root,
        seeds={42},
        include_anchor_full=True,
    )

    assert [target.policy_id for target in targets] == ["P10", "P10-no-energy", "P10-no-smooth"]


def test_validate_eval_targets_reports_missing_run_dir_and_checkpoint(local_tmp_path):
    module = _load_module()

    existing_run_dir = local_tmp_path / "2026-04-06_21-05-30_morl_p10_ablation_no_energy_seed42"
    existing_run_dir.mkdir(parents=True)
    (existing_run_dir / "model_899.pt").write_text("checkpoint", encoding="utf-8")

    targets = [
        module.AblationEvalTarget(
            run_dir_name=str(existing_run_dir),
            output_stem="morl_p10_ablation_no_energy_seed42",
            policy_id="P10-no-energy",
            seed=42,
            task=module.DEFAULT_EVAL_TASK,
            checkpoint="model_899.pt",
        ),
        module.AblationEvalTarget(
            run_dir_name=str(local_tmp_path / "missing_run_dir"),
            output_stem="morl_p10_ablation_no_energy_seed43",
            policy_id="P10-no-energy",
            seed=43,
            task=module.DEFAULT_EVAL_TASK,
            checkpoint="model_899.pt",
        ),
        module.AblationEvalTarget(
            run_dir_name=str(existing_run_dir),
            output_stem="morl_p10_ablation_no_smooth_seed42",
            policy_id="P10-no-smooth",
            seed=42,
            task=module.DEFAULT_EVAL_TASK,
            checkpoint="model_missing.pt",
        ),
    ]

    errors = module.validate_eval_targets(targets)

    assert len(errors) == 2
    assert any("Missing run directory" in err for err in errors)
    assert any("Missing checkpoint" in err for err in errors)
