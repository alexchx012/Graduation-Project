# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for eval checkpoint path resolution."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import uuid

import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "go1-ros2-test"
        / "checkpoint_utils.py"
    )
    spec = importlib.util.spec_from_file_location("go1_ros2_test_checkpoint_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_tmp_path():
    base = Path(__file__).resolve().parents[2] / ".tmp_test_eval_checkpoint_resolution"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_relative_checkpoint_is_resolved_against_absolute_load_run(local_tmp_path):
    module = _load_module()

    run_dir = local_tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_1499.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    resolved = module.resolve_eval_checkpoint_path(
        log_root_path=str(local_tmp_path / "logs"),
        load_run=str(run_dir),
        load_checkpoint="model_1499.pt",
        checkpoint_arg="model_1499.pt",
    )

    assert resolved == str(checkpoint.resolve())


def test_relative_checkpoint_is_resolved_against_log_root_and_run_name(local_tmp_path):
    module = _load_module()

    log_root = local_tmp_path / "logs" / "rsl_rl" / "unitree_go1_rough"
    run_dir = log_root / "2026-03-08_16-46-27_baseline_rough_ros2cmd"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model_1499.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    resolved = module.resolve_eval_checkpoint_path(
        log_root_path=str(log_root),
        load_run=run_dir.name,
        load_checkpoint="model_1499.pt",
        checkpoint_arg="model_1499.pt",
    )

    assert resolved == str(checkpoint.resolve())


def test_absolute_checkpoint_is_returned_without_load_run_lookup(local_tmp_path):
    module = _load_module()

    checkpoint = local_tmp_path / "model_1499.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    resolved = module.resolve_eval_checkpoint_path(
        log_root_path=str(local_tmp_path / "logs"),
        load_run="ignored_run_name",
        load_checkpoint="ignored.pt",
        checkpoint_arg=str(checkpoint),
    )

    assert resolved == str(checkpoint.resolve())
