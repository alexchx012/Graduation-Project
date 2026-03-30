# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for training checkpoint resolution helpers."""

from __future__ import annotations

import importlib.util
import shutil
import uuid
from pathlib import Path

import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "go1-ros2-test"
        / "checkpoint_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "go1_ros2_test_checkpoint_utils_training",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_tmp_path():
    base = Path(__file__).resolve().parents[2] / ".tmp_test_checkpoint_utils"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_resolve_training_checkpoint_path_returns_absolute_checkpoint():
    module = _load_module()
    local_tmp_path = Path(__file__).resolve().parents[2] / ".tmp_test_checkpoint_utils_abs"
    local_tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = local_tmp_path / "model_1499.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    try:
        resolved = module.resolve_training_checkpoint_path(
            log_root_path=str(local_tmp_path / "logs"),
            load_run="ignored_run_name",
            checkpoint_arg=str(checkpoint),
        )
        assert resolved == str(checkpoint.resolve())
    finally:
        shutil.rmtree(local_tmp_path, ignore_errors=True)


def test_resolve_training_checkpoint_path_matches_run_dir_checkpoint(local_tmp_path):
    module = _load_module()

    log_root = local_tmp_path / "logs" / "rsl_rl" / "unitree_go1_rough"
    run_dir = log_root / "2026-03-08_16-46-27_baseline_rough_ros2cmd"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model_1499.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    resolved = module.resolve_training_checkpoint_path(
        log_root_path=str(log_root),
        load_run=run_dir.name,
        checkpoint_arg="model_1499.pt",
    )

    assert resolved == str(checkpoint.resolve())
