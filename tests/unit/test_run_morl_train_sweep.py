# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MORL batch training sweep runner."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "phase_morl" / "run_morl_train_sweep.py"


def _load_module():
    module_name = "_run_morl_train_sweep_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_default_experiment_matrix_matches_phase_plan():
    module = _load_module()

    assert len(module.MORL_EXPERIMENTS) == 10
    assert module.SEEDS == [42]
    assert module.NUM_ENVS == 4096
    assert module.MAX_ITERATIONS == 1500
    assert module.DEFAULT_CLIP_PARAM == 0.3

    expected = {
        "P1": "0.7,0.1,0.1,0.1",
        "P2": "0.1,0.7,0.1,0.1",
        "P3": "0.1,0.1,0.7,0.1",
        "P4": "0.1,0.1,0.1,0.7",
        "P5": "0.4,0.3,0.2,0.1",
        "P6": "0.5,0.3,0.1,0.1",
        "P7": "0.3,0.3,0.2,0.2",
        "P8": "0.2,0.4,0.2,0.2",
        "P9": "0.3,0.2,0.3,0.2",
        "P10": "0.2,0.2,0.2,0.4",
    }

    actual = {exp["policy_id"]: exp["morl_weights"] for exp in module.MORL_EXPERIMENTS}
    assert actual == expected

    for exp in module.MORL_EXPERIMENTS:
        weights = [float(w) for w in exp["morl_weights"].split(",")]
        assert pytest.approx(sum(weights), rel=1e-9, abs=1e-9) == 1.0


def test_build_train_cmd_includes_morl_specific_flags():
    module = _load_module()
    exp = module.MORL_EXPERIMENTS[0]
    run_name = "morl_p1_seed42"

    cmd = module._build_train_cmd(ROOT, exp, seed=42, run_name=run_name)

    joined = " ".join(cmd)
    assert str(ROOT / "scripts" / "go1-ros2-test" / "train.py") in joined
    assert "--task Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v0" in joined
    assert "--num_envs 4096" in joined
    assert "--max_iterations 1500" in joined
    assert "--seed 42" in joined
    assert "--headless" in joined
    assert "--disable_ros2_tracking_tune" in joined
    assert "--clip_param 0.3" in joined
    assert f'--morl_weights {exp["morl_weights"]}' in joined
    assert f"--run_name {run_name}" in joined


def test_find_run_dir_returns_latest_match():
    module = _load_module()
    with TemporaryDirectory(dir=ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        older = tmp_path / "2026-03-15_10-00-00_morl_p1_seed42"
        newer = tmp_path / "2026-03-15_10-30-00_morl_p1_seed42"
        older.mkdir()
        newer.mkdir()
        now = time.time()
        older_mtime = now - 100
        newer_mtime = now
        import os

        os.utime(older, (older_mtime, older_mtime))
        os.utime(newer, (newer_mtime, newer_mtime))

        found = module._find_run_dir(tmp_path, "morl_p1_seed42")

        assert found == newer


class _StubLog:
    def __init__(self):
        self.messages = []

    def info(self, msg: str):
        self.messages.append(("info", msg))

    def warn(self, msg: str):
        self.messages.append(("warn", msg))

    def error(self, msg: str):
        self.messages.append(("error", msg))


def test_verify_run_artifacts_checks_checkpoint_and_configs():
    module = _load_module()
    exp = module.MORL_EXPERIMENTS[0]
    with TemporaryDirectory(dir=ROOT) as tmp_dir:
        run_dir = Path(tmp_dir) / "2026-03-15_13-00-00_morl_p1_seed42"
        params_dir = run_dir / "params"
        params_dir.mkdir(parents=True)

        (run_dir / "model_1499.pt").write_text("checkpoint", encoding="utf-8")
        (params_dir / "agent.yaml").write_text(
            "algorithm:\n  clip_param: 0.3\n",
            encoding="utf-8",
        )
        (params_dir / "env.yaml").write_text(
            "\n".join(
                [
                    "rewards:",
                    "  track_lin_vel_xy_exp:",
                    "    weight: 0.7",
                    "  morl_energy:",
                    "    weight: 0.1",
                    "  morl_smooth:",
                    "    weight: 0.1",
                    "  morl_stable:",
                    "    weight: 0.1",
                ]
            ),
            encoding="utf-8",
        )

        assert module._verify_run_artifacts(
            run_dir, exp, max_iterations=1500, log=_StubLog()
        )


def test_verify_run_artifacts_rejects_weight_mismatch():
    module = _load_module()
    exp = module.MORL_EXPERIMENTS[0]
    with TemporaryDirectory(dir=ROOT) as tmp_dir:
        run_dir = Path(tmp_dir) / "2026-03-15_13-00-00_morl_p1_seed42"
        params_dir = run_dir / "params"
        params_dir.mkdir(parents=True)

        (run_dir / "model_1499.pt").write_text("checkpoint", encoding="utf-8")
        (params_dir / "agent.yaml").write_text(
            "algorithm:\n  clip_param: 0.3\n",
            encoding="utf-8",
        )
        (params_dir / "env.yaml").write_text(
            "\n".join(
                [
                    "rewards:",
                    "  track_lin_vel_xy_exp:",
                    "    weight: 0.25",
                    "  morl_energy:",
                    "    weight: 0.25",
                    "  morl_smooth:",
                    "    weight: 0.25",
                    "  morl_stable:",
                    "    weight: 0.25",
                ]
            ),
            encoding="utf-8",
        )

        assert not module._verify_run_artifacts(
            run_dir, exp, max_iterations=1500, log=_StubLog()
        )
