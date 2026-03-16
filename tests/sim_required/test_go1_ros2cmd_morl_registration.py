# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Integration-style registration test for MORL task ids."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.sim_required

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "robot_lab" / "source" / "robot_lab"
REGISTRY_PACKAGE = (
    "robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go1_ros2"
)
REGISTRY_INIT = (
    PACKAGE_ROOT
    / "robot_lab"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "config"
    / "quadruped"
    / "unitree_go1_ros2"
    / "__init__.py"
)

MORL_TRAIN_TASK = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v0"
MORL_PLAY_TASK = "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v0"


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    module.__path__ = [str(package_path)]
    sys.modules[name] = module


def _load_registry_module():
    import gymnasium as gym

    if (
        REGISTRY_PACKAGE in sys.modules
        and MORL_TRAIN_TASK in gym.registry
        and MORL_PLAY_TASK in gym.registry
    ):
        return sys.modules[REGISTRY_PACKAGE]

    for task_id in (MORL_TRAIN_TASK, MORL_PLAY_TASK):
        gym.registry.pop(task_id, None)

    package_map = {
        "robot_lab": PACKAGE_ROOT / "robot_lab",
        "robot_lab.tasks": PACKAGE_ROOT / "robot_lab" / "tasks",
        "robot_lab.tasks.manager_based": PACKAGE_ROOT / "robot_lab" / "tasks" / "manager_based",
        "robot_lab.tasks.manager_based.locomotion": PACKAGE_ROOT / "robot_lab" / "tasks" / "manager_based" / "locomotion",
        "robot_lab.tasks.manager_based.locomotion.velocity": PACKAGE_ROOT / "robot_lab" / "tasks" / "manager_based" / "locomotion" / "velocity",
        "robot_lab.tasks.manager_based.locomotion.velocity.config": PACKAGE_ROOT / "robot_lab" / "tasks" / "manager_based" / "locomotion" / "velocity" / "config",
        "robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped": PACKAGE_ROOT / "robot_lab" / "tasks" / "manager_based" / "locomotion" / "velocity" / "config" / "quadruped",
    }
    for package_name, package_path in package_map.items():
        _ensure_package(package_name, package_path)

    spec = importlib.util.spec_from_file_location(
        REGISTRY_PACKAGE,
        REGISTRY_INIT,
        submodule_search_locations=[str(REGISTRY_INIT.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[REGISTRY_PACKAGE] = module
    spec.loader.exec_module(module)
    return module


def _get_spec(task_id: str):
    import gymnasium as gym

    _load_registry_module()
    return gym.spec(task_id)


def test_morl_ros2cmd_train_task_registered():
    spec = _get_spec(MORL_TRAIN_TASK)

    assert spec is not None
    assert (
        spec.kwargs["env_cfg_entry_point"]
        == "robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go1_ros2.morl_env_cfg:UnitreeGo1MORLEnvCfg"
    )
    assert "UnitreeGo1RoughPPORunnerCfg" in spec.kwargs["rsl_rl_cfg_entry_point"]


def test_morl_ros2cmd_play_task_registered():
    spec = _get_spec(MORL_PLAY_TASK)

    assert spec is not None
    assert (
        spec.kwargs["env_cfg_entry_point"]
        == "robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go1_ros2.morl_env_cfg:UnitreeGo1MORLEnvCfg_PLAY"
    )
    assert "UnitreeGo1RoughPPORunnerCfg" in spec.kwargs["rsl_rl_cfg_entry_point"]


def test_morl_ros2cmd_uses_rough_ppo_runner():
    for task_id in (MORL_TRAIN_TASK, MORL_PLAY_TASK):
        spec = _get_spec(task_id)
        runner_path = spec.kwargs["rsl_rl_cfg_entry_point"]
        assert "RoughPPORunnerCfg" in runner_path
        assert "FlatPPORunnerCfg" not in runner_path
