# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MORL reward functions."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src" / "go1-ros2-test" / "envs" / "mdp" / "morl_rewards.py"
MODULE_V2_PATH = ROOT / "src" / "go1-ros2-test" / "envs" / "mdp" / "morl_rewards_v2.py"


def _load_morl_rewards_module():
    module_name = "_morl_rewards_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_morl_rewards_v2_module():
    package_name = "_morl_pkg_under_test"
    module_name = f"{package_name}.morl_rewards_v2"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(MODULE_PATH.parent)]
        sys.modules[package_name] = package

    base_module_name = f"{package_name}.morl_rewards"
    if base_module_name not in sys.modules:
        base_spec = importlib.util.spec_from_file_location(base_module_name, MODULE_PATH)
        base_module = importlib.util.module_from_spec(base_spec)
        assert base_spec.loader is not None
        sys.modules[base_module_name] = base_module
        base_spec.loader.exec_module(base_module)

    spec = importlib.util.spec_from_file_location(module_name, MODULE_V2_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


AVAILABLE_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@dataclass
class MockAssetData:
    root_lin_vel_b: torch.Tensor
    root_ang_vel_b: torch.Tensor
    joint_vel: torch.Tensor
    applied_torque: torch.Tensor


class MockAsset:
    def __init__(self, data: MockAssetData):
        self.data = data


class MockScene:
    def __init__(self, robot: MockAsset):
        self._robot = robot

    def __getitem__(self, name: str) -> MockAsset:
        if name != "robot":
            raise KeyError(name)
        return self._robot


class MockCommandManager:
    def __init__(self, command: torch.Tensor):
        self._command = command

    def get_command(self, name: str) -> torch.Tensor:
        assert name == "base_velocity"
        return self._command


class MockActionManager:
    def __init__(self, action: torch.Tensor, prev_action: torch.Tensor):
        self.action = action
        self.prev_action = prev_action


class MockEnv:
    def __init__(
        self,
        *,
        root_lin_vel_b: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
        joint_vel: torch.Tensor,
        applied_torque: torch.Tensor,
        command: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
    ):
        asset = MockAsset(
            MockAssetData(
                root_lin_vel_b=root_lin_vel_b,
                root_ang_vel_b=root_ang_vel_b,
                joint_vel=joint_vel,
                applied_torque=applied_torque,
            )
        )
        self.scene = MockScene(asset)
        self.command_manager = MockCommandManager(command)
        self.action_manager = MockActionManager(action, prev_action)


def _make_env(device: str, num_envs: int = 8, num_joints: int = 12) -> MockEnv:
    return MockEnv(
        root_lin_vel_b=torch.randn(num_envs, 3, device=device),
        root_ang_vel_b=torch.randn(num_envs, 3, device=device),
        joint_vel=torch.randn(num_envs, num_joints, device=device),
        applied_torque=torch.randn(num_envs, num_joints, device=device),
        command=torch.randn(num_envs, 3, device=device),
        action=torch.randn(num_envs, num_joints, device=device),
        prev_action=torch.randn(num_envs, num_joints, device=device),
    )


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_track_reward_shape_and_range(device: str):
    morl = _load_morl_rewards_module()
    env = _make_env(device)

    reward = morl.morl_track_vel_exp(env, command_name="base_velocity", scale=5.0)

    assert reward.shape == (8,)
    assert reward.device.type == device
    assert torch.all(reward > 0.0)
    assert torch.all(reward <= 1.0)


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_track_reward_hits_one_for_perfect_tracking(device: str):
    morl = _load_morl_rewards_module()
    command = torch.randn(6, 3, device=device)
    env = MockEnv(
        root_lin_vel_b=command.clone(),
        root_ang_vel_b=torch.randn(6, 3, device=device),
        joint_vel=torch.randn(6, 12, device=device),
        applied_torque=torch.randn(6, 12, device=device),
        command=command,
        action=torch.zeros(6, 12, device=device),
        prev_action=torch.zeros(6, 12, device=device),
    )

    reward = morl.morl_track_vel_exp(env, command_name="base_velocity", scale=5.0)

    torch.testing.assert_close(reward, torch.ones(6, device=device))


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_energy_reward_uses_all_joints_and_hits_one_for_zero_power(device: str):
    morl = _load_morl_rewards_module()
    env = MockEnv(
        root_lin_vel_b=torch.randn(5, 3, device=device),
        root_ang_vel_b=torch.randn(5, 3, device=device),
        joint_vel=torch.zeros(5, 12, device=device),
        applied_torque=torch.randn(5, 12, device=device),
        command=torch.randn(5, 3, device=device),
        action=torch.randn(5, 12, device=device),
        prev_action=torch.randn(5, 12, device=device),
    )

    reward = morl.morl_energy_power_exp(env, scale=1.0)
    reward_with_cfg = morl.morl_energy_power_exp(
        env,
        asset_cfg=SimpleNamespace(name="robot", joint_ids=slice(None)),
        scale=1.0,
    )

    torch.testing.assert_close(reward, torch.ones(5, device=device))
    torch.testing.assert_close(reward_with_cfg, torch.ones(5, device=device))


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_smoothness_and_stability_rewards_hit_one_at_zero_error(device: str):
    morl = _load_morl_rewards_module()
    action = torch.randn(4, 12, device=device)
    env = MockEnv(
        root_lin_vel_b=torch.randn(4, 3, device=device),
        root_ang_vel_b=torch.zeros(4, 3, device=device),
        joint_vel=torch.randn(4, 12, device=device),
        applied_torque=torch.randn(4, 12, device=device),
        command=torch.randn(4, 3, device=device),
        action=action,
        prev_action=action.clone(),
    )

    smooth_reward = morl.morl_action_smoothness_exp(env, scale=0.01)
    stable_reward = morl.morl_stability_ang_vel_exp(env, scale=1.0)

    torch.testing.assert_close(smooth_reward, torch.ones(4, device=device))
    torch.testing.assert_close(stable_reward, torch.ones(4, device=device))


def test_morl_v2_reward_signatures_match_v1_contract():
    morl = _load_morl_rewards_module()
    morl_v2 = _load_morl_rewards_v2_module()

    assert inspect.signature(morl_v2.morl_v2_speed_pref) == inspect.signature(morl.morl_track_vel_exp)
    assert inspect.signature(morl_v2.morl_v2_energy_pref) == inspect.signature(morl.morl_energy_power_exp)
    assert inspect.signature(morl_v2.morl_v2_smooth_pref) == inspect.signature(morl.morl_action_smoothness_exp)
    assert inspect.signature(morl_v2.morl_v2_stable_pref) == inspect.signature(morl.morl_stability_ang_vel_exp)
