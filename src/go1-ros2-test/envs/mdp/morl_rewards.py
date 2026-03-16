# Canonical source: src/go1-ros2-test/envs/mdp/morl_rewards.py
# Deployed to: robot_lab/.../mdp/morl_rewards.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MORL (Multi-Objective Reinforcement Learning) reward functions.

This module implements the 4 primary objectives for the MORL phase:
- Speed: velocity tracking accuracy
- Energy: mechanical power consumption
- Smoothness: action rate penalty
- Stability: angular velocity penalty

All functions return values in (0, 1] using exponential kernels for consistent scaling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
else:
    ManagerBasedRLEnv = Any
    SceneEntityCfg = Any


def morl_track_vel_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    scale: float = 5.0,
) -> torch.Tensor:
    """Primary MORL speed objective based on xy velocity tracking error."""
    asset = env.scene["robot"]
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2]
            - asset.data.root_lin_vel_b[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-scale * lin_vel_error)


def morl_energy_power_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Primary MORL energy objective based on joint power."""
    asset_name, joint_ids = _resolve_asset_cfg(asset_cfg)
    asset = env.scene[asset_name]
    power = torch.sum(
        torch.abs(
            asset.data.joint_vel[:, joint_ids]
            * asset.data.applied_torque[:, joint_ids]
        ),
        dim=1,
    )
    return torch.exp(-scale * power)


def morl_action_smoothness_exp(
    env: ManagerBasedRLEnv,
    scale: float = 0.01,
) -> torch.Tensor:
    """Primary MORL smoothness objective based on action deltas."""
    action_diff = env.action_manager.action - env.action_manager.prev_action
    action_rate_sq = torch.sum(torch.square(action_diff), dim=1)
    return torch.exp(-scale * action_rate_sq)


def morl_stability_ang_vel_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Primary MORL stability objective based on body roll/pitch rates."""
    asset_name, _ = _resolve_asset_cfg(asset_cfg)
    asset = env.scene[asset_name]
    ang_vel_xy = asset.data.root_ang_vel_b[:, :2]
    ang_vel_sq = torch.sum(torch.square(ang_vel_xy), dim=1)
    return torch.exp(-scale * ang_vel_sq)


def _resolve_asset_cfg(asset_cfg: SceneEntityCfg | None) -> tuple[str, slice | list[int]]:
    if asset_cfg is None:
        return "robot", slice(None)
    return asset_cfg.name, asset_cfg.joint_ids
