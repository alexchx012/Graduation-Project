# Canonical source: src/go1-ros2-test/envs/mdp/morl_rewards_v2.py
# Deployed to: robot_lab/.../mdp/morl_rewards_v2.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MORL v2 secondary objective rewards built on top of a fixed locomotion scaffold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .morl_rewards import (
    morl_action_smoothness_exp,
    morl_energy_power_exp,
    morl_stability_ang_vel_exp,
    morl_track_vel_exp,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    import torch
else:
    ManagerBasedRLEnv = Any
    SceneEntityCfg = Any


def morl_v2_speed_pref(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    scale: float = 1.0,
) -> torch.Tensor:
    """Secondary speed preference term used by MORL v2."""

    return morl_track_vel_exp(env, command_name=command_name, scale=scale)


def morl_v2_energy_pref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    scale: float = 0.005,
) -> torch.Tensor:
    """Secondary energy preference term used by MORL v2."""

    return morl_energy_power_exp(env, asset_cfg=asset_cfg, scale=scale)


def morl_v2_smooth_pref(
    env: ManagerBasedRLEnv,
    scale: float = 0.01,
) -> torch.Tensor:
    """Secondary smoothness preference term used by MORL v2."""

    return morl_action_smoothness_exp(env, scale=scale)


def morl_v2_stable_pref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Secondary stability preference term used by MORL v2."""

    return morl_stability_ang_vel_exp(env, asset_cfg=asset_cfg, scale=scale)
