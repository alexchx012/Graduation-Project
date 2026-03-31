# Canonical source: src/go1-ros2-test/envs/mdp/morl_rewards_v2.py
# Deployed to: robot_lab/.../mdp/morl_rewards_v2.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MORL v2 secondary objective rewards built on top of a fixed locomotion scaffold.

Supports an optional curriculum that zeros out MORL rewards during an initial
warmup phase, then linearly ramps them to full strength.  Controlled by two
env-config attributes:

    morl_curriculum_warmup_steps : int   (default 0 → disabled)
    morl_curriculum_ramp_steps  : int   (default 0 → instant on after warmup)
"""

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


def _morl_curriculum_scale(env: ManagerBasedRLEnv) -> float:
    """Return a [0, 1] multiplier for MORL secondary rewards.

    * During warmup (step < warmup_steps): returns 0.0
    * During ramp  (warmup ≤ step < warmup + ramp): linear 0→1
    * After ramp:  returns 1.0
    """
    warmup = getattr(env.cfg, "morl_curriculum_warmup_steps", 0)
    ramp = getattr(env.cfg, "morl_curriculum_ramp_steps", 0)
    if warmup <= 0 and ramp <= 0:
        return 1.0
    step = env.common_step_counter
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    progress = (step - warmup) / ramp
    return min(progress, 1.0)


def morl_v2_speed_pref(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    scale: float = 1.0,
) -> torch.Tensor:
    """Secondary speed preference term used by MORL v2."""

    return _morl_curriculum_scale(env) * morl_track_vel_exp(
        env, command_name=command_name, scale=scale
    )


def morl_v2_energy_pref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    scale: float = 0.005,
) -> torch.Tensor:
    """Secondary energy preference term used by MORL v2."""

    return _morl_curriculum_scale(env) * morl_energy_power_exp(
        env, asset_cfg=asset_cfg, scale=scale
    )


def morl_v2_smooth_pref(
    env: ManagerBasedRLEnv,
    scale: float = 0.01,
) -> torch.Tensor:
    """Secondary smoothness preference term used by MORL v2."""

    return _morl_curriculum_scale(env) * morl_action_smoothness_exp(env, scale=scale)


def morl_v2_stable_pref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Secondary stability preference term used by MORL v2."""

    return _morl_curriculum_scale(env) * morl_stability_ang_vel_exp(
        env, asset_cfg=asset_cfg, scale=scale
    )
