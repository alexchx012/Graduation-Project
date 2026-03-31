"""Helpers for MORL-specific CLI behavior in training scripts."""

from __future__ import annotations

import math
from collections.abc import Sequence

MORL_OBJECTIVE_LABELS = (
    "speed",
    "energy",
    "smooth",
    "stable",
)

MORL_PRIMARY_REWARD_NAMES = (
    "track_lin_vel_xy_exp",
    "morl_energy",
    "morl_smooth",
    "morl_stable",
)

MORL_PRIMARY_REWARD_NAMES_V2 = (
    "morl_speed",
    "morl_energy",
    "morl_smooth",
    "morl_stable",
)

MORL_ROS2_TASK_IDS = frozenset(
    {
        "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v0",
        "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v0",
    }
)

MORL_INTERNAL_CMD_TASK_IDS = frozenset(
    {
        "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v2",
        "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v2",
    }
)

MORL_TASK_IDS = MORL_ROS2_TASK_IDS | MORL_INTERNAL_CMD_TASK_IDS

MORL_COMMAND_PROFILES = {
    "repair_forward_v1": {
        "lin_vel_x": (0.5, 1.5),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
        "heading_command": False,
        "rel_heading_envs": 0.0,
        "rel_standing_envs": 0.0,
    },
    "repair_forward_v2": {
        "lin_vel_x": (0.5, 1.5),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.5, 0.5),
        "heading_command": False,
        "rel_heading_envs": 0.0,
        "rel_standing_envs": 0.0,
    },
}

MORL_COMMAND_PROFILE_NAMES = frozenset(MORL_COMMAND_PROFILES)


def parse_morl_weights(raw_value: str) -> tuple[float, float, float, float]:
    """Parse and validate a 4D MORL weight vector."""

    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != len(MORL_OBJECTIVE_LABELS):
        raise ValueError(
            "--morl_weights must provide exactly 4 comma-separated values "
            "[speed,energy,smooth,stable]."
        )

    try:
        weights = tuple(float(part) for part in parts)
    except ValueError as err:
        raise ValueError("--morl_weights must contain only numeric values.") from err

    if any(weight < 0.0 for weight in weights):
        raise ValueError("--morl_weights cannot contain negative values.")

    total = sum(weights)
    if total <= 0.0:
        raise ValueError("--morl_weights must sum to 1.0, not 0.")
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"--morl_weights must sum to 1.0, got {total:.6f}.")

    return weights


def apply_morl_weight_override(env_cfg, weights: Sequence[float]):
    """Override only the four MORL primary objective weights on an env config."""

    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        raise AttributeError("Environment config does not expose a rewards section.")

    reward_names = _resolve_morl_reward_names(rewards)
    if len(weights) != len(reward_names):
        raise ValueError(
            f"Expected {len(reward_names)} MORL weights, got {len(weights)}."
        )

    secondary_scale = _resolve_secondary_scale(env_cfg, reward_names)
    for reward_name, weight in zip(reward_names, weights, strict=True):
        reward_term = getattr(rewards, reward_name, None)
        if reward_term is None:
            raise AttributeError(
                f"Environment config is missing MORL reward term '{reward_name}'."
            )
        reward_term.weight = float(weight * secondary_scale)

    return env_cfg


def apply_morl_command_profile(env_cfg, profile_name: str):
    """Override the MORL training command distribution for repair pilots."""

    if profile_name not in MORL_COMMAND_PROFILES:
        raise ValueError(
            f"Unknown MORL command profile '{profile_name}'. "
            f"Known: {sorted(MORL_COMMAND_PROFILE_NAMES)}."
        )

    commands = getattr(env_cfg, "commands", None)
    if commands is None or getattr(commands, "base_velocity", None) is None:
        raise AttributeError("Environment config does not expose commands.base_velocity.")

    base_velocity = commands.base_velocity
    ranges = getattr(base_velocity, "ranges", None)
    if ranges is None:
        raise AttributeError("Environment config does not expose base_velocity.ranges.")

    profile = MORL_COMMAND_PROFILES[profile_name]
    ranges.lin_vel_x = tuple(profile["lin_vel_x"])
    ranges.lin_vel_y = tuple(profile["lin_vel_y"])
    ranges.ang_vel_z = tuple(profile["ang_vel_z"])

    if hasattr(base_velocity, "heading_command"):
        base_velocity.heading_command = bool(profile["heading_command"])
    if hasattr(base_velocity, "rel_heading_envs"):
        base_velocity.rel_heading_envs = float(profile["rel_heading_envs"])
    if hasattr(base_velocity, "rel_standing_envs"):
        base_velocity.rel_standing_envs = float(profile["rel_standing_envs"])

    return env_cfg


def format_morl_weights(weights: Sequence[float]) -> str:
    """Render MORL weights in a stable log format."""

    return ", ".join(
        f"{reward_name}={weight:.3f}"
        for reward_name, weight in zip(
            MORL_OBJECTIVE_LABELS, weights, strict=True
        )
    )


def _resolve_morl_reward_names(rewards) -> tuple[str, str, str, str]:
    if hasattr(rewards, MORL_PRIMARY_REWARD_NAMES_V2[0]):
        return MORL_PRIMARY_REWARD_NAMES_V2
    return MORL_PRIMARY_REWARD_NAMES


def _resolve_secondary_scale(env_cfg, reward_names: Sequence[str]) -> float:
    if tuple(reward_names) == MORL_PRIMARY_REWARD_NAMES_V2:
        return float(getattr(env_cfg, "morl_secondary_scale", 1.0))
    return 1.0
