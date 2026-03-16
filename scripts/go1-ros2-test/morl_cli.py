"""Helpers for MORL-specific CLI behavior in training scripts."""

from __future__ import annotations

import math
from collections.abc import Sequence

MORL_PRIMARY_REWARD_NAMES = (
    "track_lin_vel_xy_exp",
    "morl_energy",
    "morl_smooth",
    "morl_stable",
)

MORL_TASK_IDS = frozenset(
    {
        "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-v0",
        "Isaac-Velocity-MORL-Unitree-Go1-ROS2Cmd-Play-v0",
    }
)


def parse_morl_weights(raw_value: str) -> tuple[float, float, float, float]:
    """Parse and validate a 4D MORL weight vector."""

    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != len(MORL_PRIMARY_REWARD_NAMES):
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

    if len(weights) != len(MORL_PRIMARY_REWARD_NAMES):
        raise ValueError(
            f"Expected {len(MORL_PRIMARY_REWARD_NAMES)} MORL weights, got {len(weights)}."
        )

    for reward_name, weight in zip(MORL_PRIMARY_REWARD_NAMES, weights, strict=True):
        reward_term = getattr(rewards, reward_name, None)
        if reward_term is None:
            raise AttributeError(
                f"Environment config is missing MORL reward term '{reward_name}'."
            )
        reward_term.weight = float(weight)

    return env_cfg


def format_morl_weights(weights: Sequence[float]) -> str:
    """Render MORL weights in a stable log format."""

    return ", ".join(
        f"{reward_name}={weight:.3f}"
        for reward_name, weight in zip(
            MORL_PRIMARY_REWARD_NAMES, weights, strict=True
        )
    )
