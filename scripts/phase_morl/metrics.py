"""Shared metric helpers for phase MORL evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def compute_j_speed(commanded_xy: torch.Tensor, actual_xy: torch.Tensor) -> torch.Tensor:
    """RMSE of xy velocity tracking error."""

    error_norm = torch.linalg.vector_norm(actual_xy - commanded_xy, dim=-1)
    return torch.sqrt(torch.mean(torch.square(error_norm), dim=-1))


def compute_j_energy(
    joint_torque: torch.Tensor,
    joint_vel: torch.Tensor,
    dt: float,
    distance: torch.Tensor | float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mechanical energy proxy per unit distance."""

    power = torch.sum(torch.abs(joint_torque * joint_vel), dim=-1)
    total_energy = torch.sum(power, dim=-1) * dt
    distance_tensor = torch.as_tensor(
        distance,
        dtype=total_energy.dtype,
        device=total_energy.device,
    )
    return total_energy / torch.clamp(distance_tensor, min=eps)


def compute_j_smooth(actions: torch.Tensor) -> torch.Tensor:
    """Mean action-rate magnitude."""

    if actions.shape[-2] < 2:
        return torch.zeros(actions.shape[:-2], dtype=actions.dtype, device=actions.device)
    action_diff = actions[..., 1:, :] - actions[..., :-1, :]
    action_rate = torch.linalg.vector_norm(action_diff, dim=-1)
    return torch.mean(action_rate, dim=-1)


def compute_j_stable(
    pose_fluctuation: torch.Tensor | None,
    ang_vel_xy: torch.Tensor,
    pose_weight: float = 0.5,
    ang_vel_weight: float = 0.5,
) -> torch.Tensor:
    """Combined pose and angular-velocity stability metric."""

    if pose_fluctuation is None:
        pose_fluctuation = torch.zeros_like(ang_vel_xy)

    pose_rms = torch.sqrt(torch.mean(torch.square(pose_fluctuation), dim=-2))
    ang_rms = torch.sqrt(torch.mean(torch.square(ang_vel_xy), dim=-2))

    return (
        pose_weight * torch.linalg.vector_norm(pose_rms, dim=-1)
        + ang_vel_weight * torch.linalg.vector_norm(ang_rms, dim=-1)
    )


def summarize_morl_metrics(
    *,
    commanded_xy: torch.Tensor,
    actual_xy: torch.Tensor,
    joint_torque: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    ang_vel_xy: torch.Tensor,
    pose_fluctuation: torch.Tensor | None,
    dt: float,
    distance: torch.Tensor | float,
) -> dict[str, float]:
    """Compute the four MORL physical metrics and return JSON-friendly scalars."""

    metrics: Mapping[str, torch.Tensor] = {
        "J_speed": compute_j_speed(commanded_xy, actual_xy),
        "J_energy": compute_j_energy(joint_torque, joint_vel, dt, distance),
        "J_smooth": compute_j_smooth(actions),
        "J_stable": compute_j_stable(pose_fluctuation, ang_vel_xy),
    }
    return {name: _to_scalar(value) for name, value in metrics.items()}


def _to_scalar(value: torch.Tensor) -> float:
    if value.numel() != 1:
        raise ValueError(f"Expected a scalar metric tensor, got shape {tuple(value.shape)}.")
    return float(value.item())
