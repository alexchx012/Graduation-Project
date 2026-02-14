# Canonical source: src/go1-ros2-test/envs/mdp/commands/ros2_velocity_command.py
# Deployed to: robot_lab/.../velocity/mdp/ros2_velocity_command.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import Any

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class Ros2VelocityCommand(CommandTerm):
    """Velocity command term fed by an external ROS2 command source."""

    cfg: Ros2VelocityCommandCfg

    def __init__(self, cfg: Ros2VelocityCommandCfg, env):
        super().__init__(cfg, env)

        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_command = torch.zeros(3, device=self.device)
        self._elapsed_time_s = 0.0
        self._last_rx_time_s = -math.inf
        self._last_source_stamp_s = -math.inf

        self.metrics["cmd_timeout_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cmd_zero_fallback_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cmd_vx"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cmd_vy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cmd_wz"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Current command tensor with shape (num_envs, 3)."""
        return self.vel_command_b

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset metrics/state without invoking base resampling."""
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long).flatten()
        else:
            env_ids_tensor = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

        extras = {}
        if env_ids_tensor.numel() == 0:
            return extras

        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value[env_ids_tensor]).item()
            metric_value[env_ids_tensor] = 0.0

        self.command_counter[env_ids_tensor] = 0
        self.time_left[env_ids_tensor] = math.inf
        self.vel_command_b[env_ids_tensor, :] = 0.0
        return extras

    def compute(self, dt: float):
        """Compute command without periodic random resampling."""
        self._update_metrics()
        self._update_command()

    def _update_metrics(self):
        # No robot-state tracking metrics for this command term.
        return

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) > 0:
            self.vel_command_b[env_ids, :] = 0.0

    def _update_command(self):
        self._elapsed_time_s += self._env.step_dt
        latest_cmd, has_new_cmd = self._read_latest_command()
        elapsed_since_rx = self._elapsed_time_s - self._last_rx_time_s

        if has_new_cmd:
            clipped = self._clip_command(latest_cmd)
            self._last_command.copy_(clipped)
            self._last_rx_time_s = self._elapsed_time_s
            self.vel_command_b[:, :] = clipped
        elif elapsed_since_rx <= self.cfg.cmd_timeout_s:
            self.vel_command_b[:, :] = self._last_command
            self.metrics["cmd_timeout_count"] += 1.0
        else:
            self.vel_command_b[:, :] = 0.0
            self.metrics["cmd_zero_fallback_count"] += 1.0

        # Record current command values for TensorBoard
        self.metrics["cmd_vx"][:] = self.vel_command_b[:, 0]
        self.metrics["cmd_vy"][:] = self.vel_command_b[:, 1]
        self.metrics["cmd_wz"][:] = self.vel_command_b[:, 2]

    def _read_latest_command(self) -> tuple[torch.Tensor, bool]:
        payload, holder = self._resolve_payload_and_holder()
        if payload is None:
            return self._last_command, False

        parsed_command, inline_stamp = self._parse_payload(payload)
        if parsed_command is None:
            return self._last_command, False

        source_stamp_s = self._resolve_source_stamp_s(holder, inline_stamp)
        if source_stamp_s is None:
            # Without timestamp, treat command as new only on first reception
            # or when the value changes. This avoids refreshing timeout forever
            # from a stale buffer that still holds the last command.
            clipped_no_stamp = self._clip_command(parsed_command)
            is_first_rx = not math.isfinite(self._last_rx_time_s)
            is_changed = not torch.allclose(clipped_no_stamp, self._last_command, rtol=0.0, atol=1.0e-6)
            return clipped_no_stamp, (is_first_rx or is_changed)

        if source_stamp_s <= self._last_source_stamp_s:
            return parsed_command, False

        self._last_source_stamp_s = source_stamp_s
        return parsed_command, True

    def _resolve_payload_and_holder(self) -> tuple[Any | None, Any | None]:
        holders = [self._env]
        unwrapped = getattr(self._env, "unwrapped", None)
        if unwrapped is not None and unwrapped is not self._env:
            holders.append(unwrapped)

        for holder in holders:
            if hasattr(holder, self.cfg.command_attr):
                return getattr(holder, self.cfg.command_attr), holder
        return None, None

    def _resolve_source_stamp_s(self, holder: Any | None, inline_stamp: Any | None) -> float | None:
        if inline_stamp is not None:
            return self._to_float(inline_stamp)

        if holder is None or self.cfg.command_stamp_attr is None:
            return None

        if not hasattr(holder, self.cfg.command_stamp_attr):
            return None

        return self._to_float(getattr(holder, self.cfg.command_stamp_attr))

    def _parse_payload(self, payload: Any) -> tuple[torch.Tensor | None, Any | None]:
        if isinstance(payload, torch.Tensor):
            flat = payload.flatten()
            if flat.numel() < 3:
                return None, None
            return flat[:3].to(device=self.device, dtype=torch.float32), None

        if isinstance(payload, (list, tuple)):
            if len(payload) < 3:
                return None, None
            try:
                return torch.tensor([float(payload[0]), float(payload[1]), float(payload[2])], device=self.device), None
            except (TypeError, ValueError):
                return None, None

        if isinstance(payload, dict):
            if "cmd" in payload:
                nested_cmd, nested_stamp = self._parse_payload(payload["cmd"])
                return nested_cmd, payload.get("stamp_s", nested_stamp)

            x = self._first_present(payload, ("linear_x", "lin_vel_x", "vx", "x"))
            y = self._first_present(payload, ("linear_y", "lin_vel_y", "vy", "y"))
            z = self._first_present(payload, ("angular_z", "ang_vel_z", "wz", "z"))
            if x is None or y is None or z is None:
                return None, None

            try:
                cmd_tensor = torch.tensor([float(x), float(y), float(z)], device=self.device, dtype=torch.float32)
            except (TypeError, ValueError):
                return None, None

            stamp = self._first_present(payload, ("stamp_s", "timestamp_s", "time_s", "stamp"))
            return cmd_tensor, stamp

        return None, None

    def _clip_command(self, cmd: torch.Tensor) -> torch.Tensor:
        clipped = cmd.to(device=self.device, dtype=torch.float32).clone()
        clipped[0] = torch.clamp(clipped[0], min=self.cfg.ranges.lin_vel_x[0], max=self.cfg.ranges.lin_vel_x[1])
        clipped[1] = torch.clamp(clipped[1], min=self.cfg.ranges.lin_vel_y[0], max=self.cfg.ranges.lin_vel_y[1])
        clipped[2] = torch.clamp(clipped[2], min=self.cfg.ranges.ang_vel_z[0], max=self.cfg.ranges.ang_vel_z[1])
        return clipped

    @staticmethod
    def _first_present(data: dict[str, Any], keys: Sequence[str]) -> Any | None:
        for key in keys:
            if key in data:
                return data[key]
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.item())
            return None
        if hasattr(value, "sec") and hasattr(value, "nanosec"):
            return float(value.sec) + float(value.nanosec) * 1.0e-9
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


@configclass
class Ros2VelocityCommandCfg(CommandTermCfg):
    """Configuration for ROS2-driven velocity command term."""

    class_type: type = Ros2VelocityCommand

    command_attr: str = "ros2_latest_cmd_vel"
    """Attribute name on env/unwrapped env that stores latest [vx, vy, wz]."""

    command_stamp_attr: str | None = "ros2_latest_cmd_stamp_s"
    """Attribute name on env/unwrapped env that stores command timestamp in seconds."""

    cmd_timeout_s: float = 0.5
    """Timeout before zero fallback. Within timeout, last command is held."""

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    def __post_init__(self):
        # External source drives updates, so we disable periodic random resampling.
        self.resampling_time_range = (math.inf, math.inf)
