#!/usr/bin/env python3
# Canonical source: scripts/go1-ros2-test/ros2_nodes/go1_cmd_script_node.py
# Standalone WSL ROS2 node — not part of robot_lab package
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ROS2 scripted velocity command publisher for Go1.

Publishes geometry_msgs/Twist on a configurable topic at a fixed rate.
Supports constant, sine, and step command profiles.

Usage (inside WSL Ubuntu-22.04):
    source /opt/ros/humble/setup.bash
    python3 go1_cmd_script_node.py --profile constant --vx 0.5 --rate 20
"""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class Go1CmdScriptNode(Node):
    """Publishes scripted velocity commands on /go1/cmd_vel."""

    def __init__(
        self,
        topic: str,
        rate: float,
        profile: str,
        vx: float,
        vy: float,
        wz: float,
        duration: float,
        qos_reliability: str,
        qos_durability: str,
        qos_history_depth: int,
    ):
        super().__init__("go1_cmd_script_node")

        reliability_policy = (
            ReliabilityPolicy.BEST_EFFORT
            if qos_reliability == "best_effort"
            else ReliabilityPolicy.RELIABLE
        )
        durability_policy = (
            DurabilityPolicy.TRANSIENT_LOCAL
            if qos_durability == "transient_local"
            else DurabilityPolicy.VOLATILE
        )
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, qos_history_depth),
            reliability=reliability_policy,
            durability=durability_policy,
        )

        self._publisher = self.create_publisher(Twist, topic, qos_profile)
        self._timer = self.create_timer(1.0 / rate, self._timer_callback)

        self._profile = profile
        self._vx = vx
        self._vy = vy
        self._wz = wz
        self._duration = duration

        self._start_time = time.time()

        self.get_logger().info(
            f"Go1CmdScriptNode started: topic={topic}, rate={rate}Hz, "
            f"profile={profile}, vx={vx}, vy={vy}, wz={wz}, duration={duration}"
        )
        self.get_logger().info(
            "QoS: "
            f"reliability={qos_reliability}, durability={qos_durability}, depth={max(1, qos_history_depth)}"
        )

    def _timer_callback(self):
        elapsed = time.time() - self._start_time

        # Check duration limit
        if math.isfinite(self._duration) and elapsed >= self._duration:
            self.get_logger().info(
                f"Duration {self._duration}s reached, shutting down."
            )
            raise SystemExit(0)

        vx, vy, wz = self._compute_command(elapsed)

        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = wz
        self._publisher.publish(msg)

        self.get_logger().info(
            f"[t={elapsed:.2f}s] pub Twist(vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f})"
        )

    def _compute_command(self, t: float) -> tuple[float, float, float]:
        """Compute (vx, vy, wz) based on the selected profile."""
        if self._profile == "constant":
            return self._vx, self._vy, self._wz

        elif self._profile == "sine":
            freq = 0.1  # Hz
            scale = math.sin(2.0 * math.pi * freq * t)
            return (
                self._vx * scale,
                self._vy * scale,
                self._wz * scale,
            )

        elif self._profile == "step":
            # Toggle every 5 seconds
            period_idx = int(t / 5.0)
            on = (period_idx % 2) == 0
            return (
                self._vx if on else 0.0,
                self._vy if on else 0.0,
                self._wz if on else 0.0,
            )

        else:
            # Fallback to constant
            return self._vx, self._vy, self._wz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Go1 scripted ROS2 velocity command publisher."
    )
    parser.add_argument(
        "--topic", type=str, default="/go1/cmd_vel", help="ROS2 topic name"
    )
    parser.add_argument(
        "--rate", type=float, default=20.0, help="Publish frequency in Hz"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="constant",
        choices=["constant", "sine", "step"],
        help="Command generation profile",
    )
    parser.add_argument("--vx", type=float, default=0.5, help="Forward velocity (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity (m/s)")
    parser.add_argument(
        "--wz", type=float, default=0.0, help="Yaw angular velocity (rad/s)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=float("inf"),
        help="Run duration in seconds (default: inf = run forever)",
    )
    parser.add_argument(
        "--qos_reliability",
        type=str,
        default="reliable",
        choices=["best_effort", "reliable"],
        help="ROS2 publisher reliability QoS",
    )
    parser.add_argument(
        "--qos_durability",
        type=str,
        default="volatile",
        choices=["volatile", "transient_local"],
        help="ROS2 publisher durability QoS",
    )
    parser.add_argument(
        "--qos_history_depth",
        type=int,
        default=5,
        help="ROS2 publisher queue depth",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = Go1CmdScriptNode(
        topic=args.topic,
        rate=args.rate,
        profile=args.profile,
        vx=args.vx,
        vy=args.vy,
        wz=args.wz,
        duration=args.duration,
        qos_reliability=args.qos_reliability,
        qos_durability=args.qos_durability,
        qos_history_depth=args.qos_history_depth,
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
