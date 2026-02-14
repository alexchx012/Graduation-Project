#!/usr/bin/env python3
# Canonical source: scripts/go1-ros2-test/ros2_nodes/go1_cmd_model_node.py
# Standalone WSL ROS2 node — not part of robot_lab package
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ROS2 model-inference velocity command publisher for Go1 (stub).

Subscribes to observation data and publishes inferred velocity commands.
Currently uses a stub inference function; model loading is TODO.

Usage (inside WSL Ubuntu-22.04):
    source /opt/ros/humble/setup.bash
    python3 go1_cmd_model_node.py --rate 20
"""

from __future__ import annotations

import argparse

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class Go1CmdModelNode(Node):
    """Subscribes to observations and publishes inferred velocity commands."""

    def __init__(
        self,
        obs_topic: str,
        cmd_topic: str,
        rate: float,
        model_path: str | None,
    ):
        super().__init__("go1_cmd_model_node")

        self._publisher = self.create_publisher(Twist, cmd_topic, 10)
        self._subscription = self.create_subscription(
            Float32MultiArray,
            obs_topic,
            self._obs_callback,
            10,
        )
        self._timer = self.create_timer(1.0 / rate, self._timer_callback)

        self._model_path = model_path
        self._latest_obs: list[float] | None = None
        self._obs_count: int = 0

        # TODO: torch.load(model_path); model = ...
        if model_path:
            self.get_logger().info(f"Model path provided: {model_path} (loading not yet implemented)")
        else:
            self.get_logger().info("No model path — using stub inference")

        self.get_logger().info(
            f"Go1CmdModelNode started: obs={obs_topic}, cmd={cmd_topic}, rate={rate}Hz"
        )

    def _obs_callback(self, msg: Float32MultiArray):
        """Store latest observation from ROS2."""
        self._latest_obs = list(msg.data)
        self._obs_count += 1
        if self._obs_count % 100 == 1:
            dim = len(self._latest_obs)
            first_5 = self._latest_obs[:5] if dim >= 5 else self._latest_obs
            self.get_logger().info(f"[obs #{self._obs_count}] dim={dim}, first_5={first_5}")

    def _timer_callback(self):
        """Publish inferred command at fixed rate."""
        cmd = self._infer(self._latest_obs)

        msg = Twist()
        msg.linear.x = cmd[0]
        msg.linear.y = cmd[1]
        msg.angular.z = cmd[2]
        self._publisher.publish(msg)

    def _infer(self, obs: list[float] | None) -> tuple[float, float, float]:
        """Stub inference function.

        Args:
            obs: Latest observation vector, or None if no observation received.

        Returns:
            Tuple of (vx, vy, wz) velocity command.
        """
        # TODO: torch.load(model_path); cmd = model.forward(obs)
        # For now, return a fixed safe command regardless of input.
        return (0.3, 0.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Go1 model inference ROS2 command publisher (stub).")
    parser.add_argument("--obs_topic", type=str, default="/go1/obs_flat", help="Observation subscription topic")
    parser.add_argument("--cmd_topic", type=str, default="/go1/cmd_vel", help="Command publish topic")
    parser.add_argument("--rate", type=float, default=20.0, help="Publish frequency in Hz")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model file (not yet implemented)")
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = Go1CmdModelNode(
        obs_topic=args.obs_topic,
        cmd_topic=args.cmd_topic,
        rate=args.rate,
        model_path=args.model_path,
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
