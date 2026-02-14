# Canonical source: src/go1-ros2-test/ros2_bridge/twist_subscriber_graph.py
# Deployed to: robot_lab/source/robot_lab/robot_lab/ros2_bridge/twist_subscriber_graph.py (runtime)
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ROS2 Twist subscriber using rclpy (not OmniGraph).

Reference: Isaac Sim standalone_examples/api/isaacsim.ros2.bridge/subscriber.py
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Literal

_logger = logging.getLogger(__name__)


@dataclass
class Ros2TwistBridgeCfg:
    """ROS2 Twist subscriber bridge configuration."""

    topic_name: str = "/go1/cmd_vel"
    queue_size: int = 10
    startup_mode: Literal["startup_blocking", "non_blocking"] = "startup_blocking"
    startup_timeout_s: float = 15.0
    command_attr: str = "ros2_latest_cmd_vel"
    command_stamp_attr: str = "ros2_latest_cmd_stamp_s"


class Ros2TwistSubscriberAdapter:
    """Adapter that subscribes ROS2 Twist via rclpy and syncs to env attributes."""

    def __init__(self, cfg: Ros2TwistBridgeCfg):
        self.cfg = cfg
        self._env = None
        self._node = None
        self._subscription = None
        self._callback_name = f"ros2_twist_sync_{id(self)}"
        self._last_cmd = [0.0, 0.0, 0.0]  # [vx, vy, wz]
        self._last_rx_time = -math.inf
        self._lock = threading.Lock()
        self._setup_done = False

    def setup(self):
        """Initialize rclpy and create subscriber node."""
        if self._setup_done:
            return

        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Twist

            # Initialize rclpy if not already done
            if not rclpy.ok():
                rclpy.init()

            # Create node
            self._node = rclpy.create_node(f"isaac_twist_sub_{id(self)}")

            # Create subscription
            self._subscription = self._node.create_subscription(
                Twist,
                self.cfg.topic_name,
                self._twist_callback,
                self.cfg.queue_size,
            )

            self._setup_done = True
            _logger.debug(f"ROS2 rclpy subscriber created for topic: {self.cfg.topic_name}")

        except Exception as err:
            raise RuntimeError(f"Failed to setup ROS2 rclpy subscriber: {err}") from err

    def _twist_callback(self, msg):
        """ROS2 callback - store latest command thread-safely."""
        with self._lock:
            self._last_cmd = [msg.linear.x, msg.linear.y, msg.angular.z]
            self._last_rx_time = time.time()

    def attach(self, env):
        """Attach to env and register physics callback."""
        self._env = env
        unwrapped = getattr(env, "unwrapped", env)

        setattr(unwrapped, self.cfg.command_attr, [0.0, 0.0, 0.0])
        setattr(unwrapped, self.cfg.command_stamp_attr, -math.inf)

        sim = getattr(unwrapped, "sim", None)
        if sim is None:
            try:
                from isaacsim.core.api import get_current_simulation_context
                sim = get_current_simulation_context()
            except ImportError:
                pass

        if sim is None or not hasattr(sim, "add_physics_callback"):
            raise RuntimeError(
                "Cannot find physics callback API. "
                "Expected env.unwrapped.sim.add_physics_callback() to be available."
            )

        try:
            sim.remove_physics_callback(self._callback_name)
        except Exception:
            pass

        sim.add_physics_callback(self._callback_name, self._sync_callback)
        _logger.debug(f"Registered physics callback: {self._callback_name}")

    def _sync_callback(self, _event):
        """Sync latest ROS2 command to env attributes on each physics step."""
        if self._env is None or self._node is None:
            return

        # Spin once to process any pending messages (non-blocking)
        try:
            import rclpy
            rclpy.spin_once(self._node, timeout_sec=0.0)
        except Exception:
            pass

        # Copy latest command thread-safely
        with self._lock:
            cmd = list(self._last_cmd)
            rx_time = self._last_rx_time

        unwrapped = getattr(self._env, "unwrapped", self._env)
        setattr(unwrapped, self.cfg.command_attr, cmd)
        setattr(unwrapped, self.cfg.command_stamp_attr, rx_time)

    def wait_for_first_message(self, timeout_s: float = 15.0) -> bool:
        """Block until first non-zero command arrives or timeout."""
        import rclpy

        start = time.time()
        while time.time() - start < timeout_s:
            if self._node is not None:
                rclpy.spin_once(self._node, timeout_sec=0.1)

            with self._lock:
                if any(abs(v) > 1.0e-9 for v in self._last_cmd):
                    _logger.debug(f"First ROS2 message received after {time.time() - start:.2f}s")
                    return True

            time.sleep(0.05)

        return False

    def close(self):
        """Cleanup node and callback."""
        if self._env is not None:
            try:
                unwrapped = getattr(self._env, "unwrapped", self._env)
                sim = getattr(unwrapped, "sim", None)
                if sim is not None:
                    sim.remove_physics_callback(self._callback_name)
            except Exception as err:
                _logger.warning(f"Failed to remove physics callback: {err}")

        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None

        self._env = None
        _logger.debug("ROS2 Twist subscriber adapter closed")


# Backwards compatible aliases
Ros2TwistSubscriberGraphAdapter = Ros2TwistSubscriberAdapter
