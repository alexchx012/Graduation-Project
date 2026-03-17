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
            import os
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Twist

            # Log ROS2 environment for debugging
            ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "0 (default)")
            rmw_impl = os.environ.get("RMW_IMPLEMENTATION", "default")
            ros_localhost = os.environ.get("ROS_LOCALHOST_ONLY", "0 (default)")
            fastrtps_cfg = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "none")
            print(f"[ROS2 Bridge] ROS_DOMAIN_ID={ros_domain_id}, RMW_IMPLEMENTATION={rmw_impl}")
            print(f"[ROS2 Bridge] ROS_LOCALHOST_ONLY={ros_localhost}")
            print(f"[ROS2 Bridge] FASTRTPS_DEFAULT_PROFILES_FILE={fastrtps_cfg}")
            print(f"[ROS2 Bridge] Subscribing to topic: {self.cfg.topic_name}")

            # Initialize rclpy if not already done
            if not rclpy.ok():
                rclpy.init()
                print("[ROS2 Bridge] rclpy initialized")

            # Create node
            self._node = rclpy.create_node(f"isaac_twist_sub_{id(self)}")
            print(f"[ROS2 Bridge] Node created: {self._node.get_name()}")

            # Create subscription
            self._subscription = self._node.create_subscription(
                Twist,
                self.cfg.topic_name,
                self._twist_callback,
                self.cfg.queue_size,
            )
            print(f"[ROS2 Bridge] Subscription created for {self.cfg.topic_name}")

            self._setup_done = True
            _logger.debug(f"ROS2 rclpy subscriber created for topic: {self.cfg.topic_name}")

        except Exception as err:
            raise RuntimeError(f"Failed to setup ROS2 rclpy subscriber: {err}") from err

    def _twist_callback(self, msg):
        """ROS2 callback - store latest command thread-safely."""
        with self._lock:
            self._last_cmd = [msg.linear.x, msg.linear.y, msg.angular.z]
            self._last_rx_time = time.time()
            # Debug: log first few messages
            if not hasattr(self, "_msg_count"):
                self._msg_count = 0
            self._msg_count += 1
            if self._msg_count <= 5:
                print(
                    f"[ROS2 Bridge] Received msg #{self._msg_count}: vx={msg.linear.x:.3f}, vy={msg.linear.y:.3f}, wz={msg.angular.z:.3f}"
                )

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
                "Cannot find physics callback API. Expected env.unwrapped.sim.add_physics_callback() to be available."
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
        except Exception as e:
            # Debug: log spin errors
            if not hasattr(self, "_spin_error_logged"):
                print(f"[ROS2 Bridge] spin_once error: {e}")
                self._spin_error_logged = True

        # Copy latest command thread-safely
        with self._lock:
            cmd = list(self._last_cmd)
            rx_time = self._last_rx_time

        unwrapped = getattr(self._env, "unwrapped", self._env)
        setattr(unwrapped, self.cfg.command_attr, cmd)
        setattr(unwrapped, self.cfg.command_stamp_attr, rx_time)

        # Debug: log sync status periodically
        if not hasattr(self, "_sync_count"):
            self._sync_count = 0
        self._sync_count += 1
        if self._sync_count == 1 or self._sync_count == 100 or self._sync_count == 1000:
            print(f"[ROS2 Bridge] Sync #{self._sync_count}: cmd={cmd}, rx_time={rx_time:.2f}")

    def wait_for_first_message(self, timeout_s: float = 15.0) -> bool:
        """Block until first non-zero command arrives or timeout.

        Prints periodic diagnostics every 3 seconds so that silent hangs
        are visible in CI / batch logs.
        """
        import rclpy

        start = time.time()
        last_diag = start
        diag_interval = 3.0

        print(
            f"[ROS2 Bridge] Waiting up to {timeout_s:.0f}s for first message "
            f"on '{self.cfg.topic_name}'..."
        )

        while time.time() - start < timeout_s:
            if self._node is not None:
                rclpy.spin_once(self._node, timeout_sec=0.1)

            with self._lock:
                if any(abs(v) > 1.0e-9 for v in self._last_cmd):
                    elapsed = time.time() - start
                    print(
                        f"[ROS2 Bridge] First message received after {elapsed:.2f}s: "
                        f"cmd={self._last_cmd}"
                    )
                    return True

            now = time.time()
            if now - last_diag >= diag_interval:
                elapsed = now - start
                remaining = timeout_s - elapsed
                print(
                    f"[ROS2 Bridge] Still waiting... "
                    f"({elapsed:.1f}s elapsed, {remaining:.1f}s remaining, "
                    f"topic='{self.cfg.topic_name}')"
                )
                last_diag = now

            time.sleep(0.05)

        elapsed = time.time() - start
        print(
            f"[ROS2 Bridge] TIMEOUT after {elapsed:.1f}s — no message received "
            f"on '{self.cfg.topic_name}'. "
            f"Check: 1) WSL publisher running?  2) DDS discovery healthy?  "
            f"3) ROS_DOMAIN_ID matching?"
        )
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
