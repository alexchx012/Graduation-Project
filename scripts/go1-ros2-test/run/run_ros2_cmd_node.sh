#!/bin/bash
# Canonical source: scripts/go1-ros2-test/run/run_ros2_cmd_node.sh
# WSL-side launcher for the Go1 ROS2 scripted command node.
#
# Usage:
#   bash run_ros2_cmd_node.sh [--profile sine] [--vx 0.5] [--rate 20] ...

set -euo pipefail

source /opt/ros/humble/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "${SCRIPT_DIR}/ros2_nodes/go1_cmd_script_node.py" "$@"
