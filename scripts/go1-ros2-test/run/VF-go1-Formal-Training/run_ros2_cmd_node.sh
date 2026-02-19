#!/bin/bash
# Formal ROS2 command publisher for Go1 vx=1.0 tracking.

set -euo pipefail

source /opt/ros/humble/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

python3 "${PROJECT_ROOT}/scripts/go1-ros2-test/ros2_nodes/go1_cmd_script_node.py" \
  --profile constant \
  --vx 1.0 \
  --vy 0.0 \
  --wz 0.0 \
  --rate 30 \
  "$@"
