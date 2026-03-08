#!/bin/bash
# Phase 1 Baseline: ROS2 command publisher for Go1 (constant vx=1.0).
# Reuses DDS config from Phase 0 fix.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# DDS config — must match Windows side
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export FASTRTPS_DEFAULT_PROFILES_FILE="${PROJECT_ROOT}/configs/ros2/fastrtps_wsl_to_win.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE}"

echo "[Phase1 ROS2] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[Phase1 ROS2] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "[Phase1 ROS2] FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS}"
echo "[Phase1 ROS2] FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}"

set +u
source /opt/ros/humble/setup.bash
set -u

echo "[Phase1 ROS2] Starting constant vx=1.0 publisher at 50Hz..."

/usr/bin/python3 "${PROJECT_ROOT}/scripts/go1-ros2-test/ros2_nodes/go1_cmd_script_node.py" \
  --profile constant \
  --vx 1.0 \
  --vy 0.0 \
  --wz 0.0 \
  --rate 50 \
  --qos_reliability reliable \
  --qos_durability volatile \
  --qos_history_depth 5 \
  "$@"

