#!/bin/bash
# Debug ROS2 command publisher for Go1 - with verbose logging.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# CRITICAL: Set ROS_DOMAIN_ID=0 to match Windows side
export ROS_DOMAIN_ID=0
# Use FastRTPS to match Windows Isaac Sim ROS2 bridge
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# Force UDP transport; do not use SHM across the Windows/WSL boundary.
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
# Keep ROS_LOCALHOST_ONLY unset; XML profile handles asymmetric locators.
# FastRTPS unicast config for WSL2 <-> Windows communication
export FASTRTPS_DEFAULT_PROFILES_FILE="${PROJECT_ROOT}/configs/ros2/fastrtps_wsl_to_win.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE}"

echo "[DEBUG ROS2] Starting ROS2 command publisher..."
echo "[DEBUG ROS2] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[DEBUG ROS2] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "[DEBUG ROS2] FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS}"
echo "[DEBUG ROS2] ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-<not set>}"
echo "[DEBUG ROS2] FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}"
echo "[DEBUG ROS2] FASTDDS_DEFAULT_PROFILES_FILE=${FASTDDS_DEFAULT_PROFILES_FILE}"
echo "[DEBUG ROS2] QoS: reliable/volatile depth=5"

# Source ROS2 setup (disable strict unbound variable check temporarily)
set +u
source /opt/ros/humble/setup.bash
set -u

echo "[DEBUG ROS2] PROJECT_ROOT: $PROJECT_ROOT"
echo "[DEBUG ROS2] python: $(command -v /usr/bin/python3)"
echo "[DEBUG ROS2] Running go1_cmd_script_node.py..."

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
