#!/bin/bash
# Formal ROS2 command publisher for Go1 vx=1.0 tracking.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Keep ROS2 runtime aligned with Windows-side Isaac Sim bridge.
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export FASTRTPS_DEFAULT_PROFILES_FILE="${PROJECT_ROOT}/configs/ros2/fastrtps_wsl_to_win.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE}"

echo "[FORMAL ROS2] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[FORMAL ROS2] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "[FORMAL ROS2] FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS}"
echo "[FORMAL ROS2] FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}"
echo "[FORMAL ROS2] QoS: reliable/volatile depth=5"

set +u
source /opt/ros/humble/setup.bash
set -u

echo "[FORMAL ROS2] python: $(command -v /usr/bin/python3)"

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
