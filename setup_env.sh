#!/usr/bin/env bash
# Source this from any shell before running the project from this checkout.

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_clean_pythonpath=""
if [ -n "${PYTHONPATH:-}" ]; then
  IFS=':' read -ra _pythonpath_parts <<< "${PYTHONPATH}"
  for _pythonpath_part in "${_pythonpath_parts[@]}"; do
    case "${_pythonpath_part}" in
      ""|"/workspace"|"/workspace/code"|"${_repo_root}"|"${_repo_root}/code") continue ;;
    esac
    _clean_pythonpath="${_clean_pythonpath:+${_clean_pythonpath}:}${_pythonpath_part}"
  done
fi
export PYTHONPATH="${_repo_root}/code:${_repo_root}${_clean_pythonpath:+:${_clean_pythonpath}}"

if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi

_ros_ws="${_repo_root}/code/ros2/shigure_recv_ws"
if [ -f "${_ros_ws}/install/setup.bash" ]; then
  source "${_ros_ws}/install/setup.bash"
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-10}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

unset _repo_root
unset _ros_ws
unset _clean_pythonpath
unset _pythonpath_parts
unset _pythonpath_part
