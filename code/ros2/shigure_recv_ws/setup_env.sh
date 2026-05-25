#!/usr/bin/env bash
# Source this before running ros2 echo/subscribers for Shigurei data.
source /opt/ros/humble/setup.bash
if [ -f /workspace/code/ros2/shigure_recv_ws/install/setup.bash ]; then
  source /workspace/code/ros2/shigure_recv_ws/install/setup.bash
fi
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-10}
export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}
