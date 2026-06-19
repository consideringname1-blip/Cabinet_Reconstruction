# Shigurei ROS2 Receive Workspace

This workspace is only for receiving Shigurei ROS2 data. It builds local message definitions so this project can deserialize Shigurei topics. It does not launch `shigure_core`.

Build:

```bash
cd /workspace_whs/code/ros2/shigure_recv_ws
source ./setup_env.sh
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
```

Use:

```bash
source /workspace_whs/code/ros2/shigure_recv_ws/setup_env.sh
ros2 topic list -t
ros2 topic echo /shigure/contacted
```

Generated `build/`, `install/`, and `log/` directories are ignored by git.
