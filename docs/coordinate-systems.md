# 坐标系统与投影规则

更新日期：2026-07-08
状态：当前实现说明

本文说明服务器、Unity、ArUco、Shigure 之间的坐标边界。核心实现位于：

- `code/coordinate_systems.py`：轴定义、基础矩阵、四元数/矩阵转换。
- `code/spatial_transforms.py`：HoloLens/ArUco/Shigure 之间的 pose、point、pixel-depth 投影/反投影。

不要在业务 stage 中新增硬编码轴翻转；需要跨系统的投影/反投影应先放到 `spatial_transforms.py`。

## 设计原则

- HoloLens2 本地不保留 ArUco 坐标数据。
- HoloLens 上传当前本地坐标。
- 服务器将 HoloLens 当前本地坐标转换到 ArUco/世界存储。
- 服务器再按当前 Unity 启动的 ArUco reference 转回 HoloLens 当前本地坐标下发。
- Unity 正式模型只消费 `object_hololens_current`。
- 同次启动、还没有 ArUco 的生成预览可以使用 HoloLens 原始坐标或 pending spatial preview，但这些数据不能用于跨启动历史。

## HoloLens 与 ArUco

存储字段：

- `object_hololens_original`：最初上传时 HoloLens 当前本地坐标，只作为同次启动恢复和 ArUco 更新源数据。
- `object_aruco`：服务器持久存储的物体 ArUco 坐标。
- `object_hololens_current`：服务器按当前 startup latest ArUco reference 转出的 Unity 下发坐标。
- `aruco_reference`：某次 startup 下 marker 在 HoloLens 当前本地坐标中的 pose。

转换函数：

```python
hololens_pose_to_aruco_pose(object_hololens_pose, aruco_reference_hololens_pose)
aruco_pose_to_hololens_pose(object_aruco_pose, aruco_reference_hololens_pose)
hololens_point_to_aruco(point_hololens, aruco_reference_hololens_pose)
aruco_points_to_hololens(points_aruco, aruco_reference_hololens_pose)
```

ArMarker 更新时，服务器用 `object_hololens_original` 和新的 `aruco_reference` 重新计算 `object_aruco`，再更新 `object_hololens_current`。retro-sync 限制在同一 `startup_session_id`。

## Shigure 与 ArUco

Shigure 的 RGB-D 和 object_detection 是固定相机视角。Shigure marker pose 由 `marker_6d_pose.json` 提供，OpenCV 表达为：

```text
marker_rotation_camera_marker_cv
marker_translation_camera_marker_cv
```

公共函数：

```python
aruco_points_to_shigure_camera(points_aruco, marker_rotation, marker_translation)
shigure_camera_points_to_aruco(points_camera_m, marker_rotation, marker_translation)
project_camera_points_to_pixels(points_camera_m, camera_matrix)
project_aruco_points_to_shigure_pixels(points_aruco, marker_rotation, marker_translation, camera_matrix)
pixel_depth_to_shigure_camera(pixel_xy, depth_m, camera_matrix)
pixel_depth_to_aruco(pixel_xy, depth_m, camera_matrix, marker_rotation, marker_translation)
```

这些函数只处理跨坐标数学。HoloLens2 的 hl2ss、本地 RGB-D 对齐、图像读取和深度图片计算仍留在各自输入边界/stage 内，不并入 `spatial_transforms.py`。

## CameraInfo

Shigure ROS camera_info 可能直接包含字段，也可能包在 `message` 内。统一解析：

```python
camera_info_message(camera_info)
camera_matrix_from_info(camera_info)
camera_info_image_shape(camera_info)
```

业务 stage 不应再次实现自己的 `K/k/camera_matrix` 正则解析。

## 预览 3D Box

`Sam3SpatialBox` 属于 pending preview：

- 输入：SAM3 mask + aligned depth + HoloLens PV camera pose。
- 输出：`coordinate_space = unity_world`。
- 深度点使用去边缘后的 mask/depth。
- 深度扩展倍数：`PREVIEW_3D_BOX_DEPTH_EXPANSION_FACTOR`，默认 `2.0`。
- 扩展方式：近相机前边不动，只向后拉远边，中心自然后移。

完成模型、历史模型、runtime 下载不能使用 `Sam3SpatialBox` 作为 fallback。

## 历史再现坐标

历史物体模型选择可以跨 startup，但显示必须回到当前 HoloLens 坐标：

1. 选择历史 completed model，要求已 `aruco_coordinate_synced`。
2. 读取历史任务的 `object_aruco` 和 bounds。
3. 用当前 startup latest ArUco reference 转成 `object_hololens_current`。
4. Unity 只显示服务器返回的 `object_hololens_current`。

历史拿取状态判断本身不依赖 3D 投影；它使用 fixed Shigure old mask 直接比较 RGB/depth。

## Unity 端约束

Unity 正式模型下载：

- `TryBuildRuntimeModelInstance` 必须解析到 `object_hololens_current`。
- `LoadModel` 不能再把缺 pose 的模型放到相机前方。
- `sam3_spatial_box` 只允许 pending preview hint 使用。

## 禁止事项

- 不要让 Unity 保存或计算 ArUco 坐标。
- 不要在 completed/history 响应里用 `sam3_spatial_box` fallback。
- 不要把 Shigure fixed view 的 old_mask 判断改回 3D 投影扫描。
- 不要在 stage 内新增重复的 camera intrinsics 解析或轴翻转。
