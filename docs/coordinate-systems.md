# 坐标系统与公开边界

更新日期：2026-07-12
状态：当前协议

坐标轴定义集中在 `code/coordinate_systems.py`，跨 HoloLens、ArUco 与 Shigure 的 pose/point 转换集中在 `code/spatial_transforms.py`。业务 stage 不自行添加轴翻转。

## 坐标空间

```text
windows_spatial    HoloLens/hl2ss 原始 PV pose 边界，右手，-Z forward
unity              Unity/HoloLens 公开运行时，左手，+Z forward
opencv_camera      OpenCV、ArUco、FoundationPose、Shigure camera，右手，+Z forward
canonical_rh       服务器重建内部坐标，右手，-Z forward
model_input        生成 OBJ 的源轴约定
unity_runtime_local RuntimeMesh/FBX local asset 轴约定
```

模型生成后端只影响 `model_input` 内容，不改变公开 Unity pose 契约。

## HoloLens 与 ArUco

HoloLens 上传 PV/Depth camera 的当前本地 `4x4` pose。服务器内部保存：

```text
object_hololens_original
object_aruco
object_hololens_current
```

转换函数：

```python
hololens_pose_to_aruco_pose(...)
aruco_pose_to_hololens_pose(...)
hololens_point_to_aruco(...)
aruco_points_to_hololens(...)
```

服务器根据请求方当前 `startup_session_id` 的 latest ArUco reference，把内部 `object_aruco` 转成公开：

```text
model_instance.pose
coordinate_space=hololens_current_local
```

Unity 不读取 `object_aruco`、marker pose 或服务器内部字段。

## History/Live pose

每个 `display_object_id` 持久保存两条独立 pose revision：

- `latest_hololens_pose_aruco`：最新 HoloLens 拍摄确认位置。
- `latest_tracking_pose_aruco`：当前模型 revision 下最近接受的 FoundationPose 位置。

服务器按当前 ArUco reference 转换后：

- history 模式下发 HoloLens pose。
- live 模式优先下发有效 tracking pose，否则下发 HoloLens pose。

公开 tracking item 只包含 `pose` 与 `coordinate_space=hololens_current_local`。`coordinate_epoch` 标识本次转换使用的 ArUco reference，Unity 拒绝不匹配的响应。

## Shigure 与 ArUco

Shigure RGB-D、object mask 和 contact bbox 都处于固定 OpenCV camera 图像/相机坐标。marker 文件必须提供：

```text
opencv_camera_pose.rotation_matrix
opencv_camera_pose.tvec_m
```

公共转换函数：

```python
aruco_points_to_shigure_camera(...)
shigure_camera_points_to_aruco(...)
project_camera_points_to_pixels(...)
project_aruco_points_to_shigure_pixels(...)
pixel_depth_to_shigure_camera(...)
pixel_depth_to_aruco(...)
```

用途：

- 将 HoloLens 最新 `Sam3SpatialBox` 投影到 Shigure 图像，筛选同一物体候选。
- 将 FoundationPose 的 Shigure camera pose 转成持久 ArUco pose。
- 将 SAM3D Body 顶点从 Shigure camera 转成 ArUco mesh。

## CameraInfo

Shigure cache 的 CameraInfo 结构为：

```json
{
  "k": [9个有限数值],
  "width": 1280,
  "height": 720
}
```

`k` 必须可 reshape 为 `3x3` 且焦距为正，width/height 必须与 RGB-D 完全一致。使用：

```python
camera_matrix_from_info(camera_info)
camera_info_image_shape(camera_info)
```

## Pending spatial box

`Sam3SpatialBox` 同时承担两个明确边界内的用途：

- Unity 当前任务生成中的临时 `unity_world` preview。
- 服务器把最新 HoloLens capture box 转到 ArUco 后投影到 Shigure 图像，进行实时身份几何过滤。

它不直接作为 completed/runtime 模型公开 pose，也不能替代 FoundationPose 结果。

## Runtime asset 与缩放

`RuntimeMesh` 保留 `model_input` 轴，pose stage 使用 `RUNTIME_LOCAL_TO_UNITY_POSE_ROTATION` 生成 Unity runtime orientation。FBX 加载后的缩放属于模型资产状态；history/live/realtime 更新只设置 position 和 rotation，不重新应用服务器 scale。

## 校验清单

- `/generate` camera pose 是否为有限数值 `4x4`。
- Shigure CameraInfo、RGB、depth 尺寸是否一致。
- marker pose 是否使用 OpenCV camera 结构。
- 公共 model、tracking、body pose 是否都标记 `hololens_current_local`。
- Unity 是否按 `coordinate_epoch` 和 revision 拒绝过期位置。
- 业务 stage 是否复用了 `spatial_transforms.py`，没有增加局部轴翻转。
