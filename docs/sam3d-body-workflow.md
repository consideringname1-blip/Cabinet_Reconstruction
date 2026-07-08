# SAM3D Body 人体 mesh 与拿取证据裁切

更新日期：2026-07-08
状态：当前实现说明

实现文件：`code/stages/sam3d_body_mesh/run_sam3d_body_mesh_from_json.py`。

## 目标

当 taken object detection 判断物体被拿走后，SAM3D Body stage 负责：

1. 在 Shigure taken frame 上生成人体 mesh。
2. 通过 depth 对人体 mesh 做轻量距离修正和缩放。
3. 选择最可能拿走物体的人体。
4. 保存人体 mesh、人体 bbox、people JSON。
5. 结合人体和物体位置裁切出给 HoloLens 显示的证据图。

## 输入

来自 `TakenObjectDetection`：

```text
result_rgb
result_depth
camera_info
active_objects
marker_pose
history_baseline.old_mask
```

来自模型任务：

```text
object_aruco
ModelBounds
aruco_reference
```

## 坐标链路

SAM3D Body 模型初始在 Shigure camera 坐标中。服务器转换：

```text
Shigure camera -> ArUco -> 当前 HoloLens local
```

公开 API 输出必须是：

```text
selected_person_pose_hololens
coordinate_space = hololens_current_local
```

内部可保存 `selected_person_pose_aruco` 和 ArUco mesh 文件，但 Unity 不直接消费 ArUco 坐标。

## 深度修正

当前修正只做轻量相机距离调整：

1. 用 SAM3D Body 输出的 body mask 找 mask 内 depth。
2. 对 mesh 可见区域和 depth 做抽样/中位数估计。
3. 只沿相机距离方向修正 offset。
4. 根据调整前后距离比例缩放人体 mesh。
5. 不做完整 ICP，不做重型全点云优化。

输出 debug 字段包含：

```text
depth_offset_m
distance_scale
valid_depth_pixels
method
```

## 最近人体选择

选择逻辑：

1. 将人体 keypoints/mesh 转到 ArUco。
2. 计算物体中心 `object_center_aruco`。
3. 比较左右手腕到物体中心距离。
4. 选择最近手腕对应人体。

如果没有可用手腕或物体中心，stage 返回对应失败状态，不应猜测。

## Subject Crop

裁切输出：

```text
08_sam3d_body_subject_crop.png
```

裁切 mask 由以下来源组合：

- selected body mask。
- selected body bbox fallback。
- taken baseline old_mask。
- object center projection mask。

裁切规则：

- 合并人体和物体区域。
- 取最小外接矩形。
- 加 padding。
- 输出 `subject_crop_path`。
- Unity 端按窗口最大约束等比显示。

## 输出文件

```text
result/08_sam3d_body_result.json
result/08_sam3d_body_people.json
result/08_sam3d_body_selected_person.obj
result/08_sam3d_body_selected_person.fbx
result/08_sam3d_body_subject_crop.png
```

API URL：

```json
"sam3d_body_mesh_urls": {
  "selected_person_fbx_url": "...",
  "selected_person_obj_url": "...",
  "people_url": "...",
  "subject_crop_url": "..."
}
```

## Unity 行为

- 历史证据图优先使用 `subject_crop_url`。
- 人体 mesh overlay 使用 `selected_person_pose_hololens`。
- 如果 pose 缺失，不应使用 ArUco 或 identity pose 在 Unity 端猜测。

## 不做的事

- 不在 Unity 端处理人体 ArUco 坐标。
- 不做完整 ICP。
- 不用人体 mesh 判断历史物体 still/missing；历史物体状态仍由 old_mask RGB-D direct compare 决定。
