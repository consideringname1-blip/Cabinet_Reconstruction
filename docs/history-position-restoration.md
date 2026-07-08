# 历史位置再现

更新日期：2026-07-08
状态：当前实现说明

历史位置再现用于判断历史物体是否仍在原位、被拿走、被遮挡或未知。当前设计固定使用 Shigure 视角直接比较，不做 3D 投影扫描。

## 触发

Unity 调用：

```text
POST /history-placement-restoration/start
GET  /history-placement-restoration/latest
```

start 请求必须包含当前 Unity 启动的 `startup_session_id`。服务器用它把历史 ArUco pose 转回当前 HoloLens 本地坐标。

## 模型选择

如果传入 `task_id`：

- 只对指定模型做历史再现。

如果不传 `task_id`：

- 全局选择 latest completed models。
- 要求 `aruco_coordinate_synced = 1`。
- 每个 display object 只用最新模型。
- 不按当前 startup 限制历史模型来源，因为跨启动历史正是通过 ArUco 回转当前 HoloLens 坐标实现。

## Baseline 来源

历史再现只接受 taken stage 保存的 explicit baseline：

```text
TakenObjectDetection.history_baseline
```

baseline 包含：

```text
old_rgb_path
old_depth_path
old_mask_path
old_reference_depth_m_path
camera_info_path
coordinate_space=fixed_shigure_image
```

如果缺失 baseline 或路径不可读，历史再现不启用旧 YOLO fallback。

## Direct Compare

在 old_mask 内比较当前 Shigure frame：

```text
depth_delta = current_depth - old_depth
depth_changed_ratio = |depth_delta| > 0.12 的比例
depth_nearer_ratio  = current_depth < old_depth - 0.12 的比例
depth_farther_ratio = current_depth > old_depth + 0.12 的比例
rgb_changed_ratio   = Lab delta > threshold 的比例
```

默认阈值：

```text
HISTORY_PLACEMENT_DIRECT_COMPARE_DEPTH_DELTA_M=0.12
HISTORY_PLACEMENT_DIRECT_COMPARE_OCCLUDED_NEARER_RATIO=0.25
HISTORY_PLACEMENT_DIRECT_COMPARE_MISSING_FARTHER_RATIO=0.40
HISTORY_PLACEMENT_DIRECT_COMPARE_STILL_MAX_DEPTH_CHANGED_RATIO=0.20
HISTORY_PLACEMENT_DIRECT_COMPARE_RGB_LAB_DELTA_THRESHOLD=18
HISTORY_PLACEMENT_DIRECT_COMPARE_STILL_MAX_RGB_CHANGED_RATIO=0.35
```

判断顺序：

1. 遮挡：`depth_nearer_ratio` 足够高。
2. 不在原位/拿走：`depth_farther_ratio` 足够高。
3. 仍在原位：depth 和 RGB 变化都足够低。
4. 其他：unknown。

## 状态与显示

状态输出：

```text
ORIGINAL
MISSING
OCCLUDED_REUSE_LAST
UNKNOWN
```

Unity 显示策略：

```text
OCCLUDED_REUSE_LAST -> tetrahedron
MISSING / moved     -> cube
ORIGINAL            -> octahedron
UNKNOWN             -> dodecahedron
```

当前不显示重叠模型：

```json
"display": {
  "show_model": false,
  "polyhedron": {"shape": "cube", "pose_hololens": {}}
}
```

## 证据图

服务端优先返回 SAM3D Body 裁切图：

```text
sam3d_body_mesh_urls.subject_crop_url
```

Unity 读取顺序：

1. `subject_crop_url`
2. `taken_object_detection_urls.result_rgb_url`

显示时按原窗口最大约束放大，保持裁切图长宽比。

## 坐标输出

历史 item 中的所有公开坐标必须是当前 HoloLens 本地坐标：

- `object_hololens_current`
- `history_placement_restoration.display.polyhedron.pose_hololens`
- `sam3d_body_mesh.selected_person_pose_hololens`

内部 `pose_aruco` 可以存在于任务 JSON，但 API 公开响应通过 `_public_spatial_payload` 转换。

## 不再使用

- 不再从 current object_detection 重新识别物体是否存在。
- 不再用 projected bbox 扫全图判断 still/missing。
- 不再在 Unity 端用 `sam3_spatial_box` fallback 显示历史模型。
