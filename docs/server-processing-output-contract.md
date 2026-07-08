# 服务器输入输出契约

更新日期：2026-07-08
状态：当前实现说明

本文记录 API、JSON 字段和稳定 artifact 契约。算法细节见坐标、Shigure、拿取判断、历史再现和 SAM3D Body 专题文档。

## 运行入口

```bash
cd /workspace_whs
python code/run_server.py
```

配置拆分：

- `code/config.py`：运行开关、阈值、worker 参数。
- `code/path_config.py`：代码路径、stage 脚本、Python/Blender 可执行文件。
- `code/artifact_layout.py`：数据目录和 artifact 命名。
- `code/task_db.py`：SQLite schema 和查询。

## 稳定下载入口

所有新稳定 artifact 使用 task-local endpoint：

```text
/task-artifacts/<task_id>/worker/<filename>
/task-artifacts/<task_id>/result/<filename>
/task-artifacts/<task_id>/debug/<filename>
```

Unity 不应直接拼 `data/` 文件路径，也不应使用旧 `/files/<folder>/<filename>` 兼容路径。

## `/generate`: object reconstruction

Multipart form：

```text
purpose=object_reconstruction
startup_session_id 或 deviceJ.startup_session_id
deviceJ
PVCameraJ 或 PVCameraFramesJ
depthCameraJ / DepthCameraJ
SelectionBoxJ
pv_image
depth_image
force_new_3d_model   # 可选；为 true 时即使命中历史模型也强制新生成
```

服务行为：

1. 创建 `task_id` 和 `task_timestamp`。
2. DB 写入 `uploading`。
3. 写入 `data/model/<task_timestamp>/task.json` 和 `worker/01_*`。
4. 状态切到 `pending`，进入 worker 队列。

关键输入字段：

```json
{
  "task_id": "uuid",
  "task_timestamp": "20260708_...",
  "purpose": "object_reconstruction",
  "device": {"startup_session_id": "..."},
  "PVCamera": {"position": [0, 0, 0], "rotation_quaternion_xyzw": [0, 0, 0, 1]},
  "DepthCamera": {"sensor": "AHAT"},
  "SelectionBox": {}
}
```

## `/generate`: aruco reference

Multipart form：

```text
purpose=aruco_reference
deviceJ.startup_session_id
PVCameraFramesJ
pv_image 或 pv_image_0, pv_image_1, ...
```

输出写入：

```text
data/aruco_processing/<task_timestamp>/
  task.json
  worker/<frame_timestamp>_color.png
  worker/<frame_timestamp>_meta.json
  result/summary.json
  result/<frame_timestamp>_marker_detect.json
  debug/<frame_timestamp>_aruco_debug_overlay.png
```

ArUco reference 会写入 DB，并触发同次 `startup_session_id` 的 model retro-sync。

## Completed Model Response

完成模型响应必须提供服务器转换后的当前 HoloLens 本地位姿：

```json
{
  "success": true,
  "status": "completed",
  "task_id": "...",
  "fbx_url": ".../task-artifacts/<task_id>/result/05_export_final.fbx",
  "model_instance": {
    "model_key": "...",
    "task_id": "...",
    "fbx_url": "...",
    "object_hololens_current": {
      "position": [0, 0, 0],
      "rotation_quaternion_xyzw": [0, 0, 0, 1],
      "scale": [1, 1, 1]
    }
  },
  "object_hololens_current": {"position": [], "rotation_quaternion_xyzw": [], "scale": []},
  "object_hololens_original": null,
  "coordinate_space": "hololens_current_local"
}
```

规则：

- `object_hololens_current` 是 Unity 正式显示唯一可信位姿。
- 非同次启动历史模型必须由 `object_aruco` + 当前 startup 的 latest ArUco reference 转换得到。
- `object_hololens_original` 只在请求 startup 与任务 startup 相同时下发。
- 缺 `object_hololens_current` 时，Unity 不下载/不显示正式模型。

## Pending Preview Response

pending 阶段可以返回 `sam3_spatial_box`：

```json
{
  "status": "sam3mask",
  "sam3_spatial_box": {
    "coordinate_space": "unity_world",
    "aabb_min_world": [],
    "aabb_max_world": [],
    "depth_expansion_factor": 2.0
  }
}
```

`sam3_spatial_box` 仅用于生成中预览。完成、历史、runtime 下载链路不得使用它作为 fallback。

## Taken Object Detection 输出

`TakenObjectDetection` 写入任务 JSON，并发布 result URLs：

```json
{
  "status": "TAKEN",
  "history_baseline": {
    "old_rgb_path": ".../rgb.png",
    "old_depth_path": ".../depth.png",
    "old_mask_path": ".../old_mask.png",
    "camera_info_path": ".../camera_info.json",
    "coordinate_space": "fixed_shigure_image"
  },
  "result_rgb": "07_taken_detection_result_rgb.png",
  "result_depth": "07_taken_detection_result_depth.png",
  "camera_info": "07_taken_detection_camera_info.json",
  "active_objects": "07_taken_detection_active_objects.json",
  "marker_pose": "07_taken_detection_marker_6d_pose.json"
}
```

服务器响应中对应：

```json
"taken_object_detection_urls": {
  "result_rgb_url": "...",
  "result_depth_url": "...",
  "camera_info_url": "...",
  "active_objects_url": "...",
  "marker_pose_url": "..."
}
```

## SAM3D Body 输出

```json
"sam3d_body_mesh": {
  "status": "SUCCESS",
  "coordinate_space": "hololens_current_local",
  "selected_person_pose_hololens": {},
  "selected_person_bbox_xyxy": [],
  "subject_crop_path": "..."
}
```

```json
"sam3d_body_mesh_urls": {
  "selected_person_fbx_url": "...",
  "selected_person_obj_url": "...",
  "people_url": "...",
  "subject_crop_url": "..."
}
```

Unity 历史证据窗口优先显示 `subject_crop_url`，缺失时才退回 `taken_object_detection_urls.result_rgb_url`。

## History Placement Response

`POST /history-placement-restoration/start` 输入：

```json
{
  "startup_session_id": "current-unity-startup",
  "task_id": "optional specific model",
  "model_limit": 20,
  "target_time": "optional"
}
```

响应 item：

```json
{
  "task_id": "...",
  "status": "ORIGINAL|MISSING|OCCLUDED_REUSE_LAST|UNKNOWN",
  "history_placement_restoration": {
    "display": {
      "show_model": false,
      "polyhedron": {
        "shape": "octahedron|cube|tetrahedron|dodecahedron",
        "pose_hololens": {}
      }
    },
    "direct_compare": {}
  },
  "model_instance": {},
  "object_hololens_current": {},
  "taken_object_detection_urls": {},
  "sam3d_body_mesh_urls": {}
}
```

历史再现不显示重叠模型，只显示状态 polyhedron 和裁切证据图。

## 配置项

常用环境变量：

```text
SHIGURE_HISTORY_RECORDING_ENABLE=1
HISTORICAL_MODEL_REUSE_ENABLE=1
FORCE_NEW_3D_MODEL=0
DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD=0.20
PREVIEW_3D_BOX_DEPTH_EXPANSION_FACTOR=2.0
TAKEN_OBJECT_TRACKING_MODE=model_diag_circle
TAKEN_OBJECT_MODEL_DIAG_CIRCLE_MIN_MASK_INSIDE_RATIO=0.80
TAKEN_OBJECT_MODEL_DIAG_CIRCLE_MAX_DEPTH_DIFF_M=0.18
HISTORY_PLACEMENT_DIRECT_COMPARE_DEPTH_DELTA_M=0.12
```

新字段应优先加到 `config.py` 或 stage-local `settings.py`，不要在 stage 内写死阈值。
