# 服务器输入输出契约

更新日期：2026-07-12
状态：当前协议

服务器只接受本文列出的字段名、类型和坐标空间。模型生成后端可选 InstantMesh 或 SAM3D Objects，但两者写入相同的公开契约。

## Artifact 下载

模型任务 artifact 使用：

```text
/task-artifacts/<task_id>/worker/<filename>
/task-artifacts/<task_id>/result/<filename>
/task-artifacts/<task_id>/debug/<filename>
```

客户端只保存服务器返回的 URL，不自行拼接服务器文件路径。

## `POST /generate`: object reconstruction

multipart form 必须恰好包含：

```text
purpose
deviceJ
PVCameraJ
DepthCameraJ
SelectionBoxJ
force_new_3d_model
```

文件必须恰好包含：

```text
pv_image
depth_image
```

字段值：

```text
purpose = object_reconstruction
force_new_3d_model = 0 | 1
```

JSON 字段结构：

```json
{
  "deviceJ": {
    "ip": "10.40.1.132",
    "startup_session_id": "unity-startup-uuid"
  },
  "PVCameraJ": {
    "width": 1920,
    "height": 1080,
    "k": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "pose": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "time": "2026-07-12T12:00:00.0000000Z"
  },
  "DepthCameraJ": {
    "pose": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "sensor": "AHAT"
  },
  "SelectionBoxJ": {
    "top_left": [0.25, 0.25],
    "bottom_right": [0.75, 0.75]
  }
}
```

约束：

- `deviceJ.ip` 必须是 IPv4。
- `PVCameraJ.k` 为有限数值 `3x3`；`PVCameraJ.pose` 和 `DepthCameraJ.pose` 为有限数值 `4x4`。
- `DepthCameraJ.sensor` 只能是 `AHAT` 或 `LONGTHROW`。
- selection 坐标在 `[0,1]`，且 `bottom_right` 必须位于 `top_left` 右下方。
- `depth_image` 必须是单通道 uint16 PNG；服务器按对应传感器可靠范围清洗。

响应：

```json
{"task_id": "server-generated-uuid"}
```

`task_id` 只由服务器生成。`force_new_3d_model=1` 表示即使命中同一 `display_object_id` 也生成新模型 revision。

## `POST /generate`: ArUco reference

multipart form 必须恰好包含：

```text
purpose
deviceJ
PVCameraFramesJ
```

```text
purpose = aruco_reference
```

`deviceJ` 只包含：

```json
{"startup_session_id": "unity-startup-uuid"}
```

`PVCameraFramesJ` 是非空数组。每帧必须恰好包含 `width`、`height`、`k`、`pose`、`time`。文件名与数组下标一一对应：

```text
pv_image_0
pv_image_1
...
```

ArUco 任务完成后保存该 startup 的 latest reference，并 retro-sync 同一 startup 的模型任务。

## `POST /check-queue`

请求：

```json
{
  "task_ids": ["task-uuid"],
  "startup_session_id": "unity-startup-uuid"
}
```

未完成时返回：

```json
{
  "ready": false,
  "pending": [
    {
      "task_id": "task-uuid",
      "status": "model_generation",
      "purpose": "object_reconstruction",
      "terminal": false,
      "stage_name": "model_generation",
      "stage_index": 3,
      "stage_count": 12,
      "progress": 0.25,
      "progress_text": "model_generation 25%"
    }
  ]
}
```

`sam3mask` 完成后，pending item 可额外包含 `sam3_spatial_box`，仅用于生成中预览。

模型任务进入唯一终态 `completed` 时，`task.model_instance` 必须恰好包含 7 个字段：

```json
{
  "model_key": "task-uuid",
  "task_id": "task-uuid",
  "display_object_id": "display-object-uuid",
  "model_revision": 2,
  "fbx_url": "http://server/task-artifacts/task-uuid/result/05_export_final.fbx",
  "pose": {
    "position": [0, 0, 0],
    "rotation_quaternion_xyzw": [0, 0, 0, 1],
    "scale": [1, 1, 1]
  },
  "coordinate_space": "hololens_current_local"
}
```

服务器不提供中间下载状态；Unity 只在 `completed` 响应中按 `model_instance` 加载正式模型。

## `POST /realtime-tracking/mode`

请求必须恰好包含：

```json
{
  "startup_session_id": "unity-startup-uuid",
  "mode": "history",
  "request_generation": 4
}
```

`mode` 只能是 `history` 或 `live`。`request_generation` 为非负整数。

响应包含 mode 状态和最多 5 个 item：

```json
{
  "success": true,
  "startup_session_id": "unity-startup-uuid",
  "mode": "history",
  "mode_epoch": 3,
  "request_generation": 4,
  "ingress_session_id": "server-ingress-uuid",
  "coordinate_epoch": "aruco-task-uuid",
  "count": 1,
  "items": []
}
```

## `GET /realtime-tracking/status`

query 必须恰好出现一次：

```text
startup_session_id=<unity-startup-uuid>
```

每个 `items[]` 的位置部分：

```json
{
  "display_object_id": "display-object-uuid",
  "model_revision": 2,
  "active_model_task_id": "task-uuid",
  "pose_source": "hololens",
  "pose_revision": 8,
  "pose": {
    "position": [0, 0, 0],
    "rotation_quaternion_xyzw": [0, 0, 0, 1],
    "scale": [1, 1, 1]
  },
  "coordinate_space": "hololens_current_local",
  "body_revision": 1
}
```

- history 模式的 `pose_source` 为 `hololens`，使用最新 HoloLens 拍摄确认位置。
- live 模式存在有效追踪结果时使用 `tracking`，否则仍使用 `hololens`。
- `model` 存在时，其结构与 canonical `model_instance` 完全相同，且 `model.pose` 必须等于 item 的 `pose`。客户端若已有相同 `display_object_id + model_revision` 的本地 FBX，则直接复用缓存。

## Canonical body evidence

有成功的人体证据时，tracking item 附带：

```json
{
  "body_evidence": {
    "task_id": "task-uuid",
    "display_object_id": "display-object-uuid",
    "body_revision": 1,
    "image_url": "http://server/task-artifacts/task-uuid/result/08_sam3d_body_subject_crop.png",
    "body_model": {
      "model_key": "body:display-object-uuid",
      "task_id": "task-uuid",
      "display_object_id": "display-object-uuid",
      "model_revision": 1,
      "fbx_url": "http://server/task-artifacts/task-uuid/result/08_sam3d_body_selected_person.fbx",
      "pose": {
        "position": [0, 0, 0],
        "rotation_quaternion_xyzw": [0, 0, 0, 1],
        "scale": [1, 1, 1]
      },
      "coordinate_space": "hololens_current_local"
    }
  }
}
```

`body_evidence` 必须包含裁切图和人体 FBX，两者 revision 必须与 item 的 `body_revision` 一致。

## 坐标与缩放规则

- 所有 Unity 正式位姿都使用 `hololens_current_local`。
- 服务器内部可保存 ArUco 位姿，但不通过 model/tracking/body 公共对象下发。
- Unity 应用 tracking/history 位姿时只更新 position 和 rotation；模型缩放保持加载时的值。
