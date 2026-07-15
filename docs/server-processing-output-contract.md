# 服务器输入输出契约

更新日期：2026-07-14
状态：Shigure v2 历史说明（v3 live/history 已取代相关章节）

> 模型任务 artifact 的通用部分仍可参考，但 Shigure live/history 当前契约不再使用 raw tracking box、`kind=take_out` 或历史照片/骨骼。请以 [`shigure-v3-runtime.md`](shigure-v3-runtime.md) 为准。

服务器只接受本文列出的字段名、类型和坐标空间。模型生成后端可选 InstantMesh 或 SAM3D Objects，但两者写入相同的公开模型契约。

## Artifact 下载

模型任务 artifact 使用：

```text
/task-artifacts/<task_id>/worker/<filename>
/task-artifacts/<task_id>/result/<filename>
/task-artifacts/<task_id>/debug/<filename>
```

Shigure 生命周期事件的证据图片使用：

```text
/shigure-event-artifacts/<event_directory>/<filename>
```

`event_directory` 对应服务器持久目录 `data/shigure_events/<event_uuid>/`；不存在 runtime/epoch 的额外路径层级。

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

文件必须恰好包含 `pv_image` 和 `depth_image`。字段值：

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
    "time": "2026-07-14T12:00:00.0000000Z"
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
- `PVCameraJ.k` 为有限数值 `3x3`；两个 camera pose 为有限数值 `4x4`。
- `DepthCameraJ.sensor` 只能是 `AHAT` 或 `LONGTHROW`。
- selection 坐标在 `[0,1]`，且 `bottom_right` 位于 `top_left` 右下方。
- `depth_image` 必须是单通道 uint16 PNG；服务器按对应传感器可靠范围清洗。

响应：

```json
{"task_id": "server-generated-uuid"}
```

`task_id` 只由服务器生成。`force_new_3d_model=1` 表示即使命中同一 `display_object_id` 也生成新模型 revision。

## `POST /generate`: ArUco reference

multipart form 必须恰好包含 `purpose`、`deviceJ`、`PVCameraFramesJ`，其中：

```text
purpose = aruco_reference
```

`deviceJ` 只包含：

```json
{"startup_session_id": "unity-startup-uuid"}
```

`PVCameraFramesJ` 是非空数组，每帧必须恰好包含 `width`、`height`、`k`、`pose`、`time`。文件名与数组下标一一对应为 `pv_image_0`、`pv_image_1` 等。任务完成后服务器保存该 startup 的 latest reference，并 retro-sync 同一 startup 的模型任务。

## `POST /check-queue`

请求：

```json
{
  "task_ids": ["task-uuid"],
  "startup_session_id": "unity-startup-uuid"
}
```

未完成时返回 task status、stage、progress 等字段。`sam3mask` 完成后 pending item 可包含 `sam3_spatial_box`，它只用于生成中的 HoloLens 当前任务预览。

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

仅由 HoloLens capture 新建的 display object 初始为 `presence=UNKNOWN`。这里返回的 completed `model_instance` 是 capture preview/模型交付，不表示 Shigure 已确认物体在场；查询 `/check-queue` 本身也不改变 presence。后续 identity sync 只有在稳定且 raw ID 可信的 Shigure recovery snapshot 中匹配成功，才可由该 snapshot 建立 binding 并激活 `PRESENT`。

completed task 还包含交付角色和可选同步摘要，例如：

```json
{
  "delivery_role": "capture_preview",
  "shigure_identity_sync": {
    "status": "PENDING",
    "sync_job_id": "0123456789abcdef0123456789abcdef",
    "display_object_id": "display-object-uuid",
    "task_id": "task-uuid",
    "reason": "waiting_for_stable_recovery_frame",
    "attempts": 0,
    "updated_utc": "2026-07-14T12:00:00+00:00",
    "status_url": "http://server/api/v2/identity-sync/0123456789abcdef0123456789abcdef"
  }
}
```

模型任务的 `terminal=true` 与辅助 identity sync 独立；因此模型已可预览时，sync 仍可为 `PENDING`。公共同步状态为 `PENDING | COMPLETED | FAILED`。成功结果的 `method` 为 `ARUCO_COLLIDER_GEOMETRY` 或 `DINOV2_FALLBACK`，并可包含 `candidate_id`、`raw_shigure_object_id`、`geometry`、`dinov2`、`identity_reference_id`、`binding_id`、`binding_established` 和 `presence_activated`。`lifecycle_authority=shigure_recovery_snapshot` 表示是 Shigure 稳定快照授权；`lifecycle_binding_changed` 明确报告本次是否建立 binding 或激活 presence。

## `GET /api/v2/identity-sync/<sync_job_id>`

`status_url` 指向持久 job 查询接口；不接受 query 或 request body。响应：

```json
{
  "sync_job_id": "0123456789abcdef0123456789abcdef",
  "kind": "HOLOLENS_CAPTURE",
  "status": "COMPLETED",
  "display_object_id": "display-object-uuid",
  "result": {
    "status": "COMPLETED",
    "method": "ARUCO_COLLIDER_GEOMETRY",
    "candidate_id": "candidate-id",
    "raw_shigure_object_id": "epoch-local-raw-id",
    "lifecycle_authority": "shigure_recovery_snapshot",
    "lifecycle_binding_changed": true
  },
  "error_message": null,
  "updated_at": "2026-07-14T12:00:00+00:00",
  "terminal": true
}
```

后续状态以该 GET 的持久 job 为准：

- 排队时 Shigure runtime 不可用会立即 `FAILED`，reason 为 `shigure_runtime_unavailable`。
- runtime 主动停止会把内存中的 pending job 终结为 `FAILED`，reason 为 `runtime_stopped_before_sync`。
- 新服务器启动会把数据库中遗留的 `PENDING/RUNNING` job 终结为 `FAILED`，reason 为 `server_restart`。
- `COMPLETED` 与 `FAILED` 的 `terminal=true`；未知 job 返回 404，非法 job ID 返回 400。

## `POST /realtime-tracking/mode`

此接口只做 live transport 握手。请求必须恰好包含：

```json
{
  "startup_session_id": "unity-startup-uuid",
  "mode": "live",
  "request_generation": 4
}
```

`request_generation` 为非负整数。`mode=history` 已废止并返回 400；历史位置只是 Unity 的局部呈现状态，不会停止服务端接收、计算或保存 live 数据。

响应：

```json
{
  "success": true,
  "startup_session_id": "unity-startup-uuid",
  "mode": "live",
  "mode_epoch": 3,
  "request_generation": 4,
  "coordinate_epoch": "aruco-task-uuid",
  "count": 0,
  "items": [],
  "tracking_box_snapshot_complete": true,
  "tracking_box_count": 0,
  "tracking_boxes": []
}
```

`POST /realtime-tracking/mode` 的成功响应与 status 使用相同模型容量语义：`items` 最多 5 个。响应中保留的 tracking box 字段只用于旧客户端兼容，不再是当前 Unity 的 box 数据源。

## `GET /realtime-tracking/status`

query 必须恰好出现一次：

```text
startup_session_id=<unity-startup-uuid>
```

调用前必须完成当前 startup 的 live 握手。`items` 是当前模型与 live pose 数据源，仍最多 5 个。下例中的 `tracking_box_*` 字段暂为旧客户端兼容字段；当前 Unity 不读取它们，也不从其 complete/count/revision 推导 raw box 状态。

```json
{
  "success": true,
  "startup_session_id": "unity-startup-uuid",
  "mode": "live",
  "mode_epoch": 3,
  "request_generation": 4,
  "coordinate_epoch": "aruco-task-uuid",
  "count": 1,
  "items": [
    {
      "display_object_id": "display-object-uuid",
      "model_revision": 2,
      "active_model_task_id": "task-uuid",
      "pose_source": "tracking",
      "pose_revision": 8,
      "pose": {
        "position": [0, 0, 0],
        "rotation_quaternion_xyzw": [0, 0, 0, 1]
      },
      "coordinate_space": "hololens_current_local",
      "presence": "PRESENT",
      "presence_epoch": 7
    }
  ],
  "tracking_box_snapshot_complete": true,
  "tracking_box_count": 1,
  "tracking_boxes": [
    {
      "tracking_id": "source-epoch:raw-tracking-id",
      "revision": 812,
      "spatial_box": {
        "status": "ready",
        "coordinate_space": "hololens_current_local",
        "revision": 30064771080,
        "corners_hololens_current_local_m": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
      }
    }
  ]
}
```

原始 tracking box 使用独立接口：

- `GET /api/v2/shigure/object-tracking-boxes/latest?startup_session_id=<uuid>`；它不依赖 realtime handshake、模型状态、历史模式或下载队列。
- Unity 从应用启动后长期约每 1 秒轮询。每次成功响应都是权威最新快照：同 tracking_id 直接覆盖，新增 ID 立即增加，响应中缺失的旧 ID 立即删除，成功空数组会清空全部线框；网络失败仅保留上一帧等待下次轮询。
- recorder 在每条 ROS object_tracking callback 内立即把当前 collider 完整快照原子写入 relay 文件；此路径不经过 runtime、DINOv2、FoundationPose、模型或启动恢复。API 只读取该最新快照并转换 Shigure-camera 到当前 HoloLens-local 坐标，不做 freshness、稳定性、平滑、debounce 或整批等待校验。
- tracking_id 直接使用当前 raw_id，仅用于原始线框显示；Unity 只画 8 点线框，不把它绑定到 RuntimeModelRecord，也不用于 PointObject、identity、pose 或其他操作。
- 当前 startup 没有 ArMarker reference 时接口成功返回空数组，因为无法安全完成坐标变换。
- raw Shigure ID 与持久物体身份严格分离。旧 ID 意外消失、随后在配置时间窗内出现新 ID 时，runtime 会另开 source epoch，并以 DINOv2 对现有 display identity 做一对一相似度恢复；这不会改变上述 raw box 的逐 ID 直接显示行为。

原 realtime status 的 `items[]` 继续只承载模型与 live pose；其中可附带 `tracking_status`、`tracking_event_uid` 和 canonical `model`。客户端已有相同 `display_object_id + model_revision` 的 FBX 时直接复用缓存。

## `GET /api/v2/display-objects/<display_object_id>/history`

query：

```text
kind=take_out
limit=1..20
startup_session_id=<unity-startup-uuid>
before_cursor=<正整数，可选>
```

当前 Unity 每次请求一条；继续点击同一物体时，把上一条 `history_cursor` 作为 `before_cursor` 取得更老记录。响应：

```json
{
  "success": true,
  "coordinate_space": "hololens_current_local",
  "coordinate_epoch": "aruco-task-uuid",
  "history_event": {
    "display_object_id": "display-object-uuid",
    "event_uid": "lifecycle-event-uid",
    "history_cursor": "42",
    "pose": {
      "position": [0, 0, 0],
      "rotation_quaternion_xyzw": [0, 0, 0, 1]
    },
    "spatial_box": null,
    "evidence": {
      "scene_image_url": "http://server/shigure-event-artifacts/event/image.jpg",
      "skeleton": {
        "people_id": "person-id",
        "joints": [
          {"name": "nose", "position": [0, 0, 0], "score": 0.9, "valid": true}
        ]
      }
    }
  }
}
```

服务端会以 100 行为内部页继续扫描，只跳过缺 pose 或坐标转换失败的行；scene image、骨骼和 spatial box 都是可选增强，缺失时 `evidence` 或 `spatial_box` 为 `null`，不能阻止历史模型再放置。`history_cursor` 始终取实际返回的有效 lifecycle row，下一次 `before_cursor` 从它继续向更老记录查找。

真正耗尽后 `history_event` 才为 `null`。证据存在时照片和 Shigure 点线骨骼随历史位置显示；证据缺失时只做模型再放置。协议不包含人体 mesh 或独立 body revision。

注意：历史事件自身的 `spatial_box` 仍可为 `null`；live 的显式 `status=no_box` 语义不改变已持久化历史证据。
