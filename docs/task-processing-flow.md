# 任务处理流程

更新日期：2026-07-12
状态：当前协议

本文描述服务器、远端 Shigure 数据入口和 Unity 使用的唯一任务流程。

## 核心标识

- `task_id`：一次上传任务的 UUID，也是 API 与数据库主键。
- `task_timestamp`：该任务的 artifact 目录名。
- `startup_session_id`：一次 HoloLens/Unity 启动的会话边界。
- `display_object_id`：跨任务持久化的物体身份。
- `model_revision`：同一 `display_object_id` 下的模型版本。
- Shigure 的 `object_id` 只在当前 ingress/startup 内作为临时绑定键，不进入持久身份。

## Artifact 布局

目录由 `code/artifact_layout.py` 统一生成：

```text
data/
  model/<task_timestamp>/
    task.json
    worker/
    result/
    debug/
    logs/
  aruco_processing/<task_timestamp>/
    task.json
    worker/
    result/
    debug/
  realtime_tracking/<display_object_id>/pose_events.jsonl
  database/tasks.db
  worker_sockets/
  shigure_history_cache/
  aruco/
  console_logs/
```

任务 JSON 只从数据库记录的 `json_path` 解析。业务代码不扫描其他目录猜测任务文件。

## Object reconstruction 主链

`task_worker.STAGE_ORDER` 的当前顺序为：

```text
hololens2depth
sam3mask
historical_model_match
model_generation
depthpointcloud
modelscale
object_alignment
pose
aruco_sync
runtime_mesh
model_bounds
display_identity
```

`model_generation` 是统一模型生成阶段。`code/config.py` 的 `MODEL_GENERATION_BACKEND` 只能是：

- `instantmesh`
- `sam3d_objects`

DINOv2 命中同一 `display_object_id` 且 `force_new_3d_model=0` 时，当前任务复用该物体最新 completed 模型资产，跳过不再需要的生成、缩放和 runtime bake；当前拍摄仍重新计算点云、对齐、位姿、ArUco 同步、bounds 和身份记录。`force_new_3d_model=1` 时生成新模型版本并保留已有版本。

`object_alignment` 的 HoloLens 拍摄任务使用高优先级 FoundationPose 请求。实时追踪请求进入独立的低优先级 latest-wins 通道。

## Shigure contact/body 辅助分支

物体上传后，`shigure_contact_body` 与主链并行启动，不占用主链 stage 顺序：

1. recorder 从远端 Shigure 数据流接收 RGB、uint16 depth、CameraInfo、`/shigure/object_detection` 和 `/shigure/contacted`。
2. `object_detection` 与 `contacted` 按完全相同的 ROS source stamp 组成事件。
3. 仅接受存在 `take_out` contact、存在对应 object mask，且 contact 与 detection 关联唯一的事件。
4. 使用当前 HoloLens 拍摄的 DINO embedding 校验该 Shigure mask 对应同一 `display_object_id`。
5. 输出 `ShigureContactEvidence`；没有被检测到联系人时输出明确的 wrong/input 状态，不做无人体回退。
6. `ShigureContactEvidence.status=TAKEN` 时，调用 `sam3d_body_mesh` 生成该对象的一份最新人体证据。

服务器停止时，辅助等待和实时追踪内存状态会清空；模型、HoloLens 拍摄位姿、模型版本、人体版本和 FoundationPose 位置日志保留在数据库及 artifact 中。

## 实时位置追踪

实时追踪最多激活最近 5 个不同的 `display_object_id`：

1. `obj_move` 或 `bring_in` 事件到达后，将 mask 与每个活动对象最新 HoloLens `Sam3SpatialBox` 在 Shigure 图像中的投影圆比较。
2. 候选必须满足 mask 在投影圆内占比至少 `0.80`，且 mask depth 中位数与投影中心 depth 差不超过 `0.18 m`。
3. 单一候选直接绑定；多个候选使用 DINOv2 判别。
4. 等待 mask 区域 depth 满足稳定窗口，再提交 FoundationPose。
5. 同一对象只保留一个 pending 观测。正在运行的推理不取消，但有更新的观测时旧结果不会提交。
6. FoundationPose 结果通过 bbox IoU 和 depth residual 校验后写入最新 tracking pose，并追加到 `pose_events.jsonl`。

`take_out` 只记录拿取事件和人体证据，不提交物体新位姿。

## History/Live 模式

Unity 的 `ToggleHistoryTrackingMode()` 调用 `/realtime-tracking/mode`：

- `history`：停止接收实时位姿，显示每个对象最新一次 HoloLens 拍摄确认的位置。
- `live`：恢复读取 Shigure 事件并显示最新有效 tracking pose；暂停期间累积的移动事件按临时 Shigure ID 合并为最新一条。

模式切换通过 `request_generation`、`mode_epoch` 和 `coordinate_epoch` 拒绝过期响应。任何重新放置只更新 position/rotation，不修改已加载模型的缩放。

## ArUco 与公开坐标

HoloLens 上传当前本地坐标；服务器将持久位姿保存为 ArUco 坐标，并按请求方当前 `startup_session_id` 的最新 ArUco reference 转回 HoloLens 当前本地坐标。

Unity 只消费：

```text
coordinate_space = hololens_current_local
model_instance.pose
tracking item.pose
body_evidence.body_model.pose
```

ArUco reference 完成后，服务器只 retro-sync 同一 `startup_session_id` 的相关任务。

## 当前 HTTP 入口

- `POST /generate`
- `POST /check-queue`
- `GET /task-artifacts/<task_id>/<area>/<filename>`
- `GET /aruco/latest-reference?startup_session_id=...`
- `GET /aruco/markers`
- `POST /aruco/markers/sync`
- `POST /realtime-tracking/mode`
- `GET /realtime-tracking/status?startup_session_id=...`
