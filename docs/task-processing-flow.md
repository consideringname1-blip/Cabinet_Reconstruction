# 任务处理流程

更新日期：2026-07-14
状态：Shigure v2 当前协议

本文描述服务器、不可修改的远端 Shigure ROS 数据入口，以及 Unity/HoloLens 的当前职责边界。

## 核心标识

- `task_id`：一次 HoloLens 上传任务的 UUID，也是 API 与数据库主键。
- `task_timestamp`：该任务的 artifact 目录名。
- `startup_session_id`：一次 Unity/HoloLens 启动的本地坐标会话。
- `display_object_id`：跨任务持久化的物体身份。
- `model_revision`：同一 `display_object_id` 下的模型版本。
- `runtime_session_id`：服务器进程内的一次 Shigure runtime 会话。
- `source_epoch_id`：一次 recorder/publisher incarnation；Shigure raw ID 只在该 epoch 内有效。

Shigure raw ID 不作为持久身份。服务器或 recorder 重启后必须建立新 source epoch，并重新完成临时 ID 到 `display_object_id` 的绑定。

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
  shigure_events/<event_uuid>/
  identity_references/<reference_id>/embedding.json
  identity_references/views/<sha256>/{scene.png,mask.png}
  shigure_debug_cache/          # 仅在 debug 开关启用时写诊断帧
  database/tasks.db
  worker_sockets/
  aruco/
  console_logs/
```

主程序从内存 socket cache 读取在线帧，绝不从 `shigure_debug_cache` 恢复业务状态。事件图片、mask、crop 和骨骼属于持久事件证据，写入 `shigure_events`。启动恢复、HoloLens identity sync、稳定视图和 FoundationPose 使用的 candidate scene/mask/crop 先写系统临时目录；一次尝试结束后统一删除。只有通过 novelty 判定而被接受的 identity 视图，才按内容摘要复制到 `identity_references/views/<sha256>` 并把持久路径写入数据库。

## Object reconstruction 主链

`task_worker.STAGE_ORDER` 的顺序为：

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

`MODEL_GENERATION_BACKEND` 只能是 `instantmesh` 或 `sam3d_objects`。HoloLens 拍摄用于生成/更新模型、提供更精确的辅助身份视角和 ArUco 当前本地坐标，不再作为 Shigure 开始追踪或存在判定的前置条件。

DINOv2 命中已有 `display_object_id` 且未强制重建时，可复用其 completed 模型资产；新拍摄仍更新对齐、位姿和身份参考。历史 capture 查询可以向后扫描较大的去重窗口，但实际 DINO 评分只保留最近 5 个不重复物体，每个物体一条最新 capture。HoloLens FoundationPose 请求与 Shigure runtime 请求进入同一个共享优先队列；`FOUNDATIONPOSE_POOL_SIZE=1..4` 是弹性 backend 上限。业务请求只等待首个 backend 可用，其余槽位在后台以非阻塞 GPU lease 尝试加入，显存不足或可选实例启动失败不会阻塞已经可工作的 backend；后续请求会继续尝试扩容。HoloLens 具有更高排队优先级，但不占专用槽，也不抢占已经运行的任务。已启动 backend 由当前 server worker 进程持有到 shutdown。

## Shigure ingress 与 canonical frame

`run_shigure_history_recorder.py` 只适配远端现有 ROS topics，不修改 `reconstruction/shigure_core`：

- RGB、同步 depth、CameraInfo
- 稀疏 `/shigure/object_detection`
- `/shigure/object_tracking`
- `/Segments`
- `/shigure/people_detection`
- 可选 `/shigure/contacted`

`ShigureCompatibilityAdapter` 按完全相同的 ROS stamp 生成 `CachedShigureFrame` schema v2。事件的 bbox-local mask 必须粘贴回原始全图坐标，禁止 resize 成全图。缺失、显式空列表和无效 mask 是不同状态。

recorder 的在线 RGB-D/canonical ring 只存在内存并通过 Unix socket 提供。稀疏事件和恢复候选只在存在同 stamp RGB 时附加精确事件 RGB-D，禁止拿邻近 RGB 冒充。可选 debug disk ring 默认关闭；开启后把这些 exact-stamp canonical/RGB-D 诊断项写入独立环形目录，最多保留 600 秒，仅用于复查。

## 身份与启动恢复

服务器启动、recorder incarnation 改变，或 adapter 检测到 `/shigure/object_tracking` raw ID 的时间前缀改变时：

1. 创建新的 runtime session/source epoch，并撤销旧 raw-ID binding。
2. 等待候选所在 exact stamp 的 `object_tracking=present` 和 exact RGB-D；输入尚未齐全时将恢复 job 保持为 `PENDING`，不提前执行 DINO。
3. 从最新完整 `/Segments`/tracking canonical frame 取得当前候选 mask。
4. 只取最近 5 个已有模型 revision 且有活动 identity reference 的持久 `display_object_id`；不要求它曾保存 HoloLens pose。为每个 candidate 计算到每个 display 的 DINOv2 edge cost。
5. 在完整 candidate×display 矩阵上做全局一对一分配：先优先覆盖带可信 raw ID 的 candidate，再最大化可绑定数量，最后最小化总代价；不会按 ROS 顺序逐个贪心。最佳与次佳同规模方案的总代价 margin 不足时，相关 candidate 保持 ambiguous。
6. 有可信 raw ID 时建立 epoch-scoped binding；没有 raw ID 时只记录 provisional 结果，并让恢复 job 保持 `PENDING`。
7. Shigure 视角与现有参考差异足够大时，保存为新的 `SHIGURE` identity reference；否则不重复记录。

`segments=explicit_empty` 且 `object_tracking=explicit_empty` 的无 candidate frame 是一份完整空 snapshot：runtime 会以 0 match 完成本次启动恢复。任一 topic 仍为 `missing` 时不能用空列表结束恢复；非空 segments 但 tracking 尚未到达、exact RGB-D 尚未到达，或存在 `UNBOUND`、`AMBIGUOUS`、`CONFLICT`、`PROVISIONAL` candidate 时，恢复均保持 `PENDING`。后续不同 source stamp 会按 `SHIGURE_STARTUP_RECOVERY_RETRY_SECONDS` 节流重试；同 epoch 已绑定 candidate 固定保留，不参与重复 DINO 分配。全部 candidate 为 `BOUND` 才 `COMPLETED`，达到 `SHIGURE_STARTUP_RECOVERY_MAX_ATTEMPTS` 后才以 `FAILED` 终止。

HoloLens 拍摄提供 `HOLOLENS` reference 作为辅助条件。已有 Shigure reference 时优先使用 Shigure 多视角参考。

HoloLens 模型任务完成后还会建立 `HOLOLENS_CAPTURE` identity sync job：先比较 HoloLens capture box 与 Shigure collider 经 ArUco AABB 重建后得到的中心和 extent；没有候选通过宽松几何门时保持失败/重试，多个候选通过时才只在门内以较宽阈值使用 DINOv2 判别。只有连续稳定、带可信 raw ID 的 Shigure recovery candidate 才能成功；它可以建立当前 epoch binding、激活 `PRESENT` 并登记 Shigure identity reference。这里的 lifecycle authority 明确是 `shigure_recovery_snapshot`，HoloLens 只提供待核对的目标 identity/model，不单独授权 presence，也不制造 `bring_in` 或 `take_out`。

仅由 HoloLens 新建的对象以 `presence=UNKNOWN` 持久化；completed `model_instance` 可供 Unity 预览，但不因此进入 Shigure live 清单或成为 `PRESENT`。只有 Shigure 生命周期事件或可信 recovery snapshot（包括上述 sync 成功）能够改变该状态。

## 生命周期事件

Shigure 是 presence/lifecycle 的权威来源：

- `bring_in`：允许使用 DINOv2 解析新 binding，并将对象置为 `PRESENT`。
- `take_out`：只接受当前 source epoch 内已有 binding；不使用 DINO 猜测。在同一数据库事务中记录拿走前最后有效 pose、scene image、mask/crop、raw collider box 和 Shigure 骨骼，把对象置为 `ABSENT`，清空 active binding，并以 `take_out_completed` 撤销旧 binding。物体再次 bring-in 时可用上游新分配的 raw ID 重新绑定到同一持久 `display_object_id`。
- `obj_move`：上游消息无法可靠证明实际移动对象。适配器对每个新的 move stamp 只轮换一次 source incarnation，立即发出空 epoch barrier，并丢弃时间不晚于该 barrier 的所有 canonical topic payload。barrier 之后、首个严格更晚的干净 tracking 之前，camera/segments/people 等辅助 topic 可以暂存在 exact-stamp bucket，但禁止 emit canonical frame、禁止触发 Holo sync 重试；解锁时只保留该干净 tracking 同 stamp 的辅助 bucket，其余隔离期 bucket 丢弃，且不进行第二次 incarnation 轮换。污染帧不进入 segment/raw-ID 映射、live box、示例采集或 FoundationPose。

每个稀疏检测项以 `(runtime session, source epoch, sec, nanosec, frame_id, index)` 唯一化。canonical event 与 lifecycle event 持久化后可供复查，临时 raw ID 不跨服务器启动复用。同一 exact stamp 的 people/contact 等 topic 迟到时，adapter 会发出 enriched canonical revision；相同 canonical event 的 replay 不再次推进 presence，也不创建第二条 lifecycle/history 记录。新 pose、box 和图片只补原记录中的 `NULL` 证据；迟到的 exact event skeleton 可以替换首次 emission 从 live state 取得的非空 fallback skeleton。

## 实时位姿、box 与骨骼

实时计算持续进行，不受 Unity 当前展示 live/history 的影响：

- 已绑定且 `PRESENT` 的 tracking 观测只有先通过连续 5 帧严格 mask 共识和 DINO identity-anchor admission，才可提交 FoundationPose；闪烁/遮挡帧不先行更新 pose。
- 同一对象使用 latest-wins 排队；旧结果不能覆盖更新的观测或模型 revision。
- FoundationPose 结果通过 bbox IoU 与 depth residual 验收后，保存为 ArUco pose。
- 人体使用 Shigure `/people_detection` 点线骨骼，不生成或下发人体 mesh。

空间框只使用 `/shigure/object_tracking` 的 raw collider：

1. 以 Shigure camera/mm 解析 collider 的最小角 `x/y/z` 和三条正 extent。
2. 构造 camera-space 八角点。
3. 转到 ArUco 后按轴取 min/max，重建 ArUco 轴对齐包围框，使前后面与竖直 ArUco marker 对齐；原 camera box 相对 ArUco 有旋转时，新框会包住旋转后的 8 点，边长可能增大，并非保留原三边。
4. 固定顺序保存 8 个 ArUco 点；API 再转换为当前 HoloLens local。

mask、depth、有效像素都不能作为 spatial box 的替代来源。无合法 collider 时返回无 box，不生成降级框。

服务端按 binding 维护 box 状态：有效 collider 需连续 2 帧获取，center/extent 经过 5 帧中值、EMA 和 8 mm/5 mm 死区；单帧 collider 无效或完整 tracking snapshot 缺席进入 `COASTING`，继续保留最后可信 box。连续明确缺失满 1.5 秒才写 `no_box`；topic `missing` 或辅助消息重放不作为负观测。take-out 立即清 live box，不走 grace，历史仍保存拿走前框。

live API 把 box 从最多 5 个模型/pose `items` 中拆出：`spatial_boxes` 是无数量上限、带 complete 标记的完整快照。合法八点为 `status=ready`；宽限期结束、ABSENT 或无框为 `status=no_box`。Unity 只有在整份 box snapshot 校验成功后才 reconcile；`no_box` 或完整快照缺席会删除 `LatestLive` 框，但不删除历史框、照片、骨骼或模型。

## Unity live 与历史展示

服务器传输模式只有 `live`。`POST /realtime-tracking/mode` 是 live handshake，`GET /realtime-tracking/status` 持续返回最新计算状态。

历史位置是 Unity presentation state，不暂停服务器计算：

- 单击 live 模型：请求该物体最近一次 `take_out` 历史，模型显示历史 pose，并自动显示对应 scene image 与彩色骨骼。
- 再次单击同一物体：带 `before_cursor` 请求更老事件。
- 全体历史再现：每个对象独立请求上一条历史。
- 全体继续追踪：所有模型恢复 `LatestLiveState`，并关闭全部历史图片、骨骼和历史 box。

历史 API 只返回具备有效 pose、scene image 和骨骼的持久事件；服务端会跨页跳过损坏或证据不完整的数据库行，直到找到下一条可用记录或真正耗尽历史。所有公开 pose、box、骨骼均使用当前 startup 的 `hololens_current_local` 与同一个 `coordinate_epoch`。

Unity 的 history URL 默认从已配置的 realtime status/mode 服务地址推导同源 `/api/v2/`，也允许显式 override。收到新的 live `coordinate_epoch` 时，任何仍显示旧 epoch 历史的模型会自动恢复 `FollowLive` 并关闭对应照片/骨骼，避免跨坐标 epoch 继续显示旧证据。

## 当前 HTTP 入口

- `POST /generate`
- `POST /check-queue`
- `GET /task-artifacts/<task_id>/<area>/<filename>`
- `GET /shigure-event-artifacts/<event_directory>/<filename>`
- `GET /aruco/latest-reference?startup_session_id=...`
- `GET /aruco/markers`
- `POST /aruco/markers/sync`
- `POST /realtime-tracking/mode`（仅 `mode=live`）
- `GET /realtime-tracking/status?startup_session_id=...`
- `GET /api/v2/identity-sync/<sync_job_id>`
- `GET /api/v2/display-objects/<display_object_id>/history?...`

## 数据库 schema 与一次性迁移

正常启动只接受两种情况：空数据库一次性创建完整 Shigure v2 schema，或已有数据库严格匹配 v2 表、索引和 `schema_metadata`。启动路径不修补、不删除也不兼容旧 schema；发现 legacy/偏差时会 fail closed。

迁移只能显式离线执行：

```bash
python code/migrate_shigure_v2_data.py          # 只读 dry-run
python code/migrate_shigure_v2_data.py --apply  # 临时备份后迁移，成功时删除全部 pre-v2/退役数据
```

`--apply` 会先生成并验证临时 pre-v2 数据库/任务 JSON 备份；只有 schema、行数和迁移结果全部校验成功后，才连同整个退役的 `data/shigure_history_cache`、任务 JSON 中的 `HistoryPlacementRestoration`/SAM3D Body 旧块及其他无法迁移的数据一起删除。迁移失败时临时备份保留供人工恢复，成功后的活动 `data/` 和 DVC 快照不保留 pre-v2 副本。SAM3D Body、旧实时追踪日志等已退役数据不进入新运行时，也没有运行期兼容读取分支。`code/reconstruction/sam3d-body` 子模块仅保留为第三方源码/历史复现参考；它不是当前 stage、socket service、API 或数据库契约的一部分。
