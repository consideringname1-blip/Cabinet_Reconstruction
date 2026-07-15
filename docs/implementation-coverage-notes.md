# Implementation Coverage Notes

更新日期：2026-07-14
状态：Shigure v2 历史说明（已被 v3 取代）

> 本文保留旧入口覆盖表用于追溯；Shigure v3 的 active 数据链、迁移与测试见 [`shigure-v3-runtime.md`](shigure-v3-runtime.md)。

本文列出主模型生成流程之外仍属于当前实现的入口和执行单元。

## Server routes

| Route | Purpose |
| --- | --- |
| `GET /` | 服务状态。 |
| `GET /task-artifacts/<task_id>/<area>/<filename>` | 模型任务 worker/result/debug artifact。 |
| `GET /shigure-event-artifacts/<event_directory>/<filename>` | 生命周期事件照片。 |
| `POST /generate` | 严格 object reconstruction 或 ArUco reference 上传。 |
| `POST /check-queue` | task 状态、capture preview、canonical `model_instance` 和 identity-sync 摘要/status URL。 |
| `GET /aruco/latest-reference` | 查询当前 startup 的 ArUco reference 是否就绪。 |
| `GET /aruco/markers` | 查询启用 marker registry。 |
| `POST /aruco/markers/sync` | 从 `aruco.json` 同步 marker registry。 |
| `GET /api/v2/identity-sync/<id>` | 查询持久 HoloLens/Shigure identity-sync job 及 terminal 状态。 |
| `POST /realtime-tracking/mode` | 建立或刷新 live transport 握手；只接受 `mode=live`。 |
| `GET /realtime-tracking/status` | 获取最多 5 个 live pose/model，以及独立、无数量上限的完整 Shigure box 快照。 |
| `GET /api/v2/display-objects/<id>/history` | 分页查询该对象的 take-out 历史位置、照片和点线骨骼。 |

## Main stages and helpers

| Script | Role |
| --- | --- |
| `run_sam3_boxmask_from_json.py` | `sam3mask` stage 与 pending `Sam3SpatialBox`。 |
| `run_historical_model_match_from_json.py` | DINOv2 display identity 匹配和模型资产复用。 |
| `run_dinov2_identity_from_json.py` | DINOv2 socket service。 |
| `run_instantmesh_from_json.py` | `model_generation` 的 InstantMesh backend。 |
| `run_sam3d_objects_from_json.py` | `model_generation` 的 SAM3D Objects backend。 |
| `run_depthpointcloud_from_json.py` | mask/depth 点云。 |
| `run_model_scale_from_json.py` | 新生成模型缩放。 |
| `run_object_alignment_from_json.py` | object alignment stage。 |
| `run_object_icp_alignment_from_json.py` | ICP/FoundationPose alignment helper。 |
| `run_foundationpose_alignment_worker.py` | FoundationPose backend socket worker。 |
| `run_pose_from_json.py` | HoloLens/ArUco pose。 |
| `run_aruco_sync_from_json.py` | ArUco sync stage。 |
| `run_runtime_mesh_from_json.py` | runtime mesh bake/export。 |
| `bake_runtime_mesh.py` | runtime mesh bake helper。 |
| `postprocess_sam3d_glb.py` | SAM3D Objects GLB 后处理。 |
| `convert_obj_to_fbx.py` | runtime OBJ 到 FBX。 |
| `run_model_bounds_from_json.py` | `ModelBounds`。 |
| `run_display_identity_from_json.py` | display object/revision commit。 |

## Shigure v2 execution

| Component | Role |
| --- | --- |
| `run_shigure_history_recorder.py` | 只适配远端 ROS topic，建立 exact-stamp canonical frame，并提供进程内 socket cache。 |
| `ShigureCompatibilityAdapter` | 将稀疏 detection、tracking、RGB-D 与骨骼归一化；为缺失 raw ID 的事件生成 source-epoch 内临时 ID，并以 tracking ID 时间前缀识别上游 node 独立重启。 |
| `ShigureRuntimeEngine` | source epoch、latest-5 identity 恢复、Holo sync、生命周期、严格 5 帧＋DINO 示例准入、DINO 后置 FoundationPose，以及带 1.5 秒 coasting 的 box 更新。 |
| `ShigureDebugDiskRing` | 仅在 debug 开关启用时保留最多 10 分钟的有界本地复查数据。 |
| `spatial_box_v2.py` | collider 毫米数据到 ArUco 轴对齐的严格 8 点线框 box；失败即无 box。 |

远端 `code/reconstruction/shigure_core` 不在本仓库修改范围。`/shigure/object_detection` 是稀疏事件输入，不被当成启动时完整静态物体清单。

debug ring 默认关闭，只写 exact-stamp 事件/RGB-D 与 canonical 诊断，业务 runtime 没有从中恢复的读取入口。SAM3D Body 已从 runtime、stage、socket 与 API 退役；人体证据只使用 Shigure joint 点线骨骼。`code/reconstruction/sam3d-body` 子模块仍作为第三方源码/历史复现参考保留，但不是当前运行依赖。

## Socket services

| Service | Socket | Notes |
| --- | --- | --- |
| `sam3mask` | `data/worker_sockets/sam3mask.sock` | SAM3 image mask。 |
| `instantmesh` | `data/worker_sockets/instantmesh.sock` | 仅在 backend 为 InstantMesh 时使用/预热。 |
| `foundationpose` pool | `FOUNDATIONPOSE_POOL_SIZE=1..4` 是弹性 backend 上限 | 全部 worker 共用一个队列，不固定保留 HoloLens 槽位；请求只等待首个 backend，其余由后台线程按非阻塞 GPU lease 加入，可选实例加载不占业务请求时延；未启动 socket 不领取任务，已启动进程持有到 shutdown。 |
| `foundationpose_dispatcher` | `data/worker_sockets/foundationpose_dispatcher.sock` | HoloLens 只提高排队优先级，不抢占运行中任务；较新同物体 generation 取代仍在排队的旧请求。 |
| `dinov2_identity` | `data/worker_sockets/dinov2_identity.sock` | HoloLens 和 Shigure identity embedding。 |
| `shigure_history` | `data/worker_sockets/shigure_history.sock` | recorder 的进程内 canonical frame/event cache。 |

GPU 启动通过预计峰值显存的 host-wide 原子 lease 做 best-fit 装箱：先在 SQLite 写锁外采集 GPU 快照，再用短 `BEGIN IMMEDIATE` 事务完成死亡 owner 清理、共享 reservation 读取、fit 与写入，因此同一主机上的独立 server/Gunicorn/Python launcher 不会同时占用同一份“空闲”显存。lease 记录 owner/worker PID、boot ID 与进程启动时刻，进程退出或 PID 被复用后自动回收；实际 managed CUDA 用量只补算超过 reservation 的部分。`GPU_LEASE_REGISTRY_PATH` 默认是 `/tmp/shigure_gpu_leases_v1.sqlite3`。该协调范围不跨主机。

## Persistent and ephemeral state

SQLite 持久保存：

```text
tasks / stage runs / timings
ArUco marker registry and startup references
display objects, model revisions and HoloLens capture poses
Shigure runtime sessions and source epochs
source-epoch temporary bindings and canonical events
object lifecycle events, take-out pose/box/skeleton evidence
identity references and identity sync jobs
```

持久 artifact 包括 `data/shigure_events` 的事件照片，以及 `data/identity_references/views/<sha256>` 中先通过严格时序/DINO anchor admission、再通过 novelty 判定的 identity scene/mask。candidate 推理文件位于系统临时目录，在 startup recovery、Holo sync、stable identity 或 FoundationPose 尝试完成后删除；数据库不得引用这些临时路径。进程内状态包括最多 5 个 identity/model-pose 候选、无数量上限的 Shigure live box registry、每 binding 的 box filter/coasting latch、canonical frame/event ring、pending/running FoundationPose generation 和 live transport handshake。debug disk ring 默认关闭，开启后保留时间上限为 600 秒。

服务器、recorder 或 object-tracking node 重启会创建新 runtime/source epoch，旧 Shigure raw ID 不会跨 epoch 复用；持久 display identity、模型 revision 和完整生命周期记录保留。成功 `take_out` 在写入历史的同一事务内释放 active binding，使下一次 bring-in 的新 raw ID 能重新绑定同一 display object。
pending identity-sync job 在 runtime stop 或服务器 restart 时转为 terminal `FAILED`，可由返回的 `status_url` 查询；完整的 segments/tracking 空 snapshot 则以 0 match 正常完成启动恢复。非空候选缺同 stamp tracking、exact RGB-D 或可信 raw ID 时，startup recovery 保持 `PENDING`。`UNBOUND`、`AMBIGUOUS`、`CONFLICT` 和 `PROVISIONAL` 同样是可重试结果；默认至少间隔 1 秒、最多尝试 10 次，且只在全部候选绑定后完成。


`initialize_task_table()` 对已有数据库只做严格 v2 校验，不包含 legacy 修复或兼容读取。旧数据必须先运行 `python code/migrate_shigure_v2_data.py` dry-run，再显式使用 `--apply` 完成一次性迁移；无法迁移的退役数据由迁移工具删除，其中 `data/shigure_history_cache` 整个目录都会删除，不保留未知文件。

## Unity layer

- `ShuJuQingQiu.cs`：object/ArUco 上传、task polling、最多 5 个模型/pose 下发，并原子校验/应用独立的无上限完整 Shigure box snapshot。
- `RuntimeModelManager.cs`：canonical model、display identity/revision、本地 FBX cache、live 与 presentation pose 分离，以及完整 box snapshot 的 update/no_box/missing 删除 reconcile。
- `HistoryPresentationController.cs`：从 realtime 服务派生 history URL，执行单物体/全体历史分页与恢复 live 呈现；coordinate epoch 改变时旧历史自动回 live。
- `ObjectEvidenceDisplay.cs`：历史照片和点线骨骼的自动显示/隐藏。
- `RuntimeSkeletonDisplay.cs`：按 joint 名称和骨段颜色绘制 Shigure 骨骼。
- `RuntimeSpatialBoxDisplay.cs`：pending preview 或严格 8 点 runtime wireframe。
- `RuntimeModelEventIdentity.cs` / `ModelEventDisplay.cs`：已加载模型的点击身份与呈现入口。
- `HoloLensDepthAquirer.cs` / `HoloLensPVAquirer.cs`：严格 RGB-D/PV 采集结构。

Unity 的正式 model、live/history pose、spatial box 和 skeleton 坐标只使用 `hololens_current_local`。历史呈现期间 live 数据仍持续接收并写入模型状态，只暂停对该模型场景 transform 的应用。
