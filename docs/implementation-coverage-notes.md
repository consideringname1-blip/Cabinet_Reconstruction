# Implementation Coverage Notes

更新日期：2026-07-12
状态：当前协议

本文列出主流程之外仍属于当前实现的入口和执行单元。

## Server routes

| Route | Purpose |
| --- | --- |
| `GET /` | 服务状态。 |
| `GET /task-artifacts/<task_id>/<area>/<filename>` | 模型任务 worker/result/debug artifact。 |
| `POST /generate` | 严格 object reconstruction 或 ArUco reference 上传。 |
| `POST /check-queue` | task 状态、pending preview 和 canonical `model_instance`。 |
| `GET /aruco/latest-reference` | 查询当前 startup 的 ArUco reference 是否就绪。 |
| `GET /aruco/markers` | 查询启用 marker registry。 |
| `POST /aruco/markers/sync` | 从 `aruco.json` 同步 marker registry。 |
| `POST /realtime-tracking/mode` | 切换 `history` / `live` 并返回一致性 snapshot。 |
| `GET /realtime-tracking/status` | 获取当前 mode 下最多 5 个对象的 pose、model 和 body evidence。 |

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

## Auxiliary execution

| Component | Role |
| --- | --- |
| `run_shigure_history_recorder.py` | 远端 Shigure RGB-D/contact/detection recorder。 |
| `ShigureRealtimeTrackingEngine` | `obj_move`/`bring_in` identity、depth stability 与 FoundationPose。 |
| `ShigureAuxiliaryBranchManager` | upload-time contact watcher 与 `ShigureContactEvidence`。 |
| `run_sam3d_body_mesh_from_json.py` | contact person bbox 到人体 mesh 与 subject crop。 |
| `export_selected_body_fbx.py` | 人体 FBX 导出。 |

## Socket services

| Service | Socket | Notes |
| --- | --- | --- |
| `sam3mask` | `data/worker_sockets/sam3mask.sock` | SAM3 image mask。 |
| `instantmesh` | `data/worker_sockets/instantmesh.sock` | 仅在 backend 为 InstantMesh 时使用/预热。 |
| `foundationpose` | `data/worker_sockets/foundationpose.sock` | FoundationPose backend。 |
| `foundationpose_dispatcher` | `data/worker_sockets/foundationpose_dispatcher.sock` | 高优先级 HoloLens capture 与低优先级 realtime 请求。 |
| `dinov2_identity` | `data/worker_sockets/dinov2_identity.sock` | HoloLens 与 Shigure 身份 embedding。 |
| `shigure_history` | `data/worker_sockets/shigure_history.sock` | recorder 进程内 RGB-D/event cache。 |

SAM3D Objects 和 SAM3D Body 由各自配置的 Python 解释器直接运行，不共享 Unity 输出协议以外的状态。

## Persistent and ephemeral state

SQLite 持久保存：

```text
tasks / stage runs / timings
ArUco marker registry and startup references
display objects and model revisions
HoloLens capture pose revisions
accepted tracking pose revisions and events
body revisions
auxiliary jobs
```

进程内状态包括：

```text
最多 5 个活动 realtime display objects
Shigure temporary ID bindings
pending/running FoundationPose tokens
RGB-D/event ring cache
History/Live session epochs
```

服务器停止会清空进程内状态，不删除 SQLite、task artifact 或 `data/realtime_tracking/<display_object_id>/pose_events.jsonl`。

## Unity layer

- `ShuJuQingQiu.cs`：object/ArUco 上传、task polling、模式握手、status、下载队列。
- `RuntimeModelManager.cs`：canonical model、display identity/revision、本地 FBX cache 和 pose 应用。
- `ObjectEvidenceDisplay.cs`：canonical `body_evidence`。
- `RuntimeSpatialBoxDisplay.cs`：pending preview。
- `RuntimeModelEventIdentity.cs` / `ModelEventDisplay.cs`：已加载模型的事件身份与显示。
- `HoloLensDepthAquirer.cs` / `HoloLensPVAquirer.cs`：严格 RGB-D/PV 采集结构。

Unity 的正式 model/body/tracking 坐标只使用 `hololens_current_local`。
