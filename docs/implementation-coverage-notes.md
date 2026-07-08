# Implementation Coverage Notes

更新日期：2026-07-08
状态：当前实现说明

本文用于补齐主流程文档之外的代码覆盖点。`task-processing-flow.md` 记录业务主链路；本文记录代码中存在但不一定出现在流程图箭头里的 API、socket service、helper 脚本和客户端辅助层。

## Server Routes

`code/server_api.py` 当前公开入口：

| Route | Purpose | Main Doc |
| --- | --- | --- |
| `GET /` | 服务状态和入口提示。 | 本文 |
| `GET /task-artifacts/<task_id>/<area>/<filename>` | 稳定 artifact 下载入口。 | `server-processing-output-contract.md` |
| `POST /generate` | object reconstruction / aruco reference 上传入口。 | `server-processing-output-contract.md` |
| `POST /check-queue` | Unity 轮询任务进度、pending preview、completed model instance。 | `server-processing-output-contract.md` |
| `GET /aruco/latest-reference` | 查询同次启动最新 ArUco reference 是否就绪。 | `server-processing-output-contract.md`, `task-processing-flow.md` |
| `GET /aruco/markers` | 查询启用的 ArUco marker registry。 | 本文 |
| `POST /aruco/markers/sync` | 从 reference folder 同步 marker registry。 | 本文 |
| `GET /latest-completed-task-ids` | 获取最近 completed 模型 ID，可按 startup session/filter 查询。 | `server-processing-output-contract.md` |
| `GET /model-bounds/latest` | 获取最近模型 bounds，服务端转换到当前 HoloLens 坐标。 | `server-processing-output-contract.md`, `model-event-tracking-state-machine.md` |
| `GET /model-bounds/range` | 按时间范围获取模型 bounds。 | `server-processing-output-contract.md`, `model-event-tracking-state-machine.md` |
| `POST /spatial-query/ray` | HoloLens 当前本地 ray 查询最近模型。 | `server-processing-output-contract.md`, `coordinate-systems.md` |
| `POST /spatial-query/ray-range` | HoloLens 当前本地 ray 查询时间范围内模型。 | `server-processing-output-contract.md`, `coordinate-systems.md` |
| `POST /history-placement-restoration/start` | 启动历史位置再现批处理。 | `history-position-restoration.md` |
| `GET /history-placement-restoration/latest` | 获取最近一次历史再现结果。 | `history-position-restoration.md` |

`/aruco/markers` 和 `/aruco/markers/sync` 是 marker registry 辅助接口，不改变“Unity 不消费 ArUco 坐标”的原则。正式模型位姿仍由服务器转成 `object_hololens_current` 后下发。

## Stage And Helper Scripts

`task_worker.STAGE_ORDER` 只包含业务 stage 名。下列脚本是代码中实际存在的执行单元：

| Script | Role |
| --- | --- |
| `run_sam3_boxmask_from_json.py` | `sam3mask` stage 和 socket service，生成 mask 与 pending preview spatial box。 |
| `run_historical_model_match_from_json.py` | 历史模型复用判断。 |
| `run_dinov2_identity_from_json.py` | DINOv2 identity socket service，被 historical model match 调用，不是独立 `STAGE_ORDER` stage。 |
| `run_instantmesh_from_json.py` | 默认模型生成后端。 |
| `run_sam3d_objects_from_json.py` | 可选模型生成后端/运行库来源，当前默认不走。 |
| `run_depthpointcloud_from_json.py` | 去边缘 mask/depth 点云与统计。 |
| `run_model_scale_from_json.py` | 模型缩放 stage。 |
| `run_object_alignment_from_json.py` | object alignment 主 stage。 |
| `run_object_icp_alignment_from_json.py` | ICP/foundationpose alignment helper，不是独立 `STAGE_ORDER` stage。 |
| `run_foundationpose_alignment_worker.py` | FoundationPose socket worker helper。 |
| `run_pose_from_json.py` | pose stage，生成 HoloLens/ArUco pose 数据。 |
| `run_aruco_sync_from_json.py` | ArUco sync stage。 |
| `run_runtime_mesh_from_json.py` | runtime mesh bake/export stage。 |
| `bake_runtime_mesh.py` | runtime mesh bake helper。 |
| `postprocess_sam3d_glb.py` | SAM3D Objects 后处理 helper。 |
| `convert_obj_to_fbx.py` | OBJ 到 FBX helper。 |
| `run_model_bounds_from_json.py` | model bounds stage。 |
| `run_display_identity_from_json.py` | display identity stage。 |
| `run_history_placement_restoration_from_json.py` | 历史位置再现。 |
| `run_taken_object_detection_from_json.py` | Shigure old mask 初始化与拿取判断。 |
| `run_sam3d_body_mesh_from_json.py` | 人体 mesh、距离修正、subject crop。 |
| `export_selected_body_fbx.py` | SAM3D Body FBX 导出 helper。 |
| `run_shigure_history_recorder.py` | Shigure RGB-D/object_detection/camera_info recorder sidecar。 |

## Socket Services

`task_worker.py` 管理这些半常驻服务：

| Service | Socket | Notes |
| --- | --- | --- |
| `sam3mask` | `data/worker_sockets/sam3mask.sock` | SAM3 mask/pending spatial box。 |
| `instantmesh` | `data/worker_sockets/instantmesh.sock` | 默认 3D 生成。 |
| `foundationpose` | `data/worker_sockets/foundationpose.sock` | object alignment helper。 |
| `dinov2_identity` | `data/worker_sockets/dinov2_identity.sock` | 历史模型复用判断。 |
| `shigure_history` | `data/worker_sockets/shigure_history.sock` | Shigure recorder 子进程内存缓存。 |

`MODEL_SERVICE_PREWARM_ENABLE=1` 时，worker 会预热模型相关服务。空闲退出时间由对应 `*_WORKER_IDLE_TIMEOUT_SEC` 控制。

## Top-Level Support Modules

| Module | Responsibility |
| --- | --- |
| `artifact_layout.py` | artifact 路径和文件命名。 |
| `config.py` | 运行开关、阈值、worker 参数。注意 `MODEL_GENERATION_BACKEND` 当前固定为 `instantmesh`。 |
| `console_output_log.py` | 控制台日志落盘到 `data/console_logs/`。 |
| `coordinate_systems.py` | 坐标系统基础定义。 |
| `spatial_transforms.py` | HoloLens/ArUco/Shigure pose、point、pixel-depth 变换。 |
| `depth_camera_config.py` | AHAT/LONGTHROW 等深度传感器范围和可靠深度限制。 |
| `display_identity.py` | 展示对象身份字段生成。 |
| `gpu_budget.py` | socket service GPU 预算和 CUDA 环境分配。 |
| `model_bounds.py` | bounds 查询、ray hit、去重逻辑。 |
| `model_generation_common.py` | 模型生成通用 helper。 |
| `object_alignment_common.py` | depth border crop、点云、alignment 共用逻辑。 |
| `subprocess_stream.py` | 子进程输出转发。 |
| `task_db.py` | SQLite schema、任务、ArUco、identity、history request 查询。 |
| `task_json.py` | task JSON 读写。 |
| `unity_coordinate_utils.py` | Unity 坐标工具。 |

## Unity Client Layer

核心数据流脚本：

- `ShuJuQingQiu.cs`: 上传、轮询、下载响应解析。
- `LoadModel.cs`: 正式模型加载；缺 `object_hololens_current` 时不显示。
- `RuntimeModelManager.cs`: runtime model registry 与 pose 管理。
- `HistoryPlacementRestorationDisplay.cs`: 历史再现状态、polyhedron、subject crop 显示。
- `RuntimeSpatialBoxDisplay.cs`: pending preview spatial box。
- `SpatialHistoryPointerQuery.cs`: HoloLens ray query。
- `RuntimeModelEventIdentity.cs` / `ModelEventDisplay.cs`: 模型事件和显示身份。

辅助/UI/调试脚本：

- `HoloLensDepthAquirer.cs`, `HoloLensPVAquirer.cs`: HoloLens RGB-D/PV 输入采集边界。
- `SelectionBoxController.cs`, `SelectionButtonsUI.cs`, `SelectionPanelManager.cs`: 选择框和 UI 操作。
- `CameraPoseDebugMarker.cs`, `GridLayoutButtonDebugActions.cs`: 调试显示与按钮。
- `DaoJiShi.cs`, `Game_M.cs`: 客户端 UI/场景辅助逻辑。

这些 Unity 辅助脚本不承担 ArUco 坐标转换。正式 runtime 坐标仍以服务器下发的 HoloLens current local pose 为准。

## Generated Or Historical Files

`data/config/source-git-manifest.tsv` 记录历史/生成数据来源，可能包含旧文档文件名；它不是当前 docs 索引，也不作为流程契约来源。
