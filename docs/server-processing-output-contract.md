# Server Processing Input / Output Contract

日期：2026-06-24  
状态：当前实现说明

这份文档记录服务器 pipeline 的稳定输入、输出、JSON 字段和文件布局。算法细节见对应专题文档：坐标系见 `docs/coordinate-systems.md`，Shigurei ROS2 topic 见 `docs/shigure-core-ros-messages.md`，拿取判断见 `docs/拿取判断状态机.md`，SAM3D Body 见 `docs/sam3dbody相关.md`。

## 入口与目录来源

服务入口：

```bash
cd /workspace_whs
python code/run_server.py
```

目录和脚本配置拆分如下：

- `config.py`：运行开关、阈值、worker 参数。
- `path_config.py`：代码目录、stage 脚本、Python/Blender 可执行文件。
- `artifact_layout.py`：`data/` 目录、任务目录、文件命名、HTTP `FOLDER_MAP`。
- `task_db.py`：SQLite schema 初始化和读写。

服务启动时会调用 `ensure_artifact_roots()`、`initialize_task_table()` 和 `start_worker()`。

## Runtime Roots

当前根目录：

```text
data/model/<task_timestamp>/              # 3D model task
  task.json
  worker/
  result/
  debug/
  logs/

data/aruco_processing/<task_timestamp>/   # ArUco reference task
  task.json
  worker/
  result/
  debug/

data/history_placement_requests/<request_timestamp>/
  worker/
  result/
  debug/

data/database/tasks.db
data/console_logs/
data/shigure_history_cache/
data/aruco/reference/
data/aruco/runtime/
data/aruco/shigure_marker_history/
data/output/                              # external scratch / compat folders
```

`data/upload/` 不再使用。

`/files/<folder>/<filename>` 仍只暴露 `artifact_layout.FOLDER_MAP` 中的兼容目录：

```text
meshes                       -> data/output/instant-mesh-large/meshes
images                       -> data/output/instant-mesh-large/images
videos                       -> data/output/instant-mesh-large/videos
sam3d_object_meshes          -> data/output/sam3d-objects/meshes
runtime_meshes               -> data/output/runtime_mesh
fbx                          -> data/output/blender/fbx
history_placement_restoration -> data/output/history_placement_restoration
sam3d_body_meshes            -> data/output/sam3d_body/meshes
sam3d_body_fbx               -> data/output/sam3d_body/fbx
```

新流程的稳定结果优先在 `data/model/<task_timestamp>/result/` 中读取；兼容 URL 只用于仍需要旧 HTTP folder 的客户端路径。

## `/generate`: Object Reconstruction

Multipart form input：

```text
purpose=object_reconstruction              # 可省略
deviceJ                                    # JSON object
PVCameraJ 或 PVCameraFramesJ               # 单帧或多帧 PV 参数
depthCameraJ / DepthCameraJ                # JSON object
SelectionBoxJ                              # JSON object
pv_image                                   # PNG bytes
depth_image                                # PNG bytes
```

服务行为：

1. 生成 `task_timestamp = make_timestamp()` 和 `task_id = uuid`。
2. DB 中创建 `status = uploading` 的 task。
3. 创建 `data/model/<task_timestamp>/worker|result|debug|logs/`。
4. 写入上传文件和 `task.json`。
5. 状态改为 `pending` 并入队。

立即写入：

```text
data/model/<task_timestamp>/task.json
data/model/<task_timestamp>/worker/01_upload_color.png
data/model/<task_timestamp>/worker/01_upload_depth.png
```

初始 task JSON 关键字段：

```json
{
  "server_received_utc": "...Z",
  "task_name": "<task_timestamp>",
  "task_timestamp": "<task_timestamp>",
  "task_id": "uuid",
  "artifact_schema_version": 1,
  "purpose": "object_reconstruction",
  "device": {"startup_session_id": "..."},
  "PVCamera": {"name": "01_upload_color.png", "width": 0, "height": 0, "k": [[0]], "pose": [[0]]},
  "PVCameraFrames": [{"name": "01_upload_color.png", "artifact_root": "model_worker"}],
  "DepthCamera": {"name": "01_upload_depth.png", "sensor": "AHAT", "stats": {}},
  "SelectionBox": {"top_left": [0, 0], "bottom_right": [1, 1]}
}
```

## `/generate`: ArUco Reference

Multipart form input：

```text
purpose=aruco_reference
deviceJ
PVCameraFramesJ                            # one or more frame objects
pv_image 或 pv_image_0, pv_image_1, ...
```

No depth image or selection box is required.

Immediately written layout：

```text
data/aruco_processing/<task_timestamp>/task.json
data/aruco_processing/<task_timestamp>/worker/<frame_timestamp>_color.png
data/aruco_processing/<task_timestamp>/worker/<frame_timestamp>_meta.json
```

ArUco stage output：

```text
data/aruco_processing/<task_timestamp>/result/summary.json
data/aruco_processing/<task_timestamp>/result/<frame_timestamp>_marker_detect.json
data/aruco_processing/<task_timestamp>/debug/<frame_timestamp>_aruco_debug_overlay.png
```

`debug` overlay 合并搜索范围、检测 ROI、marker 边框/角点/id 和摘要。

## Model Task Artifact Contract

### Worker files

```text
01_upload_color.png
01_upload_depth.png
01_upload_meta.json
01_upload_align_depth.png
02_sam3_mask.png
02_sam3_color.png
02_sam3_depth.png
03_model_source.obj
03_model_source.mtl
03_model_source_texture.png
03_generation_instantmesh_input.png
03_generation_instantmesh_raw.obj
03_generation_sam3d_raw.glb
03_generation_sam3d_postprocess.json
04_runtime_mesh.obj
04_runtime_mesh.mtl
04_runtime_mesh_texture.png
```

### Result files

```text
05_export_final.fbx
06_identity_decision.json
07_taken_detection_result.json
07_taken_detection_result_rgb.png
07_taken_detection_result_depth.png
07_taken_detection_camera_info.json
07_taken_detection_active_objects.json
07_taken_detection_marker_6d_pose.json
08_sam3d_body_result.json
08_sam3d_body_people.json
08_sam3d_body_selected_person.obj
08_sam3d_body_selected_person.fbx
```

### Debug files

```text
01_depth_alignment_align_depth_turbo.png
02_sam3_mask_overlay.png
03_generation_instantmesh_video.mp4
```

Debug output is controlled by `TASK_DEBUG_OUTPUT_ENABLE`. The pipeline must not require debug files to complete.

## Stage Order

Object reconstruction worker stage order：

```text
hololens2depth
sam3mask
instantmesh              # shared model-generation slot; backend may be InstantMesh or SAM3D Objects
depthpointcloud
modelscale
object_alignment
runtime_mesh
pose
aruco_sync
blender
model_bounds
display_identity
history_placement_restoration
taken_object_detection
sam3d_body_mesh
completed
```

ArUco reference worker stage order：

```text
aruco_detect
aruco_completed
```

## Stable Stage Effects

| Stage | Stable file output | Stable JSON/DB effect |
|---|---|---|
| `hololens2depth` | `worker/01_upload_align_depth.png`; debug turbo image if enabled | `DepthCamera.align_depth_name`, depth stats |
| `sam3mask` | `worker/02_sam3_mask.png`, `02_sam3_color.png`, `02_sam3_depth.png`; debug overlay if enabled | `sam3Name`, `Sam3SpatialBox` |
| `instantmesh` | `worker/03_model_source.*`, `03_generation_instantmesh_*`; debug video if enabled | `InstantMesh`, `ModelGeneration`; AI timing DB |
| `sam3d_objects` | `worker/03_model_source.*`, `03_generation_sam3d_*` | `SAM3DObjects`, `ModelGeneration`; AI timing DB |
| `depthpointcloud` | no stable large file | `depthpointcloud` stats |
| `modelscale` | no stable large file | `model` scale fields |
| `object_alignment` | optional scratch preview in `data/output/object_alignment/` | `object_alignment`, pose debug fields |
| `runtime_mesh` | `worker/04_runtime_mesh.*` | `RuntimeMesh` |
| `pose` | no stable large file | `object_world`, pose debug fields |
| `aruco_sync` | no stable large file | `aruco_reference`, `object_aruco` |
| `blender` | `result/05_export_final.fbx` | `Blender`, completed model response data |
| `model_bounds` | DB row | `ModelBounds`, SQLite `model_bounds` |
| `display_identity` | `result/06_identity_decision.json` | `DisplayIdentity`, identity DB tables |
| `history_placement_restoration` | request-level output when invoked through API; task JSON field when run as worker stage | `HistoryPlacementRestoration`, `history_placement_requests` DB for API runs |
| `taken_object_detection` | `result/07_taken_detection_*` | `TakenObjectDetection` |
| `sam3d_body_mesh` | `result/08_sam3d_body_*` | `SAM3DBodyMesh` |

## History Placement Request API

`POST /history-placement-restoration/start` creates a request-level record and directory：

```text
data/history_placement_requests/<request_timestamp>/worker/01_request.json
data/history_placement_requests/<request_timestamp>/worker/01_selected_model_tasks.json
data/history_placement_requests/<request_timestamp>/worker/02_result_<index>_working/
data/history_placement_requests/<request_timestamp>/worker/02_result_<index>_working_state.json
data/history_placement_requests/<request_timestamp>/result/02_result_<index>_summary.json
data/history_placement_requests/<request_timestamp>/result/02_response.json
data/history_placement_requests/<request_timestamp>/result/02_unity_display.json
```

`GET /history-placement-restoration/latest` returns the latest completed request's `02_response.json`.

## Database Tables

The SQLite database is `data/database/tasks.db`.

Important tables：

- `tasks`：task queue, status, `task_id`, `task_timestamp`, `json_path`, schema flags.
- `task_stage_runs`：stage started/completed/duration/error.
- `task_timing_events`：stage-internal timed blocks.
- `ai_model_timings`：model initialization and per-task model timing.
- `model_bounds`：spatial query index.
- `display_objects`, `capture_instances`, `capture_binding_logs`：display identity management.
- `history_placement_requests`：request-level history placement runs.
- ArUco tables：reference and marker registry.

`tasks.status` includes upload boundary states (`uploading`, `upload_failed`) plus worker stages and terminal states.

## Shigurei Sidecar

`run_shigure_history_recorder.py` runs as a sidecar under worker startup. It writes to：

```text
data/shigure_history_cache/chunks/
data/shigure_history_cache/yolo_payloads/
data/shigure_history_cache/recorder_status.json
```

History placement, taken detection and SAM3D body stages read this cache plus marker history under `data/aruco/shigure_marker_history/`.

## Cleanup Rules

- Model task cleanup: remove `data/model/<task_timestamp>/` and update DB according to maintenance policy.
- ArUco processing cleanup: remove `data/aruco_processing/<task_timestamp>/`.
- History request cleanup: remove `data/history_placement_requests/<request_timestamp>/` and mark request cancelled/failed/completed in DB.
- Do not treat `data/output/` as the source of truth for new task results; it is scratch/compat storage.
