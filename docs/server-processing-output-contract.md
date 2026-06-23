# Server Processing Input / Output Contract

这份文档按当前代码实际情况记录服务器 pipeline 的输入、输出、稳定 JSON 字段和调试文件。目标不是解释算法细节，而是让维护者能快速判断：

- 一个 stage 最小需要哪些输入才能运行。
- 这个 stage 会写回哪些 task JSON 字段。
- 会产生哪些图片、mesh、JSON、NPY/NPZ、数据库记录。
- 哪些输出是客户端/下游稳定依赖，哪些只是调试或可选产物。

坐标系细节见 `docs/coordinate-systems.md`。Shigurei ROS2 topic 和消息格式见 `docs/shigure-core-ros-messages.md`。拿取判断算法细节见 `docs/拿取判断状态机.md`。SAM3D Body 设计细节见 `docs/sam3dbody相关.md`。

## Pipeline Overview

所有正式 stage 都以同一个 task JSON 为交换边界。`server_api.py` 在 `/generate` 中创建 `data/upload/<task_name>_meta.json`，worker 后续只读写这个 JSON，并把大文件放到 `data/output/`、`data/aruco/` 或 Shigurei history 目录。

### Server Startup

常规服务启动入口：

```bash
cd /workspace_whs
python code/run_server.py
```

`run_server.py` 从 `code/config.py` 读取目标 Python 和入口脚本，并在启动子进程前自动 `source /workspace_whs/setup_env.sh`。因此 server 子进程会获得项目 `PYTHONPATH`、ROS2 Humble setup、Shigurei receive workspace setup、`ROS_DOMAIN_ID` 和 `ROS_LOCALHOST_ONLY`。外层 shell 已经 source 过也没有冲突，因为项目路径会先去重再重新写入。

这个自动 source 只影响 `run_server.py` 创建的子进程，不会修改调用者所在的父 shell。手动跑独立脚本时仍可以按需 `source ./setup_env.sh`。

### Object Reconstruction 顺序

当前 `task_worker.STAGE_ORDER`：

```text
hololens2depth
sam3mask
instantmesh              # 这是共享模型生成槽；可由 InstantMesh 或 SAM3D Objects 实际执行
depthpointcloud
modelscale
object_alignment
runtime_mesh
pose
aruco_sync
blender
model_bounds
history_placement_restoration
taken_object_detection
sam3d_body_mesh
completed
```

### ArUco Reference 顺序

```text
aruco_detect
aruco_completed
```

### Shigurei Sidecar

Shigurei RGB-D history recorder 不是普通 task stage，它随 worker 自动启停：

```text
run_shigure_history_recorder.py
```

它持续写最近 RGB-D history，并在启动后尝试更新 Shigurei ArMarker history。普通 task 在 `history_placement_restoration` / `taken_object_detection` / `sam3d_body_mesh` 阶段读取这些 sidecar 数据。

启动链路是：`server_api.py` import 时调用 `task_worker.start_worker()`，worker 启动 recorder 子进程；recorder 如发现尚未处于 ROS2 Python 环境，会自己 source `code/ros2/shigure_recv_ws/setup_env.sh` 并用 ROS Python 重新 exec 自身。

## Runtime Roots And HTTP Folders

默认路径来自 `code/config.py`。

```text
data/upload/                         # 上传图片、原始 depth、task JSON
data/database/tasks.db               # task queue、stage runs、ArUco/model bounds index
data/output/hololens2/               # HoloLens depth -> PV aligned depth
data/output/sam3/                    # SAM3 mask/color/depth/overlay
data/output/instantmesh-input/       # InstantMesh 白底输入图
data/output/instant-mesh-large/      # InstantMesh mesh/image/video
data/output/sam3d-objects/meshes/    # SAM3D Objects raw/processed mesh
data/output/object_alignment/        # 可选对齐 preview/debug
data/output/runtime_mesh/            # runtime OBJ/MTL/PNG
data/output/blender/fbx/             # 物体最终 FBX
data/output/history_placement_restoration/ # 历史摆放再现状态、备份和 debug
data/output/taken_object_detection/  # 拿取判断结果帧备份和 debug
data/output/sam3d_body/              # 人体 mesh/FBX 输出
data/shigure_history_cache/          # chunked Shigurei RGB-D + YOLO history
data/aruco/runtime/                  # HoloLens ArUco reference record
data/aruco/shigure_marker_history/   # Shigurei camera 下的 ArMarker pose history
```

`/files/<folder>/<filename>` 只暴露 `FOLDER_MAP` 中登记的目录：

```text
meshes               -> data/output/instant-mesh-large/meshes
images               -> data/output/instant-mesh-large/images
videos               -> data/output/instant-mesh-large/videos
sam3d_object_meshes  -> data/output/sam3d-objects/meshes
runtime_meshes       -> data/output/runtime_mesh
fbx                  -> data/output/blender/fbx
history_placement_restoration -> data/output/history_placement_restoration
sam3d_body_meshes    -> data/output/sam3d_body/meshes
sam3d_body_fbx       -> data/output/sam3d_body/fbx
```

## Task JSON Baseline

### `/generate` for `object_reconstruction`

Minimum multipart form input:

```text
purpose=object_reconstruction                  # 可省略，默认 object_reconstruction
deviceJ                                        # JSON object
PVCameraJ                                      # JSON object
DepthCameraJ                                   # JSON object
SelectionBoxJ                                  # JSON object
pv_image                                       # PNG bytes
depth_image                                    # PNG bytes
```

Minimum fields inside JSON objects:

```json
{
  "deviceJ": {
    "type": "HoloLens2",
    "ip": "...",
    "time": "ISO time, optional but useful for Shigurei tracking",
    "pose": [0.0, 0.0, 0.0],
    "startup_session_id": "..."
  },
  "PVCameraJ": {
    "width": 640,
    "height": 360,
    "k": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "pose": [[... 4x4 ...]],
    "time": "optional"
  },
  "DepthCameraJ": {
    "pose": [[... 4x4 ...]],
    "sensor": "AHAT | LONGTHROW | ..."
  },
  "SelectionBoxJ": {
    "top_left": [0.0, 0.0],
    "bottom_right": [1.0, 1.0]
  }
}
```

Files written immediately:

```text
data/upload/<task_name>_color.png
data/upload/<task_name>_depth.png
data/upload/<task_name>_meta.json
```

Initial task JSON fields:

```json
{
  "server_received_utc": "...Z",
  "task_name": "YYYYmmdd_HHMMSS_microZ",
  "purpose": "object_reconstruction",
  "device": {"type": "", "ip": "", "time": "", "pose": null, "startup_session_id": ""},
  "PVCamera": {
    "name": "<task_name>_color.png",
    "width": 640,
    "height": 360,
    "k": [[...]],
    "pose": [[...]],
    "position": [0, 0, 0],
    "rotation_quaternion_xyzw": [0, 0, 0, 1]
  },
  "PVCameraFrames": [{"frame_index": 0, "name": "<task_name>_color.png", "png_bytes": 0}],
  "DepthCamera": {
    "name": "<task_name>_depth.png",
    "pose": [[...]],
    "sensor": "LONGTHROW",
    "stats": {"valid_depth_pixels": 0}
  },
  "SelectionBox": {"top_left": [0, 0], "bottom_right": [1, 1]},
  "task_id": "uuid"
}
```

### `/generate` for `aruco_reference`

Minimum multipart form input:

```text
purpose=aruco_reference
deviceJ
PVCameraFramesJ                                # JSON array, one or more frame objects
pv_image or pv_image_0, pv_image_1, ...         # PNG bytes for each frame
```

No `DepthCameraJ` or `SelectionBoxJ` is required. Files written:

```text
data/upload/<task_name>_color_000.png
data/upload/<task_name>_color_001.png
data/upload/<task_name>_meta.json
```

## Sidecar: Shigurei RGB-D History

### `shigure_history` recorder

Script: `code/stages/shigure_history/run_shigure_history_recorder.py`

Minimum runtime input:

- ROS2 topics from `code/stages/shigure_history/settings.py`:
  - RGB compressed image, default `/rs/color/compressed`
  - aligned depth compressed image, default `/rs/aligned_depth_to_color/compressedDepth`
  - CameraInfo, default `/rs/aligned_depth_to_color/cameraInfo`
  - YOLO/segmentation JSON, default `/tracking/active_objects`
- ROS environment from `code/ros2/shigure_recv_ws/setup_env.sh`; normal server startup bootstraps this automatically through the recorder.

Stable output layout in `data/shigure_history_cache/`:

```text
chunks/
  <chunk_start_sample_key>/
    rgb.mp4                  # H.264 RGB video, default CRF 18
    depth.mkv                # FFV1 lossless uint16 depth video
    camera_info.json         # latest CameraInfo payload for the chunk
    chunk_manifest.json      # frame stamps, frame indexes, yolo_hash refs
yolo_payloads/
  <sha256>.json              # deduplicated /tracking/active_objects JSON with mask_b64
recorder_status.json
```

Default retention and encoding settings:

- `SHIGURE_HISTORY_HZ=5.0`
- `SHIGURE_HISTORY_CHUNK_SECONDS=10.0`, so a full chunk normally contains 50 frames
- `SHIGURE_HISTORY_SECONDS=600.0`, so the ring buffer keeps about 10 minutes
- `SHIGURE_HISTORY_DECODED_CHUNK_CACHE_MAX=5`, controlling the in-memory decoded chunk LRU

`chunk_manifest.json` records each sampled frame:

```json
{
  "chunk_id": "<chunk_start_sample_key>",
  "fps": 5.0,
  "width": 1280,
  "height": 720,
  "frame_count": 50,
  "start_seconds": 0.0,
  "end_seconds": 0.0,
  "rgb_video": "rgb.mp4",
  "depth_video": "depth.mkv",
  "camera_info": "camera_info.json",
  "frames": [
    {
      "frame_index": 0,
      "stamp": {"sec": 0, "nanosec": 0},
      "sample_key": "0000000000_000000000",
      "headers": {},
      "topic_counts": {},
      "yolo_hash": "..."
    }
  ]
}
```

The read side is `stages.shigure_history.cache.ShigureRgbdCache`. Large range reads should use `iter_samples(start=..., end=...)`; it filters manifests first, then decodes each overlapping chunk once and serves frames from an in-memory LRU. Single-frame callers can use `get_sample(stamp, mode="nearest")`. Active deletion of decoded frames is `clear_decoded_cache()`. No decoded frame cache is written to disk.

Pruning deletes old whole chunk directories and then removes YOLO payloads that are no longer referenced by any remaining manifest. This reference-based cleanup preserves a YOLO result for every retained RGB-D frame even when `/tracking/active_objects` publishes more slowly than RGB-D.

Important negative contract:

- 不再写扁平 `<timestamp>_rgb.png` / `<timestamp>_depth.png` / `<timestamp>_meta.json` cache。
- 不把 decoded chunk 帧缓存到磁盘；读取时只进入内存 LRU。
- 不写 people detection。
- 不写 skeleton / wrist / contact。
- 不写事件判断状态。

### Shigurei ArMarker history

Script/helper: `code/stages/shigure_history/marker_history.py`

Minimum input:

- Shigurei RGB sample + CameraInfo sample from history recorder.
- ArUco marker config from `data/aruco/reference/aruco.json` or DB marker registry.
- OpenCV with `cv2.aruco`.

Trigger:

- Recorder 启动后，对后续 RGB-D 样本尝试检测 marker。
- 默认累计约 5 个可接受检测后融合并更新 latest。
- 失败不阻塞 RGB-D cache。

Stable output:

```text
data/aruco/shigure_marker_history/latest_marker_6d_pose.json
data/aruco/shigure_marker_history/history/<sample_key>_marker_6d_pose.json
```

Important JSON fields:

```json
{
  "source": "shigure_history_marker_warmup_fused",
  "observation_count": 5,
  "sample_key": "0000000000_000000000",
  "marker_id": 1,
  "dictionary": "DICT_7X7_1000",
  "marker_size_mm": 200.0,
  "reprojection_error_px": 0.2,
  "corner_area_px": 1234.0,
  "opencv_camera_pose": {
    "position": [0.0, 0.0, 1.0],
    "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "rvec": [0.0, 0.0, 0.0],
    "tvec_m": [0.0, 0.0, 1.0],
    "coordinate_system": "+X right, +Y down, +Z forward; units are meters"
  },
  "unity_camera_pose": {"position": [0.0, 0.0, 1.0]},
  "source_observations": []
}
```

Downstream rule:

- 所有 Shigurei camera -> ArMarker 逻辑读取这个 latest/history 文件。
- 不扫描 `.test`、旧 fusion 输出或每帧 Shigurei cache。

## Stage Contracts: Object Reconstruction

### 1. `hololens2depth`

Runner: `code/Hololens2/DepthConvertToRGB/align_pv_depth.py`

Minimum input:

- `PVCamera.name`, `PVCamera.k`, PV size.
- `DepthCamera.name`, `DepthCamera.pose`, `DepthCamera.sensor`.
- Upload files:
  - `data/upload/<task_name>_color.png`
  - `data/upload/<task_name>_depth.png`

Output files:

```text
data/output/hololens2/<task_name>_align_depth.png          # uint16, mm, PV image size
data/output/hololens2/<task_name>_align_depth_turbo.png    # colored visualization
```

Task JSON output:

```json
{
  "DepthCamera": {
    "align_depth_name": "<task_name>_align_depth.png",
    "align_depth_stats": {
      "sensor": "LONGTHROW",
      "min_depth_mm": 200,
      "max_reliable_depth_mm": 7500,
      "valid_depth_pixels": 0
    },
    "align_depth_turbo_name": "<task_name>_align_depth_turbo.png"
  }
}
```

Minimum downstream dependency:

- `DepthCamera.align_depth_name` for `sam3mask`, `depthpointcloud`, alignment stages.

Debug-only output:

- `align_depth_turbo_name` is visualization only.

### 2. `sam3mask`

Runner: `code/stages/hololens3d_reconstruction/run_sam3_boxmask_from_json.py`

Minimum input:

- `PVCamera.name`, `PVCamera.width`, `PVCamera.height`.
- `DepthCamera.align_depth_name`.
- `SelectionBox.top_left`, `SelectionBox.bottom_right` normalized to `[0,1]`.
- Files:
  - `data/upload/<PVCamera.name>`
  - `data/output/hololens2/<DepthCamera.align_depth_name>`

Output files in `data/output/sam3/`:

```text
<task_name>_sam3_mask.png       # binary/object mask
<task_name>_sam3_color.png      # RGBA masked object color
<task_name>_sam3_depth.png      # masked aligned depth
<task_name>_sam3_overlay.png    # preview overlay
```

The stage also estimates a quick display box from the SAM3 mask and aligned depth.
It is intentionally robust/approximate, using valid mask depth percentiles and PV
camera intrinsics/pose to produce a Unity-world AABB for immediate HoloLens
feedback.

Task JSON output:

```json
{
  "sam3Name": {
    "mask": "<task_name>_sam3_mask.png",
    "color": "<task_name>_sam3_color.png",
    "depth": "<task_name>_sam3_depth.png",
    "overlay": "<task_name>_sam3_overlay.png"
  },
  "Sam3SpatialBox": {
    "status": "ready",
    "coordinate_space": "unity_world",
    "aabb_min_world": [0.0, 0.0, 0.0],
    "aabb_max_world": [0.0, 0.0, 0.0],
    "center_world": [0.0, 0.0, 0.0],
    "size_world": [0.0, 0.0, 0.0],
    "source": "sam3_mask_aligned_depth_percentile"
  }
}
```


Unity display note:

- Completed model and spatial-query responses include `sam3_spatial_box` and `model_instance.sam3_spatial_box` when `Sam3SpatialBox.status == "ready"`.
- The HoloLens client renders this as a semi-transparent filled box plus bright wireframe.
- The progress panel is placed 20 cm outside the box face nearest the viewer, billboards toward the camera, and only changes face after the user remains clearly on another side.

Minimum downstream dependency:

- `sam3Name.mask`, `sam3Name.color`, `sam3Name.depth`.

Debug-only output:

- `sam3Name.overlay`.

### 3. `instantmesh` / model generation slot

Worker stage name: `instantmesh`

Actual backend selected by `MODEL_GENERATION_BACKEND`:

- `instantmesh`: `run_instantmesh_from_json.py`
- `sam3d_objects`: `run_sam3d_objects_from_json.py`

#### Backend A: InstantMesh

Minimum input:

- `sam3Name.color`.
- File `data/output/sam3/<sam3Name.color>`.

Intermediate file:

```text
data/output/instantmesh-input/<sam3Name.color>      # white-background RGB input
```

Output files:

```text
data/output/instant-mesh-large/meshes/<stem>.obj
data/output/instant-mesh-large/meshes/<stem>.mtl
data/output/instant-mesh-large/images/<stem>.png
data/output/instant-mesh-large/videos/<stem>.mp4    # only when enabled
```

Optional cleanup output:

```text
data/output/instant-mesh-large/meshes/<stem>_clean.obj
```

Task JSON output:

```json
{
  "InstantMesh": {
    "mesh": "<stem>.obj or <stem>_clean.obj",
    "mtl": "<stem>.mtl",
    "image": "<stem>.png",
    "video": "<stem>.mp4 or null",
    "video_render_enabled": true,
    "raw_mesh": "<stem>.obj",
    "cleanup": {"removed_faces": 0}
  },
  "ModelGeneration": {
    "backend": "instantmesh",
    "source_stage": "InstantMesh",
    "mesh_folder": "meshes",
    "mesh": "...obj",
    "mtl": "...mtl",
    "image": "...png",
    "video": "...mp4",
    "video_folder": "videos",
    "runtime_ready": false
  }
}
```

Minimum downstream dependency:

- `ModelGeneration.mesh`, `mtl`, `image`, `mesh_folder`, `source_stage`.

Debug/optional:

- `video`, `cleanup`, `raw_mesh`.

#### Backend B: SAM3D Objects

Minimum input:

- `sam3Name.color` and `sam3Name.mask`.
- `SAM3D_OBJECTS_CONFIG` exists.

Output files in `data/output/sam3d-objects/meshes/`:

```text
<stem>_sam3d_raw.glb
<stem>_sam3d_processed.obj
<stem>_sam3d_processed.mtl
<stem>_sam3d_processed.png
<stem>_sam3d_processed_postprocess.json
```

Task JSON output:

```json
{
  "SAM3DObjects": {
    "backend": "sam3d_objects",
    "source_stage": "SAM3DObjects",
    "mesh_folder": "sam3d_object_meshes",
    "mesh": "<stem>_sam3d_processed.obj",
    "mtl": "<stem>_sam3d_processed.mtl",
    "image": "<stem>_sam3d_processed.png",
    "runtime_ready": false,
    "raw_glb": "<stem>_sam3d_raw.glb",
    "source_color": "...",
    "source_mask": "...",
    "config": "...pipeline.yaml",
    "seed": 42,
    "attn_backend": "sdpa",
    "postprocess": {}
  },
  "ModelGeneration": {"same_shape_as": "SAM3DObjects"}
}
```

Minimum downstream dependency:

- same `ModelGeneration` fields as InstantMesh.

Debug/optional:

- raw GLB and postprocess JSON.

### 4. `depthpointcloud`

Runner: `run_depthpointcloud_from_json.py`

Minimum input:

- `PVCamera.k`.
- `sam3Name.mask`.
- `DepthCamera.align_depth_name` and sensor limits.

No stable files are written by this stage in current code; it computes measurements and writes JSON.

Task JSON output:

```json
{
  "depthpointcloud": {
    "depth_sensor": "LONGTHROW",
    "min_depth_mm": 200,
    "max_reliable_depth_mm": 7500,
    "width_pointcloud_units": 0.0,
    "height_pointcloud_units": 0.0,
    "depth_pointcloud_units": 0.0,
    "real_width_measured": 0.0,
    "real_height_measured": 0.0,
    "mean_depth_measured": 0.0,
    "raw_point_count": 0,
    "point_count": 0,
    "valid_depth_ratio": 0.0,
    "mask_bbox_xyxy": [0, 0, 0, 0],
    "mask_pixels": 0,
    "valid_depth_pixels": 0,
    "discarded_count": 0,
    "used_count": 0
  }
}
```

Minimum downstream dependency:

- `real_width_measured`, `real_height_measured`, `mean_depth_measured`, point counts and mask bbox for scale/alignment diagnostics.

Debug/optional:

- crop/border statistics are diagnostics.

### 5. `modelscale`

Runner: `run_model_scale_from_json.py`

Minimum input:

- `depthpointcloud.real_width_measured`, `depthpointcloud.real_height_measured`.
- Generated model OBJ from `ModelGeneration`.

No files are written.

Task JSON output:

```json
{
  "model": {
    "width_model_units": 0.0,
    "height_model_units": 0.0,
    "depth_model_units": 0.0,
    "width_measured": 0.0,
    "height_measured": 0.0,
    "width_scale": 1.0,
    "height_scale": 1.0,
    "overall_scale": 1.0
  }
}
```

Minimum downstream dependency:

- `model.overall_scale`.

### 6. `object_alignment`

Runner: `run_object_alignment_from_json.py`

Minimum input:

- `ModelGeneration` mesh/mtl/image source.
- `model.overall_scale`.
- `depthpointcloud` measurements/point data resolvable from current task paths.
- `PVCamera`, aligned depth, SAM3 mask.
- Runtime setting `OBJECT_ALIGNMENT_MODE` (`foundationpose`, `camera_refine`, or `off`).

Output files:

- By default, no required stable files are written.
- If `ENABLE_ALIGNMENT_RENDER_OUTPUTS=True`, preview files are written under `data/output/object_alignment/`:

```text
<task_name>_alignment_preview_pointcloud_model.png
<task_name>_alignment_preview_model_compare.png
```

Task JSON output:

```json
{
  "object_alignment": {
    "camera_local_position": [0.0, 0.0, 0.0],
    "camera_local_rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    "model_real_scale": 1.0,
    "alignment_mode": "foundationpose",
    "alignment_solver": "foundationpose",
    "alignment_backend": "foundationpose",
    "alignment_device": "cuda",
    "alignment_device_name": "...",
    "geometry_backend": "cpu",
    "icp_mode": "foundationpose",
    "icp_enabled": false,
    "preview_image_name": null,
    "preview_image_unaligned_name": null,
    "confidence": 0.0,
    "alignment_rmse": 0.0,
    "target_point_count": 0,
    "target_front_point_count": 0,
    "discarded_point_count": 0
  },
  "debug": {
    "pose_transform_stages": {
      "object_alignment": {
        "final_camera_local_rh": {}
      }
    }
  }
}
```

Minimum downstream dependency:

- `object_alignment.camera_local_position`.
- `object_alignment.camera_local_rotation_quaternion_xyzw`.
- `object_alignment.model_real_scale`.

Debug/optional:

- preview images, RMSE, confidence, point counts, `debug.pose_transform_stages.object_alignment`.

### 7. `runtime_mesh`

Runner: `run_runtime_mesh_from_json.py`, Blender script `bake_runtime_mesh.py`

Minimum input:

- `ModelGeneration` mesh/mtl/image.
- Blender executable.
- Runtime mesh settings from `hololens3d_reconstruction/settings.py`.

Output files in `data/output/runtime_mesh/`:

```text
<stem>_runtime_<ratio>_<texture_size>.obj
<stem>_runtime_<ratio>_<texture_size>.mtl
<stem>_runtime_<ratio>_<texture_size>.png
```

If source is SAM3D Objects and already runtime-ready, files may be copied/reused with original names.

Task JSON output:

```json
{
  "RuntimeMesh": {
    "mesh": "...obj",
    "mtl": "...mtl",
    "image": "...png",
    "source_stage": "InstantMesh | SAM3DObjects",
    "source_backend": "instantmesh | sam3d_objects",
    "source_mesh_folder": "meshes | sam3d_object_meshes",
    "source_mesh": "...obj",
    "source_mtl": "...mtl",
    "source_image": "...png",
    "decimate_ratio": 0.0625,
    "texture_size": 1024,
    "bake_margin_px": 64,
    "uv_island_margin": 0.03,
    "original_vertices": 0,
    "original_faces": 0,
    "vertices": 0,
    "faces": 0
  }
}
```

Minimum downstream dependency:

- `RuntimeMesh.mesh`, `RuntimeMesh.mtl`, `RuntimeMesh.image`.

Debug/optional:

- vertex/face counts and bake parameters.

### 8. `pose`

Runner: `run_pose_from_json.py`

Minimum input:

- `object_alignment.camera_local_position`.
- `object_alignment.camera_local_rotation_quaternion_xyzw`.
- `object_alignment.model_real_scale`.
- `PVCamera.pose` or `PVCamera.position + rotation_quaternion_xyzw`.

No files are written.

Task JSON output:

```json
{
  "object_world": {
    "position": [0.0, 0.0, 0.0],
    "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    "scale": [1.0, 1.0, 1.0]
  },
  "debug": {
    "pose_transform_stages": {
      "pose_stage": {
        "camera_local_rh": {},
        "camera_local_unity": {},
        "pv_camera_world": {},
        "camera_rotation_filter": {},
        "runtime_asset": {},
        "final_object_world": {}
      }
    }
  }
}
```

Minimum downstream dependency:

- `object_world` for HoloLens world display before/without ArUco sync.

Debug/optional:

- `debug.pose_transform_stages.pose_stage`.

### 9. `aruco_sync`

Runner: `run_aruco_sync_from_json.py`

Minimum input:

- `device.startup_session_id`.
- Latest `aruco_references` DB row for that startup session.
- `object_world` with `scale` for full sync.

No files are written.

Task JSON output when reference exists but object pose is missing:

```json
{
  "aruco_reference": {"position": [0, 0, 0], "rotation_quaternion_xyzw": [0, 0, 0, 1]},
  "debug": {"pose_transform_stages": {"aruco_stage": {"sync_reason": "object_world_missing"}}}
}
```

Task JSON output when full sync succeeds:

```json
{
  "aruco_reference": {"position": [0, 0, 0], "rotation_quaternion_xyzw": [0, 0, 0, 1]},
  "object_aruco": {
    "position": [0.0, 0.0, 0.0],
    "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    "scale": [1.0, 1.0, 1.0]
  },
  "debug": {
    "pose_transform_stages": {
      "aruco_stage": {
        "sync_stage_ran": true,
        "synced_to_reference": true,
        "reference_task_id": "uuid",
        "reference_created_at": "UTC text",
        "sync_reason": "reference_applied",
        "object_aruco": {}
      }
    }
  }
}
```

Database output:

- Updates `tasks.aruco_coordinate_synced` true/false.

Minimum downstream dependency:

- `object_aruco` for `blender`, `model_bounds`, and later ArMarker display chain.

Debug/optional:

- `debug.pose_transform_stages.aruco_stage`.

### 10. `blender`

Runner: `run_blender_from_json.py`, Blender script `convert_obj_to_fbx.py`

Minimum input:

- Preferred source: `RuntimeMesh.mesh`, `RuntimeMesh.mtl`, `RuntimeMesh.image`.
- Fallback/source information via `ModelGeneration` when supported.
- `object_world` / `object_aruco` placement data depending source handling.
- Blender executable.

Output files in `data/output/blender/fbx/`:

```text
<runtime_or_source_stem>.fbx
```

SAM3D Objects FBX export may also bake texture files under `data/output/blender/fbx/` as postprocess assets.

Task JSON output:

```json
{
  "Blender": {
    "fbx": "<stem>.fbx",
    "source_stage": "RuntimeMesh | InstantMesh | SAM3DObjects",
    "source_mesh": "...obj",
    "source_mesh_folder": "runtime_meshes | meshes | sam3d_object_meshes",
    "source_backend": "runtime_mesh | instantmesh | sam3d_objects",
    "source_format": "obj_mtl_png",
    "source_texture": "...png",
    "postprocess": {"vertices": 0, "faces": 0}
  }
}
```

Minimum downstream dependency:

- `Blender.fbx`.

Debug/optional:

- `Blender.postprocess`.

### 11. `model_bounds`

Runner: `run_model_bounds_from_json.py`, logic in `model_bounds.py`

Minimum input:

- `Blender.fbx`.
- Final source OBJ, usually `RuntimeMesh.mesh` from `data/output/runtime_mesh/`.
- `object_aruco` with `position`, `rotation_quaternion_xyzw`, `scale`.

No new mesh/image files are written.

Task JSON output on success:

```json
{
  "ModelBounds": {
    "status": "ready",
    "coordinate_space": "aruco",
    "source_model_path": "data/output/runtime_mesh/<mesh>.obj",
    "aabb_min_aruco": [0.0, 0.0, 0.0],
    "aabb_max_aruco": [0.0, 0.0, 0.0],
    "corners_aruco": [[0.0, 0.0, 0.0]]
  }
}
```

Task JSON output when ArUco is missing:

```json
{
  "ModelBounds": {
    "status": "pending_reference",
    "coordinate_space": "aruco",
    "error_message": "object_aruco is missing; wait for a valid ArUco reference"
  }
}
```

Database output:

- Upserts `model_bounds` table with status, fbx name, object pose, AABB, corners.

Minimum downstream dependency:

- `ModelBounds.status == "ready"` and `corners_aruco` for taken-object projection and spatial queries.

Debug/optional:

- `source_model_path`, DB row metadata.


### 12. `history_placement_restoration`

Runner: `run_history_placement_restoration_from_json.py`

Purpose:

- Re-evaluate already modeled objects from Shigurei RGB-D + YOLO history without FoundationPose precise re-localization.
- Establish a baseline YOLO id/signature near the model's original ArUco placement.
- On the current or requested Shigurei frame, classify the object as:
  - `ORIGINAL`: still at the original placement.
  - `MOVED`: same object appears elsewhere; output a rough current ArUco pose for a floating regular polyhedron and a coarse current-to-original animation.
  - `MISSING`: not visible and depth indicates the original support area is empty/deeper.
  - `OCCLUDED_REUSE_LAST`: target id is temporarily lost but depth suggests occlusion; consumers may reuse the previous known result.
  - `UNKNOWN`: YOLO/signature/depth evidence conflicts. Unity should show the original mesh with a rotating octahedron above it; this is not treated as disappeared.

Output root:

```text
data/output/history_placement_restoration/<task_id>/
data/output/history_placement_restoration/<task_id>/<task_name>_baseline_<sample_key>/rgb.png
data/output/history_placement_restoration/<task_id>/<task_name>_baseline_<sample_key>/depth.png
data/output/history_placement_restoration/<task_id>/<task_name>_baseline_<sample_key>/active_objects.json
data/output/history_placement_restoration/<task_id>/<task_name>_current_<sample_key>/rgb.png
data/output/history_placement_restoration/<task_id>/state_visualization.png
data/output/history_placement_restoration/<task_id>/summary.json
```

Minimal task JSON:

```json
{
  "HistoryPlacementRestoration": {
    "status": "ORIGINAL | MOVED | MISSING | OCCLUDED_REUSE_LAST | UNKNOWN | SKIPPED",
    "target_object_id": "46",
    "baseline": {},
    "current": {},
    "classification": {},
    "display": {
      "mode": "original_only | current_polyhedron_to_original | restore_original_only | occluded_reuse_last | unknown_original_octahedron",
      "polyhedron": {
        "enabled": true,
        "shape": "cube | octahedron",
        "edge_length_m": 0.1,
        "pose_aruco": {}
      },
      "animation": {
        "enabled": true,
        "from_pose_aruco": {},
        "to_pose_aruco": {},
        "duration_seconds": 1.2
      }
    },
    "state_visualization_path": "data/output/history_placement_restoration/<task_id>/state_visualization.png"
  }
}
```

API trigger for the HoloLens button:

```text
POST /history-placement-restoration/start
```

Request fields:

```json
{
  "startup_session_id": "optional HoloLens startup session",
  "task_id": "optional specific model task id",
  "model_limit": 5,
  "target_time": "optional ISO timestamp"
}
```

`model_limit` defaults to `5`. `0` means no limit. The Unity side exposes one hand-menu toggle entry point, `StartHistoryPlacementRestoration()`: first click requests analysis from the server and renders the returned `display` contract, while the next click clears the generated history-restoration display. `historyPlacementRestorationModelLimit` remains editable in the Unity Inspector.

### 13. `taken_object_detection`

Runner: `run_taken_object_detection_from_json.py`

Minimum input:

- Task capture time, preferably `PVCameraFrames[].time`, `PVCamera.time`, `device.time`, or fallback `server_received_utc`.
- Shigurei RGB-D history in `data/shigure_history_cache/` covering capture time.
- Default YOLO-first mode requires:
  - deduplicated `/tracking/active_objects` payloads in Shigure history, with `object_id`, `bbox`, `x`, `y`, and full-frame `mask_b64`.
  - `ModelBounds.aabb_min_aruco` / `ModelBounds.aabb_max_aruco`, or fallback `object_aruco.position`, for object center projection.
  - latest Shigurei ArMarker history: `data/aruco/shigure_marker_history/latest_marker_6d_pose.json`.
- Legacy projected-mask mode is no longer the default. It is used only when explicitly selected or when `TAKEN_OBJECT_ENABLE_LEGACY_FALLBACK=true` after YOLO initialization failure. Its mask source priority is:
  - env `TAKEN_OBJECT_PROJECTED_MASK` or `TAKEN_OBJECT_PROJECTED_MASK_PATH`
  - `TakenObjectProjection.mask_path`
  - legacy `ModelEventTracking.current_support_mask_path`
  - `sam3Name.mask` resized fallback
  - `SelectionBox` scaled fallback, if enabled
- For downstream body stage, latest Shigurei ArMarker history is copied when available.

Output root:

```text
data/output/taken_object_detection/<task_id>/
```

YOLO-first initialization writes a post-capture stable-first-frame backup as soon as initialization succeeds. This backup is kept even when the final status later becomes `NOT_TAKEN`:

```text
data/output/taken_object_detection/<task_id>/<task_name>_yolo_init_<sample_key>/rgb.png
data/output/taken_object_detection/<task_id>/<task_name>_yolo_init_<sample_key>/depth.png
data/output/taken_object_detection/<task_id>/<task_name>_yolo_init_<sample_key>/camera_info.json
data/output/taken_object_detection/<task_id>/<task_name>_yolo_init_<sample_key>/active_objects.json
data/output/taken_object_detection/<task_id>/<task_name>_yolo_init_<sample_key>/yolo.json
data/output/taken_object_detection/<task_id>/<task_name>_yolo_init_<sample_key>/marker_6d_pose.json   # copied from Shigurei marker history when available
data/output/taken_object_detection/<task_id>/<task_name>_yolo_init_<sample_key>/meta.json             # backup_kind = yolo_init
```

On `TAKEN`, the final result-frame backup remains under the original path shape:

```text
data/output/taken_object_detection/<task_id>/<task_name>_<sample_key>/rgb.png
data/output/taken_object_detection/<task_id>/<task_name>_<sample_key>/depth.png
data/output/taken_object_detection/<task_id>/<task_name>_<sample_key>/camera_info.json
data/output/taken_object_detection/<task_id>/<task_name>_<sample_key>/active_objects.json
data/output/taken_object_detection/<task_id>/<task_name>_<sample_key>/yolo.json
data/output/taken_object_detection/<task_id>/<task_name>_<sample_key>/marker_6d_pose.json   # copied from Shigurei marker history when available
data/output/taken_object_detection/<task_id>/<task_name>_<sample_key>/meta.json             # backup_kind = result
```

Task JSON output statuses:

```text
RUNNING
TAKEN
NOT_TAKEN
INIT_FAILED
```

Minimal task JSON on success:

```json
{
  "TakenObjectDetection": {
    "status": "TAKEN",
    "result_timestamp": {"sec": 0, "nanosec": 0},
    "backup_shigurei_dir": "data/output/taken_object_detection/...",
    "init_backup_shigurei_dir": "data/output/taken_object_detection/.../<task_name>_yolo_init_<sample_key>",
    "tracking_window": {},
    "projection": {},
    "init": {},
    "depth_taken_timestamp": {"sec": 0, "nanosec": 0},
    "depth_confirm_timestamp": {"sec": 0, "nanosec": 0},
    "full_occlusion_start_timestamp": null,
    "used_full_occlusion_start": false,
    "rgb_backtrack": {},
    "checked_frame_count": 0,
    "output_dir": "data/output/taken_object_detection/<task_id>"
  }
}
```

Task JSON on no event / init failure:

```json
{
  "TakenObjectDetection": {
    "status": "NOT_TAKEN | INIT_FAILED",
    "reason": "...",
    "result_timestamp": null,
    "backup_shigurei_dir": null,
    "init_backup_shigurei_dir": "data/output/taken_object_detection/.../<task_name>_yolo_init_<sample_key> or null",
    "tracking_window": {},
    "projection": {},
    "init": {},
    "checked_frame_count": 0,
    "output_dir": "..."
  }
}
```

Debug files only when full output is enabled:

```text
data/output/taken_object_detection/<task_id>/debug/projected_mask.png
data/output/taken_object_detection/<task_id>/debug/trusted_mask.png
data/output/taken_object_detection/<task_id>/debug/reference_depth_m.npy
data/output/taken_object_detection/<task_id>/decisions.json
```

Minimum downstream dependency:

- `TakenObjectDetection.status == "TAKEN"`.
- `result_timestamp`.
- `backup_shigurei_dir` containing `rgb.png`, `depth.png`, `camera_info.json` for final taken events.
- `init_backup_shigurei_dir` containing the YOLO-stable first post-capture RGB-D frame and YOLO payload, when YOLO-first initialization succeeds.

Debug/optional:

- `tracking_window`, `projection`, `init`, RGB backtrack metadata, masks, `decisions.json`.

Configuration notes:

- `TAKEN_OBJECT_TRACKING_MODE=yolo_primary` is the default.
- Direct legacy mode requires `TAKEN_OBJECT_TRACKING_MODE=legacy_projected_mask` and `TAKEN_OBJECT_ENABLE_LEGACY_PROJECTED_MASK=true`.
- YOLO-first initialization looks back up to `TAKEN_OBJECT_YOLO_PRE_CAPTURE_UNIQUE_COUNT` unique YOLO payloads only as auxiliary id evidence, then waits up to `TAKEN_OBJECT_YOLO_INIT_HARD_TIMEOUT_SECONDS` after capture for stable post-capture YOLO.
- Large-range reads first scan chunk manifests and YOLO payload hashes. RGB-D chunks are decoded only for YOLO depth matching, initialization, depth confirmation, RGB backtracking, or legacy fallback.

### 14. `sam3d_body_mesh`

Runner: `run_sam3d_body_mesh_from_json.py`

Minimum input:

- `TakenObjectDetection.status == "TAKEN"`.
- `TakenObjectDetection.backup_shigurei_dir` with:
  - `rgb.png`
  - `depth.png`
  - `camera_info.json`
- Shigurei marker pose, either:
  - `<backup_shigurei_dir>/marker_6d_pose.json`, copied during taken detection, or
  - `data/aruco/shigure_marker_history/latest_marker_6d_pose.json`
- `ModelBounds` or `object_aruco` for object center.
- SAM3D Body checkpoint and assets under `code/reconstruction/sam3d-body/`.
- Blender executable for selected-person FBX export.

Output root:

```text
data/output/sam3d_body/meshes/<task_name>/
data/output/sam3d_body/fbx/
```

Per-person debug/working files:

```text
data/output/sam3d_body/meshes/<task_name>/person_<index>_armarker_mesh.npz
data/output/sam3d_body/meshes/<task_name>/people.json
```

Selected person stable files:

```text
data/output/sam3d_body/meshes/<task_name>/<task_name>_person_<index>_armarker.obj
data/output/sam3d_body/meshes/<task_name>/<task_name>_person_<index>_armarker.fbx_export.json
data/output/sam3d_body/fbx/<task_name>_person_<index>_body.fbx
```

Task JSON statuses:

```text
SKIPPED_NOT_TAKEN
NO_PERSON_DETECTED
NO_VALID_BODY_MESH
NO_VALID_WRIST_JOINT
SUCCESS
```

Task JSON on success:

```json
{
  "SAM3DBodyMesh": {
    "status": "SUCCESS",
    "result_timestamp": {"sec": 0, "nanosec": 0},
    "backup_shigurei_dir": "...",
    "selected_person_name": "person_0",
    "selected_person_fbx_path": "data/output/sam3d_body/fbx/<task>_person_0_body.fbx",
    "selected_person_fbx_folder": "sam3d_body_fbx",
    "selected_person_pose_armarker": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "selected_person_obj_path": "data/output/sam3d_body/meshes/<task>/<task>_person_0_armarker.obj",
    "material_color": [0.0, 0.0, 0.0],
    "material_alpha": 0.5,
    "coordinate_space": "armarker",
    "camera_to_armarker_source": "data/aruco/shigure_marker_history/latest_marker_6d_pose.json",
    "people_json_path": "data/output/sam3d_body/meshes/<task>/people.json",
    "people": [],
    "object_center_armarker": [0.0, 0.0, 0.0]
  }
}
```

Task JSON on skip/failure:

```json
{
  "SAM3DBodyMesh": {
    "status": "SKIPPED_NOT_TAKEN | NO_PERSON_DETECTED | NO_VALID_BODY_MESH | NO_VALID_WRIST_JOINT",
    "reason": "...",
    "result_timestamp": null,
    "backup_shigurei_dir": null,
    "output_dir": "data/output/sam3d_body/meshes/<task_name>"
  }
}
```

Minimum downstream dependency:

- `SAM3DBodyMesh.status == "SUCCESS"`.
- `selected_person_fbx_path` or `selected_person_fbx_folder + filename`.
- `selected_person_pose_armarker`, currently identity because vertices are already exported in ArMarker coordinates.

Debug/optional:

- `people`, `people.json`, per-person NPZ, depth offset, wrist distance, bbox.

## Stage Contracts: ArUco Reference

### `aruco_detect`

Runner: `run_aruco_detect_from_json.py`

Minimum input:

- `purpose == "aruco_reference"`.
- `PVCameraFrames` with image `name`, `k`, and pose/position/quaternion.
- Files `data/upload/<frame.name>`.
- Enabled marker config from DB or `data/aruco/reference/aruco.json`.
- OpenCV with `cv2.aruco`.

Output files:

```text
data/aruco/runtime/<task_name>/record.json
data/aruco/runtime/<task_name>/annotated_000.png
data/aruco/runtime/<task_name>/annotated_001.png
...
```

Task JSON output under debug:

```json
{
  "aruco_reference": {
    "position": [0.0, 0.0, 0.0],
    "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "debug": {
    "pose_transform_stages": {
      "aruco_stage": {
        "template_path": "data/aruco/reference/aruco.json",
        "configured": true,
        "anchor_marker_id": 1,
        "registered_marker_ids": [1],
        "detected": true,
        "detected_ids": [1],
        "matched_marker_id": 1,
        "short_circuit": true,
        "frame_count": 1,
        "detections": [],
        "raw_record_path": "data/aruco/runtime/<task>/record.json",
        "annotated_image_path": "data/aruco/runtime/<task>/annotated.png",
        "frames": [],
        "marker_pose_world": {},
        "anchor_pose_candidates": [],
        "retro_synced_completed_task_count": 0
      }
    }
  }
}
```

Database output:

- Inserts/updates `aruco_references` when anchor reference succeeds.
- May update `aruco_marker_relations` for non-anchor marker relations.
- Worker may retro-sync completed model tasks for same startup session by rerunning `aruco_sync` and `model_bounds`.

Minimum downstream dependency:

- `aruco_references.marker_pose_json` via DB.
- Task `aruco_reference` for API response.

Debug/optional:

- annotated images, detections list, relation updates, raw record.

## Database / Worker Outputs

### `tasks`

Main status values currently allowed:

```text
pending
hololens2depth
aruco_detect
sam3mask
instantmesh
depthpointcloud
modelscale
object_alignment
pose
aruco_sync
runtime_mesh
blender
model_bounds
history_placement_restoration
taken_object_detection
sam3d_body_mesh
completed
aruco_completed
failed
```

Terminal statuses:

```text
completed
aruco_completed
failed
```

### `task_stage_runs`

Each worker stage records:

```json
{
  "task_id": "uuid",
  "stage_name": "sam3mask",
  "status": "running | completed | failed",
  "started_at": "UTC text",
  "completed_at": "UTC text or null",
  "duration_ms": 0,
  "error_message": null
}
```

### `model_bounds`

Stable spatial query table populated by `model_bounds`:

```json
{
  "task_id": "uuid",
  "status": "pending | ready | failed | pending_reference",
  "uploaded_at": "UTC text",
  "model_name": "task_name",
  "fbx_name": "...fbx",
  "coordinate_space": "aruco",
  "aruco_reference_task_id": "uuid or null",
  "object_aruco_json": {},
  "aabb_min_aruco_json": [0.0, 0.0, 0.0],
  "aabb_max_aruco_json": [0.0, 0.0, 0.0],
  "corners_aruco_json": [[0.0, 0.0, 0.0]],
  "source_model_path": "data/output/runtime_mesh/...obj",
  "error_message": null
}
```

## API Response Outputs

### `/check-queue`

Pending response:

```json
{
  "ready": false,
  "pending": [{"task_id": "uuid", "status": "sam3mask", "purpose": "object_reconstruction"}]
}
```

Completed object response includes at least:

```json
{
  "ready": true,
  "status": "completed",
  "task": {
    "status": "completed",
    "task_id": "uuid",
    "purpose": "object_reconstruction",
    "terminal": true,
    "stage_runs": [],
    "aruco_coordinate_synced": true,
    "placement_status": "aruco_synced | world_temporary | missing_pose",
    "model_bounds": {},
    "model_generation": {},
    "history_placement_restoration": {},
    "taken_object_detection": {},
    "sam3d_body_mesh": {},
    "object_world": {},
    "object_aruco": {},
    "aruco_reference": {},
    "fbx_url": "http://host/files/fbx/<file>.fbx",
    "runtime_mesh_url": "http://host/files/runtime_meshes/<file>.obj"
  }
}
```

ArUco reference completed response includes:

```json
{
  "status": "aruco_completed",
  "purpose": "aruco_reference",
  "aruco_reference": {},
  "aruco_detected": true,
  "latest_model": {}
}
```

## Minimal Pipeline Requirements

### Minimal object model for Unity display

Required chain:

```text
/generate object_reconstruction
hololens2depth
sam3mask
instantmesh or sam3d_objects
depthpointcloud
modelscale
object_alignment
runtime_mesh
pose
blender
```

Minimum final fields/files:

```text
Blender.fbx                         -> data/output/blender/fbx/<file>.fbx
object_world                         -> pose in HoloLens/Unity world
RuntimeMesh mesh/mtl/image            -> optional runtime source/debug
```

### Minimal ArMarker placement and spatial query

Additional requirements:

```text
aruco_detect reference task completed for same startup_session_id
aruco_sync
model_bounds
```

Minimum final fields:

```text
aruco_reference
object_aruco
ModelBounds.status == ready
ModelBounds.corners_aruco
```

### Minimal taken-object detection

Additional requirements:

```text
Shigurei RGB-D history covers object capture time
Projected/tracking mask source exists
```

Minimum final fields/files:

```text
TakenObjectDetection.status == TAKEN
TakenObjectDetection.result_timestamp
TakenObjectDetection.backup_shigurei_dir/rgb.png
TakenObjectDetection.backup_shigurei_dir/depth.png
TakenObjectDetection.backup_shigurei_dir/camera_info.json
```

### Minimal SAM3D Body taker mesh

Additional requirements:

```text
TakenObjectDetection.status == TAKEN
Shigurei marker history latest_marker_6d_pose.json exists
ModelBounds or object_aruco gives object center
SAM3D Body checkpoint exists
Blender available
```

Minimum final fields/files:

```text
SAM3DBodyMesh.status == SUCCESS
SAM3DBodyMesh.selected_person_fbx_path
SAM3DBodyMesh.selected_person_pose_armarker == identity
```

## Debug Output Checklist

Useful debug files and fields by area:

```text
Upload/input:
  data/upload/<task>_meta.json
  data/upload/<task>_color.png
  data/upload/<task>_depth.png

Depth alignment:
  DepthCamera.align_depth_turbo_name
  DepthCamera.align_depth_stats

SAM3:
  sam3Name.overlay
  sam3Name.mask/color/depth

Model generation:
  InstantMesh.video
  InstantMesh.cleanup
  SAM3DObjects.raw_glb
  SAM3DObjects.postprocess

Scale/alignment:
  depthpointcloud.* counts/ratios/bbox
  object_alignment.confidence/rmse/point counts
  debug.pose_transform_stages.object_alignment
  optional object_alignment preview images

Pose/ArUco:
  debug.pose_transform_stages.pose_stage
  debug.pose_transform_stages.aruco_stage
  data/aruco/runtime/<task>/record.json
  data/aruco/runtime/<task>/annotated_*.png

Bounds:
  ModelBounds.source_model_path
  model_bounds DB row

Shigurei:
  data/shigure_history_cache/recorder_status.json
  data/aruco/shigure_marker_history/latest_marker_6d_pose.json
  data/aruco/shigure_marker_history/history/*.json

Taken detection:
  TakenObjectDetection.tracking_window/projection/init
  data/output/taken_object_detection/<task_id>/debug/*.png
  data/output/taken_object_detection/<task_id>/debug/reference_depth_m.npy
  data/output/taken_object_detection/<task_id>/decisions.json
  backup_shigurei_dir/meta.json

SAM3D Body:
  SAM3DBodyMesh.people
  people.json
  person_<index>_armarker_mesh.npz
  selected *_armarker.obj
  selected *.fbx_export.json
```
