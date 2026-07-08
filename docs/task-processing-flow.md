# 任务处理流程

更新日期：2026-07-08
状态：当前实现说明

本文描述服务器和 Unity 当前使用的任务流程。`docs/hwang-project-flow.drawio` 只作为数据流参考，实际触发、队列、缓存、失败处理以代码为准。

## 核心原则

- `task_id` 是 API、DB、Unity 侧追踪任务的主键。
- `task_timestamp` 是文件目录和 artifact 文件名的主键。
- HoloLens 上传 HoloLens 当前本地坐标；服务器负责转换到 ArUco/世界，再转换回当前 HoloLens 本地坐标下发。
- Unity 正式模型必须使用服务器下发的 `object_hololens_current`。缺失时不再本地 fallback 摆放。
- `Sam3SpatialBox` 只用于 pending preview，不作为历史模型或正式 runtime 位姿来源。
- Shigure 视角历史再现不走 3D 投影，使用保存的 fixed Shigure `old_rgb + old_depth + old_mask + camera_info` 直接比较。

## 目录布局

目录由 `code/artifact_layout.py` 管理：

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
  history_placement_requests/<request_timestamp>/
    worker/
    result/
    debug/
  database/tasks.db
  worker_sockets/
  shigure_history_cache/
  aruco/
  console_logs/
```

`data/upload/` 只视作旧数据/备份输入来源，不作为新任务接收缓存。

## 服务启动

`code/run_server.py` 启动 Flask API。`server_api.py` 初始化：

1. `ensure_artifact_roots()` 创建目录。
2. `initialize_task_table()` 创建或迁移 SQLite。
3. `start_worker()` 启动任务 worker。
4. worker 自动启动 Shigure history recorder sidecar，用 Unix socket 提供最近 RGB-D/object_detection/cache 数据。

## Object Reconstruction Stage 顺序

当前 `task_worker.STAGE_ORDER`：

```text
hololens2depth
sam3mask
historical_model_match
instantmesh
depthpointcloud
modelscale
object_alignment
pose
aruco_sync
runtime_mesh
model_bounds
display_identity
history_placement_restoration
taken_object_detection
sam3d_body_mesh
```

说明：

- `historical_model_match` 使用 DINOv2 识别历史模型。命中并允许复用时跳过 `instantmesh`，直接进入 `depthpointcloud`，后续仍重新计算点云、对齐、位姿和 runtime fbx。
- `instantmesh` 是模型生成 stage 名。当前默认后端是 InstantMesh；`sam3d_objects` 只作为可选后端或运行库来源，不是当前默认生成后端。
- `modelscale` 仍是独立 stage；`runtime_mesh` 内部负责 runtime mesh bake 和 FBX export。
- `pose -> aruco_sync` 之后，任务保存 `object_aruco`，后续所有历史跨启动使用 ArUco pose 转当前 HoloLens pose。

## 预览 3D Box

预览 3D box 由 `run_sam3_boxmask_from_json.py` 产生 `Sam3SpatialBox`：

- 使用 depth limits 过滤深度。
- 使用去边缘后的 mask/depth 点云。
- 深度方向只向相机后方扩展，前边不动，中心自动后移。
- 深度扩展倍数由 `config.PREVIEW_3D_BOX_DEPTH_EXPANSION_FACTOR` 控制，默认 `2.0`。
- 输出 `coordinate_space = unity_world`，仅供 pending preview 使用。

完成/历史/runtime 模型不得依赖 `Sam3SpatialBox` 放置。

## Shigure 相关 Stage

`taken_object_detection`：

1. 从 Shigure cache 取上传时刻附近的 RGB-D、camera_info、object_detection。
2. 将 HoloLens 深度点云生成的模型 box 投影到 Shigure 图像。
3. 选择覆盖投影 box 且命中 2D ray 的最小 object mask。
4. 保存 `old_rgb + old_depth + old_mask + camera_info` 作为历史再现 baseline。
5. 在 old_mask 内做拿取判断。

`history_placement_restoration`：

1. 读取 taken stage 保存的 baseline。
2. 获取当前 Shigure RGB-D。
3. 只在 old_mask 内比较 RGB Lab 和 depth delta。
4. 输出 still/missing/occluded/unknown 和 Unity 显示用 polyhedron。

`sam3d_body_mesh`：

1. 使用拿走那帧生成人体 mesh。
2. 基于 body mask + depth 只调整相对相机距离，并按距离变化缩放人体 mesh。
3. 选择离物体中心最近的人体/手腕。
4. 输出人体 mesh 和 subject crop。

## ArUco 更新与 Retro Sync

ArUco reference 完成后：

- 保存当前 `startup_session_id` 的最新 marker pose。
- 同次启动内已到 `aruco_sync` 之后或 completed 的模型会 retro-sync。
- retro-sync 只按 `startup_session_id` 查询同次启动任务，不修改其他启动批次。
- 更新逻辑从 `object_hololens_original` 重新计算 `object_aruco`，再按最新 marker pose 更新 `object_hololens_current`。

## API 触发关系

主要入口：

- `POST /generate`：上传 object reconstruction 或 aruco reference。
- `POST /check-queue`：Unity 查询任务状态和 completed model instance。
- `GET /aruco/latest-reference`：按 startup session 获取最新 ArUco reference。
- `GET /latest-completed-task-ids`：列出 completed 模型。
- `POST /history-placement-restoration/start`：启动历史再现请求。
- `GET /history-placement-restoration/latest`：获取最近 completed 历史再现结果。
- `POST /spatial-query/ray` 和 `/spatial-query/ray-range`：基于当前 HoloLens 坐标查询模型。

## 不再使用的旧逻辑

- Unity 正式模型不再从 `sam3_spatial_box` fallback 放置。
- Unity 正式模型不再在缺 pose 时放到相机前方。
- 历史再现不再扫描 Shigure 全图找 object mask。
- 历史再现不再用旧 YOLO baseline 重新恢复对象区域。
- HoloLens 本地不保存或消费 ArUco 坐标。
