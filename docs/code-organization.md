# 代码组织说明

更新日期：2026-07-08
状态：当前实现说明

本文说明主要模块职责和新增代码应放在哪里。

## 顶层目录

```text
code/
  server_api.py
  task_worker.py
  task_db.py
  config.py
  path_config.py
  artifact_layout.py
  coordinate_systems.py
  spatial_transforms.py
  depth_camera_config.py
  gpu_budget.py
  model_bounds.py
  object_alignment_common.py
  stages/
H2AI/Assets/Scripts/
docs/
data/
```

## 配置与路径

- `config.py`：运行参数、阈值、开关。可通过环境变量覆盖。
- `path_config.py`：脚本路径、第三方项目路径、Python/Blender 可执行文件。
- `artifact_layout.py`：artifact 根目录、task 文件路径和下载文件名。

原则：

- 阈值不要写死在 stage 内。
- 新 artifact 路径通过 `artifact_layout` helper 生成。
- 新 stage 脚本路径放 `path_config.py`。

## API 与 Worker

- `server_api.py`：HTTP API、response payload、公开坐标转换。
- `task_worker.py`：队列、stage 顺序、socket worker 管理、Shigure recorder 启动。
- `task_db.py`：任务表、ArUco reference、identity、history request 等 DB 访问。

公开给 Unity 的空间字段应在 `server_api.py` 转成 HoloLens current local。

## 坐标相关

- `coordinate_systems.py`：基础坐标系统定义。
- `spatial_transforms.py`：HoloLens/ArUco/Shigure pose、point、pixel-depth 变换。

不要在 stage 中新增重复的 camera intrinsics 解析或轴翻转。HoloLens2 hl2ss/RGB-D 对齐属于输入边界，不放入 `spatial_transforms.py`。

## Reconstruction Stage

```text
stages/hololens3d_reconstruction/
  run_sam3_boxmask_from_json.py
  run_historical_model_match_from_json.py
  run_dinov2_identity_from_json.py
  run_instantmesh_from_json.py
  run_sam3d_objects_from_json.py
  run_depthpointcloud_from_json.py
  run_model_scale_from_json.py
  run_object_alignment_from_json.py
  run_object_icp_alignment_from_json.py
  run_foundationpose_alignment_worker.py
  run_pose_from_json.py
  run_runtime_mesh_from_json.py
  bake_runtime_mesh.py
  postprocess_sam3d_glb.py
  convert_obj_to_fbx.py
  run_model_bounds_from_json.py
  run_display_identity_from_json.py
```

主要职责：

- mask/depth 生成。
- DINOv2 历史模型复用。
- 模型生成或复用。
- 点云、缩放、对齐、ICP/FoundationPose helper、pose。
- ArUco 同步后的 runtime mesh、bounds、identity。

## ArUco Stage

```text
stages/hololens_aruco_reference/
  run_aruco_detect_from_json.py
  run_aruco_sync_from_json.py
```

职责：

- 检测 ArUco reference。
- 保存 startup_session_id 最新 marker pose。
- 同次启动 retro-sync 已完成/后续任务。

## Shigure Stage

```text
stages/shigure_history/
  run_shigure_history_recorder.py
  cache.py
  settings.py
  marker_history.py
```

职责：

- ROS RGB-D/object_detection/camera_info 对齐。
- 1 分钟左右内存缓存。
- Unix socket 对外服务。
- marker pose 历史读取。

## Taken / History / Body

```text
stages/taken_object_detection/
stages/history_placement_restoration/
stages/sam3d_body_mesh/
  export_selected_body_fbx.py
```

职责：

- taken：选择 Shigure object old_mask，保存 baseline，判断拿走帧。
- history：fixed Shigure old_mask direct compare，输出状态 polyhedron。
- sam3d_body_mesh：人体 mesh、距离缩放、最近人体、subject crop。

## Unity Scripts

```text
H2AI/Assets/Scripts/ShuJuQingQiu.cs
H2AI/Assets/Scripts/LoadModel.cs
H2AI/Assets/Scripts/RuntimeModelManager.cs
H2AI/Assets/Scripts/HistoryPlacementRestorationDisplay.cs
H2AI/Assets/Scripts/RuntimeSpatialBoxDisplay.cs
H2AI/Assets/Scripts/SpatialHistoryPointerQuery.cs
H2AI/Assets/Scripts/HoloLensDepthAquirer.cs
H2AI/Assets/Scripts/HoloLensPVAquirer.cs
```

当前原则：

- 正式模型缺 `object_hololens_current` 不显示。
- pending preview 可用 `sam3_spatial_box`。
- 历史证据图优先 `subject_crop_url`。
- Unity 不处理 ArUco 坐标。

## 文档

`docs/hwang-project-flow.drawio` 是流程参考图，不记录触发细节。Markdown 文档记录当前实现和接口契约；当代码行为改变时，优先更新对应专题文档。

更多非主流程的 API、socket service、helper 脚本和 Unity 辅助脚本见 `implementation-coverage-notes.md`。
