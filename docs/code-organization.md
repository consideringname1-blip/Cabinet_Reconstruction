# 代码组织说明

更新日期：2026-07-12
状态：当前协议

## 服务端核心

```text
code/
  server_api.py
  task_worker.py
  task_db.py
  task_json.py
  config.py
  path_config.py
  artifact_layout.py
  coordinate_systems.py
  spatial_transforms.py
  foundationpose_dispatcher.py
```

职责：

- `server_api.py`：严格 HTTP 输入、canonical Unity 输出和 task artifact 下载。
- `task_worker.py`：主 stage 队列、GPU/socket service、Shigure recorder、实时追踪与辅助分支生命周期。
- `task_db.py`：任务、ArUco reference、持久 display identity、模型/pose/body revision 和 tracking 事件。
- `task_json.py`：只按数据库/显式路径读写 task JSON。
- `config.py`：模型后端、容量和判定阈值。
- `path_config.py`：脚本、第三方运行库和解释器路径。
- `artifact_layout.py`：唯一 artifact 目录与文件命名。
- `coordinate_systems.py` / `spatial_transforms.py`：轴定义和跨设备变换。
- `model_generation_common.py`：InstantMesh 与 SAM3D Objects 的共享输出契约。
- `display_identity.py`：HoloLens capture 与 `display_object_id`/revision 持久化。
- `foundationpose_dispatcher.py`：HoloLens 高优先级和 realtime 低优先级 FoundationPose 调度。
- `realtime_tracking.py`：最多 5 个活动对象的 ephemeral latest-wins 状态。
- `shigure_projection.py` / `shigure_identity.py`：投影圆、depth 过滤、DINO 和临时 Shigure ID 绑定。
- `shigure_realtime_tracking.py`：Shigure 移动事件到 FoundationPose 与 pose journal。
- `shigure_auxiliary_branch.py`：contact evidence 与 SAM3D Body 并行分支。

## Object reconstruction stages

```text
code/stages/hololens3d_reconstruction/
  model_generation_common.py
  object_alignment_common.py
  model_bounds.py
  display_identity.py
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
  run_model_bounds_from_json.py
  run_display_identity_from_json.py
  bake_runtime_mesh.py
  postprocess_sam3d_glb.py
  convert_obj_to_fbx.py
```

主队列 stage 顺序只在 `task_worker.STAGE_ORDER` 定义。两个模型生成脚本都占用统一的 `model_generation` slot；具体脚本由 `MODEL_GENERATION_BACKEND` 选择。

## ArUco

```text
code/stages/hololens_aruco_reference/
  aruco_common.py
  run_aruco_detect_from_json.py
  run_aruco_sync_from_json.py
```

职责是读取 marker registry、处理 HoloLens PV frames、保存 startup latest reference，并 retro-sync 同 startup 模型。Unity 不处理 ArUco 坐标。

## Shigure ingress

```text
code/stages/shigure_history/
  realtime_tracking.py
  shigure_projection.py
  shigure_identity.py
  shigure_realtime_tracking.py
  shigure_auxiliary_branch.py
  shigure_latest_mask_overlay.py
  run_shigure_history_recorder.py
  cache.py
  settings.py
  marker_history.py
```

该目录只负责远端 ROS topic 规范化、RGB-D 内存缓存、exact-stamp contact/detection event、Unix socket 和 Shigure camera marker pose。本仓库不修改 `code/reconstruction/shigure_core`。

## SAM3D Body

```text
code/stages/sam3d_body_mesh/
  run_sam3d_body_mesh_from_json.py
  export_selected_body_fbx.py
  settings.py
```

该模块由 `shigure_contact_body` 辅助分支调用，输入 `ShigureContactEvidence`，输出 `SAM3DBodyMesh`、人体 FBX 和 subject crop。它不属于主队列 stage。

## Unity scripts

```text
H2AI/Assets/Scripts/ShuJuQingQiu.cs
H2AI/Assets/Scripts/LoadModel.cs
H2AI/Assets/Scripts/RuntimeModelManager.cs
H2AI/Assets/Scripts/RuntimeSpatialBoxDisplay.cs
H2AI/Assets/Scripts/ObjectEvidenceDisplay.cs
H2AI/Assets/Scripts/RuntimeModelEventIdentity.cs
H2AI/Assets/Scripts/ModelEventDisplay.cs
H2AI/Assets/Scripts/HoloLensDepthAquirer.cs
H2AI/Assets/Scripts/HoloLensPVAquirer.cs
```

- `ShuJuQingQiu.cs`：严格上传、任务轮询、History/Live 握手、实时 status 和模型下发。
- `RuntimeModelManager.cs`：最多 5 个可见对象与 5 个本地 cache bundle，按 display identity/revision 管理模型。
- `ObjectEvidenceDisplay.cs`：按 `display_object_id + body_revision` 显示 canonical body evidence。
- `RuntimeSpatialBoxDisplay.cs`：只显示 pending preview。
- 采集脚本只负责 HoloLens 输入边界，不做 ArUco 转换。

## 修改规则

- API 或 Unity payload 改动同时更新 `server-processing-output-contract.md`。
- 跨坐标数学统一放入 `spatial_transforms.py`；stage 不复制轴翻转。
- artifact 路径统一由 `artifact_layout.py` 生成。
- 新阈值放在 `config.py` 或对应 stage `settings.py`。
- 新的模型生成实现必须写入 canonical `ModelGeneration`/`RuntimeMesh`/`Blender`，不得增加第三套 Unity 协议。
