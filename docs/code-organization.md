# 代码组织说明

更新日期：2026-07-15
状态：Shigure v3 当前协议

Shigure v3 的行为契约见 [`shigure-v3-runtime.md`](shigure-v3-runtime.md)。本页只说明代码位置与当前职责。

## 服务端核心

```text
code/
  server_api.py
  task_worker.py
  task_db.py
  task_json.py
  migrate_shigure_v2_data.py
  migrate_shigure_v3_data.py
  config.py
  path_config.py
  artifact_layout.py
  coordinate_systems.py
  spatial_transforms.py
  gpu_lease.py
  foundationpose_dispatcher.py
```

职责：

- `server_api.py`：严格 HTTP 输入、完整 display-object live snapshot、`kind=origin` 历史输出和 artifact 下载；raw box route 仅保留诊断兼容，不属于 HoloLens v3 链。
- `task_worker.py`：主 stage 队列、GPU/socket service、Shigure recorder 与 runtime 生命周期。
- `task_db.py`：任务、ArUco reference、display identity/model revision、schema v3 session/event/raw-ID alias binding，以及每物体最多 5 条持久 origin。
- `task_json.py`：只按数据库/显式路径读写 task JSON。
- `config.py`：模型后端、容量、debug 开关和判定阈值。
- `path_config.py`：当前脚本、第三方运行库和解释器路径。
- `artifact_layout.py`：唯一 artifact 目录与文件命名。
- `coordinate_systems.py` / `spatial_transforms.py`：轴定义和跨设备变换。
- `gpu_lease.py`：按预计峰值显存做 host-wide SQLite 原子 best-fit 预留；覆盖同一主机的独立 server/Gunicorn/Python 进程，并以 PID、boot ID 与进程启动时刻回收失效 lease，但不跨主机协调。
- `foundationpose_dispatcher.py`：最多 1 至 4 个弹性 backend 的共享优先队列、同物体 generation 合并与过期结果丢弃；尚无 socket 的槽位不领取任务，无 HoloLens 专用槽，运行中任务不可抢占。
- `model_generation_common.py`：InstantMesh 与 SAM3D Objects 的共享输出契约。
- `display_identity.py`：HoloLens capture 与 `display_object_id`/revision 持久化；HoloLens 图是长期 identity reference 的唯一来源。

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

主队列 stage 顺序只在 `task_worker.STAGE_ORDER` 定义。两个模型生成脚本占用统一 `model_generation` slot；具体脚本由 `MODEL_GENERATION_BACKEND` 选择。

## ArUco

```text
code/stages/hololens_aruco_reference/
  aruco_common.py
  run_aruco_detect_from_json.py
  run_aruco_sync_from_json.py
```

这里读取 marker registry、处理 HoloLens PV frames、保存 startup latest reference，并 retro-sync 同 startup 模型。Unity 不处理 ArUco 坐标。

## Shigure ingress 与 runtime

```text
code/stages/shigure_history/
  cache.py
  settings.py
  debug_cache.py
  shigure_compatibility.py
  shigure_identity.py
  spatial_box_v2.py
  shigure_runtime_v2.py
  marker_history.py
  raw_tracking_box_relay.py       # 诊断兼容，不接入 HoloLens v3 显示
  run_shigure_history_recorder.py
```

职责边界：

- recorder 只订阅并适配不可修改的远端 ROS topics，生成 exact-stamp canonical frame，并暴露内存 socket cache。
- compatibility adapter 处理稀疏 bring-in/take-out/obj_move、tracking、bbox-local mask 和 exact RGB-D；临时 raw ID 只在 source epoch 内有效。
- runtime 只用 HoloLens reference 做身份分类。已有 raw ID 继承绑定；新的实质不同 mask 才触发全库 DINOv2，并先登记为 alias。旧 primary 同帧仍可见时额外比较两张 mask 的 DINOv2 距离；仅在旧 mask 更优时保留旧 primary，否则显示最新的最优新 alias。启动恢复也采用同一逐 mask 线性策略，不做一对一全局 assignment。
- runtime 在首次可信绑定后最多尝试 5 次初始化 FoundationPose；之后只在选中的 `take_out` 上再运行 FoundationPose，不做连续 pose tracking。
- runtime 用 primary 的 exact mask/depth 反投影并在 ArUco 轴上生成大致 3D AABB。完整 snapshot 中 primary 消失会清除该 display object 的 box。
- `spatial_box_v2.py` 和 `raw_tracking_box_relay.py` 仍可供旧 collider/诊断路径使用，但 HoloLens v3 不轮询 raw tracking box route。
- debug disk ring 默认开启；只写 exact-stamp 事件/RGB-D 与 canonical 诊断，最多保留 10 分钟、18,000 条且默认不超过 10 GiB，不参与主程序判定或恢复。
- `data/shigure_recovery_debug` 独立持久保存每个 runtime/source epoch 的 bootstrap、mask reconcile、初始化报告，以及 scene/mask/depth/crop；它只用于解释等待和失败，不作为长期 identity reference 或后续业务输入。
- `migrate_shigure_v2_data.py` 是 legacy 数据进入严格 v2 的入口；`migrate_shigure_v3_data.py` 既负责带 `.pre_shigure_v3` 备份的严格 v2→v3 升级，也提供带独立 `.pre_origin_backfill_v1` 备份和幂等 metadata marker 的已-v3 origin-history one-shot 修复。正常启动只创建/验证严格 v3。
- SAM3D Body 已从当前执行路径移除；v3 HoloLens 呈现也不请求或显示人体骨骼。`code/reconstruction/sam3d-body` 只保留第三方源码/历史复现用途。
- 本仓库不修改 `code/reconstruction/shigure_core`。

## Unity scripts

```text
H2AI/Assets/Scripts/ShuJuQingQiu.cs
H2AI/Assets/Scripts/LoadModel.cs
H2AI/Assets/Scripts/RuntimeModelManager.cs
H2AI/Assets/Scripts/RuntimeSpatialBoxDisplay.cs
H2AI/Assets/Scripts/HistoryPresentationController.cs
H2AI/Assets/Scripts/ObjectEvidenceDisplay.cs
H2AI/Assets/Scripts/RuntimeSkeletonDisplay.cs
H2AI/Assets/Scripts/RuntimeModelEventIdentity.cs
H2AI/Assets/Scripts/ModelEventDisplay.cs
H2AI/Assets/Scripts/HoloLensDepthAquirer.cs
H2AI/Assets/Scripts/HoloLensPVAquirer.cs
```

- `ShuJuQingQiu.cs`：严格上传、任务轮询、live transport 握手和默认 2 秒完整 status 轮询；没有独立 raw-box poller。
- `RuntimeModelManager.cs`：以 `display_object_id` 管理模型、latest origin pose、历史 pose 与 live mask AABB；当前对象不再受 5 个可见模型限制。
- `HistoryPresentationController.cs`：用 `kind=origin` 分页，点击更早位置、到最老后回绕最新，并沿用原有飞行动画。
- `ObjectEvidenceDisplay.cs` / `RuntimeSkeletonDisplay.cs`：类仍保留，但 v3 历史放置链暂不调用照片或骨骼显示。
- `RuntimeSpatialBoxDisplay.cs`：pending preview 和每物体 primary mask-depth AABB 线框；历史 pose 不绑定历史 box。
- 采集脚本只负责 HoloLens 输入边界，不做 ArUco 转换。

## 修改规则

- Shigure live/history 或 Unity payload 改动同时更新 `shigure-v3-runtime.md`；通用模型任务 artifact 变更再同步 `server-processing-output-contract.md`。
- 跨坐标数学统一放入 `spatial_transforms.py`；stage 不复制轴翻转。
- artifact 路径统一由 `artifact_layout.py` 生成。
- 新阈值放在 `config.py` 或对应 stage `settings.py`。
- Shigure raw ID 必须受 source epoch 约束，不能作为持久 display identity。
- 新模型生成实现必须写入 canonical `ModelGeneration`/`RuntimeMesh`/`Blender`，不得增加第三套 Unity 模型协议。
