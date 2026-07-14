# 代码组织说明

更新日期：2026-07-14
状态：Shigure v2 当前协议

## 服务端核心

```text
code/
  server_api.py
  task_worker.py
  task_db.py
  task_json.py
  migrate_shigure_v2_data.py
  config.py
  path_config.py
  artifact_layout.py
  coordinate_systems.py
  spatial_transforms.py
  gpu_lease.py
  foundationpose_dispatcher.py
```

职责：

- `server_api.py`：严格 HTTP 输入、canonical model/live/history 输出和 artifact 下载。
- `task_worker.py`：主 stage 队列、GPU/socket service、Shigure recorder 与 runtime 生命周期。
- `task_db.py`：任务、ArUco reference、display identity/model revision，以及 Shigure v2 持久 session、binding、event、reference。
- `task_json.py`：只按数据库/显式路径读写 task JSON。
- `config.py`：模型后端、容量、debug 开关和判定阈值。
- `path_config.py`：当前脚本、第三方运行库和解释器路径。
- `artifact_layout.py`：唯一 artifact 目录与文件命名。
- `coordinate_systems.py` / `spatial_transforms.py`：轴定义和跨设备变换。
- `gpu_lease.py`：按预计峰值显存做 host-wide SQLite 原子 best-fit 预留；覆盖同一主机的独立 server/Gunicorn/Python 进程，并以 PID、boot ID 与进程启动时刻回收失效 lease，但不跨主机协调。
- `foundationpose_dispatcher.py`：最多 1 至 4 个弹性 backend 的共享优先队列、同物体 generation 合并与过期结果丢弃；尚无 socket 的槽位不领取任务，无 HoloLens 专用槽，运行中任务不可抢占。
- `model_generation_common.py`：InstantMesh 与 SAM3D Objects 的共享输出契约。
- `display_identity.py`：HoloLens capture 与 `display_object_id`/revision 持久化。

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
  run_shigure_history_recorder.py
```

职责边界：

- recorder 只订阅并适配不可修改的远端 ROS topics，生成 exact-stamp canonical frame，并暴露内存 socket cache。
- compatibility adapter 处理稀疏 bring-in/take-out/obj-move、tracking、bbox-local mask、RGB-D 和 Shigure 骨骼；临时 raw ID 只在 source epoch 内有效，tracking raw-ID 时间前缀变化会触发新的 incarnation/epoch。
- runtime 的 identity/模型位姿候选仍最多 5 个；另以无数量上限的 Shigure box registry 维护每 binding 的滤波与 1.5 秒 coasting。它还负责启动恢复、bring-in 绑定、take-out 原子历史、严格示例准入和 DINO 后置 FoundationPose；恢复候选缺 tracking/exact RGB-D/可信 raw ID 时保持 `PENDING`。
- `spatial_box_v2.py` 只从原始 collider 与 ArUco 校准生成严格 8 点 box，不从 mask/depth 降级。
- debug disk ring 默认关闭；开启时只写 exact-stamp 事件/RGB-D 与 canonical 诊断，最多保留 10 分钟，不参与主程序判定或恢复。
- `migrate_shigure_v2_data.py` 是 legacy 数据进入严格 v2 schema 的唯一显式一次性迁移入口；正常启动没有兼容分支，迁移会删除整个退役 `data/shigure_history_cache`。
- SAM3D Body 已从当前执行路径移除；人体历史证据只使用 Shigure joint 点线骨骼。`code/reconstruction/sam3d-body` 只保留第三方源码/历史复现用途。
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

- `ShuJuQingQiu.cs`：严格上传、任务轮询、live transport 握手/status 和模型下发。
- `RuntimeModelManager.cs`：模型/pose 与本地 cache bundle 仍遵守 5 个活动对象策略；box 独立接收无上限完整 snapshot，分离 latest live/history state，并对 ready/no_box/missing 执行更新或删除。
- `HistoryPresentationController.cs`：单击/全体 take-out 历史分页与恢复 live 呈现。
- `ObjectEvidenceDisplay.cs` / `RuntimeSkeletonDisplay.cs`：历史照片和按颜色区分的点线骨骼，随历史位置自动显示。
- `RuntimeSpatialBoxDisplay.cs`：pending preview 和 runtime 8 点线框；不填充表面。
- 采集脚本只负责 HoloLens 输入边界，不做 ArUco 转换。

## 修改规则

- API 或 Unity payload 改动同时更新 `server-processing-output-contract.md`。
- 跨坐标数学统一放入 `spatial_transforms.py`；stage 不复制轴翻转。
- artifact 路径统一由 `artifact_layout.py` 生成。
- 新阈值放在 `config.py` 或对应 stage `settings.py`。
- Shigure raw ID 必须受 source epoch 约束，不能作为持久 display identity。
- 新模型生成实现必须写入 canonical `ModelGeneration`/`RuntimeMesh`/`Blender`，不得增加第三套 Unity 模型协议。
