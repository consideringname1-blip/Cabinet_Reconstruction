# Unity 显示去重身份管理

日期：2026-06-24
状态：已接入主流程

## 作用范围

这份文档描述当前已经实现的 Unity 显示去重身份管理。它解决的是：同一个 HoloLens 物体 capture 多次建模后，服务器如何给它们分配同一个 `display_object_id`，让 Unity 默认列表里不要重复显示同一个对象。

它不证明严格真实世界同一性，只提供 Unity 显示层面的去重 id。高度相似的重复拍摄结果会尽量归到同一个显示对象；证据不足时，当前实现偏向创建新的显示对象或保持 unbound，而不是强行合并。

历史位置再现仍是独立流程，见 [历史位置再现](history-position-restoration.md)。历史位置再现负责判断物体在历史位置上的状态；本流程只负责显示去重绑定。

## 当前文件结构

当前实现分布在这些文件中：

- `code/display_identity.py`：显示去重身份绑定的核心逻辑，包括特征提取、候选搜索、距离计算、绑定决策、task JSON 写回。
- `code/stages/hololens3d_reconstruction/run_display_identity_from_json.py`：`display_identity` stage 入口。
- `code/task_db.py`：SQLite 表结构和读写函数，包括 `display_objects`、`capture_instances`、`capture_binding_logs`。
- `code/task_worker.py`：主流程 stage 顺序和 runner 注册。
- `code/server_api.py`：API 响应里追加 identity 字段，并让 model-bounds 默认按 `display_object_id` 去重。
- `code/path_config.py`：`DISPLAY_IDENTITY_STAGE_RUN` 和 `DISPLAY_IDENTITY_STAGE_PY` 路径配置。

## 主流程位置

`display_identity` 已经是正式 stage。当前 object reconstruction 的 `task_worker.STAGE_ORDER` 顺序为：

1. `hololens2depth`
2. `sam3mask`
3. `instantmesh`，这是共享模型生成槽，后端可由 InstantMesh 或 SAM3D Objects 执行
4. `depthpointcloud`
5. `modelscale`
6. `object_alignment`
7. `runtime_mesh`
8. `pose`
9. `aruco_sync`
10. `blender`
11. `model_bounds`
12. `display_identity`
13. `history_placement_restoration`
14. `taken_object_detection`
15. `sam3d_body_mesh`
16. `completed`

ArUco reference 使用独立顺序：`aruco_detect -> aruco_completed`。

也就是说，显示去重发生在模型 bounds 生成之后、历史位置再现和拿取检测之前。

服务启动时的 completed 任务补同步也会依次执行：`aruco_sync`、`model_bounds`、`display_identity`。这让旧的 completed task 在启动补同步时也能写入当前 identity 字段和数据库记录。

## 核心概念

### `display_object_id`

`display_object_id` 是 Unity 显示去重 id。Unity 默认展示列表按这个 id 去重，同一个 id 默认只显示一个代表对象。

它不是 YOLO id，也不是严格真实物体 id。YOLO id 可能因为遮挡、离开视野、重新进入或 tracker 重启而改变；当前实现不使用 YOLO id 作为候选入口，也不把 YOLO id 一致当成强绑定证据。

当前 `display_objects` 表字段：

- `display_object_id`
- `canonical_capture_instance_id`
- `capture_count`
- `notes`
- `created_at`
- `updated_at`

当前实现中，`canonical_capture_instance_id` 通常是该显示对象创建时的 capture。后续绑定会更新 `capture_count` 和 `updated_at`，但不会重新计算“最佳 canonical capture”。

### `capture_instance_id`

`capture_instance_id` 表示一次 HoloLens 拍摄或检测证据。默认值为：

`capture_{task_id}`

如果 task JSON 或已有数据库记录里已经有 `capture_instance_id`，重复执行 stage 时会复用已有值，避免同一个 task 反复创建新 capture。

当前 `capture_instances` 表字段：

- `capture_instance_id`
- `display_object_id`
- `task_id`
- `source`: 当前写入为 `hololens`
- `timestamp`
- `yolo_object_id`: 预留字段，当前 `display_identity` stage 不写入
- `binding_status`: `bound | unbound | rejected`
- `binding_reason`
- `identity_distance`
- `candidate_scores_json`
- `feature_json`
- `evidence_json`
- `created_at`
- `updated_at`

### `DisplayIdentity`

stage 会把结果写回 task JSON 的三个位置：

- 顶层 `capture_instance_id`
- 顶层 `display_object_id`
- `DisplayIdentity`

`DisplayIdentity` 当前包含：

- `capture_instance_id`
- `display_object_id`
- `is_new_display_object`
- `binding_status`
- `decision`
- `binding_reason`
- `identity_distance`
- `candidate_scores`
- `candidate_scores_close`
- `close_candidates`
- `thresholds`
- `feature_summary`
- `evidence`

## 实际绑定流程

入口是：

`code/stages/hololens3d_reconstruction/run_display_identity_from_json.py`

它通过 `stage_common.load_stage_task` 读取 task JSON，然后调用：

`display_identity.bind_capture_identity(json_path)`

### 1. 解析当前 capture

`bind_capture_identity` 先读取：

- `task_id`
- `capture_instance_id`
- 已有 `capture_instances` 记录
- 已有 `display_object_id` 绑定
- `server_received_utc` 或 `device.time` 作为 timestamp

如果当前 capture 已经在数据库里是 `bound`，且没有 `force_rebind=True`，当前实现默认复用已有绑定，决策为：

`reuse_existing_binding`

### 2. 提取 HoloLens 证据

当前实现只使用 HoloLens/SAM3 证据，不使用 Shigure，也不使用 YOLO id。

输入文件来自 task JSON 和 `artifact_layout` 推导出的 task 目录：

- mask：`data/model/<task_timestamp>/worker/02_sam3_mask.png`
- color：`data/model/<task_timestamp>/worker/02_sam3_color.png`
- depth：`data/model/<task_timestamp>/worker/02_sam3_depth.png`
- 相机内参：`PVCamera.k`，如果没有则尝试 `PVCameraFrames[0].k`
- depth 有效范围：`depth_camera_config.depth_sensor_limits_for_task(task)`

旧 task JSON 中的 `sam3Name.*` 仍可作为语义字段理解，但当前文件读取不再依赖旧全局输出或上传目录回退。

有效性要求：

- mask 像素数至少 `250`。
- masked depth 有效像素数至少 `max(50, mask_pixels * 0.03)`。
- mask、color、depth 的图像尺寸必须一致。
- depth 必须是 uint16 毫米图。

提取的特征包括：

- 颜色：HSV `H/S` 直方图、Lab `a/b` 直方图、BGR 均值、BGR 标准差。
- 形状：mask 像素数、bbox、bbox 长宽比、mask extent、Hu moments、边缘密度。
- 可见物理尺寸：使用 masked depth 和 HoloLens PV 相机内参估算可见宽、高、面积。
- depth 诊断：masked depth 中值和 IQR，只作为辅助诊断项。

可见物理尺寸当前优先使用 mask 内有效 depth pixel 的 3D 反投影点，取 5%-95% percentile extent。如果有效点不足 64 个，则使用 mask bbox 和 depth 中值估算。

### 3. 搜索候选

候选来自 SQLite 的 `capture_instances`：

- `display_object_id IS NOT NULL`
- `binding_status = 'bound'`
- 排除当前 `capture_instance_id`
- 默认最多读取 `500` 条

同一个 `display_object_id` 可能有多条历史 capture。当前实现会对每条历史 capture 计算距离，然后每个 `display_object_id` 只保留距离最小的一条作为该对象候选。

### 4. 计算距离

当前轻量距离分数为：

`identity_distance = 0.58 * masked_color_distance + 0.24 * mask_shape_distance + 0.18 * visible_physical_size_distance`

分数越小，越像同一个 Unity 显示对象。

分项含义：

- `masked_color_distance`：HSV/Lab 颜色直方图、BGR 均值、BGR 标准差。
- `mask_shape_distance`：mask 面积、bbox 长宽比、mask extent、Hu moments、边缘密度。
- `visible_physical_size_distance`：可见宽、高、面积。
- `depth_distribution_distance`：depth 中值/IQR 差异，当前会记录在候选分数里，但不参与 `identity_distance` 总分。

### 5. 做绑定决策

当前阈值：

- `strong_distance_threshold = 0.24`
- `bind_distance_threshold = 0.36`
- `gray_distance_threshold = 0.41`
- `close_candidate_margin = 0.035`
- 灰区默认：`create_new`

当前实际决策规则：

- 已有绑定且未强制重绑：复用已有 `display_object_id`，`decision=reuse_existing_binding`。
- 没有任何候选：创建新的 `display_object_id`，`decision=create_new`，`binding_reason=no_existing_candidates_create_new`。
- 最佳候选距离 `<= 0.36`：绑定到最佳候选，`decision=bind_existing`。
- 最佳候选距离在 `0.36 < distance <= 0.41`：创建新的 `display_object_id`，`binding_reason=gray_zone_create_new`。
- 最佳候选距离 `> 0.41`：创建新的 `display_object_id`，`binding_reason=distance_above_gray_create_new`。
- 如果第二候选与第一候选距离差 `<= 0.035`：记录 `candidate_scores_close=true` 和 `close_candidates`。
- 如果最佳候选已经满足 `<= 0.36`，即使存在接近候选，也会绑定到最佳候选，并把原因写为 `close_candidates_bound_to_best`。
- 如果证据不可计算，且没有可复用的旧绑定：保持 `unbound`，`decision=keep_unbound`。

当前实现没有接入 embedding 或局部特征模型，所以灰区不会自动升级为绑定。

### 6. 写入结果

绑定完成后会写入：

- `display_objects`
- `capture_instances`
- `capture_binding_logs`
- task JSON 顶层 `capture_instance_id`
- task JSON 顶层 `display_object_id`
- task JSON `DisplayIdentity`

`capture_binding_logs` 保存每次判断的候选分数、最终决策、原因和完整 detail JSON，用于后续审查。

## API 行为

### task 响应

`task_worker.get_task` 和 `task_worker.get_latest_completed_task_data` 会返回：

`display_identity: task_json.get("DisplayIdentity") or {}`

`server_api._build_completed_task_response` 会在 completed task 响应中追加：

- `display_identity`
- `display_object_id`
- `capture_instance_id`

### model-bounds 响应

`server_api._build_model_bounds_response` 和 `_build_model_instance` 会把 identity 字段追加到返回对象：

- `display_identity`
- `display_object_id`
- `capture_instance_id`

### 默认去重

以下接口默认按 `display_object_id` 去重：

- `/model-bounds/latest?limit=5`
- `/model-bounds/range?start=<uploaded_at>&end=<uploaded_at>`

响应中会包含：

- `count`：去重后的数量。
- `raw_count`：实际取出的原始 model-bounds 数量。
- `deduped_by_display_object`：默认为 `true`。

去重规则：

- 有 `display_object_id` 且 `binding_status != 'unbound'` 的对象，用 `display_object_id` 去重。
- 没有 identity 或为 `unbound` 的对象，用 task 自身作为唯一 key，不会被错误合并。

需要返回重复 capture 时，可以加：

- `include_duplicate_captures=1`
- 或 `include_duplicate_display_captures=1`

当前 `/model-bounds/latest` 在默认去重模式下会多取一部分原始记录再去重：`min(max(limit * 4, limit), 50)`。如果极端情况下最近 50 条都属于少数几个 `display_object_id`，返回数量可能少于请求的 `limit`。

## 离线阈值来源

根据 `注意事项.txt` 中 2026-06-24 标注的 21 组、119 条 `data_backup/upload` 历史 capture 做过离线试跑。试跑只纳入同时具备 meta、SAM3 mask、SAM3 masked color、SAM3 masked depth 的样本；缺 `sam3Name` 或核心数据不可用的旧数据跳过。

本轮可用样本：

- 总条目：119。
- 可用条目：114。
- 跳过条目：5，原因均为旧 meta 缺少 `sam3Name`。
- 大屏幕组按弱参考处理，因为表面纹理预期不稳定。

使用当前轻量分数和可见物理尺寸后：

- `identity_distance <= 0.36` 附近，同组召回约 `88.9%`，匹配精度约 `97.2%`，异组误合并率约 `0.44%`。
- `identity_distance <= 0.41` 附近，同组召回约 `94.5%`，匹配精度约 `93.5%`，异组误合并率约 `1.13%`。
- 按时间顺序模拟“已有对象中找候选，否则新建”时，`0.36` 阈值下 114 条里 6 条错误：2 条同物被新建，4 条异物被误合并。

因此当前线上规则使用 `0.36` 作为自动绑定阈值，`0.36~0.41` 作为灰区但默认创建新对象。

## 当前限制

当前实现已经可用，但边界很清楚：

- 还没有接入 DINOv2、CLIP/SigLIP、ALIKED、SuperPoint、LightGlue、LoFTR 等 embedding 或局部特征模型。
- 还没有生成新旧 masked crop 的对比拼图；当前保存的是 task worker/result 路径、特征摘要、候选评分和日志 JSON。
- `yolo_object_id` 是数据库预留字段，当前 `display_identity` stage 不使用也不写入。
- `display_object_id` 只用于 Unity 默认显示去重，不维护 `MOVED`、`MISSING`、`OCCLUDED`、`last_seen_at` 等追踪状态。
- 当前不处理模型版本选择，不维护 active model，不决定新模型是否替换旧模型。
- 当前没有自动合并、拆分、人工修正 `display_object_id` 的管理接口。
- `canonical_capture_instance_id` 当前不会根据后续 capture 自动重选最佳参考。
- 当前相似度基于单次视角下的 mask、颜色、形状和可见物理尺寸。视角变化、坏 mask、同类相似物体仍可能造成误合并或重复新建。

## 验收状态

当前已满足：

- 每次 HoloLens capture 能生成或复用独立 `capture_instance_id`。
- 新 capture 会根据现有 bound capture 搜索候选 `display_object_id`。
- 同一显示对象的相似 capture 可以绑定到同一个 `display_object_id`。
- 差别大或灰区默认创建新的 `display_object_id`。
- 多候选接近时会记录 `candidate_scores_close=true` 和 `close_candidates`。
- 证据不可计算且无旧绑定时不会强行绑定，会保持 `unbound`。
- 每次正常判断会写入候选评分和绑定日志。
- task JSON 会写入 `DisplayIdentity`、`display_object_id`、`capture_instance_id`。
- Unity 相关 model-bounds 默认响应会按 `display_object_id` 去重。

当前未满足：

- 没有额外视觉模型作为灰区兜底。
- 没有自动生成对比可视化图片。
- 没有 display identity 的人工合并、拆分、撤销接口。
- 没有完整的按 `display_object_id` 查询全部 capture 历史的 API。
