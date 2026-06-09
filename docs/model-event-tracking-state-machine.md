# Model Event Tracking State Machine

这份文档说明“物体被拿走”事件的整体状态机。目标是只判断一个状态：可信模型已经存在后，物体是否从 Shigurei 视角下的原始位置被拿走，并为 HoloLens 点击模型时准备事件图片、人体信息和可选人体 mesh。

## 输入和边界

拿取追踪不依赖 SAM3 video。当前正式路径使用已完成的 3D 模型和 Shigurei RGB-D 历史缓存：

1. HoloLens 完成拍摄并生成可信模型、FBX、模型 bounds。
2. Shigurei 录制器在服务启动后持续缓存必要数据，默认保留 10 分钟、约 5 Hz。
3. 追踪 stage 从拍摄后一小段稳定时间开始回放历史缓存，然后继续跟随新缓存帧。
4. 坐标链路使用文档化的 ArUco/OpenCV/Blender 投影规则，把模型投影到 Shigurei 相机。

主要代码入口：

- `code/stages/model_event_tracking/run_model_event_tracking_from_json.py`
- `code/stages/model_event_tracking/model_depth.py`
- `code/stages/model_event_tracking/event_store.py`
- `code/stages/model_event_tracking/settings.py`

## 初始化阶段

初始化的职责是建立“哪些像素可以参与拿走判定”。

1. 选择 Shigurei 历史帧窗口。
   - 优先从拍摄时间加 `MODEL_EVENT_CAPTURE_SETTLE_SECONDS` 后开始。
   - 如果设置了 debug 起止帧，则使用指定窗口。
   - 如果历史不够，会继续等待新缓存直到超时。

2. 选择 marker pose 和相机内参。
   - 当前帧缓存里有 marker pose 时直接用。
   - 没有时使用历史 marker pose，因为 Shigurei 相机相对 marker 基本固定。
   - 相机内参使用 Shigurei 缓存中的实际 camera info。

3. 渲染模型深度模板。
   - 将完成的 FBX 投影到 Shigurei RGB-D 视角。
   - 对模型投影区域内每条相机射线求 mesh 交点。
   - 每个像素得到 `model_front_depth` 和 `model_back_depth`，当前正式拿走判定使用前表面加 margin。

4. 校准固定深度偏差。
   - 使用起始若干帧估计 Shigurei 深度和模型渲染深度之间的固定 bias。
   - 得到初始 `support_mask`，只让曾经可信观测到模型表面的像素参与后续判定。
   - 模型孔洞、错误投影到背景的区域不会直接投票为“拿走”。

## 单帧深度判定

每帧进入 `DynamicDepthMaskTracker.update()` 后，会产生一个 `DepthFrameDecision`。可以把每个像素分为三类：

```text
front_match = abs(observed_depth - model_front_depth - depth_bias) <= present_tolerance
support_mask = support_mask OR front_match
occluded = observed_depth < model_front_depth + depth_bias - occlusion_margin
current_unoccluded_mask = support_mask AND valid_depth AND NOT occluded
removed = current_unoccluded_mask AND observed_depth > model_front_depth + depth_bias + removal_margin
```

调试图颜色约定：

- 绿色：完整模型投影 mask。
- 黄色：当前可信且未遮挡的 `current_unoccluded_mask`。
- 红色：在可信未遮挡区域内，被认为已经向后变深的 `removed_mask`。

遮挡不会永久删除可信区域。手、身体或其他前景挡住物体时，这些像素只是暂时从 `current_unoccluded_mask` 排除；当前景离开，如果深度重新接近模型前表面，它们会再次成为可信区域。

## 状态机

状态机以每个模型独立运行。核心状态如下：

```text
WAITING_INPUT
  -> INITIALIZING_TEMPLATE
  -> TRACKING_PRESENT
  -> FULL_OCCLUSION_CANDIDATE
  -> FULL_OCCLUDED
  -> REMOVAL_CANDIDATE
  -> TAKEN_AWAY
  -> TIMEOUT / OCCLUDED_UNRESOLVED / NO_EVENT
```

### `WAITING_INPUT`

等待模型 bounds、FBX、Shigurei RGB-D 帧、camera info、marker pose 可用。

失败或跳过条件：

- 模型 bounds 不 ready。
- 缓存帧不足且等待超时。
- camera info 或 marker pose 不可用。
- 模型投影区域太小或无法渲染有效深度模板。

### `INITIALIZING_TEMPLATE`

渲染模型深度模板，校准 depth bias，建立初始 support mask。

输出：

- `projection_and_model_mask.jpg`
- `reference_observed_mask.png`
- `diagnostics.json`
- `current_support_mask.png`
- `current_unoccluded_mask.png`

### `TRACKING_PRESENT`

默认稳定追踪状态。每帧都会更新 support mask，并判断是否进入遮挡或移除候选。

保持条件：

- 可信区域仍有足够像素。
- removed ratio 未连续超过阈值。
- 没有连续完整前景遮挡。

### `FULL_OCCLUSION_CANDIDATE`

当投影模型区域大部分被更近的前景覆盖，并且可评估区域很小，会进入完整遮挡候选。

典型场景：人背身或身体完全挡住物体。

进入条件由这些阈值控制：

- `MODEL_EVENT_DEPTH_FULL_OCCLUSION_RATIO`
- `MODEL_EVENT_DEPTH_FULL_OCCLUSION_MAX_EVALUABLE_RATIO`
- `MODEL_EVENT_DEPTH_FULL_OCCLUSION_MIN_PIXELS`

如果连续满足 `MODEL_EVENT_DEPTH_FULL_OCCLUSION_STABLE_FRAMES` 帧，则升级为 `FULL_OCCLUDED`。

### `FULL_OCCLUDED`

完整遮挡本身不等于拿走。系统只记录遮挡起点，作为可能的交互开始时间。

处理规则：

- 暂停普通 removal candidate 计数。
- 记录 `active_full_occlusion_start`。
- 如果遮挡后物体重新可见，则清除 active full occlusion，回到 `TRACKING_PRESENT`。
- 如果遮挡结束后立刻满足拿走深度判定，则事件的 depth start 回退到完整遮挡起点。

清除条件由这些阈值控制：

- `MODEL_EVENT_DEPTH_OCCLUSION_CLEAR_EVALUABLE_RATIO`
- `MODEL_EVENT_DEPTH_OCCLUSION_CLEAR_MAX_OCCLUDED_RATIO`

如果追踪超时时仍处于完整遮挡，状态为 `occluded_unresolved`，表示不能确认物体还在或已被拿走。

### `REMOVAL_CANDIDATE`

当当前可信未遮挡区域内，有足够比例像素比模型前表面加 margin 更深时，进入拿走候选。

进入条件主要是：

- `removed_ratio >= MODEL_EVENT_DEPTH_REMOVED_RATIO`
- `present_ratio <= MODEL_EVENT_DEPTH_MAX_PRESENT_RATIO`
- `evaluable_ratio >= MODEL_EVENT_DEPTH_MIN_EVALUABLE_RATIO`
- `evaluable_pixels >= MODEL_EVENT_DEPTH_MIN_EVALUABLE_PIXELS`

如果连续满足 `MODEL_EVENT_DEPTH_TAKEN_AWAY_STABLE_FRAMES` 帧，则确认 `TAKEN_AWAY`。

如果中间任一帧不满足候选条件，则候选计数清零，回到 `TRACKING_PRESENT`。

### `TAKEN_AWAY`

拿走确认后立即停止该模型对应的追踪。确认帧只负责证明“已经不在原位”，展示帧会额外通过 RGB 回溯选择。

事件时间选择顺序：

1. 深度确认得到 `depth_confirm_timestamp`。
2. 深度候选第一帧是 `removed_depth_start_timestamp`。
3. 如果前面存在有效完整遮挡且物体在遮挡后消失，则 `depth_decision_timestamp` 使用 full occlusion 起点。
4. 在深度起点前的 RGB 窗口内寻找视觉变化起点，得到 `rgb_motion_start_timestamp`。
5. HoloLens 显示图片使用 `display_timestamp`，通常就是 RGB 变化起点，也可以用 `MODEL_EVENT_RGB_MOTION_DISPLAY_OFFSET_FRAMES` 前后偏移。

## RGB 回溯

深度判定适合确认“已经被拿走”，但不一定精确到手刚开始动作的那一帧。因此拿走确认后，会向前看一段 RGB 窗口，只在模型 support mask 内计算变化。

流程：

1. 取 `MODEL_EVENT_RGB_MOTION_LOOKBACK_SECONDS` 秒窗口。
2. 用最前面的 `MODEL_EVENT_RGB_MOTION_BASELINE_FRAMES` 帧建立 RGB baseline。
3. 只在 `support_mask` 内比较 RGB 差异；如果 support 太小则退回完整模型 mask。
4. 单像素 RGB 平均差异超过 `MODEL_EVENT_RGB_MOTION_DIFF_THRESHOLD` 记为变化。
5. 连续 `MODEL_EVENT_RGB_MOTION_CONFIRM_FRAMES` 帧变化比例超过 `MODEL_EVENT_RGB_MOTION_START_RATIO`，认为找到动作开始。
6. 找不到时，事件仍成立，但展示帧回退到深度起点附近。

RGB 不决定物体是否拿走，只用于选择更适合给 HoloLens 展示的事件起点图片。

## 手腕和人体信息

拿走确认后，系统在事件时间前 `MODEL_EVENT_HAND_LOOKBACK_SECONDS` 秒内查 Shigurei people detection。

规则：

1. 只关注左右手腕。
2. 优先选择第一帧进入投影 box 的手腕。
3. 如果没有进入 box，则选择距离 box 最近的手腕。
4. 距离超过 `MODEL_EVENT_HAND_NEAREST_MAX_DISTANCE_M`，认为没有可信骨骼。
5. 为避免事件帧骨骼尚未生成，允许在显示帧前后 `MODEL_EVENT_SKELETON_SEARCH_RADIUS_FRAMES` 帧内寻找同一个 people id。

事件输出保留：

- 原始 RGB 图。
- `skeletons.json` 和调试用骨骼覆盖图。
- 可选 `body_mesh.json`，由 SAM3D Body 生成并降面后给 Unity 显示。

Unity/HoloLens 端优先显示 `body_mesh.json` 的灰色半透明人体 mesh；旧事件或生成失败时回退到骨骼点线。

## 事件落盘

确认 `TAKEN_AWAY` 后，`event_store.persist_taken_away_event()` 会写入：

```text
data/output/model_events/<task_output_name>/taken_away/
  event.json
  rgb.png
  skeletons.json
  skeleton_overlay.png
  body_mesh.json                 # 可选，生成成功时存在
  debug/
    camera_info.json
    depth.png
    marker_pose.json
    projected_box.json
    moved_mask.png
    decision_overlay.png
    body_mesh_error.json          # 可选，人体 mesh 生成失败时存在
```

`event.json` 中关键字段：

- `decision.status`: `taken_away`
- `decision.depth_decision_timestamp`: 深度事件起点，可能是完整遮挡起点。
- `decision.depth_confirm_timestamp`: 深度连续确认完成帧。
- `decision.rgb_motion_start_timestamp`: RGB 回溯找到的视觉变化起点。
- `decision.display_timestamp`: HoloLens 默认展示帧。
- `decision.rgb_motion_metadata`: RGB 回溯、完整遮挡段、是否使用遮挡起点等诊断信息。
- `hand_contact`: 最可信的手腕接触信息；没有可信手腕时为 null。
- `files`: API 会把这些文件映射为 `download_urls`。

## 超时和终止

终止条件：

- `TAKEN_AWAY`: 事件确认，停止对应模型追踪。
- `TIMEOUT`: 到达 `MODEL_EVENT_TRACKING_TIMEOUT_SEC` 且没有事件。
- `OCCLUDED_UNRESOLVED`: 超时时仍处于完整遮挡，无法确认结果。
- `NO_EVENT`: debug/replay 窗口结束但没有事件。

`MODEL_EVENT_TRACKING_TIMEOUT_SEC=0` 表示不主动超时，持续跟随 Shigurei 新帧直到确认事件或外部停止。

## 关键调参入口

深度模板和 mask：

- `MODEL_EVENT_DEPTH_RENDER_PADDING_PX`
- `MODEL_EVENT_DEPTH_MASK_ERODE_PX`
- `MODEL_EVENT_DEPTH_MIN_MODEL_PIXELS`

普通遮挡和拿走：

- `MODEL_EVENT_DEPTH_OCCLUSION_MARGIN_M`
- `MODEL_EVENT_DEPTH_REMOVAL_MARGIN_M`
- `MODEL_EVENT_DEPTH_REMOVED_RATIO`
- `MODEL_EVENT_DEPTH_MAX_PRESENT_RATIO`
- `MODEL_EVENT_DEPTH_MIN_EVALUABLE_RATIO`
- `MODEL_EVENT_DEPTH_TAKEN_AWAY_STABLE_FRAMES`

完整遮挡：

- `MODEL_EVENT_DEPTH_FULL_OCCLUSION_RATIO`
- `MODEL_EVENT_DEPTH_FULL_OCCLUSION_MAX_EVALUABLE_RATIO`
- `MODEL_EVENT_DEPTH_FULL_OCCLUSION_STABLE_FRAMES`
- `MODEL_EVENT_DEPTH_OCCLUSION_CLEAR_EVALUABLE_RATIO`
- `MODEL_EVENT_DEPTH_OCCLUSION_CLEAR_MAX_OCCLUDED_RATIO`

RGB 回溯：

- `MODEL_EVENT_RGB_MOTION_LOOKBACK_SECONDS`
- `MODEL_EVENT_RGB_MOTION_DIFF_THRESHOLD`
- `MODEL_EVENT_RGB_MOTION_START_RATIO`
- `MODEL_EVENT_RGB_MOTION_CONFIRM_FRAMES`
- `MODEL_EVENT_RGB_MOTION_DISPLAY_OFFSET_FRAMES`

手腕和人体 mesh：

- `MODEL_EVENT_HAND_NEAREST_MAX_DISTANCE_M`
- `MODEL_EVENT_SKELETON_SEARCH_RADIUS_FRAMES`
- `MODEL_EVENT_SAM3D_BODY_MESH_ENABLED`
- `MODEL_EVENT_SAM3D_BODY_MESH_DECIMATE_RATIO`
- `MODEL_EVENT_SAM3D_BODY_MESH_MAX_DISTANCE_M`

## 设计原则

1. 深度确认事件，RGB 只负责找更合适的展示起点。
2. 前景遮挡不投票为拿走，只从可评估区域里暂时排除。
3. 完整遮挡是潜在交互起点；只有遮挡后物体消失才回退到这个起点。
4. support mask 只扩大不随意缩小，避免模型孔洞和背景误投影直接造成误判。
5. 每个模型独立追踪；事件确认后立即停止对应模型的追踪任务。
