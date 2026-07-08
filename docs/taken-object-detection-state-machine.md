# 拿取判断状态机

更新日期：2026-07-08
状态：当前实现说明

实现文件：`code/stages/taken_object_detection/run_taken_object_detection_from_json.py`。该 stage 建立历史再现所需的 fixed Shigure baseline，并在后续 Shigure 帧中判断物体是否被拿走。

## 输入

- 当前 model task JSON。
- `object_aruco` / `ModelBounds`。
- Shigure history cache 中的 RGB-D、camera_info、object_detection。
- Shigure marker pose。

## 初始化目标

初始化得到：

```text
old_rgb
old_depth
old_mask
camera_info
old_reference_depth_m
```

这些 artifact 写入 taken init backup 目录，并在 `TakenObjectDetection.history_baseline` 中记录路径。历史再现 stage 后续只依赖这份 baseline。

## 初始化 mask 选择逻辑

当前使用模型中心对角圆约束：

1. 读取 `ModelBounds` 的模型中心和 AABB 对角线长度。
2. 将模型中心和 bounds corners 通过 Shigure marker pose 投影到 Shigure 图像，生成以模型中心为圆心、模型对角线为直径的投影圆。
3. 对每个 Shigure object mask 计算 `mask_inside_diag_circle_ratio = area(mask ∩ diag_circle) / area(mask)`。
4. 计算该 mask 的有效 depth 中位数，与模型中心在 Shigure 视角下的 depth 做绝对差。
5. 候选必须同时满足对角圆内占比和 depth 偏差阈值。
6. 在 accepted 候选中选择 `mask_pixels` 最大的 mask 作为 `old_mask`。

关键阈值：

```text
TAKEN_OBJECT_TRACKING_MODE=model_diag_circle
TAKEN_OBJECT_MODEL_DIAG_CIRCLE_MIN_MASK_INSIDE_RATIO=0.80
TAKEN_OBJECT_MODEL_DIAG_CIRCLE_MAX_DEPTH_DIFF_M=0.18
```

`MODEL_DIAG_CIRCLE_MAX_DEPTH_DIFF_M` 当前默认 0.18m，对应之前讨论的 0.15m ~ 0.20m 范围。

## 初始化失败条件

常见失败：

- capture time 缺失。
- Shigure cache 时间窗口内没有 frame。
- first frame 太晚。
- marker pose 缺失。
- camera_info 无法解析 K。
- 模型中心或对角圆无法投影到 Shigure 图像。
- object_detection 没有同时满足对角圆内占比和 depth 偏差阈值的 object mask。
- old_mask 内有效 depth 太少。

失败会写入 `TakenObjectDetection.status = INIT_FAILED` 和 reason。

## 跟踪阶段

初始化成功后，跟踪只在 trusted old_mask 内进行，不扫全图。

每帧计算：

```text
valid = old_mask & current_depth_valid & reference_depth_valid
delta = current_depth - reference_depth
occluded = delta < -threshold
taken = delta > threshold 或有效区域明显减少
```

状态机维护 candidate/confirmed：

- 连续满足拿走条件达到阈值，输出 `TAKEN`。
- 未达到阈值或窗口结束，输出 `NOT_TAKEN`。
- 全遮挡帧单独记录，但当前输出图片使用 depth taken frame，不再回退到“安静前帧”。

## 输出

`TAKEN` 输出：

```json
{
  "status": "TAKEN",
  "result_timestamp": {},
  "backup_shigurei_dir": "...",
  "projection": {},
  "init": {},
  "history_baseline": {},
  "depth_taken_timestamp": {},
  "depth_confirm_timestamp": {},
  "result_rgb": "07_taken_detection_result_rgb.png",
  "result_depth": "07_taken_detection_result_depth.png",
  "camera_info": "07_taken_detection_camera_info.json",
  "active_objects": "07_taken_detection_active_objects.json",
  "marker_pose": "07_taken_detection_marker_6d_pose.json"
}
```

`history_baseline` 是给历史再现使用的稳定契约。

## 与历史再现的关系

- taken stage 负责确定初始化 old_mask 和保存 old_rgb/depth。
- history stage 只读取 `history_baseline`。
- history stage 不再从 Shigure object_detection 重新找 mask。
- 如果 taken stage 没有 baseline，history stage 返回 unknown/failed，而不是使用旧 fallback。

## 不再使用的逻辑

- 初始化必须由模型对角圆规则得到 `old_mask`。
- 不再从 SAM3 mask 或 selection box 直接生成 taken mask。
- 初始化只按模型对角圆和 depth 偏差选择 `old_mask`；后续判断只在 `old_mask` 内。
