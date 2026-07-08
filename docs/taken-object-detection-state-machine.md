# 拿取判断状态机

更新日期：2026-07-08
状态：当前实现说明

实现文件：`code/stages/taken_object_detection/run_taken_object_detection_from_json.py`。该 stage 建立历史再现所需的 fixed Shigure baseline，并在后续 Shigure 帧中判断物体是否被拿走。

## 输入

- 当前 model task JSON。
- `object_aruco` / `ModelBounds`。
- Shigure history cache 中的 RGB-D、camera_info、object_detection。
- Shigure marker pose。
- HoloLens PV camera 位置，用于第 6 条 2D ray 选择。

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

## 第 6 条 mask 选择逻辑

当前仍保留第 6 条：

1. 将 HoloLens/ArUco 下的模型 box 投影到 Shigure 图像。
2. 用 projected box 生成 2D mask。
3. 从 HoloLens 相机位置指向 box center，再沿该方向向 box 后方延伸。
4. 将这条方向投到 Shigure 图像，生成 2D ray mask。
5. Shigure object mask 候选必须覆盖 projected box 足够面积，并命中 ray mask。
6. 在 accepted 候选中选择面积最小的 mask 作为 `old_mask`。

关键阈值：

```text
TAKEN_OBJECT_MODEL_BOX_OBJECTMASK_MIN_BOX_COVERAGE=0.40
```

ray 是图像上的 2D 线状 mask，不是 3D 体积选择。

## 初始化失败条件

常见失败：

- capture time 缺失。
- Shigure cache 时间窗口内没有 frame。
- first frame 太晚。
- marker pose 缺失。
- camera_info 无法解析 K。
- model box 无法投影到 Shigure 图像。
- object_detection 没有符合 projected box + ray 的 object mask。
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

- 不再支持 legacy projected mask fallback。
- 不再从 SAM3 mask 或 selection box 直接生成 taken mask。
- 不再先限制识别范围后扫全图；初始化按第 6 条选 mask，后续判断只在 old_mask 内。
