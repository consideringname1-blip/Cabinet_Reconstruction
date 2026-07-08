# 历史再现审查结论

更新日期：2026-07-08

## 当前结论

历史再现当前已符合 fixed Shigure 视角设计：

- baseline 来自 `TakenObjectDetection.history_baseline`。
- 保存并复用 `old_rgb + old_depth + old_mask + camera_info`。
- 后续只在 old_mask 内比较 RGB/depth。
- 不再使用旧 YOLO baseline fallback。
- API 公开坐标转换到当前 HoloLens 本地坐标。
- Unity 证据窗口优先显示 `subject_crop_url`。

## 仍需注意

- `history_placement_restoration` 文件中仍保留部分旧 helper，用于 debug 或历史兼容字段解析；新增逻辑不要继续扩展旧 YOLO baseline。
- 历史 item 只应选择已 `aruco_coordinate_synced` 的 completed model。
- 当前状态显示不叠加模型，只显示 polyhedron 和证据图。

## 核心风险点

如果某个历史任务没有 taken baseline，则不应尝试通过当前 Shigure object_detection 补救。这种情况应返回 unknown/failed，并在 debug 中说明 `taken_detection_history_baseline_missing`。
