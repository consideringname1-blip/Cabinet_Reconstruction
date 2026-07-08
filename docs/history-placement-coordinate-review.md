# 历史再现坐标审查

更新日期：2026-07-08

## 审查目标

确认历史再现和模型下载是否遵守：

```text
HoloLens 上传当前本地坐标
-> 服务器转 ArUco/世界存储
-> 服务器按当前 startup 转回 HoloLens 当前本地坐标
-> Unity 只消费 HoloLens 当前本地坐标
```

## 当前符合项

- completed/history response 使用 `object_aruco` + latest startup ArUco reference 转 `object_hololens_current`。
- `object_hololens_original` 只在同次启动下发。
- 非同次启动历史模型要求 `aruco_coordinate_synced`。
- `sam3_spatial_box` 只保留 pending preview。
- Unity 正式模型缺 `object_hololens_current` 会失败，不再 fallback 到相机前方。
- Unity 不读取 ArUco pose 放置模型。

## ArUco 更新

ArUco reference 完成后：

- 按 `startup_session_id` 查询同次启动任务。
- 对达到 `aruco_sync` 之后的任务重新计算 `object_aruco` 和 `object_hololens_current`。
- 对 model_bounds/display_identity 等后续结果刷新。
- 不修改其他 startup session 的历史任务。

## Pending Preview 例外

pending preview 可以使用 `Sam3SpatialBox`：

- 这是生成中的临时视觉提示。
- 它不需要跨启动稳定。
- 它不参与历史模型放置。

## 风险检查清单

新增接口或 Unity 逻辑时检查：

- 是否下发了 `object_hololens_current`。
- 是否意外下发/使用 `object_aruco`。
- 是否把 `sam3_spatial_box` 用成正式 pose fallback。
- 是否在缺 pose 时本地猜测位置。
- 是否绕过 `_public_spatial_payload` 直接返回 ArUco 字段。

## 结论

当前主链路满足设计。后续重点是保持 API response 的坐标公开层统一，不要让业务 stage 的内部 ArUco 字段泄漏给 Unity。
