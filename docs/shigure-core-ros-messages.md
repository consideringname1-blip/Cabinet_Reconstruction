# Shigure ROS 数据与内存缓存

更新日期：2026-07-08
状态：当前实现说明

Shigure 侧数据由 `code/stages/shigure_history` 负责接收、对齐和缓存。缓存常驻在 recorder 子进程内，业务 stage 通过本地 Unix socket 读取，不能依赖 Python 全局内存。

## 目标

固定 Shigure 视角提供：

- RGB 图像。
- Depth 图像。
- CameraInfo。
- `/shigure/object_detection` 输出。
- ArUco marker pose 历史。

这些数据服务于：

- taken object 初始化 old_mask。
- fixed Shigure history restoration。
- SAM3D Body 输入帧和裁切。

## 启动方式

`task_worker.start_worker()` 会在服务启动时启动 Shigure history recorder：

```text
code/stages/shigure_history/run_shigure_history_recorder.py --socket-server data/worker_sockets/shigure_history.sock
```

开关：

```text
SHIGURE_HISTORY_RECORDING_ENABLE=1
```

如果 recorder 退出，worker monitor 会尝试重启。

## Topic 配置

默认配置位于 `code/stages/shigure_history/settings.py`：

```text
SHIGURE_HISTORY_SECONDS=60
SHIGURE_HISTORY_HZ=5
SHIGURE_HISTORY_OBJECT_DETECTION_TOPIC=/shigure/object_detection
```

缓存目标是约 1 分钟最近数据。object_detection 是可选 topic，但拿取初始化依赖它，如果没有有效 object mask 会初始化失败。

## 对齐策略

recorder 以 RGB 时间戳为主样本：

1. 取最近 RGB。
2. 在允许 delta 内取最近 depth。
3. 取最近 camera_info。
4. 取最近 object_detection。
5. 打包成 `CachedRgbdSample` 放入 `ShigureMemoryStore`。

样本 metadata 包含 stamp、路径、object_detection hash、chunk/frame index 等信息。业务 stage 可以按 stamp 取 nearest sample 或遍历时间窗口。

## Unix Socket API

业务端使用：

```python
cache = ShigureRgbdCache(SHIGURE_HISTORY_CACHE_ROOT)
sample = cache.get_sample(stamp, mode="nearest")
metadata = cache.iter_sample_metadata(start=start, end=end)
```

底层通过 Unix socket 请求 recorder 进程。这样 stage 即使作为子进程运行，也能读取同一份内存缓存。

## CachedRgbdSample 字段

典型字段：

```text
stamp
rgb_bgr
depth
camera_info
yolo / object_detection
camera_info_path
yolo_path
yolo_hash
chunk_id
frame_index
```

`yolo` 字段当前也代表 Shigure object_detection payload。历史命名兼容保留，但新文档和新逻辑应使用 object_detection 语义。

## object_detection 语义

拿取初始化会读取 object_detection 中的候选 object mask：

- mask 必须能转换为与 depth 同尺寸的 bool mask。
- 候选 mask 会与模型中心对角圆投影比较，并校验 mask 有效 depth 中位数和模型中心 depth 的偏差。
- 选择规则见 `docs/taken-object-detection-state-machine.md`。

## Marker Pose

Shigure ArUco marker pose 存储在：

```text
data/aruco/shigure_marker_history/
```

最新 marker pose 用于：

- HoloLens/ArUco 模型中心和 bounds corners 投影到 Shigure 图像，生成模型对角圆。
- Shigure camera point 反投影到 ArUco。
- SAM3D Body mesh 从 Shigure camera 坐标转 ArUco，再由服务器转换到 HoloLens current。

## 使用边界

- Shigure fixed view 的历史再现不做 3D 投影扫描。
- 初始化 old_mask 使用模型对角圆内占比、depth 偏差阈值，并在 accepted 候选中选择最大 object mask。
- 一旦 old_mask 保存，后续 still/missing/occluded 判断只在 old_mask 内比较 RGB/depth。
- 不要让 stage 直接读取 recorder 内部全局变量；统一走 `ShigureRgbdCache`。
