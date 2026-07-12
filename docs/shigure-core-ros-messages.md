# 远端 Shigure 数据入口与内存缓存

更新日期：2026-07-12
状态：当前协议

Shigure 在独立服务器运行。本仓库不修改 `code/reconstruction/shigure_core`；`code/stages/shigure_history` 只订阅已配置的远端 ROS 数据流、规范化消息并向本服务器业务线程提供内存缓存。

## 输入 topics

默认配置位于 `code/stages/shigure_history/settings.py`：

```text
RGB                 /rs/color/compressed
Depth               /rs/aligned_depth_to_color/compressedDepth
CameraInfo          /rs/aligned_depth_to_color/cameraInfo
object_detection    /shigure/object_detection
contacted           /shigure/contacted
```

topic 名和消息类型可通过对应 `SHIGURE_HISTORY_*` 环境变量配置。RGB、depth、CameraInfo 是 RGB-D sample 的必要输入；两个 Shigure event topic 独立保留缺失/空列表状态。

## Recorder 生命周期

`task_worker.start_worker()` 启动：

```text
code/stages/shigure_history/run_shigure_history_recorder.py
  --socket-server data/worker_sockets/shigure_history.sock
```

`SHIGURE_HISTORY_RECORDING_ENABLE=1` 时 worker 监控并重启 recorder。缓存属于 recorder 进程内存；服务器/recorder 重启后缓存和 event sequence 重新开始。

## RGB-D sample

RGB 时间戳是 sample 主时间戳。recorder 在 `SHIGURE_HISTORY_RGB_DEPTH_MAX_DELTA_SECONDS` 内选择最近 depth，并读取 CameraInfo，构造：

```text
CachedRgbdSample
  stamp.sec
  stamp.nanosec
  rgb_bgr        uint8 HxWx3
  depth          uint16 HxW, millimetres
  camera_info.k  3x3
  camera_info.width
  camera_info.height
```

RGB、depth 和 CameraInfo 尺寸必须一致，焦距必须为正。没有 ROS source timestamp 的 RGB 不进入缓存。

## Correlated Shigure event

`/shigure/object_detection` 与 `/shigure/contacted` 只按完全相同的 ROS source stamp join：

```text
CachedShigureEvent
  source_stamp
  sequence
  contacted_state
  object_detection_state
  contacted
  object_detection
  contact_object_matches
```

topic state 只有：

```text
missing
explicit_empty
present
```

因此“该 stamp 确认没有 contact”和“该 stamp 没收到 contact topic”不会混为一谈。

规范化 detection object：

```json
{
  "object_id": "obj_move:0",
  "action": "obj_move",
  "bbox_xyxy": [10, 20, 100, 160],
  "mask_b64": "...",
  "mask_format": "png",
  "mask_bytes": 1234
}
```

规范化 contact：

```json
{
  "event_id": "...",
  "people_id": "...",
  "object_id": "...",
  "action": "take_out",
  "people_bounding_box": {"xyxy": [0, 0, 10, 10]},
  "object_bounding_box": {"xyxy": [0, 0, 10, 10]},
  "object_cube": {"x": 0, "y": 0, "z": 0, "width": 1, "height": 1, "depth": 1}
}
```

contact 与 detection 的关联先要求 action 相同，再用 object bbox IoU 消除同 action 多候选。拿取证据分支只接受 `matched_action_iou`。

## Socket API

业务代码使用 `ShigureRgbdCache`，可调用：

```text
status()
iter_samples(start, end)
newest_sample()
get_sample(stamp)                  # 返回缓存中时间最近的 RGB-D
iter_event_updates_after(sequence)
latest_event()
```

event mask 默认不通过 socket 返回；需要身份匹配或 FoundationPose 时显式设置 `include_masks=True`。

## 身份边界

Shigure `object_id` 只用于当前 `startup_session_id + ingress_session_id` 的临时绑定：

1. 将每个活动 `display_object_id` 最新 HoloLens `Sam3SpatialBox` 投影为 Shigure 图像中的对角圆。
2. 过滤 mask-inside ratio 与中心 depth 差。
3. 几何候选唯一时绑定；多个候选用 DINOv2。
4. 同一 startup 中一个临时 Shigure ID 只绑定一个 display object；HoloLens 重新拍摄该对象时释放并重新匹配。

服务器重启、startup 改变或对象重新锚定后，临时绑定失效。持久记录只使用 `display_object_id`、模型 revision 和 ArUco/HoloLens pose。

## Marker pose

Shigure camera 到 ArUco 的 marker pose 存放在：

```text
data/aruco/shigure_marker_history/latest_marker_6d_pose.json
```

当前结构只读取：

```text
opencv_camera_pose.rotation_matrix
opencv_camera_pose.tvec_m
```

它用于 HoloLens box 投影、FoundationPose 结果转换和人体 mesh 从 Shigure camera 转到 ArUco。Unity 不接收该文件或 ArUco pose。
