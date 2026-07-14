# 远端 Shigure 数据入口与内存缓存

更新日期：2026-07-14
状态：Shigure v2 当前协议

Shigure 在独立服务器运行。本仓库不修改 `code/reconstruction/shigure_core`；`code/stages/shigure_history` 只适配既有 ROS 消息，生成 canonical frame，并通过 Unix socket 向 runtime 提供有界内存数据。

## 输入 topics

默认配置位于 `code/stages/shigure_history/settings.py`：

```text
RGB                 /rs/color/compressed
Depth               /rs/aligned_depth_to_color/compressedDepth
CameraInfo          /rs/aligned_depth_to_color/cameraInfo
object_detection    /shigure/object_detection
object_tracking     /shigure/object_tracking
segments            /Segments
people              /shigure/people_detection
contacted           /shigure/contacted
```

名称和消息类型可通过对应 `SHIGURE_HISTORY_*` 环境变量覆盖。所有订阅使用 BEST_EFFORT；canonical event 仍按 ROS source stamp 精确关联，缺失消息不会被解释为空结果。`object_detection` 或 `object_tracking` 任一侧单独到达时即可形成 canonical revision：仅有 detection 时保留 `UNRESOLVED` 事件，仅有带可信 raw ID 的 tracking lifecycle action 时形成无 mask 的 `RESOLVED` 事件。另一侧同 stamp 迟到后只丰富同一个 canonical event，不重复推进 lifecycle。

`/shigure/object_detection` 只发布 bring-in/take-out/obj-move 稀疏事件，不包含启动时所有静止物体。启动恢复使用同步 tracking/segments 的全图候选，不能等待该 topic 提供完整清单。

## Recorder 生命周期

`task_worker.start_worker()` 启动：

```text
code/stages/shigure_history/run_shigure_history_recorder.py
  --socket-server data/worker_sockets/shigure_history.sock
```

`SHIGURE_HISTORY_RECORDING_ENABLE=1` 时 worker 监控并重启 recorder。RGB-D 与 canonical frame 只保存在 recorder 进程内，不写普通运行期磁盘状态。服务器/recorder 重启后内存 sequence 清空，并由 runtime 建立新的 source epoch。

上游 object-tracking node 也可能在 recorder 不重启时独立重启。其 raw ID 形如 `<时间前缀>_<数字序号>`；adapter 只在一帧 tracking ID 能解析出唯一前缀时观察该 namespace。前缀改变会生成新的 `source_incarnation_id`、清空旧 exact-join bucket，并触发 runtime 打开新 source epoch、撤销旧 binding。由于前缀只有秒级精度，同前缀下已退役 raw ID 在新 stamp 再现，或活动 raw ID 在新 stamp 再次报告 `bring_in`，也被视为上游重启并强制换 incarnation；同 stamp 重发不会误触发。旧 incarnation 残留在内存 ring 的帧会被忽略。canonical frame ring 以 recorder append sequence 判定最新权威帧，并按接收时的 monotonic age 淘汰；不能用可能乱序的 ROS source stamp 覆盖或立即删除较晚到达的 epoch barrier。

obj_move 的 raw ID 不可信。adapter 首次看到新的 move stamp 时立即轮换 incarnation 并发出空 barrier；该 stamp 及更早的 detection、tracking、segments、camera、people/contact 等 canonical 输入一律丢弃。barrier 之后、首个严格更晚的干净 tracking 之前，辅助 topic 仅可暂存，不得 emit canonical frame，也不得消耗 HoloLens identity-sync 重试；解锁时只保留该 tracking 同 stamp 的辅助 bucket，丢弃其他隔离期 bucket。干净 tracking 只在既有新 incarnation 中采用 namespace，不进行第二次轮换；barrier 水位继续保留，以拒绝之后迟到的旧包。

## RGB-D sample

RGB ROS stamp 是 sample 主时间戳。recorder 在 `SHIGURE_HISTORY_RGB_DEPTH_MAX_DELTA_SECONDS` 内选择最近 depth，并读取 CameraInfo，构造：

```text
CachedRgbdSample
  stamp.sec / stamp.nanosec
  rgb_bgr        uint8 HxWx3
  depth          uint16 HxW, millimetres
  camera_info.k  3x3
  camera_info.width / height
```

RGB、depth 和 CameraInfo 尺寸必须一致，焦距必须为正。没有 ROS source stamp 的 RGB 不进入缓存。

## Canonical exact-stamp frame

`ShigureCompatibilityAdapter` 以完全相同的 ROS stamp 聚合 detection、tracking、segments、people 和 contacted，产生 schema v2 `CachedShigureFrame`。frame 保留：

```text
sequence / source_stamp / frame_id
source_incarnation_id
events[]
tracked_objects[]
recovery_candidates[]
people[]
input_states / diagnostics
```

input state 明确区分 `missing`、`explicit_empty`、`present`。adapter 允许 frame 在迟到 topic 到达后生成更高 sequence 的修订；runtime 只按 sequence 消费新版本。canonical event 的 `frame_id` 由该 stamp 首个 detection/tracking 消息冻结；若两者都不存在，recovery-only frame 才按 segments、camera/RGB-D、people/contact 取得展示 ID。迟到 topic 因此不会改变已持久 event key。

同一 exact stamp 的 enriched revision 保持稳定 event identity。首个 detection/tracking 消息冻结该 stamp 的 event frame identity，并为每条事件分配 `canonical_event_index`；另一侧随后建立唯一映射时，把 detection/tracking alias 合并到首次分配的 index。因此无论哪一侧先到，后到的辅助或 lifecycle topic 都不会改 canonical key。people/contact/图像等迟到后，已有 lifecycle row 只补原先缺失的 pose、box、图片或骨骼证据；不会重复推进 presence 或新建历史行。segments 与 object tracking 同时 `explicit_empty` 且没有 candidate 时，是可完成启动恢复的完整空 snapshot；`missing` 不具有这个语义。非空恢复候选在同 stamp tracking 或 exact RGB-D 尚未到达时保持 `PENDING`，不会先执行 DINO 得出终态。

## Detection 临时键与 raw ID

既有 `DetectedObject` 没有 Shigure ID。adapter 先为每个条目生成：

```text
<sec>_<nanosec>:detection:<index>
```

随后只在 exact-stamp 数据中，用 action、bbox、tracking/segment 的唯一匹配解析 raw ID。segment 与 tracking 必须双向互为最高 IoU，且双方相对各自次优项的 margin 都至少为 0.1；唯一对应时事件标记 `RESOLVED`，闪烁重复、歧义、缺失或冲突保持 `UNRESOLVED/REJECTED`。这个临时键和解析出的 raw ID 都只在当前 `source_incarnation_id/source_epoch_id` 有效，不能跨 recorder 或 object-tracking node 重启持久复用。

## bbox-local mask

远端 event mask 是 bbox-local。adapter 必须：

1. 解码 mask，并验证其尺寸恰好等于 bbox 的整数宽高。
2. 把它贴回同步 RGB 的原始全图坐标；越界部分只裁剪。
3. 输出 `mask_coordinate_space=full_frame` 的 canonical PNG。

禁止把 bbox-local mask resize 到全图。mask 只用于 scene/object crop、DINO identity 和参考视图，不参与 collider 三维 box 几何。

## Tracking、collider 与骨骼

- tracking/segments 提供稳定候选、raw ID 和 collider。
- collider 的 `x/y/z/width/height/depth` 保持 Shigure camera 毫米定义，交给 `spatial_box_v2.py` 与 ArUco 校准生成严格 8 点 box。
- people joint 的 `projection_point` 保持 camera 毫米定义，runtime 转成 ArUco 点线骨骼。
- contacted 只提供人与物体的同步关联证据；不触发人体 mesh 生成。

box 无效时就是无 box，不从 mask/depth 推测替代几何。

## Socket API

业务代码使用 `ShigureRgbdCache`：

```text
status()
iter_samples(start, end)
newest_sample()
get_sample(stamp)
iter_canonical_updates_after(sequence, include_masks=False)
latest_canonical_frame(include_masks=False)
latest_recovery_frame(include_masks=False)
```

mask 默认不通过 socket 返回；identity/artifact 处理需要时显式设置 `include_masks=True`。socket status 是即时内存查询，不写入磁盘状态文件。

## 可选 debug disk ring

`SHIGURE_DEBUG_CACHE_ENABLE` 默认关闭。开启后 `ShigureDebugDiskRing` 把 RGB-D/canonical 数据写入独立 `data/shigure_debug_cache`，只供问题复查：

- 稀疏事件/恢复候选使用 exact-stamp RGB；没有完全相同 stamp 的 RGB 时只记录不可用诊断，不以邻近 RGB 冒充事件图像；
- retention 必须大于 0 且不超过 600 秒；
- entry 数量同时受上限约束；
- runtime 不能把 debug ring 当作输入或恢复源；
- 不通过 DVC 保存。

## Marker pose

Shigure camera 到 ArUco 的 marker pose 位于：

```text
data/aruco/shigure_marker_history/latest_marker_6d_pose.json
```

当前只读取 `opencv_camera_pose.rotation_matrix` 与 `opencv_camera_pose.tvec_m`。它用于 collider 八点 box、tracking pose 和 joint points 转换；Unity 不接收该文件或内部 ArUco pose。
