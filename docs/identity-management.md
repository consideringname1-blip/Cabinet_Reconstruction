# 物体身份、模型复用与 revision

更新日期：2026-07-12
状态：当前协议

系统以持久 `display_object_id` 统一 HoloLens 拍摄、历史模型资产和 Shigure 实时观测。`task_id` 表示一次拍摄任务，不能代替物体身份；Shigure `object_id` 只表示当前远端数据流中的临时实例。

## HoloLens 身份匹配

stage：

```text
historical_model_match
```

实现：

```text
code/stages/hololens3d_reconstruction/run_historical_model_match_from_json.py
```

流程：

1. 使用当前 HoloLens color + SAM mask 计算 DINOv2 embedding。
2. 从数据库读取已完成 capture 的 embedding。
3. 按 `display_object_id` 聚合，每个对象只保留距离最小的候选。
4. 最佳距离必须不超过 `DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD`。
5. 启用 margin 时，最佳候选与第二候选的距离差必须达到 `DINO_IDENTITY_MATCH_SECOND_MARGIN`。

无合格候选时创建新的 `display_object_id` 并生成模型。匹配歧义、候选结构错误、DINO service 错误或命中对象缺少 completed 模型时，stage 失败并报告原因。

## 模型复用

`force_new_3d_model` 是 object upload 的必填 0/1 字段：

- 匹配且为 `0`：复用该 `display_object_id` 最新 completed 模型的 `ModelGeneration`、`RuntimeMesh` 和 FBX 资产。
- 匹配且为 `1`：保持同一 `display_object_id`，运行 `model_generation` 并创建新的 `model_revision`。
- 未匹配：无论标志值如何，都为新对象运行 `model_generation`。

复用模型时，当前 HoloLens 拍摄仍重新计算：

```text
depthpointcloud
object_alignment
pose
aruco_sync
model_bounds
display_identity
```

因此几何资产可以复用，而物体位置始终来自最新 HoloLens RGB-D。

## 生成后端

统一 stage 名为 `model_generation`。`MODEL_GENERATION_BACKEND` 只允许：

```text
instantmesh
sam3d_objects
```

两个后端都写入 canonical `ModelGeneration`、`RuntimeMesh` 和 `Blender` 结果。选择后端不改变身份、revision、下载或 Unity 放置协议。

## Revision 与持久记录

每个 `display_object_id` 保存：

```text
active_model_task_id
active_model_revision
latest_hololens_pose_revision
latest_hololens_pose_aruco
latest_tracking_pose_revision
latest_tracking_pose_aruco
latest_body_revision
```

规则：

- 普通匹配复用不会为相同资产创建多个活动模型版本。
- 强制重建为该对象增加模型 revision，并保留服务器上的已有 revision 和任务 artifact。
- 新 HoloLens capture pose 成为 history 模式的恢复位置，并使针对先前 HoloLens pose revision 的 tracking 结果失效。
- tracking pose 必须与当前活动 `model_revision` 和 HoloLens pose revision 同时一致才能提交。
- Unity 缓存按 `display_object_id` 合并版本；当前 revision 替换该对象 superseded 的本地文件。

## Shigure 身份匹配

Shigure 实时事件先做几何过滤：

1. 读取最多最近 5 个活动 `display_object_id`。
2. 将各对象最新 HoloLens `Sam3SpatialBox` 投影到 Shigure camera，构造投影圆。
3. 要求 observation mask 至少 `80%` 位于圆内。
4. 要求 observation depth 中位数与对象投影中心 depth 差不超过 `0.18 m`。
5. 一个候选直接绑定；多个候选使用当前 mask 的 DINO embedding 与这些对象的最新 HoloLens reference 比较。

只有几何和 DINO 都不能唯一判定时才返回 ambiguous/unbound；不会创建仅由 Shigure ID 定义的新持久对象。

临时绑定键是：

```text
startup_session_id + ingress_session_id + shigure_object_id
```

服务器重启、Unity startup 改变、对象重新拍摄或 tracking epoch 改变后必须重新匹配。

## 配置

```text
DINO_IDENTITY_CANDIDATE_LIMIT=500
DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD=0.20
DINO_IDENTITY_MATCH_SECOND_MARGIN=0.05
DINO_IDENTITY_MATCH_REQUIRE_MARGIN=1

SHIGURE_IDENTITY_MIN_MASK_INSIDE_RATIO=0.80
SHIGURE_IDENTITY_MAX_CENTER_DEPTH_DIFF_M=0.18
SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD=0.20
SHIGURE_IDENTITY_MATCH_SECOND_MARGIN=0.05
SHIGURE_IDENTITY_MATCH_REQUIRE_MARGIN=1
```
