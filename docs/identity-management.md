# 物体身份、模型复用与 revision

更新日期：2026-07-14
状态：Shigure v2 当前协议

系统以持久 `display_object_id` 统一 HoloLens 模型任务、Shigure 生命周期和 Unity 呈现。`task_id` 只表示一次拍摄任务；Shigure raw ID 或 recorder 临时 ID 只在一个 `source_epoch_id` 内有效。

## HoloLens 身份匹配与模型复用

`historical_model_match` 使用当前 HoloLens color + SAM mask 计算 DINOv2 embedding，并按 `display_object_id` 聚合历史 capture。最佳距离必须通过 `DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD`；启用 margin 时还必须与第二候选拉开 `DINO_IDENTITY_MATCH_SECOND_MARGIN`。

`force_new_3d_model` 的语义：

- 匹配且为 `0`：复用该对象最新 completed 模型资产。
- 匹配且为 `1`：保持 `display_object_id`，生成新 `model_revision`。
- 未匹配：创建新的 `display_object_id` 和模型。

HoloLens 拍摄始终可更新模型、精确 capture pose 和 identity reference，但不决定物体 presence、bring-in 或 take-out。它的历史模型复用虽然最多向后扫描 500 条数据库记录来跳过重复对象，实际只对最近 5 个不重复 `display_object_id` 各取一条最新 capture 做 DINO；500 不是候选上限。

完成 capture 后，服务器建立 `HOLOLENS_CAPTURE` identity sync job。它在当前 canonical 全图候选中先比较 HoloLens/ArUco capture box 与 Shigure collider 经 ArUco AABB 重建后得到的中心和 extent；单个可信候选通过宽松几何门时直接使用 collider 几何；多个候选通过时在门内用宽松 DINOv2 判别；若可信 tracking raw-ID 候选已偏离原 capture 几何位置，则启用带距离阈值和次优间隔的 DINO fallback。用于快速建立 object-tracking binding 的 recovery candidate 必须连续 2 个不同 tracking stamp 保持可信 raw ID 与严格 bbox/mask 一致；首次完成候选判别后，第二帧复用同一可信 tracking raw ID，避免重复计算整组 DINO。segment/tracking 仍须双向互为唯一最佳，且两侧 IoU margin 均至少 0.1。成功时可建立当前 epoch binding、激活 presence；快速两帧同步本身不登记长期 Shigure identity reference，后续独立稳定视图收集仍须连续 5 帧并通过严格 DINO anchor 距离与次优 margin。授权来源记录为 `shigure_recovery_snapshot`，不是 HoloLens capture 本身。

仅由 HoloLens 新建的对象保持 `presence=UNKNOWN`。completed 模型可以交付给 Unity 作 capture preview，也会持续列入 realtime 模型目录以支持自动下载和历史访问；这不会使它成为 Shigure 权威的 `PRESENT`。`PRESENT/ABSENT` 只能来自 Shigure bring-in/take-out 或可信 recovery snapshot。

## 模型后端

统一 stage 名为 `model_generation`，`MODEL_GENERATION_BACKEND` 只允许 `instantmesh` 或 `sam3d_objects`。两者都写入 canonical `ModelGeneration`、`RuntimeMesh` 和 `Blender` 结果，不改变 identity、revision、下载或 Unity pose 协议。

## 持久状态

每个 `display_object_id` 至少维护：

```text
active_model_task_id / active_model_revision
latest_hololens_pose_revision / latest_hololens_pose_aruco
latest_tracking_pose_revision / latest_tracking_pose_aruco
presence / presence_epoch
latest spatial-box corners in ArUco
multi-view identity references
lifecycle history
```

同一对象的新模型 revision 替换活动版本，但旧任务 artifact 和生命周期记录保留。Unity 以 `display_object_id + model_revision` 复用本地 FBX；pose、presence、box 和历史 event 具有各自 revision/epoch，不互相冒充。

## Runtime/source epoch 边界

服务器启动建立 `runtime_session_id`；recorder incarnation，以及 `/shigure/object_tracking` raw ID 的时间前缀，定义短生命周期的 `source_epoch_id`。绑定键为：

```text
source_epoch_id + raw_shigure_object_id -> display_object_id
```

远端 `DetectedObject` 不带可靠 Shigure ID 时，compatibility adapter 可以根据同步 tracking/segment 解析临时 raw ID；该映射只在当前 source epoch 有效。服务器、recorder 或上游 object-tracking node 独立重启时必须新建 epoch，禁止复用上一次的临时/raw ID。adapter 通过 tracking ID 末尾数字前的时间前缀变化识别第三种情况；前缀只有秒级精度，因此同前缀下已退役 ID 在新 stamp 再现，或活动 ID 在新 stamp 再次 `bring_in`，也会强制打开新 epoch。

## Bring-in、take-out 与 move

- `bring_in`：若当前 epoch 尚无 binding，可对完整位置上的 mask crop 计算 DINO embedding，并在最近最多 5 个持久对象中建立唯一 binding。歧义或不合格时保持 unbound，不创建由 Shigure ID 定义的新持久对象。
- `take_out`：只能使用同一 source epoch 先前已建立的 binding；这里禁止重新用 DINO 猜身份。成功时在同一事务内记录拿走前 pose、collider box、scene image 和骨骼，把 presence 设为 `ABSENT`、清空 active binding，并以 `take_out_completed` 撤销旧 binding。之后同一物体的 `bring_in` 可以把上游新 raw ID 重新绑定到原 `display_object_id`。
- `obj_move`：上游不能可靠指出被移动对象。适配器首次看到新的 move stamp 时轮换 source incarnation、立即发出不含物体数据的 epoch barrier，并丢弃时间不晚于该 barrier 的所有 canonical topic payload。barrier 后但干净 tracking 尚未到达时，较新的辅助 topic 只可暂存、不可 emit 或消耗 HoloLens sync 重试；解锁时只合并该 tracking 同 stamp 的辅助 bucket，其他隔离 bucket 丢弃，且不会再次轮换 incarnation。之后在新 epoch 内重新执行严格恢复。因此 segment/raw-ID 映射、live box、identity reference 和 FoundationPose 均不会消费污染帧。

## 启动恢复

恢复候选来自 Shigure 的全图 segments/tracking snapshot，而不是稀疏 `/shigure/object_detection` 事件。候选池只包含最近 5 个已有模型且有活动 identity reference 的持久物体；HoloLens pose 不是进入候选池的前置条件。一次不确定的 DINO 结果不会关闭恢复：非 `BOUND` 结果会在后续新 source stamp 上受节流地重试，已有同 epoch binding 被固定保留，直到全部绑定或达到配置的最大尝试次数。

`/shigure/object_detection` 只有稀疏事件，不能列出启动时全部静止物体。恢复逻辑因此读取全图 tracking/segment 候选；非空候选必须等到同 stamp `object_tracking=present` 和 exact RGB-D，输入未齐时 job 保持 `PENDING` 且不执行 DINO。输入完整后：

1. 只取最近最多 5 个无重复持久 `display_object_id`，为每个 candidate×display 组合生成 DINO edge cost。
2. 对完整矩阵求一对一全局方案：优先匹配带可信 raw ID 的 candidate，再最大化匹配数量，最后最小化总距离；不按 candidate 顺序贪心。
3. 最佳与次佳同优先级/同规模方案的总距离 margin 不足时，将有差异的 candidate 标为 ambiguous。
4. 只有已解析出当前 raw ID 的 candidate 才能激活 epoch binding；没有 raw ID 的匹配只记 provisional，恢复 job 继续保持 `PENDING` 并等待 enriched frame。

恢复不依赖新的 HoloLens 拍照，但需要有效的持久 identity reference。DINO 失败、margin 不足或冲突时明确记录本次未绑定结果，并在新 source stamp 上继续受限重试；达到最大次数后才终止。

没有 candidate 且 segments/tracking 都是 `explicit_empty` 时，这是可完成的完整空 snapshot，结果为 0 match；`missing` 不等于空，不能结束恢复。非空 segments 与尚未到达的 tracking 也不是完整 snapshot。

## 多视角 identity reference

参考库只积累经过严格准入的 Shigure 视角 mask 图：必须是连续 5 个不同 tracking stamp 的真实同步帧，segment probability、bbox IoU、mask IoU 和 mask 面积变化全部通过；随后 DINO 必须把目标 `display_object_id` 判为最近对象，距离不大于 0.15，且存在第二候选时 margin 不小于 0.08。待保存图片不能自我证明，必须命中已有 HOLOLENS/SHIGURE anchor。最后仍需通过 novelty 阈值才新增，避免连续重复帧。startup 或稀疏 bring-in 的单帧图只可用于当前匹配，不直接写入示例库；FoundationPose 也只在同一帧通过严格 DINO 后调度。candidate scene/mask/crop 只在系统临时目录中服务一次推理，完成或失败后删除；通过全部准入的 scene/mask 才复制到 `data/identity_references/views/<sha256>/`。HoloLens 最新 mask 作为辅助 anchor，不作为 Shigure 生命周期的前置条件。

bbox-local mask 必须按原 bbox 贴回全图再裁切，不能直接 resize 成全图。外观 embedding 可以辅助 identity，但不生成 collider 3D box，也不替代 ArUco 校准。

## 关键配置

```text
DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD=0.20
DINO_IDENTITY_MATCH_SECOND_MARGIN=0.05
DINO_IDENTITY_MATCH_REQUIRE_MARGIN=1

SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS=5
SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD=0.20
SHIGURE_IDENTITY_MATCH_SECOND_MARGIN=0.05
SHIGURE_IDENTITY_MATCH_REQUIRE_MARGIN=1
SHIGURE_IDENTITY_VIEW_NOVELTY_DISTANCE=0.08
SHIGURE_EXAMPLE_STABLE_MASK_FRAMES=5
SHIGURE_EXAMPLE_STABLE_BBOX_IOU=0.82
SHIGURE_EXAMPLE_STABLE_MASK_IOU=0.78
SHIGURE_EXAMPLE_MAX_MASK_AREA_RATIO=1.25
SHIGURE_EXAMPLE_MIN_SEGMENT_PROBABILITY=0.70
SHIGURE_EXAMPLE_DINO_DISTANCE_THRESHOLD=0.15
SHIGURE_EXAMPLE_DINO_SECOND_MARGIN=0.08

SHIGURE_SPATIAL_BOX_MISSING_GRACE_SECONDS=1.5
SHIGURE_SPATIAL_BOX_ACQUIRE_FRAMES=2
SHIGURE_SPATIAL_BOX_FILTER_WINDOW_FRAMES=5
SHIGURE_SPATIAL_BOX_EMA_ALPHA=0.35
SHIGURE_SPATIAL_BOX_CENTER_DEADBAND_M=0.008
SHIGURE_SPATIAL_BOX_EXTENT_DEADBAND_M=0.005

SHIGURE_HOLO_SYNC_CENTER_DISTANCE_M=0.50
SHIGURE_HOLO_SYNC_SIZE_LOG_TOLERANCE=1.0
SHIGURE_HOLO_SYNC_DINO_DISTANCE_THRESHOLD=0.35
SHIGURE_HOLO_SYNC_DINO_MARGIN=0.03
SHIGURE_HOLO_SYNC_MAX_RECOVERY_ATTEMPTS=30
```
