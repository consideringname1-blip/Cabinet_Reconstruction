# 任务处理流程

更新日期：2026-07-16
状态：Shigure v3 当前流程

本页只给出端到端流程。身份、origin、空间框、HTTP 字段和迁移的详细权威契约见 [`shigure-v3-runtime.md`](shigure-v3-runtime.md)。

## 1. HoloLens 上传与模型任务

HoloLens 上传 object capture 后，服务器按 `task_worker.STAGE_ORDER` 执行：

```text
hololens2depth
sam3mask
historical_model_match
model_generation
depthpointcloud
modelscale
object_alignment
pose
aruco_sync
runtime_mesh
model_bounds
display_identity
```

`historical_model_match` 只比较 HoloLens capture：先公平选取最多 50 个 `display_object_id`，再为每个物体取最新最多 5 个 capture；单个高频拍摄物体不能耗尽全局扫描窗口。命中时可以复用既有模型或生成新的 model revision；未命中时创建新的持久 `display_object_id`。

`display_identity` 提交模型状态、HoloLens capture 指针和 `HOLOLENS` identity reference。每个物体的活动 reference 上限是 5；同一 `view_hash` 的重放不会刷新其年龄或挤掉真正较新的视图。该步骤不尝试用当前几何直接绑定 Shigure raw ID。没有当前 ArUco 时也会提交 local-only catalog state：canonical pose 保持 `NULL`，模型资产任务与最新本地 pose task 分开保存；之后 ArUco retro-sync 对同一 task 补写恰好一条 canonical pose history，不重复分配模型 revision。旧 local-only capture 的迟到重放按持久 capture 时间 fail-closed，不能倒退最新任务/模型或清空较新的 canonical pose。identity-sync 可以立即完成为“reference 已登记、等待新 Shigure mask”，不会把 HoloLens 上传队列卡在 Shigure 处理上。

## 2. Shigure 输入与本地缓存

recorder 适配不可修改的远端 ROS topics，生成 exact-stamp canonical frame，并通过 Unix socket 的内存 ring 提供给 runtime。典型输入包括 RGB、depth、CameraInfo、Segments、object tracking 和稀疏 bring-in/take-out 事件。

`data/shigure_debug_cache` 是最多 10 GiB 的本地长期诊断环；服务器启动和它的内容不要求连续，业务逻辑也不从该磁盘缓存恢复状态。跨启动状态只来自数据库、模型 artifact、HoloLens identity reference 和持久 origin。

## 3. mask 触发身份绑定

每个 source epoch 重新建立 raw Shigure ID 绑定：

1. exact RGB-D、CameraInfo、mask 和 camera-to-ArUco 校准就绪；
2. 已绑定 raw ID 直接继承现有 `display_object_id`；
3. 未绑定 raw ID 仅在新 mask 出现时，查询最多 50 个 display object；
4. 每个物体的 DINOv2 分数取最新最多 5 条 HoloLens reference 中的最小距离；
5. 通过阈值/margin 的新 ID 先成为 alias；旧 primary 仍可见时比较两张当前 mask 对同一 HoloLens bank 的 DINOv2 距离，仅在旧 ID 距离更小时保留旧 primary，否则显示最新的最优新 ID；
6. 每张 mask 独立线性判定，不做一对一全局 assignment，允许同一物体在当前 epoch 保留多个 raw-ID alias；
7. Shigure scene/mask 写入诊断，但不进入长期 reference 库。

启动恢复中，缺 exact RGB-D、CameraInfo、校准、有效非空 mask 或已解析 raw ID 时会同时写诊断报告和 `STARTUP_RECOVERY/PENDING` job，然后原帧返回，不执行身份或 FoundationPose。只有完整 `explicit_empty` 空 snapshot，或非空 snapshot 中每个候选都得到终态匹配结果，才把启动 job 标为 `COMPLETED`。完整空 snapshot 也会删除已经消失的 primary box；topic `missing` 与 `explicit_empty` 仍严格区分。

## 4. 初始化原位

服务器启动时即开始等待初始化输入。只有同一物体的绑定、exact RGB-D/mask、CameraInfo、校准与已完成模型 artifact 全部就绪后，才开始一次 FoundationPose；等待不计 attempt。身份可以在慢模型生成前先完成，但 `active_model_task_id` 尚未 `completed` 时只记录等待原因，不会提前消耗 5 次配额。每个物体每个 epoch 最多 5 次真实尝试。

成功的初始化写入持久 origin（`kind=INITIALIZATION`）。每次等待/尝试的报告和输入图保存在：

```text
data/shigure_recovery_debug/<runtime_session>/<source_epoch>/
```

## 5. 生命周期与偶发 FoundationPose

Shigure 生命周期仍决定 `bring_in`/`take_out` 和 presence。FoundationPose 不做逐帧追踪：

- `bring_in` 可以建立或激活 binding，但不持续移动模型；
- `take_out` 先保持未决，事件选择仍为 1 秒窗口，真实移动确认期默认 3 秒；
- runtime 向前回查最多 3 秒的同物体可信观测，逐候选使用同一 stamp 的 RGB-D/CameraInfo/mask，并以完整度、深度、人员重叠和 DINOv2 选择最近清晰帧；
- 20 cm 内快速回归、原位深度连续保持、前景深度变近或证据不足都失败关闭；只有连续背景显露或身份一致且移动至少 20 cm 的 `bring_in` 才提交 `take_out`；
- 确认后用回查清晰帧运行一次 FoundationPose，并把离开前原位写为 `kind=TAKE_OUT`；
- `take_out` 完成后撤销当前 epoch 中该物体的全部 raw-ID alias。

origin 跨启动持久化、按事件 `occurred_at`（`id` 只作稳定 tie-break）排列，相邻 20 cm 内去重，每个物体最多保留时间上真正最新的 5 个；异步 FoundationPose 的旧事件即使更晚完成，也不能冒充 latest 或挤掉较新历史。

canonical event 解析、lifecycle row、presence 与 binding 更新是单一数据库事务，失败全部回滚。source epoch/server runtime 关闭也先在同一事务中终结旧 lifecycle authority：拒绝未决事件，审计不可 replay 的 RESOLVED orphan，再撤销 binding 和关闭 epoch，避免迟到事件跨 epoch 生效。

## 6. live snapshot 与 Unity 呈现

HoloLens 完成 live handshake 后默认每 2 秒读取一次完整 snapshot。每个 display-object item 包含：

- 当前模型 revision 与下载信息；
- 最新 origin pose，缺失时使用 HoloLens capture pose；
- `latest_origin_cursor`；
- primary binding 和 alias 表；
- primary mask-depth AABB 的 `ready`/`no_box` 状态。

客户端不再运行独立 raw object-tracking box poller。完整 snapshot 中缺失或变为 `no_box` 的框会被删除；新增框直接加入。历史 pose 与 live box 解耦。

点击模型使用 `kind=origin` 查询更早的持久原位并执行已有飞行动画；到最老记录后回绕最新。模型历史再放置不要求照片、骨骼或历史 box，v3 暂不显示这些证据。

无 ArUco 的 live catalog 只提供当前 startup 的 local pose；复用的旧模型资产不授予跨启动坐标。没有同启动 anchor 时 history 返回成功的空事件并飞回 `LatestLive`。客户端仅在响应 epoch 与应用瞬间的非空 `LatestLive.CoordinateEpoch` 完全相等时进入历史呈现，旧 epoch 的迟到响应不会修改 transform 或 cursor。

## 7. 数据库迁移

当前 schema version 是 3。严格 v2 升级先 dry-run，再显式 apply：

```bash
python code/migrate_shigure_v3_data.py
python code/migrate_shigure_v3_data.py --apply
```

迁移前会创建并校验 `tasks.db.pre_shigure_v3` 与 metadata。正常 server 启动只创建空 v3 或验证严格 v3，不自动修改旧库。

已经是 v3、但需要补齐旧 HoloLens 校准 pose history 的数据库，使用另一条显式 one-shot：

```bash
python code/migrate_shigure_v3_data.py --repair-origin-history
python code/migrate_shigure_v3_data.py --repair-origin-history --apply
```

它使用独立且已校验的 `tasks.db.pre_origin_backfill_v1` 备份，不覆盖 v2→v3 的 `.pre_shigure_v3` 备份。修复会合并 HoloLens 历史原位并优先保留原生 v3 FoundationPose origin，随后执行 20 cm 去重和每物体 5 条上限。成功 marker 是 `schema_metadata.detail_json.hololens_origin_backfill_v1`；重复执行返回 `already_repaired`。
