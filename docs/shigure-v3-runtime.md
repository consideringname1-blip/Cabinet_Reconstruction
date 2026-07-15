# Shigure v3 运行时协议

更新日期：2026-07-15
状态：当前权威协议

本文是 Shigure 身份绑定、初始化、历史原位、空间框和 HoloLens 呈现的当前说明。其他仍标为“Shigure v2 历史说明”的文档仅用于解释旧实现；与本文冲突时以本文和代码为准。

## 目标与职责边界

- HoloLens 上传图是持久物体分类库的唯一来源；Shigure 的 scene/mask 只用于本次查询和诊断，不能写成长期 identity reference。
- Shigure raw ID 只在一个 `source_epoch_id` 内有效。服务器维护 raw ID 到持久 `display_object_id` 的本次启动绑定及变迁。
- FoundationPose 不做连续追踪，只在初始化成功绑定后和选中的 `take_out` 上运行。
- 模型 live pose 表示最近一次可信“原位”，不是每帧追踪位置。历史原位跨服务器启动持久化，每个物体最多保留 5 个。
- HoloLens 不再轮询或显示 raw `/shigure/object_tracking` collider box，只消费完整 live snapshot 中按 `display_object_id` 组织的 mask-depth AABB。
- 历史照片、人体骨骼和历史 box 暂不在 HoloLens 呈现，也不能阻塞模型历史再放置。

## 身份库与 raw ID 绑定

每个 `display_object_id` 最多保留最新 5 条活动 `HOLOLENS` identity reference。新 HoloLens 上传更新这个滑动窗口；v3 停用旧的 Shigure reference，运行时也拒绝新增 `SHIGURE` reference。

分类规则：

1. HoloLens 历史模型匹配先公平选取最多 50 个物体，再按物体聚合最新最多 5 次 HoloLens capture；一个高频物体不能占满全局读取窗口。
2. Shigure runtime 从最多 50 个当前物体中查找，以该物体最新最多 5 条 HoloLens reference 的最小 DINOv2 距离作为分数。
3. 已绑定且仍出现的 raw ID 直接继承绑定，不重复做 DINOv2。
4. 未绑定 raw ID 只在出现新的、与上次尝试不高度重合的 mask 时重新分类；失败后等待下一张实质不同的 mask。
5. 新 raw ID 通过阈值和次优 margin 后先登记为该物体的 alias。若旧 primary 同时仍有 mask，则只为这次竞争额外计算旧 mask 对同一 HoloLens reference bank 的 DINOv2 距离：仅当旧 mask 距离更小时保留旧 primary，否则提升最新的最优新 alias；旧 primary 在完整 snapshot 中消失时也从可见 alias 中选择 DINOv2 更优者（没有可比距离时选最新绑定）。
6. 启动恢复和稳态 reconcile 使用相同的逐 mask 线性判定，不做 display one-to-one 全局分配，因此多个可信 raw ID 可以独立指向同一个持久物体。一个 raw ID 仍不能同时绑定两个物体；`take_out` 完成时撤销该物体在当前 epoch 的全部 alias。

live item 会报告 `primary_binding_id`、`primary_raw_shigure_object_id` 和 `shigure_aliases`；HoloLens 只显示 primary 对应的最新空间框。

## 初始化与诊断

runtime 在服务器启动时即进入初始化等待，但只有以下输入同时可用才消耗一次 FoundationPose 尝试：

- 同一 exact stamp 的 RGB、depth 和 CameraInfo；
- 能按 RGB 尺寸解码的非空 candidate mask；
- candidate 已解析到可信 tracking raw ID；
- Shigure camera 到 ArUco 的当前校准；
- 已通过 HoloLens-only DINOv2 建立的 primary binding。
- `active_model_task_id` 对应任务已经 `completed`，FoundationPose 所需模型 artifact 已可读取。

每个物体、每个启动 epoch 最多进行 5 次初始化 FoundationPose。成功结果写入 `display_object_origin_history`，`kind=INITIALIZATION`。校准或 exact 输入尚未就绪时保持等待，不计失败次数。

启动 identity job 采用 fail-closed 终态：缺 exact RGB-D/CameraInfo/校准/有效 mask，或任一非空 candidate 尚未解析出可信 raw ID 时，写 `STARTUP_RECOVERY/PENDING` 和可取得的输入材料后立即返回。只有 `segments` 与 `object_tracking` 都是 `explicit_empty` 的完整空 snapshot，或非空 snapshot 的每个 candidate 都取得终态匹配结果，才写 `COMPLETED`；因此 `results=[]` 不能把未就绪候选误判为恢复完成。

身份绑定有意在耗时的模型生成之前建立。模型任务尚未完成时，runtime 只写 `initialization_waiting_for_completed_model_artifacts` 恢复报告，不递增初始化 attempt；因此漫长的模型生成不会提前消耗 5 次真实 FoundationPose 配额。

诊断目录为：

```text
data/shigure_recovery_debug/<runtime_session_id>/<source_epoch_id>/
```

其中保留 bootstrap、mask reconcile 和初始化尝试的 `report.json`，以及可用的 `scene.png`、`mask.png`、`depth.png`、`object_crop.png`。这些文件用于解释没有绑定或没有初始化位姿的原因，不会反向作为 identity reference 或业务恢复输入。

## 偶发 FoundationPose 与原位历史

持续 tracking/mask 不更新模型 pose。v3 只有两种 origin：

- `INITIALIZATION`：本次 epoch 首次可信绑定后，用同一物体的 exact RGB-D/mask 初始化模型原位；
- `TAKE_OUT`：`take_out` 触发后，从生命周期选择窗口中选定同一物体的候选，再用其 exact RGB-D/mask 计算离开前原位。

近时刻重复事件沿用默认 1 秒生命周期选择窗口：`take_out` 偏向更早候选，`bring_in` 偏向更后候选，DINOv2 更优者可以覆盖时间优先。相邻 `take_out`/`bring_in` 的 mask-depth 三维中心移动小于 20 cm 时视为前景遮挡或未移动，不提交新的移动结果。

canonical event 的解析、lifecycle row、presence 和 binding 变更在同一个数据库事务内提交；任一约束失败时整笔回滚，不能留下“event 已 RESOLVED、lifecycle/binding 未提交”的半状态。服务器重启、runtime 关闭或 source epoch 轮换时，也会在关闭旧 epoch 的同一事务中先终结 lifecycle authority：未决事件改为 `REJECTED`，已经 RESOLVED 但没有 lifecycle row 的异常记录保留审计并标为不可安全 replay，然后才撤销旧 binding、清 live 几何并关闭 epoch。

origin 跨启动持久化，并按物体执行：

- 以事件 `occurred_at` 排序，数据库 `id` 只作相同时刻的稳定 tie-break；
- 相邻位置小于 20 cm 时去重；
- 最多保留事件时间上最新 5 个，异步 FoundationPose 乱序完成不能改变历史先后；
- live snapshot 优先使用最新 origin，没有 origin 时才使用最新 HoloLens capture pose；
- 当前模型 revision 可以使用同一 `display_object_id` 的旧启动 origin。

点击模型时，HoloLens 以 `kind=origin` 和 `before_cursor` 查询更早 origin，沿用已有约 0.35 秒飞行动画。第一次点击从 live item 的 `latest_origin_cursor` 之前开始；到最老记录后再次点击会回绕到最新 origin。历史位置只改变模型 pose，当前 live 空间框仍独立更新。

有当前 ArUco reference 时可把跨启动 origin 转换到当前 HoloLens local。没有 ArUco 时，live catalog 只从当前 `startup_session_id` 的 `latest_hololens_task_id` 读取 `object_hololens_current`；模型资产仍可由较早任务复用，但该资产任务不能提供当前局部坐标。startup-local pose 的 revision 使用当前 capture task 的数据库自增 ID，canonical ArUco pose 保持 `NULL`，这一目录绝不授权跨启动坐标。迟到重放较旧的 local-only capture 会按持久 capture 时间拒绝倒退，不能替换新的 local task/model，也不能清掉后来补齐的 canonical pose。

没有 ArUco、但存在同启动的 `object_aruco` ↔ `object_hololens_current` anchor 时，history 只转换该 anchor 之后的同启动 origin。连同启动 anchor 也没有时，history API 不返回 400，而返回 `success=true`、`coordinate_epoch=startup-local:<startup>`、`latest_origin_cursor=null`、`history_event=null`；Unity 沿用飞行动画回到已经接受的 `LatestLive` 原位，绝不复用旧启动局部坐标。

历史响应应用瞬间必须满足：该对象已有有效 `LatestLive` pose、`LatestLive.CoordinateEpoch` 非空，并与响应 `coordinate_epoch` 按 ordinal 完全相等。迟到的旧 epoch 响应在修改 presentation、transform 或 origin cursor 之前即被拒绝；后续 live snapshot 的自动恢复只是第二道保护，不能替代该入口校验。

## mask-depth 3D AABB

空间框来自 primary raw ID 对应 candidate 的 exact mask、depth 和 CameraInfo：

1. 只取 mask 内有效深度像素并反投影到 Shigure camera；
2. 用当前 camera-to-ArUco 变换转换到 ArUco；
3. 对三个 ArUco 轴分别取稳健分位范围并设置最小边长；
4. 生成与 ArUco 轴平行的 8 个角点；
5. 以完整 snapshot 原子更新 `display_object_id` 的 `spatial_box`。

这不是 raw collider 的平滑或验证结果。primary 在新的完整 snapshot 中消失时，服务器发布 `no_box`，HoloLens 删除旧框；新的 primary 出现则新增或替换框。alias 的框不显示。

## HoloLens live snapshot

HoloLens 应用启动后先对 `POST /realtime-tracking/mode` 做 `mode=live` handshake，之后默认每 2 秒请求：

```text
GET /realtime-tracking/status?startup_session_id=<id>
```

响应是最多 50 个当前 display object 的完整 snapshot。客户端原子处理每个 item 的模型、pose、origin cursor、绑定表和 `spatial_box`，并删除完整响应中已经缺失的框。`spatial_box` 字段始终存在，状态是 `ready` 或 `no_box`。

有 ArUco 时 `coordinate_epoch` 是当前 reference task，pose 来自最新持久 origin（缺失时为最新校准 HoloLens pose）。无 ArUco 时 `coordinate_epoch=startup-local:<startup>`，只发送当前启动 capture 的 local pose；`active_model_task_id` 只决定模型资产，不能冒充 pose task。模型任务尚未完成时 item 可以先存在但不带 `model` 下载字段，初始化也继续等待而不消耗 attempt。

旧 raw tracking box endpoint 可保留作服务器诊断兼容入口，但不是 HoloLens v3 数据链；客户端没有独立 1 秒 raw-box poller，也不把 raw tracking ID 直接映射成 Unity 显示对象。

## 当前相关 HTTP

| 入口 | v3 用途 |
| --- | --- |
| `POST /generate` | HoloLens 模型/ArUco 上传。 |
| `POST /check-queue` | 异步模型任务状态与完成模型。 |
| `POST /realtime-tracking/mode` | 建立当前 startup 的 live handshake。 |
| `GET /realtime-tracking/status` | 每个 display object 的完整 live snapshot。 |
| `GET /api/v2/display-objects/<id>/history?kind=origin...` | 最多 5 个持久 origin 的游标查询。路径保留 `/api/v2/`，但 `kind=origin` 是 v3 契约。 |
| `GET /api/v2/identity-sync/<id>` | HoloLens reference 登记结果；绑定本身等待新 Shigure mask。 |

## 数据库 v3 与迁移

`task_db.py` 当前 schema version 是 3。v3 新增 `display_object_origin_history`，把每物体 origin 上限固定为 5，并允许一个 display object 在同 epoch 保留多个 raw-ID alias。正常启动只创建全新 v3 数据库或验证严格 v3 schema，不在启动路径静默修补旧库。

严格 v2 数据库升级使用：

```bash
python code/migrate_shigure_v3_data.py
python code/migrate_shigure_v3_data.py --apply
```

第一条只读检查；`--apply` 在原子迁移前创建并校验 `tasks.db.pre_shigure_v3` 及 metadata，备份不会自动覆盖。更早的 legacy 数据库必须先按 `migrate_shigure_v2_data.py` 的说明升级到严格 v2，再执行 v3 迁移。

如果数据库已经是 v3、但建立 v3 时尚未从校准后的 HoloLens pose history 回填跨启动原位，使用独立的一次性修复：

```bash
python code/migrate_shigure_v3_data.py --repair-origin-history
python code/migrate_shigure_v3_data.py --repair-origin-history --apply
```

第一条仍是只读检查；第二条才执行原子 merge。它与 v2→v3 备份相互独立，执行前创建并校验 `tasks.db.pre_origin_backfill_v1` 和对应 `.meta.json`，已存在时拒绝覆盖。merge 会把校准后的 `display_object_pose_history` 候选与现有 origin 按时间合并，保留并优先原生 v3 FoundationPose origin，移除已知不安全的旧 v2 lifecycle fallback，再应用 20 cm 邻近去重和每物体最新 5 条上限。

成功后在 `schema_metadata.detail_json.hololens_origin_backfill_v1` 写入结果与 `applied_at`。该 marker 使命令幂等：再次运行返回 `already_repaired`，不会重建历史或再创建备份。正常 server 启动不会隐式执行这项修复。

## 回归检查

```text
code/tests/test_task_db_v3.py
code/tests/test_shigure_mask_aabb.py
```

协议变更至少应同时检查数据库迁移/历史上限、HoloLens-only reference、raw-ID alias/primary、完整 snapshot 删除语义，以及 origin 分页回绕。
