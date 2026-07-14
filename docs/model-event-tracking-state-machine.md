# Unity 模型、缓存与追踪状态机

更新日期：2026-07-14
状态：Shigure v2 当前协议

Unity 以 `display_object_id` 管理模型，把服务器的 latest live state 与场景中的 presentation state 分开。历史位置只冻结所选模型的场景呈现；live status、pose、box 和模型 metadata 仍继续接收和更新。

## Pending preview

`/check-queue` 的 pending item 可包含：

```text
sam3_spatial_box.status=ready
sam3_spatial_box.coordinate_space=unity_world
sam3_spatial_box.aabb_min_world
sam3_spatial_box.aabb_max_world
```

它只在当前 HoloLens 生成任务等待期间显示临时 box，不创建正式 runtime 模型记录，也不等于 Shigure collider box。

## 正式模型加载

`TryBuildRuntimeModelInstance` 要求 `model_instance` 恰好包含：

```text
model_key
task_id
display_object_id
model_revision
fbx_url
pose
coordinate_space=hololens_current_local
```

加载流程：

1. 校验 canonical 字段和 pose。
2. 以 `display_object_id + model_revision` 查询本地 FBX 缓存。
3. 命中缓存时直接加载；未命中才下载 `fbx_url`。
4. 注册到 `RuntimeModelManager`，按服务器 pose 设置 position/rotation。
5. 同一 `display_object_id` 的新 revision 替换当前模型记录和当前 cache bundle。

无效 pose、revision 或坐标空间直接拒绝，不在客户端猜测。live item 可以只更新已有模型的位置/box，也可附带 canonical `model` 让离线缓存缺失的客户端补载模型。

## HoloLens 确认上传

Unity 保留普通上传和强制重建入口。普通上传按身份复用规则处理；强制重建发送 `force_new_3d_model=1`，为匹配到的 `display_object_id` 创建新模型 revision。近距使用 AHAT，远距使用 LONGTHROW。

HoloLens 只承担更精确的拍照、模型生成和校准辅助，不是物体存在或 take-out 的判定源。presence 与拿取前位置以 Shigure 生命周期为准。

新 HoloLens 对象保持 `presence=UNKNOWN`，completed 模型只作为 capture preview 交付。随后运行的 identity sync 只有在连续稳定且 raw ID 可信的 Shigure recovery snapshot 中匹配成功，才会建立 epoch binding、激活 `PRESENT` 并登记 identity reference；presence authority 来自该 Shigure snapshot，而不是 capture。

## Live transport

Unity 启动或重新握手时只发送 `mode=live`。`/realtime-tracking/status` 持续接收：

```text
display identity / model revision
latest live pose and pose revision
presence and presence epoch
ready/no_box spatial-box revision；ready 时包含严格 8 角点
optional canonical model metadata
```

旧 `mode=history` 服务端分支已删除。每份成功 status 中，`items` 仍只承载最多 5 个模型/live pose；独立的 `spatial_boxes` 是无数量上限的 Shigure box 完整快照，并以 `spatial_box_snapshot_complete=true` 授权客户端 reconcile/delete。Unity 按 `startup_session_id`、`request_generation`、`mode_epoch`、`coordinate_epoch`、model revision 和 pose/box revision 拒绝迟到数据。

`HistoryPresentationController` 的 URL override 默认为空；此时它从 `ShuJuQingQiu` 已配置的 realtime status/mode URL 提取 scheme、host、port 和服务路径，再派生同源 `/api/v2/`，不写死另一台服务器。

每次 live snapshot 应用前，`RuntimeModelManager` 会比较 `coordinate_epoch`。仍在显示旧 epoch 历史的对象会自动恢复 `FollowLive`、应用最新 live state，并隐藏历史照片/骨骼；历史响应本身若与当前 live epoch 不同则拒绝应用。

`ShuJuQingQiu` 先验证完整 `spatial_boxes` 的 count、complete 标记、ID 唯一性和每个 `ready|no_box`；任一项损坏时整份 box snapshot 不应用。验证成功后再原子更新并把 ready ID 集合交给 `RuntimeModelManager` reconcile。`FollowLive` 对象不在集合中时删除 `LatestLive` box/线框；`History` 对象的历史 box 不受影响，恢复 live 时再按最新状态决定是否显示。

## 单击某个模型

`ModelEventDisplay` 把单击交给 `HistoryPresentationController`：

1. 请求该 `display_object_id` 最新一条完整 `take_out` 历史。
2. 收到后把该模型的 presentation mode 设为 `History`，应用拿取前 pose 和对应历史 box。
3. 自动显示该事件的 scene image 与 Shigure 点线骨骼，不需要第二次点击图片。
4. 再次点击同一物体，把当前 `history_cursor` 作为 `before_cursor`，显示更老的一条记录及其照片/骨骼。

服务端会跳过无 pose、无 scene image、无有效骨骼或无法转换的历史行，并继续跨数据库页查找；只有真正耗尽后才返回 `history_event=null`。没有更老的完整记录或客户端校验失败时，保留当前呈现，不用不完整记录覆盖它。

模型处于 `History` 时，`RuntimeModelManager` 仍接受更新的 live pose/box revision 并保存到 `LatestLive`，但不把它们应用到该模型 transform。恢复后直接应用期间收到的最新 live state，不需要服务器回滚或重算。

## 全体历史/继续追踪按钮

`ToggleAllPresentations()` 的语义由本地呈现状态决定：

- 当前全部 `FollowLive`：为每个已加载对象请求最新完整 take-out 记录，分别显示历史 pose、box、照片和骨骼。
- 任一对象已在历史呈现，或全体历史请求仍在进行：取消待处理历史请求，全体恢复 `FollowLive`，应用各自最新 live pose/box，并关闭全部照片和骨骼。

全体进入历史后仍可继续单击某个物体，逐条向更老的 `history_cursor` 翻页。全体继续追踪不会停掉服务端任务；它只切换 Unity transform/evidence 的呈现。

## 照片与骨骼证据

历史 evidence 必须恰好包含 `scene_image_url` 和 `skeleton`。骨骼是 joint 点与按名称连接的线段，并以颜色区分骨段；不加载人体网格。证据以 `display_object_id + event_uid` 绑定，必须与正在显示的历史 pose 属于同一生命周期事件。

`ObjectEvidenceDisplay` 在历史成功应用后自动显示两者；恢复单物体或全体 live 时隐藏对应证据。证据不参与模型 identity、scale、pose 或 revision 计算。

## Runtime spatial box

live box 是带独立 revision 的 `ready | no_box` 判别联合：`ready` 包含 `hololens_current_local` 的 8 个有限角点，`no_box` 清除更旧线框；无上限完整 `spatial_boxes` 中缺席也会删除旧 `LatestLive` box。服务端对 raw collider 做 2 帧获取、5 帧中值＋EMA 与中心/边长死区；明确缺失后保持最后可信框 1.5 秒，再决定下发 `no_box`。take-out 立即清框。历史 box 为同坐标空间的 8 点或 `null`。合法八点按固定 12 条边绘制无填充长方体，线径为 5 mm；box 与 mesh tracking 分开维护，不从图像或 depth 降级。

## 本地容量与隐藏

`RuntimeModelManager` 默认：

```text
maxVisibleModels = 5
maxCachedModelFiles = 5
```

缓存按 `display_object_id` 形成 bundle；同一对象的 superseded 本地文件不占多个对象槽位。

`ClearLocalRetainedModels()` 是显式本地隐藏操作：恢复 presentation state、清空照片/骨骼、隐藏 runtime 模型，并中止客户端当前 status/download 自动下发；它保留 runtime records 和本地 FBX 文件，也不停止服务器的 Shigure/FoundationPose 计算。之后重新启用自动下发时，已有缓存优先恢复，缺失时才下载。
