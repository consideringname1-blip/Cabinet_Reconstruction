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

新 HoloLens 对象保持 `presence=UNKNOWN`，completed 模型先作为 capture preview 交付，并在后续 ArMarker 校准后继续出现在模型目录中；模型下载和历史访问不以 presence 为门。随后运行的 identity sync 只有在连续 2 帧稳定且 raw ID 可信的 Shigure recovery snapshot 中匹配成功，才会建立 epoch binding、激活 `PRESENT` 并立即发布当前 collider box；长期 identity reference 仍由独立的严格 5 帧准入收集。presence authority 来自该 Shigure snapshot，而不是 capture。

## Live transport

Unity 启动或重新握手时只发送 `mode=live`。`/realtime-tracking/status` 持续接收：

```text
display identity / model revision
latest live pose and pose revision
presence and presence epoch
optional canonical model metadata
```

旧 `mode=history` 服务端分支已删除。每份成功 realtime status 的 `items` 仍只承载最多 5 个模型/live pose；Unity 继续按 startup、request/mode/coordinate epoch、model revision 和 pose revision 拒绝迟到模型数据。raw tracking box 不再经过此验证链，而由独立 latest endpoint 长期轮询。

`HistoryPresentationController` 的 URL override 默认为空；此时它从 `ShuJuQingQiu` 已配置的 realtime status/mode URL 提取 scheme、host、port 和服务路径，再派生同源 `/api/v2/`，不写死另一台服务器。

每次 live snapshot 应用前，`RuntimeModelManager` 会比较 `coordinate_epoch`。仍在显示旧 epoch 历史的对象会自动恢复 `FollowLive`、应用最新 live state，并隐藏历史照片/骨骼；历史接口已按请求中的当前 startup reference 完成坐标转换，因此该响应的 `coordinate_epoch` 是本次历史放置的权威值，不再用可能滞后的客户端 live epoch 拒绝。

ShuJuQingQiu 从应用启动起约每 1 秒请求 `/api/v2/shigure/object-tracking-boxes/latest`，不等待 realtime handshake、模型下载或历史状态。成功响应按 tracking_id 直接 replace：能解析的条目立即新增或覆盖，缺席条目立即删除，成功空数组清空；请求失败时保留上一快照等待下一轮。服务器不做 freshness、稳定、平滑、identity 或整批等待。

## 单击某个模型

模型的 MRTK 单击和 PointObject 命中都把 `display_object_id` 直接交给 `HistoryPresentationController`：

1. 请求该 `display_object_id` 最新一条完整 `take_out` 历史。
2. 收到后把该模型的 presentation mode 设为 `History`，应用拿取前 pose 和对应历史 box。
3. 自动显示该事件的 scene image 与 Shigure 点线骨骼，不需要第二次点击图片。
4. 再次点击同一物体，把当前 `history_cursor` 作为 `before_cursor`，显示更老的一条记录及其照片/骨骼。

历史 pose 只由 Shigure `take_out` 打开的 1 秒候选窗口更新：近邻 take_out 取更早帧或 DINO 更优帧，bring_in 取更后帧或 DINO 更优帧；take_out/bring_in 的 mask+depth 三维中心移动小于 20 cm 时按遮挡未移动处理，不覆盖历史位置。服务端只跳过无 pose 或无法转换的历史行，并继续跨数据库页查找；scene image、骨骼和 spatial box 均为可选证据，只有真正耗尽后才返回 `history_event=null`。没有更老的完整记录或客户端校验失败时，保留当前呈现，不用不完整记录覆盖它。

模型处于 `History` 时，`RuntimeModelManager` 仍接受更新的 live pose/box revision 并保存到 `LatestLive`，但不把它们应用到该模型 transform。恢复后直接应用期间收到的最新 live state，不需要服务器回滚或重算。

## 全体历史/继续追踪按钮

`ToggleAllPresentations()` 的语义由本地呈现状态决定：

- 当前全部 `FollowLive`：为每个已加载对象请求最新完整 take-out 记录，分别显示历史 pose、box、照片和骨骼。
- 任一对象已在历史呈现，或全体历史请求仍在进行：取消待处理历史请求，全体恢复 `FollowLive`，应用各自最新 live pose/box，并关闭全部照片和骨骼。

全体进入历史后仍可继续单击某个物体，逐条向更老的 `history_cursor` 翻页。全体继续追踪不会停掉服务端任务；它只切换 Unity transform/evidence 的呈现。

SampleScene 的 HistoryPlacement 按钮明确调用 ShowLatestPlacements，每次为各已加载 display_object_id 重新请求最新 take-out 位置。模型的 MRTK OnPointerClicked 与 PointObject 命中都直接把 display_object_id 交给同一个 history controller；服务器返回的当前 startup 坐标是应用权威，不再因客户端缓存的旧 live coordinate epoch 拒绝。自定义食指方向只参与命中计算，MRTK debug pointing rays 关闭；ShellHandRayPointer 自带白色手掌虚线保持启用。当前 startup 有 ArMarker 时，服务器用该 startup 最新 reference 转换历史位置；没有 ArMarker 时，只允许使用同一 startup 最近 capture 的 object_aruco ↔ object_hololens_current 相对锚点，并过滤掉该锚点任务创建前的事件，返回与 live 一致的 startup-local coordinate epoch。没有同启动锚点时返回明确错误，不复用跨启动局部坐标。

## 照片与骨骼证据

历史再放置只要求有效 pose；`evidence`、spatial box、scene image 和 skeleton 均可选。证据存在时必须恰好包含 `scene_image_url` 和 `skeleton`，并以 `display_object_id + event_uid` 绑定；缺失或显示失败不会撤销已应用的历史 pose。骨骼使用 joint 点线，不加载人体网格。

`ObjectEvidenceDisplay` 在历史成功应用后自动显示两者；恢复单物体或全体 live 时隐藏对应证据。证据不参与模型 identity、scale、pose 或 revision 计算。

## Runtime spatial box

live tracking box 完全独立于模型与 display identity：recorder 在 ROS callback 内实时发布最新 /shigure/object_tracking collider，以 raw_id 作为 tracking_id；身份 runtime 不再写该文件。Unity 固定每 1 秒直接用成功响应替换本地线框集合，不做 revision、身份、稳定、平滑或延时验证；Shigure 删除的 ID 消失，新增的 ID 增加，移动的 ID 直接更新。线框不绑定模型，也不用于 PointObject、identity、pose 或其他操作。无当前启动 ArMarker 时无法安全完成坐标变换，因此成功响应为空。

## 本地容量与隐藏

`RuntimeModelManager` 默认：

```text
maxVisibleModels = 5
maxCachedModelFiles = 5
```

缓存按 `display_object_id` 形成 bundle；同一对象的 superseded 本地文件不占多个对象槽位。

`ClearLocalRetainedModels()` 是显式本地隐藏操作：恢复 presentation state、清空照片/骨骼、隐藏 runtime 模型，并中止客户端当前 status/download 自动下发；它保留 runtime records 和本地 FBX 文件，也不停止服务器的 Shigure/FoundationPose 计算。之后重新启用自动下发时，已有缓存优先恢复，缺失时才下载。
