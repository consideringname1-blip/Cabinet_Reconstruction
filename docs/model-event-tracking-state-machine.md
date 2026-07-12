# Unity 模型、缓存与追踪状态机

更新日期：2026-07-12
状态：当前协议

Unity 只处理 canonical `model_instance`、tracking item 和 `body_evidence`。每个活动物体以 `display_object_id` 为键，以 revision 防止过期结果覆盖当前状态。

## Pending preview

`/check-queue` 的 pending item 可以包含：

```text
sam3_spatial_box.status=ready
sam3_spatial_box.coordinate_space=unity_world
sam3_spatial_box.aabb_min_world
sam3_spatial_box.aabb_max_world
```

它只在当前生成任务等待期间显示临时 box，不创建正式 runtime 模型记录。

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
5. 同一 `display_object_id` 的新 revision 替换当前模型记录和当前缓存 bundle。

任何无效 pose 或坐标空间都会拒绝该模型，不在客户端猜测位置。

## HoloLens 确认上传

Unity 为后续按钮保留了 4 个入口：

```text
ShangChuanJinJingTuPian()
ShangChuanYuanJingTuPian()
UploadConfirmedPositionAndForceRebuild()
UploadConfirmedPositionAndForceRebuildLongThrow()
```

前两个按普通身份复用规则上传；后两个发送 `force_new_3d_model=1`，用于确认位置并为匹配到的 `display_object_id` 创建新模型 revision。近距使用 AHAT，远距使用 LONGTHROW。

每次 HoloLens 上传成功完成后：

- 更新该对象最新 HoloLens capture pose。
- 重新锚定实时追踪 epoch，使先前 pending/running 结果无法覆盖新位置。
- 新 revision 成为该对象活动模型；已有服务器版本仍保留。

## History/Live 按钮

`ToggleHistoryTrackingMode()` 在两个状态间切换：

- `history`：服务器 item 使用 `pose_source=hololens`；Unity 停止 status 轮询和实时位姿应用，把模型恢复到最新 HoloLens capture pose。
- `live`：恢复 status 轮询；有当前 revision 的有效 tracking pose 时使用 `pose_source=tracking`，否则使用 HoloLens pose。

请求和响应必须匹配当前 `startup_session_id`、`request_generation`、`mode_epoch` 与 `coordinate_epoch`。迟到的模式响应、旧 ArUco 坐标 epoch、旧 model revision 和旧 pose revision 均不应用。

切换和重新放置只更新 position/rotation，不修改模型已有的 Unity scale。

## 实时模型下发

`/realtime-tracking/status` item 可包含 canonical `model`。Unity 行为：

1. 先应用可接受的 pose snapshot。
2. 若当前 `display_object_id + model_revision` 未加载，则从本地缓存加载或下载。
3. 下载中的相同 identity key 不重复排队。
4. 模型加载后继续接收新的 pose revision。

服务器始终可以下发当前模型 metadata；是否需要网络下载由 Unity 本地缓存判断。

## 人体证据

item 的 `body_revision` 与 `body_evidence.body_revision` 必须一致。`ObjectEvidenceDisplay` 以 `display_object_id` 保存一个最新人体 mesh 和裁切图，并要求 `body_model` 具有与正式模型相同的 7 字段结构。

人体证据只作为对应物体的辅助显示，不参与普通物体模型的 revision、scale 或 pose 计算。

## 本地容量与隐藏

`RuntimeModelManager` 在 Unity Inspector 中默认配置：

```text
maxVisibleModels = 5
maxCachedModelFiles = 5
```

缓存按 `display_object_id` 形成 bundle；同一对象的 superseded 本地文件不占多个对象槽位。

`ClearLocalRetainedModels()` 的语义是：

- 隐藏当前 runtime 模型并清除人体显示。
- 中止客户端正在进行的 realtime status 请求、自动下发和加载队列。
- 保留 runtime 记录和本地 FBX 文件。
- 不停止服务器正在运行的 FoundationPose；其完成结果仍可写入服务器。

用户再次切换 History/Live 模式后恢复自动模型下发。若模型记录存在但场景中未加载，服务器 metadata 会触发 Unity 从本地缓存恢复；缓存不存在时才下载。
