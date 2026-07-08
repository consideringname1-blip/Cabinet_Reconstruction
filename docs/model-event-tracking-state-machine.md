# Unity 模型事件与显示状态机

更新日期：2026-07-08
状态：当前实现说明

本文描述 Unity 端围绕模型生成、下载、pending preview、历史再现和证据显示的状态规则。

## Pending Preview

pending 阶段可显示 `sam3_spatial_box`：

- 来源：服务器 pending response 的 `sam3_spatial_box`。
- 用途：生成过程中临时 3D box hint。
- 坐标：上传时 HoloLens/Unity world 口径，仅同次生成预览使用。
- 禁止：用于 completed/history/runtime 模型正式放置。

## Completed Model

completed response 必须包含：

```text
model_instance.fbx_url
model_instance.object_hololens_current
```

Unity 行为：

1. `ShuJuQingQiu.TryBuildRuntimeModelInstance` 解析 model instance。
2. 缺 `object_hololens_current` 返回 `download_ERR_missing_object_pose`。
3. `LoadModel` 下载并加载 FBX。
4. 加载完成后再次要求 `RuntimeModelManager.TryResolveWorldPose` 成功。
5. 成功后按服务器 pose 放置并注册到 `RuntimeModelManager`。
6. 失败时销毁模型，不再放到相机前方 fallback。

## ArUco 更新后的刷新

Unity 在 ArUco reference 更新后请求刷新当前加载模型：

- 请求带当前 `startup_session_id`。
- 服务端按同次 startup latest reference 转回 HoloLens current pose。
- Unity 仅更新已有模型 pose，不引入 ArUco 坐标。
- evidence overlay 不参与普通模型刷新队列。

## 历史再现显示

历史结果 item：

- 使用 `history_placement_restoration.display.polyhedron` 显示状态。
- `show_model=false` 时不叠加原模型。
- `object_hololens_current` 只作为必要的模型/状态锚点，不是缺失时的 fallback 来源。

状态到形状：

```text
OCCLUDED_REUSE_LAST -> tetrahedron
MISSING             -> cube
ORIGINAL            -> octahedron
UNKNOWN             -> dodecahedron
```

## 证据图显示

证据窗口读取：

1. `sam3d_body_mesh_urls.subject_crop_url`
2. `taken_object_detection_urls.result_rgb_url`

裁切图显示按窗口最大宽高约束等比缩放。

## 禁止状态

- completed/runtime 模型无 pose 仍下载并摆到相机前方。
- completed/history 使用 `sam3_spatial_box` 作为 runtime fallback。
- Unity 自行转换 ArUco 坐标。
- 历史状态显示重叠模型。
