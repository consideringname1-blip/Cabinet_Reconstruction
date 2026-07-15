# 坐标系统与公开边界

更新日期：2026-07-14
状态：当前共享坐标契约（适用于 Shigure v3）

坐标轴定义集中在 `code/coordinate_systems.py`，跨 HoloLens、ArUco 与 Shigure 的 pose/point 转换集中在 `code/spatial_transforms.py`。业务 stage 不自行添加轴翻转。

## 坐标空间

```text
windows_spatial     HoloLens/hl2ss 原始 PV pose 边界，右手，-Z forward
unity               Unity/HoloLens 公开运行时，左手，+Z forward
opencv_camera       OpenCV、ArUco、FoundationPose、Shigure camera，右手，+Z forward
canonical_rh        服务器重建内部坐标，右手，-Z forward
model_input         生成 OBJ 的源轴约定
unity_runtime_local RuntimeMesh/FBX local asset 轴约定
```

模型生成后端只影响 `model_input` 内容，不改变公开 Unity pose 契约。

## HoloLens 与 ArUco

HoloLens 上传 PV/Depth camera 的当前本地 `4x4` pose。服务器内部保存：

```text
object_hololens_original
object_aruco
object_hololens_current
```

统一转换函数包括：

```python
hololens_pose_to_aruco_pose(...)
aruco_pose_to_hololens_pose(...)
hololens_point_to_aruco(...)
aruco_points_to_hololens(...)
```

服务器根据请求方当前 `startup_session_id` 的 latest ArUco reference，把持久的 ArUco pose/points 转为 `hololens_current_local`。Unity 不读取 `object_aruco`、marker pose 或服务器内部字段。

## Live pose 与历史呈现

每个 `display_object_id` 持久保存校准后的 HoloLens capture pose 和最多 5 个 FoundationPose origin。live snapshot 优先使用最新 origin，缺失时使用最新校准 HoloLens pose，并携带 `coordinate_epoch`；Shigure 不再逐帧更新模型 tracking pose。

origin 同样持久保存在 ArUco 坐标，查询时才按当前 startup reference 转成 `hololens_current_local`。历史不是服务端 tracking mode：Unity 可以暂时显示历史 transform，但服务端始终继续接收、计算和保存 live 更新；Unity 的模型状态也继续接收最新 live pose，只是不覆盖正在呈现的历史 transform。

没有 ArUco reference 时，服务器只可发送当前 startup 最新 HoloLens capture 自带的 local pose，`coordinate_epoch=startup-local:<startup>`，且 canonical pose 为 `NULL`。复用自旧任务的 FBX 只提供模型资产，不提供旧启动坐标；没有可证明的同启动 anchor 时不能转换或返回跨启动 origin。历史响应只有在应用瞬间与对象当前非空 `LatestLive.CoordinateEpoch` 完全相等才可改变 Unity transform。

## Shigure camera 到 ArUco

Shigure RGB、depth、bbox-local mask、collider 与人体 joint projection point 都来自固定 Shigure camera。marker history 必须提供：

```text
opencv_camera_pose.rotation_matrix
opencv_camera_pose.tvec_m
```

相关转换函数：

```python
shigure_camera_points_to_aruco(...)
aruco_points_to_shigure_camera(...)
project_camera_points_to_pixels(...)
project_aruco_points_to_shigure_pixels(...)
pixel_depth_to_shigure_camera(...)
pixel_depth_to_aruco(...)
```

mask 只用于外观 identity/reference；bbox-local mask 必须贴回原 bbox 的全图位置，不能 resize 成整幅 mask。它不参与 collider box 的三维几何生成。

## Shigure collider box

上游 collider 提供 Shigure camera 坐标中的最小角 `x/y/z` 和 `width/height/depth`，单位为毫米。`spatial_box_v2.py` 的严格流程：

1. 在 camera 坐标按原始 collider 构造 8 个角点并把毫米转成米。
2. 用 camera-to-ArUco 刚体/反射变换转换 8 点。
3. 取 ArUco 三轴的 min/max，按固定角点顺序重建轴对齐包围框，使前后面平行于 ArUco marker 的 x/y 平面；存在相对旋转时该 AABB 的 extent 可大于原 camera box，不保证保留原三边。
4. 持久保存 8 个 ArUco 点；API 再转换为当前 HoloLens local 的 8 点。

此流程不读取 mask 或 depth，也没有几何降级。collider、校准、extent 或变换无效时 live API 返回带独立 revision 的 `status=no_box`，Unity 据此清除旧框；历史事件没有合法框时仍可保存/返回 `null`。`status=ready` 时 Unity 只连固定 12 条边，线径 `0.005 m`，不填充表面。

## Shigure 骨骼

Shigure person joint 的 `projection_point` 是 camera 坐标毫米值。服务器转成米，再经 camera-to-ArUco 保存；历史 API 按当前 startup reference 转成 `hololens_current_local`。每个 joint 保留 `name`、`score`、`valid`，Unity 按名称连接点线并区分骨段颜色。协议不生成或传输人体 mesh。

## CameraInfo 与事件图像

canonical cache 的 CameraInfo 结构为：

```json
{
  "k": [9个有限数值],
  "width": 1280,
  "height": 720
}
```

`k` 必须可 reshape 为 `3x3` 且焦距为正；width/height 必须与同步 RGB-D 一致。event mask 的几何范围由对应 bbox 决定，scene image、mask 和 object crop 只作为证据与 identity 输入，不替代三维 pose/box。

## Pending spatial box

`Sam3SpatialBox` 只用于 HoloLens 当前生成任务的临时 `unity_world` preview。它不是 Shigure runtime collider box，不作为 completed model pose，也不参与 Shigure 生命周期身份绑定。

## Runtime asset 与缩放

`RuntimeMesh` 保留 `model_input` 轴，pose stage 使用 `RUNTIME_LOCAL_TO_UNITY_POSE_ROTATION` 生成 Unity runtime orientation。FBX 加载后的缩放属于模型资产状态；live/history 呈现只设置 position 和 rotation，不重新应用服务器 scale。

## 校验清单

- `/generate` camera pose 是否为有限数值 `4x4`。
- Shigure CameraInfo、RGB、depth 尺寸是否一致，bbox-local mask 是否按 bbox 贴回。
- marker pose 是否使用 OpenCV camera 结构。
- live spatial box 是否只来自 primary mask-depth AABB，且保存严格 8 个 ArUco 点。
- 公开 model、live/history pose 和 live box 是否都标记或约定为 `hololens_current_local`；v3 不公开历史照片/骨骼呈现。
- Unity 是否只在 response epoch 与当前非空 LatestLive epoch 完全相等时应用历史，并按 revision 拒绝过期 live 结果。
- 业务 stage 是否复用了 `spatial_transforms.py`，没有增加局部轴翻转。
