# SAM3D Body 人物生成、拿取者筛选与 HoloLens 显示流程定稿

## 1. 输入来源

SAM3D Body 阶段从拿取检出状态机输出的 JSON 中读取：

```text
result_timestamp
backup_shigurei_dir
```

其中 `backup_shigurei_dir` 是拿取检出状态机在 `output` 文件夹下备份的 Shigurei 数据文件夹。

该文件夹内必须包含 `result_timestamp` 对应帧的：

```text
RGB 图像
aligned Depth 图像
CameraInfo / 相机参数
```

SAM3D Body 阶段只从 `backup_shigurei_dir` 中读取该帧数据，不再从原始 Shigurei 缓存或 rosbag 中重新查询。

---

## 2. 跳过 MoGe2，相机参数使用真实内参

SAM3D Body 原始流程为：

```text
ViTDet 人物检测
→ MoGe2 相机 FOV 估计
→ SAM3D Body mesh 生成
```

在当前处理流中，由于结果帧的 CameraInfo 可以直接获得真实相机内参，因此跳过 MoGe2。

SAM3D Body mesh 生成阶段使用真实相机参数。

真实相机参数至少包括：

```text
image_width
image_height
fx
fy
cx
cy
```

如果当前 SAM3D Body 接口只接受 FOV，而不直接接受完整内参，则 FOV 由真实内参计算得到，不调用 MoGe2 估计。

---

## 3. ViTDet 人物候选框检测

对 `result_timestamp` 对应的 RGB 图像运行 ViTDet，获得所有人物候选框。

所有 ViTDet 检出的人物候选框都进入 SAM3D Body mesh 生成阶段。

该阶段不使用拿取位置进行预筛选。
也就是说，不根据图像中心、mask 中心或物体位置提前过滤人物。

每个人物候选框分配一个稳定编号：

```text
person_<index>
```

其中 `<index>` 为 ViTDet 检测结果中的顺序编号。

---

## 4. SAM3D Body 生成人体 mesh 与 joints

对每一个 ViTDet 人物候选框，使用真实相机参数运行 SAM3D Body。

SAM3D Body 需要保留完整输出，包括：

```text
人体 mesh
人体 joints
人物候选框信息
```

后续手腕位置计算使用 SAM3D Body 输出的人体 joints。

如果某个人物候选框无法成功生成 SAM3D Body 输出，则该人物不参与后续拿取者筛选。

---

## 5. 人体 mesh 深度校正

由于 SAM3D Body 在当前情况下生成的人体距离可能不稳定，因此需要使用结果帧 aligned Depth 图对人体 mesh 进行深度校正。

深度校正在 Shigurei 相机坐标系下完成。

### 5.1 深度校正对象

不使用完整人体 mesh 的全部顶点与整张 aligned Depth 图逐点比较。

深度校正只在每个人体的局部 ROI 内进行。

对每个人体 mesh 执行：

```text
1. 使用真实相机内参，将人体 mesh 投影回结果帧 RGB 图。
2. 在该人体 ROI 内渲染人体 mesh 前表面 depth。
3. 将 rendered_mesh_front_depth 与 aligned Depth 图在同一 ROI 内比较。
```

### 5.2 比较区域

深度比较区域定义为：

```text
人体 ROI
∩ rendered_mesh_front_depth 有效区域
∩ aligned Depth 有效区域
```

其中 aligned Depth 的有效区域由深度读取层提供的 `valid_depth_mask` 决定。
SAM3D Body 阶段不额外推断无效深度语义，只使用上游提供的有效深度 mask。

### 5.3 有效重叠阈值

深度校正相关 setting 为：

```text
depth_alignment_min_overlap_ratio = 0.05
depth_alignment_min_overlap_pixels = 500
max_depth_alignment_samples = 5000
```

如果有效重叠区域满足以下任一条件，则该人体 mesh 不进行深度校正：

```text
valid_overlap_pixel_count < depth_alignment_min_overlap_pixels
valid_overlap_pixel_count / roi_pixel_count < depth_alignment_min_overlap_ratio
```

未进行深度校正的人体仍然保留，但需要在内部结果中标记：

```text
depth_corrected = false
```

该人体仍参与后续拿取者筛选。

### 5.4 depth offset 计算

如果有效重叠区域满足阈值，则计算：

```text
depth_offset = median(aligned_depth - rendered_mesh_front_depth)
```

如果有效重叠像素数量超过 `max_depth_alignment_samples`，则从有效重叠像素中采样最多 `max_depth_alignment_samples` 个像素，再计算 median。

### 5.5 平移方向

人体 mesh 沿人体中心对应的 Shigurei 相机射线方向整体平移。

人体中心像素定义为：

```text
human_center_pixel = center of ViTDet bbox
```

根据 `human_center_pixel` 和真实相机内参计算对应相机射线：

```text
human_center_camera_ray
```

然后整体平移人体 mesh：

```text
corrected_mesh = mesh + depth_offset * human_center_camera_ray
```

如果该人体不进行深度校正，则保持 SAM3D Body 原始 mesh，不执行上述平移。

---

## 6. 坐标系转换规则

深度校正阶段使用 Shigurei 相机坐标系，因为 aligned Depth 图属于 Shigurei 视角。

除深度校正外，后续所有 3D 计算统一使用 ArMarker 坐标系。

因此，每个人体在完成深度校正后，需要将以下数据从 Shigurei 相机坐标系转换到 ArMarker 坐标系：

```text
人体 mesh
人体 joints
```

物体 3D 模型坐标来源使用 ArMarker 坐标系下的物体模型数据。

后续手腕距离计算、拿取者选择、最终 FBX 输出、输出 JSON 中的位姿信息，均以 ArMarker 坐标系为准。

不保留 Shigurei 坐标系下的人体 mesh 作为最终输出。

---

## 7. 根据手腕到物体中心点距离确定拿取者

对每个成功生成 SAM3D Body 输出的人体，读取其人体 joints。

使用其中的：

```text
left_wrist_joint_3d
right_wrist_joint_3d
```

将左右手腕 joints 转换到 ArMarker 坐标系后，计算其到物体中心点的距离：

```text
left_distance = ||left_wrist_armarker - object_center_armarker||
right_distance = ||right_wrist_armarker - object_center_armarker||
person_distance = min(left_distance, right_distance)
```

其中：

```text
object_center_armarker
```

为物体 3D 模型在 ArMarker 坐标系下的中心点。

选择 `person_distance` 最小的人体作为拿取者。

如果某个人体缺少左手腕或右手腕 joint，则只使用存在的手腕 joint。
如果左右手腕 joint 均不存在，则该人体不参与拿取者筛选。

---

## 8. 生成拿取者 FBX

确定拿取者后，只对该拿取者的人体 mesh 生成用于后续显示的 FBX。

处理顺序为：

```text
SAM3D Body 输出人体 mesh
→ 在 Shigurei 相机坐标系下进行深度校正
→ 将人体 mesh 和 joints 转换到 ArMarker 坐标系
→ 对 ArMarker 坐标系下的人体 mesh 进行降面
→ 导出最终 FBX
```

最终：

```text
selected_person_fbx_path
```

指向 ArMarker 坐标系下的最终 FBX。

不保留 Shigurei 坐标系下的中间人体 mesh。

FBX 降面比例由 setting 设定：

```text
mesh_decimation_ratio = 0.125
```

即目标面数约为原始面数的 1/8。

FBX 使用黑色半透明材质。

为避免 FBX 材质在 HoloLens / Unity 侧丢失，输出 JSON 中也写入显示材质参数：

```text
material_color = black
material_alpha = 0.5
```

---

## 9. SAM3D Body 阶段输出 JSON

SAM3D Body 阶段输出一个 JSON，供 HoloLens 端点击后读取。

输出 JSON 至少包含：

```text
result_timestamp
backup_shigurei_dir
selected_person_name
selected_person_fbx_path
selected_person_pose_armarker
material_color
material_alpha
```

其中：

```text
selected_person_name = "person_<index>"
```

`<index>` 为该人体在 ViTDet 检测结果中的顺序编号。

由于最终 FBX 顶点已经导出到 ArMarker 坐标系下，因此：

```text
selected_person_pose_armarker = identity transform
```

具体写法为 4×4 单位矩阵：

```text
selected_person_pose_armarker =
[
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]
]
```

下游不得再次把 `selected_person_pose_armarker` 当作额外人体位姿重复施加到 FBX 顶点上。

HoloLens 显示时，应将 ArMarker 坐标系下的 FBX 通过 ArMarker 到 HoloLens 显示坐标系的变换链进行显示。

---

## 10. HoloLens 端显示

SAM3D Body 阶段完成后，等待 HoloLens 端点击请求。

HoloLens 端点击后，下发：

```text
result_timestamp 对应的 Shigurei RGBD 图片
选中人物的 ArMarker 坐标系 FBX
SAM3D Body 阶段输出 JSON
```

HoloLens 端显示时，需要将 ArMarker 坐标系下的人体 FBX 变换到 HoloLens 显示坐标系。

坐标系变换需要参考已有文档以及：

```text
/workspace/code/.test/sam3d_body_shigure_mesh_test.py
```

该文件用于保证人体 mesh 在 Shigurei 视角下显示正确。

从 ArMarker / Shigurei 相关坐标系变换到 HoloLens 显示坐标系时，需要参考现有 HoloLens 显示到 Shigurei 显示的对应路径，并使用对应的变换链。

---

## 11. 失败处理

如果 ViTDet 没有检测到任何人物，则 SAM3D Body 阶段返回：

```text
NO_PERSON_DETECTED
```

如果 ViTDet 检测到人物，但所有人物都无法成功生成 SAM3D Body mesh，则返回：

```text
NO_VALID_BODY_MESH
```

如果所有成功生成的人体都缺少可用手腕 joint，则返回：

```text
NO_VALID_WRIST_JOINT
```

如果成功选出拿取者并生成 FBX，则返回：

```text
SUCCESS
```
