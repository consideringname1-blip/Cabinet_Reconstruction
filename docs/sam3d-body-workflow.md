# SAM3D Body 人体 mesh 与拿取证据

更新日期：2026-07-12
状态：当前协议

实现入口：`code/stages/sam3d_body_mesh/run_sam3d_body_mesh_from_json.py`。

## 触发边界

SAM3D Body 属于 `shigure_contact_body` 辅助分支，不在 object reconstruction 的串行 stage 列表中。该分支随 HoloLens 物体上传启动，并与主链并行等待远端 Shigure contact。

只有 `ShigureContactEvidence.status=TAKEN` 才运行人体推理。实时 `obj_move`/`bring_in` 位置更新不生成人体 mesh。

## 必需输入

`ShigureContactEvidence` 提供：

```text
result_timestamp
backup_shigurei_dir
people_bounding_box.xyxy
object_bounding_box
shigure_object_id
people_id
event_id
identity_distance
```

`backup_shigurei_dir` 必须包含同一 Shigure 事件对应的：

```text
rgb.png                   # uint8 BGR
depth.png                 # uint16 millimetres
camera_info.json          # k, width, height
object_mask.png
marker_6d_pose.json       # opencv_camera_pose.rotation_matrix + tvec_m
```

RGB、depth 和 CameraInfo 尺寸必须完全一致。人体输入 box 只接受远端 contact 的 `people_bounding_box.xyxy`；缺失或无效时返回 `NO_PERSON_DETECTED`。

## 人体生成与 depth 对齐

1. 将一个 Shigure 人体 bbox 直接传给 SAM3D Body。
2. 把预测 mesh rasterize 到 Shigure 图像。
3. 在 mesh 可见区域与 depth 有效像素的重叠区计算 depth residual 中位数。
4. 沿人体 bbox 中心相机射线平移 mesh。
5. 按修正前后距离比例缩放 mesh；比例限制由 `DEPTH_SCALE_MIN/MAX` 控制。
6. 使用 `marker_6d_pose.json` 将 mesh 顶点从 Shigure OpenCV camera 坐标转换为 ArUco 坐标。

重叠像素不足时不猜测偏移或缩放，保留 SAM3D Body 原始结果并记录 `insufficient_body_mask_depth_overlap_no_translation_or_scale`。

## 证据裁切

裁切范围是两个区域的并集：

- SAM3D Body rasterized body mask；没有可用 body mask 时使用选中人体 bbox。
- 同一 Shigure event 的 `object_mask.png`。

对并集取外接矩形并添加 `SUBJECT_CROP_PAD_PX`。object mask 缺失、为空或裁切写入失败时，人体结果不能标记为 `SUCCESS`。

## 内部输出

任务辅助 JSON 写入 `SAM3DBodyMesh`：

```text
status
selected_person_name
selected_person_fbx_path
selected_person_obj_path
selected_person_pose_aruco
selected_person_bbox_xyxy
subject_crop_path
people_json_path
coordinate_space=aruco
```

结果文件：

```text
result/08_sam3d_body_result.json
result/08_sam3d_body_people.json
result/08_sam3d_body_selected_person.obj
result/08_sam3d_body_selected_person.fbx
result/08_sam3d_body_subject_crop.png
```

服务器把人体 pose 按当前 startup 的最新 ArUco reference 转为 `hololens_current_local`，再通过 canonical `body_evidence` 下发。Unity 不读取内部 ArUco 字段。

## Revision 规则

- 每个 `display_object_id` 只发布一个最新 `body_revision`。
- 新的成功人体结果替换该对象的当前人体证据引用。
- 已生成的人体 mesh、裁切图和任务历史继续保存在服务器 artifact 中。
- Unity 只接受 `body_evidence.body_revision` 与 tracking item `body_revision` 相同的结果。

## 状态

主要状态：

```text
SUCCESS
SKIPPED_NOT_TAKEN
INPUT_MISSING
NO_PERSON_DETECTED
NO_VALID_BODY_MESH
```

任何缺失的 contact person、marker pose、RGB-D、CameraInfo 或 object mask 都产生明确失败状态，不使用本地人体检测、物体中心或默认 pose 补全。
