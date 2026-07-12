# SAM3D Body 运行与配置

更新日期：2026-07-12
状态：当前协议

算法和数据契约见 [`sam3d-body-workflow.md`](sam3d-body-workflow.md)。

## 代码入口

```text
code/stages/sam3d_body_mesh/run_sam3d_body_mesh_from_json.py
code/stages/sam3d_body_mesh/export_selected_body_fbx.py
code/stages/sam3d_body_mesh/settings.py
```

`ShigureAuxiliaryBranchManager` 收到有效 `ShigureContactEvidence` 后，通过 `task_worker._run_sam3d_body_mesh()` 调用该入口。它不属于主任务 stage queue。

## 外部运行库

路径由 `code/path_config.py` 定义：

```text
SAM3D_BODY_ROOT
SAM3D_BODY_STAGE_PY
SAM3D_BODY_STAGE_RUN
SAM3D_BODY_FBX_EXPORT_SCRIPT
BLENDER_BIN
```

`SAM3D_BODY_ROOT` 必须包含模型 checkpoint 和 MHR asset；FBX 导出需要 Blender。实际 Python 解释器以 `SAM3D_BODY_STAGE_PY` 为准。

## 必需数据

执行前必须具备：

- `ShigureContactEvidence.status=TAKEN`。
- 同一 contact event 的 RGB、uint16 millimetre depth、CameraInfo、object mask。
- contact 的 `people_bounding_box.xyxy`。
- `marker_6d_pose.json` 中的 `opencv_camera_pose.rotation_matrix` 与 `tvec_m`。
- 主任务有效的 `task_id`、`task_name`、`task_timestamp`。

输入不完整时记录状态并结束辅助分支，不创建伪造人体位姿。

## 配置

`code/stages/sam3d_body_mesh/settings.py` 当前读取：

```text
SAM3D_BODY_DEVICE=cuda
SAM3D_BODY_INFERENCE_TYPE=full
SAM3D_BODY_DEPTH_OVERLAP_RATIO=0.05
SAM3D_BODY_DEPTH_OVERLAP_PIXELS=500
SAM3D_BODY_DEPTH_SAMPLE_MAX=5000
SAM3D_BODY_BBOX_DEPTH_PAD_PX=8
SAM3D_BODY_DEPTH_SCALE_MIN=0.50
SAM3D_BODY_DEPTH_SCALE_MAX=2.00
SAM3D_BODY_SUBJECT_CROP_PAD_PX=24
SAM3D_BODY_SUBJECT_CROP_BODY_BBOX_PAD_PX=8
SAM3D_BODY_FBX_DECIMATE_RATIO=0.125
SAM3D_BODY_MATERIAL_ALPHA=0.5
```

这些值分别控制推理设备、depth 对齐、裁切和 FBX 简化；不参与物体身份匹配或 realtime pose 判定。

## 排查顺序

1. 查看 auxiliary job `shigure_contact_body` 的 `status`、`result_path` 和 detail。
2. 查看 `ShigureContactEvidence.status/reason` 与 source stamp。
3. 查看 `result/08_sam3d_body_result.json`。
4. 检查 `people_bounding_box.xyxy`、RGB-D-CameraInfo 尺寸和 marker pose。
5. 查看 `result/08_sam3d_body_people.json` 中的 depth overlap、offset、scale。
6. 查看 `debug/08_sam3d_body_mesh_on_taken_rgb.png` 和 subject crop sources。

只有 `SAM3DBodyMesh.status=SUCCESS` 才会增加该 `display_object_id` 的 `body_revision` 并进入公开 `body_evidence`。
