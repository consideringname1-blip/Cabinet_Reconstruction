# SAM3D Body 运行与配置

更新日期：2026-07-08
状态：当前实现说明

本文记录 SAM3D Body stage 的运行依赖、输入输出和常用配置。算法流程见 `docs/sam3d-body-workflow.md`。

## 代码入口

```text
code/stages/sam3d_body_mesh/run_sam3d_body_mesh_from_json.py
code/stages/sam3d_body_mesh/export_selected_body_fbx.py
code/stages/sam3d_body_mesh/settings.py
```

由 `task_worker` 在 `taken_object_detection` 之后调用。

## 依赖环境

路径配置在 `code/path_config.py`：

```text
SAM3D_BODY_ROOT = <repo>/reconstruction/sam3d-body
SAM3D_BODY_STAGE_RUN = code/stages/sam3d_body_mesh/run_sam3d_body_mesh_from_json.py
SAM3D_BODY_FBX_EXPORT_SCRIPT = code/stages/sam3d_body_mesh/export_selected_body_fbx.py
```

具体 Python 环境由服务器部署决定。文档不固定某个 conda 名称；以 `path_config.py` 和实际启动脚本为准。

## 必要前置条件

任务需要已经完成：

- `aruco_sync`：生成 `object_aruco`。
- `model_bounds`：提供物体中心/包围盒。
- `taken_object_detection`：提供 taken frame 和 baseline。

缺少 marker pose、camera_info、taken result 或 depth 时，stage 应失败并写 reason，而不是继续猜测。

## 常用配置

位于 `code/stages/sam3d_body_mesh/settings.py`：

```text
SAM3D_BODY_SUBJECT_CROP_PAD_PX=24
SAM3D_BODY_SUBJECT_CROP_BODY_BBOX_PAD_PX=8
SAM3D_BODY_SUBJECT_CROP_OBJECT_CENTER_RADIUS_PX=24
SAM3D_BODY_SUBJECT_CROP_OBJECT_CENTER_MAX_RADIUS_PX=96
```

这些配置只影响裁切范围，不改变历史状态判断。

## 输出契约

任务 JSON 中写入：

```text
Sam3DBodyMesh.status
Sam3DBodyMesh.selected_person_name
Sam3DBodyMesh.selected_person_pose_aruco
Sam3DBodyMesh.selected_person_pose_hololens   # API 公开转换后
Sam3DBodyMesh.selected_person_bbox_xyxy
Sam3DBodyMesh.subject_crop_path
Sam3DBodyMesh.people_json_path
```

文件写入 `data/model/<task_timestamp>/result/`。

## Debug 建议

排查时优先看：

1. `08_sam3d_body_result.json`
2. `08_sam3d_body_people.json`
3. `subject_crop` debug 字段中的 sources。
4. `depth_alignment` 字段中的 valid pixels、offset、scale。
5. marker pose 是否存在且时间合理。

## 失败时的正确行为

- 不输出伪造人体 pose。
- 不让 Unity 用 identity/fallback pose 显示人体。
- 不影响历史物体 still/missing/occluded 的 direct compare 结果。
- 可以缺人体证据图，但 Unity 会退回 taken result RGB。
