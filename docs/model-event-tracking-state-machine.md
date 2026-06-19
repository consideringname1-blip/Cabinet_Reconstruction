# Model Event Tracking State Machine (Archived)

这份文档原本描述旧的 `code/stages/model_event_tracking/` 原型状态机。该目录和正式入口已经废弃并删除，不再作为当前 pipeline 的维护依据。

当前拿取判断链路已经拆成两个正式 stage：

```text
taken_object_detection
sam3d_body_mesh
```

当前维护入口：

- 拿取判断状态机和传给 SAM3D Body 的数据规格：`docs/拿取判断状态机.md`
- SAM3D Body 人体生成、拿取者筛选和 HoloLens 显示流程：`docs/sam3dbody相关.md`
- 服务器 pipeline 输入输出契约：`docs/server-processing-output-contract.md`
- 代码放置和新增 stage 规则：`docs/code-organization.md`

当前代码入口：

- `code/stages/taken_object_detection/run_taken_object_detection_from_json.py`
- `code/stages/sam3d_body_mesh/run_sam3d_body_mesh_from_json.py`
- `code/stages/shigure_history/run_shigure_history_recorder.py`

维护规则：不要恢复 `code/stages/model_event_tracking/` 作为正式路径。若需要调整拿取判断，请更新 `taken_object_detection` stage 和 `docs/拿取判断状态机.md`；若需要调整人体 mesh 或拿取者筛选，请更新 `sam3d_body_mesh` stage 和 `docs/sam3dbody相关.md`。
