# 历史模型身份管理与 DINOv2 复用

更新日期：2026-07-08
状态：当前实现说明

历史模型复用用于“拍摄同一物体时不再生成 3D 模型，只重新放置”。当前使用 DINOv2 embedding 做视觉身份匹配。

## 入口

stage：

```text
historical_model_match
```

实现：

```text
code/stages/hololens3d_reconstruction/run_historical_model_match_from_json.py
```

worker 会启动半常驻 `dinov2_identity` socket service。DINOv2 可复用 SAM3D 相关运行库/环境，但不代表 3D 模型生成改用 SAM3D Objects。当前模型生成默认仍是 InstantMesh。

## 配置

```text
HISTORICAL_MODEL_REUSE_ENABLE=1
FORCE_NEW_3D_MODEL=0
DINO_IDENTITY_WORKER_IDLE_TIMEOUT_SEC=300
DINO_IDENTITY_CANDIDATE_LIMIT=500
DINO_IDENTITY_MATCH_DISTANCE_THRESHOLD=0.20
DINO_IDENTITY_MATCH_SECOND_MARGIN=0.05
DINO_IDENTITY_MATCH_REQUIRE_MARGIN=1
```

请求级也可传 `force_new_3d_model`。当 force new 为 true，命中历史身份也不复用模型源。

## 复用流程

1. 当前上传图生成 DINOv2 embedding。
2. 从 identity candidate captures 中读取历史 embedding。
3. 按 display object 聚合，选距离最小候选。
4. 检查距离阈值和第二名 margin。
5. 命中后选择该 display object 最新 completed model。
6. 复制历史模型源到当前任务。
7. `task_worker` 跳过 `instantmesh`，进入 `depthpointcloud`。

复用只跳过 3D 模型生成。当前任务仍会重新计算：

```text
depthpointcloud
modelscale
object_alignment
pose
aruco_sync
runtime_mesh
model_bounds
display_identity
```

这样模型几何来自历史，放置位姿来自当前拍摄。

## 每类多个模型

每个 display object 可以有多个 completed model。复用时默认选最新 completed model，历史记录仍保留。

## 与旧分类逻辑的关系

旧的手工分类/标签逻辑不作为主路径。当前主路径以 DINOv2 identity 为准。旧字段只可作为 debug 或数据迁移参考，不能作为是否复用的权威判断。

## 失败与降级

以下情况会继续完整生成：

- 历史复用关闭。
- DINOv2 worker 启动失败。
- 当前 embedding 失败。
- 无候选。
- 最佳距离超过阈值。
- 第二名太近且要求 margin。
- 命中 display object 但没有 completed model。
- `FORCE_NEW_3D_MODEL=1` 或请求强制新模型。

## 输出字段

`HistoricalModelMatch` 常见字段：

```text
status
reuse_model
reason
display_object_id
selected_candidate_task_id
selected_model_task_id
dinov2_distance
candidate_scores
historical_reuse
```

## 注意事项

- DINOv2 判断的是“是否同一物体/类”，不是位姿。
- 复用模型后必须重新对齐和算 pose。
- 复用模型仍要经过服务器坐标链路，Unity 不做本地纠偏。
