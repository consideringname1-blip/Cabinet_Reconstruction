# 待处理事件：历史摆放再现与同一物体 ID

日期：2026-06-22

## 需求审查结论

这个需求可以做，但建议拆成两个目标，不要一次塞进现有拿取判断里。

1. **历史摆放再现**：给某个已建模物体，在“当前/指定时间”的 Shigurei 视角里判断它是原位、移动、消失，并输出可视化状态。
2. **同一实物身份管理**：新增一个只服务这个功能的 `physical_object_id`，把“同一个真实物体的多次拍摄/多次模型生成”归到一起，只保留最新可信模型或按评分选择模型。

## 核心判断流程

建议状态分三类：

- `SAME_PLACE`：物体还在原始位姿附近。HoloLens 直接显示原模型在原位，不显示蓝色正四面体，也不触发拿走人/图片。
- `MOVED`：同一物体存在，但位姿不同。用 Shigurei 当前帧 YOLO mask + 深度 + 旧模型跑 FoundationPose，得到临时新位姿。HoloLens 先把模型显示在当前位置，并在模型头顶放旋转蓝色正四面体。
- `MISSING`：没找到同一物体。HoloLens 只在历史原位显示模型，点击后显示拿走帧图片和人体 mesh。

这里最关键的是：**先验证 FoundationPose 在 Shigurei 视角的物体位姿是否可靠**。如果这个不稳，后面的“移动到新位置”和动画都会变成漂亮但不可信的效果。

## 同一物体验证

不要只信 Shigurei/YOLO 的 `object_id`。可以把它当强线索，但必须再验证。

建议评分由几部分组成：

- YOLO id 是否一致。
- 当前 YOLO mask 的 2D bbox、mask 中心、深度中值是否接近历史投影。
- 原拍摄 mask 图和当前 mask 图的图像 embedding 相似度，比如 masked CLIP/DINO 特征。
- 旧 3D 模型渲染到当前 Shigurei 视角后，和 YOLO mask 的轮廓 IoU、深度残差。
- FoundationPose 输出位姿后，渲染模型和实际 RGB-D 的对齐评分。

如果 YOLO id 不一致，就对当前帧所有 YOLO objects 逐个跑候选验证。都不达阈值就是 `MISSING`；有一个达阈值就是 `MOVED` 且记录 `yolo_id_changed=true`。

## 模型复用策略

不要简单地“同一物体就跳过生成”。保存两层身份：

- `physical_object_id`：真实物体身份。
- `model_instance_id`：某次拍照生成出的 3D 模型。

当新拍摄被判定为同一个物体时：

- 如果旧模型质量评分高，并且当前图像/深度能和旧模型对齐，就复用旧模型。
- 如果旧模型质量低，或 FoundationPose/渲染对齐差，就生成新模型，并把它设为该 `physical_object_id` 的最新推荐模型。
- 历史模型不立刻删，先标记 `superseded`，避免误判后没法回滚。

## HoloLens 交互建议

菜单按钮可以叫“复原到拍摄位置”或“显示历史位置”。

`MOVED` 时：

- 当前真实位置显示模型，头顶 10cm 边长蓝色旋转正四面体。
- 点击正四面体：模型用 1.2 到 1.5 秒，从当前位置沿轻微弧线飞到历史原位，使用 ease-in-out。
- 动画过程中显示一条细的半透明轨迹线，起点和终点各有短暂高亮。
- 再点正四面体或原模型上的返回按钮：飞回当前位置。
- 点击历史位置上的模型：显示拿走帧图片 + 人体 mesh。

`MISSING` 时：

- 直接在历史原位显示半透明或正常模型。
- 点击模型显示拿走证据。

`SAME_PLACE` 时：

- 只显示原位模型，不显示“拿走/移动”提示，避免误导。

## 需要先验证的内容

第一阶段只做验证，不做完整 UI：

1. 取几个已有物体模型。
2. 在 Shigurei 当前/历史 RGB-D 帧中找到 YOLO mask。
3. 用旧模型 + mask + depth 跑 FoundationPose。
4. 输出：
   - 渲染覆盖图。
   - mask IoU。
   - depth residual。
   - 连续几帧位姿抖动。
   - 和原始历史位姿的位移/旋转差。

如果这个验证不通过，后面应该先修 Shigurei 坐标、模型尺度、mask 或 FoundationPose 输入，而不是先做 HoloLens UI。

## 备份和清空数据建议

只读检查时看到的大致数据量：

- `data/upload`: 137M
- `data/output`: 2.1G
- `data/database`: 556K
- `data/shigure_history_cache`: 0
- `data/aruco`: 91M

建议备份并清空的范围：

- 备份：`data/upload`、`data/output`、`data/database`、`data/shigure_history_cache`、`data/aruco/runtime`、`data/aruco/shigure_marker_history`
- 保留不清：`data/config`、`data/RecordPhotoAndVideo`、`.dvc` 文件、`data/aruco/reference`

清空前需要确认：`data/aruco/runtime` 和 `data/aruco/shigure_marker_history` 要不要一起清？如果准备完全重建“历史摆放/身份数据库”，建议清；如果还想保留现有 ArUco/Shigurei 标定历史，就只清 `upload/output/database/shigure_history_cache`。
