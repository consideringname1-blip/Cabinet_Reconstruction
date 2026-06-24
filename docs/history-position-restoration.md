# 历史位置再现

日期：2026-06-23
状态：已接入主流程

## 背景结论

`history_placement_restoration` 已在历史提交 `7aed80b Add history placement restoration flow` 中作为独立流程落地：服务端读取 Shigurei RGB-D history、YOLO/segmentation history 和 ArMarker history，输出历史位置状态，并由 Unity 通过 `/history-placement-restoration/start` 触发显示。

这版文档只整理“历史位置再现”本身，暂时不讨论同一身份管理、模型版本切换或长期身份绑定。

核心修订：

- baseline 只从任务拍摄时间向后查找最多 `60s`。
- baseline 优先使用最接近拍摄时间的未遮挡可信帧，不能因为“更新”而选到物体已经移动后的帧。
- current 以当前 RGB-D 帧或指定 `target_time` 帧为状态判断时间。
- YOLO id 不再用于判断目标物体同一性。
- YOLO/segmentation 仍可提供 mask、bbox 和可配置的再现范围；使用 YOLO mask 时，必须和 YOLO 记录时间最近的 Shigurei RGB-D 帧配对。
- 当前目标候选主要用 box/depth 粗筛、原始拍照 masked crop 相似度和空间顺序判断。
- 模型不作为 current 候选选择标准，只作为历史位置、显示和 baseline 极端回退时的几何裁剪参考。
- Unity 用不同正多面体表达不同状态。

## 目标

给定一个已经建模的物体和它当初的历史摆放位置，判断当前或指定时间下它处于哪种状态。

状态定义：

- `ORIGINAL` / 原位：目标仍在历史位置附近。
- `MOVED` / 已移动：找到可信目标候选，但位置明显不同。
- `MISSING` / 消失：原位置变空，允许范围内也找不到可信目标候选。
- `OCCLUDED_REUSE_LAST` / 被遮挡，复用上次可信状态：原位置被更近物体挡住，无法确认目标是否仍在后面。
- `UNKNOWN` / 不确定：图像、depth 或候选证据冲突，不能安全判断。
- `SKIPPED` / 跳过：配置禁用、输入不足，或流程被上层跳过。

`ORIGINAL`、`MOVED`、`MISSING`、`OCCLUDED_REUSE_LAST` 是主要假设状态；`UNKNOWN` 和 `SKIPPED` 是保守退出状态。

## 非目标

- 不用 YOLO id 保证目标同一性。
- 不把 `UNKNOWN` 当作 `MISSING`。
- 不在证据冲突时自动更新模型或身份绑定。
- 不使用 3D 模型作为 current 候选选择或身份判断标准。
- 不实现两图匹配大模型兜底；这部分可以后续扩展，但不是当前文档的最小方案。

## 输入与输出

输入：

- task JSON，包含原始任务时间、原始拍照图、模型原始 ArUco 位置或 bounds。
- Shigurei RGB-D history chunk。
- `/tracking/active_objects` YOLO/segmentation JSON history。
- Shigurei ArMarker history，用于 Shigurei camera 和 ArUco 坐标转换。
- 可选 `target_time`，否则使用最新 Shigurei sample。
- 服务端配置的再现范围 YOLO id 列表，允许多个，默认为空。

再现范围配置：

- 配置项建议命名为 `tracking_region_yolo_ids` 或同义名字。
- 这些 YOLO id 指的是“限制范围用的承载物或区域物体”，例如桌子，不是目标物体 id。
- 配置为空表示不限制范围，默认即为空。
- 配置不为空时，用这些 id 对应的 mask/bbox 建立 `tracking_search_region`。
- 这些 id 指向的承载平面默认应基本静止；如果承载物或平面本身明显移动，本次范围约束无效，应输出 `UNKNOWN` 或降级为无范围搜索，而不能静默沿用旧平面。
- 范围内候选的 3D 位置应高于这些 id 所代表的平面；如果只用 depth 近似，则候选 mask 的 depth 应与该平面有合理关系，不能落到桌面后方或明显不在承载平面附近。

输出：

- `status`：本次历史位置状态。
- `baseline`：旧状态参考帧、原始 masked crop、mask、depth、baseline 可见性。
- `current`：当前 RGB-D 帧、最近可用 YOLO/segmentation payload、与 YOLO 记录时间最近的 Shigurei 配对帧及时间差。
- `tracking_search_region`：本次搜索范围；配置为空时记录为 unrestricted。
- `candidates`：候选物体及分项评分。
- `classification`：状态判断的原因、阈值和冲突点。
- `display`：Unity 可消费的模型显示、多面体形状和动画信息。
- `summary.json` 和可选 debug 可视化。

## 流程

### 1. 建立 baseline

baseline 从任务拍摄时间 `capture_seconds` 之后建立，只向后查找最多 `60s`，不再向拍摄前回看。

流程：

1. 从 `capture_seconds` 开始，按时间顺序读取后续 Shigurei history。
2. 按需逐个解封/解码视频帧到内存，在内存中的图片组里查找可用 RGB-D 帧，而不是只取拍摄后的第一帧。
3. 使用任务记录的历史位置，把目标中心和 3D box 投影到 Shigurei 图像。
4. 在投影 box 附近查找 YOLO/segmentation 候选；使用 YOLO mask 时，必须取与该 YOLO 记录时间最近的 Shigurei RGB-D 帧来读取图像和 depth。
5. 候选按 mask 中心到投影中心的距离从近到远检查。
6. 候选先做粗筛：mask 中心在投影范围附近、bbox/mask 尺度和投影 3D box 大小大致匹配、depth 在目标附近。
7. 用 YOLO mask、投影 box 和对应 Shigurei depth 判断遮挡：如果投影 box 内目标可见面被更近 depth 大面积覆盖，认为该帧被遮挡。
8. 优先选取最接近 `capture_seconds` 的未遮挡可信帧作为 baseline。
9. 如果更晚的帧被选中，必须确认候选仍在原始投影 box 附近，且尺度、depth 和原始 masked crop 没有显示已移动；否则不能作为 baseline。

如果连续多帧都被遮挡，但投影框内仍存在部分被遮挡的 YOLO mask 或有效 depth，可退化到局部初始化：只使用模型可见表面附近、Shigurei depth 可信的那部分像素建立 baseline。若有效像素过少，或完全没有可信 depth，则 baseline 失败。

这类模型辅助只允许用于 baseline 的几何可见区域裁剪，是实在没有完整可见帧时的回退；它不参与 current 候选排序，也不作为目标同一性判断标准。

baseline 不再要求同一个 YOLO `object_id` 连续稳定出现。YOLO id 不保存为目标同一性线索。

baseline 保存：

- `reference_time`
- `reference_rgb_path` 或内存帧对应备份路径
- `reference_mask`
- `reference_depth`
- `reference_masked_crop`
- `projected_box`
- `reference_signature`：mask 面积、bbox 尺度、median depth、masked crop 外观特征。
- `baseline_visibility`：是否未遮挡、是否局部初始化、有效像素比例、depth 判断依据。

baseline 建立失败时输出 `UNKNOWN / baseline_failed`。

### 2. 读取 current

current 直接使用请求指定的当前帧或最新帧，不再使用最近多帧投票。

流程：

1. 如果请求包含 `target_time`，取与 `target_time` 对齐的 Shigurei RGB-D sample；否则取最新 Shigurei RGB-D sample。这个 sample 是状态判断时间。
2. YOLO 频率低，因此 YOLO payload 通常是旧的；读取距离状态判断时间最近的一次 YOLO/segmentation payload。
3. 对于 YOLO masks 产生的候选，不直接裁状态判断 RGB-D 帧；应取与该 YOLO payload 记录时间最近的 Shigurei RGB-D 帧作为 `yolo_paired_rgbd_sample`，用它裁 masked crop 和读取候选 depth。
4. 在 `current` 中记录 `target_rgbd_time`、`yolo_payload_time`、`yolo_paired_rgbd_time`、`yolo_delta_to_target`、`yolo_pairing_delta`。
5. 如果 `yolo_delta_to_target` 超过配置阈值，YOLO masks 只能作为弱候选来源；如果没有其他可用候选，应优先返回 `UNKNOWN` 或只做原位置 depth 判断。
6. 保存 current RGB-D、yolo paired RGB-D、depth、YOLO/segmentation 备份或引用。
7. 如果状态判断帧的原位置 depth 明显被更近物体挡住，直接进入 `OCCLUDED_REUSE_LAST` 判断；不为了遮挡额外拉长时间窗口。

这里的核心变化是：状态时间由 current RGB-D 决定；YOLO mask 使用它自己记录时间附近的 Shigurei 帧配对，不能假装和 current RGB-D 严格同步。

### 3. 设定变动追踪范围

`tracking_search_region` 用于 current 阶段的移动候选搜索。它限制“目标可能移动到哪里”，不负责判断目标同一性。

配置不为空时：

1. 在最近 YOLO payload 中找到配置 id 对应的 objects。
2. 使用与该 YOLO payload 时间最近的 Shigurei RGB-D 帧，读取这些 objects 的 mask/bbox/depth。
3. 合并这些 objects 的 bbox/mask，得到搜索范围。
4. 用这些 objects 的 depth 或 3D 反投影估计承载平面。
5. 检查承载平面是否基本静止：与 baseline 或配置参考平面相比，中心、法向、median depth 的变化必须低于配置阈值。
6. 只保留位于该范围内、并且位置高于承载平面的候选。
7. 如果配置 id 缺失、YOLO 过旧、或承载平面明显移动，本次范围约束无效；按配置选择输出 `UNKNOWN`，或降级为 `unrestricted` 并在 JSON 中记录原因。

配置为空时：

1. 不限制全局搜索范围。
2. 候选仍然必须通过 box/depth/masked crop 相似度的最低阈值。
3. 候选排序从历史原位置投影中心开始，由近到远检查，优先选择最近的可信候选。
4. 距离只能用于排序或平局处理，不能替代外观、尺度和 depth 阈值。
5. 如果出现同类物体或相似包装导致误认，这是当前方案接受的剩余风险；文档不再把它作为阻塞条件。

`tracking_search_region` 建议写入：

- `source`: `configured_yolo_ids | unrestricted | debug`
- `configured_yolo_ids`
- `bbox_xyxy`
- `support_plane` 或 `support_depth_summary`
- `support_plane_static_check`
- `is_unrestricted`
- `reason`

### 4. 当前物体查找

当前物体查找完全抛弃目标 YOLO id。YOLO/segmentation 只提供候选 masks，不提供同一性结论。

流程：

1. 从最近 YOLO/segmentation payload 中收集候选 masks。
2. 每个候选 mask 都使用与 YOLO payload 记录时间最近的 Shigurei RGB-D 帧裁图和读取 depth。
3. 如果配置了 `tracking_search_region`，先过滤到范围内，并检查候选是否高于承载平面。
4. 如果未配置范围，则保留全部候选，并按候选中心到历史原位置投影中心的距离从近到远排序。
5. 用 bbox/mask 尺度和 depth 做粗筛。
6. 使用任务当初上传的原始拍照图，结合 baseline mask 裁出 `reference_masked_crop`。
7. 对每个候选裁出当前 masked crop，与 `reference_masked_crop` 做图像相似度比较。
8. 选择第一个通过最低阈值、且无明显 depth/尺寸冲突的候选。
9. 如果多个候选分数接近，按距离原位置从近到远打破平局；仍冲突则输出 `UNKNOWN`。

候选来源：

- 当前最近可用 YOLO/segmentation masks。
- 原位置 RGB-D/depth 变化区域。
- 后续可扩展 SAM/SAM2 ROI proposal，但当前最小方案不依赖它。

候选评分建议输出分项：

- `box_size_score`: bbox/mask 尺度是否接近。
- `masked_crop_score`: baseline masked crop 与候选 masked crop 的外观相似度。
- `depth_score`: 候选 median depth 和 depth 分布是否合理。
- `spatial_order`: 候选中心到历史原位置投影中心的排序。
- `support_plane_score`: 配置了范围时，候选是否高于承载平面。
- `ambiguity_margin`: 第一候选和第二候选的差距。
- `hard_thresholds_passed`: 外观、尺度、depth 是否都过最低门槛。

### 5. `ORIGINAL` / `MOVED` 判断

选到可信候选后，再判断它是在原位还是已移动。

可用检查：

- 2D 中心距离：小于 `max(45px, bbox_diag * 0.18)`。
- mask median depth 差：小于 `0.22m`。
- ArUco 3D 位置差：小于 `0.14m`。

全部关键检查通过：输出 `ORIGINAL`。

任一关键检查显示位置明显不同：输出 `MOVED`。

`MOVED` 时，使用当前候选 mask 中心和 depth 反投影出新的 ArUco pose，并在 Unity display 中把提示多面体放到新位置上方。此时不显示历史原位模型。

### 6. `MISSING` / `OCCLUDED_REUSE_LAST` 判断

如果 current 阶段找不到可信候选，进入原位置 depth 判断。

判断顺序：

1. baseline mask 或投影 box 内大面积 depth 变近：输出 `OCCLUDED_REUSE_LAST`。
2. baseline mask 或投影 box 内大面积 depth 变远，且候选搜索没有找到可信对象：输出 `MISSING`。
3. depth 变化弱、depth 缺失或变近/变远同时存在：输出 `UNKNOWN`。

默认阈值可继续作为初值：

- 遮挡：`current_depth - baseline_depth <= -0.10m` 的有效像素比例 >= `25%`。
- 消失：`current_depth - baseline_depth >= +0.10m` 的有效像素比例 >= `40%`。
- 二者同时满足：`UNKNOWN / depth_conflict_closer_and_deeper`。
- 二者都不满足：`UNKNOWN / target_missing_without_depth_change`。

有效像素比例只在 baseline mask 或投影 box 内计算，并且只统计 baseline 和 current 都有可信 depth 的像素。

这里不再用最近多帧 YOLO 去证明 missing。current RGB-D 帧被遮挡就是遮挡；current RGB-D 帧原位置变空且候选搜索找不到目标，才是 missing。

## Unity 展示约定

历史位置再现显示应由独立 Unity 脚本控制，脚本放在 `/workspace_whs/H2AI/Assets/Scripts`，并统一挂载到 SampleScene 的 `Scripts` 对象下。是否显示由 Unity 侧开关控制。

状态显示：

- `ORIGINAL`：不显示历史原位模型，在当前原位物体上方显示旋转正四面体。
- `MOVED`：不显示历史原位模型，在新的当前物体位置上方显示旋转正六面体，也就是 cube。
- `MISSING`：显示历史原位模型，在模型上方显示旋转正八面体。
- `OCCLUDED_REUSE_LAST`：显示历史原位模型，在模型上方显示旋转正十二面体。
- `UNKNOWN`：显示历史原位模型，在模型上方显示旋转正二十面体。
- `SKIPPED`：不显示历史位置再现提示。

Unity display payload 需要支持：

- `display.show_model`
- `display.model_pose`
- `display.polyhedron.enabled`
- `display.polyhedron.shape`: `tetrahedron | cube | octahedron | dodecahedron | icosahedron`
- `display.polyhedron.attach_to`: `current_object | original_model | none`
- `display.polyhedron.height_offset_m`
- `display.polyhedron.rotation_speed_deg_s`
- `display.reason`

## 与现有实现不同的部分

以下是本设计和当前 `7aed80b` 落地实现的主要差异，后续修改代码时应逐项对齐。

- baseline 搜索窗口不同：现有实现向拍摄前看 `120s`、向拍摄后看 `60s`；新方案只从拍摄时间向后查找最多 `60s`。
- baseline 取帧策略不同：现有实现偏向拍摄后第一帧和稳定 YOLO 窗口；新方案要求优先取最接近拍摄时间的未遮挡可信帧，必要时允许局部初始化。
- baseline YOLO 稳定要求不同：现有实现要求同一 YOLO `object_id` 至少 `3` 个唯一 payload 稳定；新方案不要求目标 YOLO id 稳定，也不保存目标 YOLO id 作为同一性线索。
- YOLO/RGB-D 配对不同：现有实现倾向直接把最近 YOLO payload 用到 current；新方案要求使用 YOLO 记录时间最近的 Shigurei RGB-D 帧来裁 mask 和读候选 depth，并记录配对时间差。
- current 时间策略不同：现有实现使用 `12s` lookback、`0s` forward，并保留最近 `5` 个唯一 YOLO events；新方案直接使用 current RGB-D 帧作为状态时间，YOLO 只提供候选 masks。
- 搜索范围不同：现有实现没有服务器配置的范围 YOLO id；新方案允许用配置 id 指定桌面/承载平面范围，默认空表示不限制。
- 承载平面约束不同：新方案要求配置范围的承载平面基本静止，明显移动时不能静默沿用。
- 当前物体选择标准不同：现有实现优先按旧 YOLO id 找，再做 signature fallback；新方案完全抛弃目标 YOLO id，只用 mask、box、depth、原始 masked crop 相似度和空间顺序。
- missing 判断证据不同：现有实现会要求最近唯一 YOLO 数量达到阈值后再判断 missing；新方案以 current depth 和候选搜索结果为准。
- 模型使用边界不同：现有实现中模型可参与 display 和部分几何判断；新方案只允许模型作为历史位置、显示和 baseline 局部初始化的几何参考，不作为 current 候选选择标准。
- Unity 多面体不同：现有 Unity display 只特殊支持 `octahedron`，其他 shape 退化为 cube；新方案需要 `tetrahedron`、`cube`、`octahedron`、`dodecahedron`、`icosahedron` 五种语义形状。
- 状态显示不同：现有实现 `ORIGINAL` 通常没有 polyhedron，`MOVED` 用 cube，`UNKNOWN` 用 octahedron，`MISSING` 显示原模型但没有专门多面体；新方案要求 `ORIGINAL` 正四面体、`MOVED` 正六面体、`MISSING` 正八面体、`OCCLUDED_REUSE_LAST` 正十二面体、`UNKNOWN` 正二十面体。
- Unity 挂载约定不同：新方案要求新增脚本统一挂载在 SampleScene 的 `Scripts` 对象下。

## 审查修改内容

本轮修改后的审查结论：

- baseline 已改为优先取最接近拍摄时间的未遮挡可信帧，避免“最新帧”误吸收已经移动后的状态。
- 使用 YOLO mask 时必须与 YOLO 记录时间最近的 Shigurei RGB-D 帧配对，解决旧 YOLO mask 裁当前图造成错位的问题。
- current 状态时间仍由 current RGB-D 决定；YOLO 配对帧只用于候选 mask crop 和候选 depth。
- 目标 YOLO id 已从同一性判断中删除。YOLO id 只可用于配置再现范围，例如桌面或承载区域。
- 配置范围的承载平面必须基本静止；明显移动时不能继续当作范围锚点。
- 配置范围为空时允许不限制全局搜索，但候选必须通过外观、尺度、depth 的最低阈值，距离只用于排序和平局处理。
- baseline 可在连续遮挡时退化为局部初始化，但模型只用于 baseline 几何裁剪，且必须记录有效像素比例；有效像素太少时不能硬判。
- 原始拍照图已经存在，应作为 baseline masked crop 的来源；仍需要保存或缓存对应 mask/depth crop，方便复现和 debug。
- `UNKNOWN` 已有独立显示语义：原位模型上方正二十面体，不能复用 `MISSING` 的正八面体。
- Unity 现有 `HistoryPlacementRestorationDisplay.cs` 需要新增正四面体、正十二面体、正二十面体 mesh；否则 payload 写了 shape 也会显示成 cube。
- 服务端 `display.mode` 和 `polyhedron.shape` 枚举需要扩展，至少区分 `MISSING`、`OCCLUDED_REUSE_LAST` 和 `UNKNOWN`。
- recent unique YOLO 备份不是本设计的核心，不再作为验收点。

## 验收标准

最小可验收版本：

- baseline 只向后查找最多 `60s`，并记录选中帧的未遮挡证据或局部初始化证据。
- baseline 优先取最接近拍摄时间的未遮挡可信帧；若选更晚帧，必须证明仍在原始位置。
- current 直接使用当前 RGB-D 帧或 `target_time` 对齐帧作为状态时间。
- YOLO payload 与其记录时间最近的 Shigurei RGB-D 帧配对，用于候选 mask crop 和候选 depth。
- 目标 YOLO id 不参与同一性判断。
- 支持服务端配置范围 YOLO id；配置为空时不限制范围。
- 配置范围的承载平面必须基本静止。
- 当前候选使用 YOLO/segmentation masks、box/depth 粗筛、原始 masked crop 相似度和空间顺序判断。
- 候选必须通过外观、尺度、depth 最低阈值；距离不能单独决定同一目标。
- 能输出 `ORIGINAL`、`MOVED`、`MISSING`、`OCCLUDED_REUSE_LAST`、`UNKNOWN`。
- Unity display 支持正四面体、正六面体、正八面体、正十二面体、正二十面体。
- `UNKNOWN` 不触发 missing 展示，不复用 `MISSING` 的正八面体。
