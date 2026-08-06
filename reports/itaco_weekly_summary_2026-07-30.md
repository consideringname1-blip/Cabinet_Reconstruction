# iTACO / VGGT 可动柜体重建周报

时间范围：2026-07-27 至 2026-07-30  
数据：HoloLens `2026-07-27-175228`、HoloLens `2026-07-30-002840`、iPhone RGB-D `close-to-open`

## 1. 本周结论

本周完成了从基础模型验证、误差来源隔离，到可动抽屉轴与状态恢复、内部结构重建和可动 Mesh 导出的完整实验闭环。

最终较可靠的技术路线是：

```text
HoloLens RGB-D + 设备相机位姿
→ RGB/depth/hand 有效区域约束
→ AutoSeg-SAM2 软 proposal
→ 修复 moving map（取消逐帧 min-max，加入有效支撑约束）
→ 固定相机位姿估计平移轴
→ 重新计算单调 q_t
→ static/drawer 双 volume 融合
→ NKSR 分组件重建
→ GLB/URDF 可动模型
```

最终模型能够表现抽屉约 0.300 m 的平移运动，并恢复一部分打开后显露的内部结构；结构验证已通过，可用于研究演示和运动可视化，但几何仍受 HoloLens 深度噪声影响，不是生产级干净资产。

需要强调：最终结果属于在官方 baseline 之外实现的扩展版本。官方 iTACO 核心文件未被修改，早期官方兼容结果和失败结果均单独保留。

## 2. 本周完成的工作

### 2.1 VGGT 本地部署与数据对比

- 完成本地 VGGT 环境及权重部署。
- 使用 HoloLens 数据对比真实深度、相机坐标、内参与 VGGT 输出。
- 8 帧推理耗时 1.81 s，约 0.226 s/帧，不含模型加载。
- VGGT 深度存在明显尺度差异，全局中值尺度因子为 6.104。
- 尺度对齐后，深度 MAE 为 1.644 m，AbsRel 为 0.493，说明当前近距离、窄视场 RGB-D 序列上不能直接将 VGGT 输出视为传感器深度或可靠 GT。
- VGGT 平均焦距比标定值高约 20.6%。

结果目录：[VGGT 对比结果](/workspace_whz/data/output/vggt_upload_compare/2026-07-27-175228_vggt_metrics)

代表图：[VGGT 与真实深度对比](/workspace_whz/data/output/vggt_upload_compare/2026-07-27-175228_vggt_metrics/maps/frame_000000_depth_compare.png)

### 2.2 官方 iTACO、MonST3R 与 AutoSeg-SAM2 流程复现

- 复现了 MonST3R + AutoSeg-SAM2 的官方兼容流程，并与 HoloLens 深度/位姿输入进行了比较。
- 确认 AutoSeg-SAM2 在该流程中不依赖文本 prompt；proposal 仅表示候选区域，不能直接当作 moving/static 标签。
- 早期 SAM3 结果存在明显乱分割，相关生成结果已按要求删除。
- MonST3R 在该录制上的相机和深度误差较大，容易引起点云错层、几何崩坏和运动推断污染，因此后续主线不再用其替代 HoloLens 位姿。
- 验证了视频倒序读取、初始开闭状态、动作片段和建模片段的作用，避免把“相机移动”误判成“抽屉移动”。

结果目录：

- [官方兼容 MonST3R / AutoSeg-SAM2](/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_official_monst3r_autoseg)
- [GT 与 MonST3R 对比](/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_official_monst3r_autoseg/comparison_gt_vs_monst3r)
- [AutoSeg-SAM2 可视化](/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_official_monst3r_autoseg/preview_autoseg_sam2)
- [96–127 帧动作段可视化](/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_gt_motion_frames_096_127/visualization)

### 2.3 iPhone RGB-D 与位姿误差诊断

在 iPhone `close-to-open` 数据上尝试了三状态 mask、双 volume 内部重建和 ARKit 位姿初始化。该实验没有产生可用 Mesh，但明确暴露了三个问题：

1. 三状态 mask 生长会把非抽屉区域错误吸收；
2. ARKit 位姿误差会使多帧表面分层；
3. 分层点云进入 TSDF/NKSR 后表现为毛边、尖刺和缺失背面。

失败结果保留在：[iPhone articulation-aware trial](/workspace_whz/data/output/itaco_iphone_rgbd/close_to_open_articulation_aware_trial_001)

随后使用同一份 HoloLens 点云、RGB、mask 和 TSDF 参数，仅对相机位姿加入 ARKit 风格误差，完成受控实验：

| 指标 | HoloLens 位姿 | ARKit-like 扰动位姿 |
|---|---:|---:|
| 平面厚度 p05–p95 | 1.13 cm | 11.13 cm |
| 深度层峰值数 | 1 | 4 |
| 法线偏差中位数 | 10.08° | 46.91° |
| 法线偏差 >30° | 3.09% | 71.87% |
| TSDF 三角面数 | 100,913 | 436,973 |

结论：仅相机位姿误差就能让平面厚度扩大约 10 倍并形成多层和刺状表面。TSDF 会把错误观测融合成几何，但不是误差的最初来源。

代表图：[位姿误差导致的 Mesh 差异](/workspace_whz/data/output/itaco_hololens_pose_error_diagnostic/2026-07-30-002840_arkit_like_tsdf_003_no_floor/render/mesh_comparison.png)

### 2.4 阶段 0：代码审查

完成 iTACO、AutoSeg-SAM2、MonST3R、PromptDA 等相关代码的数据流与坐标系审查，主要发现：

- AutoSeg 数组层序被隐式当作 UID，但没有显式保存 proposal 身份；
- 每个 proposal 共用一个 moving 标量，逐帧 min-max 使数值不具备跨帧概率意义；
- 原流程没有正式的 `unknown` 类；
- 相机可在联合优化中自由变化，缺少设备位姿先验；
- 第 0 帧被同时用于分割、运动零状态和几何初始化；
- coarse 相机配准存在手部排除 mask 被后一行覆盖的问题；
- 采样或倒序后缺少原始帧 ID、时间戳和位姿索引的完整回溯；
- 若干历史内部重建脚本包含固定帧、固定 UID 或空间特例。

完整报告：[阶段 0 代码审查](/workspace_whz/data/output/itaco_stage0_code_audit_20260729/STAGE0_CODE_AUDIT.md)

### 2.5 阶段 1：可回溯轨迹与三分类

在不修改官方 baseline 的独立 package 中实现：

```text
不可变 frame manifest
→ 自动参考帧选择
→ 统一 validity mask
→ 显式 AutoSeg proposal
→ 短 3D tracks / clusters
→ static / moving / unknown
```

阶段 1 固定相机、关节类型、关节轴和已知 `q_t`，没有运行相机优化、关节估计、TSDF、NKSR 或 Mesh。

主要验收结果：

- 参考帧自动选择为原始帧 180，而非默认第 0 帧；
- 第 0 帧保留原始编号、时间戳和位姿，手部 observation 为 0；
- 同一 proposal 可以拆成不同标签；
- 多个 proposal 可以因共享运动模型合并；
- 证据不足的轨迹进入 unknown。

初始轨迹支撑仍不足：637 条 raw tracks 中仅 176 条通过过滤，其中 static 2、moving 4、unknown 170。

结果目录：[阶段 1 trial 006](/workspace_whz/data/output/itaco_track_motion_stage1/2026-07-30-002840_trial_006)

### 2.6 阶段 1.5：轨迹覆盖恢复与可观测性验证

阶段 1.5 增加了：

- hand mask 时序异常检测与双向修正；
- 跨短时间缺口的轨迹重关联；
- 与真实传感器 confidence 分开的 derived depth quality；
- 少量人工评估标签；
- static/moving 空间覆盖、信息矩阵和 bootstrap 稳定性检查。

| 指标 | 阶段 1 | 阶段 1.5 |
|---|---:|---:|
| filtered tracks | 176 | 203 |
| 平均轨迹长度 | 6.09 | 12.44 |
| 平均时间跨度 | 6.09 帧 | 12.52 帧 |
| static tracks | 2 | 14 |
| moving tracks | 4 | 3 |
| unknown tracks | 170 | 186 |
| static 图像覆盖 | 6.25% | 31.25% |
| moving 图像覆盖 | 12.50% | 18.75% |
| 接受的跨缺口连接 | 0 | 13 |

分类阈值没有放宽，覆盖提升来自观测恢复。当前仍不具备直接进入联合关节优化的可观测性：

- static bootstrap 稳定率仅 7.14%；
- moving 仅有 3 条有效轨迹，且空间支撑集中；
- 人工评估准确率为 5/21（23.81%）；
- 13 个 gap connection 尚未逐个完成人工错误连接审计。

代表图：

- [hand mask 修正前后](/workspace_whz/data/output/itaco_track_motion_stage1_5/2026-07-30-002840_trial_005/visualization/summary/01_hand_mask_recovery_worst9.png)
- [阶段 1 与 1.5 轨迹数量和长度](/workspace_whz/data/output/itaco_track_motion_stage1_5/2026-07-30-002840_trial_005/visualization/summary/02_stage1_vs_stage1_5_tracks.png)
- [阶段 1 与 1.5 三维轨迹](/workspace_whz/data/output/itaco_track_motion_stage1_5/2026-07-30-002840_trial_005/visualization/summary/05_stage1_vs_stage1_5_3d_tracks.png)
- [同一 proposal 被拆分的案例](/workspace_whz/data/output/itaco_track_motion_stage1_5/2026-07-30-002840_trial_005/visualization/proposal_split_cases/proposal_000008.png)
- [多个 proposal 合并为同一 moving set](/workspace_whz/data/output/itaco_track_motion_stage1_5/2026-07-30-002840_trial_005/visualization/proposal_merge_cases/moving_set_multiple_proposals.png)

### 2.7 moving map 修复

针对地板、RGB 有效范围外深度和可见轮廓被错误赋予高 moving 分数的问题，实现了独立修复模块：

- AutoSeg proposal 只作为软区域；
- 去除逐帧 min-max；
- moving 分数必须有有效 RGB/depth、几何和时序支撑；
- 无效和证据不足区域分别保留为 invalid/unknown；
- 不使用固定 UID、颜色、地板高度、对象位置或固定帧规则。

| 指标 | 官方计算 | 修复后 gate-no-minmax |
|---|---:|---:|
| 高 moving 分数面积比例 | 58.52% | 3.19% |
| 高分区域落在有效支撑外 | 80.71% | 0% |
| 支撑外高分像素 | 1,886,763 | 0 |
| 小规模人工检查 | 8/10 | 10/10 |

确定性复跑的输出 SHA256 完全一致。moving map 本身得到修复，但首次保留联合相机优化时，相机姿态和 `q_t` 仍严重漂移，因此最终路线固定了 HoloLens 相机位姿。

代表图：[官方 / gate-only / gate-no-minmax 对比](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_trial_001/visualization/frame_000000_comparison.jpg)

### 2.8 平移运动轴与逐帧状态恢复

在固定 HoloLens 位姿和修复后的 moving labels 上，比较了 iTACO 单向 Chamfer 轴与 moving centroid 轴：

- iTACO Chamfer 轴：`[0.146, 0.225, 0.963]`
- moving centroid 轴：`[0.538, 0.089, 0.838]`
- 两轴夹角：25.06°
- centroid 轨迹线性解释率：99.50%
- centroid 轴与早期独立 GT-depth centroid 轴夹角：0.60°
- `q_t` 与 centroid 投影相关系数：0.997

因此选用 moving centroid 轴作为最终平移轴。

重新计算 `q_t` 后：

- 原始运动范围：0.3013 m；
- 单调化运动范围：0.3002 m；
- 负向跳步：7 次降为 0 次；
- 平均单调修正量：0.00018 m；
- `q=0` 自动落在最小状态平台的前 4 个处理帧，而非硬编码原始第 0 帧。

代表图：

- [两种运动轴投影对比](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/visualization/axis_projection_itaco_vs_moving_centroid.jpg)
- [关键帧运动轴](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/visualization/axis_projection_keyframes.jpg)
- [重新计算的关节状态曲线](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/visualization/joint_state_curve.jpg)
- [带点云的三维运动轴 PLY](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/visualization/axis_with_surface_moving_and_centroid_axes.ply)

### 2.9 带内部结构的可动 Mesh

最终扩展流程采用：

- 关闭状态帧：5–165；
- 交互帧：177–213；
- 打开状态帧：220–360；
- 所有阶段均按正序处理；
- 固定 HoloLens `T_world_camera`；
- 关节类型为 prismatic；
- 关闭状态抽屉前板从一开始进入 drawer volume；
- 打开后显露的静态内部结构进入 static volume；
- moving proposal 仅作为几何 seed，不作为部件所有权标签。

为避免二维 mask 膨胀泄漏，最终使用跨帧 canonical 3D 一致性恢复抽屉模板：3,593 个候选体素中接受 1,506 个，体素大小 2 cm，至少跨 4 帧且覆盖不少于 35% 的运动范围。

最终点云与 Mesh：

| 内容 | 数量 |
|---|---:|
| static 点 | 57,155 |
| drawer canonical 点 | 63,465 |
| open 状态合并点 | 102,537 |
| static NKSR 三角面 | 1,024,403 |
| drawer NKSR 三角面 | 4,108,011 |
| 预览 static 三角面 | 400,000 |
| 预览 drawer 三角面 | 200,000 |
| 关节范围 | 0–0.30023 m |

输出支持双面材质，包含 GLB 和 URDF，结构验证 `all_valid=true`。

代表图与模型：

- [双 volume 正交视图](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/geometry_interior_v1/dual_volume_fusion/dual_volume_orthographic.png)
- [三状态 mask 接触表](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/geometry_interior_v1/dual_volume_fusion/tri_state_masks_contact_sheet.jpg)
- [关闭状态渲染](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/geometry_interior_v1/articulated_model/render_closed.png)
- [打开状态渲染](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/geometry_interior_v1/articulated_model/render_open.png)
- [双面可动 GLB](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/geometry_interior_v1/articulated_model/hololens_cabinet_with_interior_double_sided.glb)
- [双面可动 URDF](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/geometry_interior_v1/articulated_model/hololens_cabinet_with_interior_double_sided.urdf)
- [完整复现实验清单](/workspace_whz/data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/geometry_interior_v1/articulated_model/reproduction_manifest.json)

## 3. 失败实验与保留价值

本周没有隐藏失败结果，以下版本均保留用于报告中的消融分析：

- SAM3：提示与分割不稳定，存在乱分割；
- MonST3R：本数据上位姿/深度误差过大；
- iPhone ARKit：位姿误差造成明显点云分层；
- 早期 AutoSeg/moving map：proposal 共享标量与逐帧 min-max 导致地板和 RGB 范围外区域高分；
- 阶段 1：轨迹过短，static/moving 支撑过少；
- 阶段 1.5：覆盖改善，但人工准确率和可观测性仍不足；
- 双 volume attempt 001：static 区域过宽；
- attempt 002：自适应裁剪前存在泄漏；
- attempt 003：二维 mask 生长仍会粘连。

这些失败版本证明最终改善并非简单更换阈值，而是依次修正了相机位姿使用、有效区域、moving map 标定、运动轴、`q_t` 和 canonical 部件融合。

## 4. 当前限制

1. HoloLens RGB 与深度视场不一致，RGB 范围外深度不能作为可靠语义或运动证据。
2. 深度噪声和遮挡边缘仍造成 Mesh 毛边、局部孔洞和表面粗糙。
3. 手遮挡后的抽屉表面无法完全依赖二维分割恢复，当前 canonical 3D 一致性只能部分补全。
4. 阶段 1.5 的 moving track 数量仍少，不适合直接放开相机、轴和标签的全联合优化。
5. 当前只验证单个 prismatic 抽屉；revolute 支持尚未在同等质量真实序列上完成验证。
6. 当前模型适合研究可视化，不应描述为完整、精确或生产级数字孪生。

## 5. 下周建议

优先级建议如下：

1. 对 13 个跨缺口重关联做人工审计，并补充 moving/static/unknown 人工标注；
2. 改进 RGB footprint 与 depth confidence 注册，继续压制深度边缘和 RGB 范围外噪声；
3. 使用清空柜体、距离更远、覆盖关闭—交互—打开三阶段的一条连续 RGB-D 视频重新采集；
4. 在固定设备位姿附近加入带先验的轻量相机 refinement，而不是自由联合优化；
5. 引入部件级遮挡补全或 canonical shape completion，恢复手遮挡后的抽屉侧板/底板；
6. 对最终 static/drawer 点云做法线一致性、离群点和局部表面修复，再比较 TSDF 与 NKSR；
7. 在可观测性通过后，再进入关节类型/轴/相机/标签的阶段 2 联合优化。

## 6. 可直接用于报告摘要的表述

本周围绕真实 RGB-D 视频中的可动柜体重建，完成了 VGGT、MonST3R、AutoSeg-SAM2 与 iTACO 的本地部署和对比实验，并系统定位了相机位姿误差、RGB/深度视场不一致、手部遮挡、AutoSeg proposal 误用以及逐帧 min-max 对 moving map 的影响。受控实验表明，ARKit-like 位姿误差可将平面厚度由 1.13 cm 放大至 11.13 cm，并使法线偏差超过 30° 的比例由 3.09% 增加至 71.87%。在保留官方 baseline 的前提下，进一步实现了不可变帧索引、统一有效区域、三维轨迹三分类、hand mask 修正、跨缺口重关联和可观测性评估。moving map 修复后，高分区域占比由 58.52% 降至 3.19%，有效支撑外高分像素由 1,886,763 降至 0。最终固定 HoloLens 相机位姿，通过 moving centroid 恢复平移轴和单调关节状态，估计抽屉行程约 0.300 m，并采用 articulation-aware 双 volume 与 NKSR 生成了包含部分内部结构的双面 GLB/URDF 可动模型。当前结果已满足研究展示和运动可视化需求，但仍存在深度噪声、遮挡补全不足和 Mesh 表面粗糙等问题。
