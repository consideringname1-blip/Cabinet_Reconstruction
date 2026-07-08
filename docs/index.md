# Documentation Index

更新日期：2026-07-08
状态：当前实现说明

本文是 `docs/` 目录索引。流程图只记录大致数据流；实际触发、接口和失败处理以 Markdown 文档和代码为准。

## Flow Diagram

- [`hwang-project-flow.drawio`](hwang-project-flow.drawio): 项目数据流参考图。

## System Contracts

- [`task-processing-flow.md`](task-processing-flow.md): 任务处理流程、stage 顺序、目录布局。
- [`server-processing-output-contract.md`](server-processing-output-contract.md): API 输入输出、artifact 和 JSON 契约。
- [`code-organization.md`](code-organization.md): 代码目录、模块职责和新增代码放置规则。
- [`implementation-coverage-notes.md`](implementation-coverage-notes.md): 代码中辅助接口、worker service、helper 脚本和 Unity 辅助层覆盖说明。
- [`coordinate-systems.md`](coordinate-systems.md): HoloLens、ArUco、Shigure、Unity 坐标边界和统一投影函数。

## Object And History Pipeline

- [`identity-management.md`](identity-management.md): DINOv2 历史模型复用、force-new 规则。
- [`taken-object-detection-state-machine.md`](taken-object-detection-state-machine.md): Shigure object mask 初始化和拿取判断状态机。
- [`history-position-restoration.md`](history-position-restoration.md): fixed Shigure 历史位置再现和 direct compare。
- [`history-placement-restoration-review.md`](history-placement-restoration-review.md): 历史再现审查结论。
- [`history-placement-coordinate-review.md`](history-placement-coordinate-review.md): 历史再现坐标链路审查。
- [`model-event-tracking-state-machine.md`](model-event-tracking-state-machine.md): Unity pending/completed/history 显示状态机。

## Shigure And Body Evidence

- [`shigure-core-ros-messages.md`](shigure-core-ros-messages.md): Shigure RGB-D、CameraInfo、object_detection 缓存与 socket 访问。
- [`sam3d-body-workflow.md`](sam3d-body-workflow.md): SAM3D Body 人体 mesh、距离修正和 subject crop 流程。
- [`sam3d-body-setup.md`](sam3d-body-setup.md): SAM3D Body stage 运行依赖、配置和排查入口。

## Naming Rule

文档文件名统一使用英文 lowercase kebab-case。保留 `.drawio` 作为流程图格式；新 Markdown 文档优先放到本索引对应章节中。
