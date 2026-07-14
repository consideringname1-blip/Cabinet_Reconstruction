# Documentation Index

更新日期：2026-07-14
状态：Shigure v2 当前协议

## 系统与 API

- [`task-processing-flow.md`](task-processing-flow.md)：主模型任务、Shigure 生命周期、身份恢复与 runtime 流程。
- [`server-processing-output-contract.md`](server-processing-output-contract.md)：严格 HTTP、canonical model/live 和历史照片/骨骼契约。
- [`coordinate-systems.md`](coordinate-systems.md)：HoloLens、ArUco、Shigure 与 Unity 的坐标边界。
- [`identity-management.md`](identity-management.md)：`display_object_id`、DINOv2、模型复用和 revision。
- [`code-organization.md`](code-organization.md)：代码目录与模块职责。
- [`implementation-coverage-notes.md`](implementation-coverage-notes.md)：辅助接口、worker service、持久状态和 Unity 层覆盖说明。

## Shigure 与 Unity

- [`shigure-core-ros-messages.md`](shigure-core-ros-messages.md)：远端 Shigure topics、exact-stamp canonical event 和内存 socket cache。
- [`model-event-tracking-state-machine.md`](model-event-tracking-state-machine.md)：Unity 模型缓存、live/history 呈现、线框 box 与照片/点线骨骼。
