# 历史位置再现与显示去重身份管理概览

日期：2026-06-24

这组文档现在是正式说明，位于 `docs/` 根目录：

- [Unity 显示去重身份管理](identity-management.md)：负责 `display_object_id`、`capture_instance_id`、HoloLens/SAM3 证据特征、SQLite 绑定记录，以及 Unity 默认物体列表去重。
- [历史位置再现](history-position-restoration.md)：负责已有模型在 Shigurei 当前或指定时间里的 `ORIGINAL`、`MOVED`、`MISSING`、`OCCLUDED_REUSE_LAST`、`UNKNOWN` 状态判断和 Unity 展示输出。

职责边界：

- 显示去重身份管理只决定多个 HoloLens captures 是否应在 Unity 中显示为同一个对象条目，不证明严格真实世界同一性。
- 历史位置再现可以读取 `display_object_id` 作为服务端去重线索，但不创建、合并或拆分 `display_object_id`。
- `UNKNOWN` 是保护状态，不能自动触发身份绑定、模型替换或历史状态覆盖。
