# Code Organization Guide

这份文档记录当前代码库已经形成的设计和文件放置习惯。新增功能时优先遵守这里的规则，至少保证文件位置、命名和配置入口统一。

## 总体结构

当前服务端以 JSON task 为边界组织 pipeline：

1. `server_api.py` 接收请求、创建任务、查询状态和提供文件下载。
2. `task_worker.py` 维护队列，并按 stage 顺序调用独立脚本。
3. 每个 stage 读取同一个 task JSON，产出文件后把结果路径和元数据写回 JSON。
4. Unity/HoloLens 端只消费 API、FBX/runtime mesh、pose、bounds 等接口结果。

核心原则是：服务内部逻辑放在 `code/`，运行时数据放在 `data/`，第三方模型源码放在 `code/reconstruction/`，文档放在 `docs/`。

## 顶层目录职责

`code/`
: 服务端源码。API、worker、数据库、JSON 工具、坐标换算、跨 stage 共享逻辑都放这里。

`code/stages/`
: pipeline stage 脚本。新增的可排队处理步骤应该放到这里的具体业务子目录。

`code/stages/hololens3d_reconstruction/`
: 物体重建主链路，包括 SAM3 mask、模型生成、深度点云、尺度估计、物体对齐、runtime mesh、pose、Blender/FBX、bounds 等。

`code/stages/hololens_aruco_reference/`
: ArUco 参考系相关 stage，包括 marker 检测和把已完成物体同步到 ArUco 坐标。

`code/Hololens2/`
: HoloLens 数据获取、标定、depth/RGB 配准等设备接入代码。depth 和 RGB 配准链路保持独立，不要为了普通 pipeline 改动随意移动或重写这里。上游依赖不要直接放这里，优先放到 `code/reconstruction/`，这里保留项目自己的设备接入脚本。

`code/reconstruction/`
: 外部模型、研究代码和上游设备 SDK 集成，例如 `InstantMesh`、`sam3`、`sam-3d-objects`、`FoundationPose`、`shigure_core`、`hl2ss`。新功能的业务封装不要直接散落到这些目录里，优先在 `code/stages/` 或 `code/Hololens2/` 写 adapter/wrapper。

`code/scripts/`
: 开发、验证、导出、手动检查脚本。可以依赖项目代码，但不作为正式 worker stage。

`code/.test/` 和 `.test/`
: 临时实验、诊断和一次性跑数脚本。不要把生产逻辑放到这里；一旦需要长期保留，移动到 `code/scripts/` 或正式 stage 目录。

`docs/`
: 设计说明和约定文档。坐标系说明见 `docs/coordinate-systems.md`。

`data/`
: 运行时输入、输出、数据库、配置快照。代码不要从 `data/` import Python 模块。

`models/`
: 模型权重和静态模型资产。

`H2AI/`
: Unity/HoloLens 客户端工程。服务端只通过 API、生成文件和约定 JSON 字段与它交互。

## 核心模块职责

`code/config.py`
: 所有路径、Python 环境、stage 脚本入口、输出目录、运行开关和 HTTP 文件目录映射的集中入口。新增硬编码路径或运行参数前，先考虑是否应该放到这里。

`code/server_api.py`
: Flask API。只做请求解析、任务创建、状态查询、文件服务和轻量接口组合，避免把重建算法塞进 API 层。

`code/task_worker.py`
: 队列和 stage 调度。新增正式 stage 时，需要在这里注册执行函数、`STAGE_RUNNERS`，并按需要插入 stage 顺序。

`code/task_db.py`
: SQLite schema、任务状态和 stage run 记录。新增 worker status 时同步更新 `ALLOWED_STATUSES`。

`code/task_json.py`
: task JSON 的加载、保存、路径解析和存储路径归一化。stage 之间交换数据优先通过这里处理。

`code/coordinate_systems.py`
: 坐标系定义和转换唯一入口。不要在业务 stage 里散写新的轴翻转矩阵；新增坐标边界时先扩展这个文件，并同步更新 `docs/coordinate-systems.md`。

`code/object_alignment_common.py`
: 物体对齐链路的共享几何、图像、mesh 工具。只有多个对齐/姿态 stage 共用的逻辑才放这里。

`code/model_generation_common.py`
: 模型生成输入解析和不同生成后端之间的共同 payload 逻辑。

`code/model_bounds.py`
: 模型 bounds、射线查询和空间查询逻辑。

`code/subprocess_stream.py`
: 子进程日志流式输出工具。worker 和 stage 调用外部命令时优先用它，方便在服务日志里定位问题。

## Stage 放置和命名

正式 stage 文件使用：

```text
code/stages/<domain>/run_<stage_name>_from_json.py
```

现有主链路 stage 顺序：

```text
hololens2depth
sam3mask
instantmesh
depthpointcloud
modelscale
object_alignment
runtime_mesh
pose
aruco_sync
blender
model_bounds
```

ArUco reference task 使用：

```text
aruco_detect
```

新增 stage 时遵守这些习惯：

1. 入口脚本接受 task JSON 路径作为参数。
2. 用 `stage_common.load_stage_task()` 读取输入并打印 stage 标识。
3. 用 `task_json.save_task_json()` 写回结果。
4. 输出文件写入 `data/output/<stage-or-domain>/` 下的专用目录。
5. 运行路径、Python 环境、外部工具路径写入 `config.py`。
6. 如果由 worker 调度，在 `task_worker.py` 增加 runner、`STAGE_RUNNERS` 和 stage 顺序。
7. 如果 stage 名会进入任务状态，在 `task_db.py` 的 `ALLOWED_STATUSES` 中登记。
8. 如果客户端需要下载新产物，在 `config.py` 的 `FOLDER_MAP` 增加目录映射。

## 共享代码放置规则

跨多个 stage 使用的通用逻辑放在 `code/` 顶层模块，例如坐标、JSON、数据库、模型 bounds、通用子进程工具。

只服务于某个 stage domain 的 helper 放在对应目录，例如：

```text
code/stages/hololens3d_reconstruction/mesh_obj_utils.py
code/stages/hololens3d_reconstruction/blender_mesh_postprocess.py
code/stages/hololens_aruco_reference/aruco_common.py
```

第三方仓库里的代码尽量不承载服务端业务逻辑。需要接入第三方模型时，把原始模型代码留在 `code/reconstruction/<project>/`，在 `code/stages/...` 写一个小的项目适配层。

## 配置和路径规则

不要在业务代码里直接写死 `/workspace`、输出目录、模型目录、Python 解释器或下载 URL。路径和运行开关优先进入 `config.py`，并用 `Path` 组合：

```python
OUTPUT_ROOT / "object_alignment"
HOLOLENS3D_RECON_STAGE_ROOT / "run_pose_from_json.py"
```

只有临时测试脚本可以少量硬编码本地样例路径；一旦脚本进入 `code/scripts/` 或 `code/stages/`，就应切换为 `config.py` 或命令行参数。

## 数据和 JSON 约定

task JSON 是 stage 之间的稳定接口。stage 不应该依赖上一个 stage 的局部变量或临时进程状态，而应从 JSON 读输入、检查文件存在、写回输出。

路径写回 JSON 时优先使用项目内可解析的相对/规范化路径。读取路径时使用 `task_json.resolve_task_json_path()` 或已有 resolver，而不是手写路径拼接。

输出字段应按现有语义分组，例如：

```text
ModelGeneration
SAM3DObjects
InstantMesh
RuntimeMesh
Blender
ModelBounds
object_alignment
object_world
```

新增字段要表达“数据属于哪个 stage 或哪个坐标边界”，不要把多个 stage 的输出混在同一个无名 dict 里。

## 坐标系边界

服务器内部计算应尽量使用统一 canonical 坐标系。外部接口的坐标差异只在边界转换：

```text
HoloLens / Unity / OpenCV / Blender / model input / Shigurei
```

坐标定义和转换函数统一放在 `code/coordinate_systems.py`。详细说明写在 `docs/coordinate-systems.md`。新增功能如果需要新坐标来源，先增加清晰命名的转换函数，再在业务代码中调用，不要局部复制轴翻转逻辑。

HoloLens depth/RGB 配准链路是设备输入边界，除非正在处理配准问题，否则不要顺手重构。

## 新增功能检查清单

新增正式 pipeline 功能时，按下面顺序落地：

1. 判断功能属于服务/API、worker stage、HoloLens 接入、第三方模型 adapter、开发脚本还是文档。
2. 选择对应目录，不把生产代码放进 `.test`。
3. 在 `config.py` 增加路径、输出目录、运行开关和 Python 环境配置。
4. 如果是 stage，创建 `run_<stage>_from_json.py`，并使用 `stage_common`、`task_json`、`subprocess_stream` 等现有工具。
5. 在 `task_worker.py` 和 `task_db.py` 注册 stage。
6. 输出写入 `data/output/` 下的专用目录，并把稳定字段写回 task JSON。
7. 如果结果要给客户端下载，更新 `FOLDER_MAP`。
8. 如果改变坐标、JSON schema、API 或文件位置，更新 `docs/`。
9. 至少运行 `python3 -m py_compile` 覆盖被改动的 Python 文件；高风险 stage 再用一份已有 task JSON 做 smoke test。

## 清理和兼容原则

历史 wrapper 可以保留在明确标注的兼容入口，例如 `run_object_icp_alignment_from_json.py`。新增代码不要继续扩散旧命名。

清理无用代码时，先确认它不是 `config.py`、`task_worker.py`、`task_db.py`、Unity 客户端或旧 task JSON 仍在引用的入口。删除第三方网络/远程调用时，保留当前本地生成逻辑和 JSON 输出契约。

如果一个临时脚本被连续用于验证当前链路，优先把它整理到 `code/scripts/`；如果它已经成为正式 pipeline 的一部分，就改造成 stage。
