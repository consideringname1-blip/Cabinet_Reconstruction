import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

from config import (
    BLENDER_STAGE_PY,
    BLENDER_STAGE_RUN,
    DEPTHPOINTCLOUD_STAGE_PY,
    DEPTHPOINTCLOUD_STAGE_RUN,
    HOLOLENS2_CONVERT_DIR,
    HOLOLENS2_CONVERT_RUN,
    HOLOLENS2_PY,
    ICPALIGNMENT_STAGE_PY,
    ICPALIGNMENT_STAGE_RUN,
    INSTANTMESH_STAGE_PY,
    INSTANTMESH_STAGE_RUN,
    MODELSCALE_STAGE_PY,
    MODELSCALE_STAGE_RUN,
    POSE_STAGE_PY,
    POSE_STAGE_RUN,
    SAM3_BOX_MASK_RUN,
    SAM3_DIR,
    SAM3_PY,
)
from task_db import (
    create_task as create_task_record,
    get_latest_completed_task,
    get_task_by_task_id,
    get_unfinished_tasks,
    initialize_task_table,
    update_task_status,
)
from task_json import (
    ensure_task_id_in_json,
    load_task_json,
    resolve_task_json_path,
    save_task_json,
)


STAGE_ORDER = [
    "hololens2depth",
    "sam3mask",
    "instantmesh",
    "objectalignment",
    "depthpointcloud",
    "modelscale",
    "icpalignment",
    "pose",
    "blender",
]

_task_queue = deque()
_task_lock = threading.Lock()
_current_task_id: Optional[str] = None
_worker_thread: Optional[threading.Thread] = None


def _resolve_python(python_path: str) -> str:
    return python_path or sys.executable


def _queue_snapshot_no_lock() -> list[str]:
    return list(reversed(_task_queue))


def _restore_unfinished_tasks() -> None:
    unfinished_tasks = get_unfinished_tasks()
    with _task_lock:
        queued = set(_task_queue)
        for task in unfinished_tasks:
            task_id = task["task_id"]
            if task_id == _current_task_id:
                continue
            if task_id in queued:
                continue
            _task_queue.append(task_id)
            queued.add(task_id)


def _run_python_script(python_path: str, script_path: Path, json_path: Path, cwd: Path) -> None:
    result = subprocess.run(
        [_resolve_python(python_path), str(script_path), str(json_path)],
        cwd=str(cwd),
        check=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)


def _run_hololens2depth(json_path: Path) -> None:
    _run_python_script(
        python_path=HOLOLENS2_PY,
        script_path=HOLOLENS2_CONVERT_RUN,
        json_path=json_path,
        cwd=HOLOLENS2_CONVERT_DIR,
    )


def _run_sam3mask(json_path: Path) -> None:
    _run_python_script(
        python_path=SAM3_PY,
        script_path=SAM3_BOX_MASK_RUN,
        json_path=json_path,
        cwd=SAM3_DIR,
    )


def _run_instantmesh(json_path: Path) -> None:
    _run_python_script(
        python_path=INSTANTMESH_STAGE_PY,
        script_path=INSTANTMESH_STAGE_RUN,
        json_path=json_path,
        cwd=INSTANTMESH_STAGE_RUN.parent,
    )


def _run_objectalignment(json_path: Path) -> None:
    return


def _run_depthpointcloud(json_path: Path) -> None:
    _run_python_script(
        python_path=DEPTHPOINTCLOUD_STAGE_PY,
        script_path=DEPTHPOINTCLOUD_STAGE_RUN,
        json_path=json_path,
        cwd=DEPTHPOINTCLOUD_STAGE_RUN.parent,
    )


def _run_modelscale(json_path: Path) -> None:
    _run_python_script(
        python_path=MODELSCALE_STAGE_PY,
        script_path=MODELSCALE_STAGE_RUN,
        json_path=json_path,
        cwd=MODELSCALE_STAGE_RUN.parent,
    )


def _run_icpalignment(json_path: Path) -> None:
    _run_python_script(
        python_path=ICPALIGNMENT_STAGE_PY,
        script_path=ICPALIGNMENT_STAGE_RUN,
        json_path=json_path,
        cwd=ICPALIGNMENT_STAGE_RUN.parent,
    )


def _run_pose(json_path: Path) -> None:
    _run_python_script(
        python_path=POSE_STAGE_PY,
        script_path=POSE_STAGE_RUN,
        json_path=json_path,
        cwd=POSE_STAGE_RUN.parent,
    )


def _run_blender(json_path: Path) -> None:
    _run_python_script(
        python_path=BLENDER_STAGE_PY,
        script_path=BLENDER_STAGE_RUN,
        json_path=json_path,
        cwd=BLENDER_STAGE_RUN.parent,
    )


STAGE_RUNNERS = {
    "hololens2depth": _run_hololens2depth,
    "sam3mask": _run_sam3mask,
    "instantmesh": _run_instantmesh,
    "objectalignment": _run_objectalignment,
    "depthpointcloud": _run_depthpointcloud,
    "modelscale": _run_modelscale,
    "icpalignment": _run_icpalignment,
    "pose": _run_pose,
    "blender": _run_blender,
}


def _process_one_task(task_id: str) -> None:
    task_record = get_task_by_task_id(task_id)
    if task_record is None:
        raise ValueError(f"Task not found in database: {task_id}")

    json_path = Path(task_record["json_path"]).expanduser().resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    ensure_task_id_in_json(json_path, task_id)

    current_status = str(task_record["status"])
    if current_status == "pending":
        current_status = STAGE_ORDER[0]
        update_task_status(task_id, current_status)

    if current_status not in STAGE_RUNNERS:
        raise ValueError(f"Task {task_id} has unsupported status: {current_status}")

    start_index = STAGE_ORDER.index(current_status)

    for index in range(start_index, len(STAGE_ORDER)):
        stage_name = STAGE_ORDER[index]
        update_task_status(task_id, stage_name)
        STAGE_RUNNERS[stage_name](json_path)

        next_status = "completed"
        if index + 1 < len(STAGE_ORDER):
            next_status = STAGE_ORDER[index + 1]
        update_task_status(task_id, next_status)


def _process_tasks_loop() -> None:
    global _current_task_id

    while True:
        task_id = None

        with _task_lock:
            if _task_queue:
                task_id = _task_queue.pop()
                _current_task_id = task_id

        if task_id is None:
            _restore_unfinished_tasks()
            time.sleep(1)
            continue

        try:
            print(f"[worker] start task: {task_id}")
            _process_one_task(task_id)
            print(f"[worker] completed task: {task_id}")
        except Exception as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                error_message = exc.stderr or exc.stdout or str(exc)
            else:
                error_message = str(exc)
            print(f"[worker] failed task {task_id}: {error_message}")
            try:
                update_task_status(task_id, "failed", error_message=error_message)
            except Exception as db_exc:
                print(f"[worker] failed to write error to database: {db_exc}")
        finally:
            with _task_lock:
                _current_task_id = None

        time.sleep(1)


def start_worker() -> threading.Thread:
    """启动后台任务线程；重复调用时复用同一个线程。"""
    global _worker_thread

    initialize_task_table()
    _restore_unfinished_tasks()

    if _worker_thread is not None and _worker_thread.is_alive():
        return _worker_thread

    _worker_thread = threading.Thread(target=_process_tasks_loop, daemon=True)
    _worker_thread.start()
    return _worker_thread


def create_task(json_path: Path | str) -> str:
    """创建任务记录并加入内存队列。"""
    task_json_path = resolve_task_json_path(json_path)
    if not task_json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {task_json_path}")

    data = load_task_json(task_json_path)
    task_id = str(data.get("task_id") or uuid.uuid4())

    data["task_id"] = task_id
    save_task_json(task_json_path, data)

    create_task_record(task_id=task_id, json_path=task_json_path)

    with _task_lock:
        _task_queue.append(task_id)

    return task_id


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """根据 task_id 查询数据库记录，并附带 json 内容。"""
    task_record = get_task_by_task_id(task_id)
    if task_record is None:
        return None

    json_path = Path(task_record["json_path"])
    if json_path.is_file():
        task_json = load_task_json(json_path)
    else:
        task_json = {}

    task_record["task_json"] = task_json
    task_record["error"] = task_record.get("error_message")
    task_record["outputs"] = {
        "instantmesh": task_json.get("InstantMesh") or {},
        "blender": task_json.get("Blender") or {},
    }
    return task_record


def get_latest_completed_task_data() -> Optional[Dict[str, Any]]:
    """Return the most recent completed task with loaded JSON outputs."""
    task_record = get_latest_completed_task()
    if task_record is None:
        return None

    json_path = Path(task_record["json_path"])
    if json_path.is_file():
        task_json = load_task_json(json_path)
    else:
        task_json = {}

    task_record["task_json"] = task_json
    task_record["error"] = task_record.get("error_message")
    task_record["outputs"] = {
        "instantmesh": task_json.get("InstantMesh") or {},
        "blender": task_json.get("Blender") or {},
    }
    return task_record


def get_current_task_id() -> Optional[str]:
    with _task_lock:
        return _current_task_id


def get_queue_snapshot() -> list[str]:
    """返回当前等待队列，列表第一个元素就是下一个任务。"""
    with _task_lock:
        return _queue_snapshot_no_lock()
