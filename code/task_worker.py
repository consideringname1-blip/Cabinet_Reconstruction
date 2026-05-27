import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

from subprocess_stream import stream_command

from config import (
    ARUCO_DETECT_STAGE_PY,
    ARUCO_DETECT_STAGE_RUN,
    ARUCO_SYNC_STAGE_PY,
    ARUCO_SYNC_STAGE_RUN,
    BLENDER_STAGE_PY,
    BLENDER_STAGE_RUN,
    DEPTHPOINTCLOUD_STAGE_PY,
    DEPTHPOINTCLOUD_STAGE_RUN,
    HOLOLENS2_CONVERT_DIR,
    HOLOLENS2_CONVERT_RUN,
    HOLOLENS2_PY,
    OBJECT_ALIGNMENT_STAGE_PY,
    OBJECT_ALIGNMENT_STAGE_RUN,
    INSTANTMESH_STAGE_PY,
    INSTANTMESH_STAGE_RUN,
    MODEL_GENERATION_BACKEND,
    MODELSCALE_STAGE_PY,
    MODELSCALE_STAGE_RUN,
    MODEL_BOUNDS_STAGE_PY,
    MODEL_BOUNDS_STAGE_RUN,
    POSE_STAGE_PY,
    POSE_STAGE_RUN,
    RUNTIME_MESH_STAGE_PY,
    RUNTIME_MESH_STAGE_RUN,
    SAM3_BOX_MASK_RUN,
    SAM3_DIR,
    SAM3D_OBJECTS_ROOT,
    SAM3D_OBJECTS_STAGE_PY,
    SAM3D_OBJECTS_STAGE_RUN,
    SAM3_PY,
)
from task_db import (
    create_task as create_task_record,
    get_completed_tasks_for_startup,
    get_latest_completed_task,
    get_task_stage_runs,
    get_task_by_task_id,
    get_unfinished_tasks,
    get_unsynced_completed_tasks,
    initialize_task_table,
    mark_task_stage_completed,
    mark_task_stage_failed,
    mark_task_stage_started,
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
    "depthpointcloud",
    "modelscale",
    "object_alignment",
    "runtime_mesh",
    "pose",
    "aruco_sync",
    "blender",
    "model_bounds",
]

PURPOSE_OBJECT_RECONSTRUCTION = "object_reconstruction"
PURPOSE_ARUCO_REFERENCE = "aruco_reference"

_aruco_task_queue = deque()
_model_task_queue = deque()
_task_lock = threading.Lock()
_current_task_id: Optional[str] = None
_worker_thread: Optional[threading.Thread] = None


def _resolve_python(python_path: str) -> str:
    return python_path or sys.executable


def _queue_snapshot_no_lock() -> list[str]:
    return list(_aruco_task_queue) + list(_model_task_queue)


def _enqueue_task_no_lock(task_id: str, purpose: str, *, front: bool = False) -> None:
    if purpose == PURPOSE_ARUCO_REFERENCE:
        queue = _aruco_task_queue
    else:
        queue = _model_task_queue

    if front:
        queue.appendleft(task_id)
    else:
        queue.append(task_id)


def _restore_unfinished_tasks() -> None:
    unfinished_tasks = get_unfinished_tasks()
    with _task_lock:
        queued = set(_queue_snapshot_no_lock())
        for task in unfinished_tasks:
            task_id = task["task_id"]
            if task_id == _current_task_id:
                continue
            if task_id in queued:
                continue
            try:
                task_json = load_task_json(resolve_task_json_path(task["json_path"]))
                purpose = _resolve_task_purpose(task_json)
            except Exception:
                purpose = PURPOSE_OBJECT_RECONSTRUCTION
            _enqueue_task_no_lock(task_id, purpose)
            queued.add(task_id)


def _run_python_script(python_path: str, script_path: Path, json_path: Path, cwd: Path) -> None:
    stream_command(
        [_resolve_python(python_path), str(script_path), str(json_path)],
        cwd=cwd,
        check=True,
    )


def _run_hololens2depth(json_path: Path) -> None:
    _run_python_script(
        python_path=HOLOLENS2_PY,
        script_path=HOLOLENS2_CONVERT_RUN,
        json_path=json_path,
        cwd=HOLOLENS2_CONVERT_DIR,
    )


def _run_aruco_detect(json_path: Path) -> None:
    _run_python_script(
        python_path=ARUCO_DETECT_STAGE_PY,
        script_path=ARUCO_DETECT_STAGE_RUN,
        json_path=json_path,
        cwd=ARUCO_DETECT_STAGE_RUN.parent,
    )


def _run_sam3mask(json_path: Path) -> None:
    _run_python_script(
        python_path=SAM3_PY,
        script_path=SAM3_BOX_MASK_RUN,
        json_path=json_path,
        cwd=SAM3_DIR,
    )


def _run_model_generation(json_path: Path) -> None:
    if MODEL_GENERATION_BACKEND == "sam3d_objects":
        _run_python_script(
            python_path=SAM3D_OBJECTS_STAGE_PY,
            script_path=SAM3D_OBJECTS_STAGE_RUN,
            json_path=json_path,
            cwd=SAM3D_OBJECTS_ROOT,
        )
        return

    _run_python_script(
        python_path=INSTANTMESH_STAGE_PY,
        script_path=INSTANTMESH_STAGE_RUN,
        json_path=json_path,
        cwd=INSTANTMESH_STAGE_RUN.parent,
    )


def _run_instantmesh(json_path: Path) -> None:
    _run_model_generation(json_path)


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


def _run_object_alignment(json_path: Path) -> None:
    _run_python_script(
        python_path=OBJECT_ALIGNMENT_STAGE_PY,
        script_path=OBJECT_ALIGNMENT_STAGE_RUN,
        json_path=json_path,
        cwd=OBJECT_ALIGNMENT_STAGE_RUN.parent,
    )


def _run_pose(json_path: Path) -> None:
    _run_python_script(
        python_path=POSE_STAGE_PY,
        script_path=POSE_STAGE_RUN,
        json_path=json_path,
        cwd=POSE_STAGE_RUN.parent,
    )


def _run_aruco_sync(json_path: Path) -> None:
    _run_python_script(
        python_path=ARUCO_SYNC_STAGE_PY,
        script_path=ARUCO_SYNC_STAGE_RUN,
        json_path=json_path,
        cwd=ARUCO_SYNC_STAGE_RUN.parent,
    )


def _run_runtime_mesh(json_path: Path) -> None:
    _run_python_script(
        python_path=RUNTIME_MESH_STAGE_PY,
        script_path=RUNTIME_MESH_STAGE_RUN,
        json_path=json_path,
        cwd=RUNTIME_MESH_STAGE_RUN.parent,
    )


def _run_blender(json_path: Path) -> None:
    _run_python_script(
        python_path=BLENDER_STAGE_PY,
        script_path=BLENDER_STAGE_RUN,
        json_path=json_path,
        cwd=BLENDER_STAGE_RUN.parent,
    )


def _run_model_bounds(json_path: Path) -> None:
    _run_python_script(
        python_path=MODEL_BOUNDS_STAGE_PY,
        script_path=MODEL_BOUNDS_STAGE_RUN,
        json_path=json_path,
        cwd=MODEL_BOUNDS_STAGE_RUN.parent,
    )


STAGE_RUNNERS = {
    "hololens2depth": _run_hololens2depth,
    "aruco_detect": _run_aruco_detect,
    "sam3mask": _run_sam3mask,
    "instantmesh": _run_instantmesh,
    "depthpointcloud": _run_depthpointcloud,
    "modelscale": _run_modelscale,
    "object_alignment": _run_object_alignment,
    "pose": _run_pose,
    "aruco_sync": _run_aruco_sync,
    "runtime_mesh": _run_runtime_mesh,
    "blender": _run_blender,
    "model_bounds": _run_model_bounds,
}


def _resolve_task_purpose(task_json: dict) -> str:
    purpose = str(task_json.get("purpose") or "").strip()
    return purpose or PURPOSE_OBJECT_RECONSTRUCTION


def _resolve_stage_order(task_json: dict) -> list[str]:
    purpose = _resolve_task_purpose(task_json)
    if purpose == PURPOSE_ARUCO_REFERENCE:
        return ["aruco_detect"]
    return STAGE_ORDER


def _process_one_task(task_id: str) -> tuple[bool, str]:
    task_record = get_task_by_task_id(task_id)
    if task_record is None:
        raise ValueError(f"Task not found in database: {task_id}")

    json_path = resolve_task_json_path(task_record["json_path"])
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    ensure_task_id_in_json(json_path, task_id)
    task_json = load_task_json(json_path)
    purpose = _resolve_task_purpose(task_json)
    stage_order = _resolve_stage_order(task_json)

    current_status = str(task_record["status"])
    if current_status == "pending":
        current_status = stage_order[0]
        update_task_status(task_id, current_status)

    if current_status not in STAGE_RUNNERS or current_status not in stage_order:
        raise ValueError(f"Task {task_id} has unsupported status: {current_status}")

    start_index = stage_order.index(current_status)

    stage_name = stage_order[start_index]
    update_task_status(task_id, stage_name)
    mark_task_stage_started(task_id, stage_name)
    try:
        STAGE_RUNNERS[stage_name](json_path)
        mark_task_stage_completed(task_id, stage_name)
    except Exception as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            error_message = exc.stderr or exc.stdout or str(exc)
        else:
            error_message = str(exc)
        mark_task_stage_failed(task_id, stage_name, error_message=error_message)
        raise

    if stage_name == "aruco_detect" and purpose == PURPOSE_ARUCO_REFERENCE:
        startup_session_id = str((task_json.get("device") or {}).get("startup_session_id") or "").strip() or None
        synced_count = _sync_completed_tasks_for_startup(startup_session_id)
        if synced_count:
            print(f"[worker] synced completed model tasks after ArUco reference: {synced_count}")
        update_task_status(task_id, "aruco_completed")
        return True, purpose

    next_status = "completed"
    if start_index + 1 < len(stage_order):
        next_status = stage_order[start_index + 1]
    update_task_status(task_id, next_status)
    return next_status == "completed", purpose


def _process_tasks_loop() -> None:
    global _current_task_id

    while True:
        task_id = None

        with _task_lock:
            if _aruco_task_queue:
                task_id = _aruco_task_queue.popleft()
                _current_task_id = task_id
            elif _model_task_queue:
                task_id = _model_task_queue.popleft()
                _current_task_id = task_id

        if task_id is None:
            _restore_unfinished_tasks()
            time.sleep(1)
            continue

        requeue_task = False
        requeue_purpose = PURPOSE_OBJECT_RECONSTRUCTION
        try:
            print(f"[worker] start task: {task_id}")
            is_terminal, requeue_purpose = _process_one_task(task_id)
            if is_terminal:
                print(f"[worker] completed task: {task_id}")
            else:
                requeue_task = True
                print(f"[worker] stage completed, requeue task: {task_id}")
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
                if _current_task_id == task_id:
                    _current_task_id = None
                if requeue_task:
                    _enqueue_task_no_lock(task_id, requeue_purpose, front=True)

        time.sleep(1)


def start_worker() -> threading.Thread:
    global _worker_thread

    initialize_task_table()
    _restore_unfinished_tasks()

    if _worker_thread is not None and _worker_thread.is_alive():
        return _worker_thread

    _worker_thread = threading.Thread(target=_process_tasks_loop, daemon=True)
    _worker_thread.start()
    return _worker_thread


def create_task(json_path: Path | str) -> str:
    task_json_path = resolve_task_json_path(json_path)
    if not task_json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {task_json_path}")

    data = load_task_json(task_json_path)
    task_id = str(data.get("task_id") or uuid.uuid4())
    startup_session_id = str((data.get("device") or {}).get("startup_session_id") or "").strip() or None

    data["task_id"] = task_id
    save_task_json(task_json_path, data)

    create_task_record(
        task_id=task_id,
        json_path=task_json_path,
        startup_session_id=startup_session_id,
    )

    with _task_lock:
        _enqueue_task_no_lock(task_id, _resolve_task_purpose(data))

    return task_id


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    task_record = get_task_by_task_id(task_id)
    if task_record is None:
        return None

    try:
        json_path = resolve_task_json_path(task_record["json_path"])
        task_json = load_task_json(json_path)
    except FileNotFoundError:
        task_json = {}

    task_record["task_json"] = task_json
    task_record["error"] = task_record.get("error_message")
    task_record["stage_runs"] = get_task_stage_runs(task_id)
    task_record["outputs"] = {
        "model_generation": task_json.get("ModelGeneration") or {},
        "sam3d_objects": task_json.get("SAM3DObjects") or {},
        "instantmesh": task_json.get("InstantMesh") or {},
        "runtime_mesh": task_json.get("RuntimeMesh") or {},
        "blender": task_json.get("Blender") or {},
    }
    return task_record


def _sync_completed_tasks_for_startup(startup_session_id: str | None = None) -> int:
    synced_count = 0
    task_rows = (
        get_completed_tasks_for_startup(startup_session_id, require_unsynced=False)
        if startup_session_id
        else get_unsynced_completed_tasks()
    )
    for task_row in task_rows:
        json_path = task_row.get("json_path")
        if not json_path:
            continue
        try:
            resolved_json_path = resolve_task_json_path(json_path)
            _run_aruco_sync(resolved_json_path)
            _run_model_bounds(resolved_json_path)
            synced_count += 1
        except Exception as exc:
            print(
                f"[worker] failed to sync completed task {task_row.get('task_id')} to ArUco reference/model bounds: {exc}"
            )
    return synced_count


def get_latest_completed_task_data(
    startup_session_id: str | None = None,
    require_aruco_coordinate_synced: bool = False,
    history_offset: int = 0,
    attempt_sync: bool = True,
) -> Optional[Dict[str, Any]]:
    history_offset = max(0, int(history_offset or 0))
    task_record = get_latest_completed_task(
        startup_session_id=startup_session_id,
        require_aruco_coordinate_synced=require_aruco_coordinate_synced,
        history_offset=history_offset,
    )
    if task_record is None and startup_session_id and require_aruco_coordinate_synced and attempt_sync:
        _sync_completed_tasks_for_startup(startup_session_id)
        task_record = get_latest_completed_task(
            startup_session_id=startup_session_id,
            require_aruco_coordinate_synced=True,
            history_offset=history_offset,
        )
    if task_record is None and require_aruco_coordinate_synced and not startup_session_id and attempt_sync:
        _sync_completed_tasks_for_startup()
        task_record = get_latest_completed_task(
            startup_session_id=startup_session_id,
            require_aruco_coordinate_synced=True,
            history_offset=history_offset,
        )
    if task_record is None:
        return None

    try:
        json_path = resolve_task_json_path(task_record["json_path"])
        task_json = load_task_json(json_path)
    except FileNotFoundError:
        task_json = {}

    task_record["task_json"] = task_json
    task_record["error"] = task_record.get("error_message")
    task_record["stage_runs"] = get_task_stage_runs(str(task_record.get("task_id") or ""))
    task_record["outputs"] = {
        "model_generation": task_json.get("ModelGeneration") or {},
        "sam3d_objects": task_json.get("SAM3DObjects") or {},
        "instantmesh": task_json.get("InstantMesh") or {},
        "runtime_mesh": task_json.get("RuntimeMesh") or {},
        "blender": task_json.get("Blender") or {},
    }
    return task_record
