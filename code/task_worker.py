import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

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
    FOUNDATIONPOSE_ALIGNMENT_PY,
    FOUNDATIONPOSE_ALIGNMENT_RUN,
    FOUNDATIONPOSE_WORKER_IDLE_TIMEOUT_SEC,
    MODEL_SERVICE_PREWARM_ENABLE,
    INSTANTMESH_WORKER_IDLE_TIMEOUT_SEC,
    HOLOLENS2_CONVERT_DIR,
    HOLOLENS2_CONVERT_RUN,
    HOLOLENS2_PY,
    HISTORY_PLACEMENT_RESTORATION_STAGE_PY,
    HISTORY_PLACEMENT_RESTORATION_STAGE_RUN,
    INSTANTMESH_GPU_IDS,
    INSTANTMESH_STAGE_PY,
    INSTANTMESH_STAGE_RUN,
    MODEL_BOUNDS_STAGE_PY,
    MODEL_BOUNDS_STAGE_RUN,
    DISPLAY_IDENTITY_STAGE_PY,
    DISPLAY_IDENTITY_STAGE_RUN,
    MODEL_GENERATION_BACKEND,
    MODELSCALE_STAGE_PY,
    MODELSCALE_STAGE_RUN,
    OBJECT_ALIGNMENT_STAGE_PY,
    OBJECT_ALIGNMENT_STAGE_RUN,
    POSE_STAGE_PY,
    POSE_STAGE_RUN,
    RUNTIME_MESH_STAGE_PY,
    RUNTIME_MESH_STAGE_RUN,
    SAM3_BOX_MASK_RUN,
    SAM3_DIR,
    SAM3_PY,
    SAM3D_OBJECTS_ROOT,
    SAM3D_OBJECTS_STAGE_PY,
    SAM3D_OBJECTS_STAGE_RUN,
    SAM3D_BODY_MESH_STAGE_PY,
    SAM3D_BODY_MESH_STAGE_RUN,
    SHIGURE_HISTORY_CACHE_ROOT,
    SHIGURE_HISTORY_RECORDER_RUN,
    SHIGURE_HISTORY_RECORDER_STAGE_PY,
    SHIGURE_HISTORY_RECORDING_ENABLE,
    TAKEN_OBJECT_DETECTION_STAGE_PY,
    TAKEN_OBJECT_DETECTION_STAGE_RUN,
    SAM3MASK_WORKER_IDLE_TIMEOUT_SEC,
    WORKER_SOCKET_ROOT,
)
from stages.hololens3d_reconstruction.settings import OBJECT_ALIGNMENT_MODE
from task_db import (
    create_task as create_task_record,
    get_completed_tasks_for_startup,
    get_latest_completed_task,
    get_task_stage_runs,
    get_task_timing_events,
    get_ai_model_timings_for_task,
    get_task_by_task_id,
    get_unfinished_tasks,
    get_unsynced_completed_tasks,
    initialize_task_table,
    mark_task_stage_completed,
    mark_task_stage_failed,
    mark_task_stage_started,
    record_ai_model_timing,
    update_task_status,
)
from task_json import (
    ensure_task_id_in_json,
    load_task_json,
    resolve_task_json_path,
    save_task_json,
)
from console_output_log import install_console_output_log

install_console_output_log()

try:
    from gpu_budget import cuda_env_for_service
except Exception:
    cuda_env_for_service = None


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
    "display_identity",
    "history_placement_restoration",
    "taken_object_detection",
    "sam3d_body_mesh",
]

PURPOSE_OBJECT_RECONSTRUCTION = "object_reconstruction"
PURPOSE_ARUCO_REFERENCE = "aruco_reference"


@dataclass(frozen=True)
class StageWorkerContext:
    stage_name: str
    worker_index: int = 0
    gpu_id: str | None = None


class SocketStageService:
    def __init__(
        self,
        *,
        name: str,
        python_path: str,
        script_path: Path,
        cwd: Path,
        socket_name: str,
        idle_timeout_sec: int,
        echo_output: bool,
        env_overrides: Mapping[str, str] | Callable[[], Mapping[str, str] | None] | None = None,
    ) -> None:
        self.name = name
        self.python_path = python_path
        self.script_path = script_path
        self.cwd = cwd
        self.socket_path = WORKER_SOCKET_ROOT / socket_name
        self.idle_timeout_sec = max(1, int(idle_timeout_sec or 300))
        self.echo_output = bool(echo_output)
        self.env_overrides = env_overrides
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._output_tail: deque[str] = deque(maxlen=80)
        self._active_requests = 0
        self._last_used_at = 0.0

    def ensure_started(
        self,
        *,
        task_id: str | None = None,
        stage_name: str | None = None,
        reason: str | None = None,
    ) -> None:
        started = False
        start_wall = time.time()
        start_perf = time.perf_counter()
        status = "completed"
        error_message = None
        detail: dict[str, Any] = {
            "socket_path": str(self.socket_path),
            "script_path": str(self.script_path),
            "cwd": str(self.cwd),
        }
        if reason:
            detail["reason"] = reason
        try:
            with self._lock:
                if self._is_running_locked() and self.socket_path.exists():
                    return
                started = True
                self._stop_locked()
                WORKER_SOCKET_ROOT.mkdir(parents=True, exist_ok=True)
                try:
                    self.socket_path.unlink()
                except FileNotFoundError:
                    pass

                env = os.environ.copy()
                env.update(self._resolve_env_overrides())
                env.setdefault("PYTHONUNBUFFERED", "1")
                command = [
                    _resolve_python(self.python_path),
                    str(self.script_path),
                    "--socket-server",
                    str(self.socket_path),
                ]
                detail["command"] = command
                self._process = subprocess.Popen(
                    command,
                    cwd=str(self.cwd),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                detail["pid"] = self._process.pid
                self._reader_thread = threading.Thread(
                    target=self._consume_output,
                    args=(self._process,),
                    daemon=True,
                    name=f"{self.name}-log-reader",
                )
                self._reader_thread.start()
                self._last_used_at = time.monotonic()

            self._wait_for_socket_ready()
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            raise
        finally:
            if started:
                try:
                    record_ai_model_timing(
                        service_name=self.name,
                        timing_kind="initialization",
                        stage_name=stage_name,
                        task_id=task_id,
                        status=status,
                        duration_ms=(time.perf_counter() - start_perf) * 1000.0,
                        started_at_unix=start_wall,
                        completed_at_unix=time.time(),
                        detail=detail,
                        error_message=error_message,
                    )
                except Exception as db_exc:
                    print(f"[worker] failed to record {self.name} initialization timing: {db_exc}")

    def _resolve_env_overrides(self) -> dict[str, str]:
        if self.env_overrides is None:
            return {}
        if callable(self.env_overrides):
            try:
                return dict(self.env_overrides() or {})
            except Exception as exc:
                print(f"[worker] failed to resolve {self.name} env overrides: {exc}")
                return {}
        return dict(self.env_overrides)

    def request(
        self,
        payload: dict[str, Any],
        *,
        task_id: str | None = None,
        stage_name: str | None = None,
    ) -> dict[str, Any]:
        self.ensure_started(task_id=task_id, stage_name=stage_name, reason="request")
        start_wall = time.time()
        start_perf = time.perf_counter()
        status = "completed"
        error_message = None
        response: dict[str, Any] | None = None
        with self._lock:
            self._active_requests += 1
            self._last_used_at = time.monotonic()
        try:
            response = _send_socket_request(self.socket_path, payload)
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or f"{self.name} worker failed"))
            return response
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - start_perf) * 1000.0
            detail: dict[str, Any] = {
                "socket_path": str(self.socket_path),
                "payload_keys": sorted(str(key) for key in payload.keys()),
            }
            if "action" in payload:
                detail["action"] = payload.get("action")
            if response is not None and isinstance(response.get("timings"), dict):
                detail["worker_timings"] = response.get("timings")
            try:
                record_ai_model_timing(
                    service_name=self.name,
                    timing_kind="task",
                    stage_name=stage_name,
                    task_id=task_id,
                    status=status,
                    duration_ms=duration_ms,
                    started_at_unix=start_wall,
                    completed_at_unix=time.time(),
                    detail=detail,
                    error_message=error_message,
                )
            except Exception as db_exc:
                print(f"[worker] failed to record {self.name} task timing: {db_exc}")
            with self._lock:
                self._active_requests = max(0, self._active_requests - 1)
                self._last_used_at = time.monotonic()

    def maybe_stop_idle(self, *, keep_alive: bool) -> None:
        with self._lock:
            if keep_alive or self._active_requests > 0 or not self._is_running_locked():
                return
            if time.monotonic() - self._last_used_at >= self.idle_timeout_sec:
                print(f"[worker] stopping idle {self.name} service")
                self._stop_locked()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _wait_for_socket_ready(self) -> None:
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            with self._lock:
                process = self._process
                if process is None:
                    raise RuntimeError(f"{self.name} worker did not start")
                return_code = process.poll()
                if return_code is not None:
                    tail = "\n".join(self._output_tail)
                    raise RuntimeError(f"{self.name} worker exited with {return_code}\n{tail}")
            if self.socket_path.exists():
                return
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for {self.name} worker socket: {self.socket_path}")

    def _consume_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            clean = line.rstrip("\n")
            with self._lock:
                self._output_tail.append(clean)
            if self.echo_output:
                print(f"[{self.name}] {line}", end="", flush=True)

    def _is_running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _stop_locked(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                _send_socket_request(self.socket_path, {"action": "shutdown"}, timeout=2.0)
            except Exception:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        self._process = None
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass


def _send_socket_request(
    socket_path: Path,
    payload: dict[str, Any],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        if timeout is not None:
            client.settimeout(timeout)
        client.connect(str(socket_path))
        client.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError(f"No response from socket worker: {socket_path}")
    return json.loads(b"".join(chunks).decode("utf-8").splitlines()[0])


def _resolve_python(python_path: str) -> str:
    return python_path or sys.executable


def _stage_queue_names() -> list[str]:
    return ["aruco_detect"] + list(STAGE_ORDER)


_stage_queues: dict[str, deque[str]] = {stage: deque() for stage in _stage_queue_names()}
_queued_task_ids: set[str] = set()
_running_task_ids: set[str] = set()
_running_task_stages: dict[str, str] = {}
_task_lock = threading.Lock()
_worker_thread: Optional[threading.Thread] = None
_worker_threads: list[threading.Thread] = []
_service_monitor_thread: Optional[threading.Thread] = None
_shigure_recorder_process: subprocess.Popen[str] | None = None
_shigure_recorder_reader_thread: threading.Thread | None = None
_last_shigure_recorder_start_attempt_at = 0.0
_service_prewarm_thread: threading.Thread | None = None
_shutdown_requested = False
_shutdown_hooks_installed = False
_restore_scan_lock = threading.Lock()
_last_restore_scan_at = 0.0



def _consume_shigure_history_recorder_output(process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        print(f"[shigure-recorder] {line}", end="", flush=True)


def _start_shigure_history_recorder(*, force: bool = False) -> None:
    global _shigure_recorder_process, _shigure_recorder_reader_thread, _last_shigure_recorder_start_attempt_at
    if _shutdown_requested or not SHIGURE_HISTORY_RECORDING_ENABLE:
        return
    if _shigure_recorder_process is not None and _shigure_recorder_process.poll() is None:
        return

    now = time.monotonic()
    if not force and now - _last_shigure_recorder_start_attempt_at < 30.0:
        return
    _last_shigure_recorder_start_attempt_at = now

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("SHIGURE_HISTORY_CACHE_ROOT", str(SHIGURE_HISTORY_CACHE_ROOT))
    command = [
        _resolve_python(SHIGURE_HISTORY_RECORDER_STAGE_PY),
        str(SHIGURE_HISTORY_RECORDER_RUN),
        "--cache-root",
        str(SHIGURE_HISTORY_CACHE_ROOT),
    ]
    try:
        _shigure_recorder_process = subprocess.Popen(
            command,
            cwd=str(SHIGURE_HISTORY_RECORDER_RUN.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _shigure_recorder_reader_thread = threading.Thread(
            target=_consume_shigure_history_recorder_output,
            args=(_shigure_recorder_process,),
            daemon=True,
            name="shigure-history-recorder-log-reader",
        )
        _shigure_recorder_reader_thread.start()
        print(f"[worker] started Shigurei history recorder: pid={_shigure_recorder_process.pid}")
    except Exception as exc:
        _shigure_recorder_process = None
        _shigure_recorder_reader_thread = None
        print(f"[worker] failed to start Shigurei history recorder: {exc}")


def _stop_shigure_history_recorder() -> None:
    global _shigure_recorder_process, _shigure_recorder_reader_thread
    process = _shigure_recorder_process
    if process is None:
        return
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        except Exception as exc:
            print(f"[worker] failed to stop Shigurei event recorder cleanly: {exc}")
    _shigure_recorder_process = None
    _shigure_recorder_reader_thread = None


def _service_gpu_env(service_name: str, allowed_ids: tuple[str, ...] | None = None) -> dict[str, str]:
    if cuda_env_for_service is None:
        return {}
    env, placement = cuda_env_for_service(service_name, allowed_ids=allowed_ids)
    required = placement.required_mib + placement.headroom_mib
    print(
        f"[worker] gpu placement {service_name}: {placement.reason} "
        f"(budget={placement.required_mib}MiB headroom={placement.headroom_mib}MiB required={required}MiB)"
    )
    if placement.gpu is not None and not placement.fits:
        print(f"[worker] warning: {service_name} may not fit on GPU {placement.gpu.index}")
    return dict(env)


_sam3mask_service = SocketStageService(
    name="sam3mask",
    python_path=SAM3_PY,
    script_path=SAM3_BOX_MASK_RUN,
    cwd=SAM3_DIR,
    socket_name="sam3mask.sock",
    idle_timeout_sec=SAM3MASK_WORKER_IDLE_TIMEOUT_SEC,
    echo_output=True,
    env_overrides=lambda: _service_gpu_env("sam3_image_mask"),
)
_instantmesh_service = SocketStageService(
    name="instantmesh",
    python_path=INSTANTMESH_STAGE_PY,
    script_path=INSTANTMESH_STAGE_RUN,
    cwd=INSTANTMESH_STAGE_RUN.parent,
    socket_name="instantmesh.sock",
    idle_timeout_sec=INSTANTMESH_WORKER_IDLE_TIMEOUT_SEC,
    echo_output=True,
    env_overrides=lambda: _service_gpu_env("instantmesh", INSTANTMESH_GPU_IDS or None),
)
_foundationpose_service = SocketStageService(
    name="foundationpose",
    python_path=FOUNDATIONPOSE_ALIGNMENT_PY,
    script_path=FOUNDATIONPOSE_ALIGNMENT_RUN,
    cwd=FOUNDATIONPOSE_ALIGNMENT_RUN.parent,
    socket_name="foundationpose.sock",
    idle_timeout_sec=FOUNDATIONPOSE_WORKER_IDLE_TIMEOUT_SEC,
    echo_output=False,
    env_overrides=lambda: _service_gpu_env("foundationpose"),
)


def _queue_snapshot_no_lock() -> list[str]:
    snapshot: list[str] = []
    for queue in _stage_queues.values():
        snapshot.extend(queue)
    return snapshot


def _enqueue_stage_task_no_lock(task_id: str, stage_name: str, *, front: bool = False) -> None:
    task_id = str(task_id)
    stage_name = str(stage_name)
    if stage_name not in STAGE_RUNNERS:
        raise ValueError(f"Unsupported stage for queue: {stage_name}")
    if task_id in _queued_task_ids or task_id in _running_task_ids:
        return
    queue = _stage_queues[stage_name]
    if front:
        queue.appendleft(task_id)
    else:
        queue.append(task_id)
    _queued_task_ids.add(task_id)


def _enqueue_task_no_lock(
    task_id: str,
    purpose: str,
    *,
    front: bool = False,
    task_json: dict | None = None,
    status: str | None = None,
) -> None:
    if purpose == PURPOSE_ARUCO_REFERENCE:
        _enqueue_stage_task_no_lock(task_id, "aruco_detect", front=front)
        return

    stage_order = _resolve_stage_order(task_json or {})
    current_status = str(status or "pending")
    stage_name = stage_order[0] if current_status == "pending" else current_status
    if stage_name in stage_order:
        _enqueue_stage_task_no_lock(task_id, stage_name, front=front)


def _restore_unfinished_tasks() -> None:
    global _last_restore_scan_at
    now = time.monotonic()
    with _restore_scan_lock:
        if now - _last_restore_scan_at < 1.0:
            return
        _last_restore_scan_at = now
    unfinished_tasks = get_unfinished_tasks()
    for task in unfinished_tasks:
        task_id = str(task["task_id"])
        with _task_lock:
            if task_id in _queued_task_ids or task_id in _running_task_ids:
                continue
        try:
            task_json = load_task_json(resolve_task_json_path(task["json_path"]))
            purpose = _resolve_task_purpose(task_json)
        except Exception:
            task_json = {}
            purpose = PURPOSE_OBJECT_RECONSTRUCTION
        with _task_lock:
            _enqueue_task_no_lock(
                task_id,
                purpose,
                task_json=task_json,
                status=str(task.get("status") or "pending"),
            )


def _prewarm_model_pipeline_services(reason: str) -> None:
    global _service_prewarm_thread
    if _shutdown_requested or not MODEL_SERVICE_PREWARM_ENABLE:
        return

    def _start_service(service: SocketStageService) -> None:
        if _shutdown_requested:
            return
        try:
            print(f"[worker] prewarming {service.name} after {reason}")
            service.ensure_started(reason=f"prewarm:{reason}")
        except Exception as exc:
            print(f"[worker] failed to prewarm {service.name}: {exc}")

    def _run() -> None:
        services = (
            _instantmesh_service,
            _sam3mask_service,
            _foundationpose_service,
        )
        starters: list[threading.Thread] = []
        for service in services:
            thread = threading.Thread(
                target=_start_service,
                args=(service,),
                daemon=True,
                name=f"prewarm-{service.name}",
            )
            thread.start()
            starters.append(thread)
        for thread in starters:
            thread.join()

    with _task_lock:
        if _service_prewarm_thread is not None and _service_prewarm_thread.is_alive():
            return
        _service_prewarm_thread = threading.Thread(
            target=_run,
            daemon=True,
            name="model-pipeline-service-prewarm",
        )
        _service_prewarm_thread.start()


def _task_id_from_json_path(json_path: Path) -> str:
    try:
        task_json = load_task_json(json_path)
        raw = task_json.get("task_id") or task_json.get("task_name")
        if raw:
            return str(raw)
    except Exception:
        pass
    return json_path.stem


def _record_worker_internal_model_init(
    *,
    service_name: str,
    stage_name: str,
    task_id: str,
    response: dict[str, Any],
) -> None:
    timings = response.get("timings") if isinstance(response, dict) else None
    if not isinstance(timings, dict):
        return
    model_load = timings.get("model_load")
    if not isinstance(model_load, dict) or not model_load.get("loaded_now"):
        return
    duration_ms = model_load.get("duration_ms")
    if duration_ms is None:
        return
    try:
        record_ai_model_timing(
            service_name=service_name,
            timing_kind="initialization",
            stage_name=stage_name,
            task_id=task_id,
            duration_ms=float(duration_ms),
            detail={"source": "worker_internal_model_load", **model_load},
        )
    except Exception as exc:
        print(f"[worker] failed to record {service_name} internal model init timing: {exc}")


def _run_python_script(
    python_path: str,
    script_path: Path,
    json_path: Path,
    cwd: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    stream_command(
        [_resolve_python(python_path), str(script_path), str(json_path)],
        cwd=cwd,
        env=env,
        check=True,
    )


def _run_hololens2depth(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=HOLOLENS2_PY,
        script_path=HOLOLENS2_CONVERT_RUN,
        json_path=json_path,
        cwd=HOLOLENS2_CONVERT_DIR,
    )


def _run_aruco_detect(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=ARUCO_DETECT_STAGE_PY,
        script_path=ARUCO_DETECT_STAGE_RUN,
        json_path=json_path,
        cwd=ARUCO_DETECT_STAGE_RUN.parent,
    )


def _run_sam3mask(json_path: Path, context: StageWorkerContext | None = None) -> None:
    task_id = _task_id_from_json_path(json_path)
    response = _sam3mask_service.request(
        {"json_path": str(json_path)},
        task_id=task_id,
        stage_name="sam3mask",
    )
    _record_worker_internal_model_init(
        service_name="sam3mask",
        stage_name="sam3mask",
        task_id=task_id,
        response=response,
    )


def _instantmesh_env(context: StageWorkerContext | None) -> dict[str, str] | None:
    if MODEL_GENERATION_BACKEND == "sam3d_objects":
        selected = _service_gpu_env("sam3d_objects")
        if selected:
            env = os.environ.copy()
            env.update(selected)
            return env
    if context is None or not context.gpu_id:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(context.gpu_id)
    return env


def _run_model_generation(json_path: Path, context: StageWorkerContext | None = None) -> None:
    env = _instantmesh_env(context)
    task_id = _task_id_from_json_path(json_path)
    if MODEL_GENERATION_BACKEND == "sam3d_objects":
        start_wall = time.time()
        start_perf = time.perf_counter()
        status = "completed"
        error_message = None
        try:
            _run_python_script(
                python_path=SAM3D_OBJECTS_STAGE_PY,
                script_path=SAM3D_OBJECTS_STAGE_RUN,
                json_path=json_path,
                cwd=SAM3D_OBJECTS_ROOT,
                env=env,
            )
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            raise
        finally:
            try:
                record_ai_model_timing(
                    service_name="sam3d_objects",
                    timing_kind="task",
                    stage_name="instantmesh",
                    task_id=task_id,
                    status=status,
                    duration_ms=(time.perf_counter() - start_perf) * 1000.0,
                    started_at_unix=start_wall,
                    completed_at_unix=time.time(),
                    detail={"backend": "sam3d_objects", "script_path": str(SAM3D_OBJECTS_STAGE_RUN)},
                    error_message=error_message,
                )
            except Exception as db_exc:
                print(f"[worker] failed to record sam3d_objects task timing: {db_exc}")
        return

    _instantmesh_service.ensure_started(task_id=task_id, stage_name="instantmesh", reason="instantmesh stage")
    _prewarm_model_pipeline_services("instantmesh start")
    _instantmesh_service.request(
        {"json_path": str(json_path)},
        task_id=task_id,
        stage_name="instantmesh",
    )


def _run_instantmesh(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_model_generation(json_path, context)


def _run_depthpointcloud(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=DEPTHPOINTCLOUD_STAGE_PY,
        script_path=DEPTHPOINTCLOUD_STAGE_RUN,
        json_path=json_path,
        cwd=DEPTHPOINTCLOUD_STAGE_RUN.parent,
    )


def _run_modelscale(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=MODELSCALE_STAGE_PY,
        script_path=MODELSCALE_STAGE_RUN,
        json_path=json_path,
        cwd=MODELSCALE_STAGE_RUN.parent,
    )


def _run_object_alignment(json_path: Path, context: StageWorkerContext | None = None) -> None:
    env = None
    if OBJECT_ALIGNMENT_MODE == "foundationpose":
        task_id = _task_id_from_json_path(json_path)
        _foundationpose_service.ensure_started(
            task_id=task_id,
            stage_name="object_alignment",
            reason="foundationpose object alignment",
        )
        env = os.environ.copy()
        env["FOUNDATIONPOSE_WORKER_SOCKET"] = str(_foundationpose_service.socket_path)
    _run_python_script(
        python_path=OBJECT_ALIGNMENT_STAGE_PY,
        script_path=OBJECT_ALIGNMENT_STAGE_RUN,
        json_path=json_path,
        cwd=OBJECT_ALIGNMENT_STAGE_RUN.parent,
        env=env,
    )


def _run_pose(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=POSE_STAGE_PY,
        script_path=POSE_STAGE_RUN,
        json_path=json_path,
        cwd=POSE_STAGE_RUN.parent,
    )


def _run_aruco_sync(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=ARUCO_SYNC_STAGE_PY,
        script_path=ARUCO_SYNC_STAGE_RUN,
        json_path=json_path,
        cwd=ARUCO_SYNC_STAGE_RUN.parent,
    )


def _run_runtime_mesh(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=RUNTIME_MESH_STAGE_PY,
        script_path=RUNTIME_MESH_STAGE_RUN,
        json_path=json_path,
        cwd=RUNTIME_MESH_STAGE_RUN.parent,
    )


def _run_blender(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=BLENDER_STAGE_PY,
        script_path=BLENDER_STAGE_RUN,
        json_path=json_path,
        cwd=BLENDER_STAGE_RUN.parent,
    )



def _run_model_bounds(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=MODEL_BOUNDS_STAGE_PY,
        script_path=MODEL_BOUNDS_STAGE_RUN,
        json_path=json_path,
        cwd=MODEL_BOUNDS_STAGE_RUN.parent,
    )


def _run_display_identity(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=DISPLAY_IDENTITY_STAGE_PY,
        script_path=DISPLAY_IDENTITY_STAGE_RUN,
        json_path=json_path,
        cwd=DISPLAY_IDENTITY_STAGE_RUN.parent,
    )


def _run_history_placement_restoration(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=HISTORY_PLACEMENT_RESTORATION_STAGE_PY,
        script_path=HISTORY_PLACEMENT_RESTORATION_STAGE_RUN,
        json_path=json_path,
        cwd=HISTORY_PLACEMENT_RESTORATION_STAGE_RUN.parent,
    )


def _run_taken_object_detection(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=TAKEN_OBJECT_DETECTION_STAGE_PY,
        script_path=TAKEN_OBJECT_DETECTION_STAGE_RUN,
        json_path=json_path,
        cwd=TAKEN_OBJECT_DETECTION_STAGE_RUN.parent,
    )


def _run_sam3d_body_mesh(json_path: Path, context: StageWorkerContext | None = None) -> None:
    _run_python_script(
        python_path=SAM3D_BODY_MESH_STAGE_PY,
        script_path=SAM3D_BODY_MESH_STAGE_RUN,
        json_path=json_path,
        cwd=SAM3D_BODY_MESH_STAGE_RUN.parent,
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
    "display_identity": _run_display_identity,
    "history_placement_restoration": _run_history_placement_restoration,
    "taken_object_detection": _run_taken_object_detection,
    "sam3d_body_mesh": _run_sam3d_body_mesh,
}


def _resolve_task_purpose(task_json: dict) -> str:
    purpose = str(task_json.get("purpose") or "").strip()
    return purpose or PURPOSE_OBJECT_RECONSTRUCTION


def _resolve_stage_order(task_json: dict) -> list[str]:
    purpose = _resolve_task_purpose(task_json)
    if purpose == PURPOSE_ARUCO_REFERENCE:
        return ["aruco_detect"]
    return STAGE_ORDER


def _dequeue_stage_task(stage_name: str) -> str | None:
    with _task_lock:
        queue = _stage_queues[stage_name]
        if not queue:
            return None
        task_id = queue.popleft()
        _queued_task_ids.discard(task_id)
        _running_task_ids.add(task_id)
        _running_task_stages[task_id] = stage_name
        return task_id


def _finish_running_task(task_id: str) -> None:
    with _task_lock:
        _running_task_ids.discard(task_id)
        _running_task_stages.pop(task_id, None)


def _process_stage_task(task_id: str, expected_stage: str, context: StageWorkerContext) -> None:
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

    if current_status != expected_stage:
        with _task_lock:
            _stage_queues[current_status].appendleft(task_id)
            _queued_task_ids.add(task_id)
        return

    stage_name = current_status
    start_index = stage_order.index(stage_name)
    worker_label = f"{stage_name}#{context.worker_index}"
    if context.gpu_id:
        worker_label += f"/gpu{context.gpu_id}"
    print(f"[worker] start {worker_label}: {task_id}")

    update_task_status(task_id, stage_name)
    mark_task_stage_started(task_id, stage_name)
    try:
        STAGE_RUNNERS[stage_name](json_path, context)
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
        print(f"[worker] completed task: {task_id}")
        return

    next_status = "completed"
    if start_index + 1 < len(stage_order):
        next_status = stage_order[start_index + 1]
    update_task_status(task_id, next_status)
    if next_status == "completed":
        print(f"[worker] completed task: {task_id}")
    else:
        with _task_lock:
            _enqueue_stage_task_no_lock(task_id, next_status, front=False)
        print(f"[worker] stage completed, queued {next_status}: {task_id}")



def _stage_worker_loop(context: StageWorkerContext) -> None:
    while True:
        task_id = _dequeue_stage_task(context.stage_name)
        if task_id is None:
            _restore_unfinished_tasks()
            time.sleep(0.5)
            continue

        try:
            _process_stage_task(task_id, context.stage_name, context)
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
            _finish_running_task(task_id)


def _instantmesh_contexts() -> list[StageWorkerContext]:
    gpu_id = str(INSTANTMESH_GPU_IDS[0]) if INSTANTMESH_GPU_IDS else None
    return [StageWorkerContext("instantmesh", worker_index=0, gpu_id=gpu_id)]


def _worker_contexts() -> list[StageWorkerContext]:
    contexts: list[StageWorkerContext] = [StageWorkerContext("aruco_detect")]
    for stage_name in STAGE_ORDER:
        if stage_name == "instantmesh":
            contexts.extend(_instantmesh_contexts())
        else:
            contexts.append(StageWorkerContext(stage_name))
    return contexts


def _has_unfinished_at_or_before(stage_name: str) -> bool:
    try:
        target_index = STAGE_ORDER.index(stage_name)
    except ValueError:
        return False
    for task in get_unfinished_tasks():
        status = str(task.get("status") or "pending")
        if status == "aruco_detect":
            continue
        if status == "pending":
            status_index = 0
        elif status in STAGE_ORDER:
            status_index = STAGE_ORDER.index(status)
        else:
            continue
        if status_index <= target_index:
            return True
    return False


def _service_monitor_loop() -> None:
    while True:
        try:
            _start_shigure_history_recorder()
            _sam3mask_service.maybe_stop_idle(
                keep_alive=_has_unfinished_at_or_before("sam3mask"),
            )
            _instantmesh_service.maybe_stop_idle(
                keep_alive=(
                    MODEL_GENERATION_BACKEND == "instantmesh"
                    and _has_unfinished_at_or_before("instantmesh")
                ),
            )
            _foundationpose_service.maybe_stop_idle(
                keep_alive=(
                    OBJECT_ALIGNMENT_MODE == "foundationpose"
                    and _has_unfinished_at_or_before("object_alignment")
                ),
            )
        except Exception as exc:
            print(f"[worker] service monitor error: {exc}")
        time.sleep(5.0)


def shutdown_worker() -> None:
    """Stop persistent model services owned by this server process."""
    global _shutdown_requested
    _shutdown_requested = True
    _stop_shigure_history_recorder()
    for service in (_sam3mask_service, _instantmesh_service, _foundationpose_service):
        try:
            service.stop()
        except Exception as exc:
            print(f"[worker] failed to stop {service.name} service: {exc}")


def _install_shutdown_hooks() -> None:
    global _shutdown_hooks_installed
    if _shutdown_hooks_installed:
        return
    _shutdown_hooks_installed = True
    atexit.register(shutdown_worker)

    def _handle_signal(signum, frame):
        shutdown_worker()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(0)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, _handle_signal)
        except Exception:
            pass


def start_worker() -> threading.Thread:
    global _worker_thread, _service_monitor_thread

    _install_shutdown_hooks()
    initialize_task_table()
    _restore_unfinished_tasks()
    _start_shigure_history_recorder(force=True)

    if _worker_thread is not None and _worker_thread.is_alive():
        return _worker_thread

    for context in _worker_contexts():
        thread = threading.Thread(
            target=_stage_worker_loop,
            args=(context,),
            daemon=True,
            name=f"stage-{context.stage_name}-{context.worker_index}",
        )
        thread.start()
        _worker_threads.append(thread)


    _service_monitor_thread = threading.Thread(
        target=_service_monitor_loop,
        daemon=True,
        name="model-service-monitor",
    )
    _service_monitor_thread.start()
    _worker_threads.append(_service_monitor_thread)
    _worker_thread = _worker_threads[0]
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

    if _resolve_task_purpose(data) == PURPOSE_OBJECT_RECONSTRUCTION:
        _prewarm_model_pipeline_services("new 3D model task")

    with _task_lock:
        _enqueue_task_no_lock(task_id, _resolve_task_purpose(data), task_json=data)

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
    task_record["timing_events"] = get_task_timing_events(task_id)
    task_record["ai_model_timings"] = get_ai_model_timings_for_task(task_id)
    task_record["outputs"] = {
        "model_generation": task_json.get("ModelGeneration") or {},
        "sam3d_objects": task_json.get("SAM3DObjects") or {},
        "instantmesh": task_json.get("InstantMesh") or {},
        "runtime_mesh": task_json.get("RuntimeMesh") or {},
        "blender": task_json.get("Blender") or {},
        "display_identity": task_json.get("DisplayIdentity") or {},
        "history_placement_restoration": task_json.get("HistoryPlacementRestoration") or {},
        "taken_object_detection": task_json.get("TakenObjectDetection") or {},
        "sam3d_body_mesh": task_json.get("SAM3DBodyMesh") or {},
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
            _run_display_identity(resolved_json_path)
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
    task_id_str = str(task_record.get("task_id") or "")
    task_record["stage_runs"] = get_task_stage_runs(task_id_str)
    task_record["timing_events"] = get_task_timing_events(task_id_str)
    task_record["ai_model_timings"] = get_ai_model_timings_for_task(task_id_str)
    task_record["outputs"] = {
        "model_generation": task_json.get("ModelGeneration") or {},
        "sam3d_objects": task_json.get("SAM3DObjects") or {},
        "instantmesh": task_json.get("InstantMesh") or {},
        "runtime_mesh": task_json.get("RuntimeMesh") or {},
        "blender": task_json.get("Blender") or {},
        "display_identity": task_json.get("DisplayIdentity") or {},
        "history_placement_restoration": task_json.get("HistoryPlacementRestoration") or {},
        "taken_object_detection": task_json.get("TakenObjectDetection") or {},
        "sam3d_body_mesh": task_json.get("SAM3DBodyMesh") or {},
    }
    return task_record
