from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from path_config import PROJECT_ROOT


DATA_ROOT = PROJECT_ROOT / "data"
ENV_CONFIG_ROOT = DATA_ROOT / "config"
WORKER_SOCKET_ROOT = DATA_ROOT / "worker_sockets"
DATABASE_ROOT = DATA_ROOT / "database"
DATABASE_PATH = DATABASE_ROOT / "tasks.db"


def ensure_database_root() -> None:
    DATABASE_ROOT.mkdir(parents=True, exist_ok=True)


_console_output_log_root_raw = os.environ.get("CONSOLE_OUTPUT_LOG_ROOT")
CONSOLE_OUTPUT_LOG_ROOT = (
    Path(_console_output_log_root_raw).expanduser()
    if _console_output_log_root_raw
    else DATA_ROOT / "console_logs"
)
if not CONSOLE_OUTPUT_LOG_ROOT.is_absolute():
    CONSOLE_OUTPUT_LOG_ROOT = PROJECT_ROOT / CONSOLE_OUTPUT_LOG_ROOT

SHIGURE_HISTORY_CACHE_ROOT = DATA_ROOT / "shigure_history_cache"
SHIGURE_HISTORY_SOCKET_PATH = WORKER_SOCKET_ROOT / "shigure_history.sock"
ARUCO_DATA_ROOT = DATA_ROOT / "aruco"
ARUCO_REFERENCE_ROOT = ARUCO_DATA_ROOT / "reference"
ARUCO_RUNTIME_ROOT = ARUCO_DATA_ROOT / "runtime"
SHIGURE_MARKER_HISTORY_ROOT = ARUCO_DATA_ROOT / "shigure_marker_history"
SHIGURE_MARKER_HISTORY_PATH = SHIGURE_MARKER_HISTORY_ROOT / "latest_marker_6d_pose.json"
ARUCO_TEMPLATE_PATH = ARUCO_REFERENCE_ROOT / "aruco.json"
ARUCO_REFERENCE_MARKER_IMAGE_PATH = ARUCO_REFERENCE_ROOT / "ar_marker_7x7_1.png"

TASK_DATA_ROOT = DATA_ROOT / "model"
ARUCO_PROCESSING_ROOT = DATA_ROOT / "aruco_processing"
REALTIME_TRACKING_ROOT = DATA_ROOT / "realtime_tracking"

ARTIFACT_ROOT_DIRS = (
    DATA_ROOT,
    ENV_CONFIG_ROOT,
    WORKER_SOCKET_ROOT,
    DATABASE_ROOT,
    CONSOLE_OUTPUT_LOG_ROOT,
    SHIGURE_HISTORY_CACHE_ROOT,
    ARUCO_DATA_ROOT,
    ARUCO_REFERENCE_ROOT,
    ARUCO_RUNTIME_ROOT,
    SHIGURE_MARKER_HISTORY_ROOT,
    TASK_DATA_ROOT,
    ARUCO_PROCESSING_ROOT,
    REALTIME_TRACKING_ROOT,
)


def ensure_artifact_roots() -> None:
    for path in ARTIFACT_ROOT_DIRS:
        path.mkdir(parents=True, exist_ok=True)



TASK_WORKER_DIRNAME = "worker"
TASK_RESULT_DIRNAME = "result"
TASK_DEBUG_DIRNAME = "debug"
TASK_LOGS_DIRNAME = "logs"
TASK_JSON_FILENAME = "task.json"


def make_timestamp(dt: datetime | None = None) -> str:
    current = dt or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")


def model_task_dir(task_timestamp: str) -> Path:
    return TASK_DATA_ROOT / task_timestamp


def model_task_json_path(task_timestamp: str) -> Path:
    return model_task_dir(task_timestamp) / TASK_JSON_FILENAME


def model_worker_dir(task_timestamp: str) -> Path:
    return model_task_dir(task_timestamp) / TASK_WORKER_DIRNAME


def model_result_dir(task_timestamp: str) -> Path:
    return model_task_dir(task_timestamp) / TASK_RESULT_DIRNAME


def model_debug_dir(task_timestamp: str) -> Path:
    return model_task_dir(task_timestamp) / TASK_DEBUG_DIRNAME


def model_logs_dir(task_timestamp: str) -> Path:
    return model_task_dir(task_timestamp) / TASK_LOGS_DIRNAME


def aruco_processing_dir(task_timestamp: str) -> Path:
    return ARUCO_PROCESSING_ROOT / task_timestamp


def aruco_task_json_path(task_timestamp: str) -> Path:
    return aruco_processing_dir(task_timestamp) / TASK_JSON_FILENAME


def aruco_worker_dir(task_timestamp: str) -> Path:
    return aruco_processing_dir(task_timestamp) / TASK_WORKER_DIRNAME


def aruco_result_dir(task_timestamp: str) -> Path:
    return aruco_processing_dir(task_timestamp) / TASK_RESULT_DIRNAME


def aruco_debug_dir(task_timestamp: str) -> Path:
    return aruco_processing_dir(task_timestamp) / TASK_DEBUG_DIRNAME


def ensure_model_task_dirs(task_timestamp: str) -> None:
    for path in (
        model_worker_dir(task_timestamp),
        model_result_dir(task_timestamp),
        model_debug_dir(task_timestamp),
        model_logs_dir(task_timestamp),
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_aruco_task_dirs(task_timestamp: str) -> None:
    for path in (
        aruco_worker_dir(task_timestamp),
        aruco_result_dir(task_timestamp),
        aruco_debug_dir(task_timestamp),
    ):
        path.mkdir(parents=True, exist_ok=True)


MODEL_WORKER_FILES = {
    "input.color": "01_upload_color.png",
    "input.depth": "01_upload_depth.png",
    "input.meta": "01_upload_meta.json",
    "input.align_depth": "01_upload_align_depth.png",
    "sam3.mask": "02_sam3_mask.png",
    "sam3.color": "02_sam3_color.png",
    "sam3.depth": "02_sam3_depth.png",
    "model.source_obj": "03_model_source.obj",
    "model.source_mtl": "03_model_source.mtl",
    "model.source_texture": "03_model_source_texture.png",
    "generation.instantmesh_input": "03_generation_instantmesh_input.png",
    "generation.instantmesh_raw_obj": "03_generation_instantmesh_raw.obj",
    "generation.sam3d_raw_glb": "03_generation_sam3d_raw.glb",
    "generation.sam3d_postprocess": "03_generation_sam3d_postprocess.json",
    "model.runtime_obj": "04_runtime_mesh.obj",
    "model.runtime_mtl": "04_runtime_mesh.mtl",
    "model.runtime_texture": "04_runtime_mesh_texture.png",
}

MODEL_RESULT_FILES = {
    "model.final_fbx": "05_export_final.fbx",
    "identity.decision": "06_identity_decision.json",
    "taken.result": "07_taken_detection_result.json",
    "taken.result_rgb": "07_taken_detection_result_rgb.png",
    "taken.result_depth": "07_taken_detection_result_depth.png",
    "taken.object_mask": "07_taken_detection_object_mask.png",
    "taken.camera_info": "07_taken_detection_camera_info.json",
    "taken.object_detection": "07_taken_detection_object_detection.json",
    "taken.marker_pose": "07_taken_detection_marker_6d_pose.json",
    "body.result": "08_sam3d_body_result.json",
    "body.people": "08_sam3d_body_people.json",
    "body.selected_obj": "08_sam3d_body_selected_person.obj",
    "body.selected_fbx": "08_sam3d_body_selected_person.fbx",
    "body.subject_crop": "08_sam3d_body_subject_crop.png",
}

MODEL_DEBUG_FILES = {
    "depth.align_turbo": "01_depth_alignment_align_depth_turbo.png",
    "sam3.overlay": "02_sam3_mask_overlay.png",
    "generation.instantmesh_video": "03_generation_instantmesh_video.mp4",
}


def model_worker_file(task_timestamp: str, artifact_key: str) -> Path:
    return model_worker_dir(task_timestamp) / MODEL_WORKER_FILES[artifact_key]


def model_result_file(task_timestamp: str, artifact_key: str) -> Path:
    return model_result_dir(task_timestamp) / MODEL_RESULT_FILES[artifact_key]


def model_debug_file(task_timestamp: str, artifact_key: str) -> Path:
    return model_debug_dir(task_timestamp) / MODEL_DEBUG_FILES[artifact_key]

def sanitize_artifact_token(value: str | None, *, fallback: str) -> str:
    raw = str(value or "").strip() or fallback
    cleaned = []
    for char in raw:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        elif char in {".", ":", "/", "\\", " ", "T"}:
            cleaned.append("_")
    token = "".join(cleaned).strip("_")
    return token or fallback


def aruco_worker_frame_color(task_timestamp: str, frame_timestamp: str) -> Path:
    return aruco_worker_dir(task_timestamp) / f"{sanitize_artifact_token(frame_timestamp, fallback='frame')}_color.png"


def aruco_worker_frame_meta(task_timestamp: str, frame_timestamp: str) -> Path:
    return aruco_worker_dir(task_timestamp) / f"{sanitize_artifact_token(frame_timestamp, fallback='frame')}_meta.json"


def aruco_result_summary_file(task_timestamp: str) -> Path:
    return aruco_result_dir(task_timestamp) / "summary.json"


def aruco_result_marker_detect_file(task_timestamp: str, frame_timestamp: str) -> Path:
    return aruco_result_dir(task_timestamp) / f"{sanitize_artifact_token(frame_timestamp, fallback='frame')}_marker_detect.json"


def aruco_debug_overlay_file(task_timestamp: str, frame_timestamp: str) -> Path:
    return aruco_debug_dir(task_timestamp) / f"{sanitize_artifact_token(frame_timestamp, fallback='frame')}_aruco_debug_overlay.png"

