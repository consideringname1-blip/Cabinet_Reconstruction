from __future__ import annotations

import os
from pathlib import Path

try:
    from artifact_layout import SHIGURE_HISTORY_CACHE_ROOT as CONFIG_SHIGURE_HISTORY_CACHE_ROOT
    from artifact_layout import WORKER_SOCKET_ROOT as CONFIG_WORKER_SOCKET_ROOT
except Exception:  # pragma: no cover - keeps this module usable in small tests.
    _project_root = Path(__file__).resolve().parents[3]
    CONFIG_SHIGURE_HISTORY_CACHE_ROOT = _project_root / "data" / "shigure_history_cache"
    CONFIG_WORKER_SOCKET_ROOT = _project_root / "data" / "worker_sockets"


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


SHIGURE_HISTORY_CACHE_ROOT = Path(
    os.environ.get("SHIGURE_HISTORY_CACHE_ROOT") or str(CONFIG_SHIGURE_HISTORY_CACHE_ROOT)
)
SHIGURE_HISTORY_SOCKET_PATH = Path(
    os.environ.get("SHIGURE_HISTORY_SOCKET_PATH") or str(CONFIG_WORKER_SOCKET_ROOT / "shigure_history.sock")
)

# Shigurei local online cache. Frames stay in recorder memory; this root only
# holds status/debug JSON and keeps the old cache constructor signature stable.
SHIGURE_HISTORY_SECONDS = _float_env("SHIGURE_HISTORY_SECONDS", 60.0)
SHIGURE_HISTORY_HZ = _float_env("SHIGURE_HISTORY_HZ", 5.0)
SHIGURE_HISTORY_MAX_SAMPLES = _int_env(
    "SHIGURE_HISTORY_MAX_SAMPLES",
    max(1, int(round(SHIGURE_HISTORY_SECONDS * SHIGURE_HISTORY_HZ))),
)
SHIGURE_HISTORY_RECORDER_LOG_INTERVAL = _float_env("SHIGURE_HISTORY_RECORDER_LOG_INTERVAL", 10.0)
SHIGURE_HISTORY_RGB_DEPTH_MAX_DELTA_SECONDS = _float_env("SHIGURE_HISTORY_RGB_DEPTH_MAX_DELTA_SECONDS", 0.2)
SHIGURE_HISTORY_OBJECT_DETECTION_MAX_DELTA_SECONDS = _float_env("SHIGURE_HISTORY_OBJECT_DETECTION_MAX_DELTA_SECONDS", 0.5)
SHIGURE_HISTORY_SOCKET_TIMEOUT_SECONDS = _float_env("SHIGURE_HISTORY_SOCKET_TIMEOUT_SECONDS", 30.0)

# Shigurei topic mapping.
RGB_TOPIC = os.environ.get("SHIGURE_HISTORY_RGB_TOPIC", "/rs/color/compressed")
RGB_TYPE = os.environ.get("SHIGURE_HISTORY_RGB_TYPE", "sensor_msgs/msg/CompressedImage")
DEPTH_TOPIC = os.environ.get("SHIGURE_HISTORY_DEPTH_TOPIC", "/rs/aligned_depth_to_color/compressedDepth")
DEPTH_TYPE = os.environ.get("SHIGURE_HISTORY_DEPTH_TYPE", "sensor_msgs/msg/CompressedImage")
CAMERA_INFO_TOPIC = os.environ.get("SHIGURE_HISTORY_CAMERA_INFO_TOPIC", "/rs/aligned_depth_to_color/cameraInfo")
CAMERA_INFO_TYPE = os.environ.get("SHIGURE_HISTORY_CAMERA_INFO_TYPE", "sensor_msgs/msg/CameraInfo")
OBJECT_DETECTION_TOPIC = os.environ.get("SHIGURE_HISTORY_OBJECT_DETECTION_TOPIC", "/shigure/object_detection")
OBJECT_DETECTION_TYPE = os.environ.get("SHIGURE_HISTORY_OBJECT_DETECTION_TYPE", "shigure_core_msgs/msg/DetectedObjectList")

# Compatibility aliases for downstream code that still calls these payloads
# "yolo" or "active_objects".
ACTIVE_OBJECTS_TOPIC = OBJECT_DETECTION_TOPIC
ACTIVE_OBJECTS_TYPE = OBJECT_DETECTION_TYPE

TOPIC_SPECS: dict[str, tuple[str, str]] = {
    "rgb": (RGB_TOPIC, RGB_TYPE),
    "depth": (DEPTH_TOPIC, DEPTH_TYPE),
    "camera_info": (CAMERA_INFO_TOPIC, CAMERA_INFO_TYPE),
    "object_detection": (OBJECT_DETECTION_TOPIC, OBJECT_DETECTION_TYPE),
}


# Shigurei ArMarker history settings are kept; only the offline RGB-D cache was removed.
SHIGURE_MARKER_HISTORY_WARMUP_ENABLE = os.environ.get("SHIGURE_MARKER_HISTORY_WARMUP_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off", ""}
SHIGURE_MARKER_HISTORY_TARGET_DETECTIONS = int(os.environ.get("SHIGURE_MARKER_HISTORY_TARGET_DETECTIONS", "5"))
SHIGURE_MARKER_HISTORY_MAX_ATTEMPTS = int(os.environ.get("SHIGURE_MARKER_HISTORY_MAX_ATTEMPTS", "80"))
SHIGURE_MARKER_HISTORY_MAX_REPROJECTION_ERROR_PX = _float_env("SHIGURE_MARKER_HISTORY_MAX_REPROJECTION_ERROR_PX", 5.0)
SHIGURE_MARKER_HISTORY_MIN_CORNER_AREA_PX = _float_env("SHIGURE_MARKER_HISTORY_MIN_CORNER_AREA_PX", 64.0)
