from __future__ import annotations

import os
from pathlib import Path

from artifact_layout import WORKER_SOCKET_ROOT as CONFIG_WORKER_SOCKET_ROOT
from artifact_layout import SHIGURE_DEBUG_CACHE_ROOT as CONFIG_SHIGURE_DEBUG_CACHE_ROOT
from config import (
    SHIGURE_DEBUG_CACHE_ENABLE as MAIN_SHIGURE_DEBUG_CACHE_ENABLE,
    SHIGURE_DEBUG_CACHE_MAX_BYTES as MAIN_SHIGURE_DEBUG_CACHE_MAX_BYTES,
    SHIGURE_DEBUG_CACHE_MAX_ENTRIES as MAIN_SHIGURE_DEBUG_CACHE_MAX_ENTRIES,
    SHIGURE_DEBUG_CACHE_RETENTION_SECONDS as MAIN_SHIGURE_DEBUG_CACHE_RETENTION_SECONDS,
)

MAX_DEBUG_CACHE_RETENTION_SECONDS = 600.0


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


SHIGURE_HISTORY_SOCKET_PATH = Path(
    os.environ.get("SHIGURE_HISTORY_SOCKET_PATH") or str(CONFIG_WORKER_SOCKET_ROOT / "shigure_history.sock")
)

# Online RGB-D/canonical frames are memory-only and are exposed only through
# the Unix socket. The debug cache is a separate write-only diagnostic ring;
# runtime consumers must never use it as an input.
SHIGURE_HISTORY_SECONDS = _float_env("SHIGURE_HISTORY_SECONDS", 60.0)
SHIGURE_HISTORY_HZ = _float_env("SHIGURE_HISTORY_HZ", 5.0)
SHIGURE_DEBUG_CACHE_ENABLE = bool(MAIN_SHIGURE_DEBUG_CACHE_ENABLE)
SHIGURE_DEBUG_CACHE_ROOT = Path(
    os.environ.get("SHIGURE_DEBUG_CACHE_ROOT") or str(CONFIG_SHIGURE_DEBUG_CACHE_ROOT)
)
SHIGURE_DEBUG_CACHE_RETENTION_SECONDS = float(MAIN_SHIGURE_DEBUG_CACHE_RETENTION_SECONDS)
if not 0.0 < SHIGURE_DEBUG_CACHE_RETENTION_SECONDS <= MAX_DEBUG_CACHE_RETENTION_SECONDS:
    raise ValueError("SHIGURE_DEBUG_CACHE_RETENTION_SECONDS must be in (0, 600]")
SHIGURE_DEBUG_CACHE_MAX_ENTRIES = int(MAIN_SHIGURE_DEBUG_CACHE_MAX_ENTRIES)
if SHIGURE_DEBUG_CACHE_MAX_ENTRIES <= 0:
    raise ValueError("SHIGURE_DEBUG_CACHE_MAX_ENTRIES must be positive")
SHIGURE_DEBUG_CACHE_MAX_BYTES = int(MAIN_SHIGURE_DEBUG_CACHE_MAX_BYTES)
if SHIGURE_DEBUG_CACHE_MAX_BYTES <= 0:
    raise ValueError("SHIGURE_DEBUG_CACHE_MAX_BYTES must be positive")
SHIGURE_HISTORY_MAX_SAMPLES = _int_env(
    "SHIGURE_HISTORY_MAX_SAMPLES",
    max(1, int(round(SHIGURE_HISTORY_SECONDS * SHIGURE_HISTORY_HZ))),
)
SHIGURE_HISTORY_MAX_FRAMES = _int_env(
    "SHIGURE_HISTORY_MAX_FRAMES",
    max(256, int(round(SHIGURE_HISTORY_SECONDS * 30.0))),
)
SHIGURE_HISTORY_RECORDER_LOG_INTERVAL = _float_env("SHIGURE_HISTORY_RECORDER_LOG_INTERVAL", 10.0)
SHIGURE_HISTORY_RGB_DEPTH_MAX_DELTA_SECONDS = _float_env("SHIGURE_HISTORY_RGB_DEPTH_MAX_DELTA_SECONDS", 0.2)
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
OBJECT_TRACKING_TOPIC = os.environ.get("SHIGURE_HISTORY_OBJECT_TRACKING_TOPIC", "/shigure/object_tracking")
OBJECT_TRACKING_TYPE = os.environ.get("SHIGURE_HISTORY_OBJECT_TRACKING_TYPE", "shigure_core_msgs/msg/TrackedObjectList")
SEGMENTS_TOPIC = os.environ.get("SHIGURE_HISTORY_SEGMENTS_TOPIC", "/Segments")
SEGMENTS_TYPE = os.environ.get("SHIGURE_HISTORY_SEGMENTS_TYPE", "bboxes_ex_msgs/msg/Segments")
PEOPLE_TOPIC = os.environ.get("SHIGURE_HISTORY_PEOPLE_TOPIC", "/shigure/people_detection")
PEOPLE_TYPE = os.environ.get("SHIGURE_HISTORY_PEOPLE_TYPE", "shigure_core_msgs/msg/PoseKeyPointsList")
CONTACTED_TOPIC = os.environ.get("SHIGURE_HISTORY_CONTACTED_TOPIC", "/shigure/contacted")
CONTACTED_TYPE = os.environ.get("SHIGURE_HISTORY_CONTACTED_TYPE", "shigure_core_msgs/msg/ContactedList")

TOPIC_SPECS: dict[str, tuple[str, str]] = {
    "rgb": (RGB_TOPIC, RGB_TYPE),
    "depth": (DEPTH_TOPIC, DEPTH_TYPE),
    "camera_info": (CAMERA_INFO_TOPIC, CAMERA_INFO_TYPE),
    "object_detection": (OBJECT_DETECTION_TOPIC, OBJECT_DETECTION_TYPE),
    "object_tracking": (OBJECT_TRACKING_TOPIC, OBJECT_TRACKING_TYPE),
    "segments": (SEGMENTS_TOPIC, SEGMENTS_TYPE),
    "people": (PEOPLE_TOPIC, PEOPLE_TYPE),
    "contacted": (CONTACTED_TOPIC, CONTACTED_TYPE),
}

# Subscribe with BEST_EFFORT for every Shigure/RealSense topic.  A
# BEST_EFFORT reader is compatible with both BEST_EFFORT and RELIABLE writers,
# while a RELIABLE reader cannot connect to Shigure's BEST_EFFORT writers.
# Event topics still get a deeper local queue; the exact-stamp join keeps a
# dropped ContactedList as ``missing`` instead of treating it as an explicit
# empty contact result.
BEST_EFFORT_TOPIC_KEYS = frozenset(TOPIC_SPECS)
# RGB/depth are rebuild triggers too: if either arrives after a sparse event,
# the same canonical stamp is replayed so exact evidence can be attached.
CANONICAL_TOPIC_KEYS = frozenset(TOPIC_SPECS)


# Shigurei ArMarker history settings are kept; only the offline RGB-D cache was removed.
SHIGURE_MARKER_HISTORY_WARMUP_ENABLE = os.environ.get("SHIGURE_MARKER_HISTORY_WARMUP_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off", ""}
SHIGURE_MARKER_HISTORY_TARGET_DETECTIONS = int(os.environ.get("SHIGURE_MARKER_HISTORY_TARGET_DETECTIONS", "5"))
SHIGURE_MARKER_HISTORY_MAX_ATTEMPTS = int(os.environ.get("SHIGURE_MARKER_HISTORY_MAX_ATTEMPTS", "80"))
SHIGURE_MARKER_HISTORY_MAX_REPROJECTION_ERROR_PX = _float_env("SHIGURE_MARKER_HISTORY_MAX_REPROJECTION_ERROR_PX", 5.0)
SHIGURE_MARKER_HISTORY_MIN_CORNER_AREA_PX = _float_env("SHIGURE_MARKER_HISTORY_MIN_CORNER_AREA_PX", 64.0)
