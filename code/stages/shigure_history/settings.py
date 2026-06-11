from __future__ import annotations

import os
from pathlib import Path

try:
    from config import SHIGURE_HISTORY_CACHE_ROOT as CONFIG_SHIGURE_HISTORY_CACHE_ROOT
except Exception:  # pragma: no cover - keeps this module usable in small tests.
    CONFIG_SHIGURE_HISTORY_CACHE_ROOT = Path('/workspace/data/shigure_history_cache')


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


SHIGURE_HISTORY_CACHE_ROOT = Path(
    os.environ.get('SHIGURE_HISTORY_CACHE_ROOT') or str(CONFIG_SHIGURE_HISTORY_CACHE_ROOT)
)

# Shigurei local RGB-D history target. These are stage-local capture settings;
# only the cache root directory convention stays in config.py.
SHIGURE_HISTORY_SECONDS = _float_env('SHIGURE_HISTORY_SECONDS', 600.0)
SHIGURE_HISTORY_HZ = _float_env('SHIGURE_HISTORY_HZ', 5.0)
SHIGURE_HISTORY_RECORDER_LOG_INTERVAL = _float_env('SHIGURE_HISTORY_RECORDER_LOG_INTERVAL', 10.0)

RGB_TOPIC = os.environ.get('SHIGURE_HISTORY_RGB_TOPIC', '/rs/color/compressed')
RGB_TYPE = os.environ.get('SHIGURE_HISTORY_RGB_TYPE', 'sensor_msgs/msg/CompressedImage')
DEPTH_TOPIC = os.environ.get('SHIGURE_HISTORY_DEPTH_TOPIC', '/rs/aligned_depth_to_color/compressedDepth')
DEPTH_TYPE = os.environ.get('SHIGURE_HISTORY_DEPTH_TYPE', 'sensor_msgs/msg/CompressedImage')
CAMERA_INFO_TOPIC = os.environ.get('SHIGURE_HISTORY_CAMERA_INFO_TOPIC', '/rs/aligned_depth_to_color/cameraInfo')
CAMERA_INFO_TYPE = os.environ.get('SHIGURE_HISTORY_CAMERA_INFO_TYPE', 'sensor_msgs/msg/CameraInfo')

TOPIC_SPECS: dict[str, tuple[str, str]] = {
    'rgb': (RGB_TOPIC, RGB_TYPE),
    'depth': (DEPTH_TOPIC, DEPTH_TYPE),
    'camera_info': (CAMERA_INFO_TOPIC, CAMERA_INFO_TYPE),
}


# Shigurei ArMarker history warmup. The recorder keeps RGB-D cache files free of
# marker/event data; these settings only control the separate global marker history.
SHIGURE_MARKER_HISTORY_WARMUP_ENABLE = os.environ.get('SHIGURE_MARKER_HISTORY_WARMUP_ENABLE', '1').strip().lower() not in {'0', 'false', 'no', 'off', ''}
SHIGURE_MARKER_HISTORY_TARGET_DETECTIONS = int(os.environ.get('SHIGURE_MARKER_HISTORY_TARGET_DETECTIONS', '5'))
SHIGURE_MARKER_HISTORY_MAX_ATTEMPTS = int(os.environ.get('SHIGURE_MARKER_HISTORY_MAX_ATTEMPTS', '80'))
SHIGURE_MARKER_HISTORY_MAX_REPROJECTION_ERROR_PX = float(os.environ.get('SHIGURE_MARKER_HISTORY_MAX_REPROJECTION_ERROR_PX', '5.0'))
SHIGURE_MARKER_HISTORY_MIN_CORNER_AREA_PX = float(os.environ.get('SHIGURE_MARKER_HISTORY_MIN_CORNER_AREA_PX', '64.0'))
