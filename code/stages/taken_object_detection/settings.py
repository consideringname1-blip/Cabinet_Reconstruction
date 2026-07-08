from __future__ import annotations

import os


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


# State-machine timing.
TRACKING_DURATION_SECONDS = float(os.environ.get("TAKEN_OBJECT_TRACKING_DURATION_SECONDS", "600"))
INIT_MAX_START_DELAY_SECONDS = float(os.environ.get("TAKEN_OBJECT_INIT_MAX_START_DELAY_SECONDS", "60"))
INIT_TIMEOUT_SECONDS = float(os.environ.get("TAKEN_OBJECT_INIT_TIMEOUT_SECONDS", "60"))
OFFLINE_STOP_WHEN_HISTORY_EXHAUSTED = _bool_env("TAKEN_OBJECT_OFFLINE_STOP_WHEN_HISTORY_EXHAUSTED", True)

# Depth thresholds, in meters.
DEPTH_MARGIN_M = float(os.environ.get("TAKEN_OBJECT_DEPTH_MARGIN_M", "0.12"))
OCCLUSION_DELTA_M = -abs(float(os.environ.get("TAKEN_OBJECT_OCCLUSION_DELTA_M", "0.12")))
TAKEN_DELTA_M = abs(float(os.environ.get("TAKEN_OBJECT_TAKEN_DELTA_M", "0.12")))

# Pixel ratios.
TRUSTED_MASK_MIN_IMAGE_RATIO = float(os.environ.get("TAKEN_OBJECT_TRUSTED_MASK_MIN_IMAGE_RATIO", "0.003"))
FULL_OCCLUSION_RATIO = float(os.environ.get("TAKEN_OBJECT_FULL_OCCLUSION_RATIO", "0.90"))
TAKEN_RATIO = float(os.environ.get("TAKEN_OBJECT_TAKEN_RATIO", "0.40"))
TAKEN_CONSECUTIVE_FRAMES = int(os.environ.get("TAKEN_OBJECT_TAKEN_CONSECUTIVE_FRAMES", "3"))
PARTIAL_OCCLUSION_CLOSER_RATIO = float(os.environ.get("TAKEN_OBJECT_PARTIAL_OCCLUSION_CLOSER_RATIO", "0.25"))
PARTIAL_OCCLUSION_UNCHANGED_RATIO = float(os.environ.get("TAKEN_OBJECT_PARTIAL_OCCLUSION_UNCHANGED_RATIO", "0.25"))
PARTIAL_OCCLUSION_DEEPER_MAX_RATIO = float(os.environ.get("TAKEN_OBJECT_PARTIAL_OCCLUSION_DEEPER_MAX_RATIO", "0.15"))

# RGB sample window used to attach the event frame artifacts.
RGB_BACKTRACK_SECONDS = float(os.environ.get("TAKEN_OBJECT_RGB_BACKTRACK_SECONDS", "5"))

# Output and fallback controls.
OUTPUT_MODE = os.environ.get("TAKEN_OBJECT_OUTPUT_MODE", "minimal_output")
FULL_OUTPUT = _bool_env("TAKEN_OBJECT_FULL_OUTPUT", OUTPUT_MODE == "full_output")
TRACKING_MODE = os.environ.get("TAKEN_OBJECT_TRACKING_MODE", "model_diag_circle").strip().lower()

# YOLO-first initialization and coarse tracking.
YOLO_PRE_CAPTURE_LOOKBACK_SECONDS = float(os.environ.get("TAKEN_OBJECT_YOLO_PRE_CAPTURE_LOOKBACK_SECONDS", "120"))
YOLO_MOVE_CENTER_PX = float(os.environ.get("TAKEN_OBJECT_YOLO_MOVE_CENTER_PX", "40"))
YOLO_MOVE_CENTER_BBOX_RATIO = float(os.environ.get("TAKEN_OBJECT_YOLO_MOVE_CENTER_BBOX_RATIO", "0.15"))
YOLO_MOVE_STRONG_CENTER_PX = float(os.environ.get("TAKEN_OBJECT_YOLO_MOVE_STRONG_CENTER_PX", "80"))
YOLO_MOVE_STRONG_BBOX_RATIO = float(os.environ.get("TAKEN_OBJECT_YOLO_MOVE_STRONG_BBOX_RATIO", "0.30"))
YOLO_MOVE_STABLE_UNIQUE_COUNT = int(os.environ.get("TAKEN_OBJECT_YOLO_MOVE_STABLE_UNIQUE_COUNT", "2"))
YOLO_TRACKING_TRIGGER_MODE = os.environ.get("TAKEN_OBJECT_YOLO_TRACKING_TRIGGER_MODE", "direct_depth_scan").strip().lower()

# Model diagonal-circle initialization.
MODEL_DIAG_CIRCLE_MIN_MASK_INSIDE_RATIO = float(os.environ.get("TAKEN_OBJECT_MODEL_DIAG_CIRCLE_MIN_MASK_INSIDE_RATIO", "0.80"))
MODEL_DIAG_CIRCLE_MAX_DEPTH_DIFF_M = float(os.environ.get("TAKEN_OBJECT_MODEL_DIAG_CIRCLE_MAX_DEPTH_DIFF_M", "0.18"))
