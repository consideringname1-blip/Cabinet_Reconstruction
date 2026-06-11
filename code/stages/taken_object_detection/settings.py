from __future__ import annotations

import os


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


# State-machine timing.
TRACKING_DURATION_SECONDS = float(os.environ.get("TAKEN_OBJECT_TRACKING_DURATION_SECONDS", "600"))
INIT_MAX_START_DELAY_SECONDS = float(os.environ.get("TAKEN_OBJECT_INIT_MAX_START_DELAY_SECONDS", "1.0"))
INIT_TIMEOUT_SECONDS = float(os.environ.get("TAKEN_OBJECT_INIT_TIMEOUT_SECONDS", "60"))
INIT_STABLE_WINDOW_SECONDS = float(os.environ.get("TAKEN_OBJECT_INIT_STABLE_WINDOW_SECONDS", "3"))

# Depth thresholds, in meters.
INIT_STABLE_DEPTH_DELTA_M = float(os.environ.get("TAKEN_OBJECT_INIT_STABLE_DEPTH_DELTA_M", "0.1"))
DEPTH_MARGIN_M = float(os.environ.get("TAKEN_OBJECT_DEPTH_MARGIN_M", "0.1"))
OCCLUSION_DELTA_M = -abs(float(os.environ.get("TAKEN_OBJECT_OCCLUSION_DELTA_M", "0.1")))
TAKEN_DELTA_M = abs(float(os.environ.get("TAKEN_OBJECT_TAKEN_DELTA_M", "0.1")))

# Pixel ratios.
INIT_STABLE_PIXEL_RATIO = float(os.environ.get("TAKEN_OBJECT_INIT_STABLE_PIXEL_RATIO", "0.95"))
TRUSTED_MASK_MIN_IMAGE_RATIO = float(os.environ.get("TAKEN_OBJECT_TRUSTED_MASK_MIN_IMAGE_RATIO", "0.003"))
FULL_OCCLUSION_RATIO = float(os.environ.get("TAKEN_OBJECT_FULL_OCCLUSION_RATIO", "0.90"))
TAKEN_RATIO = float(os.environ.get("TAKEN_OBJECT_TAKEN_RATIO", "0.65"))
TAKEN_CONSECUTIVE_FRAMES = int(os.environ.get("TAKEN_OBJECT_TAKEN_CONSECUTIVE_FRAMES", "3"))

# RGB backtracking.
RGB_BACKTRACK_SECONDS = float(os.environ.get("TAKEN_OBJECT_RGB_BACKTRACK_SECONDS", "5"))
RGB_ADJACENT_DIFF_THRESHOLD = float(os.environ.get("TAKEN_OBJECT_RGB_ADJACENT_DIFF_THRESHOLD", "18"))
RGB_INIT_DIFF_THRESHOLD = float(os.environ.get("TAKEN_OBJECT_RGB_INIT_DIFF_THRESHOLD", "24"))

# Output and fallback controls.
OUTPUT_MODE = os.environ.get("TAKEN_OBJECT_OUTPUT_MODE", "minimal_output")
FULL_OUTPUT = _bool_env("TAKEN_OBJECT_FULL_OUTPUT", OUTPUT_MODE == "full_output")
ALLOW_SELECTION_BOX_MASK_FALLBACK = _bool_env("TAKEN_OBJECT_ALLOW_SELECTION_BOX_MASK_FALLBACK", True)
