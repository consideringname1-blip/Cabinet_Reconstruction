from __future__ import annotations

import os
from pathlib import Path

try:
    from config import MODEL_EVENT_OUTPUT_ROOT as CONFIG_MODEL_EVENT_OUTPUT_ROOT
    from config import SHIGURE_EVENT_CACHE_ROOT as CONFIG_SHIGURE_EVENT_CACHE_ROOT
except Exception:  # pragma: no cover - keeps this module usable in small tests.
    CONFIG_MODEL_EVENT_OUTPUT_ROOT = Path("/workspace/data/output/model_events")
    CONFIG_SHIGURE_EVENT_CACHE_ROOT = Path("/workspace/data/shigure_event_cache")


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


MODEL_EVENT_OUTPUT_ROOT = Path(
    os.environ.get("MODEL_EVENT_OUTPUT_ROOT", str(CONFIG_MODEL_EVENT_OUTPUT_ROOT))
)
SHIGURE_EVENT_CACHE_ROOT = Path(
    os.environ.get("SHIGURE_EVENT_CACHE_ROOT", str(CONFIG_SHIGURE_EVENT_CACHE_ROOT))
)

# Shigurei local history target: ten minutes at 5 Hz. This lets model-event
# tracking replay from cached history after model generation finishes.
SHIGURE_HISTORY_SECONDS = _float_env("SHIGURE_HISTORY_SECONDS", 600.0)
SHIGURE_HISTORY_HZ = _float_env("SHIGURE_HISTORY_HZ", 5.0)

# FBX depth-template rendering. The render is cropped to the projected model
# box, so this is a one-time CPU cost per completed model.
MODEL_DEPTH_RENDER_PADDING_PX = _int_env("MODEL_EVENT_DEPTH_RENDER_PADDING_PX", 8)
MODEL_DEPTH_MASK_ERODE_PX = _int_env("MODEL_EVENT_DEPTH_MASK_ERODE_PX", 0)
MODEL_DEPTH_MAX_RAY_HITS = _int_env("MODEL_EVENT_DEPTH_MAX_RAY_HITS", 32)
MODEL_DEPTH_RAY_EPSILON_M = _float_env("MODEL_EVENT_DEPTH_RAY_EPSILON_M", 0.0002)
MODEL_DEPTH_MIN_MODEL_PIXELS = _int_env("MODEL_EVENT_DEPTH_MIN_MODEL_PIXELS", 128)

# Foreground points are temporary occlusion. Only evaluable model pixels can
# vote that the object was removed. Removal is measured behind the rendered
# front surface; the default requires a 10 cm backward depth change.
MODEL_DEPTH_OCCLUSION_MARGIN_M = _float_env("MODEL_EVENT_DEPTH_OCCLUSION_MARGIN_M", 0.10)
MODEL_DEPTH_REMOVAL_MARGIN_M = _float_env("MODEL_EVENT_DEPTH_REMOVAL_MARGIN_M", 0.10)
MODEL_DEPTH_REMOVED_RATIO = _float_env("MODEL_EVENT_DEPTH_REMOVED_RATIO", 0.45)
MODEL_DEPTH_MAX_PRESENT_RATIO = _float_env("MODEL_EVENT_DEPTH_MAX_PRESENT_RATIO", 0.55)
MODEL_DEPTH_MIN_EVALUABLE_RATIO = _float_env("MODEL_EVENT_DEPTH_MIN_EVALUABLE_RATIO", 0.20)
MODEL_DEPTH_MIN_EVALUABLE_PIXELS = _int_env("MODEL_EVENT_DEPTH_MIN_EVALUABLE_PIXELS", 128)
MODEL_DEPTH_TAKEN_AWAY_STABLE_FRAMES = _int_env(
    "MODEL_EVENT_DEPTH_TAKEN_AWAY_STABLE_FRAMES", 3
)

# A short post-capture calibration absorbs the remaining fixed marker/depth
# bias without changing the trusted coordinate-transform contract.
MODEL_DEPTH_CALIBRATION_SECONDS = _float_env("MODEL_EVENT_DEPTH_CALIBRATION_SECONDS", 5.0)
MODEL_DEPTH_CALIBRATION_MAX_RESIDUAL_M = _float_env(
    "MODEL_EVENT_DEPTH_CALIBRATION_MAX_RESIDUAL_M", 0.25
)
MODEL_DEPTH_MAX_ABS_BIAS_M = _float_env("MODEL_EVENT_DEPTH_MAX_ABS_BIAS_M", 0.15)
MODEL_DEPTH_PRESENT_TOLERANCE_M = _float_env("MODEL_EVENT_DEPTH_PRESENT_TOLERANCE_M", 0.10)

# Both left and right wrists are inspected.
HAND_SCORE_MIN = _float_env("MODEL_EVENT_HAND_SCORE_MIN", 0.15)
HAND_BOX_MARGIN_M = _float_env("MODEL_EVENT_HAND_BOX_MARGIN_M", 0.03)
HAND_NEAREST_MAX_DISTANCE_M = _float_env("MODEL_EVENT_HAND_NEAREST_MAX_DISTANCE_M", 0.50)
HAND_LOOKBACK_SECONDS = _float_env("MODEL_EVENT_HAND_LOOKBACK_SECONDS", 5.0)
SKELETON_SEARCH_RADIUS_FRAMES = _int_env("MODEL_EVENT_SKELETON_SEARCH_RADIUS_FRAMES", 5)
