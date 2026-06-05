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

# Event-mask attempts are intentionally slower than the history recorder. The
# depth prefilter can skip heavily occluded frames before SAM3 runs.
MASK_ATTEMPT_HZ = _float_env("MODEL_EVENT_MASK_ATTEMPT_HZ", 1.0)
OCCLUSION_FRONT_MARGIN_M = _float_env("MODEL_EVENT_OCCLUSION_FRONT_MARGIN_M", 0.08)
OCCLUSION_FOREGROUND_RATIO = _float_env("MODEL_EVENT_OCCLUSION_FOREGROUND_RATIO", 0.35)
OCCLUSION_FOREGROUND_PRESENCE_RATIO = _float_env(
    "MODEL_EVENT_OCCLUSION_FOREGROUND_PRESENCE_RATIO", 0.03
)
OCCLUSION_FOREGROUND_CHANGE_RATIO = _float_env(
    "MODEL_EVENT_OCCLUSION_FOREGROUND_CHANGE_RATIO", 0.08
)
OCCLUSION_FOREGROUND_DEPTH_CHANGE_M = _float_env(
    "MODEL_EVENT_OCCLUSION_FOREGROUND_DEPTH_CHANGE_M", 0.04
)
OCCLUSION_STABLE_FRAMES = _int_env("MODEL_EVENT_OCCLUSION_STABLE_FRAMES", 2)
OCCLUSION_MODEL_DEPTH_BAND_M = _float_env("MODEL_EVENT_OCCLUSION_MODEL_DEPTH_BAND_M", 0.20)
REFERENCE_MODEL_DEPTH_RATIO = _float_env("MODEL_EVENT_REFERENCE_MODEL_DEPTH_RATIO", 0.08)
MASK_MIN_INSIDE_BOX_RATIO = _float_env("MODEL_EVENT_MASK_MIN_INSIDE_BOX_RATIO", 0.50)
MASK_MIN_MODEL_DEPTH_RATIO = _float_env("MODEL_EVENT_MASK_MIN_MODEL_DEPTH_RATIO", 0.20)
REQUIRE_HAND_CONTACT = os.environ.get("MODEL_EVENT_REQUIRE_HAND_CONTACT", "0").strip().lower() not in {"0", "false", "no", "off"}

# Hand contact / event-start policy.
HAND_SCORE_MIN = _float_env("MODEL_EVENT_HAND_SCORE_MIN", 0.15)
HAND_BOX_MARGIN_M = _float_env("MODEL_EVENT_HAND_BOX_MARGIN_M", 0.03)
HAND_NEAREST_MAX_DISTANCE_M = _float_env("MODEL_EVENT_HAND_NEAREST_MAX_DISTANCE_M", 0.35)

# Mask/depth movement policy. Defaults are deliberately conservative because
# the Shigurei depth stream and SAM tracking masks both jitter.
MOVEMENT_THRESHOLD_M = _float_env("MODEL_EVENT_MOVEMENT_THRESHOLD_M", 0.10)
DEPTH_STABLE_TOLERANCE_M = _float_env("MODEL_EVENT_DEPTH_STABLE_TOLERANCE_M", 0.06)
MIN_DEPTH_POINTS = _int_env("MODEL_EVENT_MIN_DEPTH_POINTS", 128)
MIN_OVERLAP_PIXELS = _int_env("MODEL_EVENT_MIN_OVERLAP_PIXELS", 256)
MIN_VISIBLE_AREA_RATIO = _float_env("MODEL_EVENT_MIN_VISIBLE_AREA_RATIO", 0.12)
COMPLETE_AREA_RATIO = _float_env("MODEL_EVENT_COMPLETE_AREA_RATIO", 0.72)
COMPLETE_IOU_MIN = _float_env("MODEL_EVENT_COMPLETE_IOU_MIN", 0.45)
TAKEN_AWAY_STABLE_FRAMES = _int_env("MODEL_EVENT_TAKEN_AWAY_STABLE_FRAMES", 2)

# SAM3 video tracker process lifetime.
SAM3_VIDEO_TRACKER_IDLE_TIMEOUT_SEC = _int_env("SAM3_VIDEO_TRACKER_IDLE_TIMEOUT_SEC", 300)
