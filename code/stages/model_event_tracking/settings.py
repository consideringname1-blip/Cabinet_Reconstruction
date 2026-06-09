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

# Full foreground occlusion is not taken-away by itself.  It is kept as a
# possible interaction start; if the object is gone after occlusion clears, the
# event depth start is rewound to the first full-occlusion frame.
MODEL_DEPTH_FULL_OCCLUSION_RATIO = _float_env("MODEL_EVENT_DEPTH_FULL_OCCLUSION_RATIO", 0.80)
MODEL_DEPTH_FULL_OCCLUSION_MAX_EVALUABLE_RATIO = _float_env(
    "MODEL_EVENT_DEPTH_FULL_OCCLUSION_MAX_EVALUABLE_RATIO", 0.25
)
MODEL_DEPTH_FULL_OCCLUSION_STABLE_FRAMES = _int_env(
    "MODEL_EVENT_DEPTH_FULL_OCCLUSION_STABLE_FRAMES", 3
)
MODEL_DEPTH_FULL_OCCLUSION_MIN_PIXELS = _int_env(
    "MODEL_EVENT_DEPTH_FULL_OCCLUSION_MIN_PIXELS", 128
)
MODEL_DEPTH_OCCLUSION_CLEAR_EVALUABLE_RATIO = _float_env(
    "MODEL_EVENT_DEPTH_OCCLUSION_CLEAR_EVALUABLE_RATIO", 0.35
)
MODEL_DEPTH_OCCLUSION_CLEAR_MAX_OCCLUDED_RATIO = _float_env(
    "MODEL_EVENT_DEPTH_OCCLUSION_CLEAR_MAX_OCCLUDED_RATIO", 0.50
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

# After depth confirms taken-away, RGB motion is used only to look backward and
# choose the first visually useful interaction frame inside the model support.
RGB_MOTION_LOOKBACK_SECONDS = _float_env("MODEL_EVENT_RGB_MOTION_LOOKBACK_SECONDS", 5.0)
RGB_MOTION_BASELINE_FRAMES = _int_env("MODEL_EVENT_RGB_MOTION_BASELINE_FRAMES", 3)
RGB_MOTION_DIFF_THRESHOLD = _float_env("MODEL_EVENT_RGB_MOTION_DIFF_THRESHOLD", 28.0)
RGB_MOTION_START_RATIO = _float_env("MODEL_EVENT_RGB_MOTION_START_RATIO", 0.08)
RGB_MOTION_QUIET_RATIO = _float_env("MODEL_EVENT_RGB_MOTION_QUIET_RATIO", 0.03)
RGB_MOTION_CONFIRM_FRAMES = _int_env("MODEL_EVENT_RGB_MOTION_CONFIRM_FRAMES", 2)
RGB_MOTION_MIN_MASK_PIXELS = _int_env("MODEL_EVENT_RGB_MOTION_MIN_MASK_PIXELS", 128)
RGB_MOTION_DISPLAY_OFFSET_FRAMES = _int_env("MODEL_EVENT_RGB_MOTION_DISPLAY_OFFSET_FRAMES", 0)

# Optional SAM 3D Body mesh generated for HoloLens event display.  The mesh is
# projected into the event image panel and decimated before Unity downloads it.
SAM3D_BODY_EVENT_MESH_ENABLED = os.environ.get("MODEL_EVENT_SAM3D_BODY_MESH_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off", ""}
SAM3D_BODY_PY = os.environ.get("SAM3D_BODY_PY", "/opt/miniconda/envs/sam_3d_body/bin/python")
SAM3D_BODY_EVENT_MESH_SCRIPT = Path(
    os.environ.get(
        "MODEL_EVENT_SAM3D_BODY_MESH_SCRIPT",
        "/workspace/code/stages/model_event_tracking/generate_sam3d_body_event_mesh.py",
    )
)
SAM3D_BODY_EVENT_MESH_DECIMATE_RATIO = _float_env("MODEL_EVENT_SAM3D_BODY_MESH_DECIMATE_RATIO", 1.0 / 8.0)
SAM3D_BODY_EVENT_MESH_MAX_DISTANCE_M = _float_env("MODEL_EVENT_SAM3D_BODY_MESH_MAX_DISTANCE_M", 1.25)
SAM3D_BODY_EVENT_MESH_TIMEOUT_SEC = _float_env("MODEL_EVENT_SAM3D_BODY_MESH_TIMEOUT_SEC", 180.0)
