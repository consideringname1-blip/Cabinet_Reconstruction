from __future__ import annotations

import os


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}

SAM3D_BODY_DETECTOR_NAME = os.environ.get("SAM3D_BODY_DETECTOR_NAME", "vitdet")
SAM3D_BODY_DEVICE = os.environ.get("SAM3D_BODY_DEVICE", "cuda")
SAM3D_BODY_BBOX_THRESHOLD = float(os.environ.get("SAM3D_BODY_BBOX_THRESHOLD", "0.35"))
SAM3D_BODY_NMS_THRESHOLD = float(os.environ.get("SAM3D_BODY_NMS_THRESHOLD", "0.3"))
SAM3D_BODY_INFERENCE_TYPE = os.environ.get("SAM3D_BODY_INFERENCE_TYPE", "full")

DEPTH_OVERLAP_RATIO = float(os.environ.get("SAM3D_BODY_DEPTH_OVERLAP_RATIO", "0.05"))
DEPTH_OVERLAP_PIXELS = int(os.environ.get("SAM3D_BODY_DEPTH_OVERLAP_PIXELS", "500"))
DEPTH_SAMPLE_MAX = int(os.environ.get("SAM3D_BODY_DEPTH_SAMPLE_MAX", "5000"))
BBOX_DEPTH_PAD_PX = float(os.environ.get("SAM3D_BODY_BBOX_DEPTH_PAD_PX", "8"))

WRIST_INDEXES = {"right_wrist": 41, "left_wrist": 62}
MAX_WRIST_DISTANCE_M = float(os.environ.get("SAM3D_BODY_MAX_WRIST_DISTANCE_M", "1.0"))

FBX_DECIMATE_RATIO = float(os.environ.get("SAM3D_BODY_FBX_DECIMATE_RATIO", "0.125"))
MATERIAL_COLOR = [0.0, 0.0, 0.0]
MATERIAL_ALPHA = float(os.environ.get("SAM3D_BODY_MATERIAL_ALPHA", "0.5"))

# Historical HoloLens display expects SAM3D body evidence mirrored on ArUco Z.
ARMARKER_FLIP_Z = _bool_env("SAM3D_BODY_ARMARKER_FLIP_Z", True)
