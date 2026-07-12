from __future__ import annotations

import os

SAM3D_BODY_DEVICE = os.environ.get("SAM3D_BODY_DEVICE", "cuda")
SAM3D_BODY_INFERENCE_TYPE = os.environ.get("SAM3D_BODY_INFERENCE_TYPE", "full")

DEPTH_OVERLAP_RATIO = float(os.environ.get("SAM3D_BODY_DEPTH_OVERLAP_RATIO", "0.05"))
DEPTH_OVERLAP_PIXELS = int(os.environ.get("SAM3D_BODY_DEPTH_OVERLAP_PIXELS", "500"))
DEPTH_SAMPLE_MAX = int(os.environ.get("SAM3D_BODY_DEPTH_SAMPLE_MAX", "5000"))
BBOX_DEPTH_PAD_PX = float(os.environ.get("SAM3D_BODY_BBOX_DEPTH_PAD_PX", "8"))
DEPTH_SCALE_MIN = float(os.environ.get("SAM3D_BODY_DEPTH_SCALE_MIN", "0.50"))
DEPTH_SCALE_MAX = float(os.environ.get("SAM3D_BODY_DEPTH_SCALE_MAX", "2.00"))
SUBJECT_CROP_PAD_PX = int(os.environ.get("SAM3D_BODY_SUBJECT_CROP_PAD_PX", "24"))
SUBJECT_CROP_BODY_BBOX_PAD_PX = int(os.environ.get("SAM3D_BODY_SUBJECT_CROP_BODY_BBOX_PAD_PX", "8"))

FBX_DECIMATE_RATIO = float(os.environ.get("SAM3D_BODY_FBX_DECIMATE_RATIO", "0.125"))
MATERIAL_COLOR = [0.0, 0.0, 0.0]
MATERIAL_ALPHA = float(os.environ.get("SAM3D_BODY_MATERIAL_ALPHA", "0.5"))
