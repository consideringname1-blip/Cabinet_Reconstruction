"""Immediate raw Shigure collider snapshot relay.

This writer runs in the ROS recorder callback and is deliberately independent
of identity recovery, DINOv2, FoundationPose, models, and HoloLens polling.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from artifact_layout import SHIGURE_TRACKING_BOX_SNAPSHOT_PATH
from coordinate_systems import UNITY_TO_OPENCV_CAMERA_BASIS, orthonormalize_rotation
from stages.shigure_history.marker_history import latest_marker_pose_path
from stages.shigure_history.spatial_box_v2 import build_spatial_box_v2


def _camera_to_aruco() -> np.ndarray:
    path = latest_marker_pose_path()
    if path is None:
        raise FileNotFoundError("Shigure ArUco marker pose is unavailable")
    payload = json.loads(path.read_text(encoding="utf-8"))
    pose = payload.get("opencv_camera_pose")
    if not isinstance(pose, Mapping):
        raise ValueError("marker history lacks opencv_camera_pose")
    rotation = orthonormalize_rotation(
        np.asarray(pose.get("rotation_matrix"), dtype=np.float64).reshape(3, 3)
    )
    translation = np.asarray(pose.get("tvec_m"), dtype=np.float64).reshape(3)
    if not np.isfinite(translation).all():
        raise ValueError("marker translation is non-finite")
    basis = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = basis @ rotation.T
    transform[:3, 3] = basis @ rotation.T @ (-translation)
    return transform


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_raw_tracking_box_snapshot(
    tracking_payload: Mapping[str, Any],
    *,
    revision: int,
    source_stamp: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one authoritative snapshot immediately from object_tracking."""

    boxes: list[dict[str, Any]] = []
    try:
        camera_to_aruco = _camera_to_aruco()
    except Exception:
        camera_to_aruco = None

    if camera_to_aruco is not None:
        for item in tracking_payload.get("objects") or []:
            if not isinstance(item, Mapping):
                continue
            raw_id = str(item.get("object_id") or "").strip()
            action = str(item.get("action") or "").strip().lower()
            if not raw_id or action in {"take_out", "obj_move"}:
                continue
            try:
                result = build_spatial_box_v2(
                    item.get("collider"),
                    camera_to_aruco,
                    np.eye(4, dtype=np.float64),
                )
            except Exception:
                continue
            boxes.append(
                {
                    "tracking_id": raw_id,
                    "raw_tracking_id": raw_id,
                    "revision": max(1, int(revision)),
                    "corners_aruco": result["corners_aruco_m"],
                }
            )

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_stamp": dict(source_stamp or {}),
        "snapshot_complete": True,
        "count": len(boxes),
        "boxes": boxes,
    }
    _atomic_json(SHIGURE_TRACKING_BOX_SNAPSHOT_PATH, payload)
    return payload


__all__ = ["publish_raw_tracking_box_snapshot"]
