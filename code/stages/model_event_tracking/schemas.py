from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _path_to_str(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _path_to_str(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_path_to_str(item) for item in value]
    return value


def to_jsonable(value: Any) -> Any:
    return _path_to_str(value)


@dataclass(frozen=True)
class RosStamp:
    sec: int
    nanosec: int = 0

    @classmethod
    def from_header(cls, header: Mapping[str, Any] | None) -> "RosStamp | None":
        if not isinstance(header, Mapping):
            return None
        stamp = header.get("stamp")
        if not isinstance(stamp, Mapping):
            return None
        try:
            return cls(sec=int(stamp.get("sec", 0)), nanosec=int(stamp.get("nanosec", 0)))
        except Exception:
            return None

    @classmethod
    def from_message_json(cls, payload: Mapping[str, Any] | None) -> "RosStamp | None":
        if not isinstance(payload, Mapping):
            return None
        message = payload.get("message")
        if isinstance(message, Mapping):
            found = cls.from_header(message.get("header"))
            if found is not None:
                return found
        return cls.from_header(payload.get("header"))

    @property
    def seconds(self) -> float:
        return float(self.sec) + float(self.nanosec) * 1.0e-9

    def delta_seconds(self, other: "RosStamp") -> float:
        return self.seconds - other.seconds

    def to_dict(self) -> dict[str, Any]:
        return {"sec": self.sec, "nanosec": self.nanosec, "seconds": self.seconds}


@dataclass(frozen=True)
class ShigureFrame:
    stamp: RosStamp
    rgb_path: Path | None = None
    depth_path: Path | None = None
    camera_info_path: Path | None = None
    people_path: Path | None = None
    marker_pose_path: Path | None = None
    node_paths: Mapping[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(frozen=True)
class ProjectedBox:
    corners_camera_m: np.ndarray
    pixel_points: np.ndarray
    bbox_xyxy: tuple[float, float, float, float]
    coordinate_system: str = "opencv_camera"

    @property
    def center_camera_m(self) -> np.ndarray:
        return np.asarray(self.corners_camera_m, dtype=np.float64).mean(axis=0)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "coordinate_system": self.coordinate_system,
                "corners_camera_m": self.corners_camera_m,
                "pixel_points": self.pixel_points,
                "bbox_xyxy": self.bbox_xyxy,
                "center_camera_m": self.center_camera_m,
            }
        )


@dataclass(frozen=True)
class HandContact:
    timestamp: RosStamp | None
    people_id: str
    hand: str
    score: float
    point_camera_m: tuple[float, float, float]
    pixel_xy: tuple[float, float] | None
    signed_distance_m: float
    distance_m: float
    inside_box: bool
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(frozen=True)
class MaskDepthSignature:
    area_px: int
    bbox_xyxy: tuple[int, int, int, int] | None
    center_camera_m: tuple[float, float, float] | None
    median_depth_m: float | None
    valid_depth_points: int

    @property
    def has_enough_depth(self) -> bool:
        return self.center_camera_m is not None and self.valid_depth_points > 0

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(frozen=True)
class MovementDecision:
    status: str
    moved: bool
    occluded: bool
    stable_in_place: bool
    should_stop_tracking: bool
    reason: str
    timestamp: RosStamp | None = None
    trigger_contact: HandContact | None = None
    center_delta_m: float | None = None
    depth_delta_m: float | None = None
    overlap_pixels: int = 0
    visible_area_ratio: float = 0.0
    area_ratio: float = 0.0
    mask_iou: float = 0.0
    movement_candidate_frames: int = 0
    depth_decision_timestamp: RosStamp | None = None
    rgb_motion_start_timestamp: RosStamp | None = None
    display_timestamp: RosStamp | None = None
    rgb_motion_score: float | None = None
    rgb_motion_metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(frozen=True)
class ModelEventRecord:
    task_id: str
    event_type: str
    event_timestamp: RosStamp | None
    trigger_timestamp: RosStamp | None
    decision: MovementDecision
    hand_contact: HandContact | None
    output_dir: Path
    files: Mapping[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))
