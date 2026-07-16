"""Server-owned Shigure lifecycle, identity, geometry and pose runtime.

The ROS publisher remains untouched.  This consumer accepts only canonical
``CachedShigureFrame`` schema v2 records from ``ShigureCompatibilityAdapter``.
Raw Shigure ids are scoped to a recorder incarnation/source epoch and are
never treated as persistent object identities.
"""

from __future__ import annotations

import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import tempfile
from typing import Any, Callable, Mapping, Sequence
import uuid

import cv2
import numpy as np

from artifact_layout import (
    IDENTITY_REFERENCE_ROOT,
    SHIGURE_EVENT_ROOT,
    SHIGURE_RECOVERY_DEBUG_ROOT,
)
from config import (
    FOUNDATIONPOSE_POOL_SIZE,
    REALTIME_TRACKING_EVENT_POLL_SEC,
    REALTIME_TRACKING_FP_MAX_DEPTH_RESIDUAL_M,
    REALTIME_TRACKING_FP_MIN_BBOX_IOU,
    SHIGURE_EXAMPLE_DINO_DISTANCE_THRESHOLD,
    SHIGURE_EXAMPLE_DINO_SECOND_MARGIN,
    SHIGURE_EXAMPLE_MAX_MASK_AREA_RATIO,
    SHIGURE_EXAMPLE_MIN_SEGMENT_PROBABILITY,
    SHIGURE_EXAMPLE_STABLE_BBOX_IOU,
    SHIGURE_EXAMPLE_STABLE_MASK_FRAMES,
    SHIGURE_EXAMPLE_STABLE_MASK_IOU,
    SHIGURE_HOLO_SYNC_CENTER_DISTANCE_M,
    SHIGURE_HOLO_SYNC_DINO_DISTANCE_THRESHOLD,
    SHIGURE_HOLO_SYNC_DINO_MARGIN,
    SHIGURE_HOLO_SYNC_MAX_RECOVERY_ATTEMPTS,
    SHIGURE_HOLO_SYNC_STABLE_MASK_FRAMES,
    SHIGURE_HOLO_SYNC_SIZE_LOG_TOLERANCE,
    SHIGURE_LIFECYCLE_CONFIRMATION_SECONDS,
    SHIGURE_LIFECYCLE_DEPTH_CHANGE_M,
    SHIGURE_LIFECYCLE_LOOKBACK_SECONDS,
    SHIGURE_LIFECYCLE_OBSERVATION_HZ,
    SHIGURE_LIFECYCLE_MAX_FOREGROUND_OCCLUSION_RATIO,
    SHIGURE_LIFECYCLE_MAX_PERSON_OVERLAP_RATIO,
    SHIGURE_LIFECYCLE_MIN_BACKGROUND_REVEAL_RATIO,
    SHIGURE_LIFECYCLE_MIN_POST_EVIDENCE_FRAMES,
    SHIGURE_LIFECYCLE_MIN_SOURCE_AREA_RATIO,
    SHIGURE_LIFECYCLE_MIN_VALID_DEPTH_RATIO,
    SHIGURE_LIFECYCLE_NO_MOVE_DISTANCE_M,
    SHIGURE_LIFECYCLE_SELECTION_WINDOW_SECONDS,
    SHIGURE_IDENTITY_HOLOLENS_DISTANCE_PENALTY,
    SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
    SHIGURE_IDENTITY_MAX_HOLOLENS_REFERENCES,
    SHIGURE_IDENTITY_CAPTURE_MAX_ATTEMPTS,
    SHIGURE_IDENTITY_CAPTURE_MAX_NEW_VIEWS,
    SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD,
    SHIGURE_IDENTITY_MATCH_REQUIRE_MARGIN,
    SHIGURE_IDENTITY_MATCH_SECOND_MARGIN,
    SHIGURE_SPATIAL_BOX_ACQUIRE_FRAMES,
    SHIGURE_SPATIAL_BOX_CENTER_DEADBAND_M,
    SHIGURE_SPATIAL_BOX_EMA_ALPHA,
    SHIGURE_SPATIAL_BOX_EXTENT_DEADBAND_M,
    SHIGURE_SPATIAL_BOX_FILTER_WINDOW_FRAMES,
    SHIGURE_SPATIAL_BOX_MISSING_GRACE_SECONDS,
    SHIGURE_STARTUP_RECOVERY_MAX_ATTEMPTS,
    SHIGURE_STARTUP_RECOVERY_RETRY_SECONDS,
)
from coordinate_systems import (
    FBX_RUNTIME_TRANSFORM_COMPENSATION_TO_UNITY,
    MODEL_INPUT_TO_CANONICAL_RH_BASIS,
    OPENCV_CAMERA_TO_CANONICAL_RH_BASIS,
    RUNTIME_LOCAL_TO_UNITY_POSE_ROTATION,
    UNITY_TO_OPENCV_CAMERA_BASIS,
    model_pose_canonical_rh_to_unity_camera,
    orthonormalize_rotation,
)
from spatial_transforms import rt_to_pose, shigure_camera_points_to_aruco
from stages.hololens3d_reconstruction.model_generation_common import resolve_model_generation_source
from stages.hololens3d_reconstruction.object_alignment_common import read_obj_vertices
from stages.shigure_history.cache import (
    CachedRgbdSample,
    CachedShigureFrame,
    ShigureRgbdCache,
    sample_key,
)
from stages.shigure_history.marker_history import latest_marker_pose_path
from stages.shigure_history.shigure_identity import cosine_distance
from stages.shigure_history.spatial_box_v2 import NoBoxError, build_spatial_box_v2
from task_db import (
    activate_recovered_shigure_binding,
    add_display_object_origin,
    apply_object_lifecycle_event,
    close_shigure_runtime_session,
    commit_pending_bring_in_lifecycle_event,
    commit_pending_take_out_lifecycle_event,
    establish_shigure_binding,
    get_active_shigure_binding,
    get_display_object_state,
    get_task_by_task_id,
    list_active_shigure_bindings,
    list_display_object_states,
    list_object_identity_references,
    open_shigure_source_epoch,
    reject_pending_shigure_canonical_event,
    record_shigure_canonical_event,
    set_object_identity_reference_embedding,
    start_shigure_runtime_session,
    update_shigure_live_observation,
    upsert_identity_sync_job,
)
from task_json import load_task_json, resolve_project_path, resolve_task_json_path_from_record


DinoRequest = Callable[[dict[str, Any]], dict[str, Any]]
FoundationPoseRequest = Callable[[dict[str, Any], str], dict[str, Any]]

MIN_SHIGURE_SKELETON_SCORE = 0.10



def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_token(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return cleaned.strip("_") or fallback


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            temporary = file.name
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass


def _write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_bytes(path, payload)


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"failed to write image: {path}")
    _atomic_bytes(path, encoded.tobytes())


def _decode_full_mask(value: Any, shape: tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(str(value or ""), validate=True)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("mask PNG could not be decoded")
    mask = image > 0
    if mask.shape != tuple(shape):
        raise ValueError(f"canonical full-frame mask shape {mask.shape} != RGB shape {shape}")
    if not np.any(mask):
        raise ValueError("canonical mask is empty")
    return mask


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        value = value.get("bbox_xyxy") or value.get("xyxy")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result).all() or result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("mask is empty")
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 0.0
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return float(intersection / union) if union else 0.0


def _strict_example_pair_is_stable(
    previous_box: Sequence[float],
    previous_mask: np.ndarray,
    current_box: Sequence[float],
    current_mask: np.ndarray,
) -> bool:
    previous_pixels = int(np.count_nonzero(previous_mask))
    current_pixels = int(np.count_nonzero(current_mask))
    if previous_pixels <= 0 or current_pixels <= 0:
        return False
    area_ratio = max(previous_pixels, current_pixels) / min(
        previous_pixels, current_pixels
    )
    return (
        _bbox_iou(previous_box, current_box)
        >= SHIGURE_EXAMPLE_STABLE_BBOX_IOU
        and _mask_iou(previous_mask, current_mask)
        >= SHIGURE_EXAMPLE_STABLE_MASK_IOU
        and area_ratio <= SHIGURE_EXAMPLE_MAX_MASK_AREA_RATIO
    )


_BOX_CORNER_BITS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def _box_center_extent(corners: Any) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(corners, dtype=np.float64)
    if points.shape != (8, 3) or not np.isfinite(points).all():
        raise ValueError("spatial box must contain eight finite 3D corners")
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    extent = maximum - minimum
    if np.any(extent <= 0.0):
        raise ValueError("spatial box extent must be positive")
    return (minimum + maximum) * 0.5, extent


def _corners_from_center_extent(
    center: np.ndarray, extent: np.ndarray
) -> list[list[float]]:
    minimum = np.asarray(center, dtype=np.float64) - (
        np.asarray(extent, dtype=np.float64) * 0.5
    )
    corners = minimum.reshape(1, 3) + (
        _BOX_CORNER_BITS * np.asarray(extent, dtype=np.float64).reshape(1, 3)
    )
    return corners.astype(float).tolist()


def _global_identity_assignment(
    candidate_scores: Sequence[Sequence[Mapping[str, Any]]],
    *,
    candidate_priorities: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Independently select each mask so raw-ID aliases remain possible."""

    priorities = (
        [1] * len(candidate_scores)
        if candidate_priorities is None
        else [max(0, int(value)) for value in candidate_priorities]
    )
    if len(priorities) != len(candidate_scores):
        raise ValueError("candidate_priorities must match candidate_scores")
    assignments: dict[int, dict[str, Any]] = {}
    ambiguous_candidates: set[int] = set()
    candidate_margins: list[float] = []
    for candidate_index, raw_scores in enumerate(candidate_scores):
        by_display: dict[str, dict[str, Any]] = {}
        for raw_score in raw_scores:
            display_object_id = str(
                raw_score.get("display_object_id") or ""
            ).strip()
            try:
                distance = float(raw_score["distance"])
            except (KeyError, TypeError, ValueError):
                continue
            if not display_object_id or not np.isfinite(distance):
                continue
            score = dict(raw_score)
            previous = by_display.get(display_object_id)
            if previous is None or distance < float(previous["distance"]):
                by_display[display_object_id] = score
        scores = sorted(
            by_display.values(),
            key=lambda item: (
                float(item["distance"]),
                str(item["display_object_id"]),
            ),
        )
        if (
            not scores
            or float(scores[0]["distance"])
            > SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD
        ):
            continue
        margin = (
            float(scores[1]["distance"])
            - float(scores[0]["distance"])
            if len(scores) > 1
            else None
        )
        if margin is not None:
            candidate_margins.append(margin)
        if (
            SHIGURE_IDENTITY_MATCH_REQUIRE_MARGIN
            and margin is not None
            and margin < SHIGURE_IDENTITY_MATCH_SECOND_MARGIN
        ):
            ambiguous_candidates.add(candidate_index)
            continue
        assignments[candidate_index] = scores[0]
    total_distance = sum(
        float(score["distance"]) for score in assignments.values()
    )
    return {
        "assignments": assignments,
        "ambiguous_candidates": ambiguous_candidates,
        "priority_matched_count": sum(
            priorities[index] for index in assignments
        ),
        "matched_count": len(assignments),
        "total_distance": total_distance,
        "second_total_distance": None,
        "assignment_margin": min(candidate_margins, default=None),
    }


def _embedding(value: Any) -> np.ndarray:
    if isinstance(value, Mapping):
        value = value.get("embedding")
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("embedding is empty or non-finite")
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        raise ValueError("embedding has zero norm")
    return result / norm


def _camera_to_aruco() -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    path = latest_marker_pose_path()
    if path is None:
        raise FileNotFoundError("Shigure ArUco marker pose is unavailable")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    pose = payload.get("opencv_camera_pose")
    if not isinstance(pose, Mapping):
        raise ValueError("marker history lacks opencv_camera_pose")
    rotation = orthonormalize_rotation(np.asarray(pose.get("rotation_matrix"), dtype=np.float64).reshape(3, 3))
    translation = np.asarray(pose.get("tvec_m"), dtype=np.float64).reshape(3)
    if not np.isfinite(translation).all():
        raise ValueError("marker translation is non-finite")
    basis = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = basis @ rotation.T
    transform[:3, 3] = basis @ rotation.T @ (-translation)
    revision = f"{path.name}:{path.stat().st_mtime_ns}"
    return transform, rotation, translation, revision


def _person_skeleton(person: Mapping[str, Any] | None, rotation: np.ndarray, translation: np.ndarray) -> dict[str, Any] | None:
    if not isinstance(person, Mapping):
        return None
    output: list[dict[str, Any]] = []
    for joint in person.get("joints") or []:
        if not isinstance(joint, Mapping):
            continue
        point = joint.get("projection_point")
        valid = False
        position: list[float] = [0.0, 0.0, 0.0]
        score = float(joint.get("score") or 0.0)
        if isinstance(point, Mapping):
            try:
                camera_m = np.asarray([point["x"], point["y"], point["z"]], dtype=np.float64) / 1000.0
                if np.isfinite(camera_m).all() and camera_m[2] > 1.0e-6 and np.linalg.norm(camera_m) > 1.0e-6 and score >= MIN_SHIGURE_SKELETON_SCORE:
                    aruco = shigure_camera_points_to_aruco(camera_m.reshape(1, 3), rotation, translation)[0]
                    position = aruco.astype(float).tolist()
                    valid = True
            except (KeyError, TypeError, ValueError):
                pass
        output.append(
            {
                "name": str(joint.get("body_part_name") or ""),
                "position": position,
                "score": score,
                "valid": valid,
            }
        )
    if not output or not any(item["valid"] for item in output):
        return None
    return {
        "people_id": str(person.get("people_id") or ""),
        "coordinate_space": "aruco",
        "source_units": "m",
        "joints": output,
    }


def _event_person(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for contact in event.get("contact_evidence") or []:
        if isinstance(contact, Mapping) and isinstance(contact.get("person"), Mapping):
            return contact["person"]
    return None


def _nearest_person(people: Sequence[Mapping[str, Any]], object_bbox: Sequence[float] | None) -> Mapping[str, Any] | None:
    if not people:
        return None
    if object_bbox is None:
        return None
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for person in people:
        box = _bbox(person.get("bounding_box"))
        if box is not None:
            scored.append((_bbox_iou(object_bbox, box), person))
    if not scored:
        return None
    score, person = max(scored, key=lambda item: item[0])
    return person if score > 0.0 else None


@dataclass(frozen=True)
class EventArtifacts:
    scene: Path | None
    mask: Path | None
    crop: Path | None
    sample: CachedRgbdSample | None
    mask_array: np.ndarray | None
    temporary_root: Path | None = None


@dataclass(frozen=True)
class LifecyclePoseCandidate:
    action: str
    canonical_event_uid: str
    display_object_id: str
    raw_id: str
    binding_id: str | None
    resolution_method: str
    sequence: int
    stamp_seconds: float
    source_generation: int
    source_epoch_id: str
    artifacts: EventArtifacts
    mask_center_aruco: np.ndarray | None
    dino_distance: float | None
    identity: dict[str, Any]
    skeleton: Any
    calibration_revision: str | None
    marker_rotation: np.ndarray | None
    marker_translation: np.ndarray | None
    occurred_at: str


@dataclass(frozen=True)
class TrustedMaskObservation:
    display_object_id: str
    binding_id: str
    raw_id: str
    sequence: int
    stamp_seconds: float
    artifacts: EventArtifacts
    mask_area: int
    valid_depth_ratio: float
    touches_image_edge: bool
    foreground_outlier_ratio: float
    person_overlap_ratio: float
    identity_distance: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "display_object_id": self.display_object_id,
            "binding_id": self.binding_id,
            "raw_shigure_object_id": self.raw_id,
            "sequence": int(self.sequence),
            "stamp_seconds": float(self.stamp_seconds),
            "mask_area": int(self.mask_area),
            "valid_depth_ratio": float(self.valid_depth_ratio),
            "touches_image_edge": bool(self.touches_image_edge),
            "foreground_outlier_ratio": float(
                self.foreground_outlier_ratio
            ),
            "person_overlap_ratio": float(self.person_overlap_ratio),
            "identity_distance": self.identity_distance,
        }


@dataclass(frozen=True)
class LifecycleMovementConfirmation:
    status: str
    reason: str
    source: TrustedMaskObservation | None
    detail: dict[str, Any]
    report_path: Path | None = None


def _cleanup_event_artifacts(artifacts: EventArtifacts | None) -> None:
    if artifacts is None or artifacts.temporary_root is None:
        return
    shutil.rmtree(artifacts.temporary_root, ignore_errors=True)


@dataclass
class SpatialBoxObservationState:
    binding_id: str
    raw_id: str
    model_revision: int
    centers: deque[np.ndarray] = field(
        default_factory=lambda: deque(
            maxlen=SHIGURE_SPATIAL_BOX_FILTER_WINDOW_FRAMES
        )
    )
    extents: deque[np.ndarray] = field(
        default_factory=lambda: deque(
            maxlen=SHIGURE_SPATIAL_BOX_FILTER_WINDOW_FRAMES
        )
    )
    filtered_center: np.ndarray | None = None
    filtered_extent: np.ndarray | None = None
    published_center: np.ndarray | None = None
    published_extent: np.ndarray | None = None
    last_good_monotonic: float | None = None
    pending_missing_since: float | None = None
    pending_missing_sequence: int = 0
    good_streak: int = 0
    cleared: bool = True


@dataclass
class PendingHoloSync:
    job_id: str
    display_object_id: str
    task_id: str
    task_json_path: Path
    target_center_aruco: np.ndarray | None
    target_size_aruco: np.ndarray | None
    source_generation: int = 0
    attempts: int = 0
    running: bool = False
    stable_raw_id: str = ""
    stable_bbox: tuple[float, float, float, float] | None = None
    stable_mask: np.ndarray | None = None
    stable_history: deque[
        tuple[
            str,
            tuple[float, float, float, float],
            np.ndarray,
        ]
    ] = field(
        default_factory=lambda: deque(
            maxlen=SHIGURE_HOLO_SYNC_STABLE_MASK_FRAMES
        )
    )
    stable_count: int = 0
    identity_method: str = ""
    identity_dino_detail: dict[str, Any] = field(default_factory=dict)
    last_attempt_source_key: str = ""
    last_stable_source_key: str = ""
    last_reason: str = "waiting_for_recovery_frame"
    temporary_roots: list[Path] = field(default_factory=list)


class ShigureRuntimeEngine:
    """Canonical frame consumer. Unity presentation state never pauses it."""

    def __init__(
        self,
        *,
        dino_request: DinoRequest,
        foundationpose_request: FoundationPoseRequest,
        cache: ShigureRgbdCache | None = None,
        poll_seconds: float = REALTIME_TRACKING_EVENT_POLL_SEC,
    ) -> None:
        self.cache = cache or ShigureRgbdCache()
        self.dino_request = dino_request
        self.foundationpose_request = foundationpose_request
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.runtime_session_id: str | None = None
        self.source_epoch_id: str | None = None
        self.source_incarnation_id: str | None = None
        self.last_sequence = 0
        self._source_generation = 0
        self._startup_recovery_pending = False
        self._startup_recovery_attempts = 0
        self._startup_recovery_last_source_key = ""
        self._startup_recovery_last_attempt_monotonic = float("-inf")
        self._startup_recovery_last_report_key = ""
        self._startup_recovery_last_report_monotonic = float("-inf")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._pose_executor = ThreadPoolExecutor(
            max_workers=max(1, int(FOUNDATIONPOSE_POOL_SIZE)),
            thread_name_prefix="shigure-fp",
        )
        self._identity_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="shigure-holo-sync",
        )
        self._pending_holo_syncs: dict[str, PendingHoloSync] = {}
        self._stable: dict[str, deque[tuple[int, tuple[float, float, float, float], Mapping[str, Any], np.ndarray]]] = {}
        self._last_stable_source_key: dict[str, str] = {}
        self._last_strict_candidate_snapshot_source_key = ""
        self._last_view_sequence: dict[str, int] = {}
        self._view_inflight: set[str] = set()
        self._view_windows: dict[str, dict[str, Any]] = {}
        self._spatial_boxes: dict[str, SpatialBoxObservationState] = {}
        self._lifecycle_pose_windows: dict[str, dict[str, Any]] = {}
        self._trusted_mask_observations: dict[
            str, deque[TrustedMaskObservation]
        ] = {}
        self._last_geometry_stamp: tuple[int, int] | None = None
        self._runtime_obj_move_quarantine_active = False
        self._runtime_obj_move_barrier_stamp: tuple[int, int] | None = None
        self._last_camera_calibration: tuple[
            np.ndarray, np.ndarray, np.ndarray, str
        ] | None = None
        self._mask_identity_attempts: dict[str, np.ndarray] = {}
        self._initial_pose_attempts: dict[str, int] = {}
        self._initial_pose_inflight: set[str] = set()
        self._initial_pose_completed: set[str] = set()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            session = start_shigure_runtime_session(
                server_boot_id=uuid.uuid4().hex,
                config={
                    "canonical_schema": 2,
                    "identity_candidate_limit": SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
                    "spatial_box_missing_grace_seconds": (
                        SHIGURE_SPATIAL_BOX_MISSING_GRACE_SECONDS
                    ),
                    "lifecycle_confirmation_seconds": (
                        SHIGURE_LIFECYCLE_CONFIRMATION_SECONDS
                    ),
                    "lifecycle_lookback_seconds": (
                        SHIGURE_LIFECYCLE_LOOKBACK_SECONDS
                    ),
                },
            )
            self.runtime_session_id = str(session["runtime_session_id"])
            self._startup_recovery_pending = True
            bootstrap_root = SHIGURE_RECOVERY_DEBUG_ROOT / self.runtime_session_id
            _write_json(
                bootstrap_root / "bootstrap.json",
                {
                    "schema_version": 1,
                    "runtime_session_id": self.runtime_session_id,
                    "started_utc": _utc_now(),
                    "status": "PENDING",
                    "reason": "server_started_waiting_for_calibrated_image_and_masks",
                },
            )
            print(
                "[shigure-v2] startup recovery armed at server start; "
                f"report={bootstrap_root / 'bootstrap.json'}"
            )
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="shigure-runtime-v2")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._lock:
            self._source_generation += 1
            pending_jobs = list(self._pending_holo_syncs.values())
            self._pending_holo_syncs.clear()
            pending_lifecycle = [
                candidate
                for window in self._lifecycle_pose_windows.values()
                for candidate in window["candidates"]
            ]
            self._lifecycle_pose_windows.clear()
        self._reject_lifecycle_candidates(
            pending_lifecycle,
            reason="RUNTIME_STOPPED_DURING_LIFECYCLE_WINDOW",
        )
        for job in pending_jobs:
            try:
                self._record_holo_sync_status(
                    job,
                    status="FAILED",
                    reason="runtime_stopped_before_sync",
                    terminal=True,
                )
            except Exception as exc:
                print(f"[shigure-v2] failed to terminate pending Holo sync: {exc}")
        self._pose_executor.shutdown(wait=False, cancel_futures=True)
        runtime = self.runtime_session_id
        self._identity_executor.shutdown(wait=False, cancel_futures=True)
        if runtime:
            try:
                close_shigure_runtime_session(runtime, reason="server_shutdown")
            except Exception as exc:
                print(f"[shigure-v2] failed to close runtime session: {exc}")
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.process_once()
            except Exception as exc:
                print(f"[shigure-v2] poll failed: {exc}")
            self._stop.wait(self.poll_seconds)

    def process_once(self) -> int:
        latest = self.cache.latest_canonical_frame(include_masks=True)
        if latest is None:
            now_monotonic = time.monotonic()
            self._expire_pending_spatial_boxes(now_monotonic)
            self._flush_lifecycle_pose_windows(now_monotonic)
            return 0
        if latest.source_incarnation_id != self.source_incarnation_id:
            self._open_incarnation(
                latest.source_incarnation_id,
                reset_sequence=int(latest.sequence) <= int(self.last_sequence),
            )
        processed = 0
        # Bound each mask-bearing socket response so a backlog accumulated
        # during DINO/FP work cannot monopolize the server GIL and starve HTTP.
        frames = list(
            self.cache.iter_canonical_updates_after(
                self.last_sequence,
                include_masks=True,
                limit=16,
                runtime_relevant_masks=True,
            )
        )
        for frame in frames:
            # A same-process adapter rotation leaves old frames in the bounded
            # cache. Only the latest incarnation is authoritative; reopening
            # old epochs here would oscillate bindings.
            if frame.source_incarnation_id != self.source_incarnation_id:
                self.last_sequence = max(self.last_sequence, int(frame.sequence))
                continue
            self._process_frame(frame)
            self.last_sequence = max(self.last_sequence, int(frame.sequence))
            processed += 1
        if self.last_sequence == 0:
            self._process_frame(latest)
            self.last_sequence = int(latest.sequence)
            processed = 1
        now_monotonic = time.monotonic()
        self._expire_pending_spatial_boxes(now_monotonic)
        self._flush_lifecycle_pose_windows(now_monotonic)
        return processed

    def _reset_epoch_local_runtime_state(self, *, holo_reason: str) -> None:
        with self._lock:
            pending_lifecycle = [
                candidate
                for window in self._lifecycle_pose_windows.values()
                for candidate in window["candidates"]
            ]
            self._source_generation += 1
            self._startup_recovery_pending = True
            self._startup_recovery_attempts = 0
            self._startup_recovery_last_source_key = ""
            self._startup_recovery_last_attempt_monotonic = float("-inf")
            self._startup_recovery_last_report_key = ""
            self._startup_recovery_last_report_monotonic = float("-inf")
            self._stable.clear()
            self._last_stable_source_key.clear()
            self._last_strict_candidate_snapshot_source_key = ""
            self._last_view_sequence.clear()
            self._view_windows.clear()
            self._view_inflight.clear()
            self._spatial_boxes.clear()
            self._lifecycle_pose_windows.clear()
            self._trusted_mask_observations.clear()
            self._last_geometry_stamp = None
            self._runtime_obj_move_quarantine_active = False
            self._runtime_obj_move_barrier_stamp = None
            self._last_camera_calibration = None
            self._mask_identity_attempts.clear()
            self._initial_pose_attempts.clear()
            self._initial_pose_inflight.clear()
            self._initial_pose_completed.clear()
            pending_jobs = list(self._pending_holo_syncs.values())
            for job in pending_jobs:
                job.source_generation = self._source_generation
                job.stable_raw_id = ""
                job.stable_bbox = None
                job.stable_mask = None
                job.stable_history.clear()
                job.stable_count = 0
                job.last_attempt_source_key = ""
                job.last_stable_source_key = ""
                job.last_reason = str(holo_reason)
        self._reject_lifecycle_candidates(
            pending_lifecycle,
            reason="SOURCE_EPOCH_CHANGED_DURING_LIFECYCLE_WINDOW",
        )
        for job in pending_jobs:
            self._record_holo_sync_status(
                job,
                status="PENDING",
                reason=job.last_reason,
                terminal=False,
            )

    def _open_incarnation(self, incarnation_id: str, *, reset_sequence: bool = False) -> None:
        if not self.runtime_session_id:
            raise RuntimeError("Shigure runtime session has not started")
        epoch = open_shigure_source_epoch(
            runtime_session_id=self.runtime_session_id,
            reason="recorder_incarnation_changed" if self.source_incarnation_id else "initial_recorder_incarnation",
            publisher_fingerprint={"source_incarnation_id": str(incarnation_id)},
        )
        self.source_epoch_id = str(epoch["source_epoch_id"])
        self.source_incarnation_id = str(incarnation_id)
        if reset_sequence:
            self.last_sequence = 0
        self._reset_epoch_local_runtime_state(
            holo_reason="source_epoch_changed_waiting_for_stable_recovery_frame"
        )
        resumed_syncs = self.resume_unbound_hololens_capture_syncs()
        if resumed_syncs:
            print(
                "[shigure-v2] resumed Holo identity syncs after source epoch "
                f"change: count={resumed_syncs} source_epoch_id={self.source_epoch_id}"
            )
        recovery = self.cache.latest_recovery_frame(include_masks=True)
        if recovery is not None and recovery.source_incarnation_id == incarnation_id:
            self._maybe_startup_recovery(recovery)

    def _open_runtime_obj_move_epoch(
        self, frame: CachedShigureFrame
    ) -> None:
        if not self.runtime_session_id:
            raise RuntimeError("Shigure runtime session has not started")
        barrier_stamp = (
            int(frame.source_stamp.sec),
            int(frame.source_stamp.nanosec),
        )
        epoch = open_shigure_source_epoch(
            runtime_session_id=self.runtime_session_id,
            reason="runtime_obj_move_fail_closed",
            publisher_fingerprint={
                "source_incarnation_id": str(
                    self.source_incarnation_id
                    or frame.source_incarnation_id
                ),
                "obj_move_barrier_stamp": {
                    "sec": barrier_stamp[0],
                    "nanosec": barrier_stamp[1],
                },
            },
        )
        self.source_epoch_id = str(epoch["source_epoch_id"])
        self._reset_epoch_local_runtime_state(
            holo_reason="obj_move_waiting_for_clean_tracking"
        )
        self._runtime_obj_move_quarantine_active = True
        self._runtime_obj_move_barrier_stamp = barrier_stamp
        self._last_geometry_stamp = barrier_stamp

    def _maybe_startup_recovery(self, frame: CachedShigureFrame) -> None:
        if self._startup_recovery_pending:
            # Startup and steady-state identity use the same per-mask,
            # HoloLens-only matcher. A global one-to-one assignment is
            # incompatible with the required raw-ID aliases.
            self._reconcile_mask_bindings(frame)

    def _sample_exact(self, frame: CachedShigureFrame) -> CachedRgbdSample | None:
        sample = self.cache.get_sample(frame.source_stamp)
        return sample if sample is not None and sample.stamp == frame.source_stamp else None

    def _recovery_debug_epoch_root(self) -> Path:
        return (
            SHIGURE_RECOVERY_DEBUG_ROOT
            / str(self.runtime_session_id or "runtime_missing")
            / str(self.source_epoch_id or "source_epoch_pending")
        )

    @staticmethod
    def _recovery_candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "candidate_id": candidate.get("candidate_id"),
            "raw_shigure_object_id": candidate.get("shigure_object_id"),
            "tracking_match_status": candidate.get("tracking_match_status"),
            "tracking_mapping_method": candidate.get("tracking_mapping_method"),
            "tracking_candidates": candidate.get("tracking_candidates") or [],
            "bbox": _bbox(candidate),
            "mask_available": bool(candidate.get("mask_b64")),
        }

    def _write_recovery_input_artifacts(
        self,
        root: Path,
        frame: CachedShigureFrame,
        sample: CachedRgbdSample | None,
    ) -> list[dict[str, Any]]:
        root.mkdir(parents=True, exist_ok=True)
        if sample is None:
            return []
        _write_image(root / "scene.png", sample.rgb_bgr)
        manifest: list[dict[str, Any]] = []
        candidates_root = root / "candidates"
        for index, candidate in enumerate(frame.recovery_candidates):
            candidate_id = str(candidate.get("candidate_id") or f"candidate_{index}")
            safe_id = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
            item_root = candidates_root / f"{index:02d}_{safe_id}"
            entry = {
                **self._recovery_candidate_summary(candidate),
                "directory": str(item_root),
            }
            try:
                mask = _decode_full_mask(
                    candidate.get("mask_b64"),
                    sample.rgb_bgr.shape[:2],
                )
                item_root.mkdir(parents=True, exist_ok=True)
                _write_image(item_root / "mask.png", mask.astype(np.uint8) * 255)
                box = _bbox(candidate) or _mask_bbox(mask)
                x0, y0, x1, y1 = [int(round(value)) for value in box]
                x0, y0 = max(0, x0), max(0, y0)
                x1 = min(sample.rgb_bgr.shape[1], x1)
                y1 = min(sample.rgb_bgr.shape[0], y1)
                if x1 > x0 and y1 > y0:
                    crop = sample.rgb_bgr[y0:y1, x0:x1].copy()
                    crop[~mask[y0:y1, x0:x1]] = 0
                    _write_image(item_root / "object_crop.png", crop)
                entry["mask_file"] = str(item_root / "mask.png")
                entry["crop_file"] = (
                    str(item_root / "object_crop.png")
                    if (item_root / "object_crop.png").is_file()
                    else None
                )
            except Exception as exc:
                entry["artifact_error"] = f"{exc.__class__.__name__}: {exc}"
            manifest.append(entry)
        return manifest

    def _startup_recovery_calibrated_input_reason(
        self,
        frame: CachedShigureFrame,
    ) -> str | None:
        sample = self._sample_exact(frame)
        if sample is None:
            return "waiting_for_exact_rgbd"
        if not isinstance(sample.camera_info, Mapping) or not sample.camera_info:
            return "waiting_for_image_camera_info"
        try:
            _camera_to_aruco()
        except Exception:
            return "waiting_for_shigure_camera_to_armarker_calibration"
        for candidate in frame.recovery_candidates:
            try:
                mask = _decode_full_mask(
                    candidate.get("mask_b64"),
                    sample.rgb_bgr.shape[:2],
                )
            except Exception:
                return "waiting_for_image_aligned_candidate_mask"
            if not np.any(mask):
                return "waiting_for_nonempty_candidate_mask"
        return None

    @staticmethod
    def _startup_recovery_complete_empty(
        frame: CachedShigureFrame,
    ) -> bool:
        return (
            not frame.recovery_candidates
            and frame.input_states.get("segments") == "explicit_empty"
            and frame.input_states.get("object_tracking")
            == "explicit_empty"
        )

    def _startup_recovery_readiness_reason(
        self,
        frame: CachedShigureFrame,
    ) -> str | None:
        """Return why a non-empty startup snapshot cannot be reconciled yet."""

        if self._startup_recovery_complete_empty(frame):
            return None
        if frame.input_states.get("segments") != "present":
            return "waiting_for_present_segments_snapshot"
        if frame.input_states.get("object_tracking") != "present":
            return "waiting_for_present_object_tracking_snapshot"
        if not frame.recovery_candidates:
            return "waiting_for_explicit_empty_or_resolved_candidates"
        calibrated_reason = self._startup_recovery_calibrated_input_reason(
            frame
        )
        if calibrated_reason is not None:
            return calibrated_reason
        for index, candidate in enumerate(frame.recovery_candidates):
            raw_id = str(
                candidate.get("shigure_object_id") or ""
            ).strip()
            match_status = str(
                candidate.get("tracking_match_status") or ""
            ).upper()
            if not raw_id or match_status != "RESOLVED":
                candidate_id = str(
                    candidate.get("candidate_id") or f"candidate_{index}"
                )
                return (
                    "waiting_for_resolved_candidate_raw_id:"
                    f"{candidate_id}"
                )
        return None

    def _record_startup_recovery_pending(
        self,
        frame: CachedShigureFrame,
        reason: str,
    ) -> None:
        report_root = self._persist_recovery_observation(frame, reason)
        job_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            (
                f"shigure-recovery:{self.runtime_session_id}:"
                f"{self.source_epoch_id}"
            ),
        ).hex
        result = {
            "schema_version": 1,
            "status": "PENDING",
            "reason": str(reason),
            "frame_sequence": int(frame.sequence),
            "source_stamp": frame.source_stamp.to_dict(),
            "input_states": dict(frame.input_states),
            "candidate_count": len(frame.recovery_candidates),
            "candidates": [
                self._recovery_candidate_summary(candidate)
                for candidate in frame.recovery_candidates
            ],
        }
        if report_root is not None:
            result["debug_report_path"] = str(report_root / "report.json")
        upsert_identity_sync_job(
            sync_job_id=job_id,
            kind="STARTUP_RECOVERY",
            status="PENDING",
            runtime_session_id=self.runtime_session_id,
            source_epoch_id=self.source_epoch_id,
            candidate_limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
            result=result,
        )

    def _persist_recovery_observation(
        self,
        frame: CachedShigureFrame,
        reason: str,
    ) -> Path | None:
        report_key = str(reason)
        now_monotonic = float(frame.received_monotonic)
        if self._startup_recovery_last_report_key == report_key:
            return None
        if (
            now_monotonic - self._startup_recovery_last_report_monotonic
            < SHIGURE_STARTUP_RECOVERY_RETRY_SECONDS
        ):
            return None
        self._startup_recovery_last_report_key = report_key
        self._startup_recovery_last_report_monotonic = now_monotonic
        root = (
            self._recovery_debug_epoch_root()
            / "observations"
            / f"sequence_{int(frame.sequence):09d}"
        )
        artifacts = self._write_recovery_input_artifacts(
            root,
            frame,
            self._sample_exact(frame),
        )
        report = {
            "schema_version": 1,
            "runtime_session_id": self.runtime_session_id,
            "source_epoch_id": self.source_epoch_id,
            "recorded_utc": _utc_now(),
            "status": "PENDING",
            "reason": str(reason),
            "frame_sequence": int(frame.sequence),
            "source_stamp": frame.source_stamp.to_dict(),
            "input_states": dict(frame.input_states),
            "canonical_diagnostics": list(frame.diagnostics),
            "candidate_count": len(frame.recovery_candidates),
            "candidates": artifacts,
        }
        _write_json(root / "report.json", report)
        print(
            "[shigure-v2] startup recovery waiting: "
            f"reason={reason} report={root / 'report.json'}"
        )
        return root

    def _artifact_root(self, frame: CachedShigureFrame, token: str) -> Path:
        source_sample_key = sample_key(frame.source_stamp)
        directory = uuid.uuid5(
            uuid.NAMESPACE_URL,
            (
                f"shigure-artifact:{self.runtime_session_id}:{self.source_epoch_id}:"
                f"{token}:{source_sample_key}:{frame.frame_id}"
            ),
        ).hex
        return SHIGURE_EVENT_ROOT / directory

    def _event_artifacts(self, frame: CachedShigureFrame, event: Mapping[str, Any]) -> EventArtifacts:
        sample = self._sample_exact(frame)
        if sample is None:
            return EventArtifacts(None, None, None, None, None)
        root = self._artifact_root(frame, str(event.get("event_uid") or f"event_{frame.sequence}"))
        scene_path = root / "scene.png"
        _write_image(scene_path, sample.rgb_bgr)
        detection = event.get("detection") if isinstance(event.get("detection"), Mapping) else None
        mask: np.ndarray | None = None
        mask_path: Path | None = None
        if detection is not None and detection.get("mask_status") == "VALID":
            mask = _decode_full_mask(detection.get("mask_b64"), sample.rgb_bgr.shape[:2])
            mask_path = root / "mask.png"
            _write_image(mask_path, mask.astype(np.uint8) * 255)
        box = _bbox(detection) or _bbox(event.get("tracking"))
        crop_path: Path | None = None
        if box is not None:
            height, width = sample.rgb_bgr.shape[:2]
            x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
            x1, y1 = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
            if x1 > x0 and y1 > y0:
                crop = sample.rgb_bgr[y0:y1, x0:x1].copy()
                if mask is not None:
                    local_mask = mask[y0:y1, x0:x1]
                    crop[~local_mask] = 0
                crop_path = root / "object_crop.png"
                _write_image(crop_path, crop)
        return EventArtifacts(scene_path, mask_path, crop_path, sample, mask)

    def _embedding_sidecar(self, reference_id: str, response: Mapping[str, Any]) -> Path:
        payload = {
            key: value
            for key, value in response.items()
            if key in {"embedding", "dim", "model_name", "normalized", "source", "crop"}
        }
        payload["embedding"] = _embedding(response).astype(float).tolist()
        payload.setdefault("dim", len(payload["embedding"]))
        payload.setdefault("normalized", True)
        path = IDENTITY_REFERENCE_ROOT / _safe_token(reference_id, "reference") / "embedding.json"
        _write_json(path, payload)
        return path

    def _load_embedding_path(self, path_value: Any) -> np.ndarray:
        path = resolve_project_path(str(path_value), require_exists=True)
        with path.open("r", encoding="utf-8") as file:
            return _embedding(json.load(file))

    def _ensure_reference_embedding(self, reference: Mapping[str, Any]) -> np.ndarray | None:
        path_value = str(reference.get("embedding_path") or "").strip()
        if path_value:
            try:
                return self._load_embedding_path(path_value)
            except Exception:
                pass
        image_path = str(reference.get("image_path") or "").strip()
        mask_path = str(reference.get("mask_path") or "").strip()
        if not image_path or not mask_path:
            return None
        try:
            image = resolve_project_path(image_path, require_exists=True)
            mask = resolve_project_path(mask_path, require_exists=True)
            response = self.dino_request(
                {"action": "embed_files", "color_file": str(image), "mask_file": str(mask)}
            )
            vector = _embedding(response)
            sidecar = self._embedding_sidecar(str(reference["reference_id"]), response)
            set_object_identity_reference_embedding(
                str(reference["reference_id"]), embedding_path=sidecar
            )
            return vector
        except Exception as exc:
            print(f"[shigure-v2] identity reference embedding failed: {exc}")
            return None

    def _reference_vectors(self, display_object_id: str) -> list[tuple[str, str, np.ndarray]]:
        references = list_object_identity_references(display_object_id, limit=100)
        vectors: list[tuple[str, str, np.ndarray]] = []
        for reference in references:
            if str(reference.get("source") or "").upper() != "HOLOLENS":
                continue
            vector = self._ensure_reference_embedding(reference)
            if vector is not None:
                vectors.append((str(reference["reference_id"]), str(reference["source"]), vector))
            if len(vectors) >= SHIGURE_IDENTITY_MAX_HOLOLENS_REFERENCES:
                break
        return vectors

    def _recent_display_ids(self) -> list[str]:
        rows = list_display_object_states(
            limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS
        )
        return [str(row["display_object_id"]) for row in rows]

    def _identity_scores(self, observation: np.ndarray, display_ids: Sequence[str]) -> list[dict[str, Any]]:
        scores: list[dict[str, Any]] = []
        for display_object_id in display_ids:
            references = self._reference_vectors(display_object_id)
            if not references:
                continue
            candidates = []
            for reference_id, source, vector in references:
                raw_distance = cosine_distance(observation, vector)
                candidates.append(
                    (raw_distance, raw_distance, 0.0, reference_id, source)
                )
            best = min(candidates, key=lambda item: (item[0], item[3]))
            scores.append(
                {
                    "display_object_id": display_object_id,
                    "distance": float(best[0]),
                    "raw_distance": float(best[1]),
                    "source_penalty": float(best[2]),
                    "reference_id": best[3],
                    "reference_source": best[4],
                }
            )
        scores.sort(key=lambda item: (item["distance"], item["display_object_id"]))
        return scores

    def _select_identity(self, scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not scores:
            return {"status": "UNBOUND", "reason": "no_identity_references", "scores": []}
        best = scores[0]
        second = scores[1] if len(scores) > 1 else None
        margin = float(second["distance"]) - float(best["distance"]) if second else None
        if float(best["distance"]) > SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD:
            return {"status": "UNBOUND", "reason": "distance_threshold", "scores": list(scores)}
        if SHIGURE_IDENTITY_MATCH_REQUIRE_MARGIN and margin is not None and margin < SHIGURE_IDENTITY_MATCH_SECOND_MARGIN:
            return {"status": "AMBIGUOUS", "reason": "second_margin", "scores": list(scores)}
        return {
            "status": "MATCHED",
            "display_object_id": str(best["display_object_id"]),
            "distance": float(best["distance"]),
            "margin": margin,
            "reference_id": best["reference_id"],
            "reference_source": best["reference_source"],
            "scores": list(scores),
        }

    def _embed_artifacts(self, artifacts: EventArtifacts) -> tuple[np.ndarray, dict[str, Any]] | None:
        if artifacts.scene is None or artifacts.mask is None:
            return None
        response = self.dino_request(
            {"action": "embed_files", "color_file": str(artifacts.scene), "mask_file": str(artifacts.mask)}
        )
        return _embedding(response), response

    def _verify_shigure_view_admission(
        self,
        display_object_id: str,
        embedding: np.ndarray,
    ) -> dict[str, Any]:
        """Strictly prove a Shigure view against an existing identity anchor."""

        target = str(display_object_id or "").strip()
        display_ids = [target]
        display_ids.extend(
            item
            for item in self._recent_display_ids()
            if item != target
        )
        display_ids = display_ids[:SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS]
        scores = self._identity_scores(embedding, display_ids)
        target_score = next(
            (
                item
                for item in scores
                if str(item.get("display_object_id") or "") == target
            ),
            None,
        )
        base = {
            "admission_method": "DINOV2_STRICT_SHIGURE_EXAMPLE",
            "admission_display_object_id": target,
            "admission_distance_threshold": (
                SHIGURE_EXAMPLE_DINO_DISTANCE_THRESHOLD
            ),
            "admission_second_margin_threshold": (
                SHIGURE_EXAMPLE_DINO_SECOND_MARGIN
            ),
        }
        if target_score is None:
            return {
                **base,
                "admission_status": "REJECTED",
                "admission_reason": "target_has_no_identity_anchor",
            }
        competitors = [
            item
            for item in scores
            if str(item.get("display_object_id") or "") != target
        ]
        second = competitors[0] if competitors else None
        distance = float(target_score["distance"])
        margin = (
            float(second["distance"]) - distance
            if second is not None
            else None
        )
        detail = {
            **base,
            "admission_distance": distance,
            "admission_margin": margin,
            "admission_competitor_count": len(competitors),
            "admission_reference_id": target_score.get("reference_id"),
            "admission_reference_source": target_score.get(
                "reference_source"
            ),
            "admission_competitor_display_object_id": (
                second.get("display_object_id") if second else None
            ),
        }
        if not scores or str(scores[0].get("display_object_id") or "") != target:
            return {
                **detail,
                "admission_status": "REJECTED",
                "admission_reason": "different_display_object_is_closer",
            }
        if distance > SHIGURE_EXAMPLE_DINO_DISTANCE_THRESHOLD:
            return {
                **detail,
                "admission_status": "REJECTED",
                "admission_reason": "strict_distance_threshold",
            }
        if margin is not None and margin < SHIGURE_EXAMPLE_DINO_SECOND_MARGIN:
            return {
                **detail,
                "admission_status": "REJECTED",
                "admission_reason": "strict_second_margin",
            }
        return {
            **detail,
            "admission_status": "MATCHED",
            "admission_reason": "strict_identity_anchor_match",
        }

    def _register_view(
        self,
        *,
        display_object_id: str,
        artifacts: EventArtifacts,
        embedding: np.ndarray,
        embedding_response: Mapping[str, Any],
        source_event_uid: str | None,
        quality: Mapping[str, Any],
        source_epoch_id: str,
        binding_id: str,
        raw_shigure_object_id: str,
        admission: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        # Shigure scene/mask artifacts are diagnostics and query observations,
        # never long-term identity examples.
        return None

    def _candidate_artifacts(
        self, frame: CachedShigureFrame, candidate: Mapping[str, Any], sample: CachedRgbdSample
    ) -> EventArtifacts:
        mask = _decode_full_mask(candidate.get("mask_b64"), sample.rgb_bgr.shape[:2])
        root = Path(tempfile.mkdtemp(prefix="shigure-v2-candidate-"))
        scene, mask_path = root / "scene.png", root / "mask.png"
        try:
            _write_image(scene, sample.rgb_bgr)
            _write_image(mask_path, mask.astype(np.uint8) * 255)
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        box = _bbox(candidate) or _mask_bbox(mask)
        x0, y0, x1, y1 = [int(round(value)) for value in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(sample.rgb_bgr.shape[1], x1), min(sample.rgb_bgr.shape[0], y1)
        crop_path: Path | None = None
        if x1 > x0 and y1 > y0:
            crop = sample.rgb_bgr[y0:y1, x0:x1].copy()
            crop[~mask[y0:y1, x0:x1]] = 0
            crop_path = root / "object_crop.png"
            _write_image(crop_path, crop)
        return EventArtifacts(scene, mask_path, crop_path, sample, mask, root)

    def _mask_identity_attempt_is_new(
        self, raw_id: str, mask: np.ndarray
    ) -> bool:
        """Return true only for a materially new mask for an unbound raw id."""

        previous = self._mask_identity_attempts.get(str(raw_id))
        if previous is not None and _mask_iou(previous, mask) >= 0.80:
            return False
        self._mask_identity_attempts[str(raw_id)] = np.asarray(
            mask, dtype=bool
        ).copy()
        return True

    @staticmethod
    def _mask_depth_aabb_aruco(
        sample: CachedRgbdSample, mask: np.ndarray
    ) -> list[list[float]]:
        """Build a robust ArUco-axis-aligned AABB from exact mask/depth."""

        depth = np.asarray(sample.depth)
        mask = np.asarray(mask, dtype=bool)
        if depth.shape[:2] != mask.shape:
            raise ValueError("mask/depth shapes differ")
        valid = mask & np.isfinite(depth) & (depth > 0)
        ys, xs = np.nonzero(valid)
        if xs.size < 32:
            raise ValueError("mask has fewer than 32 valid depth pixels")
        if xs.size > 30000:
            indexes = np.linspace(0, xs.size - 1, 30000).astype(np.int64)
            xs, ys = xs[indexes], ys[indexes]
        z = depth[ys, xs].astype(np.float64)
        # Shigure publishes depth in millimetres.  The fallback keeps this
        # helper usable with already-metric local regression fixtures.
        if float(np.median(z)) > 20.0:
            z *= 0.001
        k = np.asarray(
            sample.camera_info.get("k"), dtype=np.float64
        ).reshape(3, 3)
        if (
            not np.isfinite(k).all()
            or abs(float(k[0, 0])) < 1.0e-9
            or abs(float(k[1, 1])) < 1.0e-9
        ):
            raise ValueError("camera intrinsics are invalid")
        camera_points = np.column_stack(
            (
                (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0],
                (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1],
                z,
                np.ones_like(z),
            )
        )
        transform = np.asarray(_camera_to_aruco()[0], dtype=np.float64)
        homogeneous = (transform @ camera_points.T).T
        valid_w = np.isfinite(homogeneous).all(axis=1) & (
            np.abs(homogeneous[:, 3]) > 1.0e-9
        )
        points = homogeneous[valid_w, :3] / homogeneous[valid_w, 3:4]
        if points.shape[0] < 32:
            raise ValueError("too few finite ArUco mask points")
        minimum = np.percentile(points, 2.0, axis=0)
        maximum = np.percentile(points, 98.0, axis=0)
        center = (minimum + maximum) * 0.5
        extent = np.maximum(maximum - minimum, 0.01)
        if not np.isfinite(center).all() or not np.isfinite(extent).all():
            raise ValueError("mask AABB is non-finite")
        return _corners_from_center_extent(center, extent)

    def _commit_primary_mask_box(
        self,
        binding: Mapping[str, Any],
        frame: CachedShigureFrame,
        sample: CachedRgbdSample,
        mask: np.ndarray,
    ) -> bool:
        state = get_display_object_state(str(binding["display_object_id"])) or {}
        if str(state.get("active_shigure_binding_id") or "") != str(
            binding["binding_id"]
        ):
            return False
        model_revision = int(state.get("active_model_revision") or 0)
        if model_revision <= 0:
            return False
        corners = self._mask_depth_aabb_aruco(sample, mask)
        result = update_shigure_live_observation(
            binding_id=str(binding["binding_id"]),
            observation_seq=int(frame.sequence),
            model_revision=model_revision,
            spatial_box_corners_aruco=corners,
            spatial_box_observed=True,
        )
        return bool(result and result.get("_spatial_box_accepted"))

    def _clear_primary_mask_box(
        self,
        binding: Mapping[str, Any],
        frame: CachedShigureFrame,
    ) -> bool:
        """Publish an explicit no-box observation for the current primary."""

        state = get_display_object_state(
            str(binding["display_object_id"])
        ) or {}
        if str(state.get("active_shigure_binding_id") or "") != str(
            binding["binding_id"]
        ):
            return False
        model_revision = int(state.get("active_model_revision") or 0)
        if model_revision <= 0:
            return False
        result = update_shigure_live_observation(
            binding_id=str(binding["binding_id"]),
            observation_seq=int(frame.sequence),
            model_revision=model_revision,
            spatial_box_corners_aruco=None,
            spatial_box_observed=True,
        )
        return bool(result and result.get("_spatial_box_accepted"))

    @staticmethod
    def _select_primary_mask_observation(
        observations: Mapping[str, Mapping[str, Any]],
        primary_binding_id: str,
        *,
        complete_snapshot: bool,
    ) -> Mapping[str, Any] | None:
        """Select one primary, preferring the lower-distance new alias."""

        def rank(observation: Mapping[str, Any]) -> tuple[str, int, float, str]:
            binding = observation.get("binding")
            binding = binding if isinstance(binding, Mapping) else {}
            try:
                binding_epoch = int(binding.get("binding_epoch") or 0)
            except (TypeError, ValueError):
                binding_epoch = 0
            try:
                confidence = float(binding.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            return (
                str(binding.get("valid_from") or ""),
                binding_epoch,
                confidence,
                str(binding.get("binding_id") or ""),
            )

        current = observations.get(str(primary_binding_id or ""))
        if current is not None:
            current_distance = current.get("identity_distance")
            try:
                current_distance_value = (
                    float(current_distance)
                    if current_distance is not None
                    else None
                )
            except (TypeError, ValueError):
                current_distance_value = None
            newcomers: list[tuple[float, Mapping[str, Any]]] = []
            for observation in observations.values():
                if observation is current or not bool(
                    observation.get("new_binding")
                ):
                    continue
                value = observation.get("identity_distance")
                try:
                    distance = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(distance):
                    newcomers.append((distance, observation))
            if current_distance_value is None or not np.isfinite(
                current_distance_value
            ) or not newcomers:
                return current
            best_distance = min(item[0] for item in newcomers)
            best_newcomer = max(
                (
                    observation
                    for distance, observation in newcomers
                    if distance == best_distance
                ),
                key=rank,
            )
            return (
                best_newcomer
                if best_distance <= current_distance_value
                else current
            )
        if not complete_snapshot or not observations:
            return None

        scored: list[tuple[float, Mapping[str, Any]]] = []
        for observation in observations.values():
            value = observation.get("identity_distance")
            try:
                distance = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(distance):
                scored.append((distance, observation))
        if scored:
            best_distance = min(item[0] for item in scored)
            return max(
                (
                    observation
                    for distance, observation in scored
                    if distance == best_distance
                ),
                key=rank,
            )

        return max(observations.values(), key=rank)

    def _initialization_artifacts(
        self,
        frame: CachedShigureFrame,
        candidate: Mapping[str, Any],
        sample: CachedRgbdSample,
        display_object_id: str,
        attempt: int,
    ) -> tuple[EventArtifacts, Path]:
        root = (
            self._recovery_debug_epoch_root()
            / "initialization"
            / _safe_token(display_object_id, "display")
            / f"attempt_{attempt:03d}_sequence_{int(frame.sequence):09d}"
        )
        root.mkdir(parents=True, exist_ok=True)
        mask = _decode_full_mask(
            candidate.get("mask_b64"), sample.rgb_bgr.shape[:2]
        )
        scene_path = root / "scene.png"
        mask_path = root / "mask.png"
        depth_path = root / "depth.png"
        _write_image(scene_path, sample.rgb_bgr)
        _write_image(mask_path, mask.astype(np.uint8) * 255)
        _write_image(depth_path, np.asarray(sample.depth, dtype=np.uint16))
        box = _bbox(candidate) or _mask_bbox(mask)
        x0, y0, x1, y1 = [int(round(value)) for value in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(sample.rgb_bgr.shape[1], x1)
        y1 = min(sample.rgb_bgr.shape[0], y1)
        crop_path: Path | None = None
        if x1 > x0 and y1 > y0:
            crop = sample.rgb_bgr[y0:y1, x0:x1].copy()
            crop[~mask[y0:y1, x0:x1]] = 0
            crop_path = root / "object_crop.png"
            _write_image(crop_path, crop)
        _write_json(
            root / "report.json",
            {
                "schema_version": 1,
                "status": "RUNNING",
                "runtime_session_id": self.runtime_session_id,
                "source_epoch_id": self.source_epoch_id,
                "display_object_id": display_object_id,
                "raw_shigure_object_id": candidate.get(
                    "shigure_object_id"
                ),
                "attempt": attempt,
                "frame_sequence": int(frame.sequence),
                "source_stamp": frame.source_stamp.to_dict(),
                "candidate": self._recovery_candidate_summary(candidate),
                "scene_path": str(scene_path),
                "mask_path": str(mask_path),
                "depth_path": str(depth_path),
                "crop_path": str(crop_path) if crop_path else None,
            },
        )
        return (
            EventArtifacts(
                scene_path,
                mask_path,
                crop_path,
                sample,
                mask,
                None,
            ),
            root,
        )

    def _schedule_initial_pose(
        self,
        *,
        binding: Mapping[str, Any],
        frame: CachedShigureFrame,
        candidate: Mapping[str, Any],
        sample: CachedRgbdSample,
    ) -> bool:
        try:
            _camera_to_aruco()
        except Exception:
            self._persist_recovery_observation(
                frame,
                "initialization_waiting_for_shigure_camera_to_armarker_calibration",
            )
            return False
        display_object_id = str(binding["display_object_id"])
        state = get_display_object_state(display_object_id) or {}
        model_revision = int(state.get("active_model_revision") or 0)
        model_task_id = str(
            state.get("active_model_task_id") or ""
        ).strip()
        model_task = (
            get_task_by_task_id(model_task_id) if model_task_id else None
        )
        if (
            model_revision <= 0
            or not model_task
            or str(model_task.get("status") or "") != "completed"
        ):
            # Display identity is intentionally established before a slow
            # model-generation stage.  Waiting here must not burn one of the
            # five real FoundationPose attempts; the bounded RGB-D cache and
            # persistent recovery report retain the evidence meanwhile.
            self._persist_recovery_observation(
                frame,
                "initialization_waiting_for_completed_model_artifacts",
            )
            return False
        with self._lock:
            if (
                display_object_id in self._initial_pose_completed
                or display_object_id in self._initial_pose_inflight
            ):
                return False
            attempt = self._initial_pose_attempts.get(display_object_id, 0) + 1
            if attempt > 5:
                return False
            self._initial_pose_attempts[display_object_id] = attempt
            self._initial_pose_inflight.add(display_object_id)
            generation = int(self._source_generation)
        try:
            artifacts, root = self._initialization_artifacts(
                frame,
                candidate,
                sample,
                display_object_id,
                attempt,
            )
            self._pose_executor.submit(
                self._run_initial_pose,
                dict(binding),
                int(frame.sequence),
                artifacts,
                root,
                generation,
                attempt,
            )
            return True
        except Exception:
            with self._lock:
                self._initial_pose_inflight.discard(display_object_id)
            raise

    def _run_initial_pose(
        self,
        binding: Mapping[str, Any],
        sequence: int,
        artifacts: EventArtifacts,
        report_root: Path,
        source_generation: int,
        attempt: int,
    ) -> None:
        display_object_id = str(binding["display_object_id"])
        report_path = report_root / "report.json"
        try:
            if source_generation != self._source_generation:
                raise RuntimeError("source epoch changed during initialization")
            current = get_active_shigure_binding(
                source_epoch_id=str(binding["source_epoch_id"]),
                raw_shigure_object_id=str(binding["raw_shigure_object_id"]),
            )
            state = get_display_object_state(display_object_id) or {}
            if (
                current is None
                or str(current["binding_id"]) != str(binding["binding_id"])
                or str(state.get("active_shigure_binding_id") or "")
                != str(binding["binding_id"])
            ):
                raise RuntimeError("initialization binding is no longer primary")
            model_revision = int(state.get("active_model_revision") or 0)
            if model_revision <= 0:
                raise RuntimeError(
                    "initialization active model revision is unavailable"
                )
            pose_aruco = self._foundationpose_pose(
                display_object_id, sequence, artifacts
            )
            if source_generation != self._source_generation:
                raise RuntimeError("source epoch changed after FoundationPose")
            if str(self.source_epoch_id or "") != str(
                binding["source_epoch_id"]
            ):
                raise RuntimeError(
                    "source epoch identity changed after FoundationPose"
                )
            current = get_active_shigure_binding(
                source_epoch_id=str(binding["source_epoch_id"]),
                raw_shigure_object_id=str(
                    binding["raw_shigure_object_id"]
                ),
            )
            state = get_display_object_state(display_object_id) or {}
            if (
                current is None
                or str(current["binding_id"]) != str(binding["binding_id"])
                or str(state.get("active_shigure_binding_id") or "")
                != str(binding["binding_id"])
            ):
                raise RuntimeError(
                    "initialization binding changed during FoundationPose"
                )
            if int(state.get("active_model_revision") or 0) != model_revision:
                raise RuntimeError(
                    "initialization model revision changed during FoundationPose"
                )
            origin = add_display_object_origin(
                display_object_id=display_object_id,
                pose_aruco=pose_aruco,
                kind="INITIALIZATION",
                model_revision=model_revision,
                source_epoch_id=str(binding["source_epoch_id"]),
                binding_id=str(binding["binding_id"]),
                raw_shigure_object_id=str(
                    binding["raw_shigure_object_id"]
                ),
            )
            with self._lock:
                self._initial_pose_completed.add(display_object_id)
            _write_json(
                report_path,
                {
                    "schema_version": 1,
                    "status": "COMPLETED",
                    "runtime_session_id": self.runtime_session_id,
                    "source_epoch_id": binding["source_epoch_id"],
                    "display_object_id": display_object_id,
                    "raw_shigure_object_id": binding[
                        "raw_shigure_object_id"
                    ],
                    "attempt": attempt,
                    "frame_sequence": sequence,
                    "pose_aruco": pose_aruco,
                    "origin_id": origin.get("id"),
                    "origin_deduplicated": bool(
                        origin.get("_deduplicated")
                    ),
                    "scene_path": str(artifacts.scene),
                    "mask_path": str(artifacts.mask),
                    "depth_path": str(report_root / "depth.png"),
                },
            )
            print(
                "[shigure-v2] initialization FoundationPose committed "
                f"display={display_object_id} attempt={attempt} "
                f"report={report_path}"
            )
        except Exception as exc:
            _write_json(
                report_path,
                {
                    "schema_version": 1,
                    "status": "FAILED",
                    "runtime_session_id": self.runtime_session_id,
                    "source_epoch_id": binding.get("source_epoch_id"),
                    "display_object_id": display_object_id,
                    "raw_shigure_object_id": binding.get(
                        "raw_shigure_object_id"
                    ),
                    "attempt": attempt,
                    "frame_sequence": sequence,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "scene_path": str(artifacts.scene),
                    "mask_path": str(artifacts.mask),
                    "depth_path": str(report_root / "depth.png"),
                },
            )
            print(
                "[shigure-v2] initialization FoundationPose failed "
                f"display={display_object_id} attempt={attempt} "
                f"error={exc} report={report_path}"
            )
        finally:
            with self._lock:
                self._initial_pose_inflight.discard(display_object_id)

    def _reconcile_mask_bindings(self, frame: CachedShigureFrame) -> None:
        """Reconcile new masks to Holo identities and publish primary AABBs."""

        if not self.runtime_session_id or not self.source_epoch_id:
            return
        startup_complete_empty = self._startup_recovery_complete_empty(frame)
        if self._startup_recovery_pending and not startup_complete_empty:
            readiness_reason = self._startup_recovery_readiness_reason(frame)
            if readiness_reason is not None:
                self._record_startup_recovery_pending(
                    frame, readiness_reason
                )
                return
        sample = self._sample_exact(frame)
        display_ids = self._recent_display_ids()
        results: list[dict[str, Any]] = []
        terminal_candidate_indexes: set[int] = set()
        observed_primary_bindings: set[str] = set()
        observations_by_display: dict[
            str, dict[str, dict[str, Any]]
        ] = {}
        attempted_identity = False
        complete_snapshot = (
            frame.input_states.get("segments")
            in {"present", "explicit_empty"}
            and frame.input_states.get("object_tracking")
            in {"present", "explicit_empty"}
            and (sample is not None or not frame.recovery_candidates)
        )

        for candidate_index, candidate in enumerate(
            frame.recovery_candidates
        ):
            raw_id = str(candidate.get("shigure_object_id") or "").strip()
            if (
                not raw_id
                or str(candidate.get("tracking_match_status") or "").upper()
                != "RESOLVED"
                or sample is None
            ):
                continue
            try:
                mask = _decode_full_mask(
                    candidate.get("mask_b64"), sample.rgb_bgr.shape[:2]
                )
            except Exception as exc:
                results.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "raw_shigure_object_id": raw_id,
                        "status": "UNBOUND",
                        "reason": f"invalid_mask:{exc}",
                    }
                )
                terminal_candidate_indexes.add(candidate_index)
                continue

            binding = get_active_shigure_binding(
                source_epoch_id=self.source_epoch_id,
                raw_shigure_object_id=raw_id,
            )
            temporary: EventArtifacts | None = None
            identity_distance: float | None = None
            new_binding = False
            try:
                if binding is None:
                    if not self._mask_identity_attempt_is_new(raw_id, mask):
                        terminal_candidate_indexes.add(candidate_index)
                        results.append(
                            {
                                "candidate_id": candidate.get(
                                    "candidate_id"
                                ),
                                "raw_shigure_object_id": raw_id,
                                "status": "UNCHANGED_MASK_PRIOR_RESULT",
                                "reason": (
                                    "waiting_for_materially_new_mask_after_"
                                    "prior_identity_attempt"
                                ),
                            }
                        )
                        continue
                    attempted_identity = True
                    temporary = self._candidate_artifacts(
                        frame, candidate, sample
                    )
                    embedded = self._embed_artifacts(temporary)
                    if embedded is None:
                        raise ValueError("candidate has no query embedding")
                    identity = self._select_identity(
                        self._identity_scores(embedded[0], display_ids)
                    )
                    result = {
                        "candidate_id": candidate.get("candidate_id"),
                        "raw_shigure_object_id": raw_id,
                        **identity,
                    }
                    if identity.get("status") != "MATCHED":
                        results.append(result)
                        terminal_candidate_indexes.add(candidate_index)
                        continue
                    identity_distance = float(identity["distance"])
                    binding = establish_shigure_binding(
                        runtime_session_id=self.runtime_session_id,
                        source_epoch_id=self.source_epoch_id,
                        raw_shigure_object_id=raw_id,
                        display_object_id=str(
                            identity["display_object_id"]
                        ),
                        established_by="NEW_MASK_HOLOLENS_DINOV2",
                        confidence=max(
                            0.0, 1.0 - float(identity["distance"])
                        ),
                        detail={
                            **identity,
                            "candidate_id": candidate.get("candidate_id"),
                            "frame_sequence": int(frame.sequence),
                            "mask_is_query_only": True,
                        },
                    )
                    new_binding = True
                    results.append({**result, "status": "BOUND_ALIAS"})

                if not new_binding:
                    results.append(
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "raw_shigure_object_id": raw_id,
                            "display_object_id": binding.get(
                                "display_object_id"
                            ),
                            "binding_id": binding.get("binding_id"),
                            "status": "BOUND_EXISTING_ALIAS",
                        }
                    )

                display_object_id = str(binding["display_object_id"])
                binding_id = str(binding["binding_id"])
                observations_by_display.setdefault(
                    display_object_id, {}
                )[binding_id] = {
                    "binding": dict(binding),
                    "candidate": candidate,
                    "mask": mask,
                    "identity_distance": identity_distance,
                    "new_binding": new_binding,
                }
                terminal_candidate_indexes.add(candidate_index)
            except Exception as exc:
                results.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "raw_shigure_object_id": raw_id,
                        "status": "CONFLICT",
                        "reason": f"{exc.__class__.__name__}: {exc}",
                    }
                )
                terminal_candidate_indexes.add(candidate_index)
            finally:
                _cleanup_event_artifacts(temporary)

        # Select after consuming the complete frame.  This lets a previously
        # known alias become primary when the newer raw ID disappears, while a
        # partial snapshot can only update the already-authoritative primary.
        for display_object_id, observations in (
            observations_by_display.items()
        ):
            state = get_display_object_state(display_object_id) or {}
            primary_binding_id = str(
                state.get("active_shigure_binding_id") or ""
            )
            primary_observation = observations.get(primary_binding_id)
            if (
                primary_observation is not None
                and primary_observation.get("identity_distance") is None
                and any(
                    bool(observation.get("new_binding"))
                    for observation in observations.values()
                )
            ):
                primary_artifacts: EventArtifacts | None = None
                try:
                    primary_artifacts = self._candidate_artifacts(
                        frame,
                        primary_observation["candidate"],
                        sample,
                    )
                    embedded = self._embed_artifacts(primary_artifacts)
                    if embedded is None:
                        raise ValueError("primary mask has no query embedding")
                    primary_scores = self._identity_scores(
                        embedded[0], [display_object_id]
                    )
                    if primary_scores:
                        primary_observation["identity_distance"] = float(
                            primary_scores[0]["distance"]
                        )
                except Exception as exc:
                    results.append(
                        {
                            "display_object_id": display_object_id,
                            "binding_id": primary_binding_id,
                            "status": "PRIMARY_RETAINED",
                            "reason": (
                                "primary_competition_dino_unavailable:"
                                f"{exc.__class__.__name__}:{exc}"
                            ),
                        }
                    )
                finally:
                    _cleanup_event_artifacts(primary_artifacts)
            selected = self._select_primary_mask_observation(
                observations,
                primary_binding_id,
                complete_snapshot=complete_snapshot,
            )
            if selected is None:
                continue
            binding = selected["binding"]
            binding_id = str(binding["binding_id"])
            raw_id = str(binding["raw_shigure_object_id"])
            latest_state = get_display_object_state(display_object_id) or {}
            if (
                binding_id
                != str(latest_state.get("active_shigure_binding_id") or "")
                or str(latest_state.get("presence") or "") != "PRESENT"
            ):
                try:
                    activate_recovered_shigure_binding(binding_id)
                    results.append(
                        {
                            "raw_shigure_object_id": raw_id,
                            "display_object_id": display_object_id,
                            "binding_id": binding_id,
                            "status": "PRIMARY_ALIAS_PROMOTED",
                            "reason": (
                                "lower_dino_distance_or_previous_primary_missing"
                            ),
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "raw_shigure_object_id": raw_id,
                            "display_object_id": display_object_id,
                            "binding_id": binding_id,
                            "status": "CONFLICT",
                            "reason": (
                                "alias_promotion_failed:"
                                f"{exc.__class__.__name__}:{exc}"
                            ),
                        }
                    )
                    continue

            try:
                self._remember_trusted_mask_observation(
                    binding=binding,
                    frame=frame,
                    sample=sample,
                    mask=np.asarray(selected["mask"], dtype=bool),
                    identity_distance=selected.get("identity_distance"),
                )
            except Exception as exc:
                print(
                    "[shigure-v2] trusted mask history unavailable "
                    f"display={display_object_id} raw={raw_id}: {exc}"
                )
            box_handled = False
            try:
                box_handled = self._commit_primary_mask_box(
                    binding,
                    frame,
                    sample,
                    np.asarray(selected["mask"], dtype=bool),
                )
            except Exception as exc:
                print(
                    "[shigure-v2] mask AABB unavailable; clearing stale box "
                    f"display={display_object_id} raw={raw_id}: {exc}"
                )
                try:
                    box_handled = self._clear_primary_mask_box(
                        binding, frame
                    )
                except Exception as clear_exc:
                    print(
                        "[shigure-v2] mask AABB clear failed "
                        f"display={display_object_id} raw={raw_id}: "
                        f"{clear_exc}"
                    )
            if box_handled:
                observed_primary_bindings.add(binding_id)
            try:
                self._schedule_initial_pose(
                    binding=binding,
                    frame=frame,
                    candidate=selected["candidate"],
                    sample=sample,
                )
            except Exception as exc:
                results.append(
                    {
                        "raw_shigure_object_id": raw_id,
                        "display_object_id": display_object_id,
                        "binding_id": binding_id,
                        "status": "CONFLICT",
                        "reason": (
                            "initialization_schedule_failed:"
                            f"{exc.__class__.__name__}:{exc}"
                        ),
                    }
                )

        if complete_snapshot:
            for binding in list_active_shigure_bindings(
                self.source_epoch_id
            ):
                state = get_display_object_state(
                    str(binding["display_object_id"])
                ) or {}
                binding_id = str(binding["binding_id"])
                if (
                    str(state.get("active_shigure_binding_id") or "")
                    != binding_id
                    or binding_id in observed_primary_bindings
                ):
                    continue
                model_revision = int(
                    state.get("active_model_revision") or 0
                )
                if model_revision <= 0:
                    continue
                update_shigure_live_observation(
                    binding_id=binding_id,
                    observation_seq=int(frame.sequence),
                    model_revision=model_revision,
                    spatial_box_corners_aruco=None,
                    spatial_box_observed=True,
                )

        all_candidates_terminal = (
            len(terminal_candidate_indexes)
            == len(frame.recovery_candidates)
        )
        startup_recovery_complete = complete_snapshot and (
            startup_complete_empty or all_candidates_terminal
        )
        if (
            self._startup_recovery_pending
            and complete_snapshot
            and not startup_recovery_complete
        ):
            self._record_startup_recovery_pending(
                frame, "waiting_for_terminal_candidate_results"
            )
        should_report = attempted_identity or (
            self._startup_recovery_pending and startup_recovery_complete
        )
        if should_report:
            root = (
                self._recovery_debug_epoch_root()
                / "mask_reconcile"
                / f"sequence_{int(frame.sequence):09d}"
            )
            input_artifacts = self._write_recovery_input_artifacts(
                root, frame, sample
            )
            report = {
                "schema_version": 1,
                "status": "COMPLETED",
                "runtime_session_id": self.runtime_session_id,
                "source_epoch_id": self.source_epoch_id,
                "frame_sequence": int(frame.sequence),
                "source_stamp": frame.source_stamp.to_dict(),
                "identity_policy": {
                    "reference_source": "HOLOLENS_ONLY",
                    "views_per_display_object": (
                        SHIGURE_IDENTITY_MAX_HOLOLENS_REFERENCES
                    ),
                    "new_mask_only": True,
                    "query_masks_persisted_as_references": False,
                },
                "candidate_count": len(frame.recovery_candidates),
                "results": results,
                "candidates": input_artifacts,
            }
            _write_json(root / "report.json", report)
            if (
                self._startup_recovery_pending
                and startup_recovery_complete
            ):
                job_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        f"shigure-recovery:{self.runtime_session_id}:"
                        f"{self.source_epoch_id}"
                    ),
                ).hex
                upsert_identity_sync_job(
                    sync_job_id=job_id,
                    kind="STARTUP_RECOVERY",
                    status="COMPLETED",
                    runtime_session_id=self.runtime_session_id,
                    source_epoch_id=self.source_epoch_id,
                    candidate_limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
                    result={
                        **report,
                        "debug_report_path": str(root / "report.json"),
                    },
                )
                self._startup_recovery_pending = False
            print(
                "[shigure-v2] mask reconcile report "
                f"results={len(results)} report={root / 'report.json'}"
            )

    def queue_hololens_capture_sync(
        self,
        *,
        display_object_id: str,
        task_id: str,
        task_json_path: str | Path,
    ) -> dict[str, Any]:
        """Queue latest-wins matching; only Shigure recovery may activate presence."""

        path = resolve_project_path(task_json_path, require_exists=True)
        task = load_task_json(path)
        job_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"shigure-hololens-sync:{task_id}",
        ).hex

        # The upload pipeline has already persisted the HoloLens identity
        # reference.  Runtime binding is intentionally deferred until a new
        # Shigure mask appears; no geometry/stability gate may bind here.
        payload = {
            "status": "COMPLETED",
            "sync_job_id": job_id,
            "display_object_id": str(display_object_id),
            "task_id": str(task_id),
            "reason": "hololens_reference_registered_waiting_for_new_shigure_mask",
            "attempts": 0,
            "method": "MASK_TRIGGERED_DINOV2",
            "lifecycle_authority": "shigure_mask_reconcile",
            "lifecycle_binding_changed": False,
            "updated_utc": _utc_now(),
        }
        task["ShigureIdentitySync"] = payload
        _write_json(path, task)
        upsert_identity_sync_job(
            sync_job_id=job_id,
            kind="HOLOLENS_CAPTURE",
            status="COMPLETED",
            runtime_session_id=self.runtime_session_id,
            source_epoch_id=self.source_epoch_id,
            display_object_id=str(display_object_id),
            candidate_limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
            result=payload,
        )
        return payload

        def fail_validation(reason: str) -> dict[str, Any]:
            payload = {
                "status": "FAILED",
                "sync_job_id": job_id,
                "display_object_id": str(display_object_id),
                "task_id": str(task_id),
                "reason": str(reason),
                "attempts": 0,
                "method": None,
                "lifecycle_authority": None,
                "lifecycle_binding_changed": False,
                "updated_utc": _utc_now(),
            }
            task["ShigureIdentitySync"] = payload
            _write_json(path, task)
            try:
                upsert_identity_sync_job(
                    sync_job_id=job_id,
                    kind="HOLOLENS_CAPTURE",
                    status="FAILED",
                    runtime_session_id=self.runtime_session_id,
                    source_epoch_id=self.source_epoch_id,
                    display_object_id=str(display_object_id),
                    candidate_limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
                    result=payload,
                    error_message=str(reason),
                )
            except Exception as exc:
                print(f"[shigure-v2] failed to persist invalid Holo sync job: {exc}")
            print(
                "[shigure-v2] Holo identity sync terminal: "
                f"task={task_id} display={display_object_id} status=FAILED "
                f"method=None raw=None candidate=None reason={reason} "
                f"sync_job_id={job_id}"
            )
            return payload

        bounds = (
            task.get("ModelBounds")
            if isinstance(task.get("ModelBounds"), Mapping)
            else None
        )
        target_center = None
        target_size = None
        if bounds is not None:
            try:
                minimum = np.asarray(
                    bounds["aabb_min_aruco"], dtype=np.float64
                ).reshape(3)
                maximum = np.asarray(
                    bounds["aabb_max_aruco"], dtype=np.float64
                ).reshape(3)
            except (KeyError, TypeError, ValueError):
                return fail_validation("hololens_model_bounds_invalid")
            size = maximum - minimum
            if (
                not np.isfinite(minimum).all()
                or not np.isfinite(maximum).all()
                or np.any(size <= 0.0)
            ):
                return fail_validation("hololens_model_bounds_invalid")
            target_center = (minimum + maximum) * 0.5
            target_size = size
        job = PendingHoloSync(
            job_id=job_id,
            display_object_id=str(display_object_id),
            task_id=str(task_id),
            task_json_path=path,
            target_center_aruco=target_center,
            target_size_aruco=target_size,
            source_generation=self._source_generation,
        )
        old: PendingHoloSync | None
        with self._lock:
            old = self._pending_holo_syncs.get(job.display_object_id)
            self._pending_holo_syncs[job.display_object_id] = job
        if old is not None and old.job_id != job.job_id:
            self._record_holo_sync_status(
                old,
                status="FAILED",
                reason="superseded_by_newer_hololens_capture",
                terminal=True,
            )
        return self._record_holo_sync_status(
            job,
            status="PENDING",
            reason="waiting_for_stable_recovery_frame",
            terminal=False,
        )

    def resume_unbound_hololens_capture_syncs(self) -> int:
        """Requeue latest active model syncs that have no Shigure binding."""

        return 0

        resumed = 0
        # Resume only the most recently updated active model. Older unbound
        # models are not evidence that they are still present, and replaying
        # all of them can monopolize identity work and starve HTTP polling.
        for state in list_display_object_states(limit=1):
            if str(state.get("active_shigure_binding_id") or "").strip():
                continue
            display_object_id = str(state.get("display_object_id") or "").strip()
            task_id = str(state.get("active_model_task_id") or "").strip()
            if not display_object_id or not task_id:
                continue
            try:
                task_record = get_task_by_task_id(task_id)
                if not task_record or str(task_record.get("status") or "") != "completed":
                    continue
                task_json_path = resolve_task_json_path_from_record(task_record)
                # A COMPLETED sync belongs to its source epoch. Once a new
                # epoch revokes that binding, the same active model must be
                # matched again against the new raw-ID namespace.
                self.queue_hololens_capture_sync(
                    display_object_id=display_object_id,
                    task_id=task_id,
                    task_json_path=task_json_path,
                )
                resumed += 1
            except Exception as exc:
                print(
                    "[shigure-v2] failed to resume Holo identity sync: "
                    f"task={task_id} display={display_object_id} error={exc}"
                )
        return resumed

    def _record_holo_sync_status(
        self,
        job: PendingHoloSync,
        *,
        status: str,
        reason: str,
        terminal: bool,
        result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": str(status),
            "sync_job_id": job.job_id,
            "display_object_id": job.display_object_id,
            "task_id": job.task_id,
            "reason": str(reason),
            "attempts": int(job.attempts),
            "source_epoch_id": self.source_epoch_id,
            "updated_utc": _utc_now(),
        }
        if result:
            payload.update(dict(result))
        upsert_identity_sync_job(
            sync_job_id=job.job_id,
            kind="HOLOLENS_CAPTURE",
            status=str(status),
            runtime_session_id=self.runtime_session_id,
            source_epoch_id=self.source_epoch_id,
            display_object_id=job.display_object_id,
            candidate_limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
            result=payload,
            error_message=reason if status == "FAILED" else None,
        )
        try:
            task = load_task_json(job.task_json_path)
            task["ShigureIdentitySync"] = payload
            _write_json(job.task_json_path, task)
        except Exception as exc:
            print(f"[shigure-v2] failed to write Holo sync task status: {exc}")
        if terminal:
            print(
                "[shigure-v2] Holo identity sync terminal: "
                f"task={job.task_id} display={job.display_object_id} "
                f"status={status} method={payload.get('method')} "
                f"raw={payload.get('raw_shigure_object_id')} "
                f"candidate={payload.get('candidate_id')} "
                f"reason={reason} sync_job_id={job.job_id}"
            )
            with self._lock:
                if self._pending_holo_syncs.get(job.display_object_id) is job:
                    self._pending_holo_syncs.pop(job.display_object_id, None)
        return payload

    def _schedule_hololens_syncs(self, frame: CachedShigureFrame) -> None:
        return
        # Holo identity matching can only make a decision from a complete,
        # non-empty recovery snapshot. Segments/tracking/RGB-D arrive as
        # separate same-stamp revisions; missing or explicit-empty revisions
        # are waiting states and must not exhaust the retry budget before the
        # enriched revision arrives (or before the target enters the view).
        if frame.input_states.get("segments") != "present":
            return
        if frame.input_states.get("object_tracking") != "present":
            return
        if not frame.recovery_candidates:
            return
        if self._sample_exact(frame) is None:
            return
        with self._lock:
            jobs = list(self._pending_holo_syncs.values())
            for job in jobs:
                if job.running:
                    continue
                job.running = True
                source_generation = job.source_generation
                try:
                    self._identity_executor.submit(
                        self._attempt_hololens_sync_guarded,
                        job,
                        frame,
                        source_generation,
                    )
                except Exception:
                    job.running = False
                    raise

    def _holo_sync_attempt_current(
        self,
        job: PendingHoloSync,
        source_generation: int,
    ) -> bool:
        with self._lock:
            return (
                self._pending_holo_syncs.get(job.display_object_id) is job
                and self._source_generation == source_generation
                and job.source_generation == source_generation
            )

    def _attempt_hololens_sync_guarded(
        self,
        job: PendingHoloSync,
        frame: CachedShigureFrame,
        source_generation: int,
    ) -> None:
        try:
            self._attempt_hololens_sync(job, frame, source_generation)
        except Exception as exc:
            self._pending_or_fail_holo_sync(
                job,
                f"sync_attempt_failed:{exc}",
                source_generation,
            )
        finally:
            with self._lock:
                job.running = False
                temporary_roots = list(job.temporary_roots)
                job.temporary_roots.clear()
            for root in temporary_roots:
                shutil.rmtree(root, ignore_errors=True)


    def _pending_or_fail_holo_sync(
        self,
        job: PendingHoloSync,
        reason: str,
        source_generation: int,
    ) -> None:
        if not self._holo_sync_attempt_current(job, source_generation):
            return
        job.last_reason = str(reason)
        terminal = job.attempts >= SHIGURE_HOLO_SYNC_MAX_RECOVERY_ATTEMPTS
        self._record_holo_sync_status(
            job,
            status="FAILED" if terminal else "PENDING",
            reason=(f"retry_limit_reached:{reason}" if terminal else reason),
            terminal=terminal,
        )

    def _reject_hololens_sync_frame(
        self,
        job: PendingHoloSync,
        reason: str,
        source_generation: int,
        source_key: str,
    ) -> None:
        with self._lock:
            if (
                self._holo_sync_attempt_current(job, source_generation)
                and job.last_stable_source_key != source_key
            ):
                job.stable_raw_id = ""
                job.stable_bbox = None
                job.stable_mask = None
                job.stable_history.clear()
                job.stable_count = 0
                job.identity_method = ""
                job.identity_dino_detail.clear()
                job.last_stable_source_key = source_key
        self._pending_or_fail_holo_sync(
            job,
            reason,
            source_generation,
        )

    def _attempt_hololens_sync(
        self,
        job: PendingHoloSync,
        frame: CachedShigureFrame,
        source_generation: int,
    ) -> None:
        source_key = sample_key(frame.source_stamp)
        if not self._holo_sync_attempt_current(job, source_generation):
            return
        if not frame.recovery_candidates:
            self._reject_hololens_sync_frame(
                job,
                "no_recovery_candidates",
                source_generation,
                source_key,
            )
            return
        sample = self._sample_exact(frame)
        if sample is None:
            self._reject_hololens_sync_frame(
                job,
                "exact_rgbd_unavailable",
                source_generation,
                source_key,
            )
            return
        try:
            camera_to_aruco, _rotation, _translation, _revision = _camera_to_aruco()
        except Exception as exc:
            self._reject_hololens_sync_frame(
                job,
                f"aruco_calibration_unavailable:{exc}",
                source_generation,
                source_key,
            )
            return

        # Collider geometry is cheap and does not need image artifacts. Gate
        # first so unrelated Segments masks are never decoded/written merely
        # to be rejected by ArUco position and size.
        plausible_geometry: list[dict[str, Any]] = []
        trusted_dino_fallback: list[dict[str, Any]] = []
        for candidate in frame.recovery_candidates:
            tracking = (
                candidate.get("tracking")
                if isinstance(candidate.get("tracking"), Mapping)
                else {}
            )
            geometry: dict[str, Any] | None = None
            try:
                box = build_spatial_box_v2(
                    tracking.get("collider"), camera_to_aruco, np.eye(4)
                )
                if (
                    job.target_center_aruco is None
                    or job.target_size_aruco is None
                ):
                    raise NoBoxError(
                        "HoloLens model bounds are not ready; use DINO"
                    )
                corners = np.asarray(box["corners_aruco_m"], dtype=np.float64)
                minimum, maximum = corners.min(axis=0), corners.max(axis=0)
                center, size = (minimum + maximum) * 0.5, maximum - minimum
                center_distance = float(
                    np.linalg.norm(center - job.target_center_aruco)
                )
                size_log_error = float(
                    np.max(np.abs(np.log(size / job.target_size_aruco)))
                )
                geometry = {
                    "center_distance_m": center_distance,
                    "size_log_error": size_log_error,
                    "plausible": (
                        center_distance <= SHIGURE_HOLO_SYNC_CENTER_DISTANCE_M
                        and size_log_error <= SHIGURE_HOLO_SYNC_SIZE_LOG_TOLERANCE
                    ),
                }
            except Exception:
                pass
            item = {
                "candidate": candidate,
                "geometry": geometry,
                "spatial_box_corners_aruco": (
                    corners.astype(float).tolist()
                    if geometry is not None
                    else None
                ),
            }
            raw_id = str(candidate.get("shigure_object_id") or "").strip()
            trusted_mapping = bool(
                raw_id
                and str(candidate.get("tracking_match_status") or "").upper()
                == "RESOLVED"
                and str(candidate.get("tracking_mapping_method") or "")
                == "SEGMENT_TRACKING_UNIQUE_IOU"
                and str(tracking.get("action") or "").strip().lower()
                in {"stay", "bring_in"}
            )
            if trusted_mapping:
                trusted_dino_fallback.append(item)
                if geometry is not None and geometry["plausible"]:
                    plausible_geometry.append(item)

        force_dino_fallback = not plausible_geometry
        reuse_prior_identity = False
        candidate_pool = (
            plausible_geometry
            if plausible_geometry
            else trusted_dino_fallback
        )
        if job.stable_raw_id and job.identity_method:
            prior = next(
                (
                    item
                    for item in trusted_dino_fallback
                    if str(
                        item["candidate"].get("shigure_object_id") or ""
                    ).strip()
                    == job.stable_raw_id
                ),
                None,
            )
            if prior is not None:
                # The first complete frame already made the expensive identity
                # decision. Confirm the same trusted tracking ID and strict
                # mask stability without embedding every scene candidate again.
                candidate_pool = [prior]
                reuse_prior_identity = True
        if not candidate_pool:
            self._reject_hololens_sync_frame(
                job,
                "no_trusted_candidate_for_geometry_or_dinov2",
                source_generation,
                source_key,
            )
            return

        plausible: list[dict[str, Any]] = []
        for item in candidate_pool:
            try:
                artifacts = self._candidate_artifacts(
                    frame, item["candidate"], sample
                )
                if artifacts.temporary_root is not None:
                    job.temporary_roots.append(artifacts.temporary_root)
            except Exception:
                continue
            plausible.append({**item, "artifacts": artifacts})
        if not plausible:
            self._reject_hololens_sync_frame(
                job,
                "no_usable_recovery_candidates",
                source_generation,
                source_key,
            )
            return

        selected: dict[str, Any] | None = None
        method = ""
        dino_detail: dict[str, Any] = {}

        if reuse_prior_identity:
            selected = plausible[0]
            method = job.identity_method
            dino_detail = dict(job.identity_dino_detail)
        elif len(plausible) == 1 and not force_dino_fallback:
            selected = plausible[0]
            method = "ARUCO_COLLIDER_GEOMETRY"
        else:
            references = [
                item for item in list_object_identity_references(job.display_object_id, limit=100)
                if str(item.get("source") or "") == "HOLOLENS"
            ]
            target = next((self._ensure_reference_embedding(item) for item in references if item), None)
            if target is None:
                self._reject_hololens_sync_frame(
                    job,
                    "hololens_reference_embedding_unavailable",
                    source_generation,
                    source_key,
                )
                return
            scored: list[tuple[float, dict[str, Any], np.ndarray, dict[str, Any]]] = []
            for item in plausible:
                embedded = self._embed_artifacts(item["artifacts"])
                if embedded is None:
                    continue
                scored.append((cosine_distance(target, embedded[0]), item, embedded[0], embedded[1]))
            scored.sort(key=lambda row: row[0])
            if not scored or scored[0][0] > SHIGURE_HOLO_SYNC_DINO_DISTANCE_THRESHOLD:
                self._reject_hololens_sync_frame(
                    job,
                    "dinov2_fallback_above_wide_threshold",
                    source_generation,
                    source_key,
                )
                return
            margin = scored[1][0] - scored[0][0] if len(scored) > 1 else None
            if margin is not None and margin < SHIGURE_HOLO_SYNC_DINO_MARGIN:
                self._reject_hololens_sync_frame(
                    job,
                    "dinov2_fallback_ambiguous",
                    source_generation,
                    source_key,
                )
                return
            selected = scored[0][1]
            selected["embedded"] = (scored[0][2], scored[0][3])
            method = "DINOV2_FALLBACK"
            dino_detail = {"distance": float(scored[0][0]), "second_margin": margin}

        assert selected is not None
        candidate = selected["candidate"]
        raw_id = str(candidate.get("shigure_object_id") or "").strip()
        tracking = (
            candidate.get("tracking")
            if isinstance(candidate.get("tracking"), Mapping)
            else {}
        )
        try:
            segment_probability = float(candidate.get("probability") or 0.0)
        except (TypeError, ValueError):
            segment_probability = 0.0
        raw_id_is_trusted = (
            raw_id
            and str(candidate.get("tracking_match_status") or "").upper()
            == "RESOLVED"
            and str(candidate.get("tracking_mapping_method") or "")
            == "SEGMENT_TRACKING_UNIQUE_IOU"
            and str(tracking.get("action") or "").strip().lower()
            in {"stay", "bring_in"}
        )
        if not raw_id_is_trusted:
            self._reject_hololens_sync_frame(
                job,
                "stable_candidate_has_no_trusted_shigure_raw_id",
                source_generation,
                source_key,
            )
            return
        # Only a plausible candidate with a trusted epoch-local raw ID is an
        # identity attempt. Empty views, unrelated scene objects, and partial
        # segment/tracking mappings must wait without exhausting the budget.
        with self._lock:
            if not self._holo_sync_attempt_current(job, source_generation):
                return
            if job.last_attempt_source_key != source_key:
                job.attempts += 1
                job.last_attempt_source_key = source_key
        candidate_bbox = _bbox(candidate)
        candidate_mask = selected["artifacts"].mask_array
        stable = False
        with self._lock:
            if not self._holo_sync_attempt_current(job, source_generation):
                return
            repeated_source = job.last_stable_source_key == source_key
            if candidate_bbox is None or candidate_mask is None:
                job.stable_history.clear()
            else:
                current = (
                    raw_id,
                    candidate_bbox,
                    candidate_mask.copy(),
                )
                prior = (
                    tuple(job.stable_history)[:-1]
                    if repeated_source and job.stable_history
                    else tuple(job.stable_history)
                )
                agrees_with_window = all(
                    prior_raw_id == raw_id
                    and _strict_example_pair_is_stable(
                        prior_bbox,
                        prior_mask,
                        candidate_bbox,
                        candidate_mask,
                    )
                    for prior_raw_id, prior_bbox, prior_mask in prior
                )
                if not agrees_with_window:
                    job.stable_history.clear()
                if repeated_source and job.stable_history:
                    job.stable_history[-1] = current
                else:
                    job.stable_history.append(current)
            job.stable_raw_id = raw_id
            job.identity_method = method
            job.identity_dino_detail = dict(dino_detail)
            job.stable_bbox = candidate_bbox
            job.stable_mask = (
                candidate_mask.copy()
                if candidate_mask is not None
                else None
            )
            job.last_stable_source_key = source_key
            job.stable_count = len(job.stable_history)
            stable = (
                job.stable_count
                >= SHIGURE_HOLO_SYNC_STABLE_MASK_FRAMES
            )
        if not stable:
            self._pending_or_fail_holo_sync(
                job,
                "candidate_not_yet_stable",
                source_generation,
            )
            return

        # Fast Holo sync only authorizes the epoch-local binding. Long-term
        # Shigure identity examples retain the stricter five-frame admission
        # window and are collected independently after the box is live.
        example_window_complete = (
            SHIGURE_HOLO_SYNC_STABLE_MASK_FRAMES
            >= SHIGURE_EXAMPLE_STABLE_MASK_FRAMES
        )
        embedded = selected.get("embedded")
        if example_window_complete and embedded is None:
            embedded = self._embed_artifacts(selected["artifacts"])
        if example_window_complete and embedded is None:
            self._reject_hololens_sync_frame(
                job,
                "selected_candidate_embedding_unavailable",
                source_generation,
                source_key,
            )
            return
        if not self._holo_sync_attempt_current(job, source_generation):
            return

        binding = get_active_shigure_binding(
            source_epoch_id=str(self.source_epoch_id),
            raw_shigure_object_id=raw_id,
        )
        if binding is not None and str(binding["display_object_id"]) != job.display_object_id:
            self._record_holo_sync_status(
                job,
                status="FAILED",
                reason="trusted_raw_id_binding_conflict",
                terminal=True,
                result={
                    "method": method,
                    "candidate_id": candidate.get("candidate_id"),
                    "raw_shigure_object_id": raw_id,
                    "conflicting_display_object_id": binding["display_object_id"],
                    "lifecycle_authority": "shigure_recovery_snapshot",
                    "lifecycle_binding_changed": False,
                },
            )
            return

        binding_established = False
        if binding is None:
            confidence = (
                max(0.0, 1.0 - float(dino_detail["distance"]))
                if "distance" in dino_detail
                else max(
                    0.0,
                    1.0
                    - float((selected.get("geometry") or {}).get("center_distance_m") or 0.0)
                    / SHIGURE_HOLO_SYNC_CENTER_DISTANCE_M,
                )
            )
            try:
                binding = establish_shigure_binding(
                    runtime_session_id=str(self.runtime_session_id),
                    source_epoch_id=str(self.source_epoch_id),
                    raw_shigure_object_id=raw_id,
                    display_object_id=job.display_object_id,
                    established_by="HOLOLENS_SYNC_SHIGURE_RECOVERY",
                    confidence=confidence,
                    detail={
                        "lifecycle_authority": "shigure_recovery_snapshot",
                        "candidate_id": candidate.get("candidate_id"),
                        "method": method,
                        "geometry": selected.get("geometry"),
                        "dinov2": dino_detail or None,
                    },
                )
                binding_established = True
            except Exception as exc:
                self._record_holo_sync_status(
                    job,
                    status="FAILED",
                    reason=f"binding_conflict_or_establish_failed:{exc}",
                    terminal=True,
                    result={
                        "method": method,
                        "candidate_id": candidate.get("candidate_id"),
                        "raw_shigure_object_id": raw_id,
                        "lifecycle_authority": "shigure_recovery_snapshot",
                        "lifecycle_binding_changed": False,
                    },
                )
                return

        state = get_display_object_state(job.display_object_id) or {}
        presence_activated = (
            str(state.get("presence") or "") != "PRESENT"
            or str(state.get("active_shigure_binding_id") or "")
            != str(binding["binding_id"])
        )
        if presence_activated:
            activate_recovered_shigure_binding(str(binding["binding_id"]))

        if not example_window_complete:
            example_admission = {
                "admission_status": "REJECTED",
                "admission_reason": "strict_stable_view_window_incomplete",
                "admission_stable_frames": SHIGURE_HOLO_SYNC_STABLE_MASK_FRAMES,
                "admission_required_frames": SHIGURE_EXAMPLE_STABLE_MASK_FRAMES,
            }
            reference = None
        elif segment_probability < SHIGURE_EXAMPLE_MIN_SEGMENT_PROBABILITY:
            example_admission = {
                "admission_status": "REJECTED",
                "admission_reason": "segment_probability_threshold",
                "admission_probability": segment_probability,
                "admission_probability_threshold": (
                    SHIGURE_EXAMPLE_MIN_SEGMENT_PROBABILITY
                ),
            }
            reference = None
        else:
            assert embedded is not None
            example_admission = self._verify_shigure_view_admission(
                job.display_object_id, embedded[0]
            )
            reference = self._register_view(
                display_object_id=job.display_object_id,
                artifacts=selected["artifacts"],
                embedding=embedded[0],
                embedding_response=embedded[1],
                source_event_uid=None,
                quality={
                    "role": "hololens_capture_sync",
                    "method": method,
                    "lifecycle_authority": "shigure_recovery_snapshot",
                    "stable_mask_frames": SHIGURE_HOLO_SYNC_STABLE_MASK_FRAMES,
                    "segment_probability": segment_probability,
                    **dino_detail,
                },
                source_epoch_id=str(self.source_epoch_id),
                binding_id=str(binding["binding_id"]),
                raw_shigure_object_id=raw_id,
                admission=example_admission,
            )
        print(
            "[shigure-v2] Holo sync strict Shigure example admission: "
            f"display_object_id={job.display_object_id} "
            f"status={example_admission.get('admission_status')} "
            f"reason={example_admission.get('admission_reason')} "
            f"distance={example_admission.get('admission_distance')} "
            f"margin={example_admission.get('admission_margin')}"
        )
        if not self._holo_sync_attempt_current(job, source_generation):
            return
        self._record_holo_sync_status(
            job,
            status="COMPLETED",
            reason=(
                "matched_shigure_recovery_view"
                if reference is not None
                else "matched_shigure_recovery_view_example_rejected"
            ),
            terminal=True,
            result={
                "method": method,
                "candidate_id": candidate.get("candidate_id"),
                "raw_shigure_object_id": raw_id,
                "geometry": selected.get("geometry"),
                "dinov2": dino_detail or None,
                "identity_reference_id": reference.get("reference_id") if reference else None,
                "shigure_example_admission": example_admission,
                "binding_id": binding["binding_id"],
                "binding_established": binding_established,
                "presence_activated": presence_activated,
                "lifecycle_authority": "shigure_recovery_snapshot",
                "lifecycle_binding_changed": binding_established or presence_activated,
            },
        )

    def _process_frame(self, frame: CachedShigureFrame) -> None:
        frame_stamp = (
            int(frame.source_stamp.sec),
            int(frame.source_stamp.nanosec),
        )
        tainted_by_obj_move = any(
            str(item.get("action") or "").strip().lower() == "obj_move"
            for item in (*frame.events, *frame.tracked_objects)
            if isinstance(item, Mapping)
        )
        barrier = self._runtime_obj_move_barrier_stamp
        if tainted_by_obj_move:
            if barrier is not None and frame_stamp <= barrier:
                return
            if not self._runtime_obj_move_quarantine_active:
                # Defense in depth for a hand-built/internal canonical frame
                # that bypasses the compatibility adapter. Revoke every old
                # raw-ID binding and invalidate in-flight identity/pose work.
                self._open_runtime_obj_move_epoch(frame)
            else:
                # A newer move while quarantined advances the fail-closed
                # watermark but does not create another source epoch.
                self._runtime_obj_move_barrier_stamp = frame_stamp
                self._last_geometry_stamp = frame_stamp
            # Nothing else in a tainted frame may drive recovery, another
            # lifecycle event, DINO, geometry, pose, or HoloLens sync. Keep
            # only disjoint rejected obj_move audit rows.
            for position, event in enumerate(frame.events):
                if (
                    isinstance(event, Mapping)
                    and str(event.get("action") or "").strip().lower()
                    == "obj_move"
                ):
                    self._process_event(frame, event, position)
            return

        barrier = self._runtime_obj_move_barrier_stamp
        if barrier is not None and frame_stamp <= barrier:
            return
        if self._runtime_obj_move_quarantine_active:
            if (
                frame.input_states.get("object_tracking")
                not in {"present", "explicit_empty"}
            ):
                return
            self._runtime_obj_move_quarantine_active = False

        self._observe_lifecycle_confirmation_frame(frame)

        lifecycle_in_frame = any(
            str(event.get("action") or "").strip().lower()
            in {"take_out", "bring_in"}
            for event in frame.events
            if isinstance(event, Mapping)
        )
        with self._lock:
            lifecycle_window_pending = bool(
                self._lifecycle_pose_windows
            )
        # Reconciliation can establish/activate a new mask binding, promote
        # an alias, clear a box, or schedule initialization.  None of those
        # writes may run before the lifecycle no-move decision for this frame
        # (or while an earlier one-second window is still open).
        if not lifecycle_in_frame and not lifecycle_window_pending:
            self._reconcile_mask_bindings(frame)
        ordered_events = sorted(
            enumerate(frame.events),
            key=lambda item: (
                {
                    "take_out": 0,
                    "bring_in": 1,
                }.get(
                    str(item[1].get("action") or "").strip().lower(),
                    2,
                ),
                item[0],
            ),
        )
        for position, event in ordered_events:
            self._process_event(frame, event, position)

    def _masked_center_aruco(
        self, artifacts: EventArtifacts
    ) -> np.ndarray | None:
        sample = artifacts.sample
        mask = artifacts.mask_array
        if sample is None or mask is None:
            return None
        depth = np.asarray(sample.depth)
        if depth.shape[:2] != mask.shape[:2]:
            return None
        valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
        ys, xs = np.nonzero(valid)
        if xs.size < 32:
            return None
        if xs.size > 20000:
            indexes = np.linspace(0, xs.size - 1, 20000).astype(np.int64)
            xs, ys = xs[indexes], ys[indexes]
        z = depth[ys, xs].astype(np.float64) / 1000.0
        k = np.asarray(sample.camera_info.get("k"), dtype=np.float64).reshape(3, 3)
        points = np.column_stack(
            (
                (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0],
                (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1],
                z,
            )
        )
        center_camera = np.median(points, axis=0)
        transform = np.asarray(_camera_to_aruco()[0], dtype=np.float64)
        center = transform @ np.append(center_camera, 1.0)
        if not np.isfinite(center).all() or abs(float(center[3])) < 1.0e-9:
            return None
        return center[:3] / center[3]

    @staticmethod
    def _mask_bbox_from_array(
        mask: np.ndarray,
    ) -> tuple[float, float, float, float] | None:
        ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
        if xs.size == 0:
            return None
        return (
            float(xs.min()),
            float(ys.min()),
            float(xs.max() + 1),
            float(ys.max() + 1),
        )

    @staticmethod
    def _object_overlap_ratio(
        object_box: Sequence[float] | None,
        other_box: Sequence[float] | None,
    ) -> float:
        if object_box is None or other_box is None:
            return 0.0
        ox0, oy0, ox1, oy1 = (float(value) for value in object_box)
        px0, py0, px1, py1 = (float(value) for value in other_box)
        object_area = max(0.0, ox1 - ox0) * max(0.0, oy1 - oy0)
        if object_area <= 0.0:
            return 0.0
        intersection = max(0.0, min(ox1, px1) - max(ox0, px0)) * max(
            0.0, min(oy1, py1) - max(oy0, py0)
        )
        return min(1.0, intersection / object_area)

    @classmethod
    def _build_trusted_mask_observation(
        cls,
        *,
        display_object_id: str,
        binding_id: str,
        raw_id: str,
        sequence: int,
        stamp_seconds: float,
        sample: CachedRgbdSample,
        mask: np.ndarray,
        people: Sequence[Mapping[str, Any]] = (),
        identity_distance: float | None = None,
    ) -> TrustedMaskObservation:
        copied_mask = np.asarray(mask, dtype=bool).copy()
        if copied_mask.shape != sample.depth.shape:
            raise ValueError("trusted mask shape does not match RGB-D")
        mask_area = int(np.count_nonzero(copied_mask))
        if mask_area < 32:
            raise ValueError("trusted mask has fewer than 32 pixels")
        depth = np.asarray(sample.depth)
        valid = copied_mask & np.isfinite(depth) & (depth > 0)
        valid_depth_ratio = float(np.count_nonzero(valid)) / float(mask_area)
        foreground_outlier_ratio = 1.0
        if np.any(valid):
            values = depth[valid].astype(np.float64) / 1000.0
            median = float(np.median(values))
            foreground_outlier_ratio = float(
                np.mean(values < median - SHIGURE_LIFECYCLE_DEPTH_CHANGE_M)
            )
        height, width = copied_mask.shape
        mask_box = cls._mask_bbox_from_array(copied_mask)
        touches_image_edge = bool(
            mask_box is None
            or mask_box[0] <= 1.0
            or mask_box[1] <= 1.0
            or mask_box[2] >= float(width - 1)
            or mask_box[3] >= float(height - 1)
        )
        person_overlap_ratio = 0.0
        for person in people:
            person_box = _bbox(person.get("bounding_box"))
            person_overlap_ratio = max(
                person_overlap_ratio,
                cls._object_overlap_ratio(mask_box, person_box),
            )
        # CachedRgbdSample is treated as immutable. One exact frame is shared
        # across objects; only the object-specific mask needs a private copy.
        artifacts = EventArtifacts(None, None, None, sample, copied_mask)
        return TrustedMaskObservation(
            display_object_id=str(display_object_id),
            binding_id=str(binding_id),
            raw_id=str(raw_id),
            sequence=int(sequence),
            stamp_seconds=float(stamp_seconds),
            artifacts=artifacts,
            mask_area=mask_area,
            valid_depth_ratio=valid_depth_ratio,
            touches_image_edge=touches_image_edge,
            foreground_outlier_ratio=foreground_outlier_ratio,
            person_overlap_ratio=person_overlap_ratio,
            identity_distance=(
                float(identity_distance)
                if identity_distance is not None
                else None
            ),
        )

    def _remember_trusted_mask_observation(
        self,
        *,
        binding: Mapping[str, Any],
        frame: CachedShigureFrame,
        sample: CachedRgbdSample,
        mask: np.ndarray,
        identity_distance: float | None,
    ) -> TrustedMaskObservation:
        display_object_id = str(binding["display_object_id"])
        observation = self._build_trusted_mask_observation(
            display_object_id=display_object_id,
            binding_id=str(binding["binding_id"]),
            raw_id=str(binding["raw_shigure_object_id"]),
            sequence=int(frame.sequence),
            stamp_seconds=frame.source_stamp.seconds,
            sample=sample,
            mask=mask,
            people=frame.people,
            identity_distance=identity_distance,
        )
        with self._lock:
            history_capacity = max(
                4,
                int(
                    np.ceil(
                        (SHIGURE_LIFECYCLE_LOOKBACK_SECONDS + 1.0)
                        * SHIGURE_LIFECYCLE_OBSERVATION_HZ
                    )
                )
                + 2,
            )
            history = self._trusted_mask_observations.setdefault(
                display_object_id, deque(maxlen=history_capacity)
            )
            if history and history[-1].sequence == observation.sequence:
                history[-1] = observation
            elif history and (
                observation.stamp_seconds <= history[-1].stamp_seconds
                or observation.stamp_seconds - history[-1].stamp_seconds
                < 1.0 / SHIGURE_LIFECYCLE_OBSERVATION_HZ
            ):
                return observation
            else:
                history.append(observation)
            cutoff = (
                observation.stamp_seconds
                - SHIGURE_LIFECYCLE_LOOKBACK_SECONDS
                - 1.0
            )
            while history and history[0].stamp_seconds < cutoff:
                history.popleft()
        return observation

    @staticmethod
    def _depth_change_evidence(
        reference: TrustedMaskObservation,
        post_sample: CachedRgbdSample,
    ) -> dict[str, Any]:
        reference_sample = reference.artifacts.sample
        mask = reference.artifacts.mask_array
        if reference_sample is None or mask is None:
            return {"usable": False, "reason": "reference_rgbd_or_mask_missing"}
        reference_depth = np.asarray(reference_sample.depth)
        post_depth = np.asarray(post_sample.depth)
        mask = np.asarray(mask, dtype=bool)
        if reference_depth.shape != post_depth.shape or mask.shape != post_depth.shape:
            return {"usable": False, "reason": "rgbd_shape_changed"}
        core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if np.count_nonzero(core) < 32:
            core = mask
        valid = (
            core
            & np.isfinite(reference_depth)
            & np.isfinite(post_depth)
            & (reference_depth > 0)
            & (post_depth > 0)
        )
        valid_count = int(np.count_nonzero(valid))
        core_count = int(np.count_nonzero(core))
        if valid_count < 32 or core_count <= 0:
            return {
                "usable": False,
                "reason": "insufficient_paired_depth",
                "valid_depth_ratio": (
                    float(valid_count) / float(core_count)
                    if core_count > 0
                    else 0.0
                ),
            }
        delta_m = (
            post_depth[valid].astype(np.float64)
            - reference_depth[valid].astype(np.float64)
        ) / 1000.0
        threshold = SHIGURE_LIFECYCLE_DEPTH_CHANGE_M
        return {
            "usable": True,
            "valid_depth_ratio": float(valid_count) / float(core_count),
            "background_reveal_ratio": float(np.mean(delta_m >= threshold)),
            "foreground_occlusion_ratio": float(np.mean(delta_m <= -threshold)),
            "unchanged_ratio": float(np.mean(np.abs(delta_m) < threshold)),
            "median_depth_delta_m": float(np.median(delta_m)),
        }

    def _observe_lifecycle_confirmation_frame(
        self, frame: CachedShigureFrame
    ) -> None:
        with self._lock:
            pending_display_ids = set(self._lifecycle_pose_windows)
        if not pending_display_ids:
            return
        sample = self._sample_exact(frame)
        if sample is None:
            return
        stamp_seconds = frame.source_stamp.seconds
        with self._lock:
            for display_object_id in pending_display_ids:
                window = self._lifecycle_pose_windows.get(display_object_id)
                if window is not None:
                    window["post_samples"].append(
                        (int(frame.sequence), stamp_seconds, sample)
                    )
        for candidate in frame.recovery_candidates:
            raw_id = str(candidate.get("shigure_object_id") or "").strip()
            if (
                not raw_id
                or str(candidate.get("tracking_match_status") or "").upper()
                != "RESOLVED"
            ):
                continue
            binding = get_active_shigure_binding(
                source_epoch_id=str(self.source_epoch_id or ""),
                raw_shigure_object_id=raw_id,
            )
            if (
                binding is None
                or str(binding["display_object_id"]) not in pending_display_ids
            ):
                continue
            try:
                mask = _decode_full_mask(
                    candidate.get("mask_b64"), sample.rgb_bgr.shape[:2]
                )
                observation = self._build_trusted_mask_observation(
                    display_object_id=str(binding["display_object_id"]),
                    binding_id=str(binding["binding_id"]),
                    raw_id=raw_id,
                    sequence=int(frame.sequence),
                    stamp_seconds=stamp_seconds,
                    sample=sample,
                    mask=mask,
                    people=frame.people,
                )
            except Exception:
                continue
            with self._lock:
                window = self._lifecycle_pose_windows.get(
                    observation.display_object_id
                )
                if window is not None:
                    window["post_observations"].append(observation)

    def _event_identity_distance(
        self, display_object_id: str, artifacts: EventArtifacts
    ) -> float | None:
        try:
            embedded = self._embed_artifacts(artifacts)
            if embedded is None:
                return None
            scores = self._identity_scores(embedded[0], [display_object_id])
            return float(scores[0]["distance"]) if scores else None
        except Exception as exc:
            print(
                "[shigure-v2] lifecycle DINO scoring unavailable "
                f"for {display_object_id}: {exc}"
            )
            return None

    def _queue_lifecycle_pose_candidate(
        self,
        *,
        action: str,
        canonical_event_uid: str,
        display_object_id: str,
        raw_id: str,
        binding_id: str | None,
        resolution_method: str,
        identity: Mapping[str, Any],
        skeleton: Any,
        calibration_revision: str | None,
        marker_rotation: np.ndarray | None,
        marker_translation: np.ndarray | None,
        occurred_at: str,
        frame: CachedShigureFrame,
        artifacts: EventArtifacts,
    ) -> None:
        center = None
        try:
            center = self._masked_center_aruco(artifacts)
        except Exception as exc:
            print(f"[shigure-v2] lifecycle mask center unavailable: {exc}")
        identity_distance = identity.get("distance")
        distance = (
            float(identity_distance)
            if identity_distance is not None
            else self._event_identity_distance(
                display_object_id, artifacts
            )
        )
        candidate = LifecyclePoseCandidate(
            action=str(action),
            canonical_event_uid=str(canonical_event_uid),
            display_object_id=str(display_object_id),
            raw_id=str(raw_id),
            binding_id=(str(binding_id) if binding_id else None),
            resolution_method=str(resolution_method),
            sequence=int(frame.sequence),
            stamp_seconds=(
                float(frame.source_stamp.sec)
                + float(frame.source_stamp.nanosec) * 1.0e-9
            ),
            source_generation=int(self._source_generation),
            source_epoch_id=str(self.source_epoch_id or ""),
            artifacts=artifacts,
            mask_center_aruco=(center.copy() if center is not None else None),
            dino_distance=distance,
            identity=dict(identity),
            skeleton=skeleton,
            calibration_revision=(
                str(calibration_revision) if calibration_revision else None
            ),
            marker_rotation=(
                np.asarray(marker_rotation, dtype=np.float64).copy()
                if marker_rotation is not None
                else None
            ),
            marker_translation=(
                np.asarray(marker_translation, dtype=np.float64).copy()
                if marker_translation is not None
                else None
            ),
            occurred_at=str(occurred_at),
        )
        now = time.monotonic()
        with self._lock:
            window = self._lifecycle_pose_windows.setdefault(
                str(display_object_id),
                {
                    "opened": now,
                    "candidates": [],
                    "post_samples": deque(maxlen=96),
                    "post_observations": deque(maxlen=96),
                },
            )
            existing_index = next(
                (
                    index
                    for index, item in enumerate(window["candidates"])
                    if item.canonical_event_uid
                    == candidate.canonical_event_uid
                ),
                None,
            )
            if existing_index is None:
                window["candidates"].append(candidate)
            else:
                previous = window["candidates"][existing_index]
                previous_identity = (
                    previous.action,
                    previous.display_object_id,
                    previous.raw_id,
                    previous.binding_id,
                    previous.source_epoch_id,
                )
                candidate_identity = (
                    candidate.action,
                    candidate.display_object_id,
                    candidate.raw_id,
                    candidate.binding_id,
                    candidate.source_epoch_id,
                )
                if candidate_identity != previous_identity:
                    print(
                        "[shigure-v2] canonical lifecycle replay identity "
                        "conflict ignored "
                        f"event={candidate.canonical_event_uid} "
                        f"stored={previous_identity} incoming={candidate_identity}"
                    )
                    return
                previous_quality = (
                    previous.mask_center_aruco is not None,
                    previous.dino_distance is not None,
                    -float(previous.dino_distance)
                    if previous.dino_distance is not None
                    else float("-inf"),
                )
                candidate_quality = (
                    candidate.mask_center_aruco is not None,
                    candidate.dino_distance is not None,
                    -float(candidate.dino_distance)
                    if candidate.dino_distance is not None
                    else float("-inf"),
                )
                if candidate_quality > previous_quality:
                    window["candidates"][existing_index] = candidate
        print(
            "[shigure-v2] lifecycle pose candidate queued "
            f"action={action} display={display_object_id} raw={raw_id} "
            f"center={center.tolist() if center is not None else None} "
            f"dino_distance={distance}"
        )

    @staticmethod
    def _choose_lifecycle_candidate(
        candidates: Sequence[LifecyclePoseCandidate], action: str
    ) -> LifecyclePoseCandidate | None:
        rows = [item for item in candidates if item.action == action]
        if not rows:
            return None
        temporal = (
            min(rows, key=lambda item: (item.stamp_seconds, item.sequence))
            if action == "take_out"
            else max(rows, key=lambda item: (item.stamp_seconds, item.sequence))
        )
        if temporal.mask_center_aruco is None:
            close = [temporal]
        else:
            close = [
                item
                for item in rows
                if item.mask_center_aruco is not None
                and abs(item.stamp_seconds - temporal.stamp_seconds)
                <= SHIGURE_LIFECYCLE_SELECTION_WINDOW_SECONDS
                and float(
                    np.linalg.norm(
                        item.mask_center_aruco - temporal.mask_center_aruco
                    )
                ) <= SHIGURE_LIFECYCLE_NO_MOVE_DISTANCE_M
            ]
        scored = [item for item in close if item.dino_distance is not None]
        if scored:
            direction = 1.0 if action == "take_out" else -1.0
            return min(
                scored,
                key=lambda item: (
                    float(item.dino_distance),
                    direction * item.stamp_seconds,
                    direction * item.sequence,
                ),
            )
        return temporal

    def _flush_lifecycle_pose_windows(self, now_monotonic: float) -> None:
        ready: list[
            tuple[
                str,
                list[LifecyclePoseCandidate],
                list[tuple[int, float, CachedRgbdSample]],
                list[TrustedMaskObservation],
            ]
        ] = []
        with self._lock:
            for display_object_id, window in tuple(
                self._lifecycle_pose_windows.items()
            ):
                if (
                    float(now_monotonic) - float(window["opened"])
                    < SHIGURE_LIFECYCLE_CONFIRMATION_SECONDS
                ):
                    continue
                ready.append(
                    (
                        display_object_id,
                        list(window["candidates"]),
                        list(window["post_samples"]),
                        list(window["post_observations"]),
                    )
                )
                self._lifecycle_pose_windows.pop(display_object_id, None)
        for (
            display_object_id,
            candidates,
            post_samples,
            post_observations,
        ) in ready:
            # Identity/presence decisions must not wait behind a long-running
            # FoundationPose job.  Only the pose estimation itself is queued.
            self._run_lifecycle_pose_window(
                display_object_id,
                candidates,
                post_samples=post_samples,
                post_observations=post_observations,
            )

    def _has_pending_take_out(self, display_object_id: str) -> bool:
        with self._lock:
            window = self._lifecycle_pose_windows.get(str(display_object_id))
            return bool(
                window
                and any(
                    candidate.action == "take_out"
                    for candidate in window["candidates"]
                )
            )

    def _reject_lifecycle_candidates(
        self,
        candidates: Sequence[LifecyclePoseCandidate],
        *,
        reason: str,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.canonical_event_uid in seen:
                continue
            seen.add(candidate.canonical_event_uid)
            audit = {
                "lifecycle_window": {
                    "action": candidate.action,
                    "display_object_id": candidate.display_object_id,
                    "raw_shigure_object_id": candidate.raw_id,
                    "sequence": candidate.sequence,
                    "source_generation": candidate.source_generation,
                    "source_epoch_id": candidate.source_epoch_id,
                }
            }
            if detail:
                audit["lifecycle_window"].update(dict(detail))
            try:
                reject_pending_shigure_canonical_event(
                    candidate.canonical_event_uid,
                    reason=reason,
                    detail=audit,
                )
            except Exception as exc:
                print(
                    "[shigure-v2] failed to reject lifecycle candidate "
                    f"event={candidate.canonical_event_uid} reason={reason}: "
                    f"{exc}"
                )

    @staticmethod
    def _lifecycle_movement_distance(
        take_out: LifecyclePoseCandidate | None,
        bring_in: LifecyclePoseCandidate | None,
    ) -> float | None:
        if (
            take_out is None
            or bring_in is None
            or take_out.mask_center_aruco is None
            or bring_in.mask_center_aruco is None
            or abs(bring_in.stamp_seconds - take_out.stamp_seconds)
            > SHIGURE_LIFECYCLE_SELECTION_WINDOW_SECONDS
        ):
            return None
        return float(
            np.linalg.norm(
                bring_in.mask_center_aruco - take_out.mask_center_aruco
            )
        )

    def _select_clear_take_out_source(
        self, take_out: LifecyclePoseCandidate
    ) -> tuple[TrustedMaskObservation | None, list[dict[str, Any]]]:
        with self._lock:
            history = list(
                self._trusted_mask_observations.get(
                    take_out.display_object_id, ()
                )
            )
        eligible = [
            observation
            for observation in history
            if observation.sequence <= take_out.sequence
            and take_out.stamp_seconds - SHIGURE_LIFECYCLE_LOOKBACK_SECONDS
            <= observation.stamp_seconds
            <= take_out.stamp_seconds
        ]
        if len(eligible) < 2:
            return None, [
                {
                    **observation.summary(),
                    "geometry_qualified": False,
                    "identity_qualified": False,
                    "rejection_reasons": [
                        "insufficient_prewindow_history"
                    ],
                }
                for observation in eligible
            ]
        reference_area = float(
            np.percentile(
                np.asarray(
                    [observation.mask_area for observation in eligible],
                    dtype=np.float64,
                ),
                75.0,
            )
        )
        foreground_baseline = float(
            np.median(
                [item.foreground_outlier_ratio for item in eligible]
            )
        )
        reports: list[dict[str, Any]] = []
        qualified: list[TrustedMaskObservation] = []
        for observation in eligible:
            area_ratio = (
                float(observation.mask_area) / reference_area
                if reference_area > 0.0
                else 0.0
            )
            reasons: list[str] = []
            if area_ratio < SHIGURE_LIFECYCLE_MIN_SOURCE_AREA_RATIO:
                reasons.append("mask_incomplete_vs_prewindow")
            if (
                observation.valid_depth_ratio
                < SHIGURE_LIFECYCLE_MIN_VALID_DEPTH_RATIO
            ):
                reasons.append("insufficient_valid_depth")
            if observation.touches_image_edge:
                reasons.append("mask_touches_image_edge")
            foreground_outlier_excess = max(
                0.0,
                observation.foreground_outlier_ratio - foreground_baseline,
            )
            if (
                foreground_outlier_excess
                > SHIGURE_LIFECYCLE_MAX_FOREGROUND_OCCLUSION_RATIO
            ):
                reasons.append("foreground_depth_contamination")
            if (
                observation.person_overlap_ratio
                > SHIGURE_LIFECYCLE_MAX_PERSON_OVERLAP_RATIO
            ):
                reasons.append("person_overlaps_object_mask")
            report = {
                **observation.summary(),
                "area_ratio_vs_prewindow_p75": area_ratio,
                "foreground_outlier_baseline": foreground_baseline,
                "foreground_outlier_excess": foreground_outlier_excess,
                "geometry_qualified": not reasons,
                "rejection_reasons": reasons,
            }
            reports.append(report)
            if not reasons:
                qualified.append(observation)

        report_by_sequence = {
            int(row["sequence"]): row for row in reports
        }
        # The closest clear frame wins. DINO is checked only on event-time
        # candidates, avoiding continuous embedding work during normal use.
        for observation in sorted(
            qualified,
            key=lambda item: (item.stamp_seconds, item.sequence),
            reverse=True,
        )[:5]:
            distance = observation.identity_distance
            if distance is None:
                distance = self._event_identity_distance(
                    take_out.display_object_id, observation.artifacts
                )
            report = report_by_sequence[observation.sequence]
            report["identity_distance"] = distance
            if distance is None:
                report["identity_qualified"] = False
                report["rejection_reasons"].append(
                    "dinov2_identity_unavailable"
                )
                continue
            if distance > SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD:
                report["identity_qualified"] = False
                report["rejection_reasons"].append(
                    "dinov2_identity_distance_above_threshold"
                )
                continue
            report["identity_qualified"] = True
            return replace(
                observation, identity_distance=float(distance)
            ), reports
        return None, reports

    @staticmethod
    def _write_lifecycle_artifacts(
        root: Path, prefix: str, artifacts: EventArtifacts
    ) -> dict[str, str]:
        sample = artifacts.sample
        mask = artifacts.mask_array
        if sample is None:
            return {}
        written: dict[str, str] = {}
        scene_path = root / f"{prefix}_scene.png"
        depth_path = root / f"{prefix}_depth.png"
        _write_image(scene_path, np.asarray(sample.rgb_bgr))
        _write_image(depth_path, np.asarray(sample.depth))
        written["scene"] = str(scene_path)
        written["depth"] = str(depth_path)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            mask_path = root / f"{prefix}_mask.png"
            _write_image(mask_path, mask.astype(np.uint8) * 255)
            written["mask"] = str(mask_path)
            box = ShigureRuntimeEngine._mask_bbox_from_array(mask)
            if box is not None:
                x0, y0, x1, y1 = (int(value) for value in box)
                crop = np.asarray(sample.rgb_bgr)[y0:y1, x0:x1].copy()
                local_mask = mask[y0:y1, x0:x1]
                crop[~local_mask] = 0
                crop_path = root / f"{prefix}_object_crop.png"
                _write_image(crop_path, crop)
                written["crop"] = str(crop_path)
        return written

    def _persist_lifecycle_confirmation_report(
        self,
        *,
        take_out: LifecyclePoseCandidate,
        source: TrustedMaskObservation | None,
        post_samples: Sequence[tuple[int, float, CachedRgbdSample]],
        detail: Mapping[str, Any],
        status: str,
        reason: str,
    ) -> Path | None:
        try:
            root = (
                self._recovery_debug_epoch_root()
                / "lifecycle_takeout"
                / _safe_token(take_out.canonical_event_uid, "event")
            )
            artifacts: dict[str, Any] = {
                "event": self._write_lifecycle_artifacts(
                    root, "event", take_out.artifacts
                )
            }
            if source is not None:
                artifacts["selected_source"] = (
                    self._write_lifecycle_artifacts(
                        root, "selected_source", source.artifacts
                    )
                )
            usable_post = [
                item
                for item in post_samples
                if item[1] > take_out.stamp_seconds
            ]
            if usable_post:
                sequence, stamp_seconds, sample = usable_post[-1]
                artifacts["latest_post"] = {
                    **self._write_lifecycle_artifacts(
                        root,
                        "latest_post",
                        EventArtifacts(None, None, None, sample, None),
                    ),
                    "sequence": int(sequence),
                    "stamp_seconds": float(stamp_seconds),
                }
            report_path = root / "report.json"
            _write_json(
                report_path,
                {
                    "schema_version": 1,
                    "runtime_session_id": self.runtime_session_id,
                    "source_epoch_id": self.source_epoch_id,
                    "recorded_utc": _utc_now(),
                    "canonical_event_uid": take_out.canonical_event_uid,
                    "display_object_id": take_out.display_object_id,
                    "raw_shigure_object_id": take_out.raw_id,
                    "take_out_sequence": int(take_out.sequence),
                    "take_out_stamp_seconds": float(
                        take_out.stamp_seconds
                    ),
                    "status": str(status),
                    "reason": str(reason),
                    "policy": {
                        "confirmation_seconds": (
                            SHIGURE_LIFECYCLE_CONFIRMATION_SECONDS
                        ),
                        "lookback_seconds": (
                            SHIGURE_LIFECYCLE_LOOKBACK_SECONDS
                        ),
                        "no_move_distance_m": (
                            SHIGURE_LIFECYCLE_NO_MOVE_DISTANCE_M
                        ),
                        "depth_change_m": (
                            SHIGURE_LIFECYCLE_DEPTH_CHANGE_M
                        ),
                    },
                    "detail": dict(detail),
                    "artifacts": artifacts,
                },
            )
            return report_path
        except Exception as exc:
            print(
                "[shigure-v2] lifecycle confirmation report failed "
                f"event={take_out.canonical_event_uid}: {exc}"
            )
            return None

    def _confirm_take_out_movement(
        self,
        take_out: LifecyclePoseCandidate,
        bring_in: LifecyclePoseCandidate | None,
        movement_distance: float | None,
        *,
        post_samples: Sequence[tuple[int, float, CachedRgbdSample]],
        post_observations: Sequence[TrustedMaskObservation],
    ) -> LifecycleMovementConfirmation:
        source, source_reports = self._select_clear_take_out_source(
            take_out
        )
        depth_evidence: list[dict[str, Any]] = []
        if source is not None:
            seen_stamps: set[float] = set()
            for sequence, stamp_seconds, sample in sorted(
                post_samples, key=lambda item: (item[1], item[0])
            ):
                if (
                    stamp_seconds <= take_out.stamp_seconds
                    or stamp_seconds
                    > take_out.stamp_seconds
                    + SHIGURE_LIFECYCLE_CONFIRMATION_SECONDS
                    or stamp_seconds in seen_stamps
                ):
                    continue
                seen_stamps.add(stamp_seconds)
                depth_evidence.append(
                    {
                        "sequence": int(sequence),
                        "stamp_seconds": float(stamp_seconds),
                        **self._depth_change_evidence(source, sample),
                    }
                )

        usable_depth = [
            row
            for row in depth_evidence
            if row.get("usable")
            and float(row.get("valid_depth_ratio") or 0.0)
            >= SHIGURE_LIFECYCLE_MIN_VALID_DEPTH_RATIO
        ]
        reveal_frames = [
            row
            for row in usable_depth
            if float(row.get("background_reveal_ratio") or 0.0)
            >= SHIGURE_LIFECYCLE_MIN_BACKGROUND_REVEAL_RATIO
            and float(row.get("foreground_occlusion_ratio") or 0.0)
            <= SHIGURE_LIFECYCLE_MAX_FOREGROUND_OCCLUSION_RATIO
        ]
        foreground_frames = [
            row
            for row in usable_depth
            if float(row.get("foreground_occlusion_ratio") or 0.0)
            > SHIGURE_LIFECYCLE_MAX_FOREGROUND_OCCLUSION_RATIO
        ]
        unchanged_frames = [
            row
            for row in usable_depth
            if float(row.get("background_reveal_ratio") or 0.0)
            < SHIGURE_LIFECYCLE_MIN_BACKGROUND_REVEAL_RATIO
            and float(row.get("foreground_occlusion_ratio") or 0.0)
            <= SHIGURE_LIFECYCLE_MAX_FOREGROUND_OCCLUSION_RATIO
            and abs(float(row.get("median_depth_delta_m") or 0.0))
            < SHIGURE_LIFECYCLE_DEPTH_CHANGE_M
        ]
        detail: dict[str, Any] = {
            "movement_distance_m": movement_distance,
            "source_candidates": source_reports,
            "selected_source": source.summary() if source else None,
            "post_depth_evidence": depth_evidence,
            "post_mask_observations": [
                observation.summary()
                for observation in post_observations
                if observation.stamp_seconds > take_out.stamp_seconds
            ],
            "reveal_frame_count": len(reveal_frames),
            "foreground_frame_count": len(foreground_frames),
            "unchanged_frame_count": len(unchanged_frames),
        }

        status = "AMBIGUOUS"
        reason = "INSUFFICIENT_TRUE_MOVEMENT_EVIDENCE"
        if source is None:
            reason = "NO_CLEAR_IDENTITY_MATCHED_PRE_TAKEOUT_FRAME"
        elif (
            bring_in is not None
            and movement_distance is not None
            and movement_distance >= SHIGURE_LIFECYCLE_NO_MOVE_DISTANCE_M
            and bring_in.dino_distance is not None
            and bring_in.dino_distance
            <= SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD
        ):
            status = "REAL_MOVE"
            reason = "DINOV2_MATCHED_BRING_IN_MOVED_AT_LEAST_20CM"
        elif len(unchanged_frames) >= SHIGURE_LIFECYCLE_MIN_POST_EVIDENCE_FRAMES:
            status = "NO_MOVE"
            reason = "OBJECT_DEPTH_REMAINED_AT_ORIGINAL_POSITION"
        elif len(reveal_frames) >= SHIGURE_LIFECYCLE_MIN_POST_EVIDENCE_FRAMES:
            status = "REAL_MOVE"
            reason = "BACKGROUND_REVEALED_AFTER_TAKE_OUT"
        elif foreground_frames:
            reason = "FOREGROUND_OCCLUSION_NOT_REAL_MOVEMENT"

        report_path = self._persist_lifecycle_confirmation_report(
            take_out=take_out,
            source=source,
            post_samples=post_samples,
            detail=detail,
            status=status,
            reason=reason,
        )
        return LifecycleMovementConfirmation(
            status=status,
            reason=reason,
            source=source,
            detail=detail,
            report_path=report_path,
        )

    def _run_lifecycle_pose_window(
        self,
        display_object_id: str,
        candidates: Sequence[LifecyclePoseCandidate],
        *,
        post_samples: Sequence[
            tuple[int, float, CachedRgbdSample]
        ] = (),
        post_observations: Sequence[TrustedMaskObservation] = (),
    ) -> None:
        take_out = self._choose_lifecycle_candidate(candidates, "take_out")
        if take_out is None:
            self._reject_lifecycle_candidates(
                candidates,
                reason="LIFECYCLE_WINDOW_HAS_NO_TAKE_OUT",
            )
            return
        bring_in = self._choose_lifecycle_candidate(
            [
                candidate
                for candidate in candidates
                if candidate.action == "bring_in"
                and candidate.stamp_seconds >= take_out.stamp_seconds
            ],
            "bring_in",
        )
        if (
            take_out.source_generation != self._source_generation
            or take_out.source_epoch_id != str(self.source_epoch_id or "")
        ):
            self._reject_lifecycle_candidates(
                candidates,
                reason="STALE_LIFECYCLE_SOURCE_EPOCH",
            )
            return
        print(
            "[shigure-v2] lifecycle pose window selected "
            f"display={display_object_id} take_out="
            f"{take_out.canonical_event_uid} bring_in="
            f"{bring_in.canonical_event_uid if bring_in is not None else None} "
            f"take_out_dino={take_out.dino_distance} "
            f"bring_in_dino="
            f"{bring_in.dino_distance if bring_in is not None else None}"
        )
        movement_distance = self._lifecycle_movement_distance(
            take_out, bring_in
        )
        if (
            movement_distance is not None
            and movement_distance < SHIGURE_LIFECYCLE_NO_MOVE_DISTANCE_M
        ):
            print(
                "[shigure-v2] lifecycle pose suppressed as foreground "
                f"occlusion/no-move display={display_object_id} "
                f"mask_move_m={movement_distance:.6f}"
            )
            fast_detail = {
                "movement_distance_m": movement_distance,
                "selected_take_out_event_uid": take_out.canonical_event_uid,
                "selected_bring_in_event_uid": bring_in.canonical_event_uid,
            }
            report_path = self._persist_lifecycle_confirmation_report(
                take_out=take_out,
                source=None,
                post_samples=post_samples,
                detail=fast_detail,
                status="NO_MOVE",
                reason="NO_MOVE_MASK_CENTER_LT_20CM",
            )
            self._reject_lifecycle_candidates(
                candidates,
                reason="NO_MOVE_MASK_CENTER_LT_20CM",
                detail={
                    "mask_move_m": movement_distance,
                    "selected_take_out_event_uid": (
                        take_out.canonical_event_uid
                    ),
                    "selected_bring_in_event_uid": (
                        bring_in.canonical_event_uid
                        if bring_in is not None
                        else None
                    ),
                    "confirmation_report_path": (
                        str(report_path)
                        if report_path is not None
                        else None
                    ),
                },
            )
            return

        selected_uids = {take_out.canonical_event_uid}
        if bring_in is not None:
            selected_uids.add(bring_in.canonical_event_uid)
        self._reject_lifecycle_candidates(
            [
                candidate
                for candidate in candidates
                if candidate.canonical_event_uid not in selected_uids
            ],
            reason="LIFECYCLE_WINDOW_CANDIDATE_NOT_SELECTED",
            detail={
                "selected_take_out_event_uid": take_out.canonical_event_uid,
                "selected_bring_in_event_uid": (
                    bring_in.canonical_event_uid
                    if bring_in is not None
                    else None
                ),
                "mask_move_m": movement_distance,
            },
        )

        active_take_out = (
            get_active_shigure_binding(
                source_epoch_id=take_out.source_epoch_id,
                raw_shigure_object_id=take_out.raw_id,
            )
            if take_out.binding_id
            else None
        )
        if (
            active_take_out is None
            or str(active_take_out["binding_id"]) != take_out.binding_id
            or str(active_take_out["display_object_id"])
            != str(display_object_id)
        ):
            self._reject_lifecycle_candidates(
                [take_out] + ([bring_in] if bring_in is not None else []),
                reason="TAKE_OUT_BINDING_CHANGED_DURING_WINDOW",
            )
            return

        confirmation = self._confirm_take_out_movement(
            take_out,
            bring_in,
            movement_distance,
            post_samples=post_samples,
            post_observations=post_observations,
        )
        print(
            "[shigure-v2] take_out movement confirmation "
            f"display={display_object_id} status={confirmation.status} "
            f"reason={confirmation.reason} "
            f"report={confirmation.report_path}"
        )
        if confirmation.status != "REAL_MOVE" or confirmation.source is None:
            self._reject_lifecycle_candidates(
                [take_out] + ([bring_in] if bring_in is not None else []),
                reason=confirmation.reason,
                detail={
                    **confirmation.detail,
                    "confirmation_status": confirmation.status,
                    "confirmation_report_path": (
                        str(confirmation.report_path)
                        if confirmation.report_path is not None
                        else None
                    ),
                },
            )
            return
        take_out = replace(
            take_out,
            artifacts=confirmation.source.artifacts,
            dino_distance=confirmation.source.identity_distance,
        )

        aliases = [
            binding
            for binding in list_active_shigure_bindings(
                take_out.source_epoch_id
            )
            if str(binding["display_object_id"]) == str(display_object_id)
        ]
        audit = {
            "selected_take_out_event_uid": take_out.canonical_event_uid,
            "selected_bring_in_event_uid": (
                bring_in.canonical_event_uid if bring_in is not None else None
            ),
            "mask_move_m": movement_distance,
            "take_out_dino_distance": take_out.dino_distance,
            "bring_in_dino_distance": (
                bring_in.dino_distance if bring_in is not None else None
            ),
            "movement_confirmation": {
                "status": confirmation.status,
                "reason": confirmation.reason,
                "report_path": (
                    str(confirmation.report_path)
                    if confirmation.report_path is not None
                    else None
                ),
                "selected_source": confirmation.source.summary(),
            },
        }
        try:
            committed_take_out = commit_pending_take_out_lifecycle_event(
                take_out.canonical_event_uid,
                binding_id=take_out.binding_id,
                display_object_id=str(display_object_id),
                raw_id=take_out.raw_id,
                resolution_method=take_out.resolution_method,
                detail={"lifecycle_window": audit},
                skeleton=take_out.skeleton,
                calibration_revision=take_out.calibration_revision,
                occurred_at=take_out.occurred_at,
            )
            lifecycle = committed_take_out["lifecycle_event"]
        except Exception as exc:
            print(
                "[shigure-v2] deferred take_out commit failed "
                f"for {display_object_id}: {exc}"
            )
            if bring_in is not None:
                self._reject_lifecycle_candidates(
                    [bring_in], reason="TAKE_OUT_COMMIT_FAILED"
                )
            return

        with self._lock:
            for alias in aliases:
                alias_raw = str(alias["raw_shigure_object_id"])
                self._view_windows.pop(alias_raw, None)
                self._spatial_boxes.pop(str(alias["binding_id"]), None)
                self._stable.pop(alias_raw, None)
                self._last_stable_source_key.pop(alias_raw, None)
                self._last_view_sequence.pop(alias_raw, None)

        if bring_in is not None:
            try:
                distance = bring_in.identity.get("distance")
                if distance is None:
                    distance = bring_in.dino_distance
                confidence = (
                    max(0.0, 1.0 - float(distance))
                    if distance is not None
                    else None
                )
                committed_bring_in = commit_pending_bring_in_lifecycle_event(
                    bring_in.canonical_event_uid,
                    runtime_session_id=str(self.runtime_session_id),
                    source_epoch_id=bring_in.source_epoch_id,
                    raw_id=bring_in.raw_id,
                    display_object_id=str(display_object_id),
                    resolution_method=bring_in.resolution_method,
                    confidence=confidence,
                    binding_detail={
                        **bring_in.identity,
                        "lifecycle_window": audit,
                    },
                    resolution_detail={"lifecycle_window": audit},
                    skeleton=bring_in.skeleton,
                    calibration_revision=bring_in.calibration_revision,
                    occurred_at=bring_in.occurred_at,
                )
                bring_in_binding = committed_bring_in["binding"]
                with self._lock:
                    self._view_windows[bring_in.raw_id] = {
                        "display_object_id": str(display_object_id),
                        "source_generation": self._source_generation,
                        "attempts": 0,
                        "added": 0,
                    }
            except Exception as exc:
                print(
                    "[shigure-v2] deferred bring_in commit failed "
                    f"for {display_object_id}: {exc}"
                )
                self._reject_lifecycle_candidates(
                    [bring_in], reason="BRING_IN_COMMIT_FAILED"
                )

        try:
            self._pose_executor.submit(
                self._run_take_out_foundationpose,
                take_out,
                movement_distance,
                int(lifecycle.get("model_revision") or 0),
            )
        except RuntimeError as exc:
            print(
                "[shigure-v2] take_out FoundationPose was not queued "
                f"for {display_object_id}: {exc}"
            )

    def _run_take_out_foundationpose(
        self,
        take_out: LifecyclePoseCandidate,
        movement_distance: float | None,
        model_revision: int,
    ) -> None:
        try:
            if (
                take_out.source_generation != self._source_generation
                or take_out.source_epoch_id
                != str(self.source_epoch_id or "")
            ):
                raise RuntimeError(
                    "source epoch changed before take_out FoundationPose"
                )
            state = get_display_object_state(take_out.display_object_id) or {}
            if int(state.get("active_model_revision") or 0) != model_revision:
                raise RuntimeError(
                    "model revision changed before take_out FoundationPose"
                )
            pose_aruco = self._foundationpose_pose_for_lifecycle(take_out)
            state = get_display_object_state(take_out.display_object_id) or {}
            if (
                take_out.source_generation != self._source_generation
                or take_out.source_epoch_id
                != str(self.source_epoch_id or "")
                or int(state.get("active_model_revision") or 0)
                != model_revision
            ):
                raise RuntimeError(
                    "source epoch/model changed during take_out FoundationPose"
                )
            apply_object_lifecycle_event(
                canonical_event_uid=take_out.canonical_event_uid,
                pose_aruco=pose_aruco,
            )
            origin = add_display_object_origin(
                display_object_id=take_out.display_object_id,
                pose_aruco=pose_aruco,
                kind="TAKE_OUT",
                model_revision=model_revision,
                source_epoch_id=take_out.source_epoch_id,
                canonical_event_uid=take_out.canonical_event_uid,
                raw_shigure_object_id=take_out.raw_id,
                occurred_at=take_out.occurred_at,
                dedup_distance_m=SHIGURE_LIFECYCLE_NO_MOVE_DISTANCE_M,
            )
            print(
                "[shigure-v2] take_out FoundationPose committed "
                f"display={take_out.display_object_id} selected_event="
                f"{take_out.canonical_event_uid} dino_distance="
                f"{take_out.dino_distance} mask_move_m={movement_distance} "
                f"origin_id={origin.get('id')} "
                f"origin_deduplicated={bool(origin.get('_deduplicated'))}"
            )
        except Exception as exc:
            print(
                "[shigure-v2] take_out FoundationPose failed "
                f"for {take_out.display_object_id}: {exc}"
            )

    def _foundationpose_pose_for_lifecycle(
        self, candidate: LifecyclePoseCandidate
    ) -> dict[str, Any]:
        return self._foundationpose_pose(
            candidate.display_object_id,
            candidate.sequence,
            candidate.artifacts,
            marker_calibration=(
                (candidate.marker_rotation, candidate.marker_translation)
                if candidate.marker_rotation is not None
                and candidate.marker_translation is not None
                else None
            ),
        )

    def _foundationpose_pose(
        self,
        display_object_id: str,
        sequence: int,
        artifacts: EventArtifacts,
        marker_calibration: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        sample = artifacts.sample
        if sample is None or artifacts.mask_array is None:
            raise ValueError("FoundationPose requires exact RGB-D and a full mask")
        state = get_display_object_state(display_object_id) or {}
        model_revision = int(state.get("active_model_revision") or 0)
        task_row = get_task_by_task_id(
            str(state.get("active_model_task_id") or "")
        )
        if model_revision <= 0 or not task_row:
            raise RuntimeError("active model revision/task is unavailable")
        task = load_task_json(resolve_task_json_path_from_record(task_row))
        source = resolve_model_generation_source(
            task, require_mtl_image=False
        )
        scale = float(
            (task.get("object_alignment") or {}).get("model_real_scale")
            or 0.0
        )
        if scale <= 0.0:
            raise RuntimeError("active model scale is missing")
        snapshot = self._foundationpose_snapshot(
            {"sequence": int(sequence)}, artifacts, sample
        )
        response = self.foundationpose_request(
            {
                "mesh_file": str(source.mesh_path),
                "color_file": snapshot["color_file"],
                "depth_file": snapshot["depth_file"],
                "mask_file": snapshot["mask_file"],
                "k": snapshot["k"],
                "model_scale": scale,
                "iteration": 5,
            },
            display_object_id,
        )
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "FoundationPose failed"))
        result = response.get("result")
        result = result if isinstance(result, Mapping) else {}
        pose_cv = np.asarray(result.get("pose"), dtype=np.float64)
        if pose_cv.shape != (4, 4) or not np.isfinite(pose_cv).all():
            raise RuntimeError("FoundationPose returned an invalid pose")
        quality = self._validate_foundationpose(
            source.mesh_path,
            pose_cv,
            scale,
            np.asarray(artifacts.mask_array, dtype=bool),
            np.asarray(sample.depth),
            np.asarray(snapshot["k"], dtype=np.float64),
        )
        if not quality["accepted"]:
            raise RuntimeError(f"FoundationPose quality rejected: {quality}")
        return self._foundationpose_to_aruco_pose(
            pose_cv, scale, marker_calibration=marker_calibration
        )

    def _process_event(self, frame: CachedShigureFrame, event: Mapping[str, Any], position: int) -> None:
        if not self.runtime_session_id or not self.source_epoch_id:
            return
        action = str(event.get("action") or "").lower()
        if action not in {"bring_in", "take_out", "obj_move"}:
            return
        detection = event.get("detection") if isinstance(event.get("detection"), Mapping) else {}
        tracking = event.get("tracking") if isinstance(event.get("tracking"), Mapping) else {}
        item_key = detection.get("item_key") if isinstance(detection.get("item_key"), Mapping) else {}
        canonical_event_index = event.get("canonical_event_index")
        if canonical_event_index is not None:
            detection_index = int(canonical_event_index)
            if detection_index < 0:
                raise ValueError("canonical_event_index must be non-negative")
        elif tracking:
            # Defensive fallback for hand-built internal frames. Recorder
            # output always carries canonical_event_index; the tracking
            # partition here only keeps an unindexed tracking event disjoint
            # from an unindexed detection event.
            detection_index = 1_000_000 + int(tracking.get("index", position))
        elif "index" in item_key:
            detection_index = int(item_key["index"])
        else:
            detection_index = 2_000_000 + int(position)
        if action == "obj_move":
            # The canonical DB key predates action. Put rejected move evidence
            # in a disjoint partition so an older RESOLVED lifecycle slot
            # cannot absorb this audit replay.
            detection_index += 10_000_000
        box = _bbox(detection) or _bbox(tracking)
        bbox_payload = {"xyxy": list(box)} if box is not None else {"xyxy": None, "status": "MISSING"}
        raw_id = str(event.get("shigure_object_id") or "").strip()
        artifacts = self._event_artifacts(frame, event)
        skeleton = None
        calibration_revision = None
        marker_rotation = None
        marker_translation = None
        try:
            _transform, rotation, translation, calibration_revision = _camera_to_aruco()
            marker_rotation = np.asarray(rotation, dtype=np.float64).copy()
            marker_translation = np.asarray(
                translation, dtype=np.float64
            ).copy()
            person = _event_person(event)
            if person is None:
                person = _nearest_person(frame.people, box)
            skeleton = _person_skeleton(person, rotation, translation)
        except Exception:
            pass
        adapter_status = str(event.get("resolution_status") or "UNRESOLVED").upper()
        binding = get_active_shigure_binding(source_epoch_id=self.source_epoch_id, raw_shigure_object_id=raw_id) if raw_id else None
        resolution_status = adapter_status if adapter_status in {"RESOLVED", "UNRESOLVED", "REJECTED"} else "UNRESOLVED"
        resolution_method = None
        identity: dict[str, Any] = {}
        embedded: tuple[np.ndarray, dict[str, Any]] | None = None
        candidate_display_id = ""
        candidate_binding_id: str | None = None
        defer_lifecycle = False
        if action == "obj_move":
            resolution_status = "REJECTED"
            identity = {
                "reason": "upstream_obj_move_id_is_not_authoritative",
                "adapter_status": adapter_status,
                "has_raw_id": bool(raw_id),
                "has_epoch_binding": binding is not None,
                "dino_attempted": False,
            }
        elif adapter_status != "RESOLVED" or not raw_id:
            resolution_status = "UNRESOLVED"
        elif action == "take_out":
            if binding is not None:
                candidate_display_id = str(binding["display_object_id"])
                candidate_binding_id = str(binding["binding_id"])
                defer_lifecycle = True
                resolution_status = "UNRESOLVED"
                resolution_method = "TAKE_OUT_WINDOW"
                identity = {
                    "status": "PENDING",
                    "reason": "awaiting_lifecycle_no_move_window",
                    "display_object_id": candidate_display_id,
                    "binding_id": candidate_binding_id,
                    "dino_attempted": True,
                }
            else:
                resolution_status = "UNRESOLVED"
                identity = {
                    "reason": "takeout_has_no_epoch_binding",
                    "dino_attempted": False,
                }
        elif action == "bring_in":
            if binding is not None:
                candidate_display_id = str(binding["display_object_id"])
                if self._has_pending_take_out(candidate_display_id):
                    defer_lifecycle = True
                    resolution_status = "UNRESOLVED"
                    resolution_method = "BRING_IN_EPOCH_WINDOW"
                    current_distance = self._event_identity_distance(
                        candidate_display_id,
                        artifacts,
                    )
                    identity = {
                        "status": "MATCHED",
                        "reason": "trusted_binding_awaiting_take_out_window",
                        "display_object_id": candidate_display_id,
                        "previous_binding_id": str(binding["binding_id"]),
                        "distance": current_distance,
                        "dino_attempted": True,
                    }
                else:
                    resolution_status = "RESOLVED"
                    resolution_method = "TRUSTED_EPOCH_BINDING"
            else:
                try:
                    embedded = self._embed_artifacts(artifacts)
                    if embedded is None:
                        raise ValueError("bring-in has no valid full-frame mask")
                    display_ids = self._recent_display_ids()
                    identity = self._select_identity(
                        self._identity_scores(embedded[0], display_ids)
                    )
                    if identity.get("status") == "MATCHED":
                        candidate_display_id = str(
                            identity["display_object_id"]
                        )
                        if self._has_pending_take_out(
                            candidate_display_id
                        ):
                            defer_lifecycle = True
                            resolution_status = "UNRESOLVED"
                            resolution_method = "BRING_IN_DINO_WINDOW"
                        else:
                            binding = establish_shigure_binding(
                                runtime_session_id=self.runtime_session_id,
                                source_epoch_id=self.source_epoch_id,
                                raw_shigure_object_id=raw_id,
                                display_object_id=candidate_display_id,
                                established_by="BRING_IN_DINO",
                                established_event_uid=(
                                    str(event.get("event_uid") or "") or None
                                ),
                                confidence=max(
                                    0.0, 1.0 - float(identity["distance"])
                                ),
                                detail=identity,
                            )
                            resolution_status = "RESOLVED"
                            resolution_method = "BRING_IN_DINO"
                    else:
                        resolution_status = (
                            "AMBIGUOUS"
                            if identity.get("status") == "AMBIGUOUS"
                            else "UNRESOLVED"
                        )
                except Exception as exc:
                    resolution_status = "UNRESOLVED"
                    identity = {"status": "UNBOUND", "reason": str(exc)}

        recorded_binding_id = None
        recorded_display_id = None
        if defer_lifecycle:
            recorded_binding_id = (
                candidate_binding_id if action == "take_out" else None
            )
            recorded_display_id = candidate_display_id or None
        elif binding is not None and resolution_status == "RESOLVED":
            recorded_binding_id = str(binding["binding_id"])
            recorded_display_id = str(binding["display_object_id"])
        canonical = record_shigure_canonical_event(
            runtime_session_id=self.runtime_session_id,
            source_epoch_id=self.source_epoch_id,
            stamp_sec=frame.source_stamp.sec,
            stamp_nanosec=frame.source_stamp.nanosec,
            frame_id=frame.frame_id,
            detection_index=detection_index,
            action=action,
            bbox=bbox_payload,
            resolution_status=resolution_status,
            raw_shigure_object_id=raw_id or None,
            binding_id=recorded_binding_id,
            display_object_id=recorded_display_id,
            resolution_method=resolution_method,
            collider=tracking.get("collider"),
            mask_artifact_path=artifacts.mask,
            scene_image_path=artifacts.scene,
            object_crop_path=artifacts.crop,
            skeleton=skeleton,
            source_stamp={**frame.source_stamp.to_dict(), "frame_id": frame.frame_id},
            detail={"adapter_event_uid": event.get("event_uid"), "adapter_mapping_method": event.get("mapping_method"), "identity": identity},
        )
        if action == "obj_move":
            # Even a malformed/colliding database response can never promote
            # untrusted move evidence into a lifecycle transition.
            return
        # Resolution is monotonic in the database. A take-out releases its
        # active binding, but a later exact-stamp RGB/person replay must still
        # reach apply_object_lifecycle_event so it can supplement the existing
        # history row without advancing presence again.
        canonical_status = str(
            canonical.get("resolution_status") or resolution_status
        ).upper()
        if defer_lifecycle and canonical_status not in {"RESOLVED", "REJECTED"}:
            self._queue_lifecycle_pose_candidate(
                action=action,
                canonical_event_uid=str(canonical["event_uid"]),
                display_object_id=candidate_display_id,
                raw_id=raw_id,
                binding_id=(
                    candidate_binding_id if action == "take_out" else None
                ),
                resolution_method=str(resolution_method),
                identity=identity,
                skeleton=skeleton,
                calibration_revision=calibration_revision,
                marker_rotation=marker_rotation,
                marker_translation=marker_translation,
                occurred_at=frame.received_utc,
                frame=frame,
                artifacts=artifacts,
            )
            return
        if canonical_status != "RESOLVED":
            return
        try:
            lifecycle = apply_object_lifecycle_event(
                canonical_event_uid=str(canonical["event_uid"]),
                skeleton=skeleton,
                calibration_revision=calibration_revision,
                occurred_at=frame.received_utc,
            )
            # A missing current binding is valid only for replay of a lifecycle
            # event whose take-out already revoked it. The database call above
            # remains the fail-closed authority for every other case.
            if binding is None:
                return
            if not bool(lifecycle.get("_replayed")):
                with self._lock:
                    if action == "bring_in":
                        self._view_windows[raw_id] = {
                            "display_object_id": str(binding["display_object_id"]),
                            "source_generation": self._source_generation,
                            "attempts": 0,
                            "added": 0,
                        }
                    elif action == "take_out":
                        self._view_windows.pop(raw_id, None)
                        self._spatial_boxes.pop(
                            str(binding["binding_id"]), None
                        )
                        self._stable.pop(raw_id, None)
                        self._last_stable_source_key.pop(raw_id, None)
                        self._last_view_sequence.pop(raw_id, None)
        except Exception as exc:
            print(f"[shigure-v2] lifecycle event rejected: {exc}")

    def _spatial_box_state(
        self,
        binding: Mapping[str, Any],
        raw_id: str,
        model_revision: int,
        persisted_corners: Any,
    ) -> SpatialBoxObservationState:
        binding_id = str(binding["binding_id"])
        current = self._spatial_boxes.get(binding_id)
        if (
            current is not None
            and current.raw_id == raw_id
            and current.model_revision == int(model_revision)
        ):
            return current
        current = SpatialBoxObservationState(
            binding_id=binding_id,
            raw_id=raw_id,
            model_revision=int(model_revision),
        )
        try:
            value = (
                json.loads(persisted_corners)
                if isinstance(persisted_corners, str)
                else persisted_corners
            )
            center, extent = _box_center_extent(value)
            current.published_center = center
            current.published_extent = extent
            current.filtered_center = center.copy()
            current.filtered_extent = extent.copy()
            current.cleared = False
        except Exception:
            pass
        self._spatial_boxes[binding_id] = current
        return current

    @staticmethod
    def _accept_spatial_box_measurement(
        state: SpatialBoxObservationState,
        corners: Any,
        received_monotonic: float,
    ) -> tuple[bool, list[list[float]] | None]:
        center, extent = _box_center_extent(corners)
        state.centers.append(center)
        state.extents.append(extent)
        median_center = np.median(
            np.stack(tuple(state.centers), axis=0), axis=0
        )
        median_extent = np.median(
            np.stack(tuple(state.extents), axis=0), axis=0
        )
        if state.filtered_center is None or state.filtered_extent is None:
            state.filtered_center = median_center
            state.filtered_extent = median_extent
        else:
            alpha = SHIGURE_SPATIAL_BOX_EMA_ALPHA
            state.filtered_center = (
                alpha * median_center
                + (1.0 - alpha) * state.filtered_center
            )
            state.filtered_extent = (
                alpha * median_extent
                + (1.0 - alpha) * state.filtered_extent
            )
        state.last_good_monotonic = float(received_monotonic)
        state.pending_missing_since = None
        state.pending_missing_sequence = 0
        state.good_streak += 1
        if state.good_streak < SHIGURE_SPATIAL_BOX_ACQUIRE_FRAMES:
            return False, None

        center_changed = (
            state.published_center is None
            or float(
                np.linalg.norm(
                    state.filtered_center - state.published_center
                )
            )
            >= SHIGURE_SPATIAL_BOX_CENTER_DEADBAND_M
        )
        extent_changed = (
            state.published_extent is None
            or float(
                np.max(
                    np.abs(state.filtered_extent - state.published_extent)
                )
            )
            >= SHIGURE_SPATIAL_BOX_EXTENT_DEADBAND_M
        )
        if not state.cleared and not center_changed and not extent_changed:
            return False, None

        published = _corners_from_center_extent(
            state.filtered_center, state.filtered_extent
        )
        return True, published

    @staticmethod
    def _mark_spatial_box_missing(
        state: SpatialBoxObservationState,
        received_monotonic: float,
        sequence: int,
    ) -> tuple[bool, None]:
        state.good_streak = 0
        if state.pending_missing_since is None:
            state.pending_missing_since = float(received_monotonic)
            state.pending_missing_sequence = int(sequence)
            if SHIGURE_SPATIAL_BOX_MISSING_GRACE_SECONDS > 0.0:
                return False, None
        if (
            float(received_monotonic) - state.pending_missing_since
            < SHIGURE_SPATIAL_BOX_MISSING_GRACE_SECONDS
            or state.cleared
        ):
            return False, None
        return True, None

    @staticmethod
    def _commit_spatial_box_ready(
        state: SpatialBoxObservationState,
    ) -> None:
        if state.filtered_center is None or state.filtered_extent is None:
            raise ValueError("cannot commit an unfiltered spatial box")
        state.published_center = state.filtered_center.copy()
        state.published_extent = state.filtered_extent.copy()
        state.cleared = False

    @staticmethod
    def _commit_spatial_box_clear(
        state: SpatialBoxObservationState,
    ) -> None:
        state.cleared = True
        state.pending_missing_since = None
        state.pending_missing_sequence = 0
        state.published_center = None
        state.published_extent = None
        state.filtered_center = None
        state.filtered_extent = None
        state.centers.clear()
        state.extents.clear()

    def _expire_pending_spatial_boxes(
        self, now_monotonic: float
    ) -> None:
        if not self.source_epoch_id:
            return
        for state in tuple(self._spatial_boxes.values()):
            if (
                state.cleared
                or state.pending_missing_since is None
                or float(now_monotonic) - state.pending_missing_since
                < SHIGURE_SPATIAL_BOX_MISSING_GRACE_SECONDS
            ):
                continue
            binding = get_active_shigure_binding(
                source_epoch_id=self.source_epoch_id,
                raw_shigure_object_id=state.raw_id,
            )
            if (
                binding is None
                or str(binding["binding_id"]) != state.binding_id
            ):
                continue
            result = update_shigure_live_observation(
                binding_id=state.binding_id,
                observation_seq=int(state.pending_missing_sequence),
                model_revision=int(state.model_revision),
                spatial_box_corners_aruco=None,
                spatial_box_observed=True,
            )
            if result is not None and bool(
                result.get("_spatial_box_accepted")
            ):
                self._commit_spatial_box_clear(state)

    def _update_tracked_geometry(self, frame: CachedShigureFrame) -> None:
        if not self.source_epoch_id:
            return
        tracking_state = str(
            frame.input_states.get("object_tracking") or "missing"
        )
        tracking_stamp = (
            int(frame.source_stamp.sec),
            int(frame.source_stamp.nanosec),
        )
        current_tracking_snapshot = (
            tracking_state in {"present", "explicit_empty"}
            and (
                self._last_geometry_stamp is None
                or tracking_stamp >= self._last_geometry_stamp
            )
        )
        fresh_tracking_snapshot = (
            current_tracking_snapshot
            and (
                self._last_geometry_stamp is None
                or tracking_stamp > self._last_geometry_stamp
            )
        )
        if fresh_tracking_snapshot and any(
            str(item.get("action") or "").strip().lower() == "obj_move"
            for item in frame.tracked_objects
            if isinstance(item, Mapping)
        ):
            # Defense in depth for direct method callers. The whole snapshot
            # is indeterminate: consume no collider and authorize no new
            # missing observation. Existing clean missing evidence is retained.
            self._last_geometry_stamp = tracking_stamp
            return

        calibration = None
        try:
            current = _camera_to_aruco()
            calibration = (
                np.asarray(current[0], dtype=np.float64),
                np.asarray(current[1], dtype=np.float64),
                np.asarray(current[2], dtype=np.float64),
                str(current[3]),
            )
            self._last_camera_calibration = calibration
        except Exception:
            # A temporary calibration read failure is not evidence that an
            # object disappeared. Reuse the last authorized transform only.
            calibration = self._last_camera_calibration

        people = [item for item in frame.people if isinstance(item, Mapping)]
        observed_raw_ids: set[str] = set()
        active_binding_ids: set[str] = set()
        if fresh_tracking_snapshot:
            active_binding_ids = {
                str(item["binding_id"])
                for item in list_active_shigure_bindings(
                    self.source_epoch_id
                )
            }
            for binding_id in list(self._spatial_boxes):
                if binding_id not in active_binding_ids:
                    self._spatial_boxes.pop(binding_id, None)

        for tracked in frame.tracked_objects:
            action = str(tracked.get("action") or "").strip().lower()
            raw_id = str(tracked.get("object_id") or "").strip()
            if not raw_id or action == "take_out":
                continue
            if action == "obj_move":
                # Current upstream obj_move IDs are not generally reliable.
                # Treat the row as observed so flicker grace does not delete a
                # valid box, but never consume its collider or identity.
                observed_raw_ids.add(raw_id)
                continue
            if action not in {"stay", "bring_in"}:
                continue
            observed_raw_ids.add(raw_id)
            binding = get_active_shigure_binding(
                source_epoch_id=self.source_epoch_id,
                raw_shigure_object_id=raw_id,
            )
            if binding is None:
                continue
            display_state = get_display_object_state(
                str(binding["display_object_id"])
            ) or {}
            model_revision = int(
                display_state.get("active_model_revision") or 0
            )
            if model_revision <= 0:
                continue
            box_state = self._spatial_box_state(
                binding,
                raw_id,
                model_revision,
                display_state.get("latest_spatial_box_aruco_json"),
            )

            skeleton = None
            if calibration is not None and current_tracking_snapshot:
                _camera_to_aruco_matrix, marker_rotation, marker_translation, _ = (
                    calibration
                )
                person = _nearest_person(people, _bbox(tracked))
                skeleton = _person_skeleton(
                    person, marker_rotation, marker_translation
                )

            spatial_box_observed = False
            spatial_box_corners = None
            if fresh_tracking_snapshot and calibration is not None:
                camera_to_aruco, _rotation, _translation, _revision = calibration
                try:
                    raw_box = build_spatial_box_v2(
                        tracked.get("collider"),
                        camera_to_aruco,
                        np.eye(4),
                    )
                    spatial_box_observed, spatial_box_corners = (
                        self._accept_spatial_box_measurement(
                            box_state,
                            raw_box["corners_aruco_m"],
                            frame.received_monotonic,
                        )
                    )
                except NoBoxError:
                    spatial_box_observed, spatial_box_corners = (
                        self._mark_spatial_box_missing(
                            box_state,
                            frame.received_monotonic,
                            int(frame.sequence),
                        )
                    )

            if spatial_box_observed or skeleton is not None:
                result = update_shigure_live_observation(
                    binding_id=str(binding["binding_id"]),
                    observation_seq=int(frame.sequence),
                    model_revision=model_revision,
                    spatial_box_corners_aruco=spatial_box_corners,
                    spatial_box_observed=spatial_box_observed,
                    skeleton=skeleton,
                )
                if (
                    spatial_box_observed
                    and result is not None
                    and bool(result.get("_spatial_box_accepted"))
                ):
                    if spatial_box_corners is None:
                        self._commit_spatial_box_clear(box_state)
                    else:
                        self._commit_spatial_box_ready(box_state)

        if not fresh_tracking_snapshot:
            return
        for binding in list_active_shigure_bindings(self.source_epoch_id):
            raw_id = str(binding["raw_shigure_object_id"])
            if raw_id in observed_raw_ids:
                continue
            display_state = get_display_object_state(
                str(binding["display_object_id"])
            ) or {}
            model_revision = int(
                display_state.get("active_model_revision") or 0
            )
            if model_revision <= 0:
                continue
            box_state = self._spatial_box_state(
                binding,
                raw_id,
                model_revision,
                display_state.get("latest_spatial_box_aruco_json"),
            )
            should_clear, _ = self._mark_spatial_box_missing(
                box_state,
                frame.received_monotonic,
                int(frame.sequence),
            )
            if should_clear:
                result = update_shigure_live_observation(
                    binding_id=str(binding["binding_id"]),
                    observation_seq=int(frame.sequence),
                    model_revision=model_revision,
                    spatial_box_corners_aruco=None,
                    spatial_box_observed=True,
                )
                if result is not None and bool(
                    result.get("_spatial_box_accepted")
                ):
                    self._commit_spatial_box_clear(box_state)
        self._last_geometry_stamp = tracking_stamp

    def _schedule_stable_identity_view(
        self,
        *,
        raw_id: str,
        binding: Mapping[str, Any],
        sequence: int,
        source_stamp: Mapping[str, Any],
        artifacts: EventArtifacts,
    ) -> bool:
        inflight_key = f"{self.source_epoch_id}:{raw_id}"
        with self._lock:
            window = self._view_windows.get(raw_id)
            if window is None:
                window = {
                    "display_object_id": str(binding["display_object_id"]),
                    "source_generation": self._source_generation,
                    "attempts": 0,
                    "added": 0,
                    "closed": False,
                }
                self._view_windows[raw_id] = window
            capture_allowed = bool(
                window is not None
                and int(window["source_generation"])
                == self._source_generation
                and str(window["display_object_id"])
                == str(binding["display_object_id"])
                and not bool(window.get("closed"))
                and int(window["attempts"])
                < SHIGURE_IDENTITY_CAPTURE_MAX_ATTEMPTS
                and int(window["added"])
                < SHIGURE_IDENTITY_CAPTURE_MAX_NEW_VIEWS
            )
            if (
                inflight_key in self._view_inflight
                or self._last_view_sequence.get(raw_id, -1)
                >= int(sequence)
            ):
                return False
            if capture_allowed and window is not None:
                window["attempts"] = int(window["attempts"]) + 1
            self._last_view_sequence[raw_id] = int(sequence)
            self._view_inflight.add(inflight_key)
            source_generation = self._source_generation
            source_epoch_id = str(self.source_epoch_id)
            binding_id = str(binding["binding_id"])
        try:
            future = self._identity_executor.submit(
                self._run_stable_identity_view_guarded,
                inflight_key,
                raw_id,
                str(binding["display_object_id"]),
                int(sequence),
                dict(source_stamp),
                artifacts,
                source_generation,
                source_epoch_id,
                binding_id,
                capture_allowed,
            )
            future.add_done_callback(
                lambda _future, key=inflight_key, owned=artifacts: (
                    self._finish_stable_identity_future(key, owned)
                )
            )
            return True
        except Exception:
            with self._lock:
                self._view_inflight.discard(inflight_key)
            raise

    def _finish_stable_identity_future(
        self, inflight_key: str, artifacts: EventArtifacts
    ) -> None:
        with self._lock:
            self._view_inflight.discard(inflight_key)
        _cleanup_event_artifacts(artifacts)

    def _run_stable_identity_view_guarded(
        self,
        inflight_key: str,
        raw_id: str,
        display_object_id: str,
        sequence: int,
        source_stamp: Mapping[str, Any],
        artifacts: EventArtifacts,
        source_generation: int,
        source_epoch_id: str,
        binding_id: str,
        capture_allowed: bool,
    ) -> None:
        reference: dict[str, Any] | None = None
        try:
            with self._lock:
                if (
                    self._source_generation != source_generation
                    or str(self.source_epoch_id) != source_epoch_id
                ):
                    return
            binding = get_active_shigure_binding(
                source_epoch_id=source_epoch_id,
                raw_shigure_object_id=raw_id,
            )
            if (
                binding is None
                or str(binding["binding_id"]) != binding_id
                or str(binding["display_object_id"]) != display_object_id
            ):
                return
            embedded = self._embed_artifacts(artifacts)
            if embedded is None:
                return
            admission = self._verify_shigure_view_admission(
                display_object_id, embedded[0]
            )
            if str(admission.get("admission_status") or "") != "MATCHED":
                return
            with self._lock:
                window = self._view_windows.get(raw_id)
                still_current = (
                    self._source_generation == source_generation
                    and str(self.source_epoch_id) == source_epoch_id
                )
                capture_still_allowed = bool(
                    capture_allowed
                    and still_current
                    and window is not None
                    and int(window["source_generation"]) == source_generation
                    and str(window["display_object_id"]) == display_object_id
                    and not bool(window.get("closed"))
                )
            if not still_current:
                return
            binding = get_active_shigure_binding(
                source_epoch_id=source_epoch_id,
                raw_shigure_object_id=raw_id,
            )
            if (
                binding is None
                or str(binding["binding_id"]) != binding_id
                or str(binding["display_object_id"]) != display_object_id
            ):
                return
            if capture_still_allowed:
                reference = self._register_view(
                    display_object_id=display_object_id,
                    artifacts=artifacts,
                    embedding=embedded[0],
                    embedding_response=embedded[1],
                    source_event_uid=None,
                    quality={
                        "role": "stable_post_bring_in_view",
                        "raw_shigure_object_id": raw_id,
                        "frame_sequence": int(sequence),
                        "source_generation": int(source_generation),
                        "stable_mask_frames": (
                            SHIGURE_EXAMPLE_STABLE_MASK_FRAMES
                        ),
                    },
                    source_epoch_id=source_epoch_id,
                    binding_id=binding_id,
                    raw_shigure_object_id=raw_id,
                    admission=admission,
                )
            # Stable tracking frames remain identity-reference inputs only.
            # A lifecycle take_out is the sole trigger allowed to update the
            # object's historical/latest position through FoundationPose.
        except Exception as exc:
            print(
                f"[shigure-v2] strict stable candidate failed for "
                f"{raw_id}: {exc}"
            )
        finally:
            with self._lock:
                self._view_inflight.discard(inflight_key)
                window = self._view_windows.get(raw_id)
                if (
                    window is not None
                    and int(window["source_generation"])
                    == source_generation
                ):
                    if reference is not None:
                        window["added"] = int(window["added"]) + 1
                    if (
                        int(window["attempts"])
                        >= SHIGURE_IDENTITY_CAPTURE_MAX_ATTEMPTS
                        or int(window["added"])
                        >= SHIGURE_IDENTITY_CAPTURE_MAX_NEW_VIEWS
                    ):
                        window["closed"] = True
            _cleanup_event_artifacts(artifacts)

    def _collect_stable_pose_candidates(
        self, frame: CachedShigureFrame
    ) -> None:
        if not self.source_epoch_id:
            return
        source_key = sample_key(frame.source_stamp)
        complete_strict_snapshot = (
            frame.input_states.get("segments")
            in {"present", "explicit_empty"}
            and frame.input_states.get("object_tracking")
            in {"present", "explicit_empty"}
            and source_key
            != self._last_strict_candidate_snapshot_source_key
        )
        if complete_strict_snapshot:
            self._last_strict_candidate_snapshot_source_key = source_key
        sample = self._sample_exact(frame)
        if sample is None:
            if complete_strict_snapshot:
                self._stable.clear()
                self._last_stable_source_key.clear()
            return
        observed_strict_raw_ids: set[str] = set()
        for candidate in frame.recovery_candidates:
            raw_id = str(
                candidate.get("shigure_object_id") or ""
            ).strip()
            tracking = (
                candidate.get("tracking")
                if isinstance(candidate.get("tracking"), Mapping)
                else {}
            )
            trusted = (
                str(candidate.get("mask_source") or "")
                == "segments_polygon"
                and str(candidate.get("tracking_match_status") or "").upper()
                == "RESOLVED"
                and str(candidate.get("tracking_mapping_method") or "")
                == "SEGMENT_TRACKING_UNIQUE_IOU"
                and str(tracking.get("action") or "").strip().lower()
                in {"stay", "bring_in"}
            )
            try:
                probability = float(candidate.get("probability") or 0.0)
            except (TypeError, ValueError):
                probability = 0.0
            box = _bbox(candidate)
            if (
                not trusted
                or probability < SHIGURE_EXAMPLE_MIN_SEGMENT_PROBABILITY
                or not raw_id
                or box is None
            ):
                if raw_id:
                    self._stable.pop(raw_id, None)
                    self._last_stable_source_key.pop(raw_id, None)
                continue
            observed_strict_raw_ids.add(raw_id)
            binding = get_active_shigure_binding(
                source_epoch_id=self.source_epoch_id,
                raw_shigure_object_id=raw_id,
            )
            if binding is None:
                continue
            try:
                candidate_mask = _decode_full_mask(
                    candidate.get("mask_b64"), sample.rgb_bgr.shape[:2]
                )
            except Exception:
                self._stable.pop(raw_id, None)
                self._last_stable_source_key.pop(raw_id, None)
                continue

            history = self._stable.setdefault(
                raw_id,
                deque(maxlen=SHIGURE_EXAMPLE_STABLE_MASK_FRAMES),
            )
            current = (
                int(frame.sequence),
                box,
                candidate,
                candidate_mask,
            )
            if (
                history
                and self._last_stable_source_key.get(raw_id) == source_key
            ):
                history[-1] = current
                if any(
                    not _strict_example_pair_is_stable(
                        item[1], item[3], box, candidate_mask
                    )
                    for item in tuple(history)[:-1]
                ):
                    history.clear()
                    history.append(current)
                continue
            if history and any(
                not _strict_example_pair_is_stable(
                    item[1], item[3], box, candidate_mask
                )
                for item in history
            ):
                history.clear()
            history.append(current)
            self._last_stable_source_key[raw_id] = source_key
            if len(history) < SHIGURE_EXAMPLE_STABLE_MASK_FRAMES:
                continue

            identity_artifacts: EventArtifacts | None = None
            try:
                identity_artifacts = self._candidate_artifacts(
                    frame, candidate, sample
                )
                scheduled = self._schedule_stable_identity_view(
                    raw_id=raw_id,
                    binding=binding,
                    sequence=int(frame.sequence),
                    source_stamp=frame.source_stamp.to_dict(),
                    artifacts=identity_artifacts,
                )
                if not scheduled:
                    _cleanup_event_artifacts(identity_artifacts)
            except Exception as exc:
                _cleanup_event_artifacts(identity_artifacts)
                print(
                    "[shigure-v2] strict stable identity candidate failed: "
                    f"{exc}"
                )

        if complete_strict_snapshot:
            for stale_raw_id in set(self._stable) - observed_strict_raw_ids:
                self._stable.pop(stale_raw_id, None)
                self._last_stable_source_key.pop(stale_raw_id, None)

    def _foundationpose_snapshot(
        self,
        source_stamp: Mapping[str, Any],
        artifacts: EventArtifacts,
        sample: CachedRgbdSample,
    ) -> dict[str, Any]:
        if artifacts.scene is None or artifacts.mask is None:
            raise ValueError("FoundationPose requires scene and full mask")
        root = artifacts.scene.parent
        depth_path = root / "depth.png"
        _write_image(depth_path, np.asarray(sample.depth, dtype=np.uint16))
        matrix = np.asarray(sample.camera_info.get("k"), dtype=np.float64).reshape(3, 3)
        return {
            "color_file": str(artifacts.scene),
            "depth_file": str(depth_path),
            "mask_file": str(artifacts.mask),
            "k": matrix.astype(float).tolist(),
            "source_stamp": dict(source_stamp),
            "temporary_root": str(artifacts.temporary_root or ""),
        }

    @staticmethod
    def _foundationpose_to_aruco_pose(
        pose_cv: np.ndarray,
        scale: float,
        marker_calibration: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        camera_basis = np.asarray(OPENCV_CAMERA_TO_CANONICAL_RH_BASIS, dtype=np.float64)
        model_basis = np.asarray(MODEL_INPUT_TO_CANONICAL_RH_BASIS, dtype=np.float64)
        local_rotation_rh = camera_basis @ pose_cv[:3, :3] @ model_basis.T
        local_translation_rh = camera_basis @ pose_cv[:3, 3]
        local_rotation, local_translation = model_pose_canonical_rh_to_unity_camera(local_rotation_rh, local_translation_rh)
        if marker_calibration is None:
            (
                _transform,
                marker_rotation,
                marker_translation,
                _revision,
            ) = _camera_to_aruco()
        else:
            marker_rotation, marker_translation = marker_calibration
        marker_rotation = np.asarray(marker_rotation, dtype=np.float64)
        marker_translation = np.asarray(
            marker_translation, dtype=np.float64
        )
        basis = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
        aruco_from_camera_rotation = orthonormalize_rotation(basis @ marker_rotation.T @ basis)
        camera_origin_aruco = basis @ (marker_rotation.T @ (-marker_translation))
        runtime_correction = (
            np.asarray(FBX_RUNTIME_TRANSFORM_COMPENSATION_TO_UNITY, dtype=np.float64)
            @ np.asarray(RUNTIME_LOCAL_TO_UNITY_POSE_ROTATION, dtype=np.float64)
        )
        position = aruco_from_camera_rotation @ np.asarray(local_translation) + camera_origin_aruco
        rotation = orthonormalize_rotation(aruco_from_camera_rotation @ np.asarray(local_rotation) @ runtime_correction)
        return rt_to_pose(rotation, position, scale=[float(scale)] * 3)

    @staticmethod
    def _validate_foundationpose(
        mesh_path: Path, pose_cv: np.ndarray, scale: float, mask: np.ndarray, depth: np.ndarray, k: np.ndarray
    ) -> dict[str, Any]:
        vertices = read_obj_vertices(mesh_path)
        if len(vertices) > 50000:
            vertices = vertices[np.linspace(0, len(vertices) - 1, 50000).astype(np.int64)]
        transformed = (pose_cv[:3, :3] @ (vertices.astype(np.float64) * scale).T).T + pose_cv[:3, 3]
        transformed = transformed[transformed[:, 2] > 1.0e-5]
        if not len(transformed):
            return {"accepted": False, "reason": "model_behind_camera"}
        pixels = np.column_stack((k[0, 0] * transformed[:, 0] / transformed[:, 2] + k[0, 2], k[1, 1] * transformed[:, 1] / transformed[:, 2] + k[1, 2]))
        finite = np.isfinite(pixels).all(axis=1)
        pixels, transformed = pixels[finite], transformed[finite]
        if not len(pixels):
            return {"accepted": False, "reason": "projection_empty"}
        projected = (float(pixels[:, 0].min()), float(pixels[:, 1].min()), float(pixels[:, 0].max() + 1), float(pixels[:, 1].max() + 1))
        observed = _mask_bbox(mask)
        iou = _bbox_iou(projected, observed)
        depth_m = depth.astype(np.float32) / 1000.0 if np.issubdtype(depth.dtype, np.integer) else depth.astype(np.float32)
        valid_depth = depth_m[mask & np.isfinite(depth_m) & (depth_m > 0.05)]
        residual = abs(float(np.median(transformed[:, 2])) - float(np.median(valid_depth))) if valid_depth.size else float("inf")
        accepted = iou >= REALTIME_TRACKING_FP_MIN_BBOX_IOU and residual <= REALTIME_TRACKING_FP_MAX_DEPTH_RESIDUAL_M
        return {"accepted": accepted, "bbox_iou": iou, "depth_residual_m": residual}


__all__ = ["EventArtifacts", "ShigureRuntimeEngine"]
