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
from dataclasses import dataclass, field
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

from artifact_layout import IDENTITY_REFERENCE_ROOT, SHIGURE_EVENT_ROOT
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
    SHIGURE_HOLO_SYNC_SIZE_LOG_TOLERANCE,
    SHIGURE_IDENTITY_HOLOLENS_DISTANCE_PENALTY,
    SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
    SHIGURE_IDENTITY_CAPTURE_MAX_ATTEMPTS,
    SHIGURE_IDENTITY_CAPTURE_MAX_NEW_VIEWS,
    SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD,
    SHIGURE_IDENTITY_MATCH_REQUIRE_MARGIN,
    SHIGURE_IDENTITY_MATCH_SECOND_MARGIN,
    SHIGURE_IDENTITY_VIEW_NOVELTY_DISTANCE,
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
    add_object_identity_reference,
    apply_object_lifecycle_event,
    close_shigure_runtime_session,
    establish_shigure_binding,
    get_active_shigure_binding,
    get_display_object_state,
    get_task_by_task_id,
    list_active_shigure_bindings,
    list_display_object_states,
    list_object_identity_references,
    open_shigure_source_epoch,
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
    """Priority/cardinality preserving minimum-cost one-to-one assignment.

    Candidate count is unbounded, but persistent display candidates are capped
    at five. A display-bitmask DP therefore avoids exponential Segments scans.
    """

    priorities = (
        [1] * len(candidate_scores)
        if candidate_priorities is None
        else [max(0, int(value)) for value in candidate_priorities]
    )
    if len(priorities) != len(candidate_scores):
        raise ValueError("candidate_priorities must match candidate_scores")
    eligible: list[list[dict[str, Any]]] = []
    for scores in candidate_scores:
        by_display: dict[str, dict[str, Any]] = {}
        for score in scores:
            display_object_id = str(score.get("display_object_id") or "").strip()
            try:
                distance = float(score["distance"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not display_object_id
                or not np.isfinite(distance)
                or distance > SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD
            ):
                continue
            item = dict(score)
            previous = by_display.get(display_object_id)
            if previous is None or distance < float(previous["distance"]):
                by_display[display_object_id] = item
        eligible.append(
            sorted(
                by_display.values(),
                key=lambda item: (float(item["distance"]), str(item["display_object_id"])),
            )
        )
    display_ids = sorted(
        {
            str(score["display_object_id"])
            for scores in eligible
            for score in scores
        }
    )
    if len(display_ids) > SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS:
        raise ValueError(
            "global identity assignment received more than five display candidates"
        )
    display_indexes = {
        display_object_id: index
        for index, display_object_id in enumerate(display_ids)
    }
    score_maps = [
        {str(score["display_object_id"]): score for score in scores}
        for scores in eligible
    ]

    # State: priority sum, distance, deterministic signature, per-candidate
    # display index (-1 means unmatched). At a fixed used-display mask, only
    # the highest-priority/lowest-cost prefix can improve any suffix.
    DpState = tuple[int, float, tuple[str, ...], tuple[int, ...]]

    def solve(
        different_at: tuple[int, str | None] | None = None,
    ) -> dict[str, Any] | None:
        states: dict[int, DpState] = {0: (0, 0.0, (), ())}

        def retain(mask: int, candidate: DpState, output: dict[int, DpState]) -> None:
            current = output.get(mask)
            candidate_rank = (-candidate[0], candidate[1], candidate[2])
            current_rank = (
                (-current[0], current[1], current[2])
                if current is not None
                else None
            )
            if current_rank is None or candidate_rank < current_rank:
                output[mask] = candidate

        for candidate_index, scores in enumerate(eligible):
            next_states: dict[int, DpState] = {}
            constrained = (
                different_at is not None and different_at[0] == candidate_index
            )
            forbidden = different_at[1] if constrained else object()
            for mask, state in states.items():
                if not constrained or forbidden is not None:
                    retain(
                        mask,
                        (
                            state[0],
                            state[1],
                            (*state[2], "\uffff"),
                            (*state[3], -1),
                        ),
                        next_states,
                    )
                for score in scores:
                    display_object_id = str(score["display_object_id"])
                    if constrained and forbidden == display_object_id:
                        continue
                    display_index = display_indexes[display_object_id]
                    bit = 1 << display_index
                    if mask & bit:
                        continue
                    retain(
                        mask | bit,
                        (
                            state[0] + priorities[candidate_index],
                            state[1] + float(score["distance"]),
                            (*state[2], display_object_id),
                            (*state[3], display_index),
                        ),
                        next_states,
                    )
            states = next_states
            if not states:
                return None

        best_mask, best_state = min(
            states.items(),
            key=lambda item: (
                -item[1][0],
                -int(item[0]).bit_count(),
                item[1][1],
                item[1][2],
            ),
        )
        assignments = {
            candidate_index: score_maps[candidate_index][display_ids[display_index]]
            for candidate_index, display_index in enumerate(best_state[3])
            if display_index >= 0
        }
        return {
            "assignments": assignments,
            "priority_matched_count": int(best_state[0]),
            "matched_count": int(best_mask).bit_count(),
            "total_distance": float(best_state[1]),
        }

    best = solve()
    assert best is not None
    assignments = best["assignments"]
    best_priority_count = int(best["priority_matched_count"])
    best_cardinality = int(best["matched_count"])
    best_distance = float(best["total_distance"])
    ambiguous_candidates: set[int] = set()
    alternative_distances: list[float] = []
    for candidate_index in range(len(eligible)):
        selected_display = (
            str(assignments[candidate_index]["display_object_id"])
            if candidate_index in assignments
            else None
        )
        alternative = solve((candidate_index, selected_display))
        if (
            alternative is None
            or int(alternative["priority_matched_count"]) != best_priority_count
            or int(alternative["matched_count"]) != best_cardinality
        ):
            continue
        distance = float(alternative["total_distance"])
        alternative_distances.append(distance)
        if (
            SHIGURE_IDENTITY_MATCH_REQUIRE_MARGIN
            and distance - best_distance < SHIGURE_IDENTITY_MATCH_SECOND_MARGIN
        ):
            ambiguous_candidates.add(candidate_index)
    second_distance = min(alternative_distances, default=None)
    return {
        "assignments": assignments,
        "ambiguous_candidates": ambiguous_candidates,
        "priority_matched_count": best_priority_count,
        "matched_count": best_cardinality,
        "total_distance": best_distance,
        "second_total_distance": second_distance,
        "assignment_margin": (
            second_distance - best_distance if second_distance is not None else None
        ),
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
    target_center_aruco: np.ndarray
    target_size_aruco: np.ndarray
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
            maxlen=SHIGURE_EXAMPLE_STABLE_MASK_FRAMES
        )
    )
    stable_count: int = 0
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
        self._last_fp_sequence: dict[str, int] = {}
        self._fp_inflight: set[str] = set()
        self._last_view_sequence: dict[str, int] = {}
        self._view_inflight: set[str] = set()
        self._view_windows: dict[str, dict[str, Any]] = {}
        self._spatial_boxes: dict[str, SpatialBoxObservationState] = {}
        self._last_geometry_stamp: tuple[int, int] | None = None
        self._runtime_obj_move_quarantine_active = False
        self._runtime_obj_move_barrier_stamp: tuple[int, int] | None = None
        self._last_camera_calibration: tuple[
            np.ndarray, np.ndarray, np.ndarray, str
        ] | None = None

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
                },
            )
            self.runtime_session_id = str(session["runtime_session_id"])
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
            self._expire_pending_spatial_boxes(time.monotonic())
            return 0
        if latest.source_incarnation_id != self.source_incarnation_id:
            self._open_incarnation(
                latest.source_incarnation_id,
                reset_sequence=int(latest.sequence) <= int(self.last_sequence),
            )
        processed = 0
        frames = list(self.cache.iter_canonical_updates_after(self.last_sequence, include_masks=True))
        for frame in frames:
            # A same-process adapter rotation leaves old frames in the bounded
            # cache. Only the latest incarnation is authoritative; reopening
            # old epochs here would oscillate bindings.
            if frame.source_incarnation_id != self.source_incarnation_id:
                continue
            self._process_frame(frame)
            self.last_sequence = max(self.last_sequence, int(frame.sequence))
            processed += 1
        if self.last_sequence == 0:
            self._process_frame(latest)
            self.last_sequence = int(latest.sequence)
            processed = 1
        self._expire_pending_spatial_boxes(time.monotonic())
        return processed

    def _reset_epoch_local_runtime_state(self, *, holo_reason: str) -> None:
        with self._lock:
            self._source_generation += 1
            self._startup_recovery_pending = True
            self._startup_recovery_attempts = 0
            self._startup_recovery_last_source_key = ""
            self._startup_recovery_last_attempt_monotonic = float("-inf")
            self._stable.clear()
            self._last_stable_source_key.clear()
            self._last_strict_candidate_snapshot_source_key = ""
            self._last_fp_sequence.clear()
            self._last_view_sequence.clear()
            self._view_windows.clear()
            self._view_inflight.clear()
            self._spatial_boxes.clear()
            self._last_geometry_stamp = None
            self._runtime_obj_move_quarantine_active = False
            self._runtime_obj_move_barrier_stamp = None
            self._last_camera_calibration = None
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
        if self._startup_recovery_pending and self._startup_recovery(frame):
            self._startup_recovery_pending = False

    def _sample_exact(self, frame: CachedShigureFrame) -> CachedRgbdSample | None:
        sample = self.cache.get_sample(frame.source_stamp)
        return sample if sample is not None and sample.stamp == frame.source_stamp else None

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
            vector = self._ensure_reference_embedding(reference)
            if vector is not None:
                vectors.append((str(reference["reference_id"]), str(reference["source"]), vector))
        # Shigure views are the primary multi-view identity bank. Preserve the
        # newest HoloLens view as an auxiliary condition even after Shigure
        # references exist; its distance receives a source penalty below so it
        # cannot displace an equally-good Shigure observation.
        shigure = [item for item in vectors if item[1] == "SHIGURE"]
        latest_hololens = next((item for item in vectors if item[1] == "HOLOLENS"), None)
        return shigure + ([latest_hololens] if latest_hololens is not None else [])

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
                source_penalty = (
                    SHIGURE_IDENTITY_HOLOLENS_DISTANCE_PENALTY
                    if source == "HOLOLENS"
                    else 0.0
                )
                candidates.append(
                    (raw_distance + source_penalty, raw_distance, source_penalty, reference_id, source)
                )
            best = min(candidates, key=lambda item: (item[0], item[4] != "SHIGURE", item[3]))
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
        if artifacts.scene is None or artifacts.mask is None:
            return None
        resolved_admission = self._verify_shigure_view_admission(
            display_object_id, embedding
        )
        if admission is not None and any(
            str(admission.get(key) or "")
            != str(resolved_admission.get(key) or "")
            for key in (
                "admission_status",
                "admission_display_object_id",
                "admission_reference_id",
            )
        ):
            return None
        if (
            str(resolved_admission.get("admission_status") or "")
            != "MATCHED"
            or str(
                resolved_admission.get("admission_display_object_id") or ""
            )
            != str(display_object_id)
        ):
            return None
        identity_quality = {
            **dict(quality),
            **resolved_admission,
            "admission_source_epoch_id": str(source_epoch_id),
            "admission_binding_id": str(binding_id),
            "admission_raw_shigure_object_id": str(
                raw_shigure_object_id
            ),
        }
        existing_vectors: list[np.ndarray] = []
        for reference in list_object_identity_references(display_object_id, limit=100):
            if str(reference.get("source") or "") != "SHIGURE":
                continue
            vector = self._ensure_reference_embedding(reference)
            if vector is not None:
                existing_vectors.append(vector)
        novelty = min((cosine_distance(embedding, item) for item in existing_vectors), default=None)
        if novelty is not None and novelty < SHIGURE_IDENTITY_VIEW_NOVELTY_DISTANCE:
            return None
        digest = hashlib.sha256()
        digest.update(artifacts.scene.read_bytes())
        digest.update(artifacts.mask.read_bytes())
        view_digest = digest.hexdigest()
        # Candidate inference files are temporary. Persist only a view that
        # passed novelty filtering, under identity storage independent of the
        # lifecycle/debug artifact tree.
        view_root = IDENTITY_REFERENCE_ROOT / "views" / view_digest
        persistent_scene = view_root / "scene.png"
        persistent_mask = view_root / "mask.png"
        scene_preexisted = persistent_scene.exists()
        mask_preexisted = persistent_mask.exists()
        _atomic_bytes(persistent_scene, artifacts.scene.read_bytes())
        _atomic_bytes(persistent_mask, artifacts.mask.read_bytes())
        try:
            reference = add_object_identity_reference(
                display_object_id=display_object_id,
                source="SHIGURE",
                source_event_uid=source_event_uid,
                image_path=persistent_scene,
                mask_path=persistent_mask,
                view_hash=f"shigure:{view_digest}",
                quality={
                    **identity_quality,
                    "novelty_distance": novelty,
                },
            )
        except Exception:
            if not scene_preexisted:
                persistent_scene.unlink(missing_ok=True)
            if not mask_preexisted:
                persistent_mask.unlink(missing_ok=True)
            try:
                view_root.rmdir()
            except OSError:
                pass
            raise
        sidecar = self._embedding_sidecar(str(reference["reference_id"]), embedding_response)
        return set_object_identity_reference_embedding(str(reference["reference_id"]), embedding_path=sidecar)

    def _startup_recovery(self, frame: CachedShigureFrame) -> bool:
        if not self.runtime_session_id or not self.source_epoch_id:
            return False
        complete_empty_snapshot = (
            not frame.recovery_candidates
            and frame.input_states.get("segments") == "explicit_empty"
            and frame.input_states.get("object_tracking") == "explicit_empty"
        )
        if not frame.recovery_candidates and not complete_empty_snapshot:
            return False
        job_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"shigure-recovery:{self.runtime_session_id}:{self.source_epoch_id}",
        ).hex
        tracking_state = frame.input_states.get("object_tracking")
        if frame.recovery_candidates and tracking_state != "present":
            # Segments is often published before same-stamp tracking. Running
            # DINO now can produce a terminal ambiguous/unbound decision before
            # the only input that supplies the epoch-local raw ID arrives.
            upsert_identity_sync_job(
                sync_job_id=job_id,
                kind="STARTUP_RECOVERY",
                status="PENDING",
                runtime_session_id=self.runtime_session_id,
                source_epoch_id=self.source_epoch_id,
                candidate_limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
                result={
                    "frame_sequence": frame.sequence,
                    "source_stamp": frame.source_stamp.to_dict(),
                    "matches": [],
                    "reason": (
                        "waiting_for_same_stamp_tracking"
                        if tracking_state == "missing"
                        else "segments_nonempty_but_tracking_empty"
                    ),
                },
            )
            return False
        if frame.recovery_candidates and self._sample_exact(frame) is None:
            upsert_identity_sync_job(
                sync_job_id=job_id,
                kind="STARTUP_RECOVERY",
                status="PENDING",
                runtime_session_id=self.runtime_session_id,
                source_epoch_id=self.source_epoch_id,
                candidate_limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
                result={"reason": "waiting_for_exact_rgbd"},
            )
            return False
        if complete_empty_snapshot:
            upsert_identity_sync_job(
                sync_job_id=job_id,
                kind="STARTUP_RECOVERY",
                status="COMPLETED",
                runtime_session_id=self.runtime_session_id,
                source_epoch_id=self.source_epoch_id,
                candidate_limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
                result={
                    "frame_sequence": frame.sequence,
                    "source_stamp": frame.source_stamp.to_dict(),
                    "assignment": {
                        "priority_matched_count": 0,
                        "matched_count": 0,
                        "total_distance": 0.0,
                        "second_total_distance": None,
                        "assignment_margin": None,
                    },
                    "matches": [],
                    "reason": "complete_empty_recovery_snapshot",
                },
            )
            return True
        source_key = sample_key(frame.source_stamp)
        attempt_monotonic = float(frame.received_monotonic)
        if self._startup_recovery_last_source_key == source_key:
            return False
        if (
            attempt_monotonic - self._startup_recovery_last_attempt_monotonic
            < SHIGURE_STARTUP_RECOVERY_RETRY_SECONDS
        ):
            return False
        self._startup_recovery_attempts += 1
        self._startup_recovery_last_source_key = source_key
        self._startup_recovery_last_attempt_monotonic = attempt_monotonic

        upsert_identity_sync_job(
            sync_job_id=job_id,
            kind="STARTUP_RECOVERY",
            status="RUNNING",
            runtime_session_id=self.runtime_session_id,
            source_epoch_id=self.source_epoch_id,
            candidate_limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
        )
        results: list[dict[str, Any]] = []
        temporary_artifacts: list[EventArtifacts] = []
        try:
            sample = self._sample_exact(frame)
            active_display_ids = {
                str(item["display_object_id"])
                for item in list_active_shigure_bindings(self.source_epoch_id)
            }
            display_ids = [item for item in self._recent_display_ids() if item not in active_display_ids]
            candidates: list[dict[str, Any]] = []
            if sample is not None:
                for candidate in frame.recovery_candidates:
                    raw_id = str(candidate.get("shigure_object_id") or "").strip()
                    existing = (
                        get_active_shigure_binding(
                            source_epoch_id=self.source_epoch_id,
                            raw_shigure_object_id=raw_id,
                        )
                        if raw_id
                        else None
                    )
                    if existing is not None:
                        state = get_display_object_state(str(existing["display_object_id"])) or {}
                        if (
                            str(state.get("presence") or "") != "PRESENT"
                            or str(state.get("active_shigure_binding_id") or "")
                            != str(existing["binding_id"])
                        ):
                            activate_recovered_shigure_binding(str(existing["binding_id"]))
                        results.append(
                            {
                                "candidate_id": candidate.get("candidate_id"),
                                "status": "BOUND",
                                "raw_shigure_object_id": raw_id,
                                "display_object_id": str(existing["display_object_id"]),
                                "reason": "existing_epoch_binding",
                            }
                        )
                        continue
                    try:
                        artifacts = self._candidate_artifacts(frame, candidate, sample)
                        temporary_artifacts.append(artifacts)
                        embedded = self._embed_artifacts(artifacts)
                        if embedded is None:
                            raise ValueError("candidate mask is unavailable")
                        vector, response = embedded
                        scores = self._identity_scores(vector, display_ids)
                        candidates.append(
                            {
                                "candidate": candidate,
                                "artifacts": artifacts,
                                "embedding": vector,
                                "response": response,
                                "scores": scores,
                            }
                        )
                    except Exception as exc:
                        results.append({"candidate_id": candidate.get("candidate_id"), "status": "UNBOUND", "reason": str(exc)})
            assignment = _global_identity_assignment(
                [item["scores"] for item in candidates],
                candidate_priorities=[
                    1
                    if str(item["candidate"].get("shigure_object_id") or "").strip()
                    else 0
                    for item in candidates
                ],
            )
            assigned = assignment["assignments"]
            ambiguous_candidates = assignment["ambiguous_candidates"]
            assignment_summary = {
                "priority_matched_count": assignment["priority_matched_count"],
                "matched_count": assignment["matched_count"],
                "total_distance": assignment["total_distance"],
                "second_total_distance": assignment["second_total_distance"],
                "assignment_margin": assignment["assignment_margin"],
            }
            for index, item in enumerate(candidates):
                item = candidates[index]
                candidate = item["candidate"]
                raw_id = str(candidate.get("shigure_object_id") or "").strip()
                selected = assigned.get(index)
                if selected is None:
                    has_viable_edge = any(
                        float(score["distance"])
                        <= SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD
                        for score in item["scores"]
                    )
                    results.append(
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "status": "UNBOUND",
                            "reason": (
                                "global_one_to_one_capacity"
                                if has_viable_edge
                                else "distance_threshold_or_no_identity_references"
                            ),
                            "scores": item["scores"],
                        }
                    )
                    continue
                display_object_id = str(selected["display_object_id"])
                distance = float(selected["distance"])
                decision = {
                    "status": "MATCHED",
                    "display_object_id": display_object_id,
                    "distance": distance,
                    "reference_id": selected.get("reference_id"),
                    "reference_source": selected.get("reference_source"),
                    **assignment_summary,
                }
                if index in ambiguous_candidates:
                    results.append(
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "status": "AMBIGUOUS",
                            "display_object_id": display_object_id,
                            "distance": distance,
                            "reason": "global_assignment_margin",
                            **assignment_summary,
                        }
                    )
                    continue
                if not raw_id:
                    results.append(
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "status": "PROVISIONAL",
                            "display_object_id": display_object_id,
                            "distance": distance,
                            "reason": "segment_has_no_resolved_raw_id",
                            **assignment_summary,
                        }
                    )
                    continue
                try:
                    binding = establish_shigure_binding(
                        runtime_session_id=self.runtime_session_id,
                        source_epoch_id=self.source_epoch_id,
                        raw_shigure_object_id=raw_id,
                        display_object_id=display_object_id,
                        established_by="STARTUP_DINO_GLOBAL_ONE_TO_ONE",
                        confidence=max(0.0, 1.0 - distance),
                        detail=decision,
                    )
                    state = get_display_object_state(display_object_id) or {}
                    if (
                        str(state.get("presence") or "") != "PRESENT"
                        or str(state.get("active_shigure_binding_id") or "")
                        != str(binding["binding_id"])
                    ):
                        activate_recovered_shigure_binding(str(binding["binding_id"]))
                    results.append({"candidate_id": candidate.get("candidate_id"), "status": "BOUND", "raw_shigure_object_id": raw_id, "display_object_id": display_object_id, "distance": distance})
                except Exception as exc:
                    results.append({"candidate_id": candidate.get("candidate_id"), "status": "CONFLICT", "reason": str(exc)})
            recovery_complete = (
                bool(frame.recovery_candidates)
                and len(results) == len(frame.recovery_candidates)
                and all(
                    str(item.get("status") or "") == "BOUND"
                    for item in results
                )
            )
            attempts_exhausted = (
                self._startup_recovery_attempts
                >= SHIGURE_STARTUP_RECOVERY_MAX_ATTEMPTS
            )
            job_status = (
                "COMPLETED"
                if recovery_complete
                else "FAILED" if attempts_exhausted else "PENDING"
            )
            reason = (
                None
                if recovery_complete
                else (
                    "retry_limit_reached_with_unresolved_candidates"
                    if attempts_exhausted
                    else "waiting_for_retryable_identity_decision"
                )
            )
            upsert_identity_sync_job(
                sync_job_id=job_id,
                kind="STARTUP_RECOVERY",
                status=job_status,
                runtime_session_id=self.runtime_session_id,
                source_epoch_id=self.source_epoch_id,
                candidate_limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
                result={
                    "frame_sequence": frame.sequence,
                    "source_stamp": frame.source_stamp.to_dict(),
                    "assignment": assignment_summary,
                    "matches": results,
                    "attempts": self._startup_recovery_attempts,
                    "max_attempts": SHIGURE_STARTUP_RECOVERY_MAX_ATTEMPTS,
                    "reason": reason,
                },
                error_message=reason if job_status == "FAILED" else None,
            )
            return recovery_complete or attempts_exhausted
        except Exception as exc:
            attempts_exhausted = (
                self._startup_recovery_attempts
                >= SHIGURE_STARTUP_RECOVERY_MAX_ATTEMPTS
            )
            failure_reason = f"startup_recovery_attempt_failed:{exc}"
            upsert_identity_sync_job(
                sync_job_id=job_id,
                kind="STARTUP_RECOVERY",
                status="FAILED" if attempts_exhausted else "PENDING",
                runtime_session_id=self.runtime_session_id,
                source_epoch_id=self.source_epoch_id,
                candidate_limit=SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
                result={
                    "partial": results,
                    "attempts": self._startup_recovery_attempts,
                    "max_attempts": SHIGURE_STARTUP_RECOVERY_MAX_ATTEMPTS,
                    "reason": failure_reason,
                },
                error_message=failure_reason if attempts_exhausted else None,
            )
            return attempts_exhausted
        finally:
            for artifacts in temporary_artifacts:
                _cleanup_event_artifacts(artifacts)

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

        bounds = task.get("ModelBounds") if isinstance(task.get("ModelBounds"), Mapping) else {}
        try:
            minimum = np.asarray(bounds["aabb_min_aruco"], dtype=np.float64).reshape(3)
            maximum = np.asarray(bounds["aabb_max_aruco"], dtype=np.float64).reshape(3)
        except (KeyError, TypeError, ValueError):
            return fail_validation("hololens_model_bounds_unavailable")
        size = maximum - minimum
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or np.any(size <= 0.0):
            return fail_validation("hololens_model_bounds_invalid")
        job = PendingHoloSync(
            job_id=job_id,
            display_object_id=str(display_object_id),
            task_id=str(task_id),
            task_json_path=path,
            target_center_aruco=(minimum + maximum) * 0.5,
            target_size_aruco=size,
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
        # Explicit-empty Segments frames are attempts too. Otherwise a live
        # stream with no current objects leaves every Holo sync PENDING forever
        # and the configured retry limit can never be reached.
        if frame.input_states.get("segments") not in {"present", "explicit_empty"}:
            return
        # Segments normally arrives a few milliseconds before its aligned
        # depth frame. The compatibility adapter emits both the early
        # Segments revision and a later same-stamp revision when depth arrives.
        # Do not let the incomplete intermediate revision consume a Holo sync
        # attempt: at 10 Hz it can otherwise exhaust the retry limit before a
        # single complete RGB-D revision is observed. Explicit-empty frames
        # still count below because they intentionally prove that no recovery
        # candidate is currently visible and do not require image evidence.
        if frame.recovery_candidates and self._sample_exact(frame) is None:
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
        with self._lock:
            if not self._holo_sync_attempt_current(job, source_generation):
                return
            if job.last_attempt_source_key != source_key:
                job.attempts += 1
                job.last_attempt_source_key = source_key
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

        entries: list[dict[str, Any]] = []
        for candidate in frame.recovery_candidates:
            try:
                artifacts = self._candidate_artifacts(frame, candidate, sample)
                if artifacts.temporary_root is not None:
                    job.temporary_roots.append(artifacts.temporary_root)
            except Exception:
                continue
            geometry: dict[str, Any] | None = None
            tracking = candidate.get("tracking") if isinstance(candidate.get("tracking"), Mapping) else {}
            try:
                box = build_spatial_box_v2(tracking.get("collider"), camera_to_aruco, np.eye(4))
                corners = np.asarray(box["corners_aruco_m"], dtype=np.float64)
                minimum, maximum = corners.min(axis=0), corners.max(axis=0)
                center, size = (minimum + maximum) * 0.5, maximum - minimum
                center_distance = float(np.linalg.norm(center - job.target_center_aruco))
                size_log_error = float(np.max(np.abs(np.log(size / job.target_size_aruco))))
                geometry = {
                    "center_distance_m": center_distance,
                    "size_log_error": size_log_error,
                    "plausible": center_distance <= SHIGURE_HOLO_SYNC_CENTER_DISTANCE_M
                    and size_log_error <= SHIGURE_HOLO_SYNC_SIZE_LOG_TOLERANCE,
                }
            except Exception:
                geometry = None
            entries.append({"candidate": candidate, "artifacts": artifacts, "geometry": geometry})
        if not entries:
            self._reject_hololens_sync_frame(
                job,
                "no_usable_recovery_candidates",
                source_generation,
                source_key,
            )
            return

        plausible = [item for item in entries if item["geometry"] and item["geometry"]["plausible"]]
        selected: dict[str, Any] | None = None
        method = ""
        dino_detail: dict[str, Any] = {}
        if not plausible:
            self._reject_hololens_sync_frame(
                job,
                "no_candidate_within_aruco_geometry_gate",
                source_generation,
                source_key,
            )
            return

        if len(plausible) == 1:
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
            and segment_probability
            >= SHIGURE_EXAMPLE_MIN_SEGMENT_PROBABILITY
        )
        if not raw_id_is_trusted:
            self._reject_hololens_sync_frame(
                job,
                "stable_candidate_has_no_trusted_shigure_raw_id",
                source_generation,
                source_key,
            )
            return
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
                >= SHIGURE_EXAMPLE_STABLE_MASK_FRAMES
            )
        if not stable:
            self._pending_or_fail_holo_sync(
                job,
                "candidate_not_yet_stable",
                source_generation,
            )
            return

        embedded = selected.get("embedded")
        if embedded is None:
            embedded = self._embed_artifacts(selected["artifacts"])
        if embedded is None:
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
                "stable_mask_frames": SHIGURE_EXAMPLE_STABLE_MASK_FRAMES,
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

        self._maybe_startup_recovery(frame)
        for position, event in enumerate(frame.events):
            self._process_event(frame, event, position)
        self._update_tracked_geometry(frame)
        self._collect_stable_pose_candidates(frame)

        self._schedule_hololens_syncs(frame)

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
        try:
            _transform, rotation, translation, calibration_revision = _camera_to_aruco()
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
        elif binding is not None:
            resolution_status = "RESOLVED"
            resolution_method = "TRUSTED_EPOCH_BINDING"
        elif action == "take_out":
            # A take-out may only consume a binding established earlier in the
            # same source epoch. DINO is intentionally forbidden here.
            resolution_status = "UNRESOLVED"
            identity = {"reason": "takeout_has_no_epoch_binding", "dino_attempted": False}
        elif action == "bring_in":
            try:
                embedded = self._embed_artifacts(artifacts)
                if embedded is None:
                    raise ValueError("bring-in has no valid full-frame mask")
                active_displays = {str(row["display_object_id"]) for row in list_active_shigure_bindings(self.source_epoch_id)}
                display_ids = [item for item in self._recent_display_ids() if item not in active_displays]
                identity = self._select_identity(self._identity_scores(embedded[0], display_ids))
                if identity.get("status") == "MATCHED":
                    binding = establish_shigure_binding(
                        runtime_session_id=self.runtime_session_id,
                        source_epoch_id=self.source_epoch_id,
                        raw_shigure_object_id=raw_id,
                        display_object_id=str(identity["display_object_id"]),
                        established_by="BRING_IN_DINO",
                        established_event_uid=str(event.get("event_uid") or "") or None,
                        confidence=max(0.0, 1.0 - float(identity["distance"])),
                        detail=identity,
                    )
                    resolution_status = "RESOLVED"
                    resolution_method = "BRING_IN_DINO"
                else:
                    resolution_status = "AMBIGUOUS" if identity.get("status") == "AMBIGUOUS" else "UNRESOLVED"
            except Exception as exc:
                resolution_status = "UNRESOLVED"
                identity = {"status": "UNBOUND", "reason": str(exc)}
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
            binding_id=str(binding["binding_id"]) if binding is not None and resolution_status == "RESOLVED" else None,
            display_object_id=str(binding["display_object_id"]) if binding is not None and resolution_status == "RESOLVED" else None,
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
                        self._last_fp_sequence.pop(raw_id, None)
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
            self._schedule_admitted_foundationpose(
                inflight_key=inflight_key,
                raw_id=raw_id,
                display_object_id=display_object_id,
                sequence=sequence,
                source_stamp=source_stamp,
                artifacts=artifacts,
                source_generation=source_generation,
                source_epoch_id=source_epoch_id,
                binding_id=binding_id,
                admission=admission,
            )
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

    def _schedule_admitted_foundationpose(
        self,
        *,
        inflight_key: str,
        raw_id: str,
        display_object_id: str,
        sequence: int,
        source_stamp: Mapping[str, Any],
        artifacts: EventArtifacts,
        source_generation: int,
        source_epoch_id: str,
        binding_id: str,
        admission: Mapping[str, Any],
    ) -> bool:
        if (
            str(admission.get("admission_status") or "") != "MATCHED"
            or str(admission.get("admission_display_object_id") or "")
            != display_object_id
        ):
            return False
        with self._lock:
            if (
                self._source_generation != source_generation
                or str(self.source_epoch_id) != source_epoch_id
                or inflight_key in self._fp_inflight
                or self._last_fp_sequence.get(raw_id, 0) >= int(sequence)
            ):
                return False
        binding = get_active_shigure_binding(
            source_epoch_id=source_epoch_id,
            raw_shigure_object_id=raw_id,
        )
        if (
            binding is None
            or str(binding["binding_id"]) != binding_id
            or str(binding["display_object_id"]) != display_object_id
        ):
            return False
        state = get_display_object_state(display_object_id) or {}
        model_revision = int(state.get("active_model_revision") or 0)
        if model_revision <= 0 or str(state.get("presence")) != "PRESENT":
            return False
        sample = artifacts.sample
        mask = artifacts.mask_array
        if sample is None or mask is None:
            return False

        root = Path(tempfile.mkdtemp(prefix="shigure-v2-fp-admitted-"))
        fp_artifacts: EventArtifacts | None = None
        try:
            scene_path = root / "scene.png"
            mask_path = root / "mask.png"
            _write_image(scene_path, sample.rgb_bgr)
            _write_image(mask_path, mask.astype(np.uint8) * 255)
            fp_artifacts = EventArtifacts(
                scene_path,
                mask_path,
                None,
                sample,
                mask.copy(),
                root,
            )
            snapshot = self._foundationpose_snapshot(
                source_stamp, fp_artifacts, sample
            )
            with self._lock:
                if (
                    self._source_generation != source_generation
                    or str(self.source_epoch_id) != source_epoch_id
                    or inflight_key in self._fp_inflight
                ):
                    _cleanup_event_artifacts(fp_artifacts)
                    return False
                self._fp_inflight.add(inflight_key)
                self._last_fp_sequence[raw_id] = int(sequence)
            future = self._pose_executor.submit(
                self._run_foundationpose_guarded,
                inflight_key,
                dict(binding),
                model_revision,
                int(sequence),
                snapshot,
                source_generation,
                source_epoch_id,
                raw_id,
                binding_id,
            )
            future.add_done_callback(
                lambda _future, key=inflight_key, owned=fp_artifacts: (
                    self._finish_foundationpose_future(key, owned)
                )
            )
            return True
        except Exception:
            with self._lock:
                self._fp_inflight.discard(inflight_key)
            _cleanup_event_artifacts(fp_artifacts)
            if fp_artifacts is None:
                shutil.rmtree(root, ignore_errors=True)
            raise

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

    def _finish_foundationpose_future(
        self, inflight_key: str, artifacts: EventArtifacts | None
    ) -> None:
        with self._lock:
            self._fp_inflight.discard(inflight_key)
        _cleanup_event_artifacts(artifacts)

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

    def _foundationpose_attempt_current(
        self,
        *,
        source_generation: int,
        source_epoch_id: str,
        raw_id: str,
        binding_id: str,
        display_object_id: str,
    ) -> bool:
        with self._lock:
            if (
                self._source_generation != int(source_generation)
                or str(self.source_epoch_id) != str(source_epoch_id)
            ):
                return False
        active = get_active_shigure_binding(
            source_epoch_id=source_epoch_id,
            raw_shigure_object_id=raw_id,
        )
        return bool(
            active is not None
            and str(active["binding_id"]) == str(binding_id)
            and str(active["display_object_id"]) == str(display_object_id)
        )

    def _run_foundationpose_guarded(
        self,
        inflight_key: str,
        binding: Mapping[str, Any],
        model_revision: int,
        sequence: int,
        snapshot: Mapping[str, Any],
        source_generation: int,
        source_epoch_id: str,
        raw_id: str,
        binding_id: str,
    ) -> None:
        try:
            self._run_foundationpose(
                binding,
                model_revision,
                sequence,
                snapshot,
                source_generation=source_generation,
                source_epoch_id=source_epoch_id,
                raw_id=raw_id,
                binding_id=binding_id,
            )
        finally:
            with self._lock:
                self._fp_inflight.discard(inflight_key)
            temporary_root = str(snapshot.get("temporary_root") or "")
            if temporary_root:
                shutil.rmtree(temporary_root, ignore_errors=True)

    def _run_foundationpose(
        self,
        binding: Mapping[str, Any],
        model_revision: int,
        sequence: int,
        snapshot: Mapping[str, Any],
        *,
        source_generation: int,
        source_epoch_id: str,
        raw_id: str,
        binding_id: str,
    ) -> None:
        display_object_id = str(binding["display_object_id"])
        try:
            if not self._foundationpose_attempt_current(
                source_generation=source_generation,
                source_epoch_id=source_epoch_id,
                raw_id=raw_id,
                binding_id=binding_id,
                display_object_id=display_object_id,
            ):
                return
            state = get_display_object_state(display_object_id) or {}
            if int(state.get("active_model_revision") or 0) != int(model_revision) or str(state.get("presence")) != "PRESENT":
                return
            task_row = get_task_by_task_id(str(state.get("active_model_task_id") or ""))
            if not task_row:
                raise RuntimeError("active model task is missing")
            task = load_task_json(resolve_task_json_path_from_record(task_row))
            source = resolve_model_generation_source(task, require_mtl_image=False)
            scale = float((task.get("object_alignment") or {}).get("model_real_scale") or 0.0)
            if scale <= 0.0:
                raise RuntimeError("active model scale is missing")
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
            result = response.get("result") if isinstance(response.get("result"), Mapping) else {}
            pose_cv = np.asarray(result.get("pose"), dtype=np.float64)
            if pose_cv.shape != (4, 4) or not np.isfinite(pose_cv).all():
                raise RuntimeError("FoundationPose returned an invalid pose")
            mask = cv2.imread(str(snapshot["mask_file"]), cv2.IMREAD_GRAYSCALE) > 0
            depth = cv2.imread(str(snapshot["depth_file"]), cv2.IMREAD_UNCHANGED)
            quality = self._validate_foundationpose(source.mesh_path, pose_cv, scale, mask, depth, np.asarray(snapshot["k"], dtype=np.float64))
            if not quality["accepted"]:
                raise RuntimeError(f"FoundationPose quality rejected: {quality}")
            pose_aruco = self._foundationpose_to_aruco_pose(pose_cv, scale)
            if not self._foundationpose_attempt_current(
                source_generation=source_generation,
                source_epoch_id=source_epoch_id,
                raw_id=raw_id,
                binding_id=binding_id,
                display_object_id=display_object_id,
            ):
                return
            update_shigure_live_observation(
                binding_id=str(binding["binding_id"]),
                observation_seq=int(sequence),
                model_revision=int(model_revision),
                pose_aruco=pose_aruco,
            )
        except Exception as exc:
            print(f"[shigure-v2] FoundationPose update failed for {display_object_id}: {exc}")

    @staticmethod
    def _foundationpose_to_aruco_pose(pose_cv: np.ndarray, scale: float) -> dict[str, Any]:
        camera_basis = np.asarray(OPENCV_CAMERA_TO_CANONICAL_RH_BASIS, dtype=np.float64)
        model_basis = np.asarray(MODEL_INPUT_TO_CANONICAL_RH_BASIS, dtype=np.float64)
        local_rotation_rh = camera_basis @ pose_cv[:3, :3] @ model_basis.T
        local_translation_rh = camera_basis @ pose_cv[:3, 3]
        local_rotation, local_translation = model_pose_canonical_rh_to_unity_camera(local_rotation_rh, local_translation_rh)
        _transform, marker_rotation, marker_translation, _revision = _camera_to_aruco()
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
