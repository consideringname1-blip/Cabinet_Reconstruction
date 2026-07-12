from __future__ import annotations

import base64
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from artifact_layout import REALTIME_TRACKING_ROOT
from config import (
    REALTIME_TRACKING_CENTROID_DRIFT_M,
    REALTIME_TRACKING_DEPTH_MAD_MAX_M,
    REALTIME_TRACKING_DEPTH_MEDIAN_DRIFT_M,
    REALTIME_TRACKING_DEPTH_STABLE_MIN_FRAMES,
    REALTIME_TRACKING_DEPTH_STABLE_WAIT_SEC,
    REALTIME_TRACKING_DEPTH_VALID_RATIO,
    REALTIME_TRACKING_EVENT_POLL_SEC,
    REALTIME_TRACKING_FP_MAX_DEPTH_RESIDUAL_M,
    REALTIME_TRACKING_FP_MIN_BBOX_IOU,
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
from model_generation_common import resolve_model_generation_source
from object_alignment_common import read_obj_vertices
from realtime_tracking import (
    MODE_LIVE,
    PendingObservation,
    RealtimeTrackingCoordinator,
    append_tracking_journal,
)
from shigure_identity import (
    AMBIGUOUS,
    MATCHED,
    EphemeralBindingRegistry,
    BindingConflictError,
    match_display_identity,
)
from spatial_transforms import camera_matrix_from_info, rt_to_pose
from stages.shigure_history.cache import CachedRgbdSample, CachedShigureEvent, RosStamp, ShigureRgbdCache
from stages.shigure_history.marker_history import latest_marker_pose_path
from stages.taken_object_detection import settings as taken_detection_settings
from stages.taken_object_detection.run_taken_object_detection_from_json import _project_model_diag_circle_to_shigure
from task_db import (
    commit_realtime_tracking_pose,
    get_display_object_state,
    get_task_by_task_id,
    record_realtime_tracking_event,
)
from task_json import load_task_json, resolve_task_json_path_from_record


RealtimeRequest = Callable[[dict[str, Any], str], dict[str, Any]]
DinoRequest = Callable[[dict[str, Any]], dict[str, Any]]


def _depth_metres(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 1000.0
    result = arr.astype(np.float32)
    finite = result[np.isfinite(result) & (result > 0)]
    if finite.size and float(np.median(finite)) > 20.0:
        result = result / 1000.0
    return result


def _decode_mask(mask_b64: str, shape: tuple[int, int]) -> np.ndarray:
    encoded = base64.b64decode(str(mask_b64 or ""))
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Shigure object mask could not be decoded")
    if image.ndim == 3:
        image = image[:, :, 0]
    mask = image > 0
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if not np.any(mask):
        raise ValueError("Shigure object mask is empty")
    return mask


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("mask is empty")
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1.0e-9, left_area + right_area - intersection)


def _depth_stats(sample: CachedRgbdSample, mask: np.ndarray, k: np.ndarray) -> dict[str, Any] | None:
    depth = _depth_metres(sample.depth)
    if depth.shape != mask.shape:
        depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = mask & np.isfinite(depth) & (depth > 0.05) & (depth < 10.0)
    valid_count = int(np.count_nonzero(valid))
    mask_count = int(np.count_nonzero(mask))
    ratio = valid_count / max(1, mask_count)
    if ratio < REALTIME_TRACKING_DEPTH_VALID_RATIO:
        return None
    values = depth[valid]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    ys, xs = np.nonzero(valid)
    z = values.astype(np.float64)
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    centroid = np.asarray([np.median(x), np.median(y), np.median(z)], dtype=np.float64)
    return {
        "valid_ratio": ratio,
        "median_depth_m": median,
        "mad_m": mad,
        "centroid_camera_m": centroid,
    }


def _score_projected_identity_candidate(
    sample: CachedRgbdSample,
    observed_mask: np.ndarray,
    projected_circle_mask: np.ndarray,
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    if projected_circle_mask.shape != observed_mask.shape:
        projected_circle_mask = cv2.resize(
            projected_circle_mask.astype(np.uint8),
            (observed_mask.shape[1], observed_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    mask_pixels = int(np.count_nonzero(observed_mask))
    inside_pixels = int(np.count_nonzero(observed_mask & projected_circle_mask))
    inside_ratio = inside_pixels / max(1, mask_pixels)
    depth = _depth_metres(sample.depth)
    if depth.shape != observed_mask.shape:
        depth = cv2.resize(depth, (observed_mask.shape[1], observed_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid_depth = depth[observed_mask & np.isfinite(depth) & (depth > 0.05) & (depth < 10.0)]
    observed_depth = float(np.median(valid_depth)) if valid_depth.size else None
    center_depth = projection.get("center_depth_m")
    depth_diff = None
    if observed_depth is not None and center_depth is not None and np.isfinite(float(center_depth)):
        depth_diff = abs(observed_depth - float(center_depth))
    reject_reasons: list[str] = []
    if inside_ratio < float(taken_detection_settings.MODEL_DIAG_CIRCLE_MIN_MASK_INSIDE_RATIO):
        reject_reasons.append("mask_not_enough_inside_model_diag_circle")
    if depth_diff is None:
        reject_reasons.append("depth_unavailable")
    elif depth_diff > float(taken_detection_settings.MODEL_DIAG_CIRCLE_MAX_DEPTH_DIFF_M):
        reject_reasons.append("depth_too_different_from_model_center")
    return {
        "accepted": not reject_reasons,
        "reject_reasons": reject_reasons,
        "mask_inside_diag_circle_pixels": inside_pixels,
        "mask_inside_diag_circle_ratio": inside_ratio,
        "min_mask_inside_diag_circle_ratio": float(taken_detection_settings.MODEL_DIAG_CIRCLE_MIN_MASK_INSIDE_RATIO),
        "observed_median_depth_m": observed_depth,
        "center_depth_m": float(center_depth) if center_depth is not None else None,
        "depth_diff_m": depth_diff,
        "max_depth_diff_m": float(taken_detection_settings.MODEL_DIAG_CIRCLE_MAX_DEPTH_DIFF_M),
    }


def _task_identity_embedding(task: Mapping[str, Any]) -> list[float] | None:
    historical = task.get("HistoricalModelMatch") if isinstance(task.get("HistoricalModelMatch"), Mapping) else {}
    current = historical.get("current_dinov2") if isinstance(historical.get("current_dinov2"), Mapping) else {}
    embedding = current.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        return None
    try:
        values = [float(value) for value in embedding]
    except (TypeError, ValueError):
        return None
    return values if values and all(math.isfinite(value) for value in values) else None


def _stable_window(samples: list[CachedRgbdSample], mask: np.ndarray, k: np.ndarray) -> tuple[CachedRgbdSample | None, dict[str, Any]]:
    required = max(2, int(REALTIME_TRACKING_DEPTH_STABLE_MIN_FRAMES))
    stats: list[tuple[CachedRgbdSample, dict[str, Any]]] = []
    for sample in samples[-max(required * 2, required) :]:
        value = _depth_stats(sample, mask, k)
        if value is not None:
            stats.append((sample, value))
    if len(stats) < required:
        return None, {"reason": "not_enough_valid_depth_frames", "valid_frames": len(stats), "required": required}
    selected = stats[-required:]
    medians = np.asarray([item[1]["median_depth_m"] for item in selected], dtype=np.float64)
    mads = np.asarray([item[1]["mad_m"] for item in selected], dtype=np.float64)
    centroids = np.asarray([item[1]["centroid_camera_m"] for item in selected], dtype=np.float64)
    median_drift = float(medians.max() - medians.min())
    centroid_drift = float(np.max(np.linalg.norm(centroids - centroids[-1], axis=1)))
    detail = {
        "reason": "stable",
        "frame_count": required,
        "median_depth_drift_m": median_drift,
        "max_depth_mad_m": float(mads.max()),
        "centroid_drift_m": centroid_drift,
        "valid_ratios": [float(item[1]["valid_ratio"]) for item in selected],
    }
    if median_drift > REALTIME_TRACKING_DEPTH_MEDIAN_DRIFT_M:
        detail["reason"] = "median_depth_moving"
        return None, detail
    if float(mads.max()) > REALTIME_TRACKING_DEPTH_MAD_MAX_M:
        detail["reason"] = "depth_region_not_compact"
        return None, detail
    if centroid_drift > REALTIME_TRACKING_CENTROID_DRIFT_M:
        detail["reason"] = "centroid_moving"
        return None, detail
    return selected[-1][0], detail


def _marker_pose_cv() -> tuple[np.ndarray, np.ndarray, Path]:
    path = latest_marker_pose_path()
    if path is None:
        raise FileNotFoundError("Shigure ArUco marker pose is unavailable")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    pose = payload.get("opencv_camera_pose") if isinstance(payload.get("opencv_camera_pose"), dict) else payload
    rotation = np.asarray(pose.get("rotation_matrix"), dtype=np.float64).reshape(3, 3)
    translation = np.asarray(pose.get("tvec_m") or pose.get("position"), dtype=np.float64).reshape(3)
    return orthonormalize_rotation(rotation), translation, path


def _foundationpose_to_aruco_pose(pose_cv: np.ndarray, scale: float) -> dict[str, Any]:
    camera_basis = np.asarray(OPENCV_CAMERA_TO_CANONICAL_RH_BASIS, dtype=np.float64)
    model_basis = np.asarray(MODEL_INPUT_TO_CANONICAL_RH_BASIS, dtype=np.float64)
    local_rotation_rh = camera_basis @ pose_cv[:3, :3] @ model_basis.T
    local_translation_rh = camera_basis @ pose_cv[:3, 3]
    local_rotation, local_translation = model_pose_canonical_rh_to_unity_camera(
        local_rotation_rh,
        local_translation_rh,
    )

    marker_rotation, marker_translation, _ = _marker_pose_cv()
    basis = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
    aruco_from_camera_rotation = orthonormalize_rotation(basis @ marker_rotation.T @ basis)
    camera_origin_aruco = basis @ (marker_rotation.T @ (-marker_translation))
    runtime_correction = (
        np.asarray(FBX_RUNTIME_TRANSFORM_COMPENSATION_TO_UNITY, dtype=np.float64)
        @ np.asarray(RUNTIME_LOCAL_TO_UNITY_POSE_ROTATION, dtype=np.float64)
    )
    world_position = aruco_from_camera_rotation @ np.asarray(local_translation, dtype=np.float64) + camera_origin_aruco
    world_rotation = orthonormalize_rotation(
        aruco_from_camera_rotation @ np.asarray(local_rotation, dtype=np.float64) @ runtime_correction
    )
    return rt_to_pose(world_rotation, world_position, scale=[float(scale)] * 3)


def _validate_foundationpose(
    *, mesh_file: Path, pose_cv: np.ndarray, scale: float, mask: np.ndarray, depth: np.ndarray, k: np.ndarray
) -> dict[str, Any]:
    vertices = read_obj_vertices(mesh_file)
    if len(vertices) > 50000:
        vertices = vertices[np.linspace(0, len(vertices) - 1, 50000).astype(np.int64)]
    transformed = (pose_cv[:3, :3] @ (vertices.astype(np.float64) * float(scale)).T).T + pose_cv[:3, 3]
    visible = transformed[:, 2] > 1.0e-5
    transformed = transformed[visible]
    if not len(transformed):
        return {"accepted": False, "reason": "model_behind_camera"}
    pixels = np.column_stack(
        (
            k[0, 0] * transformed[:, 0] / transformed[:, 2] + k[0, 2],
            k[1, 1] * transformed[:, 1] / transformed[:, 2] + k[1, 2],
        )
    )
    finite = np.isfinite(pixels).all(axis=1)
    pixels = pixels[finite]
    transformed = transformed[finite]
    if not len(pixels):
        return {"accepted": False, "reason": "projection_empty"}
    projected_bbox = (
        float(pixels[:, 0].min()),
        float(pixels[:, 1].min()),
        float(pixels[:, 0].max() + 1),
        float(pixels[:, 1].max() + 1),
    )
    observed_bbox = _mask_bbox(mask)
    bbox_iou = _bbox_iou(projected_bbox, observed_bbox)
    depth_m = _depth_metres(depth)
    if depth_m.shape != mask.shape:
        depth_m = cv2.resize(depth_m, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    observed = depth_m[mask & np.isfinite(depth_m) & (depth_m > 0.05)]
    observed_median = float(np.median(observed)) if observed.size else math.inf
    predicted_median = float(np.median(transformed[:, 2]))
    depth_residual = abs(predicted_median - observed_median)
    accepted = bbox_iou >= REALTIME_TRACKING_FP_MIN_BBOX_IOU and depth_residual <= REALTIME_TRACKING_FP_MAX_DEPTH_RESIDUAL_M
    return {
        "accepted": bool(accepted),
        "reason": "quality_gate_passed" if accepted else "quality_gate_failed",
        "bbox_iou": bbox_iou,
        "depth_residual_m": depth_residual,
        "projected_bbox": list(projected_bbox),
        "observed_bbox": list(observed_bbox),
    }


class ShigureRealtimeTrackingEngine:
    def __init__(
        self,
        *,
        coordinator: RealtimeTrackingCoordinator,
        dino_request: DinoRequest,
        foundationpose_request: RealtimeRequest,
    ) -> None:
        self.coordinator = coordinator
        self.dino_request = dino_request
        self.foundationpose_request = foundationpose_request
        self.cache = ShigureRgbdCache()
        self.bindings = EphemeralBindingRegistry(ingress_session_uuid=coordinator.ingress_session_id)
        self._stop = threading.Event()
        self._event_thread: threading.Thread | None = None
        self._fp_thread: threading.Thread | None = None
        self._last_event_sequence = 0
        self._last_startup_session_id = ""
        self._last_cache_created_at = ""
        self._known_tracking_epochs: dict[str, int] = {}

    def start(self) -> None:
        if self._event_thread is not None and self._event_thread.is_alive():
            return
        self._stop.clear()
        self._event_thread = threading.Thread(target=self._event_loop, daemon=True, name="shigure-realtime-events")
        self._fp_thread = threading.Thread(target=self._foundationpose_loop, daemon=True, name="shigure-realtime-foundationpose")
        self._event_thread.start()
        self._fp_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in (self._event_thread, self._fp_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=3.0)
        self._event_thread = None
        self._fp_thread = None
        self.bindings.reset(ingress_session_uuid=self.coordinator.ingress_session_id)
        self._known_tracking_epochs.clear()

    def _session(self) -> tuple[str, str]:
        status = self.coordinator.mode_status()
        startup = str(status.get("startup_session_id") or "").strip()
        ingress = str(status.get("ingress_session_id") or "").strip()
        if startup != self._last_startup_session_id or ingress != self.bindings.ingress_session_uuid:
            self.bindings.reset(ingress_session_uuid=ingress or self.coordinator.ingress_session_id)
            self._known_tracking_epochs.clear()
            self._last_startup_session_id = startup
        current_epochs = {
            str(display_object_id): int(value.get("tracking_epoch") or 0)
            for display_object_id, value in (self.coordinator.snapshot().get("objects") or {}).items()
            if isinstance(value, dict)
        }
        for display_object_id, old_epoch in self._known_tracking_epochs.items():
            new_epoch = current_epochs.get(display_object_id)
            if new_epoch is not None and new_epoch != old_epoch and startup:
                self.bindings.release_display_object(
                    startup,
                    display_object_id,
                    reason="coordinator_tracking_epoch_changed",
                )
        self._known_tracking_epochs = current_epochs
        return startup, ingress

    def _event_loop(self) -> None:
        while not self._stop.is_set():
            try:
                startup, _ingress = self._session()
                mode = self.coordinator.mode_status(startup).get("mode") if startup else None
                if not startup or mode != MODE_LIVE:
                    time.sleep(REALTIME_TRACKING_EVENT_POLL_SEC)
                    continue
                cache_status = self.cache.status() or {}
                cache_created_at = str(cache_status.get("created_at") or "")
                if self._last_cache_created_at and cache_created_at and cache_created_at != self._last_cache_created_at:
                    self.bindings.reset(ingress_session_uuid=self.coordinator.ingress_session_id)
                    self._known_tracking_epochs.clear()
                    self._last_event_sequence = 0
                if cache_created_at:
                    self._last_cache_created_at = cache_created_at
                updates = list(self.cache.iter_event_updates_after(self._last_event_sequence, include_masks=True))
                if not updates and self._last_event_sequence > 0:
                    if int(cache_status.get("latest_event_sequence") or 0) < self._last_event_sequence:
                        # The recorder sidecar restarted and its process-local
                        # update sequence began at one again.
                        self._last_event_sequence = 0
                        updates = list(self.cache.iter_event_updates_after(0, include_masks=True))
                max_sequence = max((int(event.sequence) for event in updates), default=self._last_event_sequence)
                for event in self._coalesce_event_updates(updates):
                    self._handle_event(startup, event)
                self._last_event_sequence = max(self._last_event_sequence, max_sequence)
            except Exception as exc:
                print(f"[realtime-tracking] event loop error: {exc}")
            time.sleep(max(0.05, float(REALTIME_TRACKING_EVENT_POLL_SEC)))

    @staticmethod
    def _coalesce_event_updates(updates: list[CachedShigureEvent]) -> list[CachedShigureEvent]:
        """Keep all take-out evidence and only the newest queued pose event per Shigure id."""

        passthrough: dict[int, CachedShigureEvent] = {}
        newest_pose_by_object: dict[str, CachedShigureEvent] = {}
        for event in updates:
            if event.contacted_state != "present":
                # Explicit-empty events carry the strict WRONG semantic. A
                # missing side is provisional and a later sequence will update
                # the same exact source stamp.
                if event.contacted_state == "explicit_empty":
                    passthrough[int(event.sequence)] = event
                continue
            contacts = (event.contacted or {}).get("contacts") or []
            retained = False
            for contact in contacts:
                if not isinstance(contact, dict):
                    continue
                action = str(contact.get("action") or "").strip().lower()
                object_id = str(contact.get("object_id") or "").strip()
                if action == "take_out":
                    passthrough[int(event.sequence)] = event
                    retained = True
                elif action in {"obj_move", "bring_in"} and object_id:
                    newest_pose_by_object[object_id] = event
                    retained = True
            if not retained and event.object_detection_state == "missing":
                continue
        selected = {int(event.sequence): event for event in newest_pose_by_object.values()}
        selected.update(passthrough)
        return [selected[key] for key in sorted(selected)]

    def _geometry_identity_candidates(
        self,
        *,
        sample: CachedRgbdSample,
        mask: np.ndarray,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Filter the five active objects using their latest HoloLens projection."""

        accepted: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for display_object_id in self.coordinator.active_display_object_ids()[-5:]:
            state = get_display_object_state(display_object_id) or {}
            task_id = str(state.get("latest_hololens_task_id") or "").strip()
            diagnostic: dict[str, Any] = {
                "display_object_id": display_object_id,
                "latest_hololens_task_id": task_id or None,
            }
            if not task_id:
                diagnostic.update({"accepted": False, "reason": "latest_hololens_task_missing"})
                diagnostics.append(diagnostic)
                continue
            task_row = get_task_by_task_id(task_id)
            if not task_row:
                diagnostic.update({"accepted": False, "reason": "latest_hololens_task_record_missing"})
                diagnostics.append(diagnostic)
                continue
            try:
                task = load_task_json(resolve_task_json_path_from_record(task_row))
                circle_mask, projection = _project_model_diag_circle_to_shigure(
                    task,
                    sample.camera_info,
                    mask.shape,
                )
            except Exception as exc:
                diagnostic.update({"accepted": False, "reason": "projection_failed", "error": str(exc)})
                diagnostics.append(diagnostic)
                continue
            diagnostic["projection"] = projection
            if circle_mask is None:
                diagnostic.update({"accepted": False, "reason": str(projection.get("reason") or "projection_unavailable")})
                diagnostics.append(diagnostic)
                continue
            score = _score_projected_identity_candidate(sample, mask, circle_mask, projection)
            diagnostic.update(score)
            diagnostics.append(diagnostic)
            if not score.get("accepted"):
                continue
            candidate: dict[str, Any] = {
                "display_object_id": display_object_id,
                "reference_id": task_id,
                "geometry_score": float(score.get("mask_inside_diag_circle_ratio") or 0.0),
                "geometry": diagnostic,
            }
            embedding = _task_identity_embedding(task)
            if embedding is not None:
                candidate["embedding"] = embedding
            accepted.append(candidate)
        return accepted, diagnostics

    def _event_detection(self, event: CachedShigureEvent, detection_id: str | None) -> dict[str, Any] | None:
        objects = ((event.object_detection or {}).get("objects") or []) if event.object_detection else []
        if detection_id:
            for item in objects:
                if isinstance(item, dict) and str(item.get("object_id") or "") == detection_id:
                    return item
        return next((item for item in objects if isinstance(item, dict)), None)

    def _identity_for_detection(
        self,
        *,
        startup: str,
        event: CachedShigureEvent,
        detection: dict[str, Any],
        shigure_object_id: str,
    ) -> tuple[str | None, dict[str, Any], CachedRgbdSample | None, np.ndarray | None]:
        sample = self.cache.get_sample(event.source_stamp, mode="nearest")
        if sample is None or sample.rgb_bgr.size == 0:
            return None, {"status": "UNBOUND", "reason": "rgbd_sample_missing"}, None, None
        mask = _decode_mask(str(detection.get("mask_b64") or ""), sample.rgb_bgr.shape[:2])
        existing = self.bindings.get(startup, shigure_object_id) if shigure_object_id else None
        active_ids = set(self.coordinator.active_display_object_ids())
        if existing is not None and existing.status == MATCHED and existing.display_object_id in active_ids:
            return (
                existing.display_object_id,
                {
                    "status": MATCHED,
                    "display_object_id": existing.display_object_id,
                    "reason": "trusted_current_shigure_session_binding",
                    "binding_epoch": int(existing.epoch),
                },
                sample,
                mask,
            )
        geometry_candidates, geometry_diagnostics = self._geometry_identity_candidates(sample=sample, mask=mask)
        if not geometry_candidates:
            result = {
                "status": "UNBOUND",
                "display_object_id": None,
                "reason": "no_projected_circle_depth_candidate",
                "geometry_candidates": geometry_diagnostics,
            }
        elif len(geometry_candidates) == 1:
            only = geometry_candidates[0]
            result = {
                "status": MATCHED,
                "display_object_id": str(only["display_object_id"]),
                "reason": "single_projected_circle_depth_candidate",
                "selected_reference_id": only.get("reference_id"),
                "candidate_scores": [only],
                "geometry_candidates": geometry_diagnostics,
                "dino_skipped": True,
            }
        else:
            root = REALTIME_TRACKING_ROOT / "identity" / f"event_{int(event.sequence):010d}"
            root.mkdir(parents=True, exist_ok=True)
            color_path = root / "color.png"
            mask_path = root / "mask.png"
            cv2.imwrite(str(color_path), sample.rgb_bgr)
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            response = self.dino_request(
                {"action": "embed_files", "color_file": str(color_path), "mask_file": str(mask_path)}
            )
            result = match_display_identity(response.get("embedding") or [], geometry_candidates)
            result["geometry_candidates"] = geometry_diagnostics
            result["dino_skipped"] = False
        if result.get("status") == MATCHED:
            display_object_id = str(result.get("display_object_id") or "")
            if not shigure_object_id:
                return display_object_id, result, sample, mask
            try:
                previous_for_display = self.bindings.get_by_display_object(startup, display_object_id)
                if (
                    previous_for_display is not None
                    and previous_for_display.key.shigure_object_id != shigure_object_id
                ):
                    self.bindings.release_display_object(
                        startup,
                        display_object_id,
                        reason="superseded_by_strong_identity_match",
                    )
                    result["replaced_shigure_object_id"] = previous_for_display.key.shigure_object_id
                self.bindings.bind(
                    startup,
                    shigure_object_id,
                    display_object_id,
                    reason=str(result.get("reason") or "identity_event_revalidated"),
                    allow_rebind=bool(existing is not None and existing.status == MATCHED),
                )
            except BindingConflictError:
                result = {**result, "status": AMBIGUOUS, "reason": "one_to_one_binding_conflict"}
                self.bindings.mark_ambiguous(
                    startup,
                    shigure_object_id,
                    candidate_display_object_ids=[display_object_id],
                    reason="one_to_one_binding_conflict",
                )
                return None, result, sample, mask
            return display_object_id, result, sample, mask
        if result.get("status") == AMBIGUOUS and shigure_object_id:
            self.bindings.mark_ambiguous(
                startup,
                shigure_object_id,
                candidate_display_object_ids=[item.get("display_object_id") for item in result.get("candidate_scores") or []],
                reason=str(result.get("reason") or "ambiguous"),
            )
        elif shigure_object_id:
            self.bindings.mark_unbound(startup, shigure_object_id, reason=str(result.get("reason") or "unbound"))
        return None, result, sample, mask

    def _wait_stable_sample(
        self, event: CachedShigureEvent, mask: np.ndarray, k: np.ndarray
    ) -> tuple[CachedRgbdSample | None, dict[str, Any]]:
        deadline = time.monotonic() + max(0.1, float(REALTIME_TRACKING_DEPTH_STABLE_WAIT_SEC))
        detail: dict[str, Any] = {"reason": "no_depth_frames"}
        while not self._stop.is_set() and time.monotonic() < deadline:
            if self.coordinator.mode_status().get("mode") != MODE_LIVE:
                return None, {"reason": "tracking_paused"}
            samples = list(self.cache.iter_samples(start=event.source_stamp))
            stable, detail = _stable_window(samples, mask, k)
            if stable is not None:
                return stable, detail
            time.sleep(0.2)
        return None, detail

    def _freeze_snapshot(
        self,
        *,
        event: CachedShigureEvent,
        sample: CachedRgbdSample,
        mask: np.ndarray,
        k: np.ndarray,
        display_object_id: str,
        shigure_object_id: str,
    ) -> dict[str, Any]:
        safe_display = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in display_object_id)
        root = REALTIME_TRACKING_ROOT / safe_display / "snapshots" / f"event_{int(event.sequence):010d}"
        root.mkdir(parents=True, exist_ok=True)
        color_path = root / "color.png"
        depth_path = root / "depth.png"
        mask_path = root / "mask.png"
        meta_path = root / "snapshot.json"
        cv2.imwrite(str(color_path), sample.rgb_bgr)
        depth = np.asarray(sample.depth)
        if np.issubdtype(depth.dtype, np.floating):
            depth = np.clip(np.rint(_depth_metres(depth) * 1000.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_path), depth)
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        meta = {
            "display_object_id": display_object_id,
            "shigure_object_id": shigure_object_id,
            "source_stamp": event.source_stamp.to_dict(),
            "event_sequence": int(event.sequence),
            "k": k.astype(float).tolist(),
            "color_file": str(color_path),
            "depth_file": str(depth_path),
            "mask_file": str(mask_path),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {**meta, "meta_file": str(meta_path), "event": event.to_dict(include_masks=False)}

    def _handle_event(self, startup: str, event: CachedShigureEvent) -> None:
        if self.coordinator.mode_status(startup).get("mode") != MODE_LIVE:
            return
        if event.contacted_state == "missing" or event.object_detection_state == "missing":
            return
        if event.contacted_state == "explicit_empty":
            relevant = [
                item
                for item in ((event.object_detection or {}).get("objects") or [])
                if isinstance(item, dict)
                and str(item.get("action") or "").strip().lower() in {"take_out", "bring_in", "obj_move"}
            ]
            recorded = False
            for detection in relevant:
                display_object_id, identity, _sample, _mask = self._identity_for_detection(
                    startup=startup,
                    event=event,
                    detection=detection,
                    shigure_object_id="",
                )
                if not display_object_id:
                    continue
                record_realtime_tracking_event(
                    status="WRONG_NO_CONTACTED_PERSON",
                    display_object_id=display_object_id,
                    startup_session_id=startup,
                    ingress_session_id=self.bindings.ingress_session_uuid,
                    model_revision=int((get_display_object_state(display_object_id) or {}).get("active_model_revision") or 0),
                    source_stamp=event.source_stamp.to_dict(),
                    reason="explicit_empty_contacted_list",
                    detail={
                        "identity": identity,
                        "detection": {key: value for key, value in detection.items() if key != "mask_b64"},
                    },
                )
                recorded = True
            if not recorded:
                record_realtime_tracking_event(
                    status="WRONG_NO_CONTACTED_PERSON",
                    startup_session_id=startup,
                    ingress_session_id=self.bindings.ingress_session_uuid,
                    source_stamp=event.source_stamp.to_dict(),
                    reason="explicit_empty_contacted_list_identity_unbound",
                    detail=event.to_dict(include_masks=False),
                )
            return
        contacts = (event.contacted or {}).get("contacts") or []
        matches = event.contact_object_matches or []
        for match in matches:
            contact_index = int(match.get("contact_index") or 0)
            if not (0 <= contact_index < len(contacts)):
                continue
            contact = contacts[contact_index]
            action = str(contact.get("action") or "").strip().lower()
            shigure_object_id = str(contact.get("object_id") or "").strip()
            if str(match.get("status") or "") != "matched_action_iou":
                record_realtime_tracking_event(
                    status="WRONG_AMBIGUOUS_OBJECT",
                    startup_session_id=startup,
                    ingress_session_id=self.bindings.ingress_session_uuid,
                    shigure_object_id=shigure_object_id or None,
                    source_stamp=event.source_stamp.to_dict(),
                    reason="contact_detection_action_bbox_mismatch",
                    detail={"match": match, "contact": contact},
                )
                continue
            best = match.get("best") if isinstance(match.get("best"), dict) else {}
            detection = self._event_detection(event, str(best.get("object_detection_id") or ""))
            if not shigure_object_id or detection is None:
                continue
            display_object_id, identity, sample, mask = self._identity_for_detection(
                startup=startup,
                event=event,
                detection=detection,
                shigure_object_id=shigure_object_id,
            )
            if not display_object_id or sample is None or mask is None:
                record_realtime_tracking_event(
                    status="WRONG_AMBIGUOUS_OBJECT" if identity.get("status") == AMBIGUOUS else "UNBOUND",
                    startup_session_id=startup,
                    ingress_session_id=self.bindings.ingress_session_uuid,
                    shigure_object_id=shigure_object_id,
                    source_stamp=event.source_stamp.to_dict(),
                    reason=str(identity.get("reason") or "identity_unbound"),
                    detail=identity,
                )
                continue
            state = get_display_object_state(display_object_id)
            if state is None:
                continue
            if action == "take_out":
                record_realtime_tracking_event(
                    status="TAKEN",
                    display_object_id=display_object_id,
                    startup_session_id=startup,
                    ingress_session_id=self.bindings.ingress_session_uuid,
                    shigure_object_id=shigure_object_id,
                    model_revision=int(state.get("active_model_revision") or 0),
                    source_stamp=event.source_stamp.to_dict(),
                    detail={"identity": identity, "contact": contact},
                )
                continue
            if action not in {"obj_move", "bring_in"}:
                continue
            k = camera_matrix_from_info(sample.camera_info)
            if k is None:
                continue
            stable_sample, stability = self._wait_stable_sample(event, mask, k)
            if stable_sample is None:
                record_realtime_tracking_event(
                    status="FAILED_UNSTABLE",
                    display_object_id=display_object_id,
                    startup_session_id=startup,
                    ingress_session_id=self.bindings.ingress_session_uuid,
                    shigure_object_id=shigure_object_id,
                    model_revision=int(state.get("active_model_revision") or 0),
                    source_stamp=event.source_stamp.to_dict(),
                    reason=str(stability.get("reason") or "depth_unstable"),
                    detail=stability,
                )
                continue
            snapshot = self._freeze_snapshot(
                event=event,
                sample=stable_sample,
                mask=mask,
                k=k,
                display_object_id=display_object_id,
                shigure_object_id=shigure_object_id,
            )
            snapshot["identity"] = identity
            snapshot["stability"] = stability
            snapshot["contact"] = contact
            self.coordinator.submit_observation(
                startup_session_id=startup,
                display_object_id=display_object_id,
                model_revision=int(state.get("active_model_revision") or 0),
                hololens_pose_revision=int(state.get("latest_hololens_pose_revision") or 0),
                payload=snapshot,
            )

    def _foundationpose_loop(self) -> None:
        while not self._stop.is_set():
            observation = self.coordinator.take_next_pending(timeout=0.5)
            if observation is None:
                continue
            self._run_foundationpose(observation)

    def _run_foundationpose(self, observation: PendingObservation) -> None:
        token = observation.token
        payload = observation.payload
        accepted = False
        result_detail: dict[str, Any] = {}
        pose_aruco: dict[str, Any] | None = None
        try:
            state = get_display_object_state(token.display_object_id)
            if state is None or int(state.get("active_model_revision") or 0) != token.model_revision:
                raise RuntimeError("STALE_MODEL_REVISION")
            if int(state.get("latest_hololens_pose_revision") or 0) != token.hololens_pose_revision:
                raise RuntimeError("STALE_HOLOLENS_POSE_REVISION")
            model_task_id = str(state.get("active_model_task_id") or "").strip()
            task_row = get_task_by_task_id(model_task_id)
            if not task_row:
                raise RuntimeError("ACTIVE_MODEL_TASK_MISSING")
            task = load_task_json(resolve_task_json_path_from_record(task_row))
            model_source = resolve_model_generation_source(task, require_mtl_image=False)
            scale = float((task.get("object_alignment") or {}).get("model_real_scale") or 0.0)
            if scale <= 0:
                raise RuntimeError("ACTIVE_MODEL_SCALE_MISSING")
            request_payload = {
                "mesh_file": str(model_source.mesh_path),
                "color_file": str(payload["color_file"]),
                "depth_file": str(payload["depth_file"]),
                "mask_file": str(payload["mask_file"]),
                "k": payload["k"],
                "model_scale": scale,
                "iteration": 5,
            }
            response = self.foundationpose_request(request_payload, token.display_object_id)
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or "FoundationPose failed"))
            result = response.get("result") if isinstance(response.get("result"), dict) else {}
            pose_cv = np.asarray(result.get("pose"), dtype=np.float64)
            if pose_cv.shape != (4, 4) or not np.isfinite(pose_cv).all():
                raise RuntimeError("FOUNDATIONPOSE_INVALID_POSE")
            mask = cv2.imread(str(payload["mask_file"]), cv2.IMREAD_GRAYSCALE) > 0
            depth = cv2.imread(str(payload["depth_file"]), cv2.IMREAD_UNCHANGED)
            k = np.asarray(payload["k"], dtype=np.float64).reshape(3, 3)
            quality = _validate_foundationpose(
                mesh_file=model_source.mesh_path,
                pose_cv=pose_cv,
                scale=scale,
                mask=mask,
                depth=depth,
                k=k,
            )
            result_detail = {"foundationpose": result, "quality": quality, "snapshot": payload.get("meta_file")}
            if not quality.get("accepted"):
                raise RuntimeError("FOUNDATIONPOSE_QUALITY_REJECTED")
            pose_aruco = _foundationpose_to_aruco_pose(pose_cv, scale)
            latest_state = get_display_object_state(token.display_object_id) or {}
            latest_revision = int(latest_state.get("active_model_revision") or 0)
            latest_hololens_pose_revision = int(latest_state.get("latest_hololens_pose_revision") or 0)
            committed = self.coordinator.commit_if_current(
                token,
                active_model_revision=latest_revision,
                active_hololens_pose_revision=latest_hololens_pose_revision,
                commit=lambda: commit_realtime_tracking_pose(
                    display_object_id=token.display_object_id,
                    model_revision=token.model_revision,
                    hololens_pose_revision=token.hololens_pose_revision,
                    observation_seq=token.observation_seq,
                    pose_aruco=pose_aruco,
                ),
            )
            if committed is None:
                refreshed = get_display_object_state(token.display_object_id) or {}
                if int(refreshed.get("active_model_revision") or 0) != token.model_revision:
                    raise RuntimeError("STALE_MODEL_REVISION")
                if int(refreshed.get("latest_hololens_pose_revision") or 0) != token.hololens_pose_revision:
                    raise RuntimeError("STALE_HOLOLENS_POSE_REVISION")
                raise RuntimeError("SUPERSEDED")
            accepted = True
            status = "ACCEPTED"
            reason = "foundationpose_quality_gate_passed"
        except Exception as exc:
            reason = str(exc)
            status = (
                "SUPERSEDED"
                if reason in {"SUPERSEDED", "STALE_MODEL_REVISION", "STALE_HOLOLENS_POSE_REVISION"}
                else "FP_FAILED"
            )
        event_payload = {
            "display_object_id": token.display_object_id,
            "observation_seq": token.observation_seq,
            "tracking_epoch": token.tracking_epoch,
            "mode_epoch": token.mode_epoch,
            "model_revision": token.model_revision,
            "hololens_pose_revision": token.hololens_pose_revision,
            "status": status,
            "reason": reason,
            "pose_aruco": pose_aruco,
            "detail": result_detail,
            "created_at_unix": time.time(),
        }
        append_tracking_journal(token.display_object_id, event_payload)
        record_realtime_tracking_event(
            status=status,
            display_object_id=token.display_object_id,
            startup_session_id=token.startup_session_id,
            ingress_session_id=token.ingress_session_id,
            observation_seq=token.observation_seq,
            tracking_epoch=token.tracking_epoch,
            mode_epoch=token.mode_epoch,
            model_revision=token.model_revision,
            reason=reason,
            source_stamp=payload.get("source_stamp"),
            pose_aruco=pose_aruco,
            detail=result_detail,
        )
        self.coordinator.finish(token, accepted=accepted)
