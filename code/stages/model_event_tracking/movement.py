from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import settings
from .schemas import HandContact, MaskDepthSignature, MovementDecision, RosStamp


@dataclass(frozen=True)
class MovementThresholds:
    movement_threshold_m: float = settings.MOVEMENT_THRESHOLD_M
    depth_stable_tolerance_m: float = settings.DEPTH_STABLE_TOLERANCE_M
    min_depth_points: int = settings.MIN_DEPTH_POINTS
    min_overlap_pixels: int = settings.MIN_OVERLAP_PIXELS
    min_visible_area_ratio: float = settings.MIN_VISIBLE_AREA_RATIO
    complete_area_ratio: float = settings.COMPLETE_AREA_RATIO
    complete_iou_min: float = settings.COMPLETE_IOU_MIN
    taken_away_stable_frames: int = settings.TAKEN_AWAY_STABLE_FRAMES


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    while array.ndim > 2 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got {array.shape}")
    return array > 0


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    array = np.asarray(depth)
    while array.ndim > 2 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"depth must be 2D, got {array.shape}")
    out = array.astype(np.float64)
    finite = out[np.isfinite(out)]
    if finite.size and float(np.nanmedian(finite)) > 20.0:
        out *= 0.001
    out[out <= 0.0] = np.nan
    return out


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _points_from_depth_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    z: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=1)


def signature_from_mask_depth(
    mask: np.ndarray,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    min_depth_points: int = settings.MIN_DEPTH_POINTS,
) -> MaskDepthSignature:
    mask_bool = normalize_mask(mask)
    depth = depth_to_meters(depth_m)
    if mask_bool.shape != depth.shape:
        raise ValueError(f"mask/depth shape mismatch: {mask_bool.shape} vs {depth.shape}")

    area_px = int(np.count_nonzero(mask_bool))
    valid = mask_bool & np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.nonzero(valid)
    if xs.size < int(min_depth_points):
        return MaskDepthSignature(
            area_px=area_px,
            bbox_xyxy=_bbox_from_mask(mask_bool),
            center_camera_m=None,
            median_depth_m=None,
            valid_depth_points=int(xs.size),
        )

    z = depth[ys, xs]
    points = _points_from_depth_pixels(xs, ys, z, camera_matrix)
    center = np.nanmedian(points, axis=0)
    return MaskDepthSignature(
        area_px=area_px,
        bbox_xyxy=_bbox_from_mask(mask_bool),
        center_camera_m=(float(center[0]), float(center[1]), float(center[2])),
        median_depth_m=float(np.nanmedian(z)),
        valid_depth_points=int(xs.size),
    )


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = normalize_mask(mask_a)
    b = normalize_mask(mask_b)
    if a.shape != b.shape:
        return 0.0
    union = int(np.count_nonzero(a | b))
    if union <= 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / float(union)


def _center_delta_m(a: MaskDepthSignature, b: MaskDepthSignature) -> float | None:
    if a.center_camera_m is None or b.center_camera_m is None:
        return None
    return float(np.linalg.norm(np.asarray(a.center_camera_m) - np.asarray(b.center_camera_m)))


def _overlap_depth_compare(
    current_mask: np.ndarray,
    current_depth_m: np.ndarray,
    reference_mask: np.ndarray,
    reference_depth_m: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[int, float | None, float | None]:
    current = normalize_mask(current_mask)
    reference = normalize_mask(reference_mask)
    current_depth = depth_to_meters(current_depth_m)
    reference_depth = depth_to_meters(reference_depth_m)
    if current.shape != reference.shape or current.shape != current_depth.shape or reference.shape != reference_depth.shape:
        raise ValueError("mask/depth arrays must share the same shape for overlap comparison")

    overlap = (
        current
        & reference
        & np.isfinite(current_depth)
        & np.isfinite(reference_depth)
        & (current_depth > 0.0)
        & (reference_depth > 0.0)
    )
    ys, xs = np.nonzero(overlap)
    if xs.size == 0:
        return 0, None, None

    current_z = current_depth[ys, xs]
    reference_z = reference_depth[ys, xs]
    depth_delta = float(np.nanmedian(np.abs(current_z - reference_z)))
    current_points = _points_from_depth_pixels(xs, ys, current_z, camera_matrix)
    reference_points = _points_from_depth_pixels(xs, ys, reference_z, camera_matrix)
    center_delta = float(
        np.linalg.norm(np.nanmedian(current_points, axis=0) - np.nanmedian(reference_points, axis=0))
    )
    return int(xs.size), depth_delta, center_delta


class MaskDepthMovementTracker:
    def __init__(self, thresholds: MovementThresholds | None = None) -> None:
        self.thresholds = thresholds or MovementThresholds()
        self.reference_mask: np.ndarray | None = None
        self.reference_depth_m: np.ndarray | None = None
        self.reference_signature: MaskDepthSignature | None = None
        self.first_hand_contact: HandContact | None = None
        self.first_inside_contact: HandContact | None = None
        self.movement_candidate_frames = 0
        self.terminal_decision: MovementDecision | None = None

    def reset_reference(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        camera_matrix: np.ndarray,
        *,
        min_depth_points: int | None = None,
    ) -> MaskDepthSignature:
        signature = signature_from_mask_depth(
            mask,
            depth_m,
            camera_matrix,
            min_depth_points=min_depth_points or self.thresholds.min_depth_points,
        )
        if not signature.has_enough_depth:
            raise ValueError("initial mask/depth does not contain enough valid depth points")
        self.reference_mask = normalize_mask(mask).copy()
        self.reference_depth_m = depth_to_meters(depth_m).copy()
        self.reference_signature = signature
        self.movement_candidate_frames = 0
        self.terminal_decision = None
        return signature

    def note_hand_contact(self, contact: HandContact | None) -> None:
        if contact is None:
            return
        if self.first_hand_contact is None:
            self.first_hand_contact = contact
        if contact.inside_box and self.first_inside_contact is None:
            self.first_inside_contact = contact

    def trigger_contact(self) -> HandContact | None:
        return self.first_inside_contact or self.first_hand_contact

    def update(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        camera_matrix: np.ndarray,
        *,
        timestamp: RosStamp | None = None,
        hand_contact: HandContact | None = None,
    ) -> MovementDecision:
        if self.terminal_decision is not None:
            return self.terminal_decision

        self.note_hand_contact(hand_contact)
        thresholds = self.thresholds
        current_mask = normalize_mask(mask)
        current_depth = depth_to_meters(depth_m)
        current_signature = signature_from_mask_depth(
            current_mask,
            current_depth,
            camera_matrix,
            min_depth_points=thresholds.min_depth_points,
        )

        if self.reference_signature is None:
            if not current_signature.has_enough_depth:
                return MovementDecision(
                    status="waiting_for_reference",
                    moved=False,
                    occluded=True,
                    stable_in_place=False,
                    should_stop_tracking=False,
                    reason="not enough valid depth points to initialize reference",
                    timestamp=timestamp,
                    trigger_contact=self.trigger_contact(),
                )
            self.reference_mask = current_mask.copy()
            self.reference_depth_m = current_depth.copy()
            self.reference_signature = current_signature
            return MovementDecision(
                status="initialized",
                moved=False,
                occluded=False,
                stable_in_place=True,
                should_stop_tracking=False,
                reason="reference mask/depth initialized",
                timestamp=timestamp,
                trigger_contact=self.trigger_contact(),
            )

        assert self.reference_mask is not None
        assert self.reference_depth_m is not None
        reference_signature = self.reference_signature
        reference_area = max(1, int(reference_signature.area_px))
        area_ratio = float(current_signature.area_px) / float(reference_area)
        iou = mask_iou(current_mask, self.reference_mask)
        complete_again = (
            area_ratio >= thresholds.complete_area_ratio
            and iou >= thresholds.complete_iou_min
            and current_signature.has_enough_depth
        )

        overlap_pixels, depth_delta, visible_center_delta = _overlap_depth_compare(
            current_mask,
            current_depth,
            self.reference_mask,
            self.reference_depth_m,
            camera_matrix,
        )
        visible_area_ratio = float(overlap_pixels) / float(reference_area)
        center_delta = _center_delta_m(current_signature, reference_signature)
        enough_visible = (
            overlap_pixels >= thresholds.min_overlap_pixels
            and visible_area_ratio >= thresholds.min_visible_area_ratio
        )

        stable_visible = (
            enough_visible
            and depth_delta is not None
            and visible_center_delta is not None
            and depth_delta <= thresholds.depth_stable_tolerance_m
            and visible_center_delta <= thresholds.movement_threshold_m
        )
        complete_stable = (
            complete_again
            and center_delta is not None
            and center_delta <= thresholds.movement_threshold_m
            and (depth_delta is None or depth_delta <= thresholds.depth_stable_tolerance_m)
        )

        movement_evidence = False
        evidence_reason = ""
        if complete_again and center_delta is not None and center_delta > thresholds.movement_threshold_m:
            movement_evidence = True
            evidence_reason = f"complete mask center moved {center_delta:.3f} m"
        elif enough_visible and depth_delta is not None and depth_delta > thresholds.depth_stable_tolerance_m:
            movement_evidence = True
            evidence_reason = f"visible overlap depth changed {depth_delta:.3f} m"
        elif enough_visible and visible_center_delta is not None and visible_center_delta > thresholds.movement_threshold_m:
            movement_evidence = True
            evidence_reason = f"visible overlap center moved {visible_center_delta:.3f} m"

        if movement_evidence:
            self.movement_candidate_frames += 1
            status = "movement_candidate"
            should_stop = False
            moved = False
            reason = evidence_reason
            if self.movement_candidate_frames >= max(1, thresholds.taken_away_stable_frames):
                status = "taken_away"
                should_stop = True
                moved = True
                reason = f"{evidence_reason}; sustained for {self.movement_candidate_frames} frame(s)"
            decision = MovementDecision(
                status=status,
                moved=moved,
                occluded=False,
                stable_in_place=False,
                should_stop_tracking=should_stop,
                reason=reason,
                timestamp=timestamp,
                trigger_contact=self.trigger_contact(),
                center_delta_m=center_delta,
                depth_delta_m=depth_delta,
                overlap_pixels=overlap_pixels,
                visible_area_ratio=visible_area_ratio,
                area_ratio=area_ratio,
                mask_iou=iou,
                movement_candidate_frames=self.movement_candidate_frames,
            )
            if should_stop:
                self.terminal_decision = decision
            return decision

        self.movement_candidate_frames = 0
        if complete_stable:
            self.reference_mask = current_mask.copy()
            self.reference_depth_m = current_depth.copy()
            self.reference_signature = current_signature
            return MovementDecision(
                status="stable",
                moved=False,
                occluded=False,
                stable_in_place=True,
                should_stop_tracking=False,
                reason="complete mask is stable; reference updated",
                timestamp=timestamp,
                trigger_contact=self.trigger_contact(),
                center_delta_m=center_delta,
                depth_delta_m=depth_delta,
                overlap_pixels=overlap_pixels,
                visible_area_ratio=visible_area_ratio,
                area_ratio=area_ratio,
                mask_iou=iou,
            )

        if stable_visible:
            return MovementDecision(
                status="occluded_in_place",
                moved=False,
                occluded=True,
                stable_in_place=True,
                should_stop_tracking=False,
                reason="partial mask still matches reference depth in the visible overlap",
                timestamp=timestamp,
                trigger_contact=self.trigger_contact(),
                center_delta_m=center_delta,
                depth_delta_m=depth_delta,
                overlap_pixels=overlap_pixels,
                visible_area_ratio=visible_area_ratio,
                area_ratio=area_ratio,
                mask_iou=iou,
            )

        return MovementDecision(
            status="occluded_lost",
            moved=False,
            occluded=True,
            stable_in_place=False,
            should_stop_tracking=False,
            reason="not enough stable visible overlap to decide movement",
            timestamp=timestamp,
            trigger_contact=self.trigger_contact(),
            center_delta_m=center_delta,
            depth_delta_m=depth_delta,
            overlap_pixels=overlap_pixels,
            visible_area_ratio=visible_area_ratio,
            area_ratio=area_ratio,
            mask_iou=iou,
        )


def decision_summary(decision: MovementDecision) -> dict[str, Any]:
    return decision.to_dict()
