from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import cv2
import numpy as np

from config import SHIGURE_ID_HANDOFF_GRACE_SECONDS
from .cache import CachedShigureFrame, RosStamp, sample_key


CANONICAL_SCHEMA_VERSION = 2
EVENT_ACTIONS = frozenset({"bring_in", "take_out", "obj_move"})
RESOLVED = "RESOLVED"
UNRESOLVED = "UNRESOLVED"
REJECTED = "REJECTED"


@dataclass(frozen=True)
class DetectionItemKey:
    """Frame-local key for a legacy DetectedObject entry.

    The legacy message has no object id.  Its list index is meaningful only
    together with the exact ROS source timestamp.
    """

    source_stamp: RosStamp
    index: int

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("detection index must be non-negative")

    @property
    def value(self) -> str:
        return f"{sample_key(self.source_stamp)}:detection:{int(self.index):04d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_stamp": self.source_stamp.to_dict(),
            "index": int(self.index),
            "value": self.value,
        }


@dataclass
class _FrameBucket:
    source_stamp: RosStamp
    frame_ids: dict[str, str] = field(default_factory=dict)
    event_frame_id: str = ""
    event_slots: dict[str, int] = field(default_factory=dict)
    next_event_slot: int = 0
    received_monotonic: float = field(default_factory=time.monotonic)
    image_shape: tuple[int, int] | None = None
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)


def _bbox_xyxy(payload: Mapping[str, Any] | None) -> tuple[float, float, float, float] | None:
    values = payload.get("bbox_xyxy") if isinstance(payload, Mapping) else None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _bbox_close(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    tolerance: float = 1.0,
) -> bool:
    return max(abs(a - b) for a, b in zip(left, right)) <= float(tolerance)


def _bbox_intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = _bbox_intersection_area(left, right)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _bbox_union(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3])


def paste_bbox_local_mask(
    mask: np.ndarray,
    bbox_xyxy: Sequence[float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Return an HxW mask without ever stretching bbox-local coordinates."""

    source = np.asarray(mask)
    if source.ndim == 3:
        source = source[:, :, 0]
    if source.ndim != 2:
        raise ValueError("mask must be a 2D image")
    source = source > 0
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image shape must be positive")
    if source.shape == (height, width):
        if not np.any(source):
            raise ValueError("mask is empty")
        return source.copy()

    if len(bbox_xyxy) != 4:
        raise ValueError("bbox must contain x0, y0, x1, y1")
    x0f, y0f, x1f, y1f = [float(value) for value in bbox_xyxy]
    if not np.isfinite([x0f, y0f, x1f, y1f]).all() or x1f <= x0f or y1f <= y0f:
        raise ValueError("bbox is invalid")
    x0, y0, x1, y1 = [int(round(value)) for value in (x0f, y0f, x1f, y1f)]
    expected_shape = (y1 - y0, x1 - x0)
    if expected_shape[0] <= 0 or expected_shape[1] <= 0 or source.shape != expected_shape:
        raise ValueError(
            f"bbox-local mask shape mismatch: got {source.shape}, expected {expected_shape}"
        )

    destination = np.zeros((height, width), dtype=bool)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(width, x1), min(height, y1)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        raise ValueError("bbox lies outside the source image")
    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)
    destination[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]
    if not np.any(destination):
        raise ValueError("mask is empty after clipping")
    return destination


def _encode_mask(mask: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", np.asarray(mask, dtype=np.uint8) * 255)
    if not ok:
        raise ValueError("failed to encode canonical mask")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _decode_legacy_mask(value: Any) -> np.ndarray:
    try:
        encoded = base64.b64decode(str(value or ""), validate=True)
    except Exception as exc:
        raise ValueError("mask base64 is invalid") from exc
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("mask PNG could not be decoded")
    return image


def _canonical_detection(
    item: Mapping[str, Any],
    *,
    source_stamp: RosStamp,
    image_shape: tuple[int, int] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    index = int(item.get("index") or 0)
    key = DetectionItemKey(source_stamp, index)
    bbox = _bbox_xyxy(item)
    payload: dict[str, Any] = {
        "item_key": key.to_dict(),
        "action": str(item.get("action") or "").strip().lower(),
        "bbox_xyxy": list(bbox) if bbox is not None else None,
        "mask_coordinate_space": "full_frame",
        "mask_status": "MISSING",
    }
    if bbox is None:
        return payload, {"code": "INVALID_DETECTION_BBOX", "detection_item_key": key.value}
    if image_shape is None:
        payload["mask_status"] = "IMAGE_SHAPE_MISSING"
        return payload, {"code": "MASK_IMAGE_SHAPE_MISSING", "detection_item_key": key.value}
    try:
        local = _decode_legacy_mask(item.get("mask_b64"))
        full = paste_bbox_local_mask(local, bbox, image_shape)
    except Exception as exc:
        payload["mask_status"] = "INVALID"
        return payload, {
            "code": "MASK_SHAPE_OR_DECODE_INVALID",
            "detection_item_key": key.value,
            "error": str(exc),
        }
    payload.update(
        {
            "mask_status": "VALID",
            "mask_b64": _encode_mask(full),
            "mask_format": "png",
            "mask_size_wh": [int(image_shape[1]), int(image_shape[0])],
            "mask_pixels": int(np.count_nonzero(full)),
        }
    )
    return payload, None


def _unique_complete_matching(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    edge: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
) -> dict[int, int] | None:
    if len(left) != len(right) or len(left) > 8:
        return None
    if not left:
        return {}
    candidates = [[index for index, item in enumerate(right) if edge(source, item)] for source in left]
    if any(not values for values in candidates):
        return None
    solutions: list[dict[int, int]] = []

    def search(position: int, used: set[int], current: dict[int, int]) -> None:
        if len(solutions) > 1:
            return
        if position == len(left):
            solutions.append(dict(current))
            return
        for target in candidates[position]:
            if target in used:
                continue
            used.add(target)
            current[position] = target
            search(position + 1, used, current)
            current.pop(position, None)
            used.remove(target)

    search(0, set(), {})
    return solutions[0] if len(solutions) == 1 else None


def _event_contacts(
    contacts: Sequence[Mapping[str, Any]],
    *,
    action: str,
    object_id: str | None,
    people: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not object_id:
        return []
    people_by_id = {str(item.get("people_id") or ""): item for item in people}
    result: list[dict[str, Any]] = []
    for item in contacts:
        if str(item.get("action") or "").strip().lower() != action:
            continue
        if str(item.get("object_id") or "").strip() != object_id:
            continue
        evidence = dict(item)
        person_id = str(item.get("people_id") or "").strip()
        if person_id and person_id in people_by_id:
            evidence["person"] = deepcopy(people_by_id[person_id])
        result.append(evidence)
    return result


def assemble_canonical_events(
    *,
    source_stamp: RosStamp,
    image_shape: tuple[int, int] | None,
    object_detection: Mapping[str, Any],
    object_tracking: Mapping[str, Any],
    previous_tracking: Mapping[str, Any] | None,
    contacted: Mapping[str, Any] | None,
    people_payload: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detections_raw = [item for item in object_detection.get("objects") or [] if isinstance(item, Mapping)]
    tracking_raw = [item for item in object_tracking.get("objects") or [] if isinstance(item, Mapping)]
    previous_raw = [
        item for item in ((previous_tracking or {}).get("objects") or []) if isinstance(item, Mapping)
    ]
    contacts = [item for item in ((contacted or {}).get("contacts") or []) if isinstance(item, Mapping)]
    people = [item for item in ((people_payload or {}).get("people") or []) if isinstance(item, Mapping)]

    canonical_detections: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for fallback_index, raw in enumerate(detections_raw):
        normalized = dict(raw)
        normalized["index"] = int(raw.get("index", fallback_index))
        detection, error = _canonical_detection(
            normalized,
            source_stamp=source_stamp,
            image_shape=image_shape,
        )
        detection["raw_index"] = int(normalized["index"])
        canonical_detections.append(detection)
        if error is not None:
            diagnostics.append(error)

    events: list[dict[str, Any]] = []
    consumed_tracking: set[int] = set()
    consumed_detection: set[int] = set()

    for action in ("bring_in", "take_out"):
        detection_indices = [
            index for index, item in enumerate(canonical_detections) if item.get("action") == action
        ]
        tracking_indices = [
            index
            for index, item in enumerate(tracking_raw)
            if str(item.get("action") or "").strip().lower() == action
        ]
        left = [canonical_detections[index] for index in detection_indices]
        right = [tracking_raw[index] for index in tracking_indices]

        if action == "bring_in":
            def edge(detection: Mapping[str, Any], tracked: Mapping[str, Any]) -> bool:
                left_bbox = _bbox_xyxy(detection)
                right_bbox = _bbox_xyxy(tracked)
                return left_bbox is not None and right_bbox is not None and _bbox_close(left_bbox, right_bbox)

            method = "TRACKING_EXACT_BBOX"
        else:
            previous_by_id = {
                str(item.get("object_id") or "").strip(): item
                for item in previous_raw
                if str(item.get("object_id") or "").strip()
                and str(item.get("action") or "").strip().lower() != "take_out"
            }

            def edge(detection: Mapping[str, Any], tracked: Mapping[str, Any]) -> bool:
                object_id = str(tracked.get("object_id") or "").strip()
                previous = previous_by_id.get(object_id)
                detection_bbox = _bbox_xyxy(detection)
                tracked_bbox = _bbox_xyxy(tracked)
                previous_bbox = _bbox_xyxy(previous)
                if detection_bbox is None or tracked_bbox is None or previous_bbox is None:
                    return False
                if _bbox_intersection_area(detection_bbox, previous_bbox) <= 0.0:
                    return False
                return _bbox_close(_bbox_union(previous_bbox, detection_bbox), tracked_bbox)

            method = "TRACKING_TAKEOUT_UNION_REPLAY"

        mapping = _unique_complete_matching(left, right, edge)
        if mapping is None:
            if left or right:
                diagnostics.append(
                    {
                        "code": "EVENT_ID_MAPPING_AMBIGUOUS",
                        "action": action,
                        "detection_count": len(left),
                        "tracking_count": len(right),
                    }
                )
            for source_index in detection_indices:
                detection = canonical_detections[source_index]
                consumed_detection.add(source_index)
                events.append(
                    {
                        "event_uid": detection["item_key"]["value"],
                        "action": action,
                        "resolution_status": UNRESOLVED,
                        "shigure_object_id": None,
                        "mapping_method": "AMBIGUOUS",
                        "detection": detection,
                        "tracking": None,
                        "contact_evidence": [],
                    }
                )
            continue

        for left_index, right_index in mapping.items():
            source_index = detection_indices[left_index]
            tracking_index = tracking_indices[right_index]
            detection = canonical_detections[source_index]
            tracked = dict(tracking_raw[tracking_index])
            object_id = str(tracked.get("object_id") or "").strip() or None
            status = RESOLVED if object_id else UNRESOLVED
            consumed_detection.add(source_index)
            consumed_tracking.add(tracking_index)
            events.append(
                {
                    "event_uid": detection["item_key"]["value"],
                    "action": action,
                    "resolution_status": status,
                    "shigure_object_id": object_id,
                    "mapping_method": method if object_id else "TRACKING_ID_MISSING",
                    "detection": detection,
                    "tracking": tracked,
                    "contact_evidence": _event_contacts(
                        contacts,
                        action=action,
                        object_id=object_id,
                        people=people,
                    ),
                }
            )

    # Legacy obj_move assigns the newest allocated id, not the moved object's id.
    # Preserve the evidence but never expose that id as a resolved canonical identity.
    move_detection_indices = [
        index for index, item in enumerate(canonical_detections) if item.get("action") == "obj_move"
    ]
    move_tracking = [
        dict(item)
        for item in tracking_raw
        if str(item.get("action") or "").strip().lower() == "obj_move"
    ]
    for offset, source_index in enumerate(move_detection_indices):
        detection = canonical_detections[source_index]
        consumed_detection.add(source_index)
        claimed = move_tracking[offset] if offset < len(move_tracking) else None
        events.append(
            {
                "event_uid": detection["item_key"]["value"],
                "action": "obj_move",
                "resolution_status": REJECTED,
                "shigure_object_id": None,
                "mapping_method": "LEGACY_OBJ_MOVE_FAIL_CLOSED",
                "detection": detection,
                "tracking": claimed,
                "claimed_shigure_object_id": (
                    (str(claimed.get("object_id") or "").strip() or None)
                    if claimed is not None
                    else None
                ),
                "contact_evidence": [],
            }
        )
    if move_detection_indices or move_tracking:
        diagnostics.append({"code": "LEGACY_OBJ_MOVE_ID_UNTRUSTED"})

    for index, detection in enumerate(canonical_detections):
        if index in consumed_detection:
            continue
        events.append(
            {
                "event_uid": detection["item_key"]["value"],
                "action": str(detection.get("action") or ""),
                "resolution_status": REJECTED,
                "shigure_object_id": None,
                "mapping_method": "UNSUPPORTED_OR_UNMATCHED_ACTION",
                "detection": detection,
                "tracking": None,
                "contact_evidence": [],
            }
        )

    # A tracking event can survive a BEST_EFFORT drop of the detection message.
    # Only synthesize it when that action has no detection at all; otherwise an
    # ambiguous detection/tracking frame must stay fail-closed.
    for action in ("bring_in", "take_out"):
        if any(item.get("action") == action for item in canonical_detections):
            continue
        for tracking_index, tracked_raw in enumerate(tracking_raw):
            if tracking_index in consumed_tracking:
                continue
            if str(tracked_raw.get("action") or "").strip().lower() != action:
                continue
            tracked = dict(tracked_raw)
            object_id = str(tracked.get("object_id") or "").strip() or None
            uid = f"{sample_key(source_stamp)}:tracking:{action}:{object_id or tracking_index}"
            events.append(
                {
                    "event_uid": uid,
                    "action": action,
                    "resolution_status": RESOLVED if object_id else UNRESOLVED,
                    "shigure_object_id": object_id,
                    "mapping_method": "TRACKING_EVENT_WITHOUT_MASK",
                    "detection": None,
                    "tracking": tracked,
                    "contact_evidence": _event_contacts(
                        contacts,
                        action=action,
                        object_id=object_id,
                        people=people,
                    ),
                }
            )

    events.sort(key=lambda item: str(item.get("event_uid") or ""))
    return events, diagnostics


def parse_segments_candidates(
    *,
    source_stamp: RosStamp,
    segments_payload: Mapping[str, Any],
    image_shape: tuple[int, int],
    object_tracking: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    height, width = int(image_shape[0]), int(image_shape[1])
    candidates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, item in enumerate(segments_payload.get("segments") or []):
        if not isinstance(item, Mapping):
            continue
        bbox = _bbox_xyxy(item)
        if bbox is None:
            diagnostics.append({"code": "SEGMENT_BBOX_INVALID", "segment_index": index})
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        y_vertices = list(item.get("x_masks") or [])
        x_vertices = list(item.get("y_masks") or [])
        if len(x_vertices) >= 3 and len(x_vertices) == len(y_vertices):
            try:
                polygon = np.asarray(
                    [[int(x), int(y)] for x, y in zip(x_vertices, y_vertices)],
                    dtype=np.int32,
                )
                cv2.fillPoly(mask, [polygon], 255)
                mask_source = "segments_polygon"
            except Exception as exc:
                diagnostics.append(
                    {"code": "SEGMENT_POLYGON_INVALID", "segment_index": index, "error": str(exc)}
                )
                continue
        else:
            diagnostics.append(
                {"code": "SEGMENT_POLYGON_MISSING", "segment_index": index}
            )
            continue
        binary = mask > 0
        if not np.any(binary):
            diagnostics.append({"code": "SEGMENT_MASK_EMPTY", "segment_index": index})
            continue
        candidates.append(
            {
                "candidate_id": f"{sample_key(source_stamp)}:segment:{index:04d}",
                "segment_index": int(index),
                "class_id": str(item.get("class_id") or ""),
                "probability": float(item.get("probability") or 0.0),
                "bbox_xyxy": list(bbox),
                "mask_coordinate_space": "full_frame",
                "mask_status": "VALID",
                "mask_source": mask_source,
                "mask_b64": _encode_mask(binary),
                "mask_format": "png",
                "mask_size_wh": [width, height],
                "mask_pixels": int(np.count_nonzero(binary)),
                "tracking_match_status": UNRESOLVED,
                "shigure_object_id": None,
                "tracking": None,
                "tracking_candidates": [],
            }
        )

    raw_tracking_items = [
        item
        for item in ((object_tracking or {}).get("objects") or [])
        if isinstance(item, Mapping)
        and str(item.get("action") or "").strip().lower() not in {"take_out", "obj_move"}
        and str(item.get("object_id") or "").strip()
    ]
    tracking_items: list[Mapping[str, Any]] = []
    for tracked in raw_tracking_items:
        tracked_bbox = _bbox_xyxy(tracked)
        duplicate_index = None
        duplicate_iou = 0.0
        if tracked_bbox is not None:
            for index, existing in enumerate(tracking_items):
                existing_bbox = _bbox_xyxy(existing)
                if existing_bbox is None:
                    continue
                overlap = _bbox_iou(tracked_bbox, existing_bbox)
                if overlap >= 0.90:
                    duplicate_index = index
                    duplicate_iou = float(overlap)
                    break
        if duplicate_index is None:
            tracking_items.append(tracked)
            continue
        previous = tracking_items[duplicate_index]
        previous_id = str(previous.get("object_id") or "")
        tracked_id = str(tracked.get("object_id") or "")
        previous_suffix = previous_id.rpartition("_")[2]
        tracked_suffix = tracked_id.rpartition("_")[2]
        if (
            tracked_suffix.isdigit()
            and (
                not previous_suffix.isdigit()
                or int(tracked_suffix) >= int(previous_suffix)
            )
        ):
            tracking_items[duplicate_index] = tracked
        diagnostics.append(
            {
                "code": "TRACKING_DUPLICATE_BBOX_COLLAPSED",
                "raw_ids": sorted({previous_id, tracked_id}),
                "representative_raw_id": str(
                    tracking_items[duplicate_index].get("object_id") or ""
                ),
                "bbox_iou": duplicate_iou,
            }
        )
    score_matrix: dict[tuple[int, int], float] = {}
    for candidate_index, candidate in enumerate(candidates):
        candidate_bbox = _bbox_xyxy(candidate)
        if candidate_bbox is None:
            continue
        rows: list[dict[str, Any]] = []
        for tracking_index, tracked in enumerate(tracking_items):
            tracked_bbox = _bbox_xyxy(tracked)
            if tracked_bbox is None:
                continue
            iou = _bbox_iou(candidate_bbox, tracked_bbox)
            rows.append(
                {
                    "shigure_object_id": str(tracked.get("object_id") or ""),
                    "bbox_iou": float(iou),
                }
            )
            score_matrix[(candidate_index, tracking_index)] = float(iou)
        rows.sort(key=lambda row: float(row["bbox_iou"]), reverse=True)
        candidate["tracking_candidates"] = rows

    # Flashing/duplicated segments must not win by greedy ordering. Accept a
    # raw-ID association only when the same pair is the unique best choice in
    # both directions and both sides clear the ambiguity margin.
    for candidate_index, candidate in enumerate(candidates):
        candidate_ranked = sorted(
            (
                (score_matrix.get((candidate_index, tracking_index), 0.0), tracking_index)
                for tracking_index in range(len(tracking_items))
            ),
            reverse=True,
        )
        if not candidate_ranked:
            continue
        score, tracking_index = candidate_ranked[0]
        candidate_second = candidate_ranked[1][0] if len(candidate_ranked) > 1 else 0.0
        if score < 0.5 or score - candidate_second < 0.1:
            continue

        tracking_ranked = sorted(
            (
                (score_matrix.get((other_index, tracking_index), 0.0), other_index)
                for other_index in range(len(candidates))
            ),
            reverse=True,
        )
        tracking_score, tracking_best_candidate = tracking_ranked[0]
        tracking_second = tracking_ranked[1][0] if len(tracking_ranked) > 1 else 0.0
        if (
            tracking_best_candidate != candidate_index
            or tracking_score != score
            or tracking_score - tracking_second < 0.1
        ):
            continue

        object_id = str(tracking_items[tracking_index].get("object_id") or "").strip()
        candidate["tracking_match_status"] = RESOLVED
        candidate["shigure_object_id"] = object_id
        # Keep the canonical method token stable; its v2 definition is the
        # stricter mutual-best/two-sided-margin rule implemented above.
        candidate["tracking_mapping_method"] = "SEGMENT_TRACKING_UNIQUE_IOU"
        candidate["tracking"] = deepcopy(dict(tracking_items[tracking_index]))
    return candidates, diagnostics


def _tracking_raw_id_namespace(payload: Mapping[str, Any]) -> str | None:
    """Return the common upstream TrackingInfo prefix when it is observable."""

    prefixes: set[str] = set()
    for item in payload.get("objects") or []:
        if not isinstance(item, Mapping):
            continue
        object_id = str(item.get("object_id") or "").strip()
        prefix, separator, suffix = object_id.rpartition("_")
        if separator and prefix and suffix.isdigit():
            prefixes.add(prefix)
    return next(iter(prefixes)) if len(prefixes) == 1 else None


class ShigureCompatibilityAdapter:
    """Assemble exact-stamp legacy ROS payloads into canonical server frames."""

    def __init__(
        self,
        emit: Callable[[CachedShigureFrame], Any],
        *,
        max_buckets: int = 512,
        source_incarnation_id: str | None = None,
    ) -> None:
        self._emit = emit
        self._max_buckets = max(8, int(max_buckets))
        self._source_incarnation_id = (
            str(source_incarnation_id).strip()
            if source_incarnation_id is not None
            else uuid.uuid4().hex
        )
        if not self._source_incarnation_id:
            raise ValueError("source_incarnation_id cannot be empty")
        self._base_source_incarnation_id = self._source_incarnation_id
        self._source_generation = 0
        self._tracking_namespace: str | None = None
        self._tracking_seen_ids: set[str] = set()
        self._tracking_active_ids: set[str] = set()
        self._unexpectedly_missing_tracking_ids: dict[str, float] = {}
        self._tracking_rotation_detail: dict[str, Any] | None = None
        self._last_tracking_stamp_key = ""
        self._obj_move_quarantine_active = False
        self._obj_move_barrier_stamp: tuple[int, int] | None = None
        self._buckets: dict[str, _FrameBucket] = {}
        self._lock = threading.RLock()

    @property
    def source_incarnation_id(self) -> str:
        return self._source_incarnation_id

    def _rotate_tracking_incarnation(
        self,
        namespace: str,
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        self._source_generation += 1
        self._source_incarnation_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            (
                f"shigure-tracking-incarnation:"
                f"{self._base_source_incarnation_id}:"
                f"{self._source_generation}:{namespace}"
            ),
        ).hex
        self._tracking_namespace = namespace
        self._tracking_rotation_detail = (
            deepcopy(dict(detail)) if detail is not None else None
        )
        self._tracking_seen_ids.clear()
        self._tracking_active_ids.clear()
        self._unexpectedly_missing_tracking_ids.clear()
        self._last_tracking_stamp_key = ""
        self._obj_move_quarantine_active = False
        # Never exact-join payloads across an upstream tracking restart.
        self._buckets.clear()


    def _obj_move_barrier_frame(
        self,
        source_stamp: RosStamp,
        *,
        frame_id: str,
    ) -> CachedShigureFrame:
        return CachedShigureFrame(
            source_stamp=source_stamp,
            source_incarnation_id=self._source_incarnation_id,
            frame_id=str(frame_id or ""),
            received_utc=datetime.now(timezone.utc).isoformat(),
            received_monotonic=time.monotonic(),
            schema_version=CANONICAL_SCHEMA_VERSION,
            input_states={
                "camera_info": "missing",
                "object_detection": "missing",
                "object_tracking": "missing",
                "segments": "missing",
                "people": "missing",
                "contacted": "missing",
            },
            events=[],
            tracked_objects=[],
            recovery_candidates=[],
            people=[],
            diagnostics=[
                {
                    "code": "UPSTREAM_OBJ_MOVE_SOURCE_EPOCH_ROTATED",
                    "reason": "obj_move_raw_id_is_not_authoritative",
                }
            ],
        )

    def ingest(
        self,
        topic_key: str,
        source_stamp: RosStamp,
        payload: Mapping[str, Any],
        *,
        frame_id: str = "",
    ) -> CachedShigureFrame | None:
        key = sample_key(source_stamp)
        stamp_order = (int(source_stamp.sec), int(source_stamp.nanosec))
        with self._lock:
            barrier = self._obj_move_barrier_stamp
            if topic_key in {"object_detection", "object_tracking"}:
                has_obj_move = any(
                    str(item.get("action") or "").strip().lower()
                    == "obj_move"
                    for item in payload.get("objects") or []
                    if isinstance(item, Mapping)
                )
                if has_obj_move:
                    if barrier is not None and stamp_order <= barrier:
                        return None
                    barrier_frame: CachedShigureFrame | None = None
                    if not self._obj_move_quarantine_active:
                        previous_namespace = self._tracking_namespace
                        namespace = (
                            _tracking_raw_id_namespace(payload)
                            if topic_key == "object_tracking"
                            else previous_namespace
                        )
                        self._rotate_tracking_incarnation(
                            namespace or "obj_move_unknown_namespace"
                        )
                        if namespace is None:
                            self._tracking_namespace = None
                        self._obj_move_quarantine_active = True
                        barrier_frame = self._obj_move_barrier_frame(
                            source_stamp,
                            frame_id=frame_id,
                        )
                    self._obj_move_barrier_stamp = stamp_order
                    # Upstream obj_move assigns the latest allocated ID rather
                    # than proving the moved object's ID. Never allow this
                    # exact-stamp detection/tracking payload into a canonical
                    # frame; emit only an empty incarnation barrier.
                    if barrier_frame is not None:
                        self._emit(barrier_frame)
                    return barrier_frame
            if barrier is not None and stamp_order <= barrier:
                # No delayed payload from the indeterminate move stamp (or
                # before it) may enter the freshly rotated source epoch.  This
                # includes auxiliary camera/segment/person inputs, not only
                # detection and tracking.
                return None
            if topic_key == "object_tracking":
                was_quarantined = self._obj_move_quarantine_active
                if was_quarantined:
                    # Keep only auxiliary inputs at this exact clean tracking
                    # stamp. Other quarantine buckets are indeterminate and
                    # must never be replayed after the barrier unlocks.
                    retained_bucket = self._buckets.get(key)
                    self._buckets.clear()
                    if retained_bucket is not None:
                        self._buckets[key] = retained_bucket
                    self._obj_move_quarantine_active = False
                tracking_objects = [
                    item
                    for item in payload.get("objects") or []
                    if isinstance(item, Mapping)
                ]
                namespace = _tracking_raw_id_namespace(payload)
                if not tracking_objects and self._tracking_namespace is not None:
                    # An authoritative empty snapshot has no ID from which to
                    # recover the namespace. It still means every previously
                    # active raw ID disappeared in the current namespace.
                    namespace = self._tracking_namespace
                tracking_ids = {
                    str(item.get("object_id") or "").strip()
                    for item in tracking_objects
                    if str(item.get("object_id") or "").strip()
                }
                take_out_ids = {
                    str(item.get("object_id") or "").strip()
                    for item in tracking_objects
                    if str(item.get("object_id") or "").strip()
                    and str(item.get("action") or "").lower() == "take_out"
                }
                current_active_ids = tracking_ids - take_out_ids
                observed_monotonic = time.monotonic()
                if was_quarantined:
                    # The obj_move marker already opened the new epoch.  The
                    # first strictly newer clean tracking snapshot establishes
                    # its namespace without rotating a second time.
                    self._tracking_namespace = namespace
                elif namespace is not None:
                    if self._tracking_namespace is None:
                        self._tracking_namespace = namespace
                    elif namespace != self._tracking_namespace:
                        self._rotate_tracking_incarnation(
                            namespace,
                            detail={
                                "code": "TRACKING_NAMESPACE_CHANGED_DINOV2_REQUIRED",
                                "missing_raw_ids": sorted(self._tracking_active_ids),
                                "new_raw_ids": sorted(current_active_ids),
                                "grace_seconds": SHIGURE_ID_HANDOFF_GRACE_SECONDS,
                                "source_stamp": source_stamp.to_dict(),
                            },
                        )
                    else:
                        cutoff = observed_monotonic - SHIGURE_ID_HANDOFF_GRACE_SECONDS
                        self._unexpectedly_missing_tracking_ids = {
                            object_id: missing_at
                            for object_id, missing_at in
                            self._unexpectedly_missing_tracking_ids.items()
                            if missing_at >= cutoff
                            and object_id not in current_active_ids
                        }
                        unexpectedly_missing = (
                            self._tracking_active_ids
                            - current_active_ids
                            - take_out_ids
                        )
                        for object_id in unexpectedly_missing:
                            self._unexpectedly_missing_tracking_ids.setdefault(
                                object_id,
                                observed_monotonic,
                            )
                        new_ids = current_active_ids - self._tracking_seen_ids
                        handoff_sources = sorted(
                            self._unexpectedly_missing_tracking_ids
                        )
                        id_handoff = bool(new_ids and handoff_sources)
                        reused = any(
                            object_id in self._tracking_seen_ids
                            and bool(self._last_tracking_stamp_key)
                            and self._last_tracking_stamp_key != key
                            and (
                                object_id not in self._tracking_active_ids
                                or str(item.get("action") or "").lower()
                                == "bring_in"
                            )
                            for item in tracking_objects
                            if (
                                object_id := str(
                                    item.get("object_id") or ""
                                ).strip()
                            )
                        )
                        if id_handoff:
                            print(
                                "[shigure_history] tracking ID handoff candidate; "
                                f"missing={handoff_sources} new={sorted(new_ids)} "
                                "opening a new source epoch for DINOv2 verification",
                                flush=True,
                            )
                            self._rotate_tracking_incarnation(
                                namespace,
                                detail={
                                    "code": "TRACKING_ID_HANDOFF_DINOV2_REQUIRED",
                                    "missing_raw_ids": handoff_sources,
                                    "new_raw_ids": sorted(new_ids),
                                    "grace_seconds": SHIGURE_ID_HANDOFF_GRACE_SECONDS,
                                    "source_stamp": source_stamp.to_dict(),
                                },
                            )
                        elif reused:
                            # The upstream prefix has only second precision.
                            # Reusing an already-retired ID (or re-bringing an
                            # active ID at a new stamp) proves an in-second restart.
                            self._rotate_tracking_incarnation(namespace)
                self._tracking_seen_ids.update(tracking_ids)
                self._tracking_active_ids = current_active_ids
                for object_id in current_active_ids:
                    self._unexpectedly_missing_tracking_ids.pop(object_id, None)
                self._last_tracking_stamp_key = key
            bucket = self._buckets.setdefault(key, _FrameBucket(source_stamp=source_stamp))
            if (
                topic_key in {"object_detection", "object_tracking"}
                and frame_id
                and not bucket.event_frame_id
            ):
                bucket.event_frame_id = str(frame_id)
            if frame_id:
                bucket.frame_ids[str(topic_key)] = str(frame_id)
            normalized = deepcopy(dict(payload))
            bucket.inputs[str(topic_key)] = normalized
            if topic_key == "camera_info":
                try:
                    width = int(normalized["width"])
                    height = int(normalized["height"])
                    if width > 0 and height > 0:
                        bucket.image_shape = (height, width)
                except (KeyError, TypeError, ValueError):
                    pass
            frame = self._build(bucket)
            self._prune()
            if frame is not None and not self._obj_move_quarantine_active:
                self._emit(frame)
                return frame
            # Newer camera/segment/person inputs may complete an exact-stamp
            # bucket while quarantined, but they are not observable until a
            # strictly newer clean tracking snapshot authorizes that stamp.
            return None

    def _previous_tracking(self, stamp: RosStamp) -> Mapping[str, Any] | None:
        stamp_key = (int(stamp.sec), int(stamp.nanosec))
        candidates = [
            bucket
            for bucket in self._buckets.values()
            if (int(bucket.source_stamp.sec), int(bucket.source_stamp.nanosec)) < stamp_key
            and "object_tracking" in bucket.inputs
        ]
        if not candidates:
            return None
        previous = max(
            candidates,
            key=lambda item: (int(item.source_stamp.sec), int(item.source_stamp.nanosec)),
        )
        return previous.inputs.get("object_tracking")

    @staticmethod
    def _input_state(bucket: _FrameBucket, key: str, count_key: str) -> str:
        if key not in bucket.inputs:
            return "missing"
        value = bucket.inputs[key]
        return "present" if int(value.get(count_key) or 0) > 0 else "explicit_empty"

    @staticmethod
    def _event_aliases(event: Mapping[str, Any], position: int) -> list[str]:
        """Return every exact-stamp identity available for one event revision."""

        aliases: list[str] = []
        action = str(event.get("action") or "").strip().lower()
        detection = event.get("detection")
        if isinstance(detection, Mapping):
            item_key = detection.get("item_key")
            if isinstance(item_key, Mapping):
                value = str(item_key.get("value") or "").strip()
                if value:
                    aliases.append(f"detection:{value}")

        tracking = event.get("tracking")
        if isinstance(tracking, Mapping):
            try:
                aliases.append(
                    f"tracking-index:{action}:{int(tracking['index'])}"
                )
            except (KeyError, TypeError, ValueError):
                pass
            object_id = str(tracking.get("object_id") or "").strip()
            if object_id:
                aliases.append(f"tracking-id:{action}:{object_id}")

        adapter_uid = str(event.get("event_uid") or "").strip()
        if adapter_uid:
            aliases.append(f"adapter:{adapter_uid}")
        if not aliases:
            aliases.append(f"position:{action}:{int(position)}")
        return aliases

    def _assign_canonical_event_indexes(
        self,
        bucket: _FrameBucket,
        events: Sequence[dict[str, Any]],
        diagnostics: list[dict[str, Any]],
    ) -> None:
        """Freeze a DB slot on the first detection/tracking revision.

        Detection and tracking can arrive in either order under BEST_EFFORT.
        Once the second side establishes their relationship, all aliases are
        joined to the slot already emitted for the first side.
        """

        for position, event in enumerate(events):
            aliases = self._event_aliases(event, position)
            existing = sorted(
                {
                    bucket.event_slots[alias]
                    for alias in aliases
                    if alias in bucket.event_slots
                }
            )
            if existing:
                slot = int(existing[0])
                if len(existing) > 1:
                    diagnostics.append(
                        {
                            "code": "CANONICAL_EVENT_SLOT_ALIAS_CONFLICT",
                            "aliases": aliases,
                            "slots": existing,
                            "selected_slot": slot,
                        }
                    )
                    for alias, value in list(bucket.event_slots.items()):
                        if value in existing:
                            bucket.event_slots[alias] = slot
            else:
                slot = int(bucket.next_event_slot)
                bucket.next_event_slot += 1
            for alias in aliases:
                bucket.event_slots[alias] = slot
            event["canonical_event_index"] = slot

    def _build(self, bucket: _FrameBucket) -> CachedShigureFrame | None:
        detection = bucket.inputs.get("object_detection")
        tracking = bucket.inputs.get("object_tracking")
        segments = bucket.inputs.get("segments")
        contacted = bucket.inputs.get("contacted")
        people = bucket.inputs.get("people")
        events_ready = detection is not None or tracking is not None
        recovery_ready = segments is not None and bucket.image_shape is not None
        # An auxiliary topic must never change the canonical event key on a
        # same-stamp replay. Detection/tracking own event frame identity;
        # Segments/camera own a recovery-only frame.
        frame_id = bucket.event_frame_id or next(
            (
                bucket.frame_ids[key]
                for key in (
                    "object_detection", "object_tracking", "segments",
                    "camera_info", "rgb", "depth", "people", "contacted",
                )
                if bucket.frame_ids.get(key)
            ),
            "",
        )
        if not events_ready and not recovery_ready:
            return None

        diagnostics: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        if events_ready:
            events, event_diagnostics = assemble_canonical_events(
                source_stamp=bucket.source_stamp,
                image_shape=bucket.image_shape,
                object_detection=detection or {},
                object_tracking=tracking or {},
                previous_tracking=self._previous_tracking(bucket.source_stamp),
                contacted=contacted,
                people_payload=people,
            )
            diagnostics.extend(event_diagnostics)
            self._assign_canonical_event_indexes(bucket, events, diagnostics)

        recovery_candidates: list[dict[str, Any]] = []
        if recovery_ready and segments is not None and bucket.image_shape is not None:
            recovery_candidates, segment_diagnostics = parse_segments_candidates(
                source_stamp=bucket.source_stamp,
                segments_payload=segments,
                image_shape=bucket.image_shape,
                object_tracking=tracking,
            )
            diagnostics.extend(segment_diagnostics)

        if self._tracking_rotation_detail is not None:
            diagnostics.append(deepcopy(self._tracking_rotation_detail))

        input_states = {
            "camera_info": "present" if "camera_info" in bucket.inputs and bucket.image_shape else "missing",
            "object_detection": self._input_state(bucket, "object_detection", "object_count"),
            "object_tracking": self._input_state(bucket, "object_tracking", "object_count"),
            "segments": self._input_state(bucket, "segments", "segment_count"),
            "people": self._input_state(bucket, "people", "people_count"),
            "contacted": self._input_state(bucket, "contacted", "contact_count"),
        }
        received_monotonic = time.monotonic()
        return CachedShigureFrame(
            source_stamp=bucket.source_stamp,
            source_incarnation_id=self._source_incarnation_id,
            frame_id=frame_id,
            received_utc=datetime.now(timezone.utc).isoformat(),
            received_monotonic=received_monotonic,
            schema_version=CANONICAL_SCHEMA_VERSION,
            input_states=input_states,
            events=events,
            tracked_objects=deepcopy((tracking or {}).get("objects") or []),
            recovery_candidates=recovery_candidates,
            people=deepcopy((people or {}).get("people") or []),
            diagnostics=diagnostics,
        )

    def _prune(self) -> None:
        if len(self._buckets) <= self._max_buckets:
            return
        ordered = sorted(
            self._buckets.items(),
            key=lambda item: (
                int(item[1].source_stamp.sec),
                int(item[1].source_stamp.nanosec),
            ),
        )
        for key, _bucket in ordered[: len(self._buckets) - self._max_buckets]:
            self._buckets.pop(key, None)
