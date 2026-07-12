from __future__ import annotations

import math
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from config import (
    SHIGURE_IDENTITY_GEOMETRY_WEIGHT,
    SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD,
    SHIGURE_IDENTITY_MATCH_REQUIRE_MARGIN,
    SHIGURE_IDENTITY_MATCH_SECOND_MARGIN,
    SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
)


MATCHED = "MATCHED"
UNBOUND = "UNBOUND"
AMBIGUOUS = "AMBIGUOUS"
IDENTITY_STATUSES = frozenset({MATCHED, UNBOUND, AMBIGUOUS})


class BindingConflictError(ValueError):
    """Raised when an explicit bind would violate a one-to-one binding."""


class BindingNotFoundError(KeyError):
    """Raised when a binding operation requires an existing matched record."""


def _non_empty_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _embedding_array(value: Any, *, label: str) -> np.ndarray:
    if isinstance(value, Mapping):
        value = value.get("embedding")
    try:
        embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise ValueError(f"{label} is not a numeric embedding") from exc
    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        raise ValueError(f"{label} must contain finite values")
    norm = float(np.linalg.norm(embedding))
    if norm <= 1.0e-12:
        raise ValueError(f"{label} has zero norm")
    return embedding / norm


def cosine_distance(left: Any, right: Any) -> float:
    """Return normalized cosine distance in the inclusive range [0, 2]."""
    a = _embedding_array(left, label="left embedding")
    b = _embedding_array(right, label="right embedding")
    if a.shape != b.shape:
        raise ValueError(f"embedding shape mismatch: {a.shape} vs {b.shape}")
    return float(max(0.0, min(2.0, 1.0 - float(np.dot(a, b)))))


def _looks_like_single_embedding(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    if not value:
        return False
    return isinstance(value[0], (int, float, np.integer, np.floating))


def _candidate_references(candidate: Mapping[str, Any]) -> list[tuple[Any, str | None]]:
    references: list[tuple[Any, str | None]] = []
    raw_references = candidate.get("references")
    if isinstance(raw_references, Sequence) and not isinstance(raw_references, (str, bytes, bytearray)):
        for index, reference in enumerate(raw_references):
            if isinstance(reference, Mapping):
                reference_id = str(
                    reference.get("reference_id")
                    or reference.get("capture_instance_id")
                    or reference.get("task_id")
                    or ""
                ).strip() or None
                references.append((reference.get("embedding"), reference_id))
            else:
                references.append((reference, str(index)))

    if references:
        return references

    raw_embeddings = candidate.get("embeddings")
    if raw_embeddings is not None:
        values = [raw_embeddings] if _looks_like_single_embedding(raw_embeddings) else list(raw_embeddings)
        for index, value in enumerate(values):
            if isinstance(value, Mapping):
                reference_id = str(
                    value.get("reference_id")
                    or value.get("capture_instance_id")
                    or value.get("task_id")
                    or ""
                ).strip() or None
                references.append((value.get("embedding"), reference_id))
            else:
                references.append((value, str(index)))
        return references

    if candidate.get("embedding") is not None:
        reference_id = str(
            candidate.get("reference_id")
            or candidate.get("capture_instance_id")
            or candidate.get("task_id")
            or ""
        ).strip() or None
        references.append((candidate.get("embedding"), reference_id))
    return references


def _geometry_score(candidate: Mapping[str, Any]) -> float | None:
    value = candidate.get("geometry_score")
    if value is None and isinstance(candidate.get("geometry"), Mapping):
        value = candidate["geometry"].get("score")
    if value is None:
        return None
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("geometry_score must be finite")
    return float(max(0.0, min(1.0, score)))


def _group_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], int]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    truncated = 0
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            continue
        display_object_id = str(raw_candidate.get("display_object_id") or "").strip()
        if not display_object_id:
            continue
        if display_object_id not in grouped:
            if len(grouped) >= max_candidates:
                truncated += 1
                continue
            grouped[display_object_id] = {
                "display_object_id": display_object_id,
                "references": [],
                "geometry_score": None,
            }
        entry = grouped[display_object_id]
        entry["references"].extend(_candidate_references(raw_candidate))
        geometry_score = _geometry_score(raw_candidate)
        if geometry_score is not None:
            previous = entry.get("geometry_score")
            entry["geometry_score"] = geometry_score if previous is None else max(float(previous), geometry_score)
    return list(grouped.values()), truncated


def match_display_identity(
    observation_embedding: Any,
    candidates: Iterable[Mapping[str, Any]],
    *,
    distance_threshold: float = SHIGURE_IDENTITY_MATCH_DISTANCE_THRESHOLD,
    second_margin: float = SHIGURE_IDENTITY_MATCH_SECOND_MARGIN,
    require_second_margin: bool = SHIGURE_IDENTITY_MATCH_REQUIRE_MARGIN,
    geometry_weight: float = SHIGURE_IDENTITY_GEOMETRY_WEIGHT,
    max_candidates: int = SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
) -> dict[str, Any]:
    """Match one Shigure observation to existing display objects.

    Each candidate may contain ``embedding``, ``embeddings``, or a
    ``references`` list with capture metadata. Multiple views are reduced by
    minimum cosine distance, matching the historical identity policy. An
    optional geometry score in [0, 1] adds ``weight * (1 - score)`` to the
    ranking distance. A missing geometry score does not penalize a candidate.

    The function never creates an ID and never mutates its inputs.
    """
    observation = _embedding_array(observation_embedding, label="observation embedding")
    distance_threshold = float(distance_threshold)
    second_margin = float(second_margin)
    geometry_weight = float(geometry_weight)
    if distance_threshold < 0.0 or second_margin < 0.0 or geometry_weight < 0.0:
        raise ValueError("identity thresholds and geometry_weight must be non-negative")
    candidate_limit = max(1, min(int(max_candidates or 1), int(SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS), 5))
    grouped, truncated = _group_candidates(candidates, max_candidates=candidate_limit)

    scores: list[dict[str, Any]] = []
    skipped_references = 0
    for candidate in grouped:
        best_distance: float | None = None
        best_reference_id: str | None = None
        valid_reference_count = 0
        for index, (raw_embedding, reference_id) in enumerate(candidate["references"]):
            try:
                reference = _embedding_array(
                    raw_embedding,
                    label=f"{candidate['display_object_id']} reference {index}",
                )
            except ValueError:
                skipped_references += 1
                continue
            if reference.shape != observation.shape:
                skipped_references += 1
                continue
            valid_reference_count += 1
            distance = float(max(0.0, min(2.0, 1.0 - float(np.dot(observation, reference)))))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_reference_id = reference_id
        if best_distance is None:
            continue

        geometry_score = candidate.get("geometry_score")
        geometry_penalty = 0.0
        if geometry_score is not None and geometry_weight > 0.0:
            geometry_penalty = geometry_weight * (1.0 - float(geometry_score))
        effective_distance = best_distance + geometry_penalty
        scores.append(
            {
                "display_object_id": candidate["display_object_id"],
                "dinov2_distance": float(best_distance),
                "geometry_score": float(geometry_score) if geometry_score is not None else None,
                "geometry_penalty": float(geometry_penalty),
                "effective_distance": float(effective_distance),
                "selected_reference_id": best_reference_id,
                "valid_reference_count": int(valid_reference_count),
            }
        )

    scores.sort(
        key=lambda item: (
            float(item["effective_distance"]),
            float(item["dinov2_distance"]),
            str(item["display_object_id"]),
        )
    )
    base_result: dict[str, Any] = {
        "status": UNBOUND,
        "display_object_id": None,
        "reason": "no_valid_candidate_embeddings",
        "candidate_scores": scores,
        "considered_display_object_count": len(grouped),
        "valid_display_object_count": len(scores),
        "truncated_display_object_count": int(truncated),
        "skipped_reference_count": int(skipped_references),
        "thresholds": {
            "distance": distance_threshold,
            "second_margin": second_margin,
            "require_second_margin": bool(require_second_margin),
            "geometry_weight": geometry_weight,
            "max_display_objects": candidate_limit,
        },
    }
    if not scores:
        return base_result

    best = scores[0]
    second = scores[1] if len(scores) > 1 else None
    best_distance = float(best["effective_distance"])
    second_distance = float(second["effective_distance"]) if second is not None else None
    margin = (second_distance - best_distance) if second_distance is not None else None
    margin_ok = second is None or float(margin) >= second_margin
    base_result.update(
        {
            "best_display_object_id": best["display_object_id"],
            "best_distance": best_distance,
            "best_dinov2_distance": float(best["dinov2_distance"]),
            "second_distance": second_distance,
            "second_margin": margin,
            "second_margin_ok": bool(margin_ok),
        }
    )
    if best_distance > distance_threshold:
        base_result["reason"] = "best_candidate_above_distance_threshold"
        return base_result
    if require_second_margin and not margin_ok:
        base_result.update(
            {
                "status": AMBIGUOUS,
                "reason": "second_candidate_too_close",
            }
        )
        return base_result

    base_result.update(
        {
            "status": MATCHED,
            "display_object_id": best["display_object_id"],
            "reason": "matched_existing_display_object",
            "selected_reference_id": best.get("selected_reference_id"),
        }
    )
    return base_result


@dataclass(frozen=True)
class BindingKey:
    ingress_session_uuid: str
    startup_session_id: str
    shigure_object_id: str


@dataclass(frozen=True)
class BindingRecord:
    key: BindingKey
    status: str
    display_object_id: str | None
    epoch: int
    generation: int
    reason: str
    candidate_display_object_ids: tuple[str, ...]
    updated_sequence: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key"] = asdict(self.key)
        payload["candidate_display_object_ids"] = list(self.candidate_display_object_ids)
        return payload


class EphemeralBindingRegistry:
    """Thread-safe, process-local Shigure-to-display binding registry."""

    def __init__(
        self,
        *,
        ingress_session_uuid: str | None = None,
        max_display_objects: int = SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
    ) -> None:
        self._ingress_session_uuid = str(ingress_session_uuid or uuid.uuid4()).strip()
        self._max_display_objects = max(1, min(int(max_display_objects or 1), 5))
        self._records: dict[BindingKey, BindingRecord] = {}
        self._reverse: dict[tuple[str, str], BindingKey] = {}
        self._namespace_generations: dict[str, int] = {}
        self._generation_serial = 1
        self._sequence = 0
        self._lock = RLock()

    @property
    def ingress_session_uuid(self) -> str:
        return self._ingress_session_uuid

    @property
    def max_display_objects(self) -> int:
        return self._max_display_objects

    def _key(self, startup_session_id: Any, shigure_object_id: Any) -> BindingKey:
        return BindingKey(
            ingress_session_uuid=self._ingress_session_uuid,
            startup_session_id=_non_empty_text(startup_session_id, "startup_session_id"),
            shigure_object_id=_non_empty_text(shigure_object_id, "shigure_object_id"),
        )

    def _generation_for(self, startup_session_id: str) -> int:
        return self._namespace_generations.setdefault(startup_session_id, self._generation_serial)

    def _remove_reverse(self, record: BindingRecord | None) -> None:
        if record is None or record.status != MATCHED or not record.display_object_id:
            return
        reverse_key = (record.key.startup_session_id, record.display_object_id)
        if self._reverse.get(reverse_key) == record.key:
            self._reverse.pop(reverse_key, None)

    def _write_record(
        self,
        key: BindingKey,
        *,
        status: str,
        display_object_id: str | None,
        reason: str,
        candidates: Sequence[str] = (),
    ) -> BindingRecord:
        if status not in IDENTITY_STATUSES:
            raise ValueError(f"unsupported identity status: {status}")
        previous = self._records.get(key)
        self._remove_reverse(previous)
        self._sequence += 1
        record = BindingRecord(
            key=key,
            status=status,
            display_object_id=display_object_id,
            epoch=(previous.epoch + 1) if previous is not None else 1,
            generation=self._generation_for(key.startup_session_id),
            reason=str(reason or ""),
            candidate_display_object_ids=tuple(
                dict.fromkeys(str(value).strip() for value in candidates if str(value).strip())
            ),
            updated_sequence=self._sequence,
        )
        self._records[key] = record
        if status == MATCHED and display_object_id:
            self._reverse[(key.startup_session_id, display_object_id)] = key
        return record

    def _matched_records(self, startup_session_id: str) -> list[BindingRecord]:
        return [
            record
            for record in self._records.values()
            if record.key.startup_session_id == startup_session_id and record.status == MATCHED
        ]

    def bind(
        self,
        startup_session_id: Any,
        shigure_object_id: Any,
        display_object_id: Any,
        *,
        reason: str = "identity_match",
        allow_rebind: bool = False,
    ) -> BindingRecord:
        """Explicitly bind an existing display ID; no persistent ID is created."""
        key = self._key(startup_session_id, shigure_object_id)
        display_id = _non_empty_text(display_object_id, "display_object_id")
        with self._lock:
            previous = self._records.get(key)
            if (
                previous is not None
                and previous.status == MATCHED
                and previous.display_object_id != display_id
                and not allow_rebind
            ):
                raise BindingConflictError(
                    f"{key.shigure_object_id} is already bound to {previous.display_object_id}"
                )
            other_key = self._reverse.get((key.startup_session_id, display_id))
            if other_key is not None and other_key != key:
                raise BindingConflictError(
                    f"{display_id} is already bound to Shigure object {other_key.shigure_object_id}"
                )

            matched = self._matched_records(key.startup_session_id)
            existing_display_ids = {record.display_object_id for record in matched}
            if display_id not in existing_display_ids and len(existing_display_ids) >= self._max_display_objects:
                oldest = min(matched, key=lambda record: record.updated_sequence)
                self._write_record(
                    oldest.key,
                    status=UNBOUND,
                    display_object_id=None,
                    reason="capacity_evicted",
                )
            return self._write_record(
                key,
                status=MATCHED,
                display_object_id=display_id,
                reason=reason,
                candidates=(display_id,),
            )

    def mark_unbound(
        self,
        startup_session_id: Any,
        shigure_object_id: Any,
        *,
        reason: str,
    ) -> BindingRecord:
        key = self._key(startup_session_id, shigure_object_id)
        with self._lock:
            return self._write_record(
                key,
                status=UNBOUND,
                display_object_id=None,
                reason=reason,
            )

    def mark_ambiguous(
        self,
        startup_session_id: Any,
        shigure_object_id: Any,
        *,
        candidate_display_object_ids: Sequence[Any],
        reason: str,
    ) -> BindingRecord:
        key = self._key(startup_session_id, shigure_object_id)
        with self._lock:
            return self._write_record(
                key,
                status=AMBIGUOUS,
                display_object_id=None,
                reason=reason,
                candidates=tuple(str(value) for value in candidate_display_object_ids),
            )

    def advance_epoch(
        self,
        startup_session_id: Any,
        shigure_object_id: Any,
        *,
        reason: str = "movement_epoch",
    ) -> BindingRecord:
        key = self._key(startup_session_id, shigure_object_id)
        with self._lock:
            previous = self._records.get(key)
            if previous is None or previous.status != MATCHED or not previous.display_object_id:
                raise BindingNotFoundError(key)
            return self._write_record(
                key,
                status=MATCHED,
                display_object_id=previous.display_object_id,
                reason=reason,
                candidates=previous.candidate_display_object_ids,
            )

    def get(self, startup_session_id: Any, shigure_object_id: Any) -> BindingRecord | None:
        key = self._key(startup_session_id, shigure_object_id)
        with self._lock:
            return self._records.get(key)

    def get_by_display_object(
        self,
        startup_session_id: Any,
        display_object_id: Any,
    ) -> BindingRecord | None:
        startup_id = _non_empty_text(startup_session_id, "startup_session_id")
        display_id = _non_empty_text(display_object_id, "display_object_id")
        with self._lock:
            key = self._reverse.get((startup_id, display_id))
            return self._records.get(key) if key is not None else None

    def release_display_object(
        self,
        startup_session_id: Any,
        display_object_id: Any,
        *,
        reason: str = "display_object_reanchored",
    ) -> BindingRecord | None:
        """Release the current temporary Shigure ID for one display object.

        A Shigure object ID is only meaningful inside its current startup
        namespace.  Even within that namespace, a fresh HoloLens capture can
        make Shigure allocate a new temporary ID for the same physical
        object.  Releasing the reverse entry lets a later strong
        geometry/DINO match bind that new ID without weakening the normal
        one-to-one guard.
        """

        startup_id = _non_empty_text(startup_session_id, "startup_session_id")
        display_id = _non_empty_text(display_object_id, "display_object_id")
        with self._lock:
            key = self._reverse.get((startup_id, display_id))
            if key is None:
                return None
            return self._write_record(
                key,
                status=UNBOUND,
                display_object_id=None,
                reason=reason,
            )

    def is_current(self, record: BindingRecord) -> bool:
        with self._lock:
            if record.key.ingress_session_uuid != self._ingress_session_uuid:
                return False
            if record.generation != self._generation_for(record.key.startup_session_id):
                return False
            current = self._records.get(record.key)
            return current == record

    def reset(
        self,
        *,
        startup_session_id: str | None = None,
        ingress_session_uuid: str | None = None,
    ) -> int:
        """Clear bindings and invalidate outstanding generation/epoch tokens."""
        with self._lock:
            self._generation_serial += 1
            if ingress_session_uuid is not None:
                self._ingress_session_uuid = _non_empty_text(
                    ingress_session_uuid,
                    "ingress_session_uuid",
                )
                startup_session_id = None
            if startup_session_id is None:
                removed = len(self._records)
                self._records.clear()
                self._reverse.clear()
                self._namespace_generations.clear()
                return removed

            startup_id = _non_empty_text(startup_session_id, "startup_session_id")
            keys = [key for key in self._records if key.startup_session_id == startup_id]
            for key in keys:
                self._remove_reverse(self._records.pop(key))
            self._namespace_generations[startup_id] = self._generation_serial
            return len(keys)

    def snapshot(self, *, startup_session_id: str | None = None) -> list[BindingRecord]:
        with self._lock:
            records = list(self._records.values())
            if startup_session_id is not None:
                startup_id = _non_empty_text(startup_session_id, "startup_session_id")
                records = [record for record in records if record.key.startup_session_id == startup_id]
            return sorted(records, key=lambda record: record.updated_sequence)
