"""Directed local transitions built from the frozen raw-frame motion timeline.

This module deliberately has no dependency on Assignment-v4 region evidence.
In particular, it cannot read ``per_target`` rows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np


class LocalMotionState(str, Enum):
    CLOSED_PLATEAU = "CLOSED_PLATEAU"
    ACTIVE_MOTION = "ACTIVE_MOTION"
    OPEN_PLATEAU = "OPEN_PLATEAU"
    INACTIVE_INTERMEDIATE = "INACTIVE_INTERMEDIATE"


@dataclass(frozen=True)
class LocalFrameMotionState:
    frame_id: int
    timestamp: float
    q: float
    velocity: float
    state: LocalMotionState
    reason: str

    def to_dict(self) -> dict:
        row = asdict(self)
        row["state"] = self.state.value
        return row


def decompose_local_motion_states(
    frame_ids: list[int], timestamps: list[float], q_values: list[float], cfg: dict
) -> list[LocalFrameMotionState]:
    """Decompose frozen q while preserving stopped intermediate states."""
    frame_ids_array = np.asarray(frame_ids, np.int64)
    timestamps_array = np.asarray(timestamps, float)
    q = np.asarray(q_values, float)
    if not (len(frame_ids_array) == len(timestamps_array) == len(q)) or not len(q):
        raise ValueError("frame_ids, timestamps, and q must be non-empty and equal length")
    if not np.all(np.diff(frame_ids_array) > 0):
        raise ValueError("frame IDs must be strictly increasing in raw recording order")
    if not np.all(np.isfinite(timestamps_array)) or not np.all(np.diff(timestamps_array) > 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if not np.all(np.isfinite(q)):
        raise ValueError("q must be finite")
    velocity = (np.gradient(q, timestamps_array, edge_order=int(cfg.get("velocity_edge_order", 1)))
                if len(q) > 1 else np.zeros_like(q))
    active = np.abs(velocity) >= float(cfg["active_velocity_threshold"])
    radius = int(cfg.get("active_dilation_frames", 0))
    if radius:
        active = np.convolve(active.astype(np.uint8), np.ones(2 * radius + 1, np.uint8), mode="same") > 0
    q_min, q_max = float(q.min()), float(q.max())
    closed_q, open_q = ((q_min, q_max) if bool(cfg.get("closed_is_minimum_q", True))
                        else (q_max, q_min))
    tolerance = float(cfg["plateau_q_tolerance"])
    rows = []
    for index in range(len(q)):
        if active[index]:
            state, reason = LocalMotionState.ACTIVE_MOTION, "frozen_q_velocity_exceeds_threshold"
        elif abs(q[index] - closed_q) <= tolerance:
            state, reason = LocalMotionState.CLOSED_PLATEAU, "near_closed_extremum_and_not_active"
        elif abs(q[index] - open_q) <= tolerance:
            state, reason = LocalMotionState.OPEN_PLATEAU, "near_open_extremum_and_not_active"
        else:
            state, reason = (LocalMotionState.INACTIVE_INTERMEDIATE,
                             "inactive_intermediate_explicit_no_ownership_creation")
        rows.append(LocalFrameMotionState(
            int(frame_ids_array[index]), float(timestamps_array[index]), float(q[index]),
            float(velocity[index]), state, reason))
    return rows


@dataclass(frozen=True)
class LocalActiveTransition:
    source_frame: int
    target_frame: int
    source_q: float
    target_q: float
    delta_q: float
    elapsed_s: float
    frame_gap: int
    chronological_direction: str
    intermediate_frames: tuple[int, ...]
    intermediate_states: tuple[str, ...]
    interval_frames: tuple[int, ...]
    interval_states: tuple[str, ...]
    valid: bool
    validity_reason: str

    @property
    def directed_key(self) -> tuple[int, int]:
        return self.source_frame, self.target_frame

    def to_dict(self) -> dict:
        return asdict(self)


def build_local_active_transitions(
    states: list[LocalFrameMotionState], cfg: dict
) -> tuple[list[LocalActiveTransition], list[LocalActiveTransition]]:
    """Return accepted and rejected chronological local transition candidates."""
    minimum = float(cfg["minimum_delta_q_m"])
    maximum = float(cfg["maximum_delta_q_m"])
    maximum_gap = int(cfg["maximum_frame_gap"])
    maximum_elapsed = float(cfg["maximum_elapsed_s"])
    require_active = bool(cfg.get("require_all_intermediate_frames_active", True))
    accepted: list[LocalActiveTransition] = []
    rejected: list[LocalActiveTransition] = []
    for source_index, source in enumerate(states):
        for target_index in range(source_index + 1, len(states)):
            target = states[target_index]
            frame_gap = target.frame_id - source.frame_id
            if frame_gap > maximum_gap:
                break
            interval = states[source_index : target_index + 1]
            intermediate = states[source_index + 1 : target_index]
            delta_q = float(target.q - source.q)
            elapsed = float(target.timestamp - source.timestamp)
            failures = []
            if target.frame_id <= source.frame_id:
                failures.append("not_chronological")
            if abs(delta_q) < minimum:
                failures.append("delta_q_below_minimum")
            if abs(delta_q) > maximum:
                failures.append("delta_q_above_maximum")
            if elapsed <= 0 or elapsed > maximum_elapsed:
                failures.append("elapsed_outside_limit")
            if require_active and any(row.state != LocalMotionState.ACTIVE_MOTION for row in interval):
                failures.append("interval_contains_non_active_motion")
            row = LocalActiveTransition(
                source_frame=int(source.frame_id), target_frame=int(target.frame_id),
                source_q=float(source.q), target_q=float(target.q), delta_q=delta_q,
                elapsed_s=elapsed, frame_gap=int(frame_gap), chronological_direction="forward",
                intermediate_frames=tuple(int(item.frame_id) for item in intermediate),
                intermediate_states=tuple(item.state.value for item in intermediate),
                interval_frames=tuple(int(item.frame_id) for item in interval),
                interval_states=tuple(item.state.value for item in interval),
                valid=not failures,
                validity_reason=("all_local_active_transition_constraints_passed"
                                 if not failures else ";".join(failures)),
            )
            (accepted if row.valid else rejected).append(row)
    return accepted, rejected


def local_targets_for_source(
    source_frame: int,
    transitions: list[LocalActiveTransition],
    allow_backward_identity_confirmation: bool = True,
) -> list[dict]:
    """Return ordered local targets without erasing transition direction."""
    rows = []
    for transition in transitions:
        if transition.source_frame == source_frame:
            target = transition.target_frame
            direction = "forward"
            causal = True
        elif allow_backward_identity_confirmation and transition.target_frame == source_frame:
            target = transition.source_frame
            direction = "backward_identity_confirmation"
            causal = False
        else:
            continue
        rows.append({
            "source_frame": int(source_frame), "target_frame": int(target),
            "identity_observation_direction": direction, "causal_eligible": causal,
            "chronological_transition": transition.to_dict(),
        })
    return sorted(rows, key=lambda row: row["target_frame"])
