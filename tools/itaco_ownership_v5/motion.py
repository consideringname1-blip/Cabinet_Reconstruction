"""Frozen-q motion-state decomposition and physically valid transitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

import numpy as np


class MotionState(str, Enum):
    CLOSED_PLATEAU = "CLOSED_PLATEAU"
    ACTIVE_MOTION = "ACTIVE_MOTION"
    OPEN_PLATEAU = "OPEN_PLATEAU"


@dataclass(frozen=True)
class FrameMotionState:
    frame_id: int
    timestamp: float
    q: float
    velocity: float
    state: MotionState
    reason: str

    def to_dict(self) -> dict:
        row = asdict(self)
        row["state"] = self.state.value
        return row


@dataclass(frozen=True)
class ActiveTransition:
    source_index: int
    target_index: int
    source_frame_id: int
    target_frame_id: int
    source_q: float
    target_q: float
    delta_q: float
    elapsed_s: float
    traversed_active_motion: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _validate(frame_ids: np.ndarray, timestamps: np.ndarray, q: np.ndarray) -> None:
    if not (len(frame_ids) == len(timestamps) == len(q)) or len(q) == 0:
        raise ValueError("frame_ids, timestamps, and q must be non-empty and equal length")
    if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if not np.all(np.isfinite(q)):
        raise ValueError("q must be finite")
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("frame IDs must be unique")


def decompose_motion_states(frame_ids: Iterable[int], timestamps: Iterable[float],
                            q: Iterable[float], cfg: dict) -> list[FrameMotionState]:
    """Classify frames from frozen q without inventing ownership on plateaus."""
    frame_ids = np.asarray(list(frame_ids), np.int64)
    timestamps = np.asarray(list(timestamps), float)
    q = np.asarray(list(q), float)
    _validate(frame_ids, timestamps, q)
    edge_order = int(cfg.get("velocity_edge_order", 1))
    velocity = np.gradient(q, timestamps, edge_order=edge_order) if len(q) > 1 else np.zeros_like(q)
    active = np.abs(velocity) >= float(cfg["active_velocity_threshold"])
    radius = int(cfg.get("active_dilation_frames", 0))
    if radius:
        active = np.convolve(active.astype(np.uint8), np.ones(2 * radius + 1, np.uint8), mode="same") > 0
    q_min, q_max = float(q.min()), float(q.max())
    closed_is_min = bool(cfg.get("closed_is_minimum_q", True))
    closed_q, open_q = (q_min, q_max) if closed_is_min else (q_max, q_min)
    tolerance = float(cfg["plateau_q_tolerance"])
    rows = []
    for index in range(len(q)):
        if active[index]:
            state, reason = MotionState.ACTIVE_MOTION, "frozen_q_velocity_exceeds_threshold"
        elif abs(q[index] - closed_q) <= tolerance:
            state, reason = MotionState.CLOSED_PLATEAU, "near_closed_extremum_and_not_active"
        elif abs(q[index] - open_q) <= tolerance:
            state, reason = MotionState.OPEN_PLATEAU, "near_open_extremum_and_not_active"
        else:
            # A stopped intermediate state has no motion evidence. Assign the nearest
            # plateau for bookkeeping, while preserving the explicit reason.
            if abs(q[index] - closed_q) <= abs(q[index] - open_q):
                state = MotionState.CLOSED_PLATEAU
            else:
                state = MotionState.OPEN_PLATEAU
            reason = "inactive_intermediate_assigned_nearest_plateau_no_ownership_creation"
        rows.append(FrameMotionState(int(frame_ids[index]), float(timestamps[index]), float(q[index]),
                                     float(velocity[index]), state, reason))
    return rows


def build_active_transitions(states: list[FrameMotionState], cfg: dict) -> list[ActiveTransition]:
    """Build transitions only when the interval traverses observed ACTIVE_MOTION."""
    minimum_delta_q = float(cfg["minimum_transition_delta_q"])
    maximum_gap = int(cfg["maximum_transition_frame_gap"])
    transitions = []
    for source in range(len(states)):
        stop = min(len(states), source + maximum_gap + 1)
        for target in range(source + 1, stop):
            delta_q = states[target].q - states[source].q
            traversed = any(row.state == MotionState.ACTIVE_MOTION for row in states[source:target + 1])
            if abs(delta_q) < minimum_delta_q or not traversed:
                continue
            transitions.append(ActiveTransition(
                source, target, states[source].frame_id, states[target].frame_id,
                states[source].q, states[target].q, float(delta_q),
                float(states[target].timestamp - states[source].timestamp), True))
    return transitions
