"""Source-indexed ACTIVE-motion observations spanning multiple q baselines.

The target of an observation is a camera/depth frame, never a target proposal.
Both chronological directions are valid for motion discrimination.  Causality
is deliberately absent from this module and remains forward-only elsewhere.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .local_transitions import LocalFrameMotionState, LocalMotionState


@dataclass(frozen=True)
class MultiBaselineObservation:
    source_frame: int
    target_frame: int
    source_q: float
    target_q: float
    delta_q: float
    abs_delta_q: float
    elapsed_s: float
    frame_gap: int
    direction: str
    baseline: str
    interval_frames: tuple[int, ...]
    interval_states: tuple[str, ...]
    valid: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _baseline_for_delta(delta: float, cfg: dict) -> str | None:
    value = abs(float(delta))
    for name in ("short", "medium", "long"):
        limits = cfg[name]
        if value >= float(limits["minimum_delta_q_m"]) and value <= float(limits["maximum_delta_q_m"]):
            return name
    return None


def build_multi_baseline_observations(
    states: list[LocalFrameMotionState], cfg: dict
) -> tuple[list[MultiBaselineObservation], list[MultiBaselineObservation]]:
    """Build directed source-to-frame observations across the full ACTIVE run."""
    accepted: list[MultiBaselineObservation] = []
    rejected: list[MultiBaselineObservation] = []
    require_active = bool(cfg.get("require_all_interval_frames_active", True))
    allow_backward = bool(cfg.get("allow_backward_motion_discrimination", True))
    for source_index, source in enumerate(states):
        if source.state != LocalMotionState.ACTIVE_MOTION:
            continue
        for target_index, target in enumerate(states):
            if source_index == target_index or target.state != LocalMotionState.ACTIVE_MOTION:
                continue
            if target_index < source_index and not allow_backward:
                continue
            left, right = sorted((source_index, target_index))
            interval = states[left:right + 1]
            delta = float(target.q - source.q)
            baseline = _baseline_for_delta(delta, cfg)
            failures = []
            if baseline is None:
                failures.append("delta_q_outside_frozen_baseline_bins")
            if require_active and any(row.state != LocalMotionState.ACTIVE_MOTION for row in interval):
                failures.append("interval_contains_non_active_motion")
            elapsed = float(target.timestamp - source.timestamp)
            direction = "forward" if elapsed > 0 else "backward_identity_confirmation"
            row = MultiBaselineObservation(
                source_frame=int(source.frame_id), target_frame=int(target.frame_id),
                source_q=float(source.q), target_q=float(target.q), delta_q=delta,
                abs_delta_q=abs(delta), elapsed_s=elapsed,
                frame_gap=abs(int(target.frame_id) - int(source.frame_id)),
                direction=direction, baseline=baseline or "outside_bins",
                interval_frames=tuple(int(item.frame_id) for item in interval),
                interval_states=tuple(item.state.value for item in interval),
                valid=not failures,
                reason=("full_active_interval_and_frozen_baseline_bin"
                        if not failures else ";".join(failures)),
            )
            (accepted if row.valid else rejected).append(row)
    accepted.sort(key=lambda row: (row.source_frame, row.baseline, row.abs_delta_q, row.target_frame))
    rejected.sort(key=lambda row: (row.source_frame, row.target_frame))
    return accepted, rejected


def observations_for_source(
    source_frame: int, observations: list[MultiBaselineObservation]
) -> list[MultiBaselineObservation]:
    return [row for row in observations if row.source_frame == int(source_frame)]
