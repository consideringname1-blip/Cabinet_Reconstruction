"""One moving-map implementation shared by training and every output consumer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MovingMapResult:
    raw_score: torch.Tensor
    gated_score: torch.Tensor
    proposal_coverage: torch.Tensor
    static_mask: torch.Tensor
    unknown_mask: torch.Tensor
    moving_mask: torch.Tensor
    invalid_mask: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "raw_score": self.raw_score,
            "gated_score": self.gated_score,
            "proposal_coverage": self.proposal_coverage,
            "static_mask": self.static_mask,
            "unknown_mask": self.unknown_mask,
            "moving_mask": self.moving_mask,
            "invalid_mask": self.invalid_mask,
        }


def _frame_minmax(score: torch.Tensor) -> torch.Tensor:
    minimum = torch.amin(score, dim=(-2, -1), keepdim=True)
    maximum = torch.amax(score, dim=(-2, -1), keepdim=True)
    return (score - minimum) / (maximum - minimum + 1e-12)


def build_moving_map(
    part_segments: torch.Tensor,
    moving_parameters: torch.Tensor,
    valid_support: torch.Tensor,
    mode: str,
    *,
    static_threshold: float = 0.30,
    moving_threshold: float = 0.70,
    overlap_aggregation: str = "mean",
) -> dict[str, torch.Tensor]:
    """Build scores and an exhaustive static/unknown/moving/invalid partition.

    ``moving_parameters`` are free proposal scalars in official/gate_only mode
    and proposal logits in gate_no_minmax mode.
    """
    if mode not in {"official", "gate_only", "gate_no_minmax"}:
        raise ValueError(f"Unsupported moving-map mode: {mode}")
    if part_segments.ndim != 4:
        raise ValueError("part_segments must have shape [N,P,H,W]")
    if moving_parameters.ndim != 1 or moving_parameters.shape[0] != part_segments.shape[1]:
        raise ValueError("moving_parameters must have shape [P]")
    if valid_support.shape != part_segments.shape[0:1] + part_segments.shape[2:]:
        raise ValueError("valid_support must have shape [N,H,W]")
    if not 0 <= static_threshold < moving_threshold <= 1:
        raise ValueError("Expected 0 <= static_threshold < moving_threshold <= 1")

    masks = part_segments.to(dtype=torch.float64)
    coverage_count = masks.sum(dim=1)
    proposal_coverage = coverage_count > 0

    if mode == "gate_no_minmax":
        proposal_scores = torch.sigmoid(moving_parameters)
        raw_score = (masks * proposal_scores.reshape(1, -1, 1, 1)).sum(dim=1)
        if overlap_aggregation == "mean":
            score = raw_score / coverage_count.clamp_min(1.0)
        elif overlap_aggregation == "max":
            expanded = proposal_scores.reshape(1, -1, 1, 1).expand_as(masks)
            score = torch.where(
                masks > 0,
                expanded,
                torch.full_like(expanded, -torch.inf),
            ).amax(dim=1)
            score = torch.where(proposal_coverage, score, torch.zeros_like(score))
        elif overlap_aggregation == "clamp":
            score = raw_score.clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unsupported overlap aggregation: {overlap_aggregation}")
        if torch.any(score < 0) or torch.any(score > 1):
            raise AssertionError("bounded_moving_score escaped [0,1]")
    else:
        raw_score = (masks * moving_parameters.reshape(1, -1, 1, 1)).sum(dim=1)
        score = _frame_minmax(raw_score)

    support = valid_support.to(dtype=torch.bool)
    gated_score = torch.where(support, score, torch.zeros_like(score))
    invalid = ~support
    moving = support & proposal_coverage & (gated_score >= moving_threshold)
    static = support & proposal_coverage & (gated_score <= static_threshold)
    unknown = support & (
        (~proposal_coverage)
        | ((gated_score > static_threshold) & (gated_score < moving_threshold))
    )

    labels_sum = (
        static.to(torch.uint8)
        + unknown.to(torch.uint8)
        + moving.to(torch.uint8)
        + invalid.to(torch.uint8)
    )
    if not torch.all(labels_sum == 1):
        raise AssertionError("Moving-map labels are not mutually exclusive and complete")
    return MovingMapResult(
        raw_score=raw_score,
        gated_score=gated_score,
        proposal_coverage=proposal_coverage,
        static_mask=static,
        unknown_mask=unknown,
        moving_mask=moving,
        invalid_mask=invalid,
    ).as_dict()
