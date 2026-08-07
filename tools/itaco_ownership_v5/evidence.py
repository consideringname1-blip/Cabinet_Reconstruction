"""Independent physical evidence-family bookkeeping and conservative decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvidenceFamily(str, Enum):
    ARTICULATED_MOTION_TRANSITION = "articulated_motion_transition"
    CAUSAL_OCCLUSION_DISOCCLUSION = "causal_occlusion_disocclusion"
    FREE_SPACE_CONTRADICTION = "free_space_contradiction"
    TRUSTED_IDENTITY_PROPAGATION = "trusted_identity_propagation"


class Decision(str, Enum):
    MOVING = "MOVING"
    STATIC = "STATIC"
    UNKNOWN = "UNKNOWN"
    CONFLICTING = "CONFLICTING"


@dataclass
class EvidenceBook:
    records: dict[EvidenceFamily, dict] = field(default_factory=dict)

    def add(self, family: EvidenceFamily, supports: str, provenance: str,
            metrics: dict | None = None, active_transition: bool = False) -> None:
        if supports not in ("moving", "static", "neither"):
            raise ValueError("supports must be moving, static, or neither")
        if family in self.records:
            previous = self.records[family]
            if previous["supports"] != supports:
                previous["supports"] = "neither"
                previous["within_family_conflict"] = True
            previous["metrics"].update(metrics or {})
            previous["provenance"].append(provenance)
            previous["active_transition"] = previous["active_transition"] or active_transition
            return
        self.records[family] = {"supports": supports, "provenance": [provenance],
                                "metrics": dict(metrics or {}), "active_transition": bool(active_transition),
                                "within_family_conflict": False}

    def decide(self) -> dict:
        moving = []
        static = []
        for family, record in self.records.items():
            # Every originating physical family must traverse ACTIVE_MOTION. Trusted
            # propagation may later carry an identity, but only when another active
            # physical family already established that same ownership.
            eligible = record["supports"] in ("moving", "static")
            if family != EvidenceFamily.TRUSTED_IDENTITY_PROPAGATION:
                eligible &= record["active_transition"]
            else:
                eligible &= any(other != family and row["supports"] == record["supports"]
                                and row["active_transition"]
                                for other, row in self.records.items())
            if not eligible:
                continue
            (moving if record["supports"] == "moving" else static).append(family.value)
        if moving and static:
            decision = Decision.CONFLICTING
            reason = "independent_physical_families_disagree"
        elif moving:
            decision = Decision.MOVING
            reason = "positive_moving_physical_evidence_no_static_conflict"
        elif static:
            decision = Decision.STATIC
            reason = "positive_static_physical_evidence_no_moving_conflict"
        else:
            decision = Decision.UNKNOWN
            reason = "insufficient_positive_physical_evidence"
        return {"decision": decision.value, "reason": reason,
                "moving_families": moving, "static_families": static,
                "records": {family.value: row for family, row in self.records.items()}}
