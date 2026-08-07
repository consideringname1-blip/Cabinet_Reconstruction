"""Conservative physical-event ownership primitives for Assignment v5.

This package deliberately stops before AutoSeg/SAM2 integration, dense ownership,
object-scope filtering, and reconstruction.
"""

from .evidence import Decision, EvidenceBook, EvidenceFamily
from .identity import verify_anchored_surface_sequence
from .motion import MotionState, build_active_transitions, decompose_motion_states
from .occlusion import detect_causal_static_disocclusion
from .transforms import ArticulatedTransform

__all__ = [
    "ArticulatedTransform",
    "Decision",
    "EvidenceBook",
    "EvidenceFamily",
    "MotionState",
    "build_active_transitions",
    "decompose_motion_states",
    "detect_causal_static_disocclusion",
    "verify_anchored_surface_sequence",
]
