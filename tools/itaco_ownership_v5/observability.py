"""Finite-boundary observability checks for tangent-motion ambiguity."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from tools.itaco_region_assignment_v4.surface_identity_visibility import canonicalize_normal


def source_finite_surface_observability(frame: dict, proposal: dict, points_world: np.ndarray,
                                        axis_world: np.ndarray, cfg: dict) -> dict:
    """Report whether proposal boundaries are physical or observation-clipped.

    A planar surface whose articulation direction is tangent remains occupancy-
    ambiguous unless finite boundaries are actually observable. Proposal edges
    coincident with the RGB footprint are observation truncation, not a physical
    surface anchor.
    """
    points = np.asarray(points_world, float).reshape(-1, 3)
    axis = np.asarray(axis_world, float); axis /= np.linalg.norm(axis)
    if len(points) < 3:
        return {"valid": False, "normal_axis_alignment": None, "axis_extent_m": None,
                "boundary_point_count": 0, "observation_clipped_boundary_fraction": 1.0,
                "finite_boundary_confidence": 0.0, "tangent_motion_ambiguity": True,
                "reason": "insufficient_source_geometry"}
    centered = points - np.median(points, axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered / max(len(points) - 1, 1))
    normal = canonicalize_normal(vectors[:, int(np.argmin(values))])
    alignment = float(abs(np.dot(normal, axis)))
    extent = float(np.percentile(points @ axis, 95) - np.percentile(points @ axis, 5))
    mask = np.asarray(proposal["eroded_mask"], bool)
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    boundary = mask & ~eroded
    footprint_distance = distance_transform_edt(np.asarray(frame["rgb_footprint"], bool))
    margin = float(cfg["observation_boundary_margin_pixels"])
    clipped = float(np.mean(footprint_distance[boundary] <= margin)) if np.any(boundary) else 1.0
    confidence = 1.0 - clipped
    tangent = alignment < float(cfg["minimum_normal_axis_alignment_for_non_tangent_motion"])
    ambiguity = bool(tangent and confidence < float(cfg["minimum_finite_boundary_confidence"]))
    return {"valid": True, "normal_axis_alignment": alignment, "axis_extent_m": extent,
            "boundary_point_count": int(boundary.sum()),
            "observation_clipped_boundary_fraction": clipped,
            "finite_boundary_confidence": confidence, "tangent_motion": bool(tangent),
            "tangent_motion_ambiguity": ambiguity,
            "reason": ("tangent_motion_without_observable_finite_boundary" if ambiguity else
                       "finite_boundary_or_non_tangent_motion_observable")}
