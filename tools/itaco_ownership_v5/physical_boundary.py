"""Depth- and geometry-supported finite-boundary observability."""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from tools.itaco_region_assignment_v4.projective_models import unproject_pixels


BOUNDARY_KINDS = (
    "rgb_footprint_truncation",
    "depth_footprint_truncation",
    "depth_discontinuity",
    "normal_or_plane_discontinuity",
    "invalid_depth_hole",
    "segmentation_only",
)


def _external_invalid(depth_valid: np.ndarray) -> np.ndarray:
    invalid = (~np.asarray(depth_valid, bool)).astype(np.uint8)
    count, labels = cv2.connectedComponents(invalid)
    exterior_labels = set(np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]]).tolist())
    exterior_labels.discard(0)
    return np.isin(labels, list(exterior_labels)) if exterior_labels else np.zeros_like(invalid, bool)


def _normal_map(depth: np.ndarray, valid: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    uv = np.column_stack((xx.ravel(), yy.ravel()))
    z = depth.ravel()
    camera = np.column_stack(((uv[:, 0] - intrinsic[0, 2]) * z / intrinsic[0, 0],
                              (uv[:, 1] - intrinsic[1, 2]) * z / intrinsic[1, 1], z)).reshape(h, w, 3)
    dx = np.zeros_like(camera); dy = np.zeros_like(camera)
    dx[:, 1:-1] = camera[:, 2:] - camera[:, :-2]
    dy[1:-1] = camera[2:] - camera[:-2]
    normal = np.cross(dx, dy)
    norm = np.linalg.norm(normal, axis=2)
    normal_valid = np.asarray(valid, bool) & np.isfinite(normal).all(axis=2) & (norm > 1e-9)
    normal[normal_valid] /= norm[normal_valid, None]
    normal[~normal_valid] = np.nan
    return normal, normal_valid


def analyze_physical_boundary(
    frame: dict,
    region_mask: np.ndarray,
    source_world: np.ndarray,
    source_uv: np.ndarray,
    axis_world: np.ndarray,
    intrinsic: np.ndarray,
    cfg: dict,
) -> dict:
    """Classify observed mask boundaries; AutoSeg alone is never physical proof."""
    mask = np.asarray(region_mask, bool)
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    boundary = mask & ~eroded
    yy, xx = np.nonzero(boundary)
    n = len(xx)
    category = np.full(n, "segmentation_only", object)
    depth = np.asarray(frame["depth"], float)
    depth_valid = np.asarray(frame["depth_valid"], bool)
    rgb_valid = np.asarray(frame["rgb_footprint"], bool)
    rgb_distance = distance_transform_edt(rgb_valid)
    exterior_invalid = _external_invalid(depth_valid)
    exterior_distance = distance_transform_edt(~exterior_invalid)
    hole_invalid = ~depth_valid & ~exterior_invalid
    hole_distance = distance_transform_edt(~hole_invalid)
    normal_map, normal_valid = _normal_map(depth, depth_valid, intrinsic)
    signed = distance_transform_edt(mask) - distance_transform_edt(~mask)
    gx = cv2.Sobel(signed.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(signed.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.maximum(np.hypot(gx[yy, xx], gy[yy, xx]), 1e-6)
    direction_x = gx[yy, xx] / magnitude
    direction_y = gy[yy, xx] / magnitude
    offset = int(cfg["cross_boundary_sample_offset_pixels"])
    inside_x = np.rint(xx + direction_x * offset).astype(int)
    inside_y = np.rint(yy + direction_y * offset).astype(int)
    outside_x = np.rint(xx - direction_x * offset).astype(int)
    outside_y = np.rint(yy - direction_y * offset).astype(int)
    h, w = mask.shape
    pair_inside = ((inside_x >= 0) & (inside_x < w) & (inside_y >= 0) & (inside_y < h) &
                   (outside_x >= 0) & (outside_x < w) & (outside_y >= 0) & (outside_y < h))
    depth_edge = np.zeros(n, bool)
    normal_edge = np.zeros(n, bool)
    valid_pair = np.zeros(n, bool)
    ids = np.flatnonzero(pair_inside)
    if len(ids):
        valid_pair[ids] = (depth_valid[inside_y[ids], inside_x[ids]] &
                           depth_valid[outside_y[ids], outside_x[ids]])
        good = ids[valid_pair[ids]]
        if len(good):
            jump = np.abs(depth[inside_y[good], inside_x[good]] - depth[outside_y[good], outside_x[good]])
            depth_edge[good] = jump >= float(cfg["minimum_depth_jump_m"])
            normal_good = (normal_valid[inside_y[good], inside_x[good]] &
                           normal_valid[outside_y[good], outside_x[good]])
            ng = good[normal_good]
            if len(ng):
                dot = np.abs(np.sum(normal_map[inside_y[ng], inside_x[ng]] *
                                    normal_map[outside_y[ng], outside_x[ng]], axis=1))
                angle = np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))
                plane_offset = np.abs(depth[inside_y[ng], inside_x[ng]] - depth[outside_y[ng], outside_x[ng]])
                normal_edge[ng] = ((angle >= float(cfg["minimum_normal_jump_degrees"])) |
                                   (plane_offset >= float(cfg["minimum_plane_offset_m"])))
    margin = float(cfg["observation_boundary_margin_pixels"])
    rgb_clipped = rgb_distance[yy, xx] <= margin
    depth_clipped = ~rgb_clipped & (exterior_distance[yy, xx] <= margin)
    hole = ~rgb_clipped & ~depth_clipped & (hole_distance[yy, xx] <= margin) & ~valid_pair
    category[rgb_clipped] = "rgb_footprint_truncation"
    category[depth_clipped] = "depth_footprint_truncation"
    category[hole] = "invalid_depth_hole"
    eligible = ~rgb_clipped & ~depth_clipped & ~hole
    category[eligible & depth_edge] = "depth_discontinuity"
    category[eligible & ~depth_edge & normal_edge] = "normal_or_plane_discontinuity"
    physical = np.isin(category, ["depth_discontinuity", "normal_or_plane_discontinuity"])
    points = np.asarray(source_world, float).reshape(-1, 3)
    axis = np.asarray(axis_world, float); axis /= np.linalg.norm(axis)
    axis_coordinate = points @ axis if len(points) else np.empty(0)
    axis_extent = (float(np.percentile(axis_coordinate, 95) - np.percentile(axis_coordinate, 5))
                   if len(axis_coordinate) else 0.0)
    # Boundary world points establish leading/trailing location in the same frame.
    valid_boundary = boundary & np.asarray(frame["valid"], bool)
    boundary_uv = np.column_stack((xx, yy))
    boundary_world = np.empty((0, 3), float)
    boundary_axis = np.empty(0, float)
    valid_boundary_ids = np.flatnonzero(valid_boundary[yy, xx])
    if len(valid_boundary_ids):
        buv = boundary_uv[valid_boundary_ids]
        boundary_world = unproject_pixels(
            buv, depth[buv[:, 1], buv[:, 0]], frame["pose"], intrinsic)
        boundary_axis = boundary_world @ axis
    low = float(np.percentile(boundary_axis, 20)) if len(boundary_axis) else 0.0
    high = float(np.percentile(boundary_axis, 80)) if len(boundary_axis) else 0.0
    leading = np.zeros(n, bool); trailing = np.zeros(n, bool)
    if len(valid_boundary_ids):
        trailing[valid_boundary_ids] = boundary_axis <= low
        leading[valid_boundary_ids] = boundary_axis >= high

    def edge_confidence(selected: np.ndarray) -> float:
        if not np.any(selected):
            return 0.0
        return float(np.mean(physical[selected]) * (1.0 - np.mean(np.isin(
            category[selected], ["rgb_footprint_truncation", "depth_footprint_truncation", "invalid_depth_hole"]))))

    # Estimate surface normal from source finite geometry only for tangent diagnosis.
    if len(points) >= 3:
        centered = points - np.median(points, axis=0)
        _, vectors = np.linalg.eigh(centered.T @ centered / max(len(points) - 1, 1))
        surface_normal = vectors[:, 0]
        normal_axis_alignment = float(abs(np.dot(surface_normal, axis)))
    else:
        normal_axis_alignment = None
    tangent = (normal_axis_alignment is None or
               normal_axis_alignment < float(cfg["maximum_normal_axis_alignment_for_tangent"]))
    counts = Counter(category.tolist())
    total = max(n, 1)
    leading_confidence = edge_confidence(leading)
    trailing_confidence = edge_confidence(trailing)
    return {
        "valid": bool(n and len(points) >= int(cfg["minimum_source_geometry_points"])),
        "boundary_point_count": int(n),
        "category_counts": {name: int(counts[name]) for name in BOUNDARY_KINDS},
        "depth_edge_fraction": float(np.mean(category == "depth_discontinuity")) if n else 0.0,
        "normal_edge_fraction": float(np.mean(category == "normal_or_plane_discontinuity")) if n else 0.0,
        "observation_clipped_fraction": float(np.mean(np.isin(category, [
            "rgb_footprint_truncation", "depth_footprint_truncation"]))) if n else 1.0,
        "invalid_hole_fraction": float(counts["invalid_depth_hole"] / total),
        "segmentation_only_fraction": float(counts["segmentation_only"] / total),
        "physical_boundary_fraction": float(np.mean(physical)) if n else 0.0,
        "leading_edge_physical_confidence": leading_confidence,
        "trailing_edge_physical_confidence": trailing_confidence,
        "axis_extent_m": axis_extent,
        "normal_axis_alignment": normal_axis_alignment,
        "tangent_motion": bool(tangent),
        "has_trusted_axis_finite_edge": bool(
            max(leading_confidence, trailing_confidence) >= float(cfg["minimum_axis_edge_physical_confidence"])),
        "arrays": {
            "boundary_uv": boundary_uv,
            "category": category,
            "physical": physical,
            "leading": leading,
            "trailing": trailing,
        },
    }


def serializable_boundary(report: dict) -> dict:
    return {key: value for key, value in report.items() if key != "arrays"}
