"""Sensor-support construction without sequence-specific semantic rules."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def build_sensor_support(
    projected_rgb_footprint: np.ndarray,
    depth_valid: np.ndarray,
    original_hand_mask: np.ndarray,
) -> np.ndarray:
    arrays = [
        np.asarray(projected_rgb_footprint, dtype=bool),
        np.asarray(depth_valid, dtype=bool),
        np.asarray(original_hand_mask, dtype=bool),
    ]
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("RGB footprint, depth validity and hand mask shapes differ")
    return arrays[0] & arrays[1] & (~arrays[2])


def build_coarse_static_mask(
    dynamic_mask: np.ndarray,
    obj_mask: np.ndarray,
    projected_rgb_footprint: np.ndarray,
    depth_valid: np.ndarray,
) -> np.ndarray:
    arrays = [
        np.asarray(dynamic_mask, dtype=bool),
        np.asarray(obj_mask, dtype=bool),
        np.asarray(projected_rgb_footprint, dtype=bool),
        np.asarray(depth_valid, dtype=bool),
    ]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("Coarse masks have inconsistent shapes")
    return (~arrays[0]) & arrays[1] & arrays[2] & arrays[3]


def _kernel(radius: int) -> np.ndarray:
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)


def load_projected_rgb_footprints(
    paths: list[Path],
    *,
    close_radius: int,
    erosion_radius: int,
    min_contour_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    exact, footprint = [], []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        nonzero = np.any(image != 0, axis=2).astype(np.uint8)
        closed = cv2.morphologyEx(nonzero, cv2.MORPH_CLOSE, _kernel(close_radius))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(nonzero)
        retained = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        cv2.drawContours(filled, retained, -1, 1, thickness=-1)
        if erosion_radius:
            filled = cv2.erode(filled, _kernel(erosion_radius), iterations=1)
        exact.append(nonzero.astype(bool))
        footprint.append(filled.astype(bool))
    return np.stack(exact), np.stack(footprint)


def load_depth_validity(
    paths: list[Path],
    *,
    minimum_m: float,
    maximum_m: float,
    erosion_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw, valid = [], []
    for path in paths:
        depth = np.load(path)
        current = np.isfinite(depth) & (depth > minimum_m) & (depth < maximum_m)
        raw.append(current)
        if erosion_radius:
            current = cv2.erode(
                current.astype(np.uint8), _kernel(erosion_radius), iterations=1
            ).astype(bool)
        valid.append(current)
    return np.stack(raw), np.stack(valid)


def load_original_hand_masks(hand_dir: Path, source_indices: list[int]) -> np.ndarray:
    masks = []
    for source_index in source_indices:
        path = hand_dir / f"{source_index:06d}.npy"
        masks.append(np.load(path).squeeze().astype(bool))
    return np.stack(masks)
