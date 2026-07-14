"""Canonical Shigure collider boxes for the HoloLens wireframe protocol.

The upstream ``Cube`` is defined by a minimum corner and three positive sizes
in millimetres in Shigure camera coordinates.  It is deliberately *not*
combined with masks or depth here.  The camera-aligned eight corners are first
transformed to ArUco coordinates, then an ArUco-axis-aligned box is rebuilt so
that its front/back faces are parallel to the ArUco marker plane.

All matrices in this module use the conventional column-vector form::

    point_destination_h = destination_from_source @ point_source_h

Invalid input is a hard ``NO_BOX`` result; this module has no geometric
fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np


SCHEMA: Final[str] = "shigure_spatial_box_v2"
STATUS_OK: Final[str] = "OK"
STATUS_NO_BOX: Final[str] = "NO_BOX"

# Bits are ArUco-axis choices after canonicalization: x, y, z respectively.
CORNER_ORDER: Final[tuple[str, ...]] = (
    "000",
    "100",
    "110",
    "010",
    "001",
    "101",
    "111",
    "011",
)

# Two z-constant faces followed by the four connecting edges.
WIREFRAME_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
WIREFRAME_DIAMETER_M: Final[float] = 0.005

_COLLIDER_KEYS: Final[tuple[str, ...]] = (
    "x",
    "y",
    "z",
    "width",
    "height",
    "depth",
)
_CORNER_BITS = np.asarray(
    [[int(bit) for bit in label] for label in CORNER_ORDER],
    dtype=np.float64,
)
_MILLIMETRES_PER_METRE: Final[float] = 1000.0
_TRANSFORM_ATOL: Final[float] = 1.0e-6
_MIN_EXTENT_M: Final[float] = 1.0e-9


class NoBoxError(ValueError):
    """A strict spatial-box validation failure.

    ``status`` is fixed to ``NO_BOX`` so API boundaries can serialize the
    failure without parsing exception text.
    """

    status: Final[str] = STATUS_NO_BOX

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f"{self.status}: {self.reason}: {self.detail}")

    def as_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "schema": SCHEMA,
            "reason": self.reason,
            "detail": self.detail,
        }


def _collider_min_and_size_m(collider_mm: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    if collider_mm is None:
        raise NoBoxError("COLLIDER_MISSING", "collider_mm is required")
    if not isinstance(collider_mm, Mapping):
        raise NoBoxError("COLLIDER_INVALID", "collider_mm must be a mapping")

    missing = [key for key in _COLLIDER_KEYS if key not in collider_mm]
    if missing:
        raise NoBoxError("COLLIDER_INVALID", f"missing fields: {', '.join(missing)}")
    try:
        values = np.asarray([collider_mm[key] for key in _COLLIDER_KEYS], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NoBoxError("COLLIDER_INVALID", "all collider fields must be numeric") from exc
    if values.shape != (6,) or not np.isfinite(values).all():
        raise NoBoxError("COLLIDER_NONFINITE", "all collider fields must be finite")

    minimum_m = values[:3] / _MILLIMETRES_PER_METRE
    size_m = values[3:] / _MILLIMETRES_PER_METRE
    if np.any(size_m <= _MIN_EXTENT_M):
        raise NoBoxError("COLLIDER_DEGENERATE", "width, height, and depth must be positive")
    return minimum_m, size_m


def _rigid_or_reflective_transform(value: Any, field_name: str) -> np.ndarray:
    """Validate an affine Euclidean transform, including handedness changes.

    Camera-to-ArUco can contain the OpenCV-to-Unity basis reflection, so both
    determinant signs are valid.  Scale, shear, singular matrices, perspective
    rows, and non-finite matrices are rejected.
    """

    if value is None:
        raise NoBoxError(f"{field_name.upper()}_MISSING", f"{field_name} is required")
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NoBoxError(f"{field_name.upper()}_INVALID", f"{field_name} must be numeric") from exc
    if transform.shape != (4, 4):
        raise NoBoxError(f"{field_name.upper()}_INVALID", f"{field_name} must be 4x4")
    if not np.isfinite(transform).all():
        raise NoBoxError(f"{field_name.upper()}_NONFINITE", f"{field_name} must be finite")
    if not np.allclose(
        transform[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=_TRANSFORM_ATOL,
    ):
        raise NoBoxError(
            f"{field_name.upper()}_INVALID",
            f"{field_name} must have affine bottom row [0, 0, 0, 1]",
        )

    basis = transform[:3, :3]
    gram = basis.T @ basis
    determinant = float(np.linalg.det(basis))
    if not np.allclose(gram, np.eye(3), rtol=0.0, atol=_TRANSFORM_ATOL) or not np.isclose(
        abs(determinant),
        1.0,
        rtol=0.0,
        atol=_TRANSFORM_ATOL,
    ):
        raise NoBoxError(
            f"{field_name.upper()}_INVALID",
            f"{field_name} must be an orthogonal Euclidean transform without scale or shear",
        )
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray, field_name: str) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed_h = (transform @ homogeneous.T).T
    if not np.isfinite(transformed_h).all():
        raise NoBoxError("TRANSFORM_RESULT_NONFINITE", f"{field_name} produced non-finite points")
    if not np.allclose(transformed_h[:, 3], 1.0, rtol=0.0, atol=_TRANSFORM_ATOL):
        raise NoBoxError("TRANSFORM_RESULT_INVALID", f"{field_name} produced invalid homogeneous w")
    return transformed_h[:, :3]


def _ordered_corners(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    return minimum.reshape(1, 3) + (_CORNER_BITS * (maximum - minimum).reshape(1, 3))


def build_spatial_box_v2(
    collider_mm: Mapping[str, Any] | None,
    camera_to_aruco: Any,
    aruco_to_hololens_current_local: Any,
) -> dict[str, Any]:
    """Build a canonical ArUco-aligned wireframe box.

    Raises:
        NoBoxError: if the collider, either transform, or a computed extent is
            missing, degenerate, non-finite, or otherwise invalid.
    """

    minimum_camera_m, size_camera_m = _collider_min_and_size_m(collider_mm)
    camera_from = _rigid_or_reflective_transform(camera_to_aruco, "camera_to_aruco")
    hololens_from = _rigid_or_reflective_transform(
        aruco_to_hololens_current_local,
        "aruco_to_hololens_current_local",
    )

    # Preserve the upstream camera box exactly for the first transform.  Do not
    # infer geometry from an image mask or depth frame.
    corners_camera_m = _ordered_corners(
        minimum_camera_m,
        minimum_camera_m + size_camera_m,
    )
    transformed_corners_aruco_m = _transform_points(
        corners_camera_m,
        camera_from,
        "camera_to_aruco",
    )

    # This is the intentional orientation canonicalization: front/back are the
    # z-min/z-max planes, hence parallel to the ArUco marker's x/y plane.
    minimum_aruco_m = np.min(transformed_corners_aruco_m, axis=0)
    maximum_aruco_m = np.max(transformed_corners_aruco_m, axis=0)
    extent_aruco_m = maximum_aruco_m - minimum_aruco_m
    if not np.isfinite(extent_aruco_m).all():
        raise NoBoxError("CANONICAL_BOX_NONFINITE", "ArUco extents are non-finite")
    if np.any(extent_aruco_m <= _MIN_EXTENT_M):
        raise NoBoxError("CANONICAL_BOX_DEGENERATE", "ArUco extents must all be positive")

    corners_aruco_m = _ordered_corners(minimum_aruco_m, maximum_aruco_m)
    corners_hololens_m = _transform_points(
        corners_aruco_m,
        hololens_from,
        "aruco_to_hololens_current_local",
    )

    return {
        "status": STATUS_OK,
        "schema": SCHEMA,
        "source": "shigure_object_tracking.collider",
        "source_units": "mm",
        "corner_order": list(CORNER_ORDER),
        "corners_aruco_m": corners_aruco_m.astype(float).tolist(),
        "corners_hololens_current_local_m": corners_hololens_m.astype(float).tolist(),
        "wireframe": {
            "edges": [list(edge) for edge in WIREFRAME_EDGES],
            "diameter_m": WIREFRAME_DIAMETER_M,
            "filled": False,
        },
        "orientation": {
            "basis": "aruco_axes",
            "front_back_faces": "parallel_to_aruco_marker",
            "front_back_axis": "aruco_z",
        },
    }


def spatial_box_v2_result(
    collider_mm: Mapping[str, Any] | None,
    camera_to_aruco: Any,
    aruco_to_hololens_current_local: Any,
) -> dict[str, Any]:
    """Serialize strict box validation as either ``OK`` or ``NO_BOX``."""

    try:
        return build_spatial_box_v2(
            collider_mm,
            camera_to_aruco,
            aruco_to_hololens_current_local,
        )
    except NoBoxError as exc:
        return exc.as_payload()


__all__ = [
    "CORNER_ORDER",
    "NoBoxError",
    "SCHEMA",
    "STATUS_NO_BOX",
    "STATUS_OK",
    "WIREFRAME_DIAMETER_M",
    "WIREFRAME_EDGES",
    "build_spatial_box_v2",
    "spatial_box_v2_result",
]
