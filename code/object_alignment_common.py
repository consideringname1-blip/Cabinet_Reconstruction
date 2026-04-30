from __future__ import annotations

import os
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np

from config import (
    AHAT_MAX_RELIABLE_DEPTH_MM,
    AHAT_MIN_DEPTH_MM,
    BLENDER_BIN,
    ICP_DEPTH_BORDER_CROP_RATIO,
    ICP_IGNORE_OCCLUDED_MODEL_POINTS,
    INSTANTMESH_OUTPUT_MESHES,
    OBJECT_ALIGNMENT_OUTPUT_ROOT,
    SAM3_OUTPUT_ROOT,
    UPLOAD_FOLDER,
)
from task_json import (
    load_task_json as load_json,
    resolve_task_json_path as resolve_json_path,
    save_task_json as save_json,
)
from unity_coordinate_utils import (
    CANONICAL_RH_TO_UNITY_BASIS,
)


MIN_DEPTH_MM = AHAT_MIN_DEPTH_MM
MAX_DEPTH_MM = AHAT_MAX_RELIABLE_DEPTH_MM

# Canonical internal basis for measurement / alignment:
# right-handed camera local, X=right, Y=up, -Z=forward.
POINTCLOUD_EXPORT_TO_CANONICAL_RH_BASIS = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

CANONICAL_RH_TO_BLENDER_WORLD = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
BLENDER_WORLD_TO_CANONICAL_RH = CANONICAL_RH_TO_BLENDER_WORLD.T

MODEL_INPUT_TO_CANONICAL_RH_BASIS = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
CANONICAL_RH_TO_MODEL_INPUT_BASIS = MODEL_INPUT_TO_CANONICAL_RH_BASIS.T

# Axis declarations used across the pipeline.
# ICP/debug rendering imports OBJ with `forward=-X`, `up=+Z`.
ICP_OBJ_IMPORT_FORWARD_AXIS = "NEGATIVE_X"
ICP_OBJ_IMPORT_UP_AXIS = "Z"
# Runtime FBX wrapping imports OBJ with Blender's default OBJ convention and
# then exports to a Unity-facing FBX basis.
FBX_CONVERT_OBJ_IMPORT_FORWARD_AXIS = "NEGATIVE_Z"
FBX_CONVERT_OBJ_IMPORT_UP_AXIS = "Y"
FBX_EXPORT_FORWARD_AXIS = "-Z"
FBX_EXPORT_UP_AXIS = "Y"

# OBJ -> Blender world basis used by the ICP/debug path
# (`bpy.ops.wm.obj_import(..., forward_axis="NEGATIVE_X", up_axis="Z")`).
ICP_OBJ_IMPORT_TO_BLENDER_WORLD = CANONICAL_RH_TO_BLENDER_WORLD @ MODEL_INPUT_TO_CANONICAL_RH_BASIS

# OBJ -> Blender world basis used when wrapping the reconstructed OBJ into FBX.
# This mirrors Blender's default OBJ import orientation
# (`forward=-Z`, `up=+Y`) so the conversion stage no longer depends on implicit
# Blender defaults.
FBX_CONVERT_OBJ_IMPORT_TO_BLENDER_WORLD = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# Blender world -> exported FBX local basis used by the runtime file wrapper
# (`axis_forward="-Z", axis_up="Y", bake_space_transform=True`).
BLENDER_WORLD_TO_FBX_EXPORT_LOCAL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)

# Original model-input axes -> runtime FBX local axes. With the explicit OBJ
# import and FBX export settings above, this composes to identity, but we keep
# the full chain here because the pose stage needs to reason about every step.
MODEL_INPUT_TO_FBX_RUNTIME_LOCAL = (
    BLENDER_WORLD_TO_FBX_EXPORT_LOCAL @ FBX_CONVERT_OBJ_IMPORT_TO_BLENDER_WORLD
)
FBX_RUNTIME_LOCAL_TO_MODEL_INPUT = MODEL_INPUT_TO_FBX_RUNTIME_LOCAL.T
# This is a geometry/basis conversion reference, not a transform-space
# rotation. It has determinant -1, so it must not be multiplied directly into a
# runtime world quaternion.
FBX_RUNTIME_LOCAL_TO_UNITY_BASIS = (
    np.asarray(CANONICAL_RH_TO_UNITY_BASIS, dtype=np.float32)
    @ MODEL_INPUT_TO_CANONICAL_RH_BASIS
    @ FBX_RUNTIME_LOCAL_TO_MODEL_INPUT
)
# The current OBJ -> FBX wrapper path bakes axis conversion into the exported
# mesh/file, and Unity/TriLib loads that FBX as a standard runtime object. So
# the transform-space correction that pose composition should apply at runtime
# is identity.
FBX_RUNTIME_TRANSFORM_COMPENSATION_TO_UNITY = np.eye(3, dtype=np.float32)


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def task_prefix(task: dict, json_path: Path) -> str:
    return str(task.get("task_name") or json_path.stem)


def resolve_task_paths(task: dict) -> dict[str, Path]:
    pv_info = task.get("PVCamera") or {}
    sam3_info = task.get("sam3Name") or {}
    mesh_info = task.get("InstantMesh") or {}

    mask_name = sam3_info.get("mask")
    depth_name = sam3_info.get("depth")
    color_name = sam3_info.get("color") or pv_info.get("name")
    mesh_name = mesh_info.get("mesh")

    if not mask_name:
        raise ValueError("sam3Name.mask is missing")
    if not depth_name:
        raise ValueError("sam3Name.depth is missing")
    if not color_name:
        raise ValueError("sam3Name.color and PVCamera.name are both missing")
    if not mesh_name:
        raise ValueError("InstantMesh.mesh is missing")

    color_candidate = (SAM3_OUTPUT_ROOT / color_name).resolve()
    color_path = color_candidate if color_candidate.is_file() else ensure_file(
        (UPLOAD_FOLDER / color_name).resolve(),
        "PVCamera color",
    )

    return {
        "mask_path": ensure_file((SAM3_OUTPUT_ROOT / mask_name).resolve(), "SAM3 mask"),
        "depth_path": ensure_file((SAM3_OUTPUT_ROOT / depth_name).resolve(), "SAM3 depth"),
        "color_path": color_path,
        "mesh_path": ensure_file((INSTANTMESH_OUTPUT_MESHES / mesh_name).resolve(), "InstantMesh mesh"),
    }


def read_depth_image(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    if depth.dtype != np.uint16:
        raise ValueError(f"Depth image must be uint16(mm), got: {depth.dtype}")
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single channel, got shape: {depth.shape}")
    return depth


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask image: {path}")

    if mask.ndim == 2:
        return mask > 0
    if mask.shape[2] == 4:
        return mask[:, :, 3] > 0
    return np.any(mask > 0, axis=2)


def read_color_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read color image: {path}")
    return image


def get_depth_border_crop_ratio() -> float:
    ratio = float(ICP_DEPTH_BORDER_CROP_RATIO)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("config.ICP_DEPTH_BORDER_CROP_RATIO must be in [0.0, 1.0)")
    return ratio


def compute_mask_border_crop(mask_bool: np.ndarray, border_ratio: float | None = None) -> dict:
    mask = np.asarray(mask_bool, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be HxW, got shape {mask.shape}")

    ratio = get_depth_border_crop_ratio() if border_ratio is None else float(border_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("Depth border crop ratio must be in [0.0, 1.0)")

    keep_mask = mask.copy()
    discard_mask = np.zeros_like(mask, dtype=bool)
    threshold_px = 0.0
    min_inside_distance_px = 0.0
    max_inside_distance_px = 0.0
    margin_x = 0
    margin_y = 0

    if np.any(mask):
        distance_px = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        inside_distances = distance_px[mask]
        min_inside_distance_px = float(inside_distances.min())
        max_inside_distance_px = float(inside_distances.max())

        if ratio > 0.0 and max_inside_distance_px > (min_inside_distance_px + 1e-6):
            threshold_px = float(
                min_inside_distance_px
                + ratio * (max_inside_distance_px - min_inside_distance_px)
            )
            keep_mask = mask & (distance_px >= (threshold_px - 1e-6))
            if not np.any(keep_mask):
                keep_mask = mask & (distance_px >= (max_inside_distance_px - 1e-6))

        discard_mask = mask & ~keep_mask
        if np.any(keep_mask):
            x0, y0, x1, y1 = mask_bbox(mask)
            kx0, ky0, kx1, ky1 = mask_bbox(keep_mask)
            margin_x = max(int(round(((kx0 - x0) + (x1 - kx1)) / 2.0)), 0)
            margin_y = max(int(round(((ky0 - y0) + (y1 - ky1)) / 2.0)), 0)

    return {
        "mode": "mask_periphery",
        "keep_mask": keep_mask,
        "discard_mask": discard_mask,
        "threshold_px": float(threshold_px),
        "min_inside_distance_px": float(min_inside_distance_px),
        "max_inside_distance_px": float(max_inside_distance_px),
        "approx_margin_x_px": int(margin_x),
        "approx_margin_y_px": int(margin_y),
    }


def depth_border_crop_margins(
    mask_bool: np.ndarray,
    border_ratio: float | None = None,
) -> tuple[int, int]:
    crop = compute_mask_border_crop(mask_bool, border_ratio=border_ratio)
    return int(crop["approx_margin_x_px"]), int(crop["approx_margin_y_px"])


def build_depth_border_keep_mask(
    mask_bool: np.ndarray,
    border_ratio: float | None = None,
) -> np.ndarray:
    crop = compute_mask_border_crop(mask_bool, border_ratio=border_ratio)
    return np.asarray(crop["keep_mask"], dtype=bool)


def apply_depth_border_crop(mask_bool: np.ndarray, border_ratio: float | None = None) -> np.ndarray:
    mask = np.asarray(mask_bool, dtype=bool)
    return mask & build_depth_border_keep_mask(mask, border_ratio=border_ratio)


def mask_bbox(mask_bool: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        raise ValueError("Mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def compute_real_measurements(mask_bool: np.ndarray, depth_mm: np.ndarray, k: np.ndarray) -> dict:
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    crop_ratio = get_depth_border_crop_ratio()
    crop = compute_mask_border_crop(mask_bool, border_ratio=crop_ratio)
    crop_margin_x = int(crop["approx_margin_x_px"])
    crop_margin_y = int(crop["approx_margin_y_px"])

    x0, y0, x1, y1 = mask_bbox(mask_bool)
    width_px = x1 - x0 + 1
    height_px = y1 - y0 + 1

    usable_mask = np.asarray(crop["keep_mask"], dtype=bool)
    valid_depth = usable_mask & (depth_mm >= MIN_DEPTH_MM) & (depth_mm <= MAX_DEPTH_MM)
    if not np.any(valid_depth):
        raise ValueError("No mask pixels remain within 20-120 cm after mask-border crop")

    mean_depth_m = float(depth_mm[valid_depth].mean()) / 1000.0
    real_width_m = width_px * mean_depth_m / fx
    real_height_m = height_px * mean_depth_m / fy

    return {
        "mask_bbox_xyxy": [x0, y0, x1, y1],
        "width_px": int(width_px),
        "height_px": int(height_px),
        "valid_depth_pixels": int(valid_depth.sum()),
        "mask_pixels": int(mask_bool.sum()),
        "usable_mask_pixels": int(usable_mask.sum()),
        "cropped_mask_pixels": int(mask_bool.sum() - usable_mask.sum()),
        "valid_ratio": float(valid_depth.sum() / max(mask_bool.sum(), 1)),
        "mean_depth_m": mean_depth_m,
        "real_width_m": real_width_m,
        "real_height_m": real_height_m,
        "depth_border_crop_ratio": crop_ratio,
        "depth_border_crop_mode": str(crop["mode"]),
        "depth_border_crop_margin_x_px": int(crop_margin_x),
        "depth_border_crop_margin_y_px": int(crop_margin_y),
        "mask_border_crop_threshold_px": float(crop["threshold_px"]),
        "mask_border_crop_min_inside_distance_px": float(crop["min_inside_distance_px"]),
        "mask_border_crop_max_inside_distance_px": float(crop["max_inside_distance_px"]),
    }


def build_depth_pointcloud(
    depth_mm: np.ndarray,
    mask_bool: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = apply_depth_border_crop(mask_bool) & (depth_mm >= MIN_DEPTH_MM) & (depth_mm <= MAX_DEPTH_MM)
    if not np.any(valid):
        raise ValueError("No valid depth points remain for pointcloud generation after mask-border crop")
    return build_depth_pointcloud_from_valid_mask(depth_mm, valid, k)


def build_depth_pointcloud_from_valid_mask(
    depth_mm: np.ndarray,
    valid_mask: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    h, w = depth_mm.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != depth_mm.shape:
        raise ValueError(f"valid_mask shape mismatch: expected {depth_mm.shape}, got {valid.shape}")
    if not np.any(valid):
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty

    z_m = depth_mm.astype(np.float32) / 1000.0
    x_cam = (uu - cx) * z_m / fx
    y_cam = (vv - cy) * z_m / fy

    canonical_points = np.stack((x_cam, -y_cam, -z_m), axis=-1)[valid]

    # Exported PLY coordinates are chosen so that importing with
    # forward=-X, up=+Y lands in Blender world as:
    # X=right, Z=up, Y=forward, matching canonical [x, y, z] -> [x, -z, y].
    export_points = np.stack((-z_m, -y_cam, -x_cam), axis=-1)[valid]
    return export_points.astype(np.float32), canonical_points.astype(np.float32)


def select_front_visible_points(
    points: np.ndarray,
    bins: int = 128,
    max_points: int | None = None,
    seed: int = 0,
    ignore_occluded_points: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points, np.empty(0, dtype=np.int32)

    if ignore_occluded_points is None:
        ignore_occluded_points = bool(ICP_IGNORE_OCCLUDED_MODEL_POINTS)

    if ignore_occluded_points:
        # Extract the camera-visible surface when the camera looks along -Z in
        # canonical right-handed camera-local space. Larger Z is closer to the
        # camera because points in front of the camera have negative Z.
        min_xy = points[:, :2].min(axis=0)
        max_xy = points[:, :2].max(axis=0)
        span_xy = np.maximum(max_xy - min_xy, 1e-6)
        uv = np.floor((points[:, :2] - min_xy) / span_xy * (bins - 1)).astype(np.int32)
        flat = uv[:, 1] * bins + uv[:, 0]
        order = np.lexsort((-points[:, 2], flat))
        flat_sorted = flat[order]

        keep = np.empty(len(order), dtype=bool)
        keep[0] = True
        keep[1:] = flat_sorted[1:] != flat_sorted[:-1]
        selected_indices = order[keep]
    else:
        selected_indices = np.arange(len(points), dtype=np.int32)

    if max_points is not None and len(selected_indices) > max_points:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(selected_indices), size=max_points, replace=False)
        selected_indices = selected_indices[pick]

    return points[selected_indices], selected_indices.astype(np.int32, copy=False)


def extract_front_visible_points(
    points: np.ndarray,
    bins: int = 128,
    max_points: int | None = None,
    seed: int = 0,
    ignore_occluded_points: bool | None = None,
) -> np.ndarray:
    selected_points, _ = select_front_visible_points(
        points,
        bins=bins,
        max_points=max_points,
        seed=seed,
        ignore_occluded_points=ignore_occluded_points,
    )
    return selected_points


def pointcloud_export_to_canonical_rh(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return (points @ POINTCLOUD_EXPORT_TO_CANONICAL_RH_BASIS.T).astype(np.float32)


def pointcloud_export_to_blender_world(points: np.ndarray) -> np.ndarray:
    points_canonical = pointcloud_export_to_canonical_rh(points)
    return canonical_rh_to_blender_world_points(points_canonical)


def obj_vertices_to_canonical_rh(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return (points @ MODEL_INPUT_TO_CANONICAL_RH_BASIS.T).astype(np.float32)


def obj_vertices_to_blender_world(points: np.ndarray) -> np.ndarray:
    points_canonical = obj_vertices_to_canonical_rh(points)
    return canonical_rh_to_blender_world_points(points_canonical)


def model_pose_canonical_rh_to_unity_camera(
    rotation_canonical: np.ndarray,
    translation_canonical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_canonical = np.asarray(rotation_canonical, dtype=np.float32)
    translation_canonical = np.asarray(translation_canonical, dtype=np.float32)
    basis = np.asarray(CANONICAL_RH_TO_UNITY_BASIS, dtype=np.float32)
    rotation_unity = basis @ rotation_canonical @ basis
    translation_unity = basis @ translation_canonical.reshape(3, 1)
    return rotation_unity.astype(np.float32), translation_unity.reshape(3).astype(np.float32)


def canonical_rh_to_blender_world_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return np.stack((points[:, 0], -points[:, 2], points[:, 1]), axis=-1).astype(np.float32)


def canonical_rh_to_blender_world_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return np.array([vector[0], -vector[2], vector[1]], dtype=np.float32)


def blender_world_to_canonical_rh_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return np.array([vector[0], vector[2], -vector[1]], dtype=np.float32)


def rotation_canonical_rh_to_blender_world(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    return CANONICAL_RH_TO_BLENDER_WORLD @ rotation @ BLENDER_WORLD_TO_CANONICAL_RH


def rotation_blender_world_to_canonical_rh(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    return BLENDER_WORLD_TO_CANONICAL_RH @ rotation @ CANONICAL_RH_TO_BLENDER_WORLD


def model_pose_canonical_rh_to_blender_world(
    rotation_canonical: np.ndarray,
    translation_canonical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_blender = rotation_canonical_rh_to_blender_world(rotation_canonical)
    translation_blender = canonical_rh_to_blender_world_vector(translation_canonical)
    return rotation_blender.astype(np.float32), translation_blender.astype(np.float32)


def model_pose_blender_world_to_canonical_rh(
    rotation_blender: np.ndarray,
    translation_blender: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_canonical = rotation_blender_world_to_canonical_rh(rotation_blender)
    translation_canonical = blender_world_to_canonical_rh_vector(translation_blender)
    return rotation_canonical.astype(np.float32), translation_canonical.astype(np.float32)


def rotation_canonical_rh_to_blender_obj_import(
    rotation: np.ndarray,
    obj_import_to_blender_world: np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    obj_import_to_blender_world = np.asarray(obj_import_to_blender_world, dtype=np.float32)
    return (
        CANONICAL_RH_TO_BLENDER_WORLD
        @ rotation
        @ MODEL_INPUT_TO_CANONICAL_RH_BASIS
        @ obj_import_to_blender_world.T
    ).astype(np.float32)


def rotation_canonical_rh_to_blender_default_obj_import(rotation: np.ndarray) -> np.ndarray:
    return rotation_canonical_rh_to_blender_obj_import(
        rotation,
        FBX_CONVERT_OBJ_IMPORT_TO_BLENDER_WORLD,
    )


def compute_front_view_extents(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float32)
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    return {
        "bbox_min": [float(v) for v in min_corner],
        "bbox_max": [float(v) for v in max_corner],
        "width_units": float(max_corner[0] - min_corner[0]),
        "height_units": float(max_corner[1] - min_corner[1]),
        "depth_units": float(max_corner[2] - min_corner[2]),
    }


def _measure_text(text: str, scale: float, thickness: int) -> tuple[int, int, int]:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return width, height, baseline


def _fit_text_scale(
    text: str,
    max_width: int,
    *,
    base_scale: float,
    thickness: int,
    min_scale: float,
) -> float:
    if not text:
        return float(base_scale)
    scale = float(base_scale)
    while scale > float(min_scale):
        width, _, _ = _measure_text(text, scale, thickness)
        if width <= max_width:
            return scale
        scale -= 0.05
    return float(min_scale)


def _wrap_text_to_width(text: str, max_width: int, *, scale: float, thickness: int) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    width, _, _ = _measure_text(text, scale, thickness)
    if width <= max_width:
        return [text]

    words = text.split(" ")
    if len(words) <= 1:
        chunks: list[str] = []
        current = ""
        for ch in text:
            candidate = f"{current}{ch}"
            candidate_width, _, _ = _measure_text(candidate, scale, thickness)
            if current and candidate_width > max_width:
                chunks.append(current)
                current = ch
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    wrapped: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        candidate_width, _, _ = _measure_text(candidate, scale, thickness)
        if current and candidate_width > max_width:
            wrapped.append(current)
            current = word
        else:
            current = candidate
    if current:
        wrapped.append(current)
    return wrapped


def draw_image_title(image: np.ndarray, title: str) -> np.ndarray:
    title = str(title or "").strip()
    if not title:
        return image
    max_width = max(image.shape[1] - 56, 80)
    scale = _fit_text_scale(
        title,
        max_width,
        base_scale=1.0,
        thickness=2,
        min_scale=0.55,
    )
    _, height, _ = _measure_text(title, scale, 2)
    y = 24 + height
    cv2.putText(
        image,
        title,
        (28, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return image


def add_text_block(image: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return image

    pad_top = 22
    pad_bottom = 24
    left_pad = 20
    max_width = max(image.shape[1] - left_pad * 2, 80)
    scale = 0.72
    thickness = 2
    wrapped_lines: list[str] = []
    for line in lines:
        wrapped_lines.extend(_wrap_text_to_width(line, max_width, scale=scale, thickness=thickness))
    if not wrapped_lines:
        return image

    _, sample_height, sample_baseline = _measure_text("Ag", scale, thickness)
    line_h = sample_height + sample_baseline + 10
    footer_color = np.array((242, 244, 247), dtype=np.uint8)
    canvas = np.empty(
        (image.shape[0] + pad_top + pad_bottom + line_h * len(wrapped_lines), image.shape[1], 3),
        dtype=np.uint8,
    )
    canvas[:, :, :] = footer_color
    canvas[: image.shape[0], :, :] = image
    cv2.line(canvas, (0, image.shape[0]), (image.shape[1] - 1, image.shape[0]), (214, 219, 226), 2)

    y = image.shape[0] + pad_top + sample_height
    for line in wrapped_lines:
        cv2.putText(
            canvas,
            line,
            (left_pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )
        y += line_h

    return canvas


def annotate_rendered_image(image_path: Path, title: str, info_lines: list[str]) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read rendered image: {image_path}")

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        matte = np.full_like(bgr, (244.0, 246.0, 249.0))
        rgb = np.clip(bgr * alpha[:, :, None] + matte * (1.0 - alpha[:, :, None]), 0, 255).astype(np.uint8)
    else:
        rgb = image[:, :, :3]

    rgb = draw_image_title(rgb, title)
    rgb = add_text_block(rgb, info_lines)
    cv2.imwrite(str(image_path), rgb)


def write_binary_ply(path: Path, points: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")

    with path.open("wb") as f:
        f.write(header)
        for point in points:
            f.write(struct.pack("<fff", float(point[0]), float(point[1]), float(point[2])))


def read_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No OBJ vertices found: {path}")
    return np.asarray(vertices, dtype=np.float32)


def resolve_blender_path(cli_arg: str | None = None) -> Path:
    candidates: list[Path] = []

    if cli_arg:
        candidates.append(Path(cli_arg).expanduser())

    env_path = os.environ.get("BLENDER_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    if BLENDER_BIN:
        candidates.append(Path(str(BLENDER_BIN)).expanduser())

    which_blender = shutil.which("blender")
    if which_blender:
        candidates.append(Path(which_blender))

    for root in (
        Path(r"C:/Program Files/Blender Foundation"),
        Path(r"C:/Program Files (x86)/Blender Foundation"),
    ):
        candidates.extend(sorted(root.glob("Blender */blender.exe"), reverse=True))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Blender executable not found. Pass it as an argument or set BLENDER_PATH."
    )


def object_alignment_output_path(name: str) -> Path:
    path = (OBJECT_ALIGNMENT_OUTPUT_ROOT / name).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
