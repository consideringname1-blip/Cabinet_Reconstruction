from __future__ import annotations

import os
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np

from config import (
    BLENDER_BIN,
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


MIN_DEPTH_MM = 200
MAX_DEPTH_MM = 1200

# Canonical internal basis for measurement / alignment / JSON output:
# X = right, Y = up, Z = forward
POINTCLOUD_INPUT_TO_UNITY_BASIS = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
UNITY_TO_POINTCLOUD_INPUT_BASIS = POINTCLOUD_INPUT_TO_UNITY_BASIS.T

MODEL_INPUT_TO_UNITY_BASIS = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
UNITY_TO_MODEL_INPUT_BASIS = MODEL_INPUT_TO_UNITY_BASIS.T

UNITY_TO_BLENDER_WORLD = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
BLENDER_WORLD_TO_UNITY = UNITY_TO_BLENDER_WORLD.copy()


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


def mask_bbox(mask_bool: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        raise ValueError("Mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def compute_real_measurements(mask_bool: np.ndarray, depth_mm: np.ndarray, k: np.ndarray) -> dict:
    fx = float(k[0, 0])
    fy = float(k[1, 1])

    x0, y0, x1, y1 = mask_bbox(mask_bool)
    width_px = x1 - x0 + 1
    height_px = y1 - y0 + 1

    valid_depth = mask_bool & (depth_mm >= MIN_DEPTH_MM) & (depth_mm <= MAX_DEPTH_MM)
    if not np.any(valid_depth):
        raise ValueError("No mask pixels remain within 20-120 cm")

    mean_depth_m = float(depth_mm[valid_depth].mean()) / 1000.0
    real_width_m = width_px * mean_depth_m / fx
    real_height_m = height_px * mean_depth_m / fy

    return {
        "mask_bbox_xyxy": [x0, y0, x1, y1],
        "width_px": int(width_px),
        "height_px": int(height_px),
        "valid_depth_pixels": int(valid_depth.sum()),
        "mask_pixels": int(mask_bool.sum()),
        "valid_ratio": float(valid_depth.sum() / max(mask_bool.sum(), 1)),
        "mean_depth_m": mean_depth_m,
        "real_width_m": real_width_m,
        "real_height_m": real_height_m,
    }


def build_depth_pointcloud(
    depth_mm: np.ndarray,
    mask_bool: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    h, w = depth_mm.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    valid = mask_bool & (depth_mm >= MIN_DEPTH_MM) & (depth_mm <= MAX_DEPTH_MM)
    if not np.any(valid):
        raise ValueError("No valid depth points remain for pointcloud generation")

    z_m = depth_mm.astype(np.float32) / 1000.0
    x_cam = (uu - cx) * z_m / fx
    y_cam = (vv - cy) * z_m / fy

    unity_points = np.stack((x_cam, -y_cam, z_m), axis=-1)[valid]

    # Exported PLY coordinates are chosen so that importing with
    # forward=-X, up=+Y lands in Blender world as:
    # X=right, Z=up, Y=forward, which corresponds to the same object pose.
    export_points = np.stack((-z_m, -y_cam, -x_cam), axis=-1)[valid]
    return export_points.astype(np.float32), unity_points.astype(np.float32)


def pointcloud_export_to_unity(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return (points @ POINTCLOUD_INPUT_TO_UNITY_BASIS.T).astype(np.float32)


def obj_vertices_to_unity(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return (points @ MODEL_INPUT_TO_UNITY_BASIS.T).astype(np.float32)


def model_pose_unity_to_pointcloud_input(
    rotation_unity: np.ndarray,
    translation_unity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Convert the final ICP pose from the internal Unity basis to:
    # raw model input basis -> raw pointcloud input basis.
    rotation_unity = np.asarray(rotation_unity, dtype=np.float32)
    translation_unity = np.asarray(translation_unity, dtype=np.float32)
    rotation_pointcloud = UNITY_TO_POINTCLOUD_INPUT_BASIS @ rotation_unity @ MODEL_INPUT_TO_UNITY_BASIS
    translation_pointcloud = translation_unity @ POINTCLOUD_INPUT_TO_UNITY_BASIS
    return rotation_pointcloud.astype(np.float32), translation_pointcloud.astype(np.float32)


def model_pose_pointcloud_input_to_unity(
    rotation_pointcloud: np.ndarray,
    translation_pointcloud: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Inverse of model_pose_unity_to_pointcloud_input for downstream consumers
    # that still expect the internal Unity basis.
    rotation_pointcloud = np.asarray(rotation_pointcloud, dtype=np.float32)
    translation_pointcloud = np.asarray(translation_pointcloud, dtype=np.float32)
    rotation_unity = POINTCLOUD_INPUT_TO_UNITY_BASIS @ rotation_pointcloud @ UNITY_TO_MODEL_INPUT_BASIS
    translation_unity = translation_pointcloud @ UNITY_TO_POINTCLOUD_INPUT_BASIS
    return rotation_unity.astype(np.float32), translation_unity.astype(np.float32)


def unity_to_blender_world_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return points[:, [0, 2, 1]].copy()


def unity_to_blender_world_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector[[0, 2, 1]].copy()


def blender_world_to_unity_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector[[0, 2, 1]].copy()


def rotation_unity_to_blender_world(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    return UNITY_TO_BLENDER_WORLD @ rotation @ BLENDER_WORLD_TO_UNITY


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


def add_text_block(image: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return image

    pad = 24
    line_h = 34
    canvas = np.full((image.shape[0] + pad * 2 + line_h * len(lines), image.shape[1], 3), 255, dtype=np.uint8)
    canvas[: image.shape[0], :, :] = image

    y = image.shape[0] + pad + 8
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        y += line_h

    return canvas


def render_front_view_points(
    points: np.ndarray,
    image_path: Path,
    title: str,
    info_lines: list[str] | None = None,
    point_color: tuple[int, int, int] = (20, 20, 20),
    bbox_color: tuple[int, int, int] = (0, 140, 255),
) -> None:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError("Cannot render empty point set")

    image_size = 1024
    margin = 64
    canvas = np.full((image_size, image_size, 3), 255, dtype=np.uint8)

    x = points[:, 0]
    y = points[:, 1]
    min_x, max_x = float(x.min()), float(x.max())
    min_y, max_y = float(y.min()), float(y.max())
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min((image_size - 2 * margin) / span_x, (image_size - 2 * margin) / span_y)

    u = np.round((x - min_x) * scale + margin).astype(np.int32)
    v = np.round((max_y - y) * scale + margin).astype(np.int32)
    keep = (u >= 0) & (u < image_size) & (v >= 0) & (v < image_size)
    u = u[keep]
    v = v[keep]
    canvas[v, u] = point_color

    if len(u) > 0:
        cv2.rectangle(
            canvas,
            (int(u.min()), int(v.min())),
            (int(u.max()), int(v.max())),
            bbox_color,
            2,
        )

    cv2.putText(canvas, title, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    canvas = add_text_block(canvas, info_lines or [])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), canvas)


def annotate_rendered_model_front_view(image_path: Path, title: str, info_lines: list[str]) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read rendered image: {image_path}")

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = image < 250
    elif image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        rgb = np.clip(bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None]), 0, 255).astype(np.uint8)
        mask = image[:, :, 3] > 0
    else:
        rgb = image[:, :, :3]
        mask = np.any(rgb < 250, axis=2)

    if np.any(mask):
        ys, xs = np.where(mask)
        cv2.rectangle(
            rgb,
            (int(xs.min()), int(ys.min())),
            (int(xs.max()), int(ys.max())),
            (0, 140, 255),
            2,
        )

    cv2.putText(rgb, title, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    rgb = add_text_block(rgb, info_lines)
    cv2.imwrite(str(image_path), rgb)


def annotate_rendered_image(image_path: Path, title: str, info_lines: list[str]) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read rendered image: {image_path}")

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        rgb = np.clip(bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None]), 0, 255).astype(np.uint8)
    else:
        rgb = image[:, :, :3]

    cv2.putText(rgb, title, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
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


def read_binary_ply_points(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header_lines.append(line.decode("ascii").strip())
            if header_lines[-1] == "end_header":
                break

        vertex_count = None
        for line in header_lines:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                break
        if vertex_count is None:
            raise ValueError(f"PLY vertex count missing: {path}")

        data = f.read(vertex_count * 12)
        points = np.frombuffer(data, dtype="<f4").reshape(vertex_count, 3)
        return points.astype(np.float32)


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
