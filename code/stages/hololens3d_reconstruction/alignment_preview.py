from __future__ import annotations

import subprocess
from pathlib import Path

import _bootstrap
import numpy as np

from object_alignment_common import annotate_rendered_image, resolve_blender_path


HELPER_SCRIPT = Path(__file__).resolve().with_name("blender_render_measure.py")


def _format_vec3(values: object, precision: int = 3) -> str:
    arr = np.asarray(values if values is not None else [0.0, 0.0, 0.0], dtype=np.float32).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), constant_values=0.0)
    arr = arr[:3]
    return f"({arr[0]:.{precision}f}, {arr[1]:.{precision}f}, {arr[2]:.{precision}f})"


def _format_vec4(values: object, precision: int = 3) -> str:
    arr = np.asarray(values if values is not None else [0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(-1)
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - arr.size), constant_values=0.0)
    arr = arr[:4]
    return (
        f"({arr[0]:.{precision}f}, {arr[1]:.{precision}f}, "
        f"{arr[2]:.{precision}f}, {arr[3]:.{precision}f})"
    )


def build_preview_info_lines(
    task: dict,
    *,
    header_line: str,
    model_legend_line: str,
    extra_lines: list[str] | None = None,
) -> list[str]:
    depthpointcloud = task.get("depthpointcloud") or {}
    object_alignment = task.get("object_alignment") or {}
    lines = [
        header_line,
        model_legend_line,
        f"Mode         : {str(object_alignment.get('icp_mode') or 'off')}",
        f"Scale        : {float(object_alignment.get('model_real_scale') or 0.0):.4f}",
        f"Camera pos   : {_format_vec3(object_alignment.get('camera_local_position'))}",
        f"Camera quat  : {_format_vec4(object_alignment.get('camera_local_rotation_quaternion_xyzw'))}",
        f"Depth mean   : {float(depthpointcloud.get('mean_depth_measured') or 0.0):.4f} m",
        f"ICP rmse     : {float(object_alignment.get('icp_rmse') or 0.0):.4f} m",
        f"Confidence   : {float(object_alignment.get('confidence') or 0.0):.3f}",
        (
            "Fit points   : "
            f"model={int(object_alignment.get('icp_fit_model_point_count') or 0)} "
            f"target={int(depthpointcloud.get('used_count') or 0)}"
        ),
    ]
    if extra_lines:
        lines.extend(extra_lines)
    return lines


def resolve_preview_camera_intrinsics(task: dict) -> dict[str, float | int]:
    pv_info = task.get("PVCamera") or {}
    k = np.asarray(pv_info.get("k"), dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"PVCamera.k must be 3x3, got {k.shape}")

    width = int(pv_info.get("width") or 0)
    height = int(pv_info.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("PVCamera.width and PVCamera.height must be positive")

    return {
        "width": width,
        "height": height,
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def render_overlay_preview_image(
    mesh_path: Path,
    discarded_pointcloud_path: Path,
    icp_pointcloud_path: Path,
    render_path: Path,
    blender_translation: np.ndarray,
    blender_delta_euler_deg: np.ndarray,
    scale: float,
    task: dict,
    blender_arg: str | None = None,
    title: str = "Alignment Perspective Preview",
    header_line: str = "Perspective preview: point cloud + aligned model",
    model_legend_line: str = "Orange = mask-border-discarded points, green = ICP-used points, blue = aligned model",
) -> str:
    blender_path = resolve_blender_path(blender_arg)
    camera_intrinsics = resolve_preview_camera_intrinsics(task)

    info_lines = build_preview_info_lines(
        task,
        header_line=header_line,
        model_legend_line=model_legend_line,
    )

    command = [
        str(blender_path),
        "--background",
        "--python",
        str(HELPER_SCRIPT),
        "--",
        "overlay_preview",
        str(mesh_path),
        str(discarded_pointcloud_path),
        str(icp_pointcloud_path),
        str(render_path),
        *[f"{float(v):.9f}" for v in blender_translation],
        *[f"{float(v):.9f}" for v in blender_delta_euler_deg],
        f"{float(scale):.9f}",
        str(int(camera_intrinsics["width"])),
        str(int(camera_intrinsics["height"])),
        f"{float(camera_intrinsics['fx']):.9f}",
        f"{float(camera_intrinsics['fy']):.9f}",
        f"{float(camera_intrinsics['cx']):.9f}",
        f"{float(camera_intrinsics['cy']):.9f}",
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Blender overlay preview render failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    annotate_rendered_image(render_path, title, info_lines)
    return str(blender_path)


def render_model_compare_preview_image(
    mesh_path: Path,
    render_path: Path,
    aligned_blender_translation: np.ndarray,
    aligned_blender_delta_euler_deg: np.ndarray,
    aligned_scale: float,
    reference_blender_translation: np.ndarray,
    reference_blender_delta_euler_deg: np.ndarray,
    reference_scale: float,
    task: dict,
    blender_arg: str | None = None,
    title: str = "Model Pose Comparison Preview",
    header_line: str = "Perspective preview: aligned model + initial-distance model",
    model_legend_line: str = "Blue = ICP-aligned model, orange = initial-distance model",
) -> str:
    blender_path = resolve_blender_path(blender_arg)
    camera_intrinsics = resolve_preview_camera_intrinsics(task)

    info_lines = build_preview_info_lines(
        task,
        header_line=header_line,
        model_legend_line=model_legend_line,
        extra_lines=[
            f"Initial scale: {float(reference_scale):.4f}",
            f"Aligned scale: {float(aligned_scale):.4f}",
        ],
    )

    command = [
        str(blender_path),
        "--background",
        "--python",
        str(HELPER_SCRIPT),
        "--",
        "model_compare_preview",
        str(mesh_path),
        str(render_path),
        *[f"{float(v):.9f}" for v in aligned_blender_translation],
        *[f"{float(v):.9f}" for v in aligned_blender_delta_euler_deg],
        f"{float(aligned_scale):.9f}",
        *[f"{float(v):.9f}" for v in reference_blender_translation],
        *[f"{float(v):.9f}" for v in reference_blender_delta_euler_deg],
        f"{float(reference_scale):.9f}",
        str(int(camera_intrinsics["width"])),
        str(int(camera_intrinsics["height"])),
        f"{float(camera_intrinsics['fx']):.9f}",
        f"{float(camera_intrinsics['fy']):.9f}",
        f"{float(camera_intrinsics['cx']):.9f}",
        f"{float(camera_intrinsics['cy']):.9f}",
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Blender model comparison preview render failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    annotate_rendered_image(render_path, title, info_lines)
    return str(blender_path)
