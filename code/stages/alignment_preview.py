from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from object_alignment_common import annotate_rendered_image, resolve_blender_path


HELPER_SCRIPT = Path(__file__).resolve().with_name("blender_render_measure.py")


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
) -> str:
    depthpointcloud = task.get("depthpointcloud") or {}
    object_alignment = task.get("object_alignment") or {}
    blender_path = resolve_blender_path(blender_arg)

    info_lines = [
        "Perspective preview: point cloud + aligned model",
        "Point cloud import: forward=-X, up=+Y",
        "Model import: forward=-X, up=+Z",
        "Orange = mask-border-discarded points, green = ICP-used points, blue = aligned model",
        f"Depth mean    : {float(depthpointcloud.get('mean_depth_measured') or 0.0):.4f} m",
        f"ICP rmse      : {float(object_alignment.get('icp_rmse') or 0.0):.4f} m",
        f"Confidence    : {float(object_alignment.get('confidence') or 0.0):.3f}",
    ]

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
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Blender overlay preview render failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    annotate_rendered_image(render_path, "Alignment Perspective Preview", info_lines)
    return str(blender_path)
