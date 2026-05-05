from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
STAGE_ROOT = CODE_ROOT / "stages" / "hololens3d_reconstruction"
for path in (CODE_ROOT, STAGE_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from config import (  # noqa: E402
    ICP_TARGET_FRONT_MAX_POINTS,
    INSTANTMESH_CLEAN_COMPONENT_MIN_FACE_RATIO,
    INSTANTMESH_CLEAN_COMPONENT_MIN_FACES,
)
from mesh_obj_utils import clean_obj_connected_components, read_obj_geometry  # noqa: E402
from object_alignment_common import (  # noqa: E402
    build_depth_pointcloud,
    obj_vertices_to_canonical_rh,
    read_depth_image,
    read_mask,
    resolve_task_paths,
    select_front_visible_points,
    task_prefix,
)
from pose_math import quat_xyzw_to_rotation_matrix  # noqa: E402
from run_object_icp_alignment_from_json import (  # noqa: E402
    build_bbox_surface_alignment,
    build_target_context,
    transform_points,
)
from task_db import get_latest_completed_task  # noqa: E402
from task_json import load_task_json, resolve_task_json_path  # noqa: E402


def _rotation_delta_degrees(new_rotation: np.ndarray, old_rotation: np.ndarray) -> float:
    delta = np.asarray(new_rotation, dtype=np.float64) @ np.asarray(old_rotation, dtype=np.float64).T
    cos_angle = (float(np.trace(delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _write_materials(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "newmtl icp_blue",
                "Kd 0.05 0.20 1.00",
                "d 0.58",
                "Tr 0.42",
                "illum 2",
                "",
                "newmtl bbox_orange",
                "Kd 1.00 0.45 0.05",
                "d 0.58",
                "Tr 0.42",
                "illum 2",
                "",
                "newmtl delta_green",
                "Kd 0.05 0.95 0.25",
                "illum 2",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_combined_obj(
    output_obj: Path,
    *,
    faces: list[list[int]],
    old_vertices: np.ndarray,
    new_vertices: np.ndarray,
    old_center: np.ndarray,
    new_center: np.ndarray,
) -> None:
    output_mtl = output_obj.with_suffix(".mtl")
    _write_materials(output_mtl)

    lines: list[str] = [
        f"mtllib {output_mtl.name}",
        "# camera-local canonical RH coordinates",
        "# blue = stored ICP/camera_refine result",
        "# orange = new bbox-front-surface result",
        "",
        "o icp_result_blue",
        "usemtl icp_blue",
    ]
    for vertex in old_vertices:
        lines.append("v " + " ".join(f"{float(coord):.9g}" for coord in vertex))
    for face in faces:
        lines.append("f " + " ".join(str(index + 1) for index in face))

    new_offset = len(old_vertices)
    lines.extend(["", "o bbox_surface_result_orange", "usemtl bbox_orange"])
    for vertex in new_vertices:
        lines.append("v " + " ".join(f"{float(coord):.9g}" for coord in vertex))
    for face in faces:
        lines.append("f " + " ".join(str(new_offset + index + 1) for index in face))

    center_offset = len(old_vertices) + len(new_vertices)
    lines.extend(["", "o center_delta_line", "usemtl delta_green"])
    lines.append("v " + " ".join(f"{float(coord):.9g}" for coord in old_center))
    lines.append("v " + " ".join(f"{float(coord):.9g}" for coord in new_center))
    lines.append(f"l {center_offset + 1} {center_offset + 2}")

    output_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _iter_recent_completed(limit: int, max_offset: int = 40):
    collected = 0
    offset = 0
    while collected < limit and offset < max_offset:
        row = get_latest_completed_task(history_offset=offset)
        offset += 1
        if not row:
            break
        yield row
        collected += 1


def _build_comparison_for_task(row: dict, output_dir: Path) -> dict:
    task_id = str(row.get("task_id") or "")
    json_path = resolve_task_json_path(row["json_path"])
    task = load_task_json(json_path)
    prefix = task_prefix(task, json_path)
    task_dir = output_dir / f"{prefix}_{task_id[:8]}"
    task_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_task_paths(task)
    clean_mesh_path = task_dir / f"{Path(paths['mesh_path']).stem}_comparison_clean.obj"
    cleanup = clean_obj_connected_components(
        paths["mesh_path"],
        clean_mesh_path,
        min_face_ratio=float(INSTANTMESH_CLEAN_COMPONENT_MIN_FACE_RATIO),
        min_faces=int(INSTANTMESH_CLEAN_COMPONENT_MIN_FACES),
    )

    raw_vertices, faces = read_obj_geometry(clean_mesh_path)
    model_vertices_unity = obj_vertices_to_canonical_rh(raw_vertices)
    if not faces:
        raise ValueError(f"No OBJ faces found for comparison: {clean_mesh_path}")

    k = np.asarray((task.get("PVCamera") or {}).get("k"), dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"PVCamera.k must be 3x3, got {k.shape}")
    mask_bool = read_mask(paths["mask_path"])
    depth_mm = read_depth_image(paths["depth_path"])
    _export_points, pointcloud_points_unity = build_depth_pointcloud(depth_mm, mask_bool, k)
    target_front_fit, _target_front_indices = select_front_visible_points(
        pointcloud_points_unity,
        bins=160,
        max_points=ICP_TARGET_FRONT_MAX_POINTS,
        seed=7,
    )
    target_context = build_target_context(target_front_fit)

    old_alignment = task.get("object_alignment") or {}
    old_position = np.asarray(old_alignment.get("camera_local_position"), dtype=np.float32)
    old_quat = np.asarray(old_alignment.get("camera_local_rotation_quaternion_xyzw"), dtype=np.float32)
    if old_position.shape != (3,) or old_quat.shape != (4,):
        raise ValueError(f"Task {task_id} has no stored object_alignment pose")
    old_rotation = quat_xyzw_to_rotation_matrix(old_quat).astype(np.float32)
    old_scale = float(old_alignment.get("model_real_scale") or 0.0)
    if old_scale <= 0.0:
        raise ValueError(f"Task {task_id} has invalid stored object_alignment scale")

    overall_scale = float((task.get("model") or {}).get("overall_scale") or old_scale)
    new_best, _new_debug = build_bbox_surface_alignment(
        model_vertices_unity=model_vertices_unity,
        target_points=pointcloud_points_unity,
        target_front_fit=target_front_fit,
        overall_scale=overall_scale,
        target_context=target_context,
    )

    old_vertices = transform_points(model_vertices_unity, old_scale, old_rotation, old_position)
    new_vertices = transform_points(
        model_vertices_unity,
        float(new_best["scale"]),
        new_best["rotation"],
        new_best["translation"],
    )
    old_center = 0.5 * (old_vertices.min(axis=0) + old_vertices.max(axis=0))
    new_center = 0.5 * (new_vertices.min(axis=0) + new_vertices.max(axis=0))

    combined_obj = task_dir / f"{prefix}_icp_vs_bbox_surface.obj"
    _write_combined_obj(
        combined_obj,
        faces=faces,
        old_vertices=old_vertices,
        new_vertices=new_vertices,
        old_center=old_center,
        new_center=new_center,
    )

    position_delta = np.asarray(new_best["translation"], dtype=np.float64) - old_position.astype(np.float64)
    summary = {
        "task_id": task_id,
        "task_name": prefix,
        "json_path": str(json_path.relative_to(PROJECT_ROOT)),
        "combined_obj": str(combined_obj.relative_to(PROJECT_ROOT)),
        "old_icp_mode": old_alignment.get("icp_mode"),
        "old_icp_enabled": bool(old_alignment.get("icp_enabled")),
        "old_position": [float(v) for v in old_position],
        "new_position": [float(v) for v in new_best["translation"]],
        "position_delta": [float(v) for v in position_delta],
        "position_delta_m": float(np.linalg.norm(position_delta)),
        "position_delta_cm": float(np.linalg.norm(position_delta) * 100.0),
        "rotation_delta_deg": _rotation_delta_degrees(new_best["rotation"], old_rotation),
        "old_scale": float(old_scale),
        "new_scale": float(new_best["scale"]),
        "scale_delta": float(float(new_best["scale"]) - old_scale),
        "old_rmse": float(old_alignment.get("icp_rmse") or 0.0),
        "new_rmse": float(new_best["rmse"]),
        "new_surface_rmse_3d": float(new_best["surface_rmse_3d"]),
        "new_placement": new_best.get("placement"),
        "cleanup": cleanup,
    }
    (task_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or (PROJECT_ROOT / ".test" / f"alignment_compare_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    errors: list[dict] = []
    for row in _iter_recent_completed(args.limit):
        try:
            summaries.append(_build_comparison_for_task(row, output_dir))
        except Exception as exc:
            errors.append({"task_id": row.get("task_id"), "error": str(exc)})

    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "task_id",
        "task_name",
        "old_icp_mode",
        "position_delta_cm",
        "rotation_delta_deg",
        "old_scale",
        "new_scale",
        "scale_delta",
        "old_rmse",
        "new_rmse",
        "combined_obj",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: item.get(key) for key in fieldnames})

    (output_dir / "summary.json").write_text(
        json.dumps({"items": summaries, "errors": errors}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.txt").write_text(
        "\n".join(
            [
                "Blue mesh: stored ICP/camera_refine result.",
                "Orange mesh: new bbox-front-surface result used by ICP_MODE=off.",
                "Green line: center offset from stored ICP result to new result.",
                "Coordinates are camera-local canonical RH, same space as object_alignment.camera_local_position.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(output_dir)
    print(f"generated={len(summaries)} errors={len(errors)}")
    if errors:
        for error in errors:
            print(f"[WARN] {error['task_id']}: {error['error']}", file=sys.stderr)
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
