from __future__ import annotations

import sys

from _bootstrap import CODE_ROOT
from object_alignment_common import (
    compute_front_view_extents,
    object_alignment_output_path,
    obj_vertices_to_unity,
    read_obj_vertices,
    resolve_task_paths,
    task_prefix,
)
from task_json import load_task_json, resolve_task_json_path, save_task_json


def remove_legacy_outputs(prefix: str) -> None:
    for name in (
        f"{prefix}_model_front.png",
        f"{prefix}_model_front_tmp.png",
        f"{prefix}_size_compare.png",
    ):
        path = object_alignment_output_path(name)
        if path.exists():
            path.unlink()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python code/stages/run_model_scale_from_json.py <task_meta.json or filename>", file=sys.stderr)
        return 2

    json_path = resolve_task_json_path(argv[1])
    task = load_task_json(json_path)
    print(f"[STAGE] modelscale : {json_path}")

    if "depthpointcloud" not in task:
        raise ValueError("depthpointcloud is missing. Run pointcloud stage first.")

    paths = resolve_task_paths(task)
    prefix = task_prefix(task, json_path)

    model_vertices_raw = read_obj_vertices(paths["mesh_path"])
    model_vertices_unity = obj_vertices_to_unity(model_vertices_raw)
    extents = compute_front_view_extents(model_vertices_unity)

    real_width = float((task.get("depthpointcloud") or {}).get("real_width_measured") or 0.0)
    real_height = float((task.get("depthpointcloud") or {}).get("real_height_measured") or 0.0)
    width_measured = extents["width_units"]
    height_measured = extents["height_units"]

    if width_measured <= 0 or height_measured <= 0:
        raise ValueError("Model width/height must be positive")

    width_scale = real_width / width_measured
    height_scale = real_height / height_measured
    overall_scale = (width_scale + height_scale) / 2.0

    model_info = {
        "coordinate_basis": "unity_x_right_y_up_z_forward",
        "blender_import_axes": {"forward": "-X", "up": "+Z"},
        "width_model_units": extents["width_units"],
        "height_model_units": extents["height_units"],
        "depth_model_units": extents["depth_units"],
        "bbox_min": extents["bbox_min"],
        "bbox_max": extents["bbox_max"],
        "width_measured": width_measured,
        "height_measured": height_measured,
        "width_scale": width_scale,
        "height_scale": height_scale,
        "overall_scale": overall_scale,
    }
    task["model"] = model_info
    save_task_json(json_path, task)
    remove_legacy_outputs(prefix)

    print(
        f"[INFO] modelscale : size={width_measured:.4f}x{height_measured:.4f} "
        f"scale=({width_scale:.6f}, {height_scale:.6f}, overall={overall_scale:.6f})"
    )
    print("[OK] modelscale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
