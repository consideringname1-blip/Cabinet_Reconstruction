from __future__ import annotations

import sys

import _bootstrap
from stages.hololens3d_reconstruction.object_alignment_common import (
    compute_front_view_extents,
    obj_vertices_to_canonical_rh,
    read_obj_vertices,
    resolve_task_paths,
)
from stage_common import load_stage_task
from task_json import save_task_json


def main(argv: list[str]) -> int:
    json_path, task = load_stage_task(
        argv,
        usage="Usage: python code/stages/hololens3d_reconstruction/run_model_scale_from_json.py <task_meta.json or filename>",
        stage_name="modelscale",
    )

    if "depthpointcloud" not in task:
        raise ValueError("depthpointcloud is missing. Run pointcloud stage first.")

    paths = resolve_task_paths(task)
    model_vertices_raw = read_obj_vertices(paths["mesh_path"])
    model_vertices_canonical = obj_vertices_to_canonical_rh(model_vertices_raw)
    extents = compute_front_view_extents(model_vertices_canonical)

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
        "width_model_units": extents["width_units"],
        "height_model_units": extents["height_units"],
        "depth_model_units": extents["depth_units"],
        "width_measured": width_measured,
        "height_measured": height_measured,
        "width_scale": width_scale,
        "height_scale": height_scale,
        "overall_scale": overall_scale,
    }
    task["model"] = model_info
    save_task_json(json_path, task)

    print(
        f"[INFO] modelscale : size={width_measured:.4f}x{height_measured:.4f} "
        f"scale=({width_scale:.6f}, {height_scale:.6f}, overall={overall_scale:.6f})"
    )
    print("[OK] modelscale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
