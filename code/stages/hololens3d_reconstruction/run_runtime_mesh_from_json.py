import subprocess
import sys
from pathlib import Path

import _bootstrap
from config import RUNTIME_MESH_BAKE_SCRIPT, RUNTIME_MESH_OUTPUT_ROOT
from object_alignment_common import resolve_blender_path
from stage_common import ensure_file, load_stage_task
from task_json import load_task_json


def run_runtime_mesh(json_path: Path) -> None:
    blender_bin = resolve_blender_path()
    command = [
        str(blender_bin),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(RUNTIME_MESH_BAKE_SCRIPT),
        "--",
        str(json_path),
    ]

    print("[DEBUG] running:", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Runtime mesh stage failed with return code {exc.returncode}") from exc

    task = load_task_json(json_path)
    runtime_mesh = task.get("RuntimeMesh") or {}
    mesh_name = runtime_mesh.get("mesh")
    mtl_name = runtime_mesh.get("mtl")
    image_name = runtime_mesh.get("image")
    if not mesh_name or not mtl_name or not image_name:
        raise RuntimeError(
            "Runtime mesh stage finished without RuntimeMesh.mesh / mtl / image. "
            "Check Blender logs from bake_runtime_mesh.py."
        )

    ensure_file(RUNTIME_MESH_OUTPUT_ROOT / mesh_name, "runtime mesh obj")
    ensure_file(RUNTIME_MESH_OUTPUT_ROOT / mtl_name, "runtime mesh mtl")
    ensure_file(RUNTIME_MESH_OUTPUT_ROOT / image_name, "runtime mesh texture")


def main() -> int:
    try:
        json_path, _ = load_stage_task(
            sys.argv,
            usage="Usage: python code/stages/hololens3d_reconstruction/run_runtime_mesh_from_json.py <task_meta.json or filename>",
            stage_name="runtime_mesh",
        )
        run_runtime_mesh(json_path)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
