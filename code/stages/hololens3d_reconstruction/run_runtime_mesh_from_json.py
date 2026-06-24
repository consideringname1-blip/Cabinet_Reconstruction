import subprocess
import sys
from pathlib import Path

import _bootstrap
from path_config import RUNTIME_MESH_BAKE_SCRIPT
from model_generation_common import resolve_runtime_mesh_source
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
    runtime_source = resolve_runtime_mesh_source(task, require_mtl_image=True)
    if runtime_source is None or runtime_source.mtl_path is None or runtime_source.image_path is None:
        raise RuntimeError(
            "Runtime mesh stage finished without RuntimeMesh.mesh / mtl / image. "
            "Check Blender logs from bake_runtime_mesh.py."
        )

    ensure_file(runtime_source.mesh_path, "runtime mesh obj")
    ensure_file(runtime_source.mtl_path, "runtime mesh mtl")
    ensure_file(runtime_source.image_path, "runtime mesh texture")


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
