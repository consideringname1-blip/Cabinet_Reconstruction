import subprocess
import sys
from pathlib import Path

import _bootstrap
from artifact_layout import model_result_file
from path_config import CONVERT_SCRIPT, RUNTIME_MESH_BAKE_SCRIPT
from stages.hololens3d_reconstruction.model_generation_common import resolve_runtime_mesh_source
from stages.hololens3d_reconstruction.object_alignment_common import resolve_blender_path
from stage_common import ensure_file, load_stage_task
from task_json import load_task_json


def _run_blender_script(blender_bin: Path, script_path: Path, json_path: Path, label: str) -> None:
    command = [
        str(blender_bin),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(script_path),
        "--",
        str(json_path),
    ]

    print("[DEBUG] running:", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{label} failed with return code {exc.returncode}") from exc


def _verify_runtime_mesh(json_path: Path) -> None:
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


def _verify_final_fbx(json_path: Path) -> None:
    task = load_task_json(json_path)
    blender_info = task.get("Blender") or {}
    fbx_name = blender_info.get("fbx")
    if not fbx_name:
        raise RuntimeError(
            "Runtime mesh stage finished without producing Blender.fbx. "
            "Check Blender Python imports/export logs from convert_obj_to_fbx.py."
        )

    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if not task_timestamp:
        raise RuntimeError("task_timestamp is required for Blender artifacts")
    fbx_path = model_result_file(task_timestamp, "model.final_fbx")
    ensure_file(fbx_path, "Blender fbx")


def run_runtime_mesh(json_path: Path) -> None:
    blender_bin = resolve_blender_path()

    _run_blender_script(blender_bin, RUNTIME_MESH_BAKE_SCRIPT, json_path, "Runtime mesh bake")
    _verify_runtime_mesh(json_path)

    _run_blender_script(blender_bin, CONVERT_SCRIPT, json_path, "Runtime mesh FBX export")
    _verify_final_fbx(json_path)


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
