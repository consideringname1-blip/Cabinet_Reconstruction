import subprocess
import sys
from pathlib import Path

import _bootstrap
from artifact_layout import model_result_file
from path_config import CONVERT_SCRIPT
from object_alignment_common import resolve_blender_path
from stage_common import ensure_file, load_stage_task
from task_json import load_task_json


def run_blender(json_path: Path) -> None:
    blender_bin = resolve_blender_path()

    command = [
        str(blender_bin),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(CONVERT_SCRIPT),
        "--",
        str(json_path),
    ]

    print("[DEBUG] running:", " ".join(command), flush=True)

    try:
        subprocess.run(
            command,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Blender stage failed with return code {exc.returncode}") from exc

    task = load_task_json(json_path)
    blender_info = task.get("Blender") or {}
    fbx_name = blender_info.get("fbx")
    if not fbx_name:
        raise RuntimeError(
            "Blender stage finished without producing Blender.fbx. "
            "Check Blender Python imports/export logs from convert_obj_to_fbx.py."
        )

    task_timestamp = str(task.get("task_timestamp") or "").strip()
    if not task_timestamp:
        raise RuntimeError("task_timestamp is required for Blender artifacts")
    fbx_path = model_result_file(task_timestamp, "model.final_fbx")
    ensure_file(fbx_path, "Blender fbx")


def main() -> int:
    try:
        json_path, _ = load_stage_task(
            sys.argv,
            usage="Usage: python code/stages/hololens3d_reconstruction/run_blender_from_json.py <task_meta.json or filename>",
            stage_name="blender",
        )
        run_blender(json_path)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
