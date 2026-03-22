from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from object_alignment_common import load_json, resolve_json_path


STAGE_SCRIPTS = [
    "run_depthpointcloud_from_json.py",
    "run_model_scale_from_json.py",
    "run_object_icp_alignment_from_json.py",
]


def run_stage(stage_script: Path, json_arg: str, extra_args: list[str]) -> None:
    command = [sys.executable, str(stage_script), json_arg, *extra_args]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage failed: {stage_script.name}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    if completed.stdout.strip():
        print(completed.stdout.strip())


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(
            "Usage: python code/test_object_alignment_depth.py <task_meta.json or filename> [blender_path]",
            file=sys.stderr,
        )
        return 2

    json_path = resolve_json_path(argv[1])
    blender_path = argv[2] if len(argv) == 3 else None
    code_root = Path(__file__).resolve().parent

    for stage_name in STAGE_SCRIPTS:
        stage_script = code_root / stage_name
        extra_args = [blender_path] if blender_path and stage_name == "run_object_icp_alignment_from_json.py" else []
        print(f"[RUN] {stage_name}")
        run_stage(stage_script, str(json_path), extra_args)

    task = load_json(json_path)
    alignment = task.get("object_alignment") or {}
    print("[DONE] object alignment pipeline completed")
    print(f"[DONE] scale      : {alignment.get('model_real_scale')}")
    print(f"[DONE] confidence : {alignment.get('confidence')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
