from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _bootstrap import CODE_ROOT
from config import (
    DEPTHPOINTCLOUD_STAGE_RUN,
    ICPALIGNMENT_STAGE_RUN,
    MODELSCALE_STAGE_RUN,
)
from task_json import load_task_json, resolve_task_json_path


STAGE_SCRIPTS = [
    DEPTHPOINTCLOUD_STAGE_RUN,
    MODELSCALE_STAGE_RUN,
    ICPALIGNMENT_STAGE_RUN,
]


def run_stage(stage_script: Path, json_arg: str, extra_args: list[str]) -> None:
    command = [sys.executable, str(stage_script), json_arg, *extra_args]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        cwd=str(stage_script.parent),
    )
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
            "Usage: python code/scripts/test_object_alignment_depth.py <task_meta.json or filename> [blender_path]",
            file=sys.stderr,
        )
        return 2

    json_path = resolve_task_json_path(argv[1])
    blender_path = argv[2] if len(argv) == 3 else None

    for stage_script in STAGE_SCRIPTS:
        extra_args = [blender_path] if blender_path and stage_script == ICPALIGNMENT_STAGE_RUN else []
        print(f"[RUN] {stage_script.name}")
        run_stage(stage_script, str(json_path), extra_args)

    task = load_task_json(json_path)
    alignment = task.get("object_alignment") or {}
    print("[DONE] object alignment pipeline completed")
    print(f"[DONE] scale      : {alignment.get('model_real_scale')}")
    print(f"[DONE] confidence : {alignment.get('confidence')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
