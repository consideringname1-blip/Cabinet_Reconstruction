import subprocess
import sys
from pathlib import Path

from _bootstrap import CODE_ROOT
from config import BLENDER_FBX_DIR, CONVERT_SCRIPT
from object_alignment_common import resolve_blender_path
from task_json import load_task_json, resolve_task_json_path


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def run_blender(json_path: Path) -> None:
    blender_bin = resolve_blender_path()

    command = [
        str(blender_bin),
        "--background",
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
        raise ValueError("Blender.fbx is missing")

    ensure_file(BLENDER_FBX_DIR / fbx_name, "Blender fbx")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python code/stages/run_blender_from_json.py <task_meta.json or filename>", file=sys.stderr)
        return 2

    json_path = resolve_task_json_path(sys.argv[1])
    ensure_file(json_path, "JSON file")
    try:
        run_blender(json_path)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
