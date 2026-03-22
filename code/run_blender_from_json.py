import json
import subprocess
import sys
from pathlib import Path

from config import BLENDER_FBX_DIR, CONVERT_SCRIPT
from object_alignment_common import resolve_blender_path


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def run_blender(json_path: Path) -> None:
    blender_bin = resolve_blender_path()

    try:
        result = subprocess.run(
            [
                str(blender_bin),
                "--background",
                "--python",
                str(CONVERT_SCRIPT),
                "--",
                str(json_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr or exc.stdout or str(exc)) from exc

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    task = load_json(json_path)
    blender_info = task.get("Blender") or {}
    fbx_name = blender_info.get("fbx")
    if not fbx_name:
        raise ValueError("Blender.fbx is missing")

    ensure_file(BLENDER_FBX_DIR / fbx_name, "Blender fbx")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python run_blender_from_json.py /path/to/task.json", file=sys.stderr)
        return 2

    json_path = Path(sys.argv[1]).expanduser().resolve()
    ensure_file(json_path, "JSON file")
    try:
        run_blender(json_path)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
