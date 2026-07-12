from __future__ import annotations

import sys

import _bootstrap

from stages.hololens3d_reconstruction.model_bounds import compute_and_store_model_bounds
from stage_common import load_stage_task


def main(argv: list[str]) -> int:
    try:
        json_path, _task = load_stage_task(
            argv,
            usage="Usage: python code/stages/hololens3d_reconstruction/run_model_bounds_from_json.py <task_meta.json or filename>",
            stage_name="model_bounds",
        )
        row = compute_and_store_model_bounds(json_path)
        status = row.get("status")
        print(f"[INFO] model_bounds : status={status}")
        print("[OK] model_bounds")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
