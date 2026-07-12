from __future__ import annotations

import json
import sys

import _bootstrap

from stages.hololens3d_reconstruction.display_identity import bind_capture_identity
from stage_common import load_stage_task


def main(argv: list[str]) -> int:
    try:
        json_path, _task = load_stage_task(
            argv,
            usage="Usage: python code/stages/hololens3d_reconstruction/run_display_identity_from_json.py <task_meta.json or filename>",
            stage_name="display_identity",
        )
        result = bind_capture_identity(json_path)
        print(
            "[INFO] display_identity : "
            f"decision={result.get('decision')} "
            f"status={result.get('binding_status')} "
            f"display_object_id={result.get('display_object_id')} "
            f"distance={result.get('identity_distance')}"
        )
        if result.get("candidate_scores_close"):
            print("[INFO] display_identity : close candidates recorded")
        print("[OK] display_identity")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
