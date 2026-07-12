from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import task_worker  # noqa: E402


class ArucoRetroSyncOrderTests(unittest.TestCase):
    def test_newest_capture_is_replayed_last(self) -> None:
        newest = {"task_id": "new", "status": "completed", "json_path": "new.json"}
        oldest = {"task_id": "old", "status": "completed", "json_path": "old.json"}
        display_identity_order: list[str] = []

        with (
            patch.object(task_worker, "get_tasks_for_startup_statuses", return_value=[newest, oldest]),
            patch.object(task_worker, "resolve_task_json_path_from_record", side_effect=lambda row: Path(row["json_path"])),
            patch.object(task_worker, "_run_aruco_sync"),
            patch.object(task_worker, "_run_model_bounds"),
            patch.object(
                task_worker,
                "_run_display_identity",
                side_effect=lambda path: display_identity_order.append(Path(path).stem),
            ),
        ):
            synced = task_worker._sync_completed_tasks_for_startup("startup-a")

        self.assertEqual(2, synced)
        self.assertEqual(["old", "new"], display_identity_order)


if __name__ == "__main__":
    unittest.main()
