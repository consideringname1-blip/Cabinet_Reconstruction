from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import task_db  # noqa: E402
import task_worker  # noqa: E402


def _pose() -> dict:
    return {
        "position": [0.0, 0.0, 0.0],
        "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    }


class CompletedAuxiliaryBodyRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="aux-body-recovery-")
        self.root = Path(self._temporary.name)
        self._old_database_path = task_db.DATABASE_PATH
        self._old_schema_initialized = task_db._SCHEMA_INITIALIZED
        task_db.DATABASE_PATH = self.root / "tasks.db"
        task_db._SCHEMA_INITIALIZED = False
        task_db.initialize_task_table()

    def tearDown(self) -> None:
        task_db.DATABASE_PATH = self._old_database_path
        task_db._SCHEMA_INITIALIZED = self._old_schema_initialized
        self._temporary.cleanup()

    def test_completed_job_is_linked_after_restart_crash_window(self) -> None:
        task_id = "capture-1"
        display_object_id = "display-1"
        main_path = self.root / "main.json"
        branch_path = self.root / "branch.json"
        main_path.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "task_timestamp": "capture-ts",
                    "display_object_id": display_object_id,
                    "DisplayIdentity": {"display_object_id": display_object_id},
                }
            ),
            encoding="utf-8",
        )
        branch_path.write_text(
            json.dumps({"SAM3DBodyMesh": {"status": "SUCCESS"}}),
            encoding="utf-8",
        )
        task_db.create_task(
            task_id=task_id,
            json_path=main_path,
            task_timestamp="capture-ts",
            status="completed",
        )
        task_db.commit_display_object_capture_state(
            display_object_id=display_object_id,
            capture_task_id=task_id,
            pose_aruco=_pose(),
            generated_new_model=True,
        )
        task_db.upsert_auxiliary_job(
            task_id=task_id,
            branch_name="shigure_contact_body",
            status="completed",
            result_path=branch_path,
            detail={"body_status": "SUCCESS"},
        )
        self.assertEqual(
            0,
            task_db.get_display_object_state(display_object_id)["latest_body_revision"],
        )

        task_worker._reconcile_completed_auxiliary_outputs()
        first = task_db.get_display_object_state(display_object_id)
        task_worker._reconcile_completed_auxiliary_outputs()
        replayed = task_db.get_display_object_state(display_object_id)

        self.assertEqual(task_id, first["latest_body_task_id"])
        self.assertEqual(1, first["latest_body_revision"])
        self.assertEqual(first["latest_body_revision"], replayed["latest_body_revision"])
        history = task_db.list_display_object_pose_history(display_object_id)
        self.assertEqual(1, len(history))
        self.assertEqual(1, history[0]["body_revision"])


if __name__ == "__main__":
    unittest.main()
