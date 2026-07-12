from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import task_db


def _pose(x: float) -> dict:
    return {
        "position": [float(x), 0.0, 0.0],
        "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    }


class DisplayObjectStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="display-object-state-")
        self._old_database_path = task_db.DATABASE_PATH
        self._old_schema_initialized = task_db._SCHEMA_INITIALIZED
        task_db.DATABASE_PATH = Path(self._temporary.name) / "tasks.db"
        task_db._SCHEMA_INITIALIZED = False
        task_db.initialize_task_table()

    def tearDown(self) -> None:
        task_db.DATABASE_PATH = self._old_database_path
        task_db._SCHEMA_INITIALIZED = self._old_schema_initialized
        self._temporary.cleanup()

    def _capture(self, task_id: str, x: float, *, new_model: bool) -> dict:
        return task_db.commit_display_object_capture_state(
            display_object_id="object-1",
            capture_task_id=task_id,
            pose_aruco=_pose(x),
            generated_new_model=new_model,
            active_model_task_id=None if new_model else "capture-1",
        )

    def test_new_hololens_capture_invalidates_old_tracking_pointer(self) -> None:
        self._capture("capture-1", 1.0, new_model=True)
        tracked = task_db.commit_realtime_tracking_pose(
            display_object_id="object-1",
            model_revision=1,
            hololens_pose_revision=1,
            observation_seq=1,
            pose_aruco=_pose(9.0),
        )
        self.assertEqual(1, tracked["latest_tracking_pose_revision"])

        state = self._capture("capture-2", 2.0, new_model=False)

        self.assertEqual(1, state["active_model_revision"])
        self.assertEqual(0, state["latest_tracking_model_revision"])
        self.assertIsNone(state["latest_tracking_pose_aruco_json"])
        self.assertEqual(0, state["latest_tracking_observation_seq"])
        # Keep this counter monotonic so Unity does not reject the next live pose.
        self.assertEqual(1, state["latest_tracking_pose_revision"])

        stale = task_db.commit_realtime_tracking_pose(
            display_object_id="object-1",
            model_revision=1,
            hololens_pose_revision=1,
            observation_seq=2,
            pose_aruco=_pose(99.0),
        )
        self.assertIsNone(stale)

        retracked = task_db.commit_realtime_tracking_pose(
            display_object_id="object-1",
            model_revision=1,
            hololens_pose_revision=2,
            observation_seq=2,
            pose_aruco=_pose(3.0),
        )
        self.assertEqual(2, retracked["latest_tracking_pose_revision"])

    def test_body_pointer_follows_capture_order_not_completion_order(self) -> None:
        self._capture("capture-1", 1.0, new_model=True)
        self._capture("capture-2", 2.0, new_model=False)

        newest = task_db.set_latest_body_revision(
            display_object_id="object-1",
            task_id="capture-2",
        )
        late_old = task_db.set_latest_body_revision(
            display_object_id="object-1",
            task_id="capture-1",
        )
        replayed_old = task_db.set_latest_body_revision(
            display_object_id="object-1",
            task_id="capture-1",
        )

        self.assertEqual("capture-2", newest["latest_body_task_id"])
        self.assertEqual("capture-2", late_old["latest_body_task_id"])
        self.assertEqual("capture-2", replayed_old["latest_body_task_id"])
        history = {
            row["task_id"]: row
            for row in task_db.list_display_object_pose_history("object-1")
        }
        self.assertEqual(1, history["capture-1"]["body_revision"])
        self.assertEqual(2, history["capture-2"]["body_revision"])

    def test_body_link_waits_until_capture_identity_is_committed(self) -> None:
        self._capture("capture-1", 1.0, new_model=True)

        result = task_db.set_latest_body_revision(
            display_object_id="object-1",
            task_id="capture-not-committed",
        )

        self.assertIsNone(result)
        state = task_db.get_display_object_state("object-1")
        self.assertIsNone(state["latest_body_task_id"])

    def test_idempotent_aruco_resync_refreshes_pose_without_new_revision(self) -> None:
        self._capture("capture-1", 1.0, new_model=True)

        refreshed = task_db.commit_display_object_capture_state(
            display_object_id="object-1",
            capture_task_id="capture-1",
            pose_aruco=_pose(5.0),
            generated_new_model=True,
        )

        self.assertEqual(1, refreshed["active_model_revision"])
        self.assertEqual(1, refreshed["latest_hololens_pose_revision"])
        self.assertEqual(
            [5.0, 0.0, 0.0],
            json.loads(refreshed["latest_hololens_pose_aruco_json"])["position"],
        )
        history = task_db.list_display_object_pose_history("object-1")
        self.assertEqual(1, len(history))
        self.assertEqual(
            [5.0, 0.0, 0.0],
            json.loads(history[0]["pose_aruco_json"])["position"],
        )

    def test_latest_tracking_event_getter_returns_newest_object_diagnostic(self) -> None:
        self._capture("capture-1", 1.0, new_model=True)
        first_id = task_db.record_realtime_tracking_event(
            status="WRONG_NO_CONTACTED_PERSON",
            display_object_id="object-1",
            reason="explicit_empty_contacted_list",
        )
        second_id = task_db.record_realtime_tracking_event(
            status="ACCEPTED",
            display_object_id="object-1",
            observation_seq=3,
            reason="foundationpose_quality_gate_passed",
        )

        latest = task_db.get_latest_realtime_tracking_event("object-1")

        self.assertGreater(second_id, first_id)
        self.assertEqual(second_id, latest["id"])
        self.assertEqual("ACCEPTED", latest["status"])
        self.assertEqual(3, latest["observation_seq"])
        self.assertIsNone(task_db.get_latest_realtime_tracking_event("missing"))


if __name__ == "__main__":
    unittest.main()
