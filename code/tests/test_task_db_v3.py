from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import migrate_shigure_v3_data
import task_db


def _pose(x: float) -> dict:
    return {
        "position": [x, 0.0, 0.0],
        "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


class TaskDbV3Test(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "tasks.db"
        self.old_database_path = task_db.DATABASE_PATH
        self.old_marker_sync = task_db.ARUCO_SYNC_MARKER_REGISTRY_ON_START
        task_db.DATABASE_PATH = self.database_path
        task_db.ARUCO_SYNC_MARKER_REGISTRY_ON_START = False
        task_db._SCHEMA_INITIALIZED = False

    def tearDown(self) -> None:
        task_db.DATABASE_PATH = self.old_database_path
        task_db.ARUCO_SYNC_MARKER_REGISTRY_ON_START = self.old_marker_sync
        task_db._SCHEMA_INITIALIZED = False
        self.temporary.cleanup()

    def _initialize_display(self) -> None:
        task_db.initialize_task_table()
        task_db.create_display_object(display_object_id="display-1")
        with task_db._get_connection() as connection:
            connection.execute(
                f"""
                INSERT INTO {task_db.DISPLAY_OBJECT_STATE_TABLE} (
                    display_object_id, active_model_revision,
                    active_model_task_id, latest_hololens_pose_revision,
                    latest_hololens_pose_aruco_json
                ) VALUES ('display-1', 1, 'model-task', 1, ?)
                """,
                (json.dumps(_pose(0.0)),),
            )

    def _start_runtime_binding(
        self, raw_id: str = "raw-a"
    ) -> tuple[dict, dict, dict]:
        runtime = task_db.start_shigure_runtime_session(
            server_boot_id=f"test-{raw_id}"
        )
        epoch = task_db.open_shigure_source_epoch(
            runtime_session_id=runtime["runtime_session_id"], reason="test"
        )
        binding = task_db.establish_shigure_binding(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            raw_shigure_object_id=raw_id,
            display_object_id="display-1",
            established_by="test",
        )
        return runtime, epoch, binding

    def test_pending_canonical_event_resolve_is_strict_and_idempotent(self) -> None:
        self._initialize_display()
        runtime, epoch, binding = self._start_runtime_binding()
        pending = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=10,
            stamp_nanosec=20,
            frame_id="camera",
            detection_index=0,
            action="TAKE_OUT",
            bbox={},
            resolution_status="UNRESOLVED",
            raw_shigure_object_id="raw-a",
            detail={"capture_audit": "preserved"},
        )
        resolved = task_db.resolve_pending_shigure_canonical_event(
            pending["event_uid"],
            binding["binding_id"],
            "display-1",
            "raw-a",
            "DEFERRED_DINO_MATCH",
            detail={"distance": 0.12},
        )
        self.assertEqual(resolved["resolution_status"], "RESOLVED")
        self.assertEqual(resolved["binding_id"], binding["binding_id"])
        self.assertEqual(resolved["display_object_id"], "display-1")
        self.assertEqual(resolved["raw_shigure_object_id"], "raw-a")
        self.assertEqual(resolved["resolution_method"], "DEFERRED_DINO_MATCH")
        resolved_detail = json.loads(resolved["detail_json"])
        self.assertEqual(resolved_detail["capture_audit"], "preserved")
        self.assertEqual(resolved_detail["distance"], 0.12)

        task_db.revoke_shigure_binding(
            binding["binding_id"], reason="lifecycle_completed"
        )
        replay = task_db.resolve_pending_shigure_canonical_event(
            pending["event_uid"],
            binding["binding_id"],
            "display-1",
            "raw-a",
            "DEFERRED_DINO_MATCH",
            detail={"must_not_overwrite": True},
        )
        self.assertEqual(replay, resolved)
        with self.assertRaisesRegex(ValueError, "different identity"):
            task_db.resolve_pending_shigure_canonical_event(
                pending["event_uid"],
                binding["binding_id"],
                "display-1",
                "raw-a",
                "DIFFERENT_METHOD",
            )

    def test_pending_canonical_event_requires_its_active_epoch_binding(self) -> None:
        self._initialize_display()
        runtime, epoch, binding = self._start_runtime_binding()
        pending = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=11,
            stamp_nanosec=21,
            frame_id="camera",
            detection_index=0,
            action="BRING_IN",
            bbox={},
            resolution_status="AMBIGUOUS",
            raw_shigure_object_id="raw-a",
        )
        other_binding = task_db.establish_shigure_binding(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            raw_shigure_object_id="raw-b",
            display_object_id="display-1",
            established_by="test",
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            task_db.resolve_pending_shigure_canonical_event(
                pending["event_uid"],
                other_binding["binding_id"],
                "display-1",
                "raw-b",
                "DEFERRED_DINO_MATCH",
            )

        task_db.revoke_shigure_binding(binding["binding_id"], reason="test")
        with self.assertRaisesRegex(ValueError, "not active"):
            task_db.resolve_pending_shigure_canonical_event(
                pending["event_uid"],
                binding["binding_id"],
                "display-1",
                "raw-a",
                "DEFERRED_DINO_MATCH",
            )
        self.assertEqual(
            task_db.get_shigure_canonical_event(pending["event_uid"])[
                "resolution_status"
            ],
            "AMBIGUOUS",
        )

    def test_pending_canonical_event_rejection_is_terminal_and_audited(self) -> None:
        self._initialize_display()
        runtime, epoch, binding = self._start_runtime_binding()
        pending = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=12,
            stamp_nanosec=22,
            frame_id="camera",
            detection_index=0,
            action="TAKE_OUT",
            bbox={},
            resolution_status="CONFLICT",
            raw_shigure_object_id="raw-a",
            detail={"capture_audit": "preserved"},
        )
        rejected = task_db.reject_pending_shigure_canonical_event(
            pending["event_uid"],
            "DEFERRED_IDENTITY_TIMEOUT",
            detail={"attempts": 5},
        )
        self.assertEqual(rejected["resolution_status"], "REJECTED")
        self.assertEqual(
            rejected["resolution_method"], "DEFERRED_IDENTITY_TIMEOUT"
        )
        rejected_detail = json.loads(rejected["detail_json"])
        self.assertEqual(rejected_detail["capture_audit"], "preserved")
        self.assertEqual(rejected_detail["attempts"], 5)
        self.assertEqual(
            rejected_detail["rejection_reason"], "DEFERRED_IDENTITY_TIMEOUT"
        )
        rerecorded = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=12,
            stamp_nanosec=22,
            frame_id="camera",
            detection_index=0,
            action="TAKE_OUT",
            bbox={},
            resolution_status="RESOLVED",
            raw_shigure_object_id="raw-a",
            binding_id=binding["binding_id"],
            display_object_id="display-1",
            resolution_method="MUST_NOT_PROMOTE",
            detail={"must_not_overwrite": True},
        )
        self.assertEqual(rerecorded, rejected)
        replay = task_db.reject_pending_shigure_canonical_event(
            pending["event_uid"], "DIFFERENT_REASON", detail={"attempts": 99}
        )
        self.assertEqual(replay, rejected)
        with self.assertRaisesRegex(ValueError, "cannot be resolved"):
            task_db.resolve_pending_shigure_canonical_event(
                pending["event_uid"],
                binding["binding_id"],
                "display-1",
                "raw-a",
                "DEFERRED_DINO_MATCH",
            )

        resolved = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=13,
            stamp_nanosec=23,
            frame_id="camera",
            detection_index=1,
            action="BRING_IN",
            bbox={},
            resolution_status="RESOLVED",
            raw_shigure_object_id="raw-a",
            binding_id=binding["binding_id"],
            display_object_id="display-1",
            resolution_method="TRUSTED_EPOCH_BINDING",
        )
        protected = task_db.reject_pending_shigure_canonical_event(
            resolved["event_uid"], "MUST_NOT_DOWNGRADE"
        )
        self.assertEqual(protected, resolved)
        self.assertEqual(protected["resolution_status"], "RESOLVED")

    def test_atomic_take_out_then_fresh_bring_in_commits_consistently(self) -> None:
        self._initialize_display()
        runtime, epoch, binding = self._start_runtime_binding()
        task_db.activate_recovered_shigure_binding(binding["binding_id"])
        take_out = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=20,
            stamp_nanosec=1,
            frame_id="camera",
            detection_index=0,
            action="TAKE_OUT",
            bbox={},
            resolution_status="UNRESOLVED",
            raw_shigure_object_id="raw-a",
            binding_id=binding["binding_id"],
            display_object_id="display-1",
            resolution_method="TAKE_OUT_WINDOW",
        )
        taken = task_db.commit_pending_take_out_lifecycle_event(
            take_out["event_uid"],
            binding["binding_id"],
            "display-1",
            "raw-a",
            "TAKE_OUT_WINDOW",
            detail={"selected": True},
        )
        self.assertEqual(
            taken["canonical_event"]["resolution_status"], "RESOLVED"
        )
        self.assertEqual(taken["lifecycle_event"]["action"], "TAKE_OUT")
        self.assertEqual(taken["binding"]["status"], "REVOKED")
        self.assertEqual(
            task_db.get_display_object_state("display-1")["presence"], "ABSENT"
        )

        bring_in = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=20,
            stamp_nanosec=2,
            frame_id="camera",
            detection_index=1,
            action="BRING_IN",
            bbox={},
            resolution_status="UNRESOLVED",
            raw_shigure_object_id="raw-b",
            display_object_id="display-1",
            resolution_method="BRING_IN_DINO_WINDOW",
        )
        brought = task_db.commit_pending_bring_in_lifecycle_event(
            bring_in["event_uid"],
            runtime["runtime_session_id"],
            epoch["source_epoch_id"],
            "raw-b",
            "display-1",
            "BRING_IN_DINO_WINDOW",
            confidence=0.9,
            binding_detail={"distance": 0.1},
            resolution_detail={"selected": True},
        )
        self.assertEqual(
            brought["canonical_event"]["resolution_status"], "RESOLVED"
        )
        self.assertEqual(brought["lifecycle_event"]["action"], "BRING_IN")
        self.assertEqual(brought["binding"]["status"], "ACTIVE")
        state = task_db.get_display_object_state("display-1")
        self.assertEqual(state["presence"], "PRESENT")
        self.assertEqual(
            state["active_shigure_binding_id"], brought["binding"]["binding_id"]
        )

    def test_atomic_lifecycle_failures_roll_back_resolution_and_binding(self) -> None:
        self._initialize_display()
        runtime, epoch, binding = self._start_runtime_binding()
        take_out = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=21,
            stamp_nanosec=1,
            frame_id="camera",
            detection_index=0,
            action="TAKE_OUT",
            bbox={},
            resolution_status="UNRESOLVED",
            raw_shigure_object_id="raw-a",
            binding_id=binding["binding_id"],
            display_object_id="display-1",
            resolution_method="TAKE_OUT_WINDOW",
        )
        with self.assertRaisesRegex(ValueError, "requires a present object"):
            task_db.commit_pending_take_out_lifecycle_event(
                take_out["event_uid"],
                binding["binding_id"],
                "display-1",
                "raw-a",
                "TAKE_OUT_WINDOW",
            )
        self.assertEqual(
            task_db.get_shigure_canonical_event(take_out["event_uid"])[
                "resolution_status"
            ],
            "UNRESOLVED",
        )
        self.assertEqual(task_db.get_shigure_binding(binding["binding_id"])["status"], "ACTIVE")

        task_db.activate_recovered_shigure_binding(binding["binding_id"])
        bring_in = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=21,
            stamp_nanosec=2,
            frame_id="camera",
            detection_index=1,
            action="BRING_IN",
            bbox={},
            resolution_status="UNRESOLVED",
            raw_shigure_object_id="raw-new",
            display_object_id="display-1",
            resolution_method="BRING_IN_DINO_WINDOW",
        )
        with self.assertRaisesRegex(ValueError, "duplicate bring-in"):
            task_db.commit_pending_bring_in_lifecycle_event(
                bring_in["event_uid"],
                runtime["runtime_session_id"],
                epoch["source_epoch_id"],
                "raw-new",
                "display-1",
                "BRING_IN_DINO_WINDOW",
            )
        self.assertEqual(
            task_db.get_shigure_canonical_event(bring_in["event_uid"])[
                "resolution_status"
            ],
            "UNRESOLVED",
        )
        self.assertIsNone(
            task_db.get_active_shigure_binding(
                source_epoch_id=epoch["source_epoch_id"],
                raw_shigure_object_id="raw-new",
            )
        )
        with task_db._get_connection() as connection:
            lifecycle_count = connection.execute(
                f"SELECT COUNT(*) FROM {task_db.OBJECT_LIFECYCLE_EVENT_TABLE}"
            ).fetchone()[0]
        self.assertEqual(lifecycle_count, 0)

    def test_pending_canonical_identity_is_first_non_null_and_fail_closed(self) -> None:
        self._initialize_display()
        runtime, epoch, _binding = self._start_runtime_binding()

        def record(**overrides: object) -> dict:
            values = {
                "raw_shigure_object_id": "raw-a",
                "binding_id": "binding-a",
                "display_object_id": "display-1",
                "resolution_method": "WINDOW-A",
            }
            values.update(overrides)
            return task_db.record_shigure_canonical_event(
                runtime_session_id=runtime["runtime_session_id"],
                source_epoch_id=epoch["source_epoch_id"],
                stamp_sec=22,
                stamp_nanosec=1,
                frame_id="camera",
                detection_index=0,
                action="TAKE_OUT",
                bbox={},
                resolution_status="UNRESOLVED",
                **values,
            )

        original = record()
        replay = record(
            raw_shigure_object_id=None,
            binding_id=None,
            display_object_id=None,
            resolution_method=None,
        )
        for column in (
            "raw_shigure_object_id",
            "binding_id",
            "display_object_id",
            "resolution_method",
        ):
            self.assertEqual(replay[column], original[column])
        for column, conflict in (
            ("raw_shigure_object_id", "raw-b"),
            ("binding_id", "binding-b"),
            ("display_object_id", "display-2"),
            ("resolution_method", "WINDOW-B"),
        ):
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, "identity conflict"):
                    record(**{column: conflict})
        self.assertEqual(
            task_db.get_shigure_canonical_event(original["event_uid"])[
                "display_object_id"
            ],
            "display-1",
        )

    def test_epoch_transitions_terminally_close_pending_lifecycle_authority(self) -> None:
        self._initialize_display()
        runtime, epoch, binding = self._start_runtime_binding()
        pending = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=23,
            stamp_nanosec=1,
            frame_id="camera",
            detection_index=0,
            action="TAKE_OUT",
            bbox={},
            resolution_status="UNRESOLVED",
            raw_shigure_object_id="raw-a",
            binding_id=binding["binding_id"],
            display_object_id="display-1",
            resolution_method="TAKE_OUT_WINDOW",
        )
        orphan = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=23,
            stamp_nanosec=2,
            frame_id="camera",
            detection_index=1,
            action="BRING_IN",
            bbox={},
            resolution_status="RESOLVED",
            raw_shigure_object_id="raw-a",
            binding_id=binding["binding_id"],
            display_object_id="display-1",
            resolution_method="TRUSTED_EPOCH_BINDING",
        )
        next_epoch = task_db.open_shigure_source_epoch(
            runtime_session_id=runtime["runtime_session_id"], reason="rotate"
        )
        self.assertEqual(
            next_epoch["_lifecycle_epoch_close_audit"],
            {
                "rejected_pending_lifecycle_events": 1,
                "audited_resolved_without_lifecycle": 1,
            },
        )
        closed_pending = task_db.get_shigure_canonical_event(pending["event_uid"])
        self.assertEqual(closed_pending["resolution_status"], "REJECTED")
        self.assertEqual(
            closed_pending["resolution_method"],
            "SOURCE_EPOCH_CHANGED_PENDING_LIFECYCLE",
        )
        audited_orphan = task_db.get_shigure_canonical_event(orphan["event_uid"])
        self.assertEqual(audited_orphan["resolution_status"], "RESOLVED")
        orphan_audit = json.loads(audited_orphan["detail_json"])[
            "lifecycle_terminal_audit"
        ]
        self.assertEqual(
            orphan_audit["status"], "ORPHANED_RESOLVED_WITHOUT_LIFECYCLE"
        )
        self.assertFalse(orphan_audit["safe_replay"])

    def test_local_only_capture_later_commits_one_calibrated_pose(self) -> None:
        task_db.initialize_task_table()
        task_db.create_display_object(display_object_id="display-local")
        previous = task_db.commit_display_object_capture_state(
            display_object_id="display-local",
            capture_task_id="capture-previous",
            pose_aruco=_pose(1.0),
            captured_at="2025-12-31T00:00:00+00:00",
            generated_new_model=True,
            active_model_task_id="model-local",
            asset_hash="asset-local",
        )
        self.assertEqual(previous["latest_hololens_pose_revision"], 1)
        self.assertIsNotNone(previous["latest_hololens_pose_aruco_json"])
        local = task_db.commit_display_object_capture_state(
            display_object_id="display-local",
            capture_task_id="capture-local",
            pose_aruco=None,
            captured_at="2026-01-01T00:00:00+00:00",
            generated_new_model=False,
            active_model_task_id="model-local",
            asset_hash="asset-local",
        )
        self.assertEqual(local["active_model_revision"], 1)
        self.assertEqual(local["active_model_task_id"], "model-local")
        self.assertEqual(local["latest_hololens_task_id"], "capture-local")
        self.assertEqual(local["latest_hololens_pose_revision"], 1)
        self.assertIsNone(local["latest_hololens_pose_aruco_json"])
        history_before_sync = task_db.list_display_object_pose_history(
            "display-local"
        )
        self.assertEqual(len(history_before_sync), 1)
        self.assertEqual(history_before_sync[0]["task_id"], "capture-previous")
        self.assertEqual(
            [
                row["display_object_id"]
                for row in task_db.list_live_display_object_states(limit=10)
            ],
            ["display-local"],
        )

        replay_local = task_db.commit_display_object_capture_state(
            display_object_id="display-local",
            capture_task_id="capture-local",
            pose_aruco=None,
            captured_at="2026-01-01T00:00:00+00:00",
            generated_new_model=False,
            active_model_task_id="model-local",
            asset_hash="asset-local",
        )
        self.assertEqual(replay_local["active_model_revision"], 1)
        calibrated = task_db.commit_display_object_capture_state(
            display_object_id="display-local",
            capture_task_id="capture-local",
            pose_aruco=_pose(3.0),
            captured_at="2026-01-01T00:00:00+00:00",
            generated_new_model=False,
            active_model_task_id="model-local",
            asset_hash="asset-local",
        )
        self.assertEqual(calibrated["active_model_revision"], 1)
        self.assertEqual(calibrated["latest_hololens_pose_revision"], 2)
        history = task_db.list_display_object_pose_history("display-local")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["task_id"], "capture-local")
        replay_calibrated = task_db.commit_display_object_capture_state(
            display_object_id="display-local",
            capture_task_id="capture-local",
            pose_aruco=_pose(4.0),
            captured_at="2026-01-01T00:00:00+00:00",
            generated_new_model=False,
            active_model_task_id="model-local",
            asset_hash="asset-local",
        )
        self.assertEqual(replay_calibrated["latest_hololens_pose_revision"], 2)
        self.assertEqual(
            len(task_db.list_display_object_pose_history("display-local")), 2
        )

    def test_old_local_capture_replay_cannot_roll_back_new_capture(self) -> None:
        task_db.initialize_task_table()
        task_db.create_display_object(display_object_id="display-local-order")
        for task_id, timestamp in (
            ("capture-old-local", "2026-01-01T00:00:00+00:00"),
            ("capture-new", "2026-01-02T00:00:00+00:00"),
        ):
            task_db.upsert_capture_instance(
                capture_instance_id=f"instance-{task_id}",
                task_id=task_id,
                display_object_id="display-local-order",
                timestamp=timestamp,
                binding_status="bound",
            )

        task_db.commit_display_object_capture_state(
            display_object_id="display-local-order",
            capture_task_id="capture-old-local",
            pose_aruco=None,
            generated_new_model=True,
            active_model_task_id="model-local-order",
        )
        newest = task_db.commit_display_object_capture_state(
            display_object_id="display-local-order",
            capture_task_id="capture-new",
            pose_aruco=_pose(2.0),
            generated_new_model=False,
            active_model_task_id="model-local-order",
        )
        replay = task_db.commit_display_object_capture_state(
            display_object_id="display-local-order",
            capture_task_id="capture-old-local",
            pose_aruco=None,
            generated_new_model=True,
            active_model_task_id="model-local-order",
        )

        self.assertEqual(replay["latest_hololens_task_id"], "capture-new")
        self.assertEqual(
            replay["latest_hololens_captured_at"],
            "2026-01-02T00:00:00+00:00",
        )
        self.assertEqual(
            replay["latest_hololens_pose_aruco_json"],
            newest["latest_hololens_pose_aruco_json"],
        )
        self.assertEqual(
            replay["latest_hololens_pose_revision"],
            newest["latest_hololens_pose_revision"],
        )
        self.assertEqual(replay["active_model_revision"], 1)

    def test_origins_keep_five_and_deduplicate_nearby_pose(self) -> None:
        self._initialize_display()
        first = task_db.add_display_object_origin(
            "display-1", 1, _pose(0.0), "INITIALIZATION", binding_id="binding-0"
        )
        replay = task_db.add_display_object_origin(
            "display-1", 1, _pose(0.0), "INITIALIZATION", binding_id="binding-0"
        )
        self.assertTrue(first["_inserted"])
        self.assertFalse(replay["_inserted"])
        self.assertEqual(first["origin_uid"], replay["origin_uid"])

        nearby = task_db.add_display_object_origin(
            "display-1",
            1,
            _pose(0.1),
            "TAKE_OUT",
            canonical_event_uid="event-near",
        )
        self.assertTrue(nearby["_deduplicated"])
        for index in range(1, 7):
            task_db.add_display_object_origin(
                "display-1",
                1,
                _pose(float(index)),
                "TAKE_OUT",
                canonical_event_uid=f"event-{index}",
            )
        replay_near = task_db.add_display_object_origin(
            "display-1",
            1,
            _pose(0.1),
            "TAKE_OUT",
            canonical_event_uid="event-near",
        )
        self.assertFalse(replay_near["_inserted"])
        self.assertTrue(replay_near["_deduplicated"])
        origins = task_db.list_display_object_origins("display-1", limit=5)
        self.assertEqual(len(origins), 5)
        self.assertEqual(json.loads(origins[0]["pose_aruco_json"])["position"][0], 6.0)
        self.assertEqual(task_db.get_latest_display_object_origin("display-1")["id"], origins[0]["id"])

    def test_origin_order_cap_cursor_and_dedup_use_occurred_at(self) -> None:
        self._initialize_display()
        for day in range(2, 7):
            task_db.add_display_object_origin(
                "display-1",
                1,
                _pose(float(day)),
                "TAKE_OUT",
                canonical_event_uid=f"event-{day}",
                occurred_at=f"2026-01-{day:02d}T00:00:00+00:00",
            )

        # An old event that finishes last must neither become latest nor evict
        # any of the five genuinely newest origins.
        task_db.add_display_object_origin(
            "display-1",
            1,
            _pose(1.0),
            "TAKE_OUT",
            canonical_event_uid="event-late-old",
            occurred_at="2026-01-01T00:00:00+00:00",
        )
        origins = task_db.list_display_object_origins("display-1", limit=5)
        self.assertEqual(
            [json.loads(row["pose_aruco_json"])["position"][0] for row in origins],
            [6.0, 5.0, 4.0, 3.0, 2.0],
        )
        self.assertEqual(
            task_db.get_latest_display_object_origin("display-1")["id"],
            origins[0]["id"],
        )
        self.assertEqual(
            [row["id"] for row in task_db.list_display_object_origins(
                "display-1", limit=5, before_id=origins[1]["id"]
            )],
            [row["id"] for row in origins[2:]],
        )

        # A delayed middle event is compared with its chronological neighbors,
        # not whichever row happened to be inserted most recently.
        deduplicated = task_db.add_display_object_origin(
            "display-1",
            1,
            _pose(3.9),
            "TAKE_OUT",
            canonical_event_uid="event-middle-near-four",
            occurred_at="2026-01-03T12:00:00+00:00",
        )
        self.assertTrue(deduplicated["_deduplicated"])
        self.assertEqual(
            json.loads(deduplicated["pose_aruco_json"])["position"][0],
            4.0,
        )

    def test_aliases_promote_primary_and_take_out_revokes_all(self) -> None:
        self._initialize_display()
        runtime = task_db.start_shigure_runtime_session(server_boot_id="test")
        epoch = task_db.open_shigure_source_epoch(
            runtime_session_id=runtime["runtime_session_id"], reason="test"
        )
        bindings = []
        for raw_id in ("raw-a", "raw-b"):
            binding = task_db.establish_shigure_binding(
                runtime_session_id=runtime["runtime_session_id"],
                source_epoch_id=epoch["source_epoch_id"],
                raw_shigure_object_id=raw_id,
                display_object_id="display-1",
                established_by="test",
            )
            task_db.activate_recovered_shigure_binding(binding["binding_id"])
            bindings.append(binding)
        aliases = task_db.list_display_object_alias_bindings(
            epoch["source_epoch_id"], "display-1"
        )
        self.assertEqual(len(aliases), 2)
        self.assertEqual(aliases[0]["binding_id"], bindings[1]["binding_id"])

        event = task_db.record_shigure_canonical_event(
            runtime_session_id=runtime["runtime_session_id"],
            source_epoch_id=epoch["source_epoch_id"],
            stamp_sec=1,
            stamp_nanosec=2,
            frame_id="camera",
            detection_index=0,
            action="TAKE_OUT",
            bbox={},
            resolution_status="RESOLVED",
            raw_shigure_object_id="raw-a",
            binding_id=bindings[0]["binding_id"],
            display_object_id="display-1",
            resolution_method="test_alias",
        )
        task_db.apply_object_lifecycle_event(canonical_event_uid=event["event_uid"])
        self.assertEqual(
            task_db.list_display_object_alias_bindings(
                epoch["source_epoch_id"], "display-1"
            ),
            [],
        )

    def test_only_five_newest_hololens_references_remain_active(self) -> None:
        self._initialize_display()
        for index in range(7):
            task_db.add_object_identity_reference(
                display_object_id="display-1",
                source="HOLOLENS",
                image_path=Path(self.temporary.name) / f"view-{index}.png",
                view_hash=f"view-{index}",
            )
        references = task_db.list_object_identity_references(
            "display-1", source="HOLOLENS", limit=20
        )
        self.assertEqual(len(references), 5)
        self.assertEqual(references[0]["view_hash"], "view-6")
        self.assertEqual(len(task_db.list_display_object_states(limit=10)), 1)

    def test_view_hash_replay_does_not_refresh_reference_age(self) -> None:
        self._initialize_display()
        for index in range(5):
            task_db.add_object_identity_reference(
                display_object_id="display-1",
                source="HOLOLENS",
                image_path=Path(self.temporary.name) / f"view-{index}.png",
                view_hash=f"view-{index}",
            )
        with task_db._get_connection() as connection:
            for index in range(5):
                connection.execute(
                    f"""
                    UPDATE {task_db.OBJECT_IDENTITY_REFERENCE_TABLE}
                    SET created_at = ?
                    WHERE display_object_id = 'display-1' AND view_hash = ?
                    """,
                    (f"2026-01-0{index + 1}T00:00:00", f"view-{index}"),
                )
        replay = task_db.add_object_identity_reference(
            display_object_id="display-1",
            source="HOLOLENS",
            image_path=Path(self.temporary.name) / "view-0-replayed.png",
            view_hash="view-0",
        )
        self.assertEqual(replay["created_at"], "2026-01-01T00:00:00")
        task_db.add_object_identity_reference(
            display_object_id="display-1",
            source="HOLOLENS",
            image_path=Path(self.temporary.name) / "view-5.png",
            view_hash="view-5",
        )
        active_hashes = {
            row["view_hash"]
            for row in task_db.list_object_identity_references(
                "display-1", source="HOLOLENS", limit=20
            )
        }
        self.assertEqual(active_hashes, {"view-1", "view-2", "view-3", "view-4", "view-5"})

    def test_identity_sync_candidate_limit_supports_fifty(self) -> None:
        job = task_db.upsert_identity_sync_job(
            sync_job_id="sync-fifty",
            kind="STARTUP_RECOVERY",
            status="PENDING",
            candidate_limit=50,
        )
        self.assertEqual(job["candidate_limit"], 50)
        updated = task_db.upsert_identity_sync_job(
            sync_job_id="sync-fifty",
            kind="STARTUP_RECOVERY",
            status="RUNNING",
            candidate_limit=37,
        )
        self.assertEqual(updated["candidate_limit"], 37)
        capped = task_db.upsert_identity_sync_job(
            sync_job_id="sync-over-limit",
            kind="STARTUP_RECOVERY",
            status="PENDING",
            candidate_limit=500,
        )
        self.assertEqual(capped["candidate_limit"], 50)

    def test_historical_candidates_keep_latest_five_for_fifty_objects(self) -> None:
        task_db.initialize_task_table()
        rows = []
        for capture_index in range(600):
            rows.append(
                (
                    f"dense-{capture_index:03d}",
                    "display-dense",
                    f"dense-task-{capture_index:03d}",
                    f"2026-01-01T{capture_index // 3600:02d}:"
                    f"{(capture_index // 60) % 60:02d}:{capture_index % 60:02d}",
                )
            )
        for display_index in range(49):
            display_object_id = f"display-{display_index:02d}"
            for view_index in range(5):
                rows.append(
                    (
                        f"capture-{display_index:02d}-{view_index}",
                        display_object_id,
                        f"task-{display_index:02d}-{view_index}",
                        f"2026-02-{display_index + 1:02d}T00:00:0{view_index}",
                    )
                )
        with task_db._get_connection() as connection:
            connection.executemany(
                f"""
                INSERT INTO {task_db.CAPTURE_INSTANCE_TABLE} (
                    capture_instance_id, display_object_id, task_id,
                    timestamp, binding_status
                ) VALUES (?, ?, ?, ?, 'bound')
                """,
                rows,
            )

        candidates = task_db.list_identity_candidate_captures_by_display(
            display_object_limit=50,
            references_per_display=5,
        )
        counts: dict[str, int] = {}
        for candidate in candidates:
            display_object_id = str(candidate["display_object_id"])
            counts[display_object_id] = counts.get(display_object_id, 0) + 1
        self.assertEqual(len(counts), 50)
        self.assertEqual(set(counts.values()), {5})
        self.assertEqual(len(candidates), 250)
        self.assertEqual(
            {
                row["task_id"]
                for row in candidates
                if row["display_object_id"] == "display-dense"
            },
            {f"dense-task-{index:03d}" for index in range(595, 600)},
        )
        excluded = task_db.list_identity_candidate_captures_by_display(
            display_object_limit=50,
            references_per_display=5,
            exclude_capture_instance_id="dense-599",
        )
        self.assertEqual(
            {
                row["task_id"]
                for row in excluded
                if row["display_object_id"] == "display-dense"
            },
            {f"dense-task-{index:03d}" for index in range(594, 599)},
        )

    def test_shigure_images_cannot_be_identity_references(self) -> None:
        self._initialize_display()
        with self.assertRaisesRegex(ValueError, "query-only"):
            task_db.add_object_identity_reference(
                display_object_id="display-1",
                source="SHIGURE",
                image_path=Path(self.temporary.name) / "scene.png",
                mask_path=Path(self.temporary.name) / "mask.png",
                view_hash="forbidden-shigure-reference",
            )

    def test_explicit_migration_creates_backup_and_v3_schema(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            for _, builder in task_db._v2_table_builders():
                connection.execute(builder())
            for index_sql in task_db._v2_index_sql():
                connection.execute(index_sql)
            connection.execute(
                f"""
                INSERT INTO {task_db.SCHEMA_METADATA_TABLE} (
                    schema_name, schema_version, migrated_at, detail_json
                ) VALUES ('shigure_runtime', 2, '2026-01-01T00:00:00', '{{}}')
                """
            )
            connection.execute(
                f"""
                INSERT INTO {task_db.DISPLAY_OBJECT_STATE_TABLE} (
                    display_object_id, active_model_revision,
                    active_model_task_id, latest_hololens_pose_revision,
                    latest_hololens_pose_aruco_json
                ) VALUES ('display-1', 1, 'model-task', 7, ?)
                """,
                (json.dumps(_pose(5.0)),),
            )
            for index, x_position in enumerate((0.0, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0), 1):
                connection.execute(
                    f"""
                    INSERT INTO {task_db.DISPLAY_OBJECT_POSE_HISTORY_TABLE} (
                        display_object_id, task_id, model_revision,
                        pose_revision, pose_aruco_json, captured_at
                    ) VALUES ('display-1', ?, 1, ?, ?, ?)
                    """,
                    (
                        f"capture-{index}",
                        index,
                        json.dumps(_pose(x_position)),
                        f"2026-01-{index:02d}T00:00:00+00:00",
                    ),
                )
            connection.execute(
                f"""
                INSERT INTO {task_db.OBJECT_LIFECYCLE_EVENT_TABLE} (
                    lifecycle_event_uid, canonical_event_uid,
                    display_object_id, binding_id, source_epoch_id,
                    raw_shigure_object_id, action, presence_before,
                    presence_after, presence_epoch, model_revision,
                    pose_revision, pose_aruco_json, source_stamp_json,
                    occurred_at
                ) VALUES (
                    'legacy-lifecycle', 'legacy-event', 'display-1',
                    'legacy-binding', 'legacy-epoch', 'legacy-raw',
                    'TAKE_OUT', 'PRESENT', 'ABSENT', 1, 1, 1, ?,
                    '{{}}', '2026-02-01T00:00:00+00:00'
                )
                """,
                (json.dumps(_pose(99.0)),),
            )
        connection.close()
        task_db._SCHEMA_INITIALIZED = False
        exit_code = migrate_shigure_v3_data.main(
            ["--database", str(self.database_path), "--apply"]
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue(
            self.database_path.with_name(
                self.database_path.name + migrate_shigure_v3_data.BACKUP_SUFFIX
            ).is_file()
        )
        self.assertEqual(migrate_shigure_v3_data._schema_version(self.database_path), 3)
        task_db.DATABASE_PATH = self.database_path
        task_db._SCHEMA_INITIALIZED = False
        origins = task_db.list_display_object_origins("display-1", limit=5)
        self.assertEqual(len(origins), 5)
        self.assertTrue(all(row["kind"] == "INITIALIZATION" for row in origins))
        positions = [
            json.loads(row["pose_aruco_json"])["position"][0]
            for row in reversed(origins)
        ]
        self.assertEqual(positions, [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertNotIn(99.0, positions)

    def test_v3_origin_repair_merges_history_and_preserves_native_origin(self) -> None:
        self._initialize_display()
        with task_db._get_connection() as connection:
            for index, x_position in enumerate((0.0, 2.0), 1):
                connection.execute(
                    f"""
                    INSERT INTO {task_db.DISPLAY_OBJECT_POSE_HISTORY_TABLE} (
                        display_object_id, task_id, model_revision,
                        pose_revision, pose_aruco_json, captured_at
                    ) VALUES ('display-1', ?, 1, ?, ?, ?)
                    """,
                    (
                        f"old-capture-{index}",
                        index,
                        json.dumps(_pose(x_position)),
                        f"2026-01-0{index}T00:00:00+00:00",
                    ),
                )
        native = task_db.add_display_object_origin(
            "display-1",
            1,
            _pose(4.0),
            "TAKE_OUT",
            canonical_event_uid="native-event",
            occurred_at="2026-01-03T00:00:00+00:00",
        )
        with task_db._get_connection() as connection:
            connection.execute(
                f"""
                INSERT INTO {task_db.DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE} (
                    origin_uid, display_object_id, kind, model_revision,
                    pose_aruco_json, occurred_at, detail_json
                ) VALUES (
                    'unsafe-origin', 'display-1', 'TAKE_OUT', 1, ?,
                    '2026-01-04T00:00:00+00:00',
                    '{{"migrated_from":"object_lifecycle_events"}}'
                )
                """,
                (json.dumps(_pose(99.0)),),
            )

        task_db._SCHEMA_INITIALIZED = False
        result = task_db.repair_v3_origin_history_from_hololens_once()
        self.assertEqual(result["status"], "repaired")
        self.assertEqual(result["removed_unsafe_v2_take_out_origins"], 1)
        origins = task_db.list_display_object_origins("display-1", limit=5)
        positions = [
            json.loads(row["pose_aruco_json"])["position"][0]
            for row in reversed(origins)
        ]
        self.assertEqual(positions, [0.0, 2.0, 4.0])
        self.assertIn(native["origin_uid"], {row["origin_uid"] for row in origins})
        first_ids = [row["id"] for row in origins]

        task_db._SCHEMA_INITIALIZED = False
        replay = task_db.repair_v3_origin_history_from_hololens_once()
        self.assertEqual(replay["status"], "already_repaired")
        self.assertEqual(
            [
                row["id"]
                for row in task_db.list_display_object_origins("display-1", limit=5)
            ],
            first_ids,
        )

    def test_v3_origin_repair_cli_uses_independent_verified_backup(self) -> None:
        self._initialize_display()
        with task_db._get_connection() as connection:
            connection.execute(
                f"""
                INSERT INTO {task_db.DISPLAY_OBJECT_POSE_HISTORY_TABLE} (
                    display_object_id, task_id, model_revision,
                    pose_revision, pose_aruco_json, captured_at
                ) VALUES (
                    'display-1', 'old-capture', 1, 1, ?,
                    '2026-01-01T00:00:00+00:00'
                )
                """,
                (json.dumps(_pose(0.0)),),
            )
        task_db._SCHEMA_INITIALIZED = False
        exit_code = migrate_shigure_v3_data.main(
            [
                "--database",
                str(self.database_path),
                "--repair-origin-history",
                "--apply",
            ]
        )
        self.assertEqual(exit_code, 0)
        backup = self.database_path.with_name(
            self.database_path.name
            + migrate_shigure_v3_data.ORIGIN_REPAIR_BACKUP_SUFFIX
        )
        self.assertTrue(backup.is_file())
        metadata = json.loads(
            backup.with_name(
                backup.name + migrate_shigure_v3_data.BACKUP_METADATA_SUFFIX
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["source_schema_version"], 3)
        self.assertEqual(metadata["target_schema_version"], 3)
        self.assertEqual(
            migrate_shigure_v3_data.main(
                [
                    "--database",
                    str(self.database_path),
                    "--repair-origin-history",
                    "--apply",
                ]
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
