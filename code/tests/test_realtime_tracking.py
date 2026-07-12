from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from realtime_tracking import MODE_HISTORY, MODE_LIVE, RealtimeTrackingCoordinator  # noqa: E402


class RealtimeTrackingCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinator = RealtimeTrackingCoordinator(max_objects=5)
        self.coordinator.mode_status("startup-a")
        self.coordinator.activate_display_object("display-a")

    def _submit(self, value: int):
        return self.coordinator.submit_observation(
            startup_session_id="startup-a",
            display_object_id="display-a",
            model_revision=3,
            hololens_pose_revision=7,
            payload={"value": value},
        )

    def test_running_job_continues_but_pending_queue_keeps_only_latest(self) -> None:
        token1 = self._submit(1)
        running = self.coordinator.take_next_pending(timeout=0.0)
        self.assertIsNotNone(token1)
        self.assertEqual(token1, running.token)

        token2 = self._submit(2)
        token3 = self._submit(3)
        self.assertIsNotNone(token2)
        self.assertIsNotNone(token3)
        self.assertFalse(
            self.coordinator.should_accept(
                token1,
                active_model_revision=3,
                active_hololens_pose_revision=7,
            )
        )
        self.assertEqual(3, self.coordinator.snapshot()["objects"]["display-a"]["pending_seq"])

        self.coordinator.finish(token1, accepted=False)
        latest = self.coordinator.take_next_pending(timeout=0.0)
        self.assertEqual(token3, latest.token)
        self.assertEqual(3, latest.payload["value"])

    def test_pause_invalidates_running_and_resume_waits_for_fresh_latest_event(self) -> None:
        token1 = self._submit(1)
        running = self.coordinator.take_next_pending(timeout=0.0)
        self.assertEqual(token1, running.token)

        paused = self.coordinator.set_mode(
            startup_session_id="startup-a",
            mode=MODE_HISTORY,
            request_generation=1,
        )
        self.assertEqual(MODE_HISTORY, paused["mode"])
        self.assertFalse(
            self.coordinator.should_accept(
                token1,
                active_model_revision=3,
                active_hololens_pose_revision=7,
            )
        )
        self.assertIsNone(self._submit(2))

        resumed = self.coordinator.set_mode(
            startup_session_id="startup-a",
            mode=MODE_LIVE,
            request_generation=2,
        )
        self.assertEqual(MODE_LIVE, resumed["mode"])
        self.coordinator.finish(token1, accepted=False)
        self.assertIsNone(self.coordinator.take_next_pending(timeout=0.0))

        # The realtime event reader performs one latest-only cache scan after
        # resume and submits that current observation.
        latest_token = self._submit(3)
        latest = self.coordinator.take_next_pending(timeout=0.0)
        self.assertIsNotNone(latest)
        self.assertEqual(latest_token, latest.token)
        self.assertEqual(3, latest.payload["value"])
        self.assertNotEqual(token1.tracking_epoch, latest.token.tracking_epoch)

    def test_out_of_order_mode_request_cannot_roll_state_back(self) -> None:
        paused = self.coordinator.set_mode(
            startup_session_id="startup-a",
            mode=MODE_HISTORY,
            request_generation=10,
        )
        stale = self.coordinator.set_mode(
            startup_session_id="startup-a",
            mode=MODE_LIVE,
            request_generation=9,
        )
        conflict = self.coordinator.set_mode(
            startup_session_id="startup-a",
            mode=MODE_LIVE,
            request_generation=10,
        )

        self.assertEqual(MODE_HISTORY, paused["mode"])
        self.assertEqual(MODE_HISTORY, stale["mode"])
        self.assertEqual("stale_request_generation", stale["request_ignored_reason"])
        self.assertEqual(MODE_HISTORY, conflict["mode"])
        self.assertEqual("generation_mode_conflict", conflict["request_ignored_reason"])

    def test_commit_if_current_rejects_result_after_newer_observation(self) -> None:
        token1 = self._submit(1)
        running = self.coordinator.take_next_pending(timeout=0.0)
        self.assertEqual(token1, running.token)
        self._submit(2)
        commits: list[str] = []

        result = self.coordinator.commit_if_current(
            token1,
            active_model_revision=3,
            active_hololens_pose_revision=7,
            commit=lambda: commits.append("committed") or {"ok": True},
        )

        self.assertIsNone(result)
        self.assertEqual([], commits)

    def test_reanchor_invalidates_old_model_work(self) -> None:
        token = self._submit(1)
        self.coordinator.take_next_pending(timeout=0.0)
        self.coordinator.activate_display_object("display-a", reanchor=True)
        self.assertFalse(
            self.coordinator.should_accept(
                token,
                active_model_revision=3,
                active_hololens_pose_revision=7,
            )
        )

    def test_new_hololens_capture_revision_rejects_old_foundationpose(self) -> None:
        token = self._submit(1)
        self.coordinator.take_next_pending(timeout=0.0)
        self.assertFalse(
            self.coordinator.should_accept(
                token,
                active_model_revision=3,
                active_hololens_pose_revision=8,
            )
        )


if __name__ == "__main__":
    unittest.main()
