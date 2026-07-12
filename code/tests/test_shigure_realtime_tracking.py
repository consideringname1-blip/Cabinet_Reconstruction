from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import shigure_realtime_tracking as realtime_module  # noqa: E402
from realtime_tracking import RealtimeTrackingCoordinator  # noqa: E402
from shigure_realtime_tracking import (  # noqa: E402
    ShigureRealtimeTrackingEngine,
    _score_projected_identity_candidate,
)
from stages.shigure_history.cache import CachedRgbdSample, CachedShigureEvent, RosStamp  # noqa: E402


def _mask_b64(mask: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("mask encode failed")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _sample() -> CachedRgbdSample:
    return CachedRgbdSample(
        stamp=RosStamp(100, 0),
        rgb_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=np.full((10, 10), 1000, dtype=np.uint16),
        camera_info_path=None,
        camera_info={"k": [100, 0, 5, 0, 100, 5, 0, 0, 1]},
    )


class _Cache:
    def __init__(self, sample: CachedRgbdSample) -> None:
        self.sample = sample

    def get_sample(self, *_args, **_kwargs):
        return self.sample


class ShigureGeometryIdentityTests(unittest.TestCase):
    def test_projection_gate_requires_eighty_percent_and_center_depth(self) -> None:
        sample = _sample()
        observed = np.zeros((10, 10), dtype=bool)
        observed[0, :10] = True
        circle = np.zeros_like(observed)
        circle[0, :8] = True

        accepted = _score_projected_identity_candidate(
            sample,
            observed,
            circle,
            {"center_depth_m": 1.0},
        )
        too_little = _score_projected_identity_candidate(
            sample,
            observed,
            np.pad(np.ones((1, 7), dtype=bool), ((0, 9), (0, 3))),
            {"center_depth_m": 1.0},
        )
        wrong_depth = _score_projected_identity_candidate(
            sample,
            observed,
            circle,
            {"center_depth_m": 1.3},
        )

        self.assertTrue(accepted["accepted"])
        self.assertAlmostEqual(0.8, accepted["mask_inside_diag_circle_ratio"])
        self.assertIn("mask_not_enough_inside_model_diag_circle", too_little["reject_reasons"])
        self.assertIn("depth_too_different_from_model_center", wrong_depth["reject_reasons"])

    def _engine(self):
        coordinator = RealtimeTrackingCoordinator(max_objects=5)
        coordinator.mode_status("startup-a")
        coordinator.activate_display_object("display-a")
        dino_calls: list[dict] = []
        engine = ShigureRealtimeTrackingEngine(
            coordinator=coordinator,
            dino_request=lambda payload: dino_calls.append(payload) or {"embedding": [1.0, 0.0]},
            foundationpose_request=lambda _payload, _display: {"ok": False},
        )
        engine.cache = _Cache(_sample())
        return coordinator, engine, dino_calls

    def test_single_geometry_candidate_binds_without_dino_then_session_id_is_trusted(self) -> None:
        _coordinator, engine, dino_calls = self._engine()
        event = CachedShigureEvent(
            source_stamp=RosStamp(100, 0),
            received_utc="now",
            received_monotonic=1.0,
            contacted_state="present",
            object_detection_state="present",
            sequence=1,
        )
        detection = {"action": "obj_move", "mask_b64": _mask_b64(np.ones((10, 10), dtype=bool))}
        geometry = [{"display_object_id": "display-a", "reference_id": "task-a", "geometry_score": 1.0}]
        with patch.object(engine, "_geometry_identity_candidates", return_value=(geometry, [{"accepted": True}])):
            display_id, result, _sample_value, _mask = engine._identity_for_detection(
                startup="startup-a",
                event=event,
                detection=detection,
                shigure_object_id="shigure-1",
            )
        self.assertEqual("display-a", display_id)
        self.assertTrue(result["dino_skipped"])
        self.assertEqual([], dino_calls)

        with patch.object(engine, "_geometry_identity_candidates", side_effect=AssertionError("must use binding fast path")):
            display_id2, result2, *_ = engine._identity_for_detection(
                startup="startup-a",
                event=event,
                detection=detection,
                shigure_object_id="shigure-1",
            )
        self.assertEqual("display-a", display_id2)
        self.assertEqual("trusted_current_shigure_session_binding", result2["reason"])

    def test_strong_new_temporary_id_replaces_old_reverse_binding(self) -> None:
        _coordinator, engine, _dino_calls = self._engine()
        event = CachedShigureEvent(
            source_stamp=RosStamp(100, 0),
            received_utc="now",
            received_monotonic=1.0,
            contacted_state="present",
            object_detection_state="present",
            sequence=3,
        )
        detection = {"action": "obj_move", "mask_b64": _mask_b64(np.ones((10, 10), dtype=bool))}
        geometry = [{"display_object_id": "display-a", "reference_id": "task-a", "geometry_score": 1.0}]
        with patch.object(engine, "_geometry_identity_candidates", return_value=(geometry, [{"accepted": True}])):
            first, *_ = engine._identity_for_detection(
                startup="startup-a",
                event=event,
                detection=detection,
                shigure_object_id="shigure-old",
            )
            second, result, *_ = engine._identity_for_detection(
                startup="startup-a",
                event=event,
                detection=detection,
                shigure_object_id="shigure-new",
            )

        self.assertEqual("display-a", first)
        self.assertEqual("display-a", second)
        self.assertEqual("shigure-old", result["replaced_shigure_object_id"])
        self.assertEqual("UNBOUND", engine.bindings.get("startup-a", "shigure-old").status)
        self.assertEqual(
            "shigure-new",
            engine.bindings.get_by_display_object("startup-a", "display-a").key.shigure_object_id,
        )

    def test_reanchor_epoch_releases_existing_temporary_binding(self) -> None:
        coordinator, engine, _dino_calls = self._engine()
        engine._session()
        engine.bindings.bind("startup-a", "shigure-old", "display-a")
        coordinator.activate_display_object("display-a", reanchor=True)

        engine._session()

        record = engine.bindings.get("startup-a", "shigure-old")
        self.assertIsNotNone(record)
        self.assertEqual("UNBOUND", record.status)
        self.assertEqual("coordinator_tracking_epoch_changed", record.reason)

    def test_explicit_empty_contact_records_wrong_for_geometry_matched_display_object(self) -> None:
        _coordinator, engine, _calls = self._engine()
        event = CachedShigureEvent(
            source_stamp=RosStamp(100, 0),
            received_utc="now",
            received_monotonic=1.0,
            contacted_state="explicit_empty",
            object_detection_state="present",
            object_detection={
                "objects": [
                    {"action": "obj_move", "object_id": "obj_move:0", "mask_b64": _mask_b64(np.ones((10, 10), dtype=bool))}
                ]
            },
            sequence=2,
        )
        recorded: list[dict] = []
        with (
            patch.object(
                engine,
                "_identity_for_detection",
                return_value=("display-a", {"status": "MATCHED"}, _sample(), np.ones((10, 10), dtype=bool)),
            ),
            patch.object(realtime_module, "get_display_object_state", return_value={"active_model_revision": 3}),
            patch.object(realtime_module, "record_realtime_tracking_event", side_effect=lambda **kwargs: recorded.append(kwargs)),
        ):
            engine._handle_event("startup-a", event)

        self.assertEqual(1, len(recorded))
        self.assertEqual("WRONG_NO_CONTACTED_PERSON", recorded[0]["status"])
        self.assertEqual("display-a", recorded[0]["display_object_id"])


if __name__ == "__main__":
    unittest.main()
