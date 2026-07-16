from __future__ import annotations

import base64
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
HOLOLENS_STAGE_ROOT = (
    CODE_ROOT / "stages" / "hololens3d_reconstruction"
)
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(HOLOLENS_STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(HOLOLENS_STAGE_ROOT))

from coordinate_systems import UNITY_TO_OPENCV_CAMERA_BASIS
from stages.hololens3d_reconstruction.run_historical_model_match_from_json import (
    _candidate_distance,
)
from stages.shigure_history.cache import (
    CachedRgbdSample,
    CachedShigureFrame,
    RosStamp,
)
from stages.shigure_history.shigure_runtime_v2 import (
    EventArtifacts,
    LifecyclePoseCandidate,
    ShigureRuntimeEngine,
    _global_identity_assignment,
)


class ShigureMaskAabbTest(unittest.TestCase):
    @staticmethod
    def _lifecycle_candidate(
        action: str,
        *,
        stamp_seconds: float,
        center_x: float,
    ) -> LifecyclePoseCandidate:
        is_take_out = action == "take_out"
        return LifecyclePoseCandidate(
            action=action,
            canonical_event_uid=f"event-{action}-{stamp_seconds}",
            display_object_id="display-1",
            raw_id=("raw-old" if is_take_out else "raw-new"),
            binding_id=("binding-old" if is_take_out else None),
            resolution_method=(
                "TRUSTED_EPOCH_BINDING" if is_take_out else "BRING_IN_DINO"
            ),
            sequence=(1 if is_take_out else 2),
            stamp_seconds=stamp_seconds,
            source_generation=7,
            source_epoch_id="epoch-1",
            artifacts=EventArtifacts(None, None, None, None, None),
            mask_center_aruco=np.asarray(
                [center_x, 0.0, 0.0], dtype=np.float64
            ),
            dino_distance=None,
            identity={},
            skeleton=None,
            calibration_revision=None,
            marker_rotation=None,
            marker_translation=None,
            occurred_at="2026-07-15T00:00:00+00:00",
        )

    @staticmethod
    def _lifecycle_engine() -> ShigureRuntimeEngine:
        engine = object.__new__(ShigureRuntimeEngine)
        engine._source_generation = 7
        engine.source_epoch_id = "epoch-1"
        engine.runtime_session_id = "session-1"
        engine._lock = threading.RLock()
        engine._lifecycle_pose_windows = {}
        engine._view_windows = {}
        engine._spatial_boxes = {}
        engine._stable = {}
        engine._trusted_mask_observations = {}
        engine._last_stable_source_key = {}
        engine._last_view_sequence = {}
        engine._pose_executor = Mock()
        engine._initial_pose_attempts = {}
        engine._initial_pose_inflight = set()
        engine._initial_pose_completed = set()
        engine._initial_pose_failed = set()
        return engine

    @staticmethod
    def _visibility_observation(
        *,
        sequence: int,
        stamp_seconds: float,
        mask_slice: tuple[slice, slice] = (slice(2, 14), slice(2, 14)),
        depth_mm: int = 1000,
        identity_distance: float | None = 0.10,
    ):
        depth = np.full((16, 16), depth_mm, dtype=np.uint16)
        sample = CachedRgbdSample(
            stamp=RosStamp(
                int(stamp_seconds),
                int(round((stamp_seconds % 1.0) * 1_000_000_000)),
            ),
            rgb_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
            depth=depth,
            camera_info={
                "k": [100.0, 0.0, 7.5, 0.0, 100.0, 7.5, 0.0, 0.0, 1.0],
                "width": 16,
                "height": 16,
            },
        )
        mask = np.zeros((16, 16), dtype=bool)
        mask[mask_slice] = True
        return ShigureRuntimeEngine._build_trusted_mask_observation(
            display_object_id="display-1",
            binding_id="binding-old",
            raw_id="raw-old",
            sequence=sequence,
            stamp_seconds=stamp_seconds,
            sample=sample,
            mask=mask,
            identity_distance=identity_distance,
        )

    @staticmethod
    def _post_depth_sample(stamp_seconds: float, depth_mm: int):
        observation = ShigureMaskAabbTest._visibility_observation(
            sequence=99,
            stamp_seconds=stamp_seconds,
            depth_mm=depth_mm,
        )
        return observation.artifacts.sample

    @staticmethod
    def _startup_mask_b64() -> str:
        mask = np.full((4, 4), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            raise AssertionError("failed to encode startup test mask")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @classmethod
    def _startup_frame(
        cls,
        *,
        candidates: list[dict] | None = None,
        segments: str = "present",
        object_tracking: str = "present",
    ) -> CachedShigureFrame:
        return CachedShigureFrame(
            source_stamp=RosStamp(100, 1),
            source_incarnation_id="incarnation-1",
            frame_id="camera",
            received_utc="2026-07-15T00:00:00+00:00",
            received_monotonic=100.0,
            schema_version=2,
            input_states={
                "segments": segments,
                "object_tracking": object_tracking,
            },
            events=[],
            tracked_objects=[],
            recovery_candidates=(
                candidates
                if candidates is not None
                else [
                    {
                        "candidate_id": "candidate-1",
                        "shigure_object_id": "raw-1",
                        "tracking_match_status": "RESOLVED",
                        "mask_b64": cls._startup_mask_b64(),
                        "bbox_xyxy": [0, 0, 4, 4],
                    }
                ]
            ),
            people=[],
            diagnostics=[],
            sequence=1,
        )

    @staticmethod
    def _startup_sample() -> CachedRgbdSample:
        return CachedRgbdSample(
            stamp=RosStamp(100, 1),
            rgb_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=np.full((4, 4), 1000, dtype=np.uint16),
            camera_info={
                "k": [100.0, 0.0, 1.5, 0.0, 100.0, 1.5, 0.0, 0.0, 1.0],
                "width": 4,
                "height": 4,
            },
        )

    def test_lifecycle_0199m_no_move_has_no_authoritative_side_effects(
        self,
    ) -> None:
        engine = self._lifecycle_engine()
        take_out = self._lifecycle_candidate(
            "take_out", stamp_seconds=10.0, center_x=0.0
        )
        bring_in = self._lifecycle_candidate(
            "bring_in", stamp_seconds=10.5, center_x=0.199
        )

        with (
            patch.object(engine, "_reject_lifecycle_candidates") as reject,
            patch.object(
                engine, "_foundationpose_pose_for_lifecycle"
            ) as foundationpose,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "apply_object_lifecycle_event"
            ) as apply_lifecycle,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "establish_shigure_binding"
            ) as establish_binding,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "commit_pending_take_out_lifecycle_event"
            ) as commit_take_out,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "commit_pending_bring_in_lifecycle_event"
            ) as commit_bring_in,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "add_display_object_origin"
            ) as add_origin,
        ):
            engine._run_lifecycle_pose_window(
                "display-1", [take_out, bring_in]
            )

        apply_lifecycle.assert_not_called()
        establish_binding.assert_not_called()
        commit_take_out.assert_not_called()
        commit_bring_in.assert_not_called()
        foundationpose.assert_not_called()
        add_origin.assert_not_called()
        engine._pose_executor.submit.assert_not_called()
        self.assertEqual(
            reject.call_args.kwargs["reason"],
            "NO_MOVE_MASK_CENTER_LT_20CM",
        )

    def test_duplicate_lifecycle_event_keeps_better_dino_observation(
        self,
    ) -> None:
        engine = self._lifecycle_engine()
        frame = Mock(sequence=1)
        frame.source_stamp = Mock(sec=10, nanosec=0)
        artifacts = EventArtifacts(None, None, None, None, None)

        with patch.object(
            engine,
            "_masked_center_aruco",
            return_value=np.asarray([0.1, 0.0, 0.0], dtype=np.float64),
        ):
            for distance in (0.30, 0.10):
                engine._queue_lifecycle_pose_candidate(
                    action="take_out",
                    canonical_event_uid="same-event",
                    display_object_id="display-1",
                    raw_id="raw-old",
                    binding_id="binding-old",
                    resolution_method="TAKE_OUT_WINDOW",
                    identity={"distance": distance},
                    skeleton=None,
                    calibration_revision=None,
                    marker_rotation=None,
                    marker_translation=None,
                    occurred_at="2026-07-15T00:00:00+00:00",
                    frame=frame,
                    artifacts=artifacts,
                )

        candidates = engine._lifecycle_pose_windows["display-1"][
            "candidates"
        ]
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].dino_distance, 0.10)

    def test_lifecycle_020m_boundary_is_not_no_move(self) -> None:
        engine = self._lifecycle_engine()
        take_out = self._lifecycle_candidate(
            "take_out", stamp_seconds=10.0, center_x=0.0
        )
        bring_in = self._lifecycle_candidate(
            "bring_in", stamp_seconds=10.5, center_x=0.20
        )
        early_bring_in = self._lifecycle_candidate(
            "bring_in", stamp_seconds=9.9, center_x=0.80
        )
        old_binding = {
            "binding_id": "binding-old",
            "display_object_id": "display-1",
            "raw_shigure_object_id": "raw-old",
        }
        new_binding = {
            "binding_id": "binding-new",
            "display_object_id": "display-1",
            "raw_shigure_object_id": "raw-new",
        }

        commit_order = []

        def commit_take_out(*args, **kwargs):
            commit_order.append("take_out")
            return {"lifecycle_event": {"model_revision": 3}}

        def commit_bring_in(*args, **kwargs):
            commit_order.append("bring_in")
            return {"binding": new_binding, "lifecycle_event": {}}

        with (
            patch.object(engine, "_reject_lifecycle_candidates") as reject,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "get_active_shigure_binding",
                return_value=old_binding,
            ),
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "list_active_shigure_bindings",
                return_value=[],
            ),
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "commit_pending_take_out_lifecycle_event",
                side_effect=commit_take_out,
            ) as take_out_transaction,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "commit_pending_bring_in_lifecycle_event",
                side_effect=commit_bring_in,
            ) as bring_in_transaction,
            patch.object(
                engine,
                "_confirm_take_out_movement",
                return_value=Mock(
                    status="REAL_MOVE",
                    reason="TEST_CONFIRMED",
                    source=Mock(
                        artifacts=take_out.artifacts,
                        identity_distance=0.10,
                        summary=Mock(return_value={}),
                    ),
                    detail={},
                    report_path=None,
                ),
            ),
        ):
            engine._run_lifecycle_pose_window(
                "display-1", [early_bring_in, take_out, bring_in]
            )

        self.assertAlmostEqual(
            ShigureRuntimeEngine._lifecycle_movement_distance(
                take_out, bring_in
            ),
            0.20,
        )
        take_out_transaction.assert_called_once()
        bring_in_transaction.assert_called_once()
        self.assertEqual(
            bring_in_transaction.call_args.args[0],
            bring_in.canonical_event_uid,
        )
        self.assertEqual(commit_order, ["take_out", "bring_in"])
        engine._pose_executor.submit.assert_called_once()
        self.assertNotIn(
            "NO_MOVE_MASK_CENTER_LT_20CM",
            [call.kwargs.get("reason") for call in reject.call_args_list],
        )

    def test_lifecycle_source_time_pairing_includes_exactly_one_second(
        self,
    ) -> None:
        take_out = self._lifecycle_candidate(
            "take_out", stamp_seconds=10.0, center_x=0.0
        )
        exactly_one_second = self._lifecycle_candidate(
            "bring_in", stamp_seconds=11.0, center_x=0.1
        )
        over_one_second = self._lifecycle_candidate(
            "bring_in", stamp_seconds=11.000001, center_x=0.1
        )

        self.assertAlmostEqual(
            ShigureRuntimeEngine._lifecycle_movement_distance(
                take_out, exactly_one_second
            ),
            0.1,
        )
        self.assertIsNone(
            ShigureRuntimeEngine._lifecycle_movement_distance(
                take_out, over_one_second
            )
        )

    def test_takeout_source_uses_latest_complete_pre_event_frame(self) -> None:
        engine = self._lifecycle_engine()
        complete_old = self._visibility_observation(
            sequence=4, stamp_seconds=8.0
        )
        complete_latest = self._visibility_observation(
            sequence=5, stamp_seconds=8.5
        )
        partial_near_event = self._visibility_observation(
            sequence=6,
            stamp_seconds=9.5,
            mask_slice=(slice(5, 11), slice(5, 11)),
        )
        engine._trusted_mask_observations["display-1"] = deque(
            [complete_old, complete_latest, partial_near_event]
        )
        take_out = replace(
            self._lifecycle_candidate(
                "take_out", stamp_seconds=10.0, center_x=0.0
            ),
            sequence=10,
        )

        selected, reports = engine._select_clear_take_out_source(take_out)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.sequence, complete_latest.sequence)
        self.assertIs(
            selected.artifacts.sample, complete_latest.artifacts.sample
        )
        partial_report = next(
            row for row in reports if row["sequence"] == 6
        )
        self.assertIn(
            "mask_incomplete_vs_prewindow",
            partial_report["rejection_reasons"],
        )

    def test_depth_change_distinguishes_removal_occlusion_and_overlap(self) -> None:
        source = self._visibility_observation(
            sequence=5, stamp_seconds=9.0, depth_mm=1000
        )
        revealed = ShigureRuntimeEngine._depth_change_evidence(
            source, self._post_depth_sample(10.5, 1200)
        )
        occluded = ShigureRuntimeEngine._depth_change_evidence(
            source, self._post_depth_sample(10.6, 800)
        )
        overlapping_object = ShigureRuntimeEngine._depth_change_evidence(
            source, self._post_depth_sample(10.7, 1010)
        )

        self.assertGreater(revealed["background_reveal_ratio"], 0.99)
        self.assertGreater(occluded["foreground_occlusion_ratio"], 0.99)
        self.assertGreater(overlapping_object["unchanged_ratio"], 0.99)

    def test_takeout_confirmation_requires_repeated_background_reveal(self) -> None:
        engine = self._lifecycle_engine()
        source = self._visibility_observation(
            sequence=5, stamp_seconds=9.0
        )
        engine._trusted_mask_observations["display-1"] = deque(
            [
                replace(source, sequence=4, stamp_seconds=8.5),
                source,
            ]
        )
        take_out = replace(
            self._lifecycle_candidate(
                "take_out", stamp_seconds=10.0, center_x=0.0
            ),
            sequence=10,
        )
        post_samples = [
            (11, 10.5, self._post_depth_sample(10.5, 1200)),
            (12, 11.0, self._post_depth_sample(11.0, 1200)),
        ]

        with patch.object(
            engine,
            "_persist_lifecycle_confirmation_report",
            return_value=None,
        ):
            confirmation = engine._confirm_take_out_movement(
                take_out,
                None,
                None,
                post_samples=post_samples,
                post_observations=(),
            )

        self.assertEqual(confirmation.status, "REAL_MOVE")
        self.assertEqual(
            confirmation.reason, "BACKGROUND_REVEALED_AFTER_TAKE_OUT"
        )
        self.assertIs(
            confirmation.source.artifacts.sample, source.artifacts.sample
        )

    def test_takeout_confirmation_suppresses_person_and_overlap_false_positive(self) -> None:
        engine = self._lifecycle_engine()
        source = self._visibility_observation(
            sequence=5, stamp_seconds=9.0
        )
        engine._trusted_mask_observations["display-1"] = deque(
            [
                replace(source, sequence=4, stamp_seconds=8.5),
                source,
            ]
        )
        take_out = replace(
            self._lifecycle_candidate(
                "take_out", stamp_seconds=10.0, center_x=0.0
            ),
            sequence=10,
        )

        with patch.object(
            engine,
            "_persist_lifecycle_confirmation_report",
            return_value=None,
        ):
            overlap = engine._confirm_take_out_movement(
                take_out,
                None,
                None,
                post_samples=[
                    (11, 10.5, self._post_depth_sample(10.5, 1010)),
                    (12, 11.0, self._post_depth_sample(11.0, 1010)),
                ],
                post_observations=(),
            )
            foreground = engine._confirm_take_out_movement(
                take_out,
                None,
                None,
                post_samples=[
                    (11, 10.5, self._post_depth_sample(10.5, 800)),
                    (12, 11.0, self._post_depth_sample(11.0, 800)),
                ],
                post_observations=(),
            )

        self.assertEqual(overlap.status, "NO_MOVE")
        self.assertEqual(
            overlap.reason, "OBJECT_DEPTH_REMAINED_AT_ORIGINAL_POSITION"
        )
        self.assertEqual(foreground.status, "AMBIGUOUS")
        self.assertEqual(
            foreground.reason, "FOREGROUND_OCCLUSION_NOT_REAL_MOVEMENT"
        )

    def test_takeout_without_clear_identity_source_fails_closed(self) -> None:
        engine = self._lifecycle_engine()
        take_out = replace(
            self._lifecycle_candidate(
                "take_out", stamp_seconds=10.0, center_x=0.0
            ),
            sequence=10,
        )
        with patch.object(
            engine,
            "_persist_lifecycle_confirmation_report",
            return_value=None,
        ):
            confirmation = engine._confirm_take_out_movement(
                take_out,
                None,
                None,
                post_samples=(),
                post_observations=(),
            )
        self.assertEqual(confirmation.status, "AMBIGUOUS")
        self.assertEqual(
            confirmation.reason,
            "NO_CLEAR_IDENTITY_MATCHED_PRE_TAKEOUT_FRAME",
        )

    def test_ambiguous_takeout_never_commits_lifecycle_or_pose(self) -> None:
        engine = self._lifecycle_engine()
        take_out = self._lifecycle_candidate(
            "take_out", stamp_seconds=10.0, center_x=0.0
        )
        old_binding = {
            "binding_id": "binding-old",
            "display_object_id": "display-1",
            "raw_shigure_object_id": "raw-old",
        }
        with (
            patch.object(engine, "_reject_lifecycle_candidates") as reject,
            patch.object(
                engine,
                "_confirm_take_out_movement",
                return_value=Mock(
                    status="AMBIGUOUS",
                    reason="FOREGROUND_OCCLUSION_NOT_REAL_MOVEMENT",
                    source=None,
                    detail={"foreground_frame_count": 2},
                    report_path=Path("/tmp/report.json"),
                ),
            ),
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "get_active_shigure_binding",
                return_value=old_binding,
            ),
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "commit_pending_take_out_lifecycle_event"
            ) as commit_take_out,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "commit_pending_bring_in_lifecycle_event"
            ) as commit_bring_in,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "apply_object_lifecycle_event"
            ) as apply_lifecycle,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "add_display_object_origin"
            ) as add_origin,
        ):
            engine._run_lifecycle_pose_window("display-1", [take_out])

        commit_take_out.assert_not_called()
        commit_bring_in.assert_not_called()
        apply_lifecycle.assert_not_called()
        add_origin.assert_not_called()
        engine._pose_executor.submit.assert_not_called()
        self.assertIn(
            "FOREGROUND_OCCLUSION_NOT_REAL_MOVEMENT",
            [call.kwargs.get("reason") for call in reject.call_args_list],
        )

    def test_trusted_history_is_bounded_5hz_and_shares_rgbd(self) -> None:
        engine = self._lifecycle_engine()
        binding = {
            "binding_id": "binding-old",
            "display_object_id": "display-1",
            "raw_shigure_object_id": "raw-old",
        }
        first_sample = None
        for index in range(11):
            stamp_seconds = 100.0 + index * 0.1
            template = self._visibility_observation(
                sequence=index + 1,
                stamp_seconds=stamp_seconds,
            )
            sample = template.artifacts.sample
            if first_sample is None:
                first_sample = sample
            frame = Mock(
                sequence=index + 1,
                source_stamp=sample.stamp,
                people=[],
            )
            engine._remember_trusted_mask_observation(
                binding=binding,
                frame=frame,
                sample=sample,
                mask=template.artifacts.mask_array,
                identity_distance=0.10,
            )

        history = engine._trusted_mask_observations["display-1"]
        self.assertEqual(history.maxlen, 22)
        self.assertGreaterEqual(len(history), 4)
        self.assertLessEqual(len(history), 6)
        self.assertIs(history[0].artifacts.sample, first_sample)
        self.assertTrue(
            all(
                observation.artifacts.mask_array is not None
                for observation in history
            )
        )

    def test_takeout_confirmation_report_writes_rgb_depth_mask_and_crop(self) -> None:
        engine = self._lifecycle_engine()
        source = self._visibility_observation(
            sequence=5, stamp_seconds=9.0
        )
        take_out = replace(
            self._lifecycle_candidate(
                "take_out", stamp_seconds=10.0, center_x=0.0
            ),
            sequence=10,
            artifacts=source.artifacts,
        )
        post_sample = self._post_depth_sample(10.5, 1200)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                engine,
                "_recovery_debug_epoch_root",
                return_value=root,
            ):
                report_path = engine._persist_lifecycle_confirmation_report(
                    take_out=take_out,
                    source=source,
                    post_samples=[(11, 10.5, post_sample)],
                    detail={"test": True},
                    status="REAL_MOVE",
                    reason="BACKGROUND_REVEALED_AFTER_TAKE_OUT",
                )

            self.assertIsNotNone(report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "REAL_MOVE")
            event_root = report_path.parent
            for name in (
                "event_scene.png",
                "event_depth.png",
                "event_mask.png",
                "event_object_crop.png",
                "selected_source_scene.png",
                "selected_source_depth.png",
                "selected_source_mask.png",
                "selected_source_object_crop.png",
                "latest_post_scene.png",
                "latest_post_depth.png",
            ):
                self.assertTrue((event_root / name).is_file(), name)

    def test_primary_mask_observation_wins_when_still_visible(self) -> None:
        observations = {
            "old": {
                "new_binding": True,
                "identity_distance": 0.20,
                "binding": {
                    "binding_id": "old",
                    "valid_from": "2026-07-15T10:00:00+00:00",
                    "confidence": 0.95,
                }
            },
            "primary": {
                "new_binding": False,
                "identity_distance": 0.10,
                "binding": {
                    "binding_id": "primary",
                    "valid_from": "2026-07-15T09:00:00+00:00",
                    "confidence": 0.80,
                }
            },
        }
        selected = ShigureRuntimeEngine._select_primary_mask_observation(
            observations,
            "primary",
            complete_snapshot=True,
        )
        self.assertIs(selected, observations["primary"])

    def test_lower_dino_new_alias_becomes_primary(self) -> None:
        observations = {
            "primary": {
                "new_binding": False,
                "identity_distance": 0.20,
                "binding": {"binding_id": "primary"},
            },
            "new": {
                "new_binding": True,
                "identity_distance": 0.10,
                "binding": {"binding_id": "new"},
            },
        }
        selected = ShigureRuntimeEngine._select_primary_mask_observation(
            observations,
            "primary",
            complete_snapshot=True,
        )
        self.assertIs(selected, observations["new"])

    def test_equal_dino_new_alias_replaces_visible_old_primary(self) -> None:
        observations = {
            "primary": {
                "new_binding": False,
                "identity_distance": 0.10,
                "binding": {
                    "binding_id": "primary",
                    "valid_from": "2026-07-15T09:00:00+00:00",
                },
            },
            "new": {
                "new_binding": True,
                "identity_distance": 0.10,
                "binding": {
                    "binding_id": "new",
                    "valid_from": "2026-07-15T10:00:00+00:00",
                },
            },
        }
        selected = ShigureRuntimeEngine._select_primary_mask_observation(
            observations,
            "primary",
            complete_snapshot=True,
        )
        self.assertIs(selected, observations["new"])

    def test_startup_recovery_uses_alias_compatible_mask_reconcile(self) -> None:
        engine = self._lifecycle_engine()
        engine._startup_recovery_pending = True
        frame = Mock()
        with (
            patch.object(engine, "_reconcile_mask_bindings") as reconcile,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "_global_identity_assignment"
            ) as retired,
        ):
            engine._maybe_startup_recovery(frame)
        reconcile.assert_called_once_with(frame)
        retired.assert_not_called()

    def test_startup_missing_rgbd_stays_pending_without_identity_or_fp(
        self,
    ) -> None:
        engine = self._lifecycle_engine()
        engine._startup_recovery_pending = True
        engine.cache = Mock()
        engine.cache.get_sample.return_value = None
        frame = self._startup_frame()
        with (
            patch.object(
                engine,
                "_persist_recovery_observation",
                return_value=Path("/tmp/startup-wait"),
            ) as persist,
            patch.object(engine, "_recent_display_ids") as recent,
            patch.object(engine, "_schedule_initial_pose") as schedule,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "upsert_identity_sync_job"
            ) as upsert,
        ):
            engine._reconcile_mask_bindings(frame)

        self.assertTrue(engine._startup_recovery_pending)
        persist.assert_called_once_with(frame, "waiting_for_exact_rgbd")
        recent.assert_not_called()
        schedule.assert_not_called()
        self.assertEqual(upsert.call_args.kwargs["status"], "PENDING")
        self.assertEqual(
            upsert.call_args.kwargs["result"]["reason"],
            "waiting_for_exact_rgbd",
        )

    def test_startup_unresolved_raw_id_does_not_complete(self) -> None:
        engine = self._lifecycle_engine()
        engine._startup_recovery_pending = True
        engine.cache = Mock()
        engine.cache.get_sample.return_value = self._startup_sample()
        candidate = {
            "candidate_id": "candidate-unresolved",
            "shigure_object_id": "",
            "tracking_match_status": "UNRESOLVED",
            "mask_b64": self._startup_mask_b64(),
            "bbox_xyxy": [0, 0, 4, 4],
        }
        frame = self._startup_frame(candidates=[candidate])
        with (
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "_camera_to_aruco",
                return_value=Mock(),
            ),
            patch.object(
                engine,
                "_persist_recovery_observation",
                return_value=Path("/tmp/startup-unresolved"),
            ) as persist,
            patch.object(engine, "_recent_display_ids") as recent,
            patch.object(engine, "_schedule_initial_pose") as schedule,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "upsert_identity_sync_job"
            ) as upsert,
        ):
            engine._reconcile_mask_bindings(frame)

        reason = "waiting_for_resolved_candidate_raw_id:candidate-unresolved"
        self.assertTrue(engine._startup_recovery_pending)
        persist.assert_called_once_with(frame, reason)
        recent.assert_not_called()
        schedule.assert_not_called()
        self.assertEqual(upsert.call_args.kwargs["status"], "PENDING")
        self.assertEqual(upsert.call_args.kwargs["result"]["reason"], reason)

    def test_startup_mixed_segments_processes_only_resolved_subset(
        self,
    ) -> None:
        engine = self._lifecycle_engine()
        engine._startup_recovery_pending = True
        engine.cache = Mock()
        engine.cache.get_sample.return_value = self._startup_sample()
        frame = self._startup_frame(
            candidates=[
                {
                    "candidate_id": "ordinary-untracked-segment",
                    "shigure_object_id": "",
                    "tracking_match_status": "UNRESOLVED",
                    # Unresolved masks are diagnostics only.
                    "mask_b64": "not-an-image",
                    "bbox_xyxy": [0, 0, 2, 2],
                },
                {
                    "candidate_id": "tracked-segment",
                    "shigure_object_id": "raw-1",
                    "tracking_match_status": "RESOLVED",
                    "mask_b64": self._startup_mask_b64(),
                    "bbox_xyxy": [0, 0, 4, 4],
                },
            ]
        )
        binding = {
            "binding_id": "binding-1",
            "display_object_id": "display-1",
            "raw_shigure_object_id": "raw-1",
            "valid_from": "2026-07-15T00:00:00+00:00",
            "confidence": 0.9,
        }
        state = {
            "active_shigure_binding_id": "binding-1",
            "active_model_revision": 1,
            "presence": "PRESENT",
        }

        with (
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "_camera_to_aruco",
                return_value=Mock(),
            ),
            patch.object(engine, "_recent_display_ids", return_value=[]),
            patch.object(
                engine,
                "_write_recovery_input_artifacts",
                return_value=[],
            ),
            patch.object(engine, "_remember_trusted_mask_observation"),
            patch.object(
                engine, "_commit_primary_mask_box", return_value=True
            ),
            patch.object(
                engine,
                "_schedule_initial_pose",
                side_effect=lambda **_kwargs: (
                    engine._initial_pose_completed.add("display-1") or True
                ),
            ) as schedule,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "get_active_shigure_binding",
                return_value=binding,
            ),
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "get_display_object_state",
                return_value=state,
            ),
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "list_active_shigure_bindings",
                return_value=[binding],
            ),
            patch(
                "stages.shigure_history.shigure_runtime_v2._write_json"
            ),
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "upsert_identity_sync_job"
            ) as upsert,
        ):
            engine._reconcile_mask_bindings(frame)

        self.assertFalse(engine._startup_recovery_pending)
        schedule.assert_called_once()
        self.assertEqual(upsert.call_args.kwargs["status"], "COMPLETED")
        report = upsert.call_args.kwargs["result"]
        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["resolved_candidate_count"], 1)
        self.assertEqual(report["ignored_unresolved_candidate_count"], 1)
        result_ids = {
            row.get("candidate_id") for row in report["results"]
        }
        self.assertIn("tracked-segment", result_ids)
        self.assertNotIn("ordinary-untracked-segment", result_ids)

    def test_startup_missing_camera_to_aruco_stays_pending(self) -> None:
        engine = self._lifecycle_engine()
        engine._startup_recovery_pending = True
        engine.cache = Mock()
        engine.cache.get_sample.return_value = self._startup_sample()
        frame = self._startup_frame()
        with (
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "_camera_to_aruco",
                side_effect=FileNotFoundError("calibration missing"),
            ),
            patch.object(
                engine,
                "_persist_recovery_observation",
                return_value=Path("/tmp/startup-calibration"),
            ) as persist,
            patch.object(engine, "_recent_display_ids") as recent,
            patch.object(engine, "_schedule_initial_pose") as schedule,
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "upsert_identity_sync_job"
            ) as upsert,
        ):
            engine._reconcile_mask_bindings(frame)

        reason = "waiting_for_shigure_camera_to_armarker_calibration"
        self.assertTrue(engine._startup_recovery_pending)
        persist.assert_called_once_with(frame, reason)
        recent.assert_not_called()
        schedule.assert_not_called()
        self.assertEqual(upsert.call_args.kwargs["status"], "PENDING")
        self.assertEqual(upsert.call_args.kwargs["result"]["reason"], reason)

    def test_startup_complete_empty_snapshot_completes(self) -> None:
        engine = self._lifecycle_engine()
        engine._startup_recovery_pending = True
        engine.cache = Mock()
        engine.cache.get_sample.return_value = None
        frame = self._startup_frame(
            candidates=[],
            segments="explicit_empty",
            object_tracking="explicit_empty",
        )
        with (
            patch.object(engine, "_recent_display_ids", return_value=[]),
            patch.object(
                engine, "_write_recovery_input_artifacts", return_value=[]
            ),
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "list_active_shigure_bindings",
                return_value=[],
            ),
            patch(
                "stages.shigure_history.shigure_runtime_v2._write_json"
            ),
            patch(
                "stages.shigure_history.shigure_runtime_v2."
                "upsert_identity_sync_job"
            ) as upsert,
        ):
            engine._reconcile_mask_bindings(frame)

        self.assertFalse(engine._startup_recovery_pending)
        self.assertEqual(upsert.call_args.kwargs["status"], "COMPLETED")
        self.assertEqual(
            upsert.call_args.kwargs["result"]["candidate_count"], 0
        )

    def test_independent_assignment_allows_two_raw_aliases(self) -> None:
        assignment = _global_identity_assignment(
            [
                [{"display_object_id": "display-1", "distance": 0.10}],
                [{"display_object_id": "display-1", "distance": 0.12}],
            ]
        )
        self.assertEqual(assignment["matched_count"], 2)
        self.assertEqual(
            {
                row["display_object_id"]
                for row in assignment["assignments"].values()
            },
            {"display-1"},
        )

    def test_zero_dino_distance_is_not_treated_as_missing(self) -> None:
        self.assertEqual(_candidate_distance({"dinov2_distance": 0.0}), 0.0)
        self.assertEqual(_candidate_distance({"dinov2_distance": None}), 999.0)

    def test_complete_snapshot_falls_back_to_newest_visible_alias(self) -> None:
        observations = {
            "older": {
                "binding": {
                    "binding_id": "older",
                    "valid_from": "2026-07-15T09:00:00+00:00",
                    "confidence": 0.99,
                }
            },
            "newer": {
                "binding": {
                    "binding_id": "newer",
                    "valid_from": "2026-07-15T10:00:00+00:00",
                    "confidence": 0.75,
                }
            },
        }
        partial = ShigureRuntimeEngine._select_primary_mask_observation(
            observations,
            "missing-primary",
            complete_snapshot=False,
        )
        selected = ShigureRuntimeEngine._select_primary_mask_observation(
            observations,
            "missing-primary",
            complete_snapshot=True,
        )
        self.assertIsNone(partial)
        self.assertIs(selected, observations["newer"])

    def test_equal_dino_aliases_choose_the_newest_binding(self) -> None:
        observations = {
            "older": {
                "identity_distance": 0.10,
                "binding": {
                    "binding_id": "older",
                    "valid_from": "2026-07-15T09:00:00+00:00",
                },
            },
            "newer": {
                "identity_distance": 0.10,
                "binding": {
                    "binding_id": "newer",
                    "valid_from": "2026-07-15T10:00:00+00:00",
                },
            },
        }
        selected = ShigureRuntimeEngine._select_primary_mask_observation(
            observations,
            "missing-primary",
            complete_snapshot=True,
        )
        self.assertIs(selected, observations["newer"])

    def test_metric_mask_points_form_axis_aligned_box(self) -> None:
        height, width = 10, 12
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.full((height, width), 1000, dtype=np.uint16)
        mask = np.ones((height, width), dtype=bool)
        sample = CachedRgbdSample(
            RosStamp(1, 0),
            rgb,
            depth,
            {
                "k": [[100.0, 0.0, 5.5], [0.0, 100.0, 4.5], [0.0, 0.0, 1.0]],
                "width": width,
                "height": height,
            },
        )
        identity = np.eye(4, dtype=np.float64)
        with patch(
            "stages.shigure_history.shigure_runtime_v2._camera_to_aruco",
            return_value=(identity, np.eye(3), np.zeros(3), "test"),
        ):
            corners = np.asarray(
                ShigureRuntimeEngine._mask_depth_aabb_aruco(sample, mask)
            )

        self.assertEqual(corners.shape, (8, 3))
        np.testing.assert_allclose(corners[:, 2], 1.0, atol=0.005)
        self.assertGreater(float(np.ptp(corners[:, 0])), 0.09)
        self.assertGreater(float(np.ptp(corners[:, 1])), 0.07)

    def test_local_id46_fixture_matches_recorded_aruco_aabb(self) -> None:
        fixture = PROJECT_ROOT / ".test" / "foundationpose_shigure_id46_20260623"
        marker_path = (
            PROJECT_ROOT
            / "data"
            / "aruco"
            / "shigure_marker_history"
            / "history"
            / "1784118381_078602877_marker_6d_pose.json"
        )
        if not fixture.is_dir() or not marker_path.is_file():
            self.skipTest("local Shigure RGB-D/ArUco regression fixture is unavailable")

        rgb = cv2.imread(str(fixture / "rgb.png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(fixture / "depth.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(
            str(fixture / "mask_id46.png"), cv2.IMREAD_GRAYSCALE
        ) > 0
        camera_info = json.loads(
            (fixture / "camera_info.json").read_text(encoding="utf-8")
        )["message"]
        matrix = np.fromstring(
            str(camera_info["k"]).strip("[]"), sep=" "
        ).reshape(3, 3)
        sample = CachedRgbdSample(
            RosStamp(1, 0),
            rgb,
            depth,
            {
                "k": matrix.tolist(),
                "width": int(camera_info["width"]),
                "height": int(camera_info["height"]),
            },
        )

        marker = json.loads(marker_path.read_text(encoding="utf-8"))[
            "opencv_camera_pose"
        ]
        rotation = np.asarray(marker["rotation_matrix"], dtype=np.float64)
        translation = np.asarray(marker["tvec_m"], dtype=np.float64)
        basis = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = basis @ rotation.T
        transform[:3, 3] = basis @ rotation.T @ (-translation)
        with patch(
            "stages.shigure_history.shigure_runtime_v2._camera_to_aruco",
            return_value=(transform, rotation, translation, "fixture"),
        ):
            corners = np.asarray(
                ShigureRuntimeEngine._mask_depth_aabb_aruco(sample, mask)
            )

        np.testing.assert_allclose(
            corners.min(axis=0),
            [-0.9900401, -0.1584722, 1.8147353],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            corners.max(axis=0),
            [-0.7574495, 0.0909692, 2.0177937],
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
