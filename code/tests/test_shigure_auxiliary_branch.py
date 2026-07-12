from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import shigure_auxiliary_branch as auxiliary_module  # noqa: E402
from shigure_auxiliary_branch import (  # noqa: E402
    _freeze_watched_event,
    _load_watched_events,
    _publish_taken_artifacts,
)
from stages.shigure_history.cache import CachedRgbdSample, CachedShigureEvent, RosStamp  # noqa: E402


class ShigureAuxiliaryArtifactTests(unittest.TestCase):
    def test_sparse_watcher_event_survives_rgbd_cache_eviction_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stamp = RosStamp(123, 456)
            event = CachedShigureEvent(
                source_stamp=stamp,
                received_utc="now",
                received_monotonic=1.25,
                contacted_state="present",
                object_detection_state="present",
                contacted={"contacts": [{"action": "take_out", "people_id": "p1"}]},
                object_detection={"objects": [{"action": "take_out", "mask_b64": "mask-data"}]},
                contact_object_matches=[{"status": "matched_action_iou"}],
                sequence=7,
            )
            sample = CachedRgbdSample(
                stamp=stamp,
                rgb_bgr=np.full((4, 5, 3), 17, dtype=np.uint8),
                depth=np.full((4, 5), 1234, dtype=np.uint16),
                camera_info_path=None,
                camera_info={"k": [1, 0, 2, 0, 1, 2, 0, 0, 1]},
            )

            frozen_dir = _freeze_watched_event(root, event, sample)
            restored = _load_watched_events(root)

            self.assertTrue((frozen_dir / "event.json").is_file())
            self.assertEqual(1, len(restored))
            restored_event, restored_sample = next(iter(restored.values()))
            self.assertEqual(7, restored_event.sequence)
            self.assertEqual("mask-data", restored_event.object_detection["objects"][0]["mask_b64"])
            self.assertIsNotNone(restored_sample)
            np.testing.assert_array_equal(sample.rgb_bgr, restored_sample.rgb_bgr)
            np.testing.assert_array_equal(sample.depth, restored_sample.depth)
            self.assertEqual(sample.camera_info, restored_sample.camera_info)

    def test_selected_take_frame_is_published_in_model_result_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backup = root / "backup"
            result = root / "result"
            backup.mkdir()
            sources = {
                "rgb.png": b"rgb",
                "depth.png": b"depth",
                "camera_info.json": json.dumps({"k": [1]}).encode("utf-8"),
                "active_objects.json": b"{}",
                "marker_6d_pose.json": b"{}",
            }
            for name, value in sources.items():
                (backup / name).write_bytes(value)

            def result_file(_timestamp: str, artifact_key: str) -> Path:
                return result / f"{artifact_key}.artifact"

            with patch.object(auxiliary_module, "model_result_file", side_effect=result_file):
                payload = _publish_taken_artifacts("task-time", backup)

            self.assertEqual("model_result", payload["artifact_root"])
            for key in ("result_rgb", "result_depth", "camera_info", "active_objects", "marker_pose"):
                self.assertIn(key, payload)
                self.assertTrue((result / payload[key]).is_file())


if __name__ == "__main__":
    unittest.main()
