from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from stages.sam3d_body_mesh.run_sam3d_body_mesh_from_json import _contact_bbox_policy  # noqa: E402


class Sam3dBodyContactPolicyTests(unittest.TestCase):
    def test_remote_contact_requires_valid_people_bbox(self) -> None:
        bbox, policy = _contact_bbox_policy(
            {
                "source": "remote_shigure_contacted",
                "people_bounding_box": {"x": 1, "y": 2, "width": 30, "height": 40},
            }
        )
        self.assertEqual([1.0, 2.0, 31.0, 42.0], bbox)
        self.assertEqual("remote_contact_bbox", policy)

    def test_remote_contact_never_falls_back_to_local_person_detector(self) -> None:
        bbox, policy = _contact_bbox_policy(
            {
                "source": "remote_shigure_contacted",
                "people_bounding_box": {"x": 1, "y": 2, "width": 0, "height": 40},
            }
        )
        self.assertIsNone(bbox)
        self.assertEqual("remote_contact_bbox_missing", policy)

        bbox, policy = _contact_bbox_policy(
            {
                "source": "remote_shigure_contacted",
                "people_bounding_box": {"xyxy": [0.0, 0.0, float("inf"), 10.0]},
            }
        )
        self.assertIsNone(bbox)
        self.assertEqual("remote_contact_bbox_missing", policy)

    def test_legacy_taken_path_keeps_detector_fallback(self) -> None:
        bbox, policy = _contact_bbox_policy({"source": "taken_object_detection_shigure_object_mask"})
        self.assertIsNone(bbox)
        self.assertEqual("legacy_detector_fallback", policy)


if __name__ == "__main__":
    unittest.main()
