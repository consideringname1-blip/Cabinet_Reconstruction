from __future__ import annotations

import unittest

import numpy as np

from stages.hololens_aruco_reference.run_aruco_sync_from_json import (
    _sync_sam3_spatial_box_to_aruco,
)


class Sam3SpatialBoxArUcoSyncTests(unittest.TestCase):
    def test_derives_canonical_corners_from_immutable_hololens_box(self) -> None:
        task = {
            "Sam3SpatialBox": {
                "status": "ready",
                "coordinate_space": "unity_world",
                "aabb_min_world": [1.0, 2.0, 3.0],
                "aabb_max_world": [3.0, 6.0, 5.0],
            }
        }
        reference = {
            "position": [1.0, 1.0, 1.0],
            "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }

        result = _sync_sam3_spatial_box_to_aruco(task, reference)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("aruco_local", result["canonical_coordinate_space"])
        self.assertEqual(8, len(result["corners_aruco"]))
        np.testing.assert_allclose(result["center_aruco"], [1.0, 3.0, 3.0])
        self.assertAlmostEqual(float(np.sqrt(24.0)), result["diagonal_m"])
        self.assertEqual([1.0, 2.0, 3.0], result["aabb_min_world"])

    def test_unavailable_preview_box_is_not_promoted(self) -> None:
        task = {"Sam3SpatialBox": {"status": "unavailable"}}
        reference = {
            "position": [0.0, 0.0, 0.0],
            "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        self.assertIsNone(_sync_sam3_spatial_box_to_aruco(task, reference))
        self.assertNotIn("corners_aruco", task["Sam3SpatialBox"])


if __name__ == "__main__":
    unittest.main()
