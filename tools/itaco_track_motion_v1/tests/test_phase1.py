from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.itaco_track_motion_v1.classification import build_clusters, classify_tracks, proposal_split_merge_diagnostics
from tools.itaco_track_motion_v1.geometry import KnownMotion, rebase_poses
from tools.itaco_track_motion_v1.reference import select_reference_frame
from tools.itaco_track_motion_v1.tracking import _observation

from tools.itaco_track_motion_v1.validity import compose_coarse_static_mask

def observation(track_id: int, index: int, point: np.ndarray, proposal: str) -> dict:
    return {
        "track_id": track_id, "processing_index": index, "original_frame_id": 1000 + index,
        "timestamp": 100000 + index, "point_world": point.tolist(), "proposal_id": proposal,
        "depth_confidence": 1.0, "depth_uncertainty_m": 0.001,
    }


def track(track_id: int, points: list[np.ndarray], proposal: str) -> dict:
    return {
        "track_id": track_id, "observations": [observation(track_id, index, point, proposal) for index, point in enumerate(points)],
        "attempted_transitions": len(points) - 1, "occlusion_failures": 0, "boundary_failures": 0,
    }


class Phase1Tests(unittest.TestCase):
    def test_nonzero_reference_rebase(self) -> None:
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 0, 3] = [2.0, 3.0, 4.0]
        rebased = rebase_poses(poses, 1)
        np.testing.assert_allclose(rebased[1], np.eye(4), atol=1e-12)
        self.assertFalse(np.allclose(rebased[0], np.eye(4)))

    def test_reference_selection_rejects_hand_frame_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            rng = np.random.default_rng(3)
            texture = rng.integers(20, 235, size=(64, 64, 3), dtype=np.uint8)
            for index in range(3):
                rgb_path, depth_path, hand_path = root / f"rgb{index}.png", root / f"depth{index}.png", root / f"hand{index}.npy"
                cv2.imwrite(str(rgb_path), texture); cv2.imwrite(str(depth_path), np.full((64, 64), 1000, np.uint16))
                hand = np.zeros((64, 64), bool)
                if index == 0: hand[:, :32] = True
                np.save(hand_path, hand)
                records.append({"original_frame_id": 20 + index, "processing_index": index, "timestamp": index * 1000000,
                                "rgb_path": str(rgb_path), "depth_path": str(depth_path), "hand_mask_path": str(hand_path)})
            poses = np.repeat(np.eye(4)[None], 3, axis=0)
            config = {
                "validity": {"depth_scale_to_m": 0.001, "depth_min_m": 0.2, "depth_max_m": 4.0,
                             "rgb_min_intensity": 2, "rgb_close_radius_px": 0, "rgb_keep_largest_component": True,
                             "depth_boundary_erosion_px": 1.0},
                "reference_selection": {"timestamp_scale_seconds": 1e-7, "max_hand_fraction": 0.2,
                    "min_valid_depth_fraction": 0.5, "min_static_valid_fraction": 0.5, "max_camera_speed_mps": 1.0,
                    "max_camera_angular_speed_radps": 1.0, "min_laplacian_variance": 1.0, "min_tracking_quality": 0.2,
                    "require_pre_action": True, "pre_action_baseline_fraction": 0.5, "pre_action_max_state_delta": 0.01,
                    "pre_action_max_range_fraction": 0.05,
                    "weights": {"hand": 1.0, "depth": 1.0, "static": 1.0, "linear_speed": 1.0, "angular_speed": 1.0,
                                "blur": 1.0, "tracking": 1.0, "before_action": 1.0},
                    "tracking_probe": {"max_corners": 100, "quality_level": 0.01, "min_distance_px": 3.0, "fb_max_error_px": 1.0}},
            }
            selected, candidates, _ = select_reference_frame(records, poses, config, np.asarray([0.0, 0.0, 0.2]))
            self.assertEqual(selected, 1)
            self.assertIn("hand_coverage_too_high", candidates[0]["rejection_reasons"])

    def test_fixed_model_classification_split_merge_and_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); axis_path, state_path = root / "axis.npy", root / "q.npy"
            q = np.linspace(0.0, 0.2, 8); np.save(axis_path, np.asarray([1.0, 0.0, 0.0])); np.save(state_path, q)
            motion = KnownMotion({"type": "prismatic", "axis_path": str(axis_path), "state_path": str(state_path), "canonicalization_sign": 1.0}, np.repeat(np.eye(4)[None], 8, axis=0), 0, 8)
            base_static = np.asarray([0.0, 0.0, 1.0]); base_moving = np.asarray([0.2, 0.0, 1.0])
            tracks = [
                track(0, [base_static + np.asarray([0.0, 0.0002 * i, 0.0]) for i in range(8)], "proposal_a"),
                track(1, [base_moving - np.asarray([value, 0.0, 0.0]) for value in q], "proposal_a"),
                track(2, [base_moving + np.asarray([0.0, 0.05, 0.0]) - np.asarray([value, 0.0, 0.0]) for value in q], "proposal_b"),
                track(3, [base_static, base_static + np.asarray([0.001, 0.0, 0.0])], "proposal_c"),
            ]
            cfg = {"residual_sigma_floor_m": 0.001, "bic_temperature": 5.0, "min_valid_observations": 4,
                   "min_coverage": 0.5, "min_abs_delta_bic": 2.0, "max_absolute_rmse_m": 0.03,
                   "min_median_depth_confidence": 0.5, "max_median_depth_uncertainty_m": 0.02,
                   "max_known_moving_rmse_m": 0.03, "min_moving_proposal_support": 0.5,
                   "max_occlusion_rate": 0.5, "max_boundary_failure_rate": 0.5, "min_motion_excitation": 0.03,
                   "unknown_probability_floor_on_failure": 0.8, "unknown_probability_without_failure": 0.02,
                   "unknown_reason_temperature": 0.5, "label_posterior_threshold": 0.55}
            residuals, labels, representative = classify_tracks(tracks, motion, cfg)
            by_id = {item["track_id"]: item["label"] for item in labels}
            self.assertEqual(by_id[0], "static"); self.assertEqual(by_id[1], "moving"); self.assertEqual(by_id[2], "moving"); self.assertEqual(by_id[3], "unknown")
            clusters = build_clusters(tracks, labels, representative, {"voxel_size_m": 1.0, "max_tracks_per_cluster": 20})
            diagnostics = proposal_split_merge_diagnostics(labels, clusters)
            self.assertIn("proposal_a", diagnostics["mixed_proposals_split"])
            self.assertTrue(diagnostics["moving_set_merges_multiple_proposals"])
            self.assertEqual(len(residuals), 4)

    def test_hand_pixel_cannot_form_observation(self) -> None:
        payload = {"depth": np.ones((8, 8), np.float32), "valid_for_tracking": np.ones((8, 8), bool),
                   "hand": np.zeros((8, 8), bool), "occlusion": np.zeros((8, 8), bool), "fb_consistent": np.ones((8, 8), bool),
                   "boundary_distance": np.full((8, 8), 5.0), "depth_confidence": np.ones((8, 8), np.float32),
                   "rgb": np.zeros((8, 8, 3), np.uint8)}
        payload["hand"][4, 4] = True
        result = _observation(0, {"processing_index": 0, "original_frame_id": 0, "timestamp": 0}, payload, [], np.asarray([4.0, 4.0]),
                              np.eye(4), np.asarray([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]), 1.0,
                              {"depth_uncertainty_patch_radius_px": 1, "depth_uncertainty_scale_m": 0.02})
        self.assertIsNone(result)

    def test_coarse_static_mask_keeps_object_and_hand_exclusion(self) -> None:
        dynamic = np.asarray([[False, True], [False, False]])
        object_mask = np.asarray([[True, True], [False, True]])
        valid_camera = np.asarray([[True, True], [True, False]])
        result = compose_coarse_static_mask(dynamic, object_mask, valid_camera)
        np.testing.assert_array_equal(result, np.asarray([[True, False], [False, False]]))

    def test_static_track_is_independent_of_proposal_identity(self) -> None:
        self.assertNotIn("proposal_id", __import__("inspect").signature(classify_tracks).parameters)


if __name__ == "__main__":
    unittest.main()
