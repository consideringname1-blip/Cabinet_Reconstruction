import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.itaco_funrec_assignment_v3.depth_projective import candidate_metrics, decompose_track_observations
from tools.itaco_funrec_assignment_v3.track_only_report import write_track_only_report


K = np.asarray([[200.0, 0, 160.0], [0, 200.0, 144.0], [0, 0, 1.0]])
POSE = np.eye(4)
AXIS = np.asarray([1.0, 0, 0])
CFG = {"pixel_sigma_px": 3.0, "depth_sigma_m": .04, "maximum_depth_residual_m": .08,
       "maximum_association_cost": 3.0, "occlusion_margin_m": .03, "model_cost_margin": .25}


def frame(depth=1.0, valid=True):
    return {"depth": np.full((288, 320), depth, np.float32), "valid": np.full((288, 320), valid, bool),
            "pose": POSE, "edge_distance": np.full((288, 320), 10.0), "source": 1, "proposals": []}


class ProjectiveTests(unittest.TestCase):
    def test_static_candidate(self):
        result = candidate_metrics(np.asarray([0., 0, 1]), 0., np.asarray([160., 144.]), frame(), .1, AXIS, K, CFG)
        self.assertTrue(result["association_accepted"]); self.assertEqual(result["association_model"], "static")

    def test_drawer_candidate(self):
        result = candidate_metrics(np.asarray([0., 0, 1]), 0., np.asarray([180., 144.]), frame(), .1, AXIS, K, CFG)
        self.assertTrue(result["association_accepted"]); self.assertEqual(result["association_model"], "drawer")

    def test_invalid_target_rejected(self):
        self.assertIsNone(candidate_metrics(np.asarray([0., 0, 1]), 0., np.asarray([160., 144.]), frame(valid=False), .1, AXIS, K, CFG))

    def test_occluded_under_both_models(self):
        result = candidate_metrics(np.asarray([0., 0, 2]), 0., np.asarray([160., 144.]), frame(depth=1.), .1, AXIS, K, CFG)
        self.assertFalse(result["association_accepted"]); self.assertTrue(result["static_occluded"]); self.assertTrue(result["drawer_occluded"])

    def test_axis_perpendicular_decomposition(self):
        observations = [{"track_id": 1, "original_frame_id": i, "pixel_uv": [0, 0], "depth_m": 1., "point_world": p,
                         "q_t": q, "tracking_confidence": 1., "forward_backward_error": 0.}
                        for i, (p, q) in enumerate([(np.asarray([0., 0., 1.]), 0.), (np.asarray([.1, .02, 1.]), .1)])]
        rows = decompose_track_observations([{"track_id": 1, "seed_frame_id": 0, "observations": observations}], AXIS)
        self.assertAlmostEqual(abs(rows[0]["drawer_track_axis_residual_m"]), 0., places=7)
        self.assertAlmostEqual(rows[0]["drawer_track_perpendicular_residual_m"], .01, places=7)

    def test_gate_never_runs_propagation(self):
        evidence = [{"track_id": i, "label": "moving", "reason": "moving_track"} for i in range(13)]
        cfg = {"region_propagation_gate": {"baseline_reliable_moving_tracks": 8, "minimum_absolute_increase": 5,
                                            "minimum_increase_factor": 1.5}}
        diagnostics = {"accepted_attempts": 10, "rejected_attempts": 2, "high_residual_track_count": 0}
        with tempfile.TemporaryDirectory() as directory:
            report = write_track_only_report(Path(directory), evidence, diagnostics, cfg)
            self.assertTrue(report["gate"]["reliable_moving_tracks_clearly_increased"])
            self.assertFalse(report["gate"]["region_propagation_ran"])


if __name__ == "__main__":
    unittest.main()
