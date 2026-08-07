import inspect
import unittest

import numpy as np

from tools.itaco_ownership_v5.local_transitions import decompose_local_motion_states
from tools.itaco_ownership_v5.motion_discrimination import discriminate_motion_model
from tools.itaco_ownership_v5.multi_baseline import build_multi_baseline_observations
from tools.itaco_ownership_v5.plateau_noise import FrozenPlateauNoiseModel


MOTION = {"active_velocity_threshold": .001, "active_dilation_frames": 0,
          "velocity_edge_order": 1, "plateau_q_tolerance": .001,
          "closed_is_minimum_q": True}
BASELINES = {
    "short": {"minimum_delta_q_m": .005, "maximum_delta_q_m": .015},
    "medium": {"minimum_delta_q_m": .020, "maximum_delta_q_m": .050},
    "long": {"minimum_delta_q_m": .050000001, "maximum_delta_q_m": .35},
    "require_all_interval_frames_active": True,
    "allow_backward_motion_discrimination": True,
}
DECISION = {"minimum_discriminative_to_plateau_p90_ratio": 1.,
            "minimum_informative_observations": 2,
            "minimum_posterior_probability": .99,
            "minimum_direction_consistency_fraction": .6}


def noise():
    values = np.linspace(.001, .010, 1000)
    return FrozenPlateauNoiseModel(
        {"surface_interior": values, "depth_or_geometry_edge": values * 1.5},
        {"combined": {"p90_m": .009}, "categories": {"surface_interior": {"tail_scale_m": .002}, "depth_or_geometry_edge": {"tail_scale_m": .003}}, "observation_mean_surprisal_p95": 3.,
         "frozen_model_sha256": "frozen"})


def observations(moving=True):
    rows = []
    for baseline, d, index in (("short", .012, 1), ("medium", .035, 2), ("long", .070, 3)):
        if moving:
            static_r, moving_r, evidence = .010 + .20 * d, .006, 5.
            static_nll, moving_nll = 5., .8
        else:
            static_r, moving_r, evidence = .006, .010 + .20 * d, -5.
            static_nll, moving_nll = .8, 5.
        rows.append({
            "accepted_for_discrimination": True, "log_evidence_moving_over_static": evidence,
            "discriminative_to_plateau_p90_ratio": 2. + index, "d_discriminative_m": d,
            "baseline": baseline,
            "static": {"median_residual_m": static_r, "mean_noise_surprisal": static_nll},
            "moving": {"median_residual_m": moving_r, "mean_noise_surprisal": moving_nll},
        })
    return rows


class MultiBaselineTests(unittest.TestCase):
    def test_01_builder_has_no_target_proposal_or_v4_per_target(self):
        source = inspect.getsource(build_multi_baseline_observations)
        self.assertNotIn("proposal", source); self.assertNotIn("per_target", source)

    def test_02_forward_and_backward_baselines_cover_active_motion(self):
        states = decompose_local_motion_states(
            list(range(7)), [i * .2 for i in range(7)],
            [0., .006, .014, .028, .050, .080, .081], MOTION)
        accepted, _ = build_multi_baseline_observations(states, BASELINES)
        directions = {row.direction for row in accepted}
        bins = {row.baseline for row in accepted}
        self.assertEqual(directions, {"forward", "backward_identity_confirmation"})
        self.assertTrue({"short", "medium", "long"}.issubset(bins))

    def test_03_moving_model_is_selected_by_frozen_noise_evidence(self):
        result = discriminate_motion_model(
            observations(True), {"tangent_motion": False, "has_trusted_axis_finite_edge": False},
            noise(), DECISION)
        self.assertEqual(result["label"], "MOVING_LINK")

    def test_04_world_static_route_is_symmetric(self):
        result = discriminate_motion_model(
            observations(False), {"tangent_motion": False, "has_trusted_axis_finite_edge": False},
            noise(), DECISION)
        self.assertEqual(result["label"], "WORLD_STATIC")

    def test_05_tangent_repeated_plane_stays_unknown_without_finite_edge(self):
        result = discriminate_motion_model(
            observations(True), {"tangent_motion": True, "has_trusted_axis_finite_edge": False},
            noise(), DECISION)
        self.assertEqual(result["label"], "UNKNOWN")

    def test_06_plateau_cannot_create_ownership(self):
        result = discriminate_motion_model(
            observations(True), {"tangent_motion": False, "has_trusted_axis_finite_edge": True},
            noise(), DECISION, evaluation_mode="plateau_only")
        self.assertEqual(result["label"], "UNKNOWN")

    def test_07_empirical_noise_surprisal_penalizes_large_residual(self):
        model = noise(); labels = np.asarray(["surface_interior"] * 2, object)
        score = model.surprisal(np.asarray([.003, .030]), labels)
        self.assertGreater(score[1], score[0])


if __name__ == "__main__":
    unittest.main()
