import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.itaco_ownership_v5.observation_censor import (
    ObservationState,
    censor_observation,
    discriminate_censored_motion_model,
    load_frozen_noise_model,
)
from tools.itaco_ownership_v5.plateau_noise import FrozenPlateauNoiseModel


DECISION = {
    "minimum_discriminative_to_plateau_p90_ratio": 1.,
    "minimum_informative_observations": 2,
    "minimum_posterior_probability": .99,
    "minimum_direction_consistency_fraction": .6,
}
BOUNDARY = {"tangent_motion": False, "has_trusted_axis_finite_edge": False}


def noise():
    values = np.linspace(.001, .010, 100)
    return FrozenPlateauNoiseModel(
        {"surface_interior": values, "depth_or_geometry_edge": values * 1.5},
        {"combined": {"p90_m": .009},
         "categories": {
             "surface_interior": {"sample_count": 100, "tail_scale_m": .002},
             "depth_or_geometry_edge": {"sample_count": 100, "tail_scale_m": .003}},
         "observation_mean_surprisal_p95": 3.,
         "frozen_model_sha256": "frozen"})


def raw(static_nll, moving_nll, baseline="medium", d=.03, raw_evidence=None):
    if raw_evidence is None:
        raw_evidence = static_nll - moving_nll
    return {
        "baseline": baseline, "abs_delta_q": d, "d_discriminative_m": d,
        "discriminative_to_plateau_p90_ratio": 3.,
        "accepted_for_discrimination": True,
        "common_source_sample_count": 100, "common_source_coverage": .8,
        "common_source_spatial_coverage": .75, "visibility": "common_source_anchor_observable",
        "static": {"median_residual_m": .01 + d, "p90_residual_m": .02 + d,
                   "mean_noise_surprisal": static_nll, "median_noise_surprisal": static_nll},
        "moving": {"median_residual_m": .005, "p90_residual_m": .008,
                   "mean_noise_surprisal": moving_nll, "median_noise_surprisal": moving_nll},
        "log_evidence_moving_over_static": raw_evidence,
    }


class ObservationCensorTests(unittest.TestCase):
    def test_01_static_compatible_moving_ood_is_world_static_evidence(self):
        row = censor_observation(raw(1., 8.), noise())
        self.assertEqual(row["observation_state"], ObservationState.WORLD_STATIC_EVIDENCE.value)
        self.assertTrue(row["eligible_for_aggregation"])

    def test_02_moving_compatible_static_ood_is_moving_evidence(self):
        row = censor_observation(raw(8., 1.), noise())
        self.assertEqual(row["observation_state"], ObservationState.MOVING_LINK_EVIDENCE.value)
        self.assertTrue(row["eligible_for_aggregation"])

    def test_03_both_compatible_is_neutral(self):
        row = censor_observation(raw(1., 2.), noise())
        self.assertEqual(row["observation_state"], ObservationState.AMBIGUOUS_COMPATIBLE.value)
        self.assertFalse(row["eligible_for_aggregation"]); self.assertFalse(row["censored"])

    def test_04_both_ood_is_censored_without_smaller_residual_selection(self):
        row = censor_observation(raw(20., 8., raw_evidence=12.), noise())
        self.assertEqual(row["observation_state"], ObservationState.IDENTITY_LOST_CENSORED.value)
        self.assertTrue(row["censored"]); self.assertIsNone(row["log_evidence_moving_over_static"])
        self.assertFalse(row["both_ood_smaller_residual_selected"])

    def test_05_censored_extreme_evidence_cannot_change_region_decision(self):
        retained = [
            censor_observation(raw(8., 1., "short", .01), noise()),
            censor_observation(raw(10., 1., "medium", .04), noise()),
        ]
        censored = censor_observation(raw(1000., 999., "long", .2, 10000.), noise())
        result = discriminate_censored_motion_model(retained + [censored], BOUNDARY, noise(), DECISION)
        self.assertEqual(result["label"], "MOVING_LINK")
        self.assertEqual(result["effective_chain_length"], 2)
        self.assertEqual(result["censored_observation_count"], 1)

    def test_06_all_both_ood_becomes_insufficient_not_smaller_model_winner(self):
        rows = [censor_observation(raw(9., 8., "long", .2), noise())]
        result = discriminate_censored_motion_model(rows, BOUNDARY, noise(), DECISION)
        self.assertEqual(result["label"], "UNKNOWN")
        self.assertEqual(result["reason"], "insufficient_retained_one_model_compatible_observations")

    def test_07_tangent_gate_is_preserved(self):
        rows = [
            censor_observation(raw(8., 1., "short", .01), noise()),
            censor_observation(raw(10., 1., "medium", .04), noise()),
        ]
        result = discriminate_censored_motion_model(
            rows, {"tangent_motion": True, "has_trusted_axis_finite_edge": False},
            noise(), DECISION)
        self.assertEqual(result["label"], "UNKNOWN")

    def test_08_plateau_never_creates_ownership(self):
        rows = [censor_observation(raw(8., 1.), noise())]
        result = discriminate_censored_motion_model(
            rows, BOUNDARY, noise(), DECISION, evaluation_mode="plateau_only")
        self.assertEqual(result["label"], "UNKNOWN")

    def test_09_insufficient_visibility_is_censored(self):
        row = raw(1., 8.); row["accepted_for_discrimination"] = False
        result = censor_observation(row, noise())
        self.assertEqual(result["observation_state"], ObservationState.IDENTITY_LOST_CENSORED.value)
        self.assertIn("visibility", result["censored_reason"])

    def test_10_frozen_noise_loader_verifies_original_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.linspace(.001, .01, 4)
            summary = {
                "categories": {
                    "surface_interior": {"sample_count": 4},
                    "depth_or_geometry_edge": {"sample_count": 4}},
                "observation_mean_surprisal_p95": 3.,
            }
            canonical = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(canonical).hexdigest()
            summary["frozen_model_sha256"] = digest
            (root / "plateau_noise_model.json").write_text(json.dumps(summary))
            np.savez(root / "plateau_noise_residuals.npz",
                     surface_interior=values, depth_or_geometry_edge=values)
            model, report = load_frozen_noise_model(root, digest)
            self.assertEqual(model.summary["frozen_model_sha256"], digest)
            self.assertTrue(report["loaded_without_refit"])

    def test_11_wrong_noise_hash_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plateau_noise_model.json").write_text(json.dumps({
                "frozen_model_sha256": "wrong", "categories": {},
                "observation_mean_surprisal_p95": 3.}))
            np.savez(root / "plateau_noise_residuals.npz",
                     surface_interior=[.1], depth_or_geometry_edge=[.1])
            with self.assertRaises(RuntimeError):
                load_frozen_noise_model(root, "expected")


if __name__ == "__main__":
    unittest.main()
