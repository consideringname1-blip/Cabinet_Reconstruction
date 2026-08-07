import unittest

import numpy as np

from tools.itaco_ownership_v5.real_diagnostic import anchor_compatible, finite_metrics, _identity_state


CFG = {"minimum_predicted_overlap": .55, "minimum_observed_overlap": .3,
       "minimum_symmetric_overlap": .45, "maximum_normal_angle_degrees": 10.,
       "maximum_plane_offset_m": .02, "maximum_centroid_distance_m": .08,
       "reject_multiple_anchor_compatible_candidates": True}


def patch(z=1., x=0., n=15):
    u, v = np.meshgrid(np.linspace(-.05, .05, n), np.linspace(-.05, .05, n))
    return np.column_stack((u.ravel() + x, v.ravel(), np.full(u.size, z)))


class RealDiagnosticTests(unittest.TestCase):
    def test_same_finite_surface_matches_anchor(self):
        metrics = finite_metrics(patch(), patch(x=.002), .015)
        accepted, failures = anchor_compatible(metrics, CFG)
        self.assertTrue(accepted); self.assertEqual(failures, [])

    def test_parallel_neighbor_fails_anchor(self):
        metrics = finite_metrics(patch(), patch(z=1.05), .015)
        accepted, failures = anchor_compatible(metrics, CFG)
        self.assertFalse(accepted); self.assertIn("plane_offset", failures)

    def test_partial_finite_surface_allowed(self):
        source = patch(); observed = source[source[:, 0] > 0]
        metrics = finite_metrics(source, observed, .015)
        accepted, _ = anchor_compatible(metrics, CFG)
        self.assertTrue(accepted)

    def test_two_anchor_compatible_candidates_are_ambiguous(self):
        candidates = [{"anchor_compatible": True, "symmetric_overlap": .8},
                      {"anchor_compatible": True, "symmetric_overlap": .7}]
        state, selected, reason = _identity_state(candidates, CFG)
        self.assertEqual(state, "IDENTITY_AMBIGUOUS"); self.assertIsNone(selected)
        self.assertIn("no_substitution", reason)

    def test_no_candidate_is_identity_lost(self):
        state, selected, _ = _identity_state([], CFG)
        self.assertEqual(state, "IDENTITY_LOST"); self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
