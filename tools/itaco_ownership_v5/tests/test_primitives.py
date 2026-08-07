import unittest

import numpy as np

from tools.itaco_ownership_v5.evidence import Decision, EvidenceBook, EvidenceFamily
from tools.itaco_ownership_v5.identity import verify_anchored_surface_sequence
from tools.itaco_ownership_v5.motion import MotionState, build_active_transitions, decompose_motion_states
from tools.itaco_ownership_v5.occlusion import detect_causal_static_disocclusion
from tools.itaco_ownership_v5.transforms import ArticulatedTransform


MOTION = {"active_velocity_threshold": .05, "active_dilation_frames": 0,
          "plateau_q_tolerance": .005, "closed_is_minimum_q": True}
TRANSITION = {"minimum_transition_delta_q": .02, "maximum_transition_frame_gap": 4}
IDENTITY = {"point_distance_threshold": .01, "minimum_symmetric_overlap": .6}
OCCLUSION = {"maximum_local_depth_gap_m": .08, "maximum_boundary_distance_px": 12,
             "maximum_reveal_delay_frames": 2, "minimum_persistent_frames": 3,
             "maximum_world_residual_m": .01}


def patch(origin=(0., 0., 1.), half=.05, n=12):
    x, y = np.meshgrid(np.linspace(-half, half, n), np.linspace(-half, half, n))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    return points + np.asarray(origin)


class AssignmentV5PrimitiveTests(unittest.TestCase):
    def test_01_static_plane_near_active_drawer_remains_static(self):
        source = patch()
        observations = [{"frame_id": 2, "q": .05, "points": source.copy(), "active_transition": True}]
        static = verify_anchored_surface_sequence(source, 0., observations, lambda p, qi, qj: p, IDENTITY)
        moving_transform = ArticulatedTransform("prismatic", [1, 0, 0], [0, 0, 0])
        moving = verify_anchored_surface_sequence(source, 0., observations, moving_transform.apply_between, IDENTITY)
        self.assertEqual(static["verified_active_observations"], 1)
        self.assertEqual(moving["verified_active_observations"], 0)

    def test_02_moving_finite_drawer_surface_follows_transform(self):
        transform = ArticulatedTransform("prismatic", [1, 0, 0], [0, 0, 0])
        source = patch()
        observed = transform.apply_between(source, 0., .06)
        result = verify_anchored_surface_sequence(source, 0.,
            [{"frame_id": 2, "q": .06, "points": observed, "active_transition": True}],
            transform.apply_between, IDENTITY)
        self.assertEqual(result["verified_active_observations"], 1)

    def test_03_infinite_plane_parallel_motion_ambiguity_is_unknown(self):
        book = EvidenceBook()
        # Both occupancy predictions being compatible adds no positive family.
        book.add(EvidenceFamily.ARTICULATED_MOTION_TRANSITION, "neither", "occupancy_supported_both", active_transition=True)
        self.assertEqual(book.decide()["decision"], Decision.UNKNOWN.value)

    def test_04_parallel_neighbor_plane_does_not_replace_anchor(self):
        source = patch(origin=(0, 0, 1))
        neighbor = patch(origin=(0, 0, 1.04))
        result = verify_anchored_surface_sequence(source, 0.,
            [{"frame_id": 2, "q": .1, "points": neighbor, "active_transition": True}],
            lambda p, qi, qj: p, IDENTITY)
        self.assertEqual(result["rows"][0]["identity_state"], "IDENTITY_LOST")

    def test_05_gradual_a_ab_b_cannot_surface_hop(self):
        source = patch(half=.04)
        shifted = patch(origin=(.06, 0, 1), half=.04)
        mixed = np.vstack((source[::2], shifted[1::2]))
        rows = [{"frame_id": 1, "q": .03, "points": mixed, "active_transition": True},
                {"frame_id": 2, "q": .06, "points": shifted, "active_transition": True}]
        result = verify_anchored_surface_sequence(source, 0., rows, lambda p, qi, qj: p, IDENTITY)
        self.assertEqual(result["rows"][-1]["identity_state"], "IDENTITY_LOST")
        self.assertTrue(result["target_observations_never_become_anchors"])

    def test_06_trusted_drawer_moves_away_reveals_persistent_static(self):
        event = {"active_transition": True, "verified_moving_occluder": True,
                 "occluder_motion_matches_articulation": True, "depth_gap_m": .03,
                 "boundary_distance_px": 4, "reveal_delay_frames": 1,
                 "silhouette_leaves_location": True, "new_surface_appears": True,
                 "persistent_frames": 4, "world_residual_m": .003}
        self.assertTrue(detect_causal_static_disocclusion(event, OCCLUSION)["positive_static_evidence"])

    def test_07_large_same_ray_depth_gap_is_not_local_disocclusion(self):
        event = {"active_transition": True, "verified_moving_occluder": True,
                 "occluder_motion_matches_articulation": True, "depth_gap_m": .73,
                 "boundary_distance_px": 4, "reveal_delay_frames": 1,
                 "silhouette_leaves_location": True, "new_surface_appears": True,
                 "persistent_frames": 4, "world_residual_m": .003}
        result = detect_causal_static_disocclusion(event, OCCLUSION)
        self.assertFalse(result["positive_static_evidence"])
        self.assertIn("local_depth_gap", result["failed_checks"])

    def test_08_plateau_only_cannot_create_ownership(self):
        states = decompose_motion_states([0, 1, 2], [0., 1., 2.], [0., 0., 0.], MOTION)
        self.assertTrue(all(row.state == MotionState.CLOSED_PLATEAU for row in states))
        self.assertEqual(build_active_transitions(states, TRANSITION), [])
        book = EvidenceBook()
        book.add(EvidenceFamily.ARTICULATED_MOTION_TRANSITION, "moving", "plateau_similarity", active_transition=False)
        self.assertEqual(book.decide()["decision"], Decision.UNKNOWN.value)

    def test_09_correlated_metrics_count_as_one_family(self):
        book = EvidenceBook()
        book.add(EvidenceFamily.ARTICULATED_MOTION_TRANSITION, "moving", "continuity", {"continuity": .9}, True)
        book.add(EvidenceFamily.ARTICULATED_MOTION_TRANSITION, "moving", "attachment_same_chain", {"attachment": .8}, True)
        result = book.decide()
        self.assertEqual(result["moving_families"], [EvidenceFamily.ARTICULATED_MOTION_TRANSITION.value])
        self.assertEqual(len(result["records"]), 1)

    def test_10_conflicting_independent_cues_remain_conflicting(self):
        book = EvidenceBook()
        book.add(EvidenceFamily.ARTICULATED_MOTION_TRANSITION, "moving", "finite_motion", active_transition=True)
        book.add(EvidenceFamily.CAUSAL_OCCLUSION_DISOCCLUSION, "static", "causal_reveal", active_transition=True)
        self.assertEqual(book.decide()["decision"], Decision.CONFLICTING.value)

    def test_11_world_static_does_not_imply_cabinet_static(self):
        book = EvidenceBook()
        book.add(EvidenceFamily.CAUSAL_OCCLUSION_DISOCCLUSION, "static", "world_static_only", active_transition=True)
        result = book.decide()
        self.assertEqual(result["decision"], Decision.STATIC.value)
        self.assertNotIn("cabinet", result)

    def test_12_revolute_transform_obeys_same_identity_logic(self):
        transform = ArticulatedTransform("revolute", [0, 0, 1], [0, 0, 1])
        source = patch(origin=(.2, 0, 1), half=.03)
        observed = transform.apply_between(source, 0., np.pi / 6)
        result = verify_anchored_surface_sequence(source, 0.,
            [{"frame_id": 2, "q": np.pi / 6, "points": observed, "active_transition": True}],
            transform.apply_between, IDENTITY)
        self.assertEqual(result["verified_active_observations"], 1)

    def test_motion_state_decomposition_and_active_transition(self):
        states = decompose_motion_states(range(6), np.arange(6.), [0, 0, .08, .16, .2, .2], MOTION)
        self.assertIn(MotionState.ACTIVE_MOTION, [row.state for row in states])
        transitions = build_active_transitions(states, TRANSITION)
        self.assertTrue(transitions)
        self.assertTrue(all(row.traversed_active_motion and abs(row.delta_q) >= .02 for row in transitions))

    def test_causal_family_without_active_transition_cannot_originate_static(self):
        book = EvidenceBook()
        book.add(EvidenceFamily.CAUSAL_OCCLUSION_DISOCCLUSION, "static", "plateau_reveal", active_transition=False)
        self.assertEqual(book.decide()["decision"], Decision.UNKNOWN.value)

    def test_trusted_propagation_cannot_originate_identity(self):
        book = EvidenceBook()
        book.add(EvidenceFamily.TRUSTED_IDENTITY_PROPAGATION, "moving", "sam2_seed_overlap")
        self.assertEqual(book.decide()["decision"], Decision.UNKNOWN.value)


if __name__ == "__main__":
    unittest.main()
