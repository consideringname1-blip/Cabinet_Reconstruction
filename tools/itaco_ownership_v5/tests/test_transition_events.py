import inspect
import unittest

import cv2
import numpy as np

from tools.itaco_ownership_v5.causal_reveal import (
    compare_predicted_actual_silhouette,
    measure_causal_reveal,
    transform_trusted_drawer_silhouette,
)
from tools.itaco_ownership_v5.local_transitions import (
    LocalMotionState,
    build_local_active_transitions,
    decompose_local_motion_states,
    local_targets_for_source,
)
from tools.itaco_ownership_v5.physical_boundary import analyze_physical_boundary
from tools.itaco_ownership_v5.projected_source_anchor import (
    AnchorState,
    evaluate_source_indexed_patch,
    extract_source_patch,
)
from tools.itaco_ownership_v5.transition_chain import (
    decide_motion_ownership,
    moving_edge_chain_confidence,
    summarize_transition_chain,
)
from tools.itaco_ownership_v5.transforms import ArticulatedTransform
from tools.itaco_region_assignment_v4.frame_data import evaluate_scale_consistency_gate


K = np.asarray([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
MOTION = {"active_velocity_threshold": .002, "active_dilation_frames": 0,
          "velocity_edge_order": 1, "plateau_q_tolerance": .002,
          "closed_is_minimum_q": True}
LOCAL = {"minimum_delta_q_m": .005, "maximum_delta_q_m": .060,
         "maximum_frame_gap": 3, "maximum_elapsed_s": .8,
         "require_all_intermediate_frames_active": True}
ANCHOR = {"depth_support_threshold_m": .03, "residual_inlier_threshold_m": .035,
          "occlusion_margin_m": .03, "free_space_margin_m": .03,
          "minimum_valid_projected_samples": 20, "minimum_source_index_coverage": .2,
          "minimum_spatial_coverage": .2, "minimum_inlier_fraction": .45,
          "maximum_median_residual_m": .03, "maximum_p90_residual_m": .06,
          "minimum_contradiction_fraction": .15,
          "minimum_contradiction_spatial_coherence": .2,
          "minimum_occlusion_fraction": .25,
          "minimum_trusted_drawer_occlusion_fraction": .5,
          "spatial_dilation_pixels": 2, "coverage_grid_size": 4}
CHAIN = {"minimum_consecutive_verified_frames": 2, "minimum_verified_q_span_m": .008,
         "minimum_verified_elapsed_s": .2, "maximum_chain_median_residual_m": .03,
         "maximum_chain_p90_residual_m": .06}
BOUNDARY = {"observation_boundary_margin_pixels": 3., "cross_boundary_sample_offset_pixels": 2,
            "minimum_depth_jump_m": .045, "minimum_normal_jump_degrees": 18.,
            "minimum_plane_offset_m": .025, "maximum_normal_axis_alignment_for_tangent": .3,
            "minimum_axis_edge_physical_confidence": .3, "minimum_source_geometry_points": 20,
            "minimum_moving_edge_support_fraction": .4}
CAUSAL = {"minimum_silhouette_iou": .3, "maximum_silhouette_boundary_displacement_px": 8.,
          "maximum_local_depth_gap_m": .08, "maximum_boundary_distance_px": 12.,
          "maximum_reveal_frame_gap": 3, "maximum_reveal_elapsed_s": .8,
          "minimum_reveal_samples": 4, "minimum_reveal_fraction": .1,
          "minimum_trusted_occluder_pixels": 10, "minimum_persistent_frames": 2,
          "minimum_persistent_sample_fraction": .2, "maximum_world_residual_m": .03,
          "silhouette": {"maximum_silhouette_samples": 6000, "silhouette_dilation_pixels": 2}}


def frame(frame_id, q, depth=None, footprint=None):
    if depth is None: depth = np.full((100, 100), 2., np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if footprint is None: footprint = np.ones_like(valid)
    return {"source": frame_id, "q": q, "depth": depth, "depth_valid": valid,
            "rgb_footprint": footprint, "valid": valid & footprint,
            "pose": np.eye(4), "rgb": np.zeros((100, 100, 3), np.uint8)}


def rectangle_mask(x0=40, x1=60, y0=40, y1=60):
    mask = np.zeros((100, 100), bool); mask[y0:y1, x0:x1] = True
    return mask


def source_and_targets(moving=True, tangent=False):
    mask = rectangle_mask()
    source_depth = np.full((100, 100), 2., np.float32); source_depth[mask] = 1.
    source = frame(0, 0., source_depth)
    axis = np.asarray([1., 0., 0.]) if tangent else np.asarray([0., 0., -1.])
    targets = []
    for frame_id, q in ((1, .02), (2, .04)):
        depth = np.full((100, 100), 2., np.float32)
        if moving:
            if tangent:
                shift = int(round(100 * q)); moved = np.roll(mask, shift, axis=1)
                if shift: moved[:, :shift] = False
                depth[moved] = 1.
            else:
                depth[mask] = 1. - q
        else:
            depth[mask] = 1.
        targets.append(frame(frame_id, q, depth))
    patch = extract_source_patch(source, mask, K, {"maximum_source_samples": 800})
    return source, targets, mask, patch, axis


class TransitionEventRepairTests(unittest.TestCase):
    def test_01_local_transitions_do_not_read_v4_per_target(self):
        self.assertNotIn("per_target", inspect.getsource(build_local_active_transitions))

    def test_02_directed_transition_reverse_is_not_equivalent(self):
        states = decompose_local_motion_states([1, 2, 3], [0., .2, .4], [0., .02, .04], MOTION)
        accepted, _ = build_local_active_transitions(states, LOCAL)
        keys = {row.directed_key for row in accepted}
        self.assertIn((1, 2), keys); self.assertNotIn((2, 1), keys)

    def test_03_nonactive_interval_is_rejected(self):
        states = decompose_local_motion_states([1, 2, 3], [0., .2, .4], [0., 0., .04], MOTION)
        accepted, rejected = build_local_active_transitions(states, LOCAL)
        self.assertNotIn((1, 3), {row.directed_key for row in accepted})
        self.assertTrue(any(row.directed_key == (1, 3) and
                            "interval_contains_non_active_motion" in row.validity_reason for row in rejected))

    def test_04_projective_patch_engine_has_no_target_proposal_input(self):
        signature = inspect.signature(evaluate_source_indexed_patch)
        self.assertNotIn("proposals", signature.parameters)
        self.assertNotIn("for proposal", inspect.getsource(evaluate_source_indexed_patch))

    def test_05_nearby_parallel_proposal_cannot_replace_source(self):
        source, _, mask, patch, axis = source_and_targets(False)
        target = frame(1, .02, np.full((100, 100), 1.10, np.float32))
        result = evaluate_source_indexed_patch(patch, target, axis, K, "static", ANCHOR)
        self.assertNotEqual(result["identity_state"], AnchorState.VERIFIED_SOURCE_ANCHOR.value)

    def test_06_moving_finite_front_active_chain_is_moving_link(self):
        source, targets, mask, patch, axis = source_and_targets(True, False)
        rows = [evaluate_source_indexed_patch(patch, target, axis, K, "drawer", ANCHOR)
                for target in targets]
        static = [evaluate_source_indexed_patch(patch, target, axis, K, "static", ANCHOR)
                  for target in targets]
        times = {0: 0., 1: .2, 2: .4}
        moving_chain = summarize_transition_chain(rows, 0, 0., times, CHAIN)
        static_chain = summarize_transition_chain(static, 0, 0., times, CHAIN)
        boundary = {"tangent_motion": False, "has_trusted_axis_finite_edge": False}
        decision = decide_motion_ownership(static_chain, moving_chain, boundary, {"passed": False})
        self.assertEqual(decision["label"], "MOVING_LINK")

    def test_07_world_static_finite_surface_has_symmetric_route(self):
        source, targets, mask, patch, axis = source_and_targets(False, False)
        static = [evaluate_source_indexed_patch(patch, target, axis, K, "static", ANCHOR) for target in targets]
        moving = [evaluate_source_indexed_patch(patch, target, axis, K, "drawer", ANCHOR) for target in targets]
        times = {0: 0., 1: .2, 2: .4}
        decision = decide_motion_ownership(
            summarize_transition_chain(static, 0, 0., times, CHAIN),
            summarize_transition_chain(moving, 0, 0., times, CHAIN),
            {"tangent_motion": False, "has_trusted_axis_finite_edge": False}, {"passed": False})
        self.assertEqual(decision["label"], "WORLD_STATIC")

    def test_08_infinite_tangent_plane_both_compatible_is_unknown(self):
        source, targets, mask, patch, axis = source_and_targets(False, True)
        for target in targets: target["depth"][:] = 1.
        static = [evaluate_source_indexed_patch(patch, target, axis, K, "static", ANCHOR) for target in targets]
        moving = [evaluate_source_indexed_patch(patch, target, axis, K, "drawer", ANCHOR) for target in targets]
        times = {0: 0., 1: .2, 2: .4}
        decision = decide_motion_ownership(
            summarize_transition_chain(static, 0, 0., times, CHAIN),
            summarize_transition_chain(moving, 0, 0., times, CHAIN),
            {"tangent_motion": True, "has_trusted_axis_finite_edge": False}, {"passed": False})
        self.assertEqual(decision["label"], "UNKNOWN")
        self.assertEqual(decision["reason"], "motion_tangent_or_repeated_surface_ambiguity")

    def test_09_tangent_finite_panel_true_edge_allows_moving(self):
        source, targets, mask, patch, axis = source_and_targets(True, True)
        boundary = analyze_physical_boundary(source, mask, patch["source_world"], patch["source_uv"], axis, K, BOUNDARY)
        moving = [evaluate_source_indexed_patch(patch, target, axis, K, "drawer", ANCHOR) for target in targets]
        static = [evaluate_source_indexed_patch(patch, target, axis, K, "static", ANCHOR) for target in targets]
        edge = moving_edge_chain_confidence(patch, moving, axis, BOUNDARY)
        times = {0: 0., 1: .2, 2: .4}
        decision = decide_motion_ownership(
            summarize_transition_chain(static, 0, 0., times, CHAIN),
            summarize_transition_chain(moving, 0, 0., times, CHAIN), boundary, edge)
        self.assertEqual(decision["label"], "MOVING_LINK")

    def test_10_observation_clipped_boundary_is_not_physical(self):
        depth = np.ones((100, 100), np.float32); footprint = np.ones((100, 100), bool); footprint[:, :12] = False
        source = frame(0, 0., depth, footprint); mask = rectangle_mask(10, 30)
        patch = extract_source_patch(source, mask, K, {"maximum_source_samples": 800})
        report = analyze_physical_boundary(source, mask, patch["source_world"], patch["source_uv"], [1,0,0], K, BOUNDARY)
        self.assertGreater(report["observation_clipped_fraction"], 0.)

    def test_11_segmentation_only_boundary_is_not_physical(self):
        source = frame(0, 0., np.ones((100, 100), np.float32)); mask = rectangle_mask()
        patch = extract_source_patch(source, mask, K, {"maximum_source_samples": 800})
        report = analyze_physical_boundary(source, mask, patch["source_world"], patch["source_uv"], [1,0,0], K, BOUNDARY)
        self.assertFalse(report["has_trusted_axis_finite_edge"])
        self.assertGreater(report["segmentation_only_fraction"], .5)

    def test_12_floor_parallel_plane_is_not_moving_link(self):
        source, targets, mask, patch, axis = source_and_targets(False, True)
        for target in targets: target["depth"][:] = 1.
        rows = {model: [evaluate_source_indexed_patch(patch, target, axis, K, model, ANCHOR)
                        for target in targets] for model in ("static", "drawer")}
        times = {0: 0., 1: .2, 2: .4}
        decision = decide_motion_ownership(
            summarize_transition_chain(rows["static"], 0, 0., times, CHAIN),
            summarize_transition_chain(rows["drawer"], 0, 0., times, CHAIN),
            {"tangent_motion": True, "has_trusted_axis_finite_edge": False}, {"passed": False})
        self.assertNotEqual(decision["label"], "MOVING_LINK")

    def test_13_reveal_uses_real_frame_delay_not_filtered_index(self):
        patch, before, after, future, masks, transition = self._causal_fixture(3, .6, .05)
        event = measure_causal_reveal(patch, before, after, future, masks, [1,0,0], K, transition, CAUSAL)
        self.assertEqual(event["frame_gap"], 3); self.assertAlmostEqual(event["elapsed_s"], .6)

    def test_14_actual_drawer_silhouette_transform_compares_target_mask(self):
        source = frame(1, 0., np.ones((100,100), np.float32)); target = frame(2, .04, np.ones((100,100), np.float32))
        mask = rectangle_mask(); transformed = transform_trusted_drawer_silhouette(
            source, target, mask, [1,0,0], K, CAUSAL["silhouette"])
        comparison = compare_predicted_actual_silhouette(transformed["predicted_mask"], transformed["predicted_mask"])
        self.assertEqual(comparison["silhouette_iou"], 1.)

    def _causal_fixture(self, gap=1, elapsed=.2, depth_gap=.05):
        mask = rectangle_mask(40, 60, 40, 60)
        before_depth = np.full((100,100), 2., np.float32); before_depth[mask] = 1.
        before = frame(1, 0., before_depth)
        after = frame(1 + gap, .04, np.full((100,100), 1. + depth_gap, np.float32))
        predicted = transform_trusted_drawer_silhouette(before, after, mask, [1,0,0], K, CAUSAL["silhouette"])["predicted_mask"]
        # Candidate world patch is the revealed plane behind the drawer.
        candidate_source = frame(8, .04, np.full((100,100), 1. + depth_gap, np.float32))
        patch = extract_source_patch(candidate_source, mask, K, {"maximum_source_samples": 800})
        future = [frame(after["source"] + 1, .04, np.full((100,100), 1. + depth_gap, np.float32))]
        masks = {before["source"]: mask, after["source"]: predicted,
                 future[0]["source"]: predicted}
        transition = {"valid": True, "frame_gap": gap, "elapsed_s": elapsed,
                      "delta_q": .04, "chronological_direction": "forward"}
        return patch, before, after, future, masks, transition

    def test_15_large_same_ray_depth_gap_rejects_causal_reveal(self):
        patch, before, after, future, masks, transition = self._causal_fixture(1, .2, .5)
        event = measure_causal_reveal(patch, before, after, future, masks, [1,0,0], K, transition, CAUSAL)
        self.assertFalse(event["positive_causal_static_disocclusion"])
        self.assertFalse(event["checks"]["local_depth_gap"])

    def test_16_local_trusted_drawer_reveal_is_accepted(self):
        patch, before, after, future, masks, transition = self._causal_fixture()
        event = measure_causal_reveal(patch, before, after, future, masks, [1,0,0], K, transition, CAUSAL)
        self.assertTrue(event["positive_causal_static_disocclusion"], event["failed_checks"])

    def test_17_world_residual_is_measured_not_fixed_zero(self):
        patch, before, after, future, masks, transition = self._causal_fixture()
        after["depth"][:] += .01; future[0]["depth"][:] += .01
        event = measure_causal_reveal(patch, before, after, future, masks, [1,0,0], K, transition, CAUSAL)
        self.assertIsNotNone(event["persistent_world_residual_median"])
        self.assertGreater(event["persistent_world_residual_median"], 0.)

    def test_18_plateau_only_cannot_create_ownership(self):
        states = decompose_local_motion_states([1,2,3], [0,.2,.4], [.3,.3,.3], MOTION)
        transitions, _ = build_local_active_transitions(states, LOCAL)
        self.assertEqual(transitions, [])
        self.assertEqual(local_targets_for_source(2, transitions), [])

    def test_19_correlated_chain_metrics_form_one_route(self):
        chain = {"positive_chain": True, "contradiction_intervals": [], "verified_target_count": 2}
        absent = {"positive_chain": False, "contradiction_intervals": [], "verified_target_count": 0}
        decision = decide_motion_ownership(absent, chain,
            {"tangent_motion": False, "has_trusted_axis_finite_edge": False}, {"passed": False})
        self.assertEqual(sum(decision["evidence_families"].values()), 1)

    def test_20_independent_moving_static_conflict_is_conflicting(self):
        moving = {"positive_chain": True, "contradiction_intervals": [], "verified_target_count": 2}
        absent = {"positive_chain": False, "contradiction_intervals": [], "verified_target_count": 0}
        decision = decide_motion_ownership(absent, moving,
            {"tangent_motion": False, "has_trusted_axis_finite_edge": False}, {"passed": False}, True)
        self.assertEqual(decision["label"], "CONFLICTING")
        self.assertEqual(decision["formal_ownership_label"], "UNKNOWN")

    def test_21_revolute_transform_regression(self):
        transform = ArticulatedTransform("revolute", [0,0,1], [0,0,0])
        point = np.asarray([[1., 0., 0.]])
        result = transform.apply_between(point, 0., np.pi/2)
        np.testing.assert_allclose(result, [[0.,1.,0.]], atol=1e-7)

    def test_22_registered_depth_hard_gate_regression(self):
        per_frame = [{"fitted_scale_to_m": .0002, "valid_mask_iou": .999,
                      "median_abs_error_m": .0002, "p90_abs_error_m": .001} for _ in range(3)]
        cfg = {"enabled": True, "contract_scale_to_m": .0002, "minimum_frames": 3,
               "maximum_scale_relative_error_to_contract": .01,
               "maximum_per_frame_scale_drift": .005, "minimum_valid_mask_iou": .98,
               "maximum_median_abs_error_m": .001, "maximum_p90_abs_error_m": .005}
        self.assertTrue(evaluate_scale_consistency_gate(per_frame, .0002, cfg)["passed"])


if __name__ == "__main__":
    unittest.main()
