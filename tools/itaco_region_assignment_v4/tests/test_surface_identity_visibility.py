import unittest

import numpy as np

from tools.itaco_region_assignment_v4.frame_data import evaluate_scale_consistency_gate
from tools.itaco_region_assignment_v4.projective_models import evaluate_model
from tools.itaco_region_assignment_v4.surface_identity_visibility import (
    attribute_occlusion, contradiction_run, descriptor_compatibility,
    proposal_memberships, select_continuity_chain, surface_descriptor,
    surface_switches, temporal_cues,
)


DESC = {"voxel_sizes_m": [.01], "minimum_valid_points": 3}
CONT = {"overlap_radius_voxels": 1.5, "normal_scale_degrees": 20., "plane_offset_scale_m": .04,
        "centroid_scale_m": .12, "weights": {"normal": .15, "plane_offset": .2, "centroid": .25, "extent": .1, "overlap": .3},
        "support_emission_weight": .15, "transition_weight": .35, "switch_normal_degrees": 18.,
        "switch_plane_offset_m": .035, "switch_centroid_m": .1}
CONTRA = {"per_frame_fraction_threshold": .15, "minimum_spatial_coherence_fraction": .25,
          "minimum_consecutive_frames": 2, "minimum_q_span_m": .015}
TEMP = {"early_q_fraction": .34, "late_q_fraction": .34, "minimum_early_drawer_occluded_fraction": .25,
        "minimum_late_same_surface_support_fraction": .45, "minimum_temporal_monotonicity": .1,
        "minimum_drawer_attachment_score": .45}


def plane(offset=np.zeros(3), size=.12):
    x, y = np.meshgrid(np.linspace(-size, size, 15), np.linspace(-size, size, 15))
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))) + np.asarray(offset)


def descriptor(points):
    return surface_descriptor(points, [.01], 3)


def candidate(points, supported=1.):
    desc = descriptor(points)
    return {"proposal_id": "metadata_only", "descriptor": desc, "voxel_m": .01, "supported_fraction": supported,
            "source_compatibility": descriptor_compatibility(descriptor(plane()), desc, CONT, .01)}


class SurfaceIdentityVisibilityTests(unittest.TestCase):
    def test_same_static_plane_world_continuity_high(self):
        score = descriptor_compatibility(descriptor(plane()), descriptor(plane(offset=[.002, 0, 0])), CONT, .01)["score"]
        self.assertGreater(score, .8)

    def test_moving_finite_plane_drawer_canonical_continuity_high(self):
        axis = np.asarray([1., 0, 0]); q = .2
        source = plane(); observed = source + q * axis
        canonical = observed - q * axis
        self.assertGreater(descriptor_compatibility(descriptor(source), descriptor(canonical), CONT, .01)["score"], .95)
        self.assertLess(descriptor_compatibility(descriptor(source), descriptor(observed), CONT, .01)["score"], .8)

    def test_large_plane_occupancy_is_not_identity(self):
        k = np.asarray([[100., 0, 50.], [0, 100., 50.], [0, 0, 1.]])
        target = {"depth": np.ones((100, 100)), "valid": np.ones((100, 100), bool), "pose": np.eye(4)}
        point = np.asarray([[0., 0, 1.]])
        cfg = {"depth_support_threshold_m": .03, "occlusion_margin_m": .03, "free_space_margin_m": .03}
        self.assertEqual(evaluate_model(point, 0, target, .1, np.asarray([1., 0, 0]), k, "static", cfg)["support_ratio"], 1.)
        self.assertEqual(evaluate_model(point, 0, target, .1, np.asarray([1., 0, 0]), k, "drawer", cfg)["support_ratio"], 1.)

    def test_surface_switch_parallel_plane_is_penalized(self):
        selected = [candidate(plane()), candidate(plane(offset=[0, 0, .08]))]
        result = surface_switches(selected, CONT)
        self.assertEqual(result["switch_count"], 1)

    def test_explained_drawer_occlusion(self):
        evidence = {"uv": np.asarray([[1, 1]]), "status": np.asarray([2], np.uint8),
                    "predicted_depth_m": np.asarray([2.]), "observed_depth_m": np.asarray([1.])}
        drawer = np.zeros((3, 3), bool); drawer[1, 1] = True
        result = attribute_occlusion(evidence, drawer, np.zeros_like(drawer))
        self.assertEqual(result["counts"]["explained_occluded_by_trusted_drawer"], 1)

    def test_unknown_occluder_not_positive(self):
        evidence = {"uv": np.asarray([[1, 1]]), "status": np.asarray([2], np.uint8),
                    "predicted_depth_m": np.asarray([2.]), "observed_depth_m": np.asarray([1.])}
        empty = np.zeros((3, 3), bool)
        self.assertEqual(attribute_occlusion(evidence, empty, empty)["counts"]["occluded_by_unknown"], 1)

    def test_disocclusion_sequence(self):
        static = [{"target_q_m": q, "explained_drawer_occlusion_fraction": occ,
                   "selected_same_surface_supported": support} for q, occ, support in
                  [(0., .8, 0), (.1, .5, 0), (.2, 0, 1), (.3, 0, 1)]]
        drawer = [{**row, "selected_same_surface_supported": 0} for row in static]
        result = temporal_cues({"static": static, "drawer": drawer}, TEMP)
        self.assertTrue(result["strong_static_disocclusion"])

    def test_continuous_contradiction_run_not_hidden_by_median(self):
        rows = [{"target_frame_id": frame, "target_q_m": q, "contradiction_fraction": fraction,
                 "contradiction_spatial_coherence_fraction": .9} for frame, q, fraction in
                [(1, 0., 0), (2, .02, .3), (3, .04, .4), (4, .06, 0)]]
        result = contradiction_run(rows, CONTRA)
        self.assertTrue(result["coherent_contradiction_run"]); self.assertEqual(result["longest_consecutive_contradiction_run"], 2)

    def test_overlapping_proposals_not_independent_evidence(self):
        uv = np.asarray([[1, 1]]); selected = np.asarray([True])
        a = np.zeros((3, 3), bool); b = np.zeros((3, 3), bool); a[1, 1] = b[1, 1] = True
        proposals = [{"proposal_id": "a", "source_layer": 0, "mask": a}, {"proposal_id": "b", "source_layer": 1, "mask": b}]
        result = proposal_memberships(uv, selected, proposals)
        self.assertEqual(result["unique_supported_point_count"], 1)
        self.assertAlmostEqual(sum(row["fractional_hit_count"] for row in result["all_memberships"]), 1.)

    def test_no_cross_frame_uid_identity(self):
        a = candidate(plane()); b = candidate(plane()); a["proposal_id"] = "frame1_layer25"; b["proposal_id"] = "frame2_layer3"
        first = select_continuity_chain([[a], [b]], CONT)
        b["proposal_id"] = "frame2_layer25"
        second = select_continuity_chain([[a], [b]], CONT)
        self.assertEqual(first[1]["source_compatibility"], second[1]["source_compatibility"])

    def test_deterministic_rerun(self):
        frames = [[candidate(plane(offset=[0, 0, .001]))], [candidate(plane(offset=[.002, 0, 0]))]]
        a = select_continuity_chain(frames, CONT); b = select_continuity_chain(frames, CONT)
        self.assertEqual([x["proposal_id"] for x in a], [x["proposal_id"] for x in b])

    def test_registered_depth_gate_regression(self):
        rows = [{"fitted_scale_to_m": value, "valid_mask_iou": .999, "median_abs_error_m": .0001, "p90_abs_error_m": .0002}
                for value in (.0002, .00020001, .00019999)]
        cfg = {"enabled": True, "contract_scale_to_m": .0002, "minimum_frames": 3,
               "maximum_scale_relative_error_to_contract": .01, "maximum_per_frame_scale_drift": .005,
               "minimum_valid_mask_iou": .98, "maximum_median_abs_error_m": .001, "maximum_p90_abs_error_m": .005}
        self.assertTrue(evaluate_scale_consistency_gate(rows, .0002, cfg)["passed"])
        self.assertFalse(evaluate_scale_consistency_gate(rows, .001, cfg)["passed"])


if __name__ == "__main__": unittest.main()
