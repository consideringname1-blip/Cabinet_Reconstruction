import unittest

import numpy as np

from tools.itaco_region_assignment_v4 import LABEL_DRAWER,LABEL_STATIC,LABEL_UNKNOWN
from tools.itaco_region_assignment_v4.frame_data import validity_masks
from tools.itaco_region_assignment_v4.projective_models import CONTRADICTION,OCCLUDED,SUPPORTED,evaluate_model
from tools.itaco_region_assignment_v4.region_assignment import merge_seed_propagations,resolve_interaction,resolve_positive_support
from tools.itaco_region_assignment_v4.region_evidence import classify_region,deterministic_sample,eligible_target_indices,evaluate_region

K=np.asarray([[100.,0,50.],[0,100.,50.],[0,0,1.]])
AXIS=np.asarray([1.,0,0]); POSE=np.eye(4)
PROJECTIVE={"depth_support_threshold_m":.03,"occlusion_margin_m":.03,"free_space_margin_m":.03}
REGION={**PROJECTIVE,"minimum_pair_q_span_fraction":.15,"maximum_target_frames":8,"maximum_sampled_points_per_region":100,
        "minimum_valid_region_pixels":4,"minimum_sampled_points":4,"minimum_target_frames":1,"minimum_testable_points_per_target":1,
        "minimum_total_testable_points":1,"frame_support_ratio_threshold":.5,"minimum_support_ratio_median":.5,"minimum_support_ratio_p25":.5,
        "maximum_contradiction_ratio":.25,"minimum_supported_frames":1,"minimum_point_preferred_fraction":.4,"minimum_model_score_margin":.1,
        "point_preference_score_margin":.2,"minimum_point_testable_observations":1,"mixed_surface_min_fraction":.2,"seed_conflict_overlap_ratio":.1}


def frame(depth=3.,pose=POSE,q=0.,source=0,valid=True):
    return {"depth":np.full((100,100),depth,np.float32),"valid":np.full((100,100),valid,bool),"pose":np.asarray(pose,float),
            "q":q,"source":source,"rgb":np.full((100,100,3),120,np.uint8)}


def aggregate(score,support=.8,contradiction=.1):
    return {"support_ratio_median":support,"support_ratio_p25":support,"contradiction_ratio_median":contradiction,
            "supported_frames":3,"testable_points":100,"usable_target_frames":3,"score":score}


def evidence(static_score=-.5,drawer_score=.7,seed=0.,static_fraction=.05,drawer_fraction=.8):
    return {"region_valid_pixel_count":100,"region_sampled_point_count":100,"target_frame_count":3,"seed_overlap_ratio":seed,
            "static":aggregate(static_score,support=.2,contradiction=.7),"drawer":aggregate(drawer_score),
            "static_preferred_fraction":static_fraction,"drawer_preferred_fraction":drawer_fraction,"ambiguous_fraction":.15}


class RegionAssignmentV4Tests(unittest.TestCase):
    def test_synthetic_static_plane_different_pose(self):
        target_pose=np.eye(4); target_pose[0,3]=.02; target=frame(pose=target_pose,q=.1)
        target["depth"][50,48]=1.; target["depth"][50,58]=2.
        point=np.asarray([[0.,0.,1.]])
        static=evaluate_model(point,0,target,.1,AXIS,K,"static",PROJECTIVE)
        drawer=evaluate_model(point,0,target,.1,AXIS,K,"drawer",PROJECTIVE)
        self.assertEqual(static["status"][0],SUPPORTED); self.assertEqual(drawer["status"][0],CONTRADICTION)

    def test_synthetic_drawer_plane(self):
        target=frame(q=.1); target["depth"][50,60]=1.; target["depth"][50,50]=2.
        point=np.asarray([[0.,0.,1.]])
        self.assertEqual(evaluate_model(point,0,target,.1,AXIS,K,"drawer",PROJECTIVE)["status"][0],SUPPORTED)
        self.assertEqual(evaluate_model(point,0,target,.1,AXIS,K,"static",PROJECTIVE)["status"][0],CONTRADICTION)

    def test_occlusion_is_neutral(self):
        target=frame(depth=1.)
        result=evaluate_model(np.asarray([[0.,0.,2.]]),0,target,0,AXIS,K,"static",PROJECTIVE)
        self.assertEqual(result["status"][0],OCCLUDED); self.assertEqual(result["testable_count"],0)

    def test_free_space_is_contradiction(self):
        target=frame(depth=2.)
        result=evaluate_model(np.asarray([[0.,0.,1.]]),0,target,0,AXIS,K,"static",PROJECTIVE)
        self.assertEqual(result["status"][0],CONTRADICTION)

    def test_low_q_pair_not_discriminative(self):
        self.assertEqual(eligible_target_indices(0,np.asarray([0.,.01]),.3,{"minimum_pair_q_span_fraction":.1,"maximum_target_frames":8}),[])

    def test_mixed_region_is_unknown(self):
        item=evidence(static_fraction=.35,drawer_fraction=.35)
        self.assertEqual(classify_region(item,REGION)[:2],("unknown","unknown_mixed_surface"))

    def test_invalid_hand_and_depth_edge_excluded(self):
        rgb=np.full((20,20,3),100,np.uint8); depth=np.ones((20,20),np.float32); depth[:,10:]=1.3; hand=np.zeros((20,20),bool); hand[5,5]=True
        masks=validity_masks(rgb,depth,hand,{"rgb_erosion_pixels":0,"depth_min_m":.2,"depth_max_m":4.,"depth_discontinuity_threshold_m":.08,"depth_boundary_erosion_pixels":1})
        self.assertFalse(masks["valid"][5,5]); self.assertFalse(masks["valid"][10,10]); self.assertTrue(masks["valid"][5,2])

    def test_overlapping_accepted_regions_conflict(self):
        valid=np.ones((5,5),bool); a=np.zeros((5,5),bool); b=np.zeros((5,5),bool); a[2,2]=True; b[2,2]=True
        labels,_=resolve_interaction(valid,[{"mask":a},{"mask":b}],[{"label":"static"},{"label":"drawer"}])
        self.assertEqual(labels[2,2],LABEL_UNKNOWN)

    def test_repaired_seed_cannot_override_geometry(self):
        item=evidence(static_score=.7,drawer_score=-.5,seed=.5,static_fraction=.8,drawer_fraction=.05)
        item["static"]=aggregate(.7); item["drawer"]=aggregate(-.5,support=.2,contradiction=.7)
        self.assertEqual(classify_region(item,REGION)[:2],("unknown","unknown_seed_geometry_conflict"))

    def test_open_only_without_positive_evidence_unknown(self):
        labels=resolve_positive_support(np.ones((4,4),bool),np.zeros((4,4),bool),np.zeros((4,4),bool))
        self.assertTrue(np.all(labels==LABEL_UNKNOWN))

    def test_sam2_seed_disagreement_unknown(self):
        valid=np.ones((3,3),bool); yes=np.zeros((3,3),bool); yes[1,1]=True; no=np.zeros_like(yes)
        drawer,conflict,_=merge_seed_propagations([yes,no],valid,.6)
        self.assertFalse(drawer[1,1]); self.assertTrue(conflict[1,1])

    def test_deterministic_region_evidence(self):
        source=frame(depth=1.,q=0.,source=10); target=frame(depth=1.,q=.2,source=20); mask=np.zeros((100,100),bool); mask[40:60,40:60]=True
        first=evaluate_region(0,source,mask,[source,target],np.asarray([0.,.2]),AXIS,K,.2,REGION,np.zeros_like(mask))
        second=evaluate_region(0,source,mask,[source,target],np.asarray([0.,.2]),AXIS,K,.2,REGION,np.zeros_like(mask))
        self.assertTrue(np.array_equal(first["sample_uv"],second["sample_uv"])); self.assertTrue(np.array_equal(first["point_preference"],second["point_preference"]))
        self.assertEqual(first["label"],second["label"]); self.assertEqual(first["per_target"],second["per_target"])

    def test_deterministic_spatial_sampling_covers_region(self):
        mask=np.zeros((100,100),bool); mask[10:90,10:90]=True; a=deterministic_sample(mask,100); b=deterministic_sample(mask,100)
        self.assertTrue(np.array_equal(a,b)); self.assertLessEqual(len(a),100); self.assertGreater(np.ptp(a[:,0]),60); self.assertGreater(np.ptp(a[:,1]),60)

if __name__=="__main__": unittest.main()
