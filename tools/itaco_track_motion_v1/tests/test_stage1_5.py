from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.itaco_track_motion_v1.hand_recovery import diagnose
from tools.itaco_track_motion_v1.reassociation import reassociate


class IdentityMotion:
    type="prismatic"
    state=np.zeros(8)
    def canonicalize(self,points,indices): return np.asarray(points)
    def decanonicalize(self,points,indices): return np.asarray(points)


def observation(track_id,index,point,uv=(50.0,50.0),proposal=""):
    return {"track_id":track_id,"processing_index":index,"original_frame_id":100+index,"timestamp":1000+index,
            "point_world":list(point),"pixel_uv":list(uv),"depth":float(point[2]),"rgb":[100,120,140],"proposal_id":proposal}


def track(track_id,index,point,uv=(50.0,50.0),proposal=""):
    return {"track_id":track_id,"observations":[observation(track_id,index,point,uv,proposal)],"attempted_transitions":0,
            "occlusion_failures":0,"boundary_failures":0,"fb_failures":0,"termination_reason":"test"}


ASSOCIATION={"max_gap_frames":2,"max_static_3d_error_m":.05,"max_moving_3d_error_m":.05,
             "max_static_reprojection_error_px":5.0,"max_moving_reprojection_error_px":5.0,
             "min_appearance_similarity":-.1,"max_depth_difference_m":.1,"depth_consistency_scale_m":.04,
             "proposal_soft_bonus":.04,"min_association_confidence":.05,"missing_normal_factor":.75}


class Stage15Tests(unittest.TestCase):
    def test_gap_association_bridges_short_static_gap(self):
        tracks=[track(0,0,[0,0,1]),track(1,2,[.001,0,1],uv=(50.1,50),proposal="different")]
        merged,records=reassociate(tracks,np.repeat(np.eye(4)[None],4,axis=0),np.asarray([[100,0,50],[0,100,50],[0,0,1.]]),IdentityMotion(),ASSOCIATION)
        self.assertEqual(len(merged),1); self.assertTrue(any(item["accepted"] for item in records))

    def test_proposal_identity_cannot_bridge_geometrically_bad_gap(self):
        tracks=[track(0,0,[0,0,1],proposal="same"),track(1,2,[1,0,1],uv=(150,50),proposal="same")]
        merged,records=reassociate(tracks,np.repeat(np.eye(4)[None],4,axis=0),np.asarray([[100,0,50],[0,100,50],[0,0,1.]]),IdentityMotion(),ASSOCIATION)
        self.assertEqual(len(merged),2); self.assertFalse(any(item["accepted"] for item in records))
        self.assertIn("neither_fixed_model",records[0]["rejection_reason"])

    def test_large_temporal_hand_mask_is_anomaly_without_position_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); records=[]; detector=[]
            for index,size in enumerate((10,70,10)):
                mask=np.zeros((80,80),bool); mask[5:5+size,5:5+size]=True
                path=root/f"{index}.npy"; np.save(path,mask)
                records.append({"original_frame_id":10+index,"timestamp":100+index,"hand_mask_path":str(path),"rgb_path":str(root/f"{index}.png")})
                detector.append({"image":str(root/f"{index}.png"),"grounding_confidences":[.4],"sam_scores":[.9]})
            manifest=root/"manifest.json"; manifest.write_text(json.dumps({"records":detector}))
            config={"image_boundary_band_px":2,"max_neighbor_area_change_ratio":2.,"max_rgb_footprint_coverage":.45,
                    "min_neighbor_iou":.12,"max_centroid_motion_fraction":.16,"max_boundary_contact_ratio":.35,
                    "low_detector_confidence":.3,"large_area_with_low_confidence":.2}
            _,rows,anomalies=diagnose(records,[np.ones((80,80),bool)]*3,manifest,config)
            self.assertTrue(rows[1]["hand_mask_temporal_anomaly"]); self.assertIn("excessive_rgb_footprint_coverage",rows[1]["anomaly_reasons"])
            self.assertNotIn("original_frame_id",config)

    def test_stage15_config_contains_no_forbidden_operation(self):
        import yaml
        path=Path(__file__).parents[1]/"configs/hololens_2026-07-30-002840_stage1_5.yaml"
        config=yaml.safe_load(path.read_text())
        forbidden={"optimize_camera","estimate_axis","model_selection","free_se3","tsdf","nksr","mesh"}
        self.assertFalse(forbidden & set(config))
        self.assertIn("fixed_stage1_classifier" if False else "stage1_config_path",config)


if __name__=="__main__": unittest.main()
