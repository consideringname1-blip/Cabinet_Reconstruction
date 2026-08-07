import tempfile
import unittest
from pathlib import Path
import numpy as np

from tools.itaco_funrec_assignment_v3.core import classify_track, four_state, merge_propagations, periodic_seed_frames, vote_region, LABEL_DRAWER, LABEL_INVALID, LABEL_STATIC, LABEL_UNKNOWN


CFG={"min_track_observations":4,"min_track_q_span_fraction":.15,"residual_margin_m":.003}
AXIS=np.array([1.,0,0]); TRAVEL=.3
def obs(points,q):
    return [{"track_id":1,"original_frame_id":i,"point_world":p,"q_t":x,"tracking_confidence":1.} for i,(p,x) in enumerate(zip(points,q))]

class CoreTests(unittest.TestCase):
    def test_static(self):
        q=np.linspace(0,.3,6); e=classify_track(obs(np.tile([1.,2,3],(6,1)),q),AXIS,TRAVEL,CFG); self.assertEqual(e["label"],"static")
    def test_prismatic(self):
        q=np.linspace(0,.3,6); p=np.array([1.,2,3])+q[:,None]*AXIS; self.assertEqual(classify_track(obs(p,q),AXIS,TRAVEL,CFG)["label"],"moving")
    def test_low_q_unknown(self):
        q=np.linspace(0,.01,6); self.assertEqual(classify_track(obs(np.tile([1.,2,3],(6,1)),q),AXIS,TRAVEL,CFG)["reason"],"unknown_low_excitation")
    def test_ambiguous_unknown(self):
        q=np.linspace(0,.3,6); p=np.array([1.,2,3])+q[:,None]*AXIS*.5; self.assertEqual(classify_track(obs(p,q),AXIS,TRAVEL,{**CFG,"residual_margin_m":.2})["label"],"unknown")
    def test_periodic_reseed_includes_late(self): self.assertEqual(periodic_seed_frames(8,3),[0,3,6,7])
    def test_mixed_region_unknown(self):
        c={"min_labeled_tracks":2,"moving_ratio_threshold":.7,"static_ratio_threshold":.3,"max_unknown_track_fraction":.6}; self.assertEqual(vote_region([1,2],{1:"moving",2:"static"},c)["label"],"unknown")
    def test_propagation_conflict(self):
        votes=np.array([[[1]], [[-1]]]); drawer,conflict=merge_propagations(votes,np.ones((1,1),bool)); self.assertFalse(drawer[0,0]); self.assertTrue(conflict[0,0])
    def test_open_only_unknown(self):
        labels=four_state(np.ones((1,1),bool),np.zeros((1,1),bool),np.zeros((1,1),bool)); self.assertEqual(labels[0,0],LABEL_UNKNOWN)
    def test_four_state_exclusive_complete(self):
        valid=np.array([[0,1,1,1]],bool); labels=four_state(valid,np.array([[0,1,0,1]],bool),np.array([[0,0,1,1]],bool)); self.assertEqual(labels.tolist(),[[0,2,1,3]])
    def test_invalid_cannot_be_owned(self):
        labels=four_state(np.zeros((1,2),bool),np.ones((1,2),bool),np.ones((1,2),bool)); self.assertTrue(np.all(labels==LABEL_INVALID))
    def test_absolute_residual_unknown(self):
        q=np.linspace(0,.3,6); p=np.array([1.,2,3])+q[:,None]*AXIS+np.arange(6)[:,None]*np.array([0.,0.1,0.]); self.assertEqual(classify_track(obs(p,q),AXIS,TRAVEL,{**CFG,"max_absolute_residual_m":.05})["reason"],"unknown_absolute_residual")

    def test_deterministic_repeat(self):
        q=np.linspace(0,.3,6); p=np.array([1.,2,3])+q[:,None]*AXIS; self.assertEqual(classify_track(obs(p,q),AXIS,TRAVEL,CFG),classify_track(obs(p,q),AXIS,TRAVEL,CFG))

if __name__=="__main__": unittest.main()
