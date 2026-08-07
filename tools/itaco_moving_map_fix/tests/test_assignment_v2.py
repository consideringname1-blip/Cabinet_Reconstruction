import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2]/'fuse_hololens_articulation_assignment_v2.py'; S=importlib.util.spec_from_file_location('v2',P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)
CFG={'minimum_static_frames':3,'minimum_drawer_frames':3,'maximum_spread_m':.02,'evidence_spread_margin_m':.002,'minimum_drawer_q_span_fraction':.2}
def fake(frames,spread,qspan,n=8): return {'inverse':np.arange(n),'frames':np.full(n,frames),'p90':np.full(n,spread),'qspan':np.full(n,qspan)}
def test_static_surface(): assert np.all(M.classify_motion(fake(5,.002,0),fake(1,.02,.3),.3,CFG)[0]==2)
def test_drawer_surface(): assert np.all(M.classify_motion(fake(1,.02,0),fake(5,.002,.3),.3,CFG)[0]==3)
def test_ambiguous_is_unknown(): assert np.all(M.classify_motion(fake(5,.003,.3),fake(5,.003,.3),.3,CFG)[0]==4)
def test_insufficient_is_unknown(): assert np.all(M.classify_motion(fake(1,.003,0),fake(1,.003,0),.3,CFG)[0]==5)
def test_depth_inconsistent_excluded():
    pts=np.array([[0.,0.,1.],[0.,0.,1.1]]); k=np.eye(3); support=np.ones((2,2),bool); z=M.projected_visibility(pts,np.eye(4),k,(2,2),support,np.zeros((2,2),bool),.2,4,.025,0); assert z[4].tolist()==[True,False]
def test_no_remainder_acceptance():
    lab=M.classify_motion(fake(1,.1,0),fake(1,.1,0),.3,CFG)[0]; assert not np.any((lab==2)|(lab==3))
