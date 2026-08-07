"""Periodic, spatially balanced LoFTR tracks anchored in each seed frame."""
from __future__ import annotations
import cv2, numpy as np, torch
from scipy.spatial import cKDTree
from kornia.feature import LoFTR
from .core import periodic_seed_frames

def _infer(model, gray0, gray1):
    a=torch.from_numpy(gray0).float().div_(255)[None,None].cuda(); b=torch.from_numpy(gray1).float().div_(255)[None,None].cuda()
    with torch.inference_mode(): result=model({'image0':a,'image1':b})
    return result['keypoints0'].cpu().numpy(),result['keypoints1'].cpu().numpy(),result['confidence'].cpu().numpy()

def _observation(track, frame, uv, q, k, confidence, fb):
    u,v=np.rint(uv).astype(int); h,w=frame['depth'].shape
    if not (0<=u<w and 0<=v<h and frame['valid'][v,u]): return None
    depth=float(frame['depth'][v,u]); pc=np.asarray([(u-k[0,2])*depth/k[0,0],(v-k[1,2])*depth/k[1,1],depth]); pw=pc@frame['pose'][:3,:3].T+frame['pose'][:3,3]
    proposals=[p['proposal_id'] for p in frame['proposals'] if p['mask'][v,u]]
    return {'track_id':track['track_id'],'seed_frame_id':track['seed_frame_id'],'original_frame_id':int(frame['source']),
            'pixel_uv':[float(uv[0]),float(uv[1])],'depth_m':depth,'point_camera':pc.tolist(),'point_world':pw.tolist(),'q_t':float(q),
            'proposal_ids':proposals,'tracking_confidence':float(confidence),'forward_backward_error':float(fb),'visibility':True,
            'occlusion':False,'depth_edge_distance':float(frame['edge_distance'][v,u]),'hand_flag':False}

def _balanced_seed_keypoints(kp, confidence, valid, cfg):
    stride=int(cfg['grid_stride_pixels']); chosen={}
    order=np.argsort(-confidence,kind='stable')
    for i in order:
        u,v=np.rint(kp[i]).astype(int)
        if not (0<=u<valid.shape[1] and 0<=v<valid.shape[0] and valid[v,u]): continue
        cell=(u//stride,v//stride)
        if cell not in chosen: chosen[cell]=int(i)
    return [chosen[key] for key in sorted(chosen,key=lambda x:(x[1],x[0]))]

def build_loftr_tracks(frames, axis, q, k, cfg):
    del axis
    tcfg=cfg['tracking']; torch.manual_seed(int(cfg['determinism']['seed'])); torch.cuda.manual_seed_all(int(cfg['determinism']['seed']))
    model=LoFTR(pretrained='indoor').eval().cuda(); grays=[cv2.cvtColor(f['rgb'],cv2.COLOR_BGR2GRAY) for f in frames]; tracks=[]; next_id=0
    for seed in periodic_seed_frames(len(frames),int(tcfg['reseed_interval_frames'])):
        if seed==len(frames)-1: continue
        kp0,_,conf0=_infer(model,grays[seed],grays[seed+1]); usable=conf0>=float(tcfg['loftr_min_confidence']); indices=_balanced_seed_keypoints(kp0[usable],conf0[usable],frames[seed]['valid'],tcfg)
        seed_points=kp0[usable][indices]; batch=[]
        for uv in seed_points:
            track={'track_id':next_id,'seed_frame_id':int(frames[seed]['source']),'seed_uv':uv.astype(np.float32),'observations':[]}; next_id+=1
            first=_observation(track,frames[seed],uv,q[seed],k,1.0,0.0)
            if first is not None: track['observations'].append(first); batch.append(track); tracks.append(track)
        for target_index in range(seed+1,len(frames)):
            kp_seed,kp_target,confidence=_infer(model,grays[seed],grays[target_index]); rkp_target,rkp_seed,rconfidence=_infer(model,grays[target_index],grays[seed])
            if not len(kp_seed) or not len(rkp_target): continue
            forward=cKDTree(kp_seed); reverse=cKDTree(rkp_target)
            for track in batch:
                distance,match=forward.query(track['seed_uv']); target=kp_target[match]; reverse_distance,back_match=reverse.query(target); back=rkp_seed[back_match]
                fb=float(np.linalg.norm(back-track['seed_uv'])); score=float(min(confidence[match],rconfidence[back_match]))
                if distance>float(tcfg['loftr_seed_association_radius_pixels']) or reverse_distance>float(tcfg['loftr_reverse_association_radius_pixels']) or fb>float(tcfg['forward_backward_threshold_pixels']) or score<float(tcfg['loftr_min_confidence']): continue
                observation=_observation(track,frames[target_index],target,q[target_index],k,score,fb)
                if observation is not None: track['observations'].append(observation)
    for track in tracks: track.pop('seed_uv',None)
    return tracks
