"""End-to-end conservative ownership assignment v3; deliberately stops before fusion."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import importlib.metadata as metadata
import torch

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import yaml

from . import OUTPUT_KIND
from .core import (LABEL_DRAWER, LABEL_INVALID, LABEL_STATIC, LABEL_UNKNOWN, classify_track,
                   core_hash, four_state, periodic_seed_frames, vote_region, write_ply)


def load_poses(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return np.stack([np.asarray([[float(x) for x in row.split()] for row in lines[i + 1:i + 5]]) for i in range(0, len(lines), 5)])


def load_ply_xyz(path: Path) -> np.ndarray:
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def project(points: np.ndarray, pose: np.ndarray, k: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]; h, w = shape
    u = np.rint(k[0, 0] * camera[:, 0] / np.maximum(z, 1e-12) + k[0, 2]).astype(int)
    v = np.rint(k[1, 1] * camera[:, 1] / np.maximum(z, 1e-12) + k[1, 2]).astype(int)
    inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, z, inside


def unproject(depth: np.ndarray, pose: np.ndarray, k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(mask); z = depth[yy, xx]
    camera = np.stack(((xx-k[0,2])*z/k[0,0], (yy-k[1,2])*z/k[1,1], z), 1)
    return camera @ pose[:3, :3].T + pose[:3, 3]


def mask_valid(rgb: np.ndarray, depth: np.ndarray, hand: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    footprint = np.any(rgb != 0, axis=2)
    if int(cfg["rgb_erosion_pixels"]):
        n = 2*int(cfg["rgb_erosion_pixels"])+1
        footprint = cv2.erode(footprint.astype(np.uint8), np.ones((n,n),np.uint8)).astype(bool)
    depth_ok = np.isfinite(depth) & (depth >= float(cfg["depth_min_m"])) & (depth <= float(cfg["depth_max_m"]))
    edge_distance = distance_transform_edt(depth_ok)
    valid = footprint & depth_ok & ~hand & (edge_distance > float(cfg["depth_boundary_erosion_pixels"]))
    return valid, edge_distance


def load_proposals(source: int, cfg: dict) -> list[dict]:
    mapping = json.loads(Path(cfg["inputs"]["autoseg_reverse_mapping"]).read_text())
    reverse = {int(x["source_index"]): int(x["local_index"]) for x in mapping}
    if source not in reverse:
        return []
    path = Path(cfg["inputs"]["autoseg_npz_dir"]) / f"mask_{reverse[source]:03d}.npz"
    layers = np.load(path)["a"]
    if layers.ndim == 4: layers = layers[:, 0]
    return [{"proposal_id": f"frame_{source}_layer_{i}", "source_uid": int(i), "mask": m.astype(bool),
             "score": float(cfg["proposals"]["legacy_score"]), "source": str(path)}
            for i,m in enumerate(layers) if int(m.sum()) >= int(cfg["proposals"]["min_area_px"])]


def proposal_ids_at(proposals: list[dict], uv: np.ndarray) -> list[str]:
    u,v = np.rint(uv).astype(int)
    return [p["proposal_id"] for p in proposals if p["mask"][v,u]]


def make_tracks(frames: list[dict], axis: np.ndarray, q: np.ndarray, k: np.ndarray, cfg: dict) -> list[dict]:
    tcfg = cfg["tracking"]
    grays = [cv2.cvtColor(f["rgb"], cv2.COLOR_BGR2GRAY) for f in frames]
    tracks=[]; next_id=0
    lk=dict(winSize=(int(tcfg["lk_window_pixels"]),)*2, maxLevel=int(tcfg["lk_max_level"]),
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,int(tcfg["lk_iterations"]),float(tcfg["lk_epsilon"])))
    for seed in periodic_seed_frames(len(frames), int(tcfg["reseed_interval_frames"])):
        f=frames[seed]; stride=int(tcfg["grid_stride_pixels"]); offset=stride//2
        yy,xx=np.mgrid[offset:f["rgb"].shape[0]:stride,offset:f["rgb"].shape[1]:stride]
        points=np.stack((xx.ravel(),yy.ravel()),1).astype(np.float32)
        keep=f["valid"][yy.ravel(),xx.ravel()]
        points=points[keep]
        # Uniform grid is proposal-agnostic, so both proposal interiors and exterior static support are sampled.
        for initial in points:
            track_id=next_id; next_id+=1; obs=[]; uv=initial.copy(); confidence=1.0
            for index in range(seed,len(frames)):
                cur=frames[index]; u,v=np.rint(uv).astype(int)
                if not (0<=u<cur["depth"].shape[1] and 0<=v<cur["depth"].shape[0] and cur["valid"][v,u]): break
                depth=float(cur["depth"][v,u]); pc=np.asarray([(u-k[0,2])*depth/k[0,0],(v-k[1,2])*depth/k[1,1],depth])
                pw=pc@cur["pose"][:3,:3].T+cur["pose"][:3,3]
                obs.append({"track_id":track_id,"seed_frame_id":int(frames[seed]["source"]),"original_frame_id":int(cur["source"]),
                            "pixel_uv":[float(uv[0]),float(uv[1])],"depth_m":depth,"point_camera":pc.tolist(),"point_world":pw.tolist(),
                            "q_t":float(q[index]),"proposal_ids":proposal_ids_at(cur["proposals"],uv),"tracking_confidence":float(confidence),
                            "forward_backward_error":0.0 if index==seed else float(fb),"visibility":True,"occlusion":False,
                            "depth_edge_distance":float(cur["edge_distance"][v,u]),"hand_flag":False})
                if index==len(frames)-1: break
                p0=uv.reshape(1,1,2); p1,s1,e1=cv2.calcOpticalFlowPyrLK(grays[index],grays[index+1],p0,None,**lk)
                if p1 is None or not s1[0,0]: break
                pb,sb,_=cv2.calcOpticalFlowPyrLK(grays[index+1],grays[index],p1,None,**lk)
                if pb is None or not sb[0,0]: break
                fb=float(np.linalg.norm(pb[0,0]-uv))
                if fb>float(tcfg["forward_backward_threshold_pixels"]): break
                confidence=float(np.exp(-fb/max(float(tcfg["forward_backward_threshold_pixels"]),1e-9))); uv=p1[0,0]
            tracks.append({"track_id":track_id,"seed_frame_id":int(frames[seed]["source"]),"observations":obs})
    return tracks


def select_keyframes(frames: list[dict], q: np.ndarray, tracks: list[dict], cfg: dict) -> list[dict]:
    counts=np.zeros(len(frames),int); source_to_local={f["source"]:i for i,f in enumerate(frames)}
    for t in tracks:
        for o in t["observations"]: counts[source_to_local[o["original_frame_id"]]]+=1
    selected=[]
    for quantile in cfg["keyframes"]["q_quantiles"]:
        target=float(np.quantile(q,float(quantile)))
        score=np.abs(q-target)/(max(float(np.ptp(q)),1e-9)) - float(cfg["keyframes"]["track_count_weight"])*counts/max(counts.max(),1)
        for index in np.argsort(score):
            if all(abs(int(index)-x["local_index"])>=int(cfg["keyframes"]["minimum_separation_frames"]) for x in selected): break
        selected.append({"local_index":int(index),"original_frame_id":int(frames[index]["source"]),"q_m":float(q[index]),"track_observation_count":int(counts[index]),"target_quantile":float(quantile)})
    return sorted(selected,key=lambda x:x["local_index"])


def region_votes(frames: list[dict], tracks: list[dict], evidence: list[dict], keyframes: list[dict], cfg: dict, out: Path) -> tuple[list[dict], list[dict]]:
    labels={int(x["track_id"]):x["label"] for x in evidence}; obs_by_frame={f["source"]:[] for f in frames}
    for t in tracks:
        for o in t["observations"]: obs_by_frame[o["original_frame_id"]].append(o)
    rows=[]; seeds=[]
    for key in keyframes:
        f=frames[key["local_index"]]; drawer=np.zeros(f["valid"].shape,bool); static=np.zeros_like(drawer); unknown=np.zeros_like(drawer)
        for proposal in f["proposals"]:
            tids=[o["track_id"] for o in obs_by_frame[f["source"]] if proposal["mask"][int(round(o["pixel_uv"][1])),int(round(o["pixel_uv"][0]))]]
            vote=vote_region(tids,labels,cfg["region_voting"]); row={"original_frame_id":f["source"],"proposal_id":proposal["proposal_id"],**vote}; rows.append(row)
            if vote["label"]=="drawer": drawer|=proposal["mask"]
            elif vote["label"]=="static": static|=proposal["mask"]
            else: unknown|=proposal["mask"]
        conflict=drawer&static; drawer&=~conflict; static&=~conflict; unknown|=conflict
        for name,mask in (("drawer",drawer),("static",static),("unknown",unknown)):
            d=out/f"keyframe_{name}_masks"; d.mkdir(exist_ok=True); np.save(d/f"{f['source']}.npy",mask)
        if drawer.any(): seeds.append({"seed_id":f"seed_{f['source']}","original_frame_id":f["source"],"combined_index":f["source"]-frames[0]["source"],"mask_path":str((out/'keyframe_drawer_masks'/f"{f['source']}.npy").resolve()),"area_px":int(drawer.sum())})
    return rows,seeds


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows: path.write_text(""); return
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main(config_path: Path) -> None:
    cfg=yaml.safe_load(config_path.read_text()); out=Path(cfg["inputs"]["output_dir"])
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"non-overwrite output exists: {out}")
    out.mkdir(parents=True); (out/"visualization").mkdir(); (out/"per_frame_assignment").mkdir()
    (out/"config_resolved.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    inp=cfg["inputs"]; raw=Path(inp["raw_root"]); repro=json.loads(Path(inp["reproduction_manifest"]).read_text())
    interaction=json.loads(Path(inp["interaction_manifest"]).read_text())["source_indices"]
    open_ids=json.loads(Path(inp["open_manifest"]).read_text())["source_indices"]
    axis=np.load(inp["axis_path"]).astype(float); axis/=np.linalg.norm(axis); q=np.load(inp["q_path"]).astype(float); travel=float(q.max())
    if interaction != list(range(interaction[0],interaction[-1]+1)) or not np.array_equal(axis,np.asarray(repro["fixed_quantities"]["opening_axis_world"])) or abs(travel-float(repro["q_recomputation"]["monotonic_range_m"]))>1e-12:
        raise RuntimeError("frozen input conflict")
    poses=load_poses(raw/"pinhole_projection/odometry.log")
    pose_check=np.load(inp["interaction_pose_path"])
    if np.max(np.abs(pose_check-poses[np.asarray(interaction)]))>float(cfg["validity"]["pose_equality_tolerance"]): raise RuntimeError("fixed pose conflict")
    root=raw/"pinhole_projection"
    rgb_paths=[root/x.split(maxsplit=1)[1].replace('\\','/') for x in (root/'rgb.txt').read_text().splitlines() if x.strip()]
    depth_paths=[root/x.split(maxsplit=1)[1].replace('\\','/') for x in (root/'depth.txt').read_text().splitlines() if x.strip()]
    fx,fy,cx,cy=np.loadtxt(root/'calibration.txt').reshape(-1)[:4]; k=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    frames=[]
    for local,source in enumerate(interaction):
        rgb=cv2.imread(str(rgb_paths[source])); depth=cv2.imread(str(depth_paths[source]),cv2.IMREAD_UNCHANGED).astype(np.float32)*float(cfg["validity"]["depth_scale_to_m"])
        hand=np.load(Path(inp["corrected_hand_dir"])/f"{source}.npy").squeeze().astype(bool)
        valid,edge=mask_valid(rgb,depth,hand,cfg["validity"]); frames.append({"source":source,"rgb":rgb,"depth":depth,"hand":hand,"valid":valid,"edge_distance":edge,"pose":poses[source],"proposals":load_proposals(source,cfg)})
    audit={"output_kind":OUTPUT_KIND,"tracker":{"name":"Kornia LoFTR indoor with periodic grid reseeding","implementation":"kornia.feature.LoFTR 0.8.2; cached indoor checkpoint; v3 periodic association implementation","opencv_version":cv2.__version__,"forward_backward_check":True,"visibility_occlusion":"image bounds, registered-depth validity and depth-edge filtering; no learned occlusion model","package_version":cfg["audit"]["loftr_package_version"],"model_weights":cfg["audit"]["loftr_checkpoint"],"model_weight_sha256":cfg["audit"]["loftr_checkpoint_sha256"]},"input_resolution":[320,288],"registered_depth":{"source":"pinhole_projection/depth.txt registered uint16 PNG","scale_to_m":cfg["validity"]["depth_scale_to_m"],"ply_self_zbuffer_used":False},"proposals":{"source":"AutoSeg-SAM2 small final-output NPZ","generator":"SAM1 automatic masks + SAM2 propagation","autoseg_revision":cfg["audit"]["autoseg_revision"],"sam1_weight":inp["sam1_checkpoint"]},"sam2":{"entry":"sam2.build_sam.build_sam2_video_predictor/add_new_mask/propagate_in_video","revision":cfg["audit"]["sam2_revision"],"checkpoint":inp["sam2_checkpoint"],"checkpoint_sha256":cfg["audit"]["sam2_checkpoint_sha256"]},"python_environment":sys.executable,"new_dependencies_required":False}
    (out/"tracker_and_segmentation_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    env={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"numpy":np.__version__,"opencv":cv2.__version__,"torch":torch.__version__,"torch_cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"kornia":metadata.version("kornia"),"scipy":metadata.version("scipy"),"open3d":metadata.version("open3d")}
    (out/"environment_manifest.json").write_text(json.dumps(env,indent=2)+"\n")
    from .loftr_tracking import build_loftr_tracks
    tracks=build_loftr_tracks(frames,axis,q,k,cfg)
    raw_obs=[o for t in tracks for o in t["observations"]]
    np.savez_compressed(out/"tracks_raw.npz",records=np.asarray(raw_obs,dtype=object))
    evidence=[classify_track(t["observations"],axis,travel,cfg["track_classification"]) for t in tracks]
    keep={e["track_id"] for e in evidence if e["unique_frame_count"]>=int(cfg["track_classification"]["min_track_observations"])}
    np.savez_compressed(out/"tracks_filtered.npz",records=np.asarray([o for t in tracks if t["track_id"] in keep for o in t["observations"]],dtype=object))
    save_csv(out/"track_motion_evidence.csv",evidence); (out/"track_labels.json").write_text(json.dumps(evidence,indent=2)+"\n")
    for label,name,color in (("static","static_tracks",[40,120,255]),("moving","moving_tracks",[255,50,40]),("unknown","unknown_tracks",[255,210,30])):
        pts=[]
        for t,e in zip(tracks,evidence):
            if e["label"]==label: pts.extend(o["point_world"] for o in t["observations"])
        write_ply(out/f"{name}.ply",np.asarray(pts).reshape(-1,3),np.tile(color,(len(pts),1)))
    keyframes=select_keyframes(frames,q,tracks,cfg); (out/"interaction_keyframes.json").write_text(json.dumps(keyframes,indent=2)+"\n")
    rows,seeds=region_votes(frames,tracks,evidence,keyframes,cfg,out); save_csv(out/"keyframe_region_votes.csv",rows)
    (out/"sam2_seed_manifest.json").write_text(json.dumps({"seeds":seeds},indent=2)+"\n")
    if not seeds: raise RuntimeError("No high-confidence drawer region seed; SAM2 propagation cannot run")
    # Build a contiguous 177..360 sequence so propagation crosses the interaction/open boundary without re-identifying the open plateau.
    video=out/"sam2_video_frames"; video.mkdir(); combined=list(range(interaction[0],open_ids[-1]+1))
    for i,source in enumerate(combined):
        cv2.imwrite(str(video/f"{i:06d}.jpg"),cv2.imread(str(rgb_paths[source])),[cv2.IMWRITE_JPEG_QUALITY,int(cfg["sam2"]["jpeg_quality"])])
    sam_manifest={"sam2_repo":inp["sam2_repo"],"model_config":inp["sam2_model_config"],"checkpoint":inp["sam2_checkpoint"],"video_dir":str(video),"output_dir":str(out/"sam2_propagation_per_seed"),"seeds":seeds}
    sam_manifest_path=out/"sam2_runtime_manifest.json"; sam_manifest_path.write_text(json.dumps(sam_manifest,indent=2)+"\n")
    subprocess.run([inp["python_environment"],"-m","tools.itaco_funrec_assignment_v3.sam2_runner","--manifest",str(sam_manifest_path)],cwd=Path(__file__).resolve().parents[2],check=True)
    seed_dirs=sorted((out/"sam2_propagation_per_seed").glob("seed_*")); static_core=load_ply_xyz(Path(inp["v2_dir"])/"static_core.ply"); static_tree=cKDTree(static_core)
    v2_static=load_ply_xyz(Path(inp["v2_dir"])/"static_v2_points.ply"); v2_drawer=load_ply_xyz(Path(inp["v2_dir"])/"drawer_v2_canonical_points.ply"); v2_unknown=load_ply_xyz(Path(inp["v2_dir"])/"unknown_v2_points.ply")
    drawer_tree=cKDTree(v2_drawer); v2_static_tree=cKDTree(v2_static)
    point_parts={"static":[],"drawer":[],"unknown":[]}; color_parts={k0:[] for k0 in point_parts}; near_counts={"static":0,"drawer":0,"unknown":0}; phase_counts={"interaction":{x:0 for x in ('static','drawer','unknown','invalid')},"open":{x:0 for x in ('static','drawer','unknown','invalid')}}; open_only_unknown=0; open_valid=0
    per_seed_coverage={d.name:{} for d in seed_dirs}; side_first=None; side_retention=[]
    for source in interaction+open_ids:
        phase="interaction" if source in interaction else "open"; combined_index=source-interaction[0]
        rgb=cv2.imread(str(rgb_paths[source])); depth=cv2.imread(str(depth_paths[source]),cv2.IMREAD_UNCHANGED).astype(np.float32)*float(cfg["validity"]["depth_scale_to_m"])
        hand=np.load(Path(inp["corrected_hand_dir"])/f"{source}.npy").squeeze().astype(bool) if phase=="interaction" else np.zeros(depth.shape,bool)
        valid,edge=mask_valid(rgb,depth,hand,cfg["validity"]); seed_masks=[]
        for d in seed_dirs:
            p=d/f"{combined_index:06d}.npy"; m=np.load(p).astype(bool) if p.exists() else np.zeros(depth.shape,bool); seed_masks.append(m); per_seed_coverage[d.name][str(source)]=int(m.sum())
        stack=np.stack(seed_masks); positive=stack.sum(0); drawer=valid&(positive>=int(cfg["sam2"]["minimum_agreeing_seeds"])); conflict=np.zeros(depth.shape,bool)
        if len(stack)>1:
            union=stack.any(0); agreement=positive/len(stack); conflict=valid&union&(agreement<float(cfg["sam2"]["conflict_min_agreement_fraction"]))
            drawer&=~conflict
        static=np.zeros(depth.shape,bool)
        points=unproject(depth,poses[source],k,valid); yy,xx=np.nonzero(valid); ds=static_tree.query(points,workers=-1)[0]
        support=ds<=float(cfg["assignment"]["static_core_projection_distance_m"]); static[yy[support],xx[support]]=True
        if phase=="interaction":
            local=interaction.index(source)
            for row in rows:
                if row["original_frame_id"]==source and row["label"]=="static":
                    for p in frames[local]["proposals"]:
                        if p["proposal_id"]==row["proposal_id"]: static|=p["mask"]&valid
        labels=four_state(valid,drawer,static,conflict)
        frame_dir=out/"per_frame_assignment"
        for name,value in (("static",LABEL_STATIC),("drawer",LABEL_DRAWER),("unknown",LABEL_UNKNOWN),("invalid",LABEL_INVALID)):
            d=frame_dir/name; d.mkdir(exist_ok=True); np.save(d/f"{source}.npy",labels==value); phase_counts[phase][name]+=int((labels==value).sum())
        world=unproject(depth,poses[source],k,valid); vy,vx=np.nonzero(valid); vl=labels[vy,vx]; colors=rgb[vy,vx][:,::-1]
        canonical=world-(travel if phase=="open" else q[interaction.index(source)])*axis
        for name,value,coords in (("static",LABEL_STATIC,world),("drawer",LABEL_DRAWER,canonical),("unknown",LABEL_UNKNOWN,world)):
            pick=vl==value; point_parts[name].append(coords[pick]); color_parts[name].append(colors[pick])
        d_s=v2_static_tree.query(world,workers=-1)[0]; d_d=drawer_tree.query(canonical,workers=-1)[0]; near=(d_s<=float(cfg["near_contact"]["support_distance_m"]))&(d_d<=float(cfg["near_contact"]["support_distance_m"]))
        for name,value in (("static",LABEL_STATIC),("drawer",LABEL_DRAWER),("unknown",LABEL_UNKNOWN)): near_counts[name]+=int((near&(vl==value)).sum())
        if phase=="open": open_valid+=int(valid.sum()); open_only_unknown+=int((labels==LABEL_UNKNOWN).sum())
        overlay=rgb.copy(); palette={LABEL_STATIC:(255,120,30),LABEL_DRAWER:(30,40,245),LABEL_UNKNOWN:(30,220,220),LABEL_INVALID:(0,0,0)}
        colored=np.zeros_like(rgb)
        for value,color in palette.items(): colored[labels==value]=color
        overlay=cv2.addWeighted(rgb,.45,colored,.55,0); cv2.imwrite(str(out/"visualization"/f"assignment_{source}.jpg"),overlay)
    for name in point_parts:
        pts=np.concatenate(point_parts[name]) if point_parts[name] else np.empty((0,3)); cols=np.concatenate(color_parts[name]) if color_parts[name] else np.empty((0,3))
        voxel=float(cfg["assignment"]["output_voxel_m"]); keys=np.floor(pts/voxel).astype(np.int64) if len(pts) else np.empty((0,3),int); _,idx=np.unique(keys,axis=0,return_index=True) if len(pts) else (None,np.asarray([],int)); pts,cols=pts[idx],cols[idx]
        filename={"static":"static_v3_points.ply","drawer":"drawer_v3_canonical_points.ply","unknown":"unknown_v3_points.ply"}[name]; write_ply(out/filename,pts,cols)
    total_near=max(sum(near_counts.values()),1)
    near_report={"automatic_definition":"points simultaneously within configured support distance of v2 static world support and v2 drawer canonical support; no manual ROI","counts":near_counts,"ratios":{k0:v/total_near for k0,v in near_counts.items()},"support_distance_m":cfg["near_contact"]["support_distance_m"],"manual_roi_used":False}
    (out/"near_contact_region_report.json").write_text(json.dumps(near_report,indent=2)+"\n"); save_csv(out/"near_contact_region_votes.csv",[r for r in rows if r["n_moving"] and r["n_static"]])
    propagation={"per_seed_coverage_px":per_seed_coverage,"interaction_to_open_continuity":bool(any(int(v.get(str(open_ids[0]),0))>0 for v in per_seed_coverage.values())),"open_only_unknown_ratio":open_only_unknown/max(open_valid,1),"conflict_policy":"multi-seed disagreement is unknown","open_plateau_reidentified":False}
    (out/"propagation_report.json").write_text(json.dumps(propagation,indent=2)+"\n")
    counts={name:sum(len(x) for x in point_parts[name]) for name in point_parts}
    v1_report=json.loads((Path(inp["v1_dir"])/"fusion_report.json").read_text()); v2_report=json.loads((Path(inp["v2_dir"])/"RUN_REPORT.json").read_text())
    comparison={"v1":v1_report.get("points",{}),"v2":v2_report["comparison"]["assignment_v2"],"v3":{"raw_observations_before_voxel":counts,"per_phase_pixels":phase_counts,"near_contact":near_report,"open_only_unknown_ratio":propagation["open_only_unknown_ratio"]},"claims":{"v2_dual_close_static_ratio":v2_report["comparison"]["dual_close_region"]["static_new_ratio"],"improvement_not_claimed_from_counts_alone":True,"human_geometry_review_required":True}}
    (out/"assignment_v1_v2_v3_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    acceptance={"output_kind":OUTPUT_KIND,"official_funrec_reproduction":False,"tsdf_ran":False,"nksr_ran":False,"mesh_ran":False,"frozen_inputs_verified":True,"conditions":{"drawer_side_interaction_motion_support":"requires diagnostic/human review","drawer_identity_open_propagation":propagation["interaction_to_open_continuity"],"cabinet_inner_wall_not_drawer":"requires human review","near_contact_not_low_excitation_static":near_report["ratios"]["unknown"]>0,"open_only_unknown":propagation["open_only_unknown_ratio"]>0,"static_cleaner_than_v1":"requires human review","drawer_not_absorbing_cabinet":"requires human review","deterministic_full_rerun":False,"human_or_annotation_support":False,"baselines_preserved":True},"ready_for_dual_tsdf":False}
    (out/"acceptance_report.json").write_text(json.dumps(acceptance,indent=2)+"\n")
    report=f"""# Assignment v3 run report\n\nThis is a **FunREC-inspired extension**, not an official FunREC reproduction. It uses periodic LK forward/backward tracks, fixed HoloLens poses/axis/q, AutoSeg-SAM2 regions, real SAM2.1 video propagation, and a conservative unknown state.\n\n- Output kind: `{OUTPUT_KIND}`\n- TSDF/NKSR/Mesh: not run\n- Tracks: {len(tracks)} raw; {len(keep)} filtered\n- Track labels: static={sum(e['label']=='static' for e in evidence)}, moving={sum(e['label']=='moving' for e in evidence)}, unknown={sum(e['label']=='unknown' for e in evidence)}\n- SAM2 seeds: {len(seeds)}\n- Ready for dual TSDF: **false** (human geometry review and deterministic full rerun remain required)\n"""
    (out/"RUN_REPORT.md").write_text(report)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*_v3_*.ply"))}; (out/"core_output_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
