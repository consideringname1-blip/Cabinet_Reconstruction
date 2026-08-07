"""Frozen input validation and registered RGB-D frame loading for Assignment v4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def load_poses(path: Path) -> np.ndarray:
    lines=[line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines)%5: raise ValueError(f"invalid odometry blocks: {path}")
    return np.stack([np.asarray([[float(value) for value in row.split()] for row in lines[i+1:i+5]]) for i in range(0,len(lines),5)])


def load_phase(path: Path) -> list[int]:
    ids=[int(value) for value in json.loads(path.read_text())["source_indices"]]
    if not ids or any(b<=a for a,b in zip(ids,ids[1:])): raise ValueError(f"phase is not nonempty forward order: {path}")
    return ids


def rgb_footprint(rgb: np.ndarray, erosion_pixels: int) -> np.ndarray:
    mask=np.any(rgb!=0,axis=2).astype(np.uint8)
    if erosion_pixels:
        size=2*int(erosion_pixels)+1; mask=cv2.erode(mask,np.ones((size,size),np.uint8))
    return mask.astype(bool)


def validity_masks(rgb: np.ndarray, depth: np.ndarray, hand: np.ndarray, cfg: dict) -> dict:
    footprint=rgb_footprint(rgb,int(cfg["rgb_erosion_pixels"]))
    depth_valid=np.isfinite(depth)&(depth>=float(cfg["depth_min_m"]))&(depth<=float(cfg["depth_max_m"]))
    threshold=float(cfg["depth_discontinuity_threshold_m"]); discontinuity=np.zeros_like(depth_valid)
    horizontal=depth_valid[:,1:]&depth_valid[:,:-1]&(np.abs(depth[:,1:]-depth[:,:-1])>threshold)
    vertical=depth_valid[1:]&depth_valid[:-1]&(np.abs(depth[1:]-depth[:-1])>threshold)
    discontinuity[:,1:]|=horizontal; discontinuity[:,:-1]|=horizontal; discontinuity[1:]|=vertical; discontinuity[:-1]|=vertical
    edge_distance=distance_transform_edt(~discontinuity)
    valid=footprint&depth_valid&~np.asarray(hand,bool)&(edge_distance>float(cfg["depth_boundary_erosion_pixels"]))
    return {"rgb_footprint":footprint,"depth_valid":depth_valid,"depth_edge_distance":edge_distance,"valid":valid}


def load_and_validate(cfg: dict) -> dict:
    inp=cfg["inputs"]; raw=Path(inp["raw_root"]); root=raw/"pinhole_projection"
    required=[Path(value) for key,value in inp.items() if key.endswith(("_path","_manifest","_dir")) or key in {"raw_root","sam2_repo","sam2_checkpoint"}]
    missing=[str(path) for path in required if not path.exists()]
    if missing: raise FileNotFoundError(json.dumps({"code":"missing_frozen_inputs","paths":missing},indent=2))
    manifest=json.loads(Path(inp["reproduction_manifest"]).read_text())
    phases={name:load_phase(Path(inp[f"{name}_manifest"])) for name in ("closed","interaction","open")}
    recorded=manifest["frame_phases"]
    expected={"closed":[phases["closed"][0],phases["closed"][-1]],"interaction":[phases["interaction"][0],phases["interaction"][-1]],"open":[phases["open"][0],phases["open"][-1]]}
    actual={"closed":recorded["closed_original_frame_ids"],"interaction":recorded["interaction_original_frame_ids"],"open":recorded["open_original_frame_ids"]}
    if expected!=actual or recorded["order"]!="forward in all phases": raise RuntimeError(f"frozen frame policy conflict: expected={expected}, manifest={actual}")
    axis=np.load(inp["axis_path"]).astype(float); axis/=np.linalg.norm(axis); manifest_axis=np.asarray(manifest["fixed_quantities"]["opening_axis_world"],float)
    q=np.load(inp["q_path"]).astype(float); travel=float(q.max())
    if manifest["fixed_quantities"]["joint_type"]!="prismatic" or not np.array_equal(axis,manifest_axis): raise RuntimeError("frozen joint type/axis conflict")
    if len(q)!=len(phases["interaction"]) or np.any(np.diff(q)<0) or abs(travel-float(manifest["q_recomputation"]["monotonic_range_m"]))>1e-12: raise RuntimeError("frozen q_t conflict")
    poses=load_poses(root/"odometry.log"); pose_copy=np.load(inp["interaction_pose_path"])
    pose_difference=float(np.max(np.abs(pose_copy-poses[np.asarray(phases["interaction"])])))
    if pose_difference>float(cfg["validity"]["pose_equality_tolerance"]): raise RuntimeError(f"frozen pose conflict: {pose_difference}")
    rgb_paths=[root/line.split(maxsplit=1)[1].replace("\\","/") for line in (root/"rgb.txt").read_text().splitlines() if line.strip()]
    depth_paths=[root/line.split(maxsplit=1)[1].replace("\\","/") for line in (root/"depth.txt").read_text().splitlines() if line.strip()]
    if len(rgb_paths)!=len(depth_paths) or len(poses)!=len(rgb_paths): raise RuntimeError("RGB/depth/pose count conflict")
    fx,fy,cx,cy=np.loadtxt(root/"calibration.txt").reshape(-1)[:4]; intrinsic=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1.]],float)
    moving=np.load(inp["moving_labels_path"])["a"]
    if moving.shape!=(len(q),288,320): raise RuntimeError(f"repaired moving label shape conflict: {moving.shape}")
    frames={}
    for phase,sources in phases.items():
        collection=[]
        for local,source in enumerate(sources):
            rgb=cv2.imread(str(rgb_paths[source])); raw_depth=cv2.imread(str(depth_paths[source]),cv2.IMREAD_UNCHANGED)
            if rgb is None or raw_depth is None: raise FileNotFoundError(f"missing RGB-D source frame {source}")
            depth=raw_depth.astype(np.float32)*float(cfg["validity"]["depth_scale_to_m"])
            hand=np.load(Path(inp["corrected_hand_dir"])/f"{source}.npy").squeeze().astype(bool) if phase=="interaction" else np.zeros(depth.shape,bool)
            masks=validity_masks(rgb,depth,hand,cfg["validity"])
            state=0.0 if phase=="closed" else travel if phase=="open" else float(q[local])
            collection.append({"phase":phase,"local_index":local,"source":source,"q":state,"rgb":rgb,"depth":depth,"hand":hand,"pose":poses[source],**masks})
        frames[phase]=collection
    hashes={name:{"path":inp[name],"sha256":sha256(Path(inp[name]))} for name in ("axis_path","q_path","moving_labels_path","interaction_pose_path","sam2_checkpoint")}
    expected_checkpoint_hash=str(cfg["audit"]["sam2_checkpoint_sha256"])
    if hashes["sam2_checkpoint"]["sha256"]!=expected_checkpoint_hash:
        raise RuntimeError("frozen SAM2 checkpoint hash conflict")
    audit={"verified":True,"manifest":inp["reproduction_manifest"],"joint_type":"prismatic","axis_world":axis.tolist(),"travel_m":travel,
           "frame_ranges":{name:[values[0],values[-1],len(values)] for name,values in phases.items()},"frame_order":"forward",
           "pose_max_abs_difference":pose_difference,"input_hashes":hashes,"camera_optimization_ran":False,"axis_optimization_ran":False,
           "q_t_optimization_ran":False,"moving_map_optimization_ran":False,"calibration_optimization_ran":False}
    return {"frames":frames,"phases":phases,"axis":axis,"q":q,"travel":travel,"poses":poses,"intrinsic":intrinsic,
            "moving_labels":moving,"rgb_paths":rgb_paths,"depth_paths":depth_paths,"manifest":manifest,"audit":audit}
