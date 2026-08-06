"""Gap-tolerant track association under fixed static and known-moving models."""

from __future__ import annotations

import math
import numpy as np

from .geometry import KnownMotion
import cv2


def _project(point_reference: np.ndarray, pose_reference_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray | None:
    point_camera = np.linalg.inv(pose_reference_camera)[:3, :3] @ (point_reference - pose_reference_camera[:3, 3])
    if point_camera[2] <= 1e-8 or not np.isfinite(point_camera).all(): return None
    return np.asarray([intrinsic[0,0] * point_camera[0] / point_camera[2] + intrinsic[0,2], intrinsic[1,1] * point_camera[1] / point_camera[2] + intrinsic[1,2]])


def _cosine_rgb(a: dict, b: dict) -> float:
    x = np.asarray(a.get("appearance_descriptor",a["rgb"]), dtype=np.float64)
    y = np.asarray(b.get("appearance_descriptor",b["rgb"]), dtype=np.float64)
    x -= x.mean(); y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0)) if denom > 1e-9 else float(np.exp(-np.linalg.norm(x-y)))
def annotate_association_features(tracks: list[dict], payloads: list[dict], intrinsic: np.ndarray, config: dict) -> None:
    radius=int(config["appearance_patch_radius_px"])
    for track in tracks:
        for obs in track["observations"]:
            index=int(obs["processing_index"]); payload=payloads[index]; u,v=(int(round(x)) for x in obs["pixel_uv"])
            image=payload["rgb"]; y0,y1=max(0,v-radius),min(image.shape[0],v+radius+1); x0,x1=max(0,u-radius),min(image.shape[1],u+radius+1)
            patch=cv2.resize(image[y0:y1,x0:x1],(3,3),interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)/255.0
            obs["appearance_descriptor"]=patch.tolist()
            depth=payload["depth"]
            if 0<u<depth.shape[1]-1 and 0<v<depth.shape[0]-1 and all(depth[y,x]>0 for y,x in ((v,u-1),(v,u+1),(v-1,u),(v+1,u))):
                def point(x,y):
                    z=float(depth[y,x]); return np.asarray([(x-intrinsic[0,2])*z/intrinsic[0,0],(y-intrinsic[1,2])*z/intrinsic[1,1],z])
                normal=np.cross(point(u+1,v)-point(u-1,v),point(u,v+1)-point(u,v-1)); norm=np.linalg.norm(normal)
                obs["surface_normal_camera"]=(normal/norm).tolist() if norm>1e-9 else None
            else: obs["surface_normal_camera"]=None




def _candidate(before: dict, after: dict, poses: np.ndarray, intrinsic: np.ndarray, motion: KnownMotion, config: dict) -> dict:
    end, start = before["observations"][-1], after["observations"][0]
    i, j = int(end["processing_index"]), int(start["processing_index"])
    p_end, p_start = np.asarray(end["point_world"]), np.asarray(start["point_world"])
    static_error = float(np.linalg.norm(p_end - p_start))
    canonical = motion.canonicalize(p_end[None], np.asarray([i]))
    moving_prediction = motion.decanonicalize(canonical, np.asarray([j]))[0]
    moving_error = float(np.linalg.norm(moving_prediction - p_start))
    uv_static = _project(p_end, poses[j], intrinsic)
    uv_moving = _project(moving_prediction, poses[j], intrinsic)
    target_uv = np.asarray(start["pixel_uv"])
    static_reprojection = float(np.linalg.norm(uv_static-target_uv)) if uv_static is not None else float("inf")
    moving_reprojection = float(np.linalg.norm(uv_moving-target_uv)) if uv_moving is not None else float("inf")
    appearance = _cosine_rgb(end, start)
    depth_consistency = float(abs(end["depth"] - start["depth"]))
    same_proposal = bool(end.get("proposal_id") and end.get("proposal_id") == start.get("proposal_id"))
    na,nb=end.get("surface_normal_camera"),start.get("surface_normal_camera")
    normal_similarity=float(np.dot(na,nb)) if na is not None and nb is not None else None
    normal_factor=max((normal_similarity+1.0)/2.0,0.0) if normal_similarity is not None else float(config["missing_normal_factor"])
    static_ok = static_error <= float(config["max_static_3d_error_m"]) and static_reprojection <= float(config["max_static_reprojection_error_px"])
    moving_ok = moving_error <= float(config["max_moving_3d_error_m"]) and moving_reprojection <= float(config["max_moving_reprojection_error_px"])
    model_error = min(static_error / float(config["max_static_3d_error_m"]), moving_error / float(config["max_moving_3d_error_m"]))
    reproj_error = min(static_reprojection / float(config["max_static_reprojection_error_px"]), moving_reprojection / float(config["max_moving_reprojection_error_px"]))
    confidence = float(np.exp(-model_error) * np.exp(-reproj_error) * max((appearance + 1.0) / 2.0, 0.0) * normal_factor * np.exp(-depth_consistency / float(config["depth_consistency_scale_m"])))
    if same_proposal: confidence = min(1.0, confidence + float(config["proposal_soft_bonus"]))
    reasons = []
    if not (static_ok or moving_ok): reasons.append("neither_fixed_model_explains_connection")
    if appearance < float(config["min_appearance_similarity"]): reasons.append("appearance_inconsistent")
    if depth_consistency > float(config["max_depth_difference_m"]): reasons.append("depth_inconsistent")
    if confidence < float(config["min_association_confidence"]): reasons.append("association_confidence_too_low")
    return {
        "track_id_before": int(before["track_id"]), "track_id_after": int(after["track_id"]), "gap_length": j-i-1,
        "static_3d_error_m": static_error, "moving_3d_error_m": moving_error,
        "static_reprojection_error": static_reprojection, "moving_reprojection_error": moving_reprojection,
        "appearance_similarity": appearance, "depth_consistency": depth_consistency,
        "normal_consistency": normal_similarity, "proposal_soft_agreement": same_proposal,
        "association_confidence": confidence, "accepted": not reasons, "rejection_reason": ";".join(reasons),
        "explaining_models": [name for name, ok in (("static", static_ok), ("known_moving", moving_ok)) if ok],
    }


def reassociate(tracks: list[dict], poses: np.ndarray, intrinsic: np.ndarray, motion: KnownMotion, config: dict) -> tuple[list[dict], list[dict]]:
    starts: dict[int, list[dict]] = {}
    for track in tracks:
        starts.setdefault(int(track["observations"][0]["processing_index"]), []).append(track)
    candidates = []
    max_gap = int(config["max_gap_frames"])
    for before in tracks:
        end_index = int(before["observations"][-1]["processing_index"])
        for gap in range(1, max_gap + 1):
            for after in starts.get(end_index + gap + 1, []):
                candidates.append(_candidate(before, after, poses, intrinsic, motion, config))
    candidates.sort(key=lambda item: item["association_confidence"], reverse=True)
    successor, predecessor = {}, {}
    for item in candidates:
        if not item["accepted"]: continue
        before, after = item["track_id_before"], item["track_id_after"]
        if before in successor or after in predecessor:
            item["accepted"] = False; item["rejection_reason"] = "conflict_with_higher_confidence_connection"; continue
        successor[before] = after; predecessor[after] = before
    by_id = {int(track["track_id"]): track for track in tracks}
    roots = sorted(track_id for track_id in by_id if track_id not in predecessor)
    merged = []
    for new_id, root in enumerate(roots):
        chain, current = [], root
        while True:
            chain.append(current)
            if current not in successor: break
            current = successor[current]
        observations = []
        for old_id in chain: observations.extend(by_id[old_id]["observations"])
        observations.sort(key=lambda obs: obs["processing_index"])
        for obs in observations: obs["track_id"] = new_id
        merged.append({
            "track_id": new_id, "observations": observations, "source_track_ids": chain,
            "attempted_transitions": sum(by_id[x].get("attempted_transitions", 0) for x in chain) + len(chain)-1,
            "occlusion_failures": sum(by_id[x].get("occlusion_failures", 0) for x in chain),
            "boundary_failures": sum(by_id[x].get("boundary_failures", 0) for x in chain),
            "fb_failures": sum(by_id[x].get("fb_failures", 0) for x in chain),
            "termination_reason": by_id[chain[-1]].get("termination_reason"),
        })
    return merged, candidates


def filter_tracks(tracks: list[dict], tracking_config: dict) -> list[dict]:
    result = []
    for track in tracks:
        obs = track["observations"]
        span = int(obs[-1]["processing_index"] - obs[0]["processing_index"] + 1)
        if len(obs) >= int(tracking_config["min_observations"]) and span >= int(tracking_config["min_temporal_span_frames"]):
            track["temporal_span_frames"] = span; track["coverage"] = len(obs) / span; result.append(track)
    return result


def synthetic_false_connection_count(records: list[dict]) -> int | None:
    # Real recordings have no algorithmic segment identity ground truth; manual evaluation is reported separately.
    return None
