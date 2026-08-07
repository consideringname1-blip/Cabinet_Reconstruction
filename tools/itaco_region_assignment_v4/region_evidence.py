"""Deterministic surface-region sampling, evidence aggregation and classification."""
from __future__ import annotations

import numpy as np

from .projective_models import CONTRADICTION, SUPPORTED, evaluate_model, unproject_pixels


def eligible_target_indices(source_index: int, q: np.ndarray, total_travel: float, cfg: dict) -> list[int]:
    minimum = float(cfg["minimum_pair_q_span_fraction"]) * float(total_travel)
    eligible = [i for i in range(len(q)) if i != source_index and abs(float(q[i] - q[source_index])) >= minimum]
    maximum = int(cfg.get("maximum_target_frames", 0))
    if maximum and len(eligible) > maximum:
        order = sorted(eligible, key=lambda i: (float(q[i]), i))
        picks = np.linspace(0, len(order) - 1, maximum).round().astype(int)
        eligible = [order[i] for i in sorted(set(picks.tolist()))]
    return eligible


def deterministic_sample(mask: np.ndarray, maximum_points: int) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return np.empty((0, 2), np.int64)
    if len(xx) <= maximum_points:
        return np.column_stack((xx, yy))
    y0, y1, x0, x1 = yy.min(), yy.max(), xx.min(), xx.max()
    area = max((y1 - y0 + 1) * (x1 - x0 + 1), 1)
    stride = max(1, int(np.floor(np.sqrt(area / maximum_points))))
    chosen = np.flatnonzero(((xx - x0) % stride == 0) & ((yy - y0) % stride == 0))
    if len(chosen) > maximum_points:
        chosen = chosen[np.linspace(0, len(chosen) - 1, maximum_points).round().astype(int)]
    elif len(chosen) < maximum_points:
        remaining = np.setdiff1d(np.arange(len(xx)), chosen, assume_unique=True)
        need = min(maximum_points - len(chosen), len(remaining))
        if need:
            chosen = np.sort(np.r_[chosen, remaining[np.linspace(0, len(remaining) - 1, need).round().astype(int)]])
    return np.column_stack((xx[chosen], yy[chosen]))


def _aggregate(rows: list[dict], prefix: str, cfg: dict) -> dict:
    useful = [r for r in rows if r[f"{prefix}_testable_count"] >= int(cfg["minimum_testable_points_per_target"])]
    support = np.asarray([r[f"{prefix}_support_ratio"] for r in useful], float)
    contradiction = np.asarray([r[f"{prefix}_contradiction_ratio"] for r in useful], float)
    if not len(useful):
        return {"support_ratio_median": 0.0, "support_ratio_p25": 0.0,
                "contradiction_ratio_median": 1.0, "supported_frames": 0,
                "testable_points": 0, "usable_target_frames": 0, "score": -1.0}
    frame_good = (support >= float(cfg["frame_support_ratio_threshold"])) & (contradiction <= float(cfg["maximum_contradiction_ratio"]))
    return {"support_ratio_median": float(np.median(support)), "support_ratio_p25": float(np.percentile(support, 25)),
            "contradiction_ratio_median": float(np.median(contradiction)), "supported_frames": int(frame_good.sum()),
            "testable_points": int(sum(r[f"{prefix}_testable_count"] for r in useful)),
            "usable_target_frames": len(useful), "score": float(np.median(support - contradiction))}


def classify_region(evidence: dict, cfg: dict) -> tuple[str, str, bool]:
    if evidence["region_valid_pixel_count"] < int(cfg["minimum_valid_region_pixels"]) or evidence["region_sampled_point_count"] < int(cfg["minimum_sampled_points"]):
        return "invalid", "invalid_insufficient_depth", False
    if evidence["target_frame_count"] < int(cfg["minimum_target_frames"]):
        return "unknown", "unknown_insufficient_evidence", False
    static, drawer = evidence["static"], evidence["drawer"]
    if min(static["testable_points"], drawer["testable_points"]) < int(cfg["minimum_total_testable_points"]):
        return "unknown", "unknown_insufficient_evidence", False
    if evidence["drawer_preferred_fraction"] >= float(cfg["mixed_surface_min_fraction"]) and evidence["static_preferred_fraction"] >= float(cfg["mixed_surface_min_fraction"]):
        return "unknown", "unknown_mixed_surface", False
    def strong(model: dict, preferred: float) -> bool:
        return (model["support_ratio_median"] >= float(cfg["minimum_support_ratio_median"])
                and model["support_ratio_p25"] >= float(cfg["minimum_support_ratio_p25"])
                and model["contradiction_ratio_median"] <= float(cfg["maximum_contradiction_ratio"])
                and model["supported_frames"] >= int(cfg["minimum_supported_frames"])
                and preferred >= float(cfg["minimum_point_preferred_fraction"]))
    static_strong = strong(static, evidence["static_preferred_fraction"])
    drawer_strong = strong(drawer, evidence["drawer_preferred_fraction"])
    margin = float(drawer["score"] - static["score"])
    seed_conflict = evidence["seed_overlap_ratio"] >= float(cfg["seed_conflict_overlap_ratio"]) and not (drawer_strong and margin >= float(cfg["minimum_model_score_margin"]))
    if seed_conflict:
        return "unknown", "unknown_seed_geometry_conflict", False
    if drawer_strong and margin >= float(cfg["minimum_model_score_margin"]):
        return "drawer", "drawer_region_geometry_with_seed" if evidence["seed_overlap_ratio"] > 0 else "drawer_region_geometry_revealed", True
    if static_strong and -margin >= float(cfg["minimum_model_score_margin"]):
        return "static", "static_region_geometry", False
    return "unknown", "unknown_ambiguous", False


def evaluate_region(source_index: int, source_frame: dict, region_mask: np.ndarray,
                    frames: list[dict], q: np.ndarray, axis: np.ndarray, intrinsic: np.ndarray,
                    total_travel: float, cfg: dict, seed_mask: np.ndarray) -> dict:
    valid_region = np.asarray(region_mask, bool) & source_frame["valid"]
    uv = deterministic_sample(valid_region, int(cfg["maximum_sampled_points_per_region"]))
    result = {"region_valid_pixel_count": int(valid_region.sum()), "region_sampled_point_count": int(len(uv)),
              "seed_overlap_ratio": float((valid_region & seed_mask).sum() / max(valid_region.sum(), 1))}
    if not len(uv):
        result.update({"target_frame_count": 0, "q_span_used_m": 0.0, "per_target": [],
                       "static": _aggregate([], "static", cfg), "drawer": _aggregate([], "drawer", cfg),
                       "static_preferred_fraction": 0.0, "drawer_preferred_fraction": 0.0, "ambiguous_fraction": 1.0,
                       "sample_uv": uv, "point_preference": np.empty(0, np.int8)})
        result["label"], result["reason"], result["high_confidence_drawer"] = classify_region(result, cfg)
        return result
    depth = source_frame["depth"][uv[:, 1], uv[:, 0]]
    world = unproject_pixels(uv, depth, source_frame["pose"], intrinsic)
    targets = eligible_target_indices(source_index, q, total_travel, cfg)
    per_target=[]; point_support={m:np.zeros(len(uv),int) for m in ("static","drawer")}; point_contra={m:np.zeros(len(uv),int) for m in ("static","drawer")}
    for target_index in targets:
        row={"source_frame_id":int(source_frame["source"]),"target_frame_id":int(frames[target_index]["source"]),"q_delta_m":float(q[target_index]-q[source_index])}
        for model in ("static","drawer"):
            ev=evaluate_model(world,float(q[source_index]),frames[target_index],float(q[target_index]),axis,intrinsic,model,cfg)
            for key in ("support_ratio","contradiction_ratio","testable_count","observable_fraction"):
                row[f"{model}_{key}"]=ev[key]
            row[f"{model}_counts"]=ev["counts"]
            point_support[model] += ev["status"] == SUPPORTED; point_contra[model] += ev["status"] == CONTRADICTION
        per_target.append(row)
    result["per_target"]=per_target; result["target_frame_count"]=len(targets)
    result["q_span_used_m"]=float(max((abs(q[i]-q[source_index]) for i in targets),default=0.0))
    result["static"]=_aggregate(per_target,"static",cfg); result["drawer"]=_aggregate(per_target,"drawer",cfg)
    static_testable=point_support["static"]+point_contra["static"]; drawer_testable=point_support["drawer"]+point_contra["drawer"]
    static_score=np.divide(point_support["static"]-point_contra["static"],static_testable,out=np.zeros(len(uv),float),where=static_testable>0)
    drawer_score=np.divide(point_support["drawer"]-point_contra["drawer"],drawer_testable,out=np.zeros(len(uv),float),where=drawer_testable>0)
    testable=(static_testable>=int(cfg["minimum_point_testable_observations"]))&(drawer_testable>=int(cfg["minimum_point_testable_observations"]))
    margin=float(cfg["point_preference_score_margin"]); pref=np.zeros(len(uv),np.int8); pref[testable&(drawer_score>=static_score+margin)]=1; pref[testable&(static_score>=drawer_score+margin)]=-1
    result["drawer_preferred_fraction"]=float((pref==1).mean()); result["static_preferred_fraction"]=float((pref==-1).mean()); result["ambiguous_fraction"]=float((pref==0).mean())
    result["sample_uv"]=uv; result["point_preference"]=pref
    result["label"],result["reason"],result["high_confidence_drawer"]=classify_region(result,cfg)
    return result
