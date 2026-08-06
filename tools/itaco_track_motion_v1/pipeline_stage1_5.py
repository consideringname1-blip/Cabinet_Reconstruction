"""Stage 1.5 orchestration: observation recovery and observability only."""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import cv2
import numpy as np
import yaml

from .classification import build_clusters, classify_tracks, proposal_split_merge_diagnostics, save_posteriors, write_label_ply
from .config import load_config
from .depth_quality import annotate_tracks, inspect_sensor_metadata, save_quality_npz, write_report
from .diagnostics import draw_split_merge_cases, draw_track_label_overlays, draw_validity_overlays
from .evaluation import audit_gap_associations, coverage_report, manual_evaluate, stage1_metrics, support_report, unknown_distribution
from .geometry import KnownMotion
from .hand_recovery import correct, diagnose, write_anomalies, write_quality_csv
from .proposals import build_explicit_proposals
from .reassociation import annotate_association_features, filter_tracks, reassociate
from .tracking import build_tracks, load_tracks_npz, save_tracks_npz
from .validity import build_validity


def _clean(value):
    if isinstance(value, dict): return {str(k): _clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_clean(v) for v in value]
    if isinstance(value, np.generic): return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(_clean(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_stage15(path: Path) -> dict:
    raw=yaml.safe_load(path.read_text(encoding="utf-8"))
    required={"schema_version","stage1_output_dir","stage1_config_path","hand_quality","hand_correction","gap_association","depth_quality","manual_evaluation","observability","runtime"}
    missing=sorted(required-set(raw))
    if missing: raise ValueError(f"stage 1.5 config missing sections: {missing}")
    if str(raw["schema_version"])!="1.5": raise ValueError("stage 1.5 schema_version must be 1.5")
    forbidden={"optimize_camera","estimate_axis","model_selection","free_se3","tsdf","nksr","mesh"}
    if forbidden & set(raw): raise ValueError(f"forbidden stage 1.5 sections: {sorted(forbidden&set(raw))}")
    base=path.parent.resolve()
    def resolve(value,key=""):
        if isinstance(value,dict): return {k:resolve(v,k) for k,v in value.items()}
        if isinstance(value,list): return [resolve(v,key) for v in value]
        if isinstance(value,str) and (key.endswith("_path") or key.endswith("_dir")):
            p=Path(value).expanduser(); return str((base/p).resolve() if not p.is_absolute() else p.resolve())
        return value
    config=resolve(raw); config["_config_source"]=str(path.resolve()); return config


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _poses(path: Path) -> np.ndarray:
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"): continue
        values=line.split(); rows.append(np.asarray([float(x) for x in values[2:]], dtype=np.float64).reshape(4,4))
    return np.stack(rows)


def _stage1_spatial(stage1_dir: Path, records: list[dict], label: str, grid_rows: int, grid_cols: int) -> dict:
    tracks=load_tracks_npz(stage1_dir/"tracks_filtered.npz")
    labels=json.loads((stage1_dir/"track_labels.json").read_text(encoding="utf-8"))["labels"]
    by_id={item["track_id"]:item["label"] for item in labels}
    obs=[o for t in tracks if by_id[t["track_id"]]==label for o in t["observations"]]
    image=cv2.imread(records[0]["rgb_path"],cv2.IMREAD_COLOR); h,w=image.shape[:2]
    cells={(min(grid_rows-1,int(o["pixel_uv"][1]/h*grid_rows)),min(grid_cols-1,int(o["pixel_uv"][0]/w*grid_cols))) for o in obs}
    points=np.asarray([o["point_world"] for o in obs],dtype=np.float64).reshape(-1,3)
    return {"observation_count":len(obs),"image_grid_coverage":len(cells)/(grid_rows*grid_cols),"grid_rows":grid_rows,"grid_cols":grid_cols,"world_bbox_extents_m":np.ptp(points,axis=0).tolist() if len(points) else [0,0,0]}


def _write_manifest(records: list[dict], path: Path) -> None:
    with path.open("w",encoding="utf-8") as handle:
        for record in records: handle.write(json.dumps(record,ensure_ascii=False)+"\n")


def _git_state(path: Path) -> dict:
    def command(*args):
        result=subprocess.run(["git","-C",str(path),*args],capture_output=True,text=True,check=False)
        return result.stdout.strip() if result.returncode==0 else f"unavailable: {result.stderr.strip()}"
    return {"path":str(path.resolve()),"head":command("rev-parse","HEAD"),"status_short":command("status","--short")}


def _source_hashes(package_dir: Path) -> dict:
    return {str(path.relative_to(package_dir)):_sha256(path) for path in sorted(package_dir.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}


def _environment(config: dict, stage1_config: dict) -> dict:
    package_dir=Path(__file__).resolve().parent; workspace=package_dir.parents[1]
    stage1_dir=Path(config["stage1_output_dir"]); redetection_manifest=Path(config["hand_quality"]["detector_manifest_path"])
    return {"stage":"1.5","command":sys.argv,"entry_script":str(Path(__file__).resolve()),"python":platform.python_version(),
            "python_executable":sys.executable,"config_source":config["_config_source"],"stage1_config_source":stage1_config["_config_source"],
            "source_revisions":{"workspace":_git_state(workspace),"official_itaco":_git_state(workspace/"code/reconstruction/video2articulation")},
            "source_sha256":_source_hashes(package_dir),
            "input_sha256":{"stage1_tracks_filtered":_sha256(stage1_dir/"tracks_filtered.npz"),"stage1_labels":_sha256(stage1_dir/"track_labels.json"),
                            "stage1_manifest":_sha256(stage1_dir/"frame_manifest.jsonl"),"hand_redetection_manifest":_sha256(redetection_manifest)},
            "hand_redetection":{"command":config["provenance"]["hand_redetection_command"],"manifest":str(redetection_manifest),"manifest_sha256":_sha256(redetection_manifest)},
            "fixed_inputs":{"camera_pose":True,"joint_type":True,"joint_axis":True,"joint_state_q_t":True,"stage1_classifier":True},
            "forbidden_operations":{"camera_optimization":False,"joint_estimation":False,"joint_model_selection":False,"free_se3":False,"tsdf":False,"nksr":False,"mesh":False}}


def run(config_path: Path, output_dir: Path) -> dict:
    output_dir=output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()): raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True,exist_ok=True)
    try:
        config=_load_stage15(config_path); stage1_dir=Path(config["stage1_output_dir"]); stage1_config=load_config(Path(config["stage1_config_path"]))
        records=copy.deepcopy(_records(stage1_dir/"frame_manifest.jsonl"))
        poses_original=_poses(stage1_dir/"camera_pose_original.txt"); poses_rebased=_poses(stage1_dir/"camera_pose_rebased.txt")
        if not (len(records)==len(poses_original)==len(poses_rebased)): raise ValueError("stage1 manifest/pose counts disagree")
        reference=json.loads((stage1_dir/"reference_frame_selection.json").read_text(encoding="utf-8"))
        reference_index=int(reference["processing_index"])
        if np.linalg.norm(poses_rebased[reference_index]-np.eye(4))>float(stage1_config["reference_selection"]["identity_tolerance"]): raise ValueError("stage1 rebased reference pose is not identity")
        classification_hash_before=hashlib.sha256(yaml.safe_dump(stage1_config["classification"],sort_keys=True).encode()).hexdigest()
        config["fixed_stage1_classifier"]={"source":stage1_config["classification"],"sha256":classification_hash_before,"thresholds_changed":False}
        (output_dir/"config_resolved.yaml").write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding="utf-8")
        _json(output_dir/"environment_manifest.json",_environment(config,stage1_config))
        _write_manifest(records,output_dir/"frame_manifest_input_immutable.jsonl")
        (output_dir/"camera_pose_original.txt").write_bytes((stage1_dir/"camera_pose_original.txt").read_bytes())
        (output_dir/"camera_pose_rebased.txt").write_bytes((stage1_dir/"camera_pose_rebased.txt").read_bytes())
        (output_dir/"reference_frame_selection.json").write_bytes((stage1_dir/"reference_frame_selection.json").read_bytes())

        footprints=[np.load(record["rgb_valid_path"]).astype(bool) for record in records]
        original_masks, quality_rows, anomalies=diagnose(records,footprints,Path(config["hand_quality"]["detector_manifest_path"]),config["hand_quality"])
        corrected, uncertain, corrections=correct(records,original_masks,quality_rows,config["hand_correction"],output_dir)
        write_quality_csv(output_dir/"hand_mask_quality.csv",quality_rows)
        write_anomalies(output_dir/"hand_mask_anomalies.json",anomalies,corrections,{"quality":config["hand_quality"],"correction":config["hand_correction"]})

        payloads,usage=build_validity(records,stage1_config["validity"],output_dir/"validity")
        import csv
        with (output_dir/"frame_usage.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(usage[0])); writer.writeheader(); writer.writerows(usage)
        _write_manifest(records,output_dir/"frame_manifest.jsonl")
        proposal_records,proposals=build_explicit_proposals(records,stage1_config["proposals"],output_dir/"autoseg_masks")
        _json(output_dir/"autoseg_proposals.json",{"policy":"soft association feature only","proposals":proposal_records})

        np.random.seed(int(config["runtime"]["random_seed"])); cv2.setRNGSeed(int(config["runtime"]["random_seed"])); cv2.setNumThreads(int(config["runtime"]["opencv_num_threads"]))
        raw_tracks,_=build_tracks(records,payloads,proposals,poses_rebased,stage1_config["tracking"])
        sensor=inspect_sensor_metadata(records,config["depth_quality"])
        derived=annotate_tracks(raw_tracks,payloads,records,config["depth_quality"])
        annotate_association_features(raw_tracks,payloads,np.asarray(records[0]["intrinsics"]),config["gap_association"])
        save_tracks_npz(output_dir/"tracks_raw_stage1_5.npz",raw_tracks); save_quality_npz(output_dir/"depth_quality_observations.npz",raw_tracks)
        write_report(output_dir/"depth_quality_report.json",sensor,derived)
        motion=KnownMotion(stage1_config["known_motion"],poses_original,reference_index,len(records))
        merged,associations=reassociate(raw_tracks,poses_rebased,np.asarray(records[0]["intrinsics"]),motion,config["gap_association"])
        filtered=filter_tracks(merged,stage1_config["tracking"])
        save_tracks_npz(output_dir/"tracks_reassociated_all.npz",merged); save_tracks_npz(output_dir/"tracks_reassociated.npz",filtered)
        _json(output_dir/"track_gap_associations.json",{"thresholds":config["gap_association"],"records":associations,
              "summary":{"attempted":len(associations),"accepted":sum(item["accepted"] for item in associations),"proposal_id_only_connections":0}})

        residuals,labels,representative=classify_tracks(filtered,motion,stage1_config["classification"])
        classification_hash_after=hashlib.sha256(yaml.safe_dump(stage1_config["classification"],sort_keys=True).encode()).hexdigest()
        if classification_hash_before!=classification_hash_after: raise RuntimeError("fixed stage1 classifier config changed")
        clusters=build_clusters(filtered,labels,representative,stage1_config["clustering"]); split_merge=proposal_split_merge_diagnostics(labels,clusters)
        _json(output_dir/"track_model_residuals.json",{"known_motion":motion.source,"records":residuals}); save_posteriors(output_dir/"track_posteriors.npz",labels)
        _json(output_dir/"track_labels.json",{"labels":labels,"summary":dict(Counter(item["label"] for item in labels))})
        _json(output_dir/"track_clusters.json",{"clusters":clusters,"split_merge_diagnostics":split_merge})
        for label in ("static","moving","unknown"): write_label_ply(output_dir/f"{label}_tracks.ply",filtered,labels,label)

        static=support_report("static",filtered,labels,records,motion,stage1_config["classification"],config["observability"])
        moving=support_report("moving",filtered,labels,records,motion,stage1_config["classification"],config["observability"])
        observability={"static":static,"moving":moving,"classifier_thresholds_changed":False}
        _json(output_dir/"observability_report.json",observability)
        coverage=coverage_report(filtered,labels,records,observability); _json(output_dir/"track_coverage_report.json",coverage)
        manual=manual_evaluate(Path(config["manual_evaluation"]["annotation_path"]),filtered,labels,config["manual_evaluation"])
        _json(output_dir/"manual_evaluation_report.json",manual)
        (output_dir/"manual_track_evaluation.json").write_bytes(Path(config["manual_evaluation"]["annotation_path"]).read_bytes())
        gap_audit=audit_gap_associations(Path(config["manual_evaluation"]["annotation_path"]),associations,raw_tracks,float(config["manual_evaluation"]["max_match_distance_px"])); _json(output_dir/"gap_association_manual_audit.json",gap_audit)

        s1=stage1_metrics(stage1_dir); s15={"raw_track_count":len(raw_tracks),"filtered_track_count":len(filtered),
            "mean_track_length":coverage["mean_track_length"],"mean_temporal_span_frames":coverage["mean_temporal_span_frames"],
            "label_counts":coverage["label_counts"],"unknown_reason_distribution":unknown_distribution(labels),
            "static_spatial_coverage":static["image_spatial_coverage"],"moving_spatial_coverage":moving["image_spatial_coverage"],
            "manual_accuracy":manual["accuracy_on_evaluated"],"accepted_gap_connections":sum(item["accepted"] for item in associations),
            "false_gap_connections":gap_audit["verified_false_connection_count"],"gap_connection_audit":gap_audit}
        s1["static_spatial_coverage"]=_stage1_spatial(stage1_dir,records,"static",int(config["observability"]["image_grid_rows"]),int(config["observability"]["image_grid_cols"])); s1["moving_spatial_coverage"]=_stage1_spatial(stage1_dir,records,"moving",int(config["observability"]["image_grid_rows"]),int(config["observability"]["image_grid_cols"]))
        s1["manual_accuracy"]=None
        unknown_reduced=s15["label_counts"].get("unknown",0)<s1["label_counts"].get("unknown",0)
        comparison={"stage1":s1,"stage1_5":s15,"unknown_count_change":s15["label_counts"].get("unknown",0)-s1["label_counts"].get("unknown",0),
                    "unknown_fraction_change":s15["label_counts"].get("unknown",0)/max(s15["filtered_track_count"],1)-s1["label_counts"].get("unknown",0)/max(s1["filtered_track_count"],1),
                    "unknown_count_reduced":unknown_reduced,"classifier_thresholds_identical":classification_hash_before==classification_hash_after}
        _json(output_dir/"stage1_vs_stage1_5_comparison.json",comparison)

        visualization=output_dir/"visualization"; visualization.mkdir(exist_ok=True)
        draw_validity_overlays(records,payloads,visualization,stage1_config["visualization"])
        draw_track_label_overlays(records,filtered,labels,visualization,stage1_config["visualization"])
        draw_split_merge_cases(records,filtered,labels,split_merge,visualization,stage1_config["visualization"])
        hand_observations=0
        for track in raw_tracks:
            for obs in track["observations"]:
                u,v=(int(round(x)) for x in obs["pixel_uv"]); index=int(obs["processing_index"])
                hand_observations+=int(payloads[index]["hand"][v,u])
        def information_ok(item):
            info=item["residual_information_matrix"]
            return info["rank"]>=int(config["observability"]["min_information_rank"]) and info["condition_number"] is not None and info["condition_number"]<=float(config["observability"]["max_information_condition_number"])
        entry={
            "hand_observation_count_zero":hand_observations==0,
            "hand_mask_anomaly_coverage_reduced":sum(row["corrected_hand_area_ratio"] for row in quality_rows if row["hand_mask_temporal_anomaly"]) < sum(row["hand_area_ratio"] for row in quality_rows if row["hand_mask_temporal_anomaly"]),
            "gap_reassociation_not_known_to_increase_errors":gap_audit["sufficient_to_claim_no_error_increase"],
            "static_support_nondegenerate":not static["static_support_degenerate"],"moving_support_nondegenerate":not moving["moving_support_degenerate"],
            "information_matrices_nondegenerate":information_ok(static) and information_ok(moving),
            "bootstrap_stable":static["bootstrap"]["stable_fraction"]>=config["observability"]["min_bootstrap_stable_fraction"] and moving["bootstrap"]["stable_fraction"]>=config["observability"]["min_bootstrap_stable_fraction"],
            "manual_no_obvious_leakage":manual["accuracy_on_evaluated"] is not None and manual["accuracy_on_evaluated"]>=config["manual_evaluation"]["min_accuracy_for_stage2"],
            "unknown_reduced_from_observation_recovery_not_threshold_relaxation":unknown_reduced and classification_hash_before==classification_hash_after,
            "baseline_and_stage1_preserved":all((stage1_dir/name).exists() for name in ("stage1_summary.json","tracks_filtered.npz","track_labels.json")),
        }
        entry["ready_for_stage2"]=all(entry.values())
        summary={"stage":"1.5","output_dir":str(output_dir),"reference_frame_id":records[reference_index]["original_frame_id"],
                 "raw_track_count":len(raw_tracks),"filtered_track_count":len(filtered),"label_counts":dict(Counter(i["label"] for i in labels)),
                 "hand_anomaly_count":len(anomalies),"accepted_gap_connections":sum(i["accepted"] for i in associations),
                 "entry_stage2":entry,"forbidden_operations":{"camera_optimization":False,"joint_estimation":False,"joint_model_selection":False,"free_se3":False,"tsdf":False,"nksr":False,"mesh":False}}
        _json(output_dir/"stage1_5_summary.json",summary); _json(output_dir/"failure_reasons.json",{"failures":[],"unknown_tracks":{str(i["track_id"]):i["unknown_reasons"] for i in labels if i["unknown_reasons"]}})
        return summary
    except Exception as exc:
        _json(output_dir/"failure_reasons.json",{"failures":[{"stage":"stage1_5","code":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}]})
        raise
