"""Assignment v4 end-to-end region-level ownership; no reconstruction code."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
import yaml

from . import LABEL_DRAWER,LABEL_INVALID,LABEL_STATIC,LABEL_UNKNOWN,OUTPUT_KIND
from .diagnostics import (assignment_overlay,json_ready_region,make_contact_sheet,projection_panel,proposal_overlay,
                          render_representative_frames,save_csv,sha_files,voxel_reduce,write_ply)
from .frame_data import load_and_validate,validate_registered_depth_scale
from .projective_models import supported_template_pixels,unproject_pixels
from .proposals import attach_interaction_proposals
from .propagation import load_seed_masks,run_sam2
from .region_assignment import merge_seed_propagations,resolve_interaction,resolve_positive_support
from .region_evidence import evaluate_region


def load_cloud(path: Path) -> np.ndarray:
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def save_assignment(output: Path,phase: str,source: int,labels: np.ndarray) -> None:
    for name,value in (("static",LABEL_STATIC),("drawer",LABEL_DRAWER),("unknown",LABEL_UNKNOWN),("invalid",LABEL_INVALID)):
        directory=output/"per_frame_assignment"/name; directory.mkdir(parents=True,exist_ok=True); np.save(directory/f"{source}.npy",labels==value)


def region_row(item: dict) -> dict:
    return {"original_frame_id":item["original_frame_id"],"proposal_id":item["proposal_id"],"source_layer":item["source_layer"],
            "label":item["label"],"reason":item["reason"],"valid_pixels":item["region_valid_pixel_count"],"sampled_points":item["region_sampled_point_count"],
            "target_frame_count":item["target_frame_count"],"q_span_used_m":item["q_span_used_m"],"seed_overlap_ratio":item["seed_overlap_ratio"],
            "static_support_median":item["static"]["support_ratio_median"],"static_support_p25":item["static"]["support_ratio_p25"],
            "static_contradiction_median":item["static"]["contradiction_ratio_median"],"static_supported_frames":item["static"]["supported_frames"],
            "drawer_support_median":item["drawer"]["support_ratio_median"],"drawer_support_p25":item["drawer"]["support_ratio_p25"],
            "drawer_contradiction_median":item["drawer"]["contradiction_ratio_median"],"drawer_supported_frames":item["drawer"]["supported_frames"],
            "evidence_margin":item["drawer"]["score"]-item["static"]["score"],"drawer_preferred_fraction":item["drawer_preferred_fraction"],
            "static_preferred_fraction":item["static_preferred_fraction"],"ambiguous_fraction":item["ambiguous_fraction"],
            "high_confidence_drawer":item["high_confidence_drawer"]}


def region_preview(frame: dict,proposal: dict,item: dict) -> np.ndarray:
    image=frame["rgb"].copy(); color={"static":(255,120,30),"drawer":(30,40,245),"unknown":(30,220,220),"invalid":(60,60,60)}[item["label"]]
    mask=proposal["mask"]; painted=np.full_like(image,color); image[mask]=cv2.addWeighted(image,.35,painted,.65,0)[mask]
    text=f"{frame['source']} L{proposal['source_layer']} {item['label']} {item['reason']}"; cv2.rectangle(image,(0,0),(image.shape[1],20),(0,0,0),-1); cv2.putText(image,text,(3,14),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1,cv2.LINE_AA)
    return image


def quantiles(values: list[float]) -> dict | None:
    a=np.asarray(values,float); a=a[np.isfinite(a)]
    if not len(a): return None
    return {"p10":float(np.percentile(a,10)),"p25":float(np.percentile(a,25)),"p50":float(np.median(a)),"p75":float(np.percentile(a,75)),"p90":float(np.percentile(a,90))}


def fate(points: np.ndarray,current: dict[str,np.ndarray],threshold: float) -> dict:
    if not len(points): return {"static":0,"drawer":0,"unknown":0,"invalid_or_unmatched":0}
    names=("static","drawer","unknown"); distances=[]
    for name in names:
        cloud=current[name]; distances.append(cKDTree(cloud).query(points,workers=-1)[0] if len(cloud) else np.full(len(points),np.inf))
    stack=np.stack(distances); owner=np.argmin(stack,axis=0); minimum=np.min(stack,axis=0)
    return {**{name:int(((owner==index)&(minimum<=threshold)).sum()) for index,name in enumerate(names)},"invalid_or_unmatched":int((minimum>threshold).sum())}


def main(config_path: Path) -> None:
    cfg=yaml.safe_load(config_path.read_text()); output=Path(cfg["inputs"]["output_dir"])
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"non-overwrite output exists: {output}")
    try:
        scale_gate=validate_registered_depth_scale(cfg)
    except Exception as error:
        scale_gate={"gate":"registered_depth_scale_consistency","passed":False,"status":"failed","failure_message":"registered depth physical-unit mismatch","failure_codes":["gate_execution_error"],"error_type":type(error).__name__,"error":str(error),"evaluated_before_assignment_frame_loading":True}
    output.mkdir(parents=True); (output/"visualization").mkdir(); (output/"config_resolved.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    command={"argv":sys.argv,"cwd":str(Path.cwd()),"entry_script":"tools/fuse_hololens_articulation_assignment_v4.py","parent_commit":cfg["audit"]["parent_commit"]}
    (output/"execution_manifest.json").write_text(json.dumps(command,indent=2)+"\n")
    (output/"scale_consistency_gate_report.json").write_text(json.dumps(scale_gate,indent=2)+"\n")
    if not scale_gate["passed"]: raise RuntimeError(f"registered depth physical-unit mismatch: {scale_gate['failure_codes']}")
    context=load_and_validate(cfg); (output/"frozen_input_audit.json").write_text(json.dumps(context["audit"],indent=2)+"\n")
    dependency_audit={"formal_primary_ownership":"AutoSeg/SAM surface regions with fixed-model projective RGB-D evidence",
                      "point_tracking_primary":False,"loftr_called":False,"lk_called":False,"tapip3d_called":False,
                      "reason":["cabinet surfaces are low-texture","long-lived point identity is unreliable","v3 depth-projective filtering reduced geometric outliers but reliable tracks remained sparse","v4 evaluates motion consistency at region/surface level"],
                      "sam2_role":"identity propagation only after direct geometry acceptance","official_baseline_modified":False}
    (output/"method_dependency_audit.json").write_text(json.dumps(dependency_audit,indent=2)+"\n")
    frames=context["frames"]["interaction"]; proposal_audit=attach_interaction_proposals(frames,cfg)
    (output/"autoseg_proposals.json").write_text(json.dumps(proposal_audit,indent=2)+"\n")
    frame_evidence={}; rows=[]; drawer_candidates=[]; all_items=[]
    for index,frame in enumerate(frames):
        evidence=[]; seed=context["moving_labels"][index]==2
        for proposal in frame["proposals"]:
            item=evaluate_region(index,frame,proposal["eroded_mask"],frames,context["q"],context["axis"],context["intrinsic"],context["travel"],cfg["region_evidence"],seed)
            item.update({"original_frame_id":frame["source"],"proposal_id":proposal["proposal_id"],"source_layer":proposal["source_layer"],"proposal_area_pixels":proposal["area_pixels"]})
            uv=item["sample_uv"]
            if len(uv):
                world=unproject_pixels(uv,frame["depth"][uv[:,1],uv[:,0]],frame["pose"],context["intrinsic"]); item["world_centroid"]=np.median(world,axis=0).tolist(); item["canonical_centroid"]=np.median(world-frame["q"]*context["axis"],axis=0).tolist()
            else: item["world_centroid"]=None; item["canonical_centroid"]=None
            evidence.append(item); rows.append(region_row(item)); all_items.append(item)
            if item["high_confidence_drawer"]:
                drawer_candidates.append({"original_frame_id":frame["source"],"proposal_id":proposal["proposal_id"],"mask":proposal["mask"]&frame["valid"],
                                          "seed_overlap_ratio":item["seed_overlap_ratio"],"drawer_score":item["drawer"]["score"],"static_score":item["static"]["score"],
                                          "drawer_support":item["drawer"]["support_ratio_median"]})
        frame_evidence[frame["source"]]=evidence
        print(f"[region {index+1:02d}/{len(frames):02d}] source={frame['source']} proposals={len(evidence)} labels={dict(Counter(x['label'] for x in evidence))}",flush=True)
    save_csv(output/"region_evidence.csv",rows)
    (output/"region_evidence.json").write_text(json.dumps([json_ready_region(item) for item in all_items],indent=2)+"\n")
    propagation=run_sam2(drawer_candidates,context,cfg,output)
    labels_by_phase={"closed":{},"interaction":{},"open":{}}; propagation_rows=[]
    interaction_start=context["phases"]["interaction"][0]
    for frame in frames:
        masks=load_seed_masks(propagation["seed_dirs"],frame["source"]-interaction_start,frame["depth"].shape)
        propagated,conflict,stats=merge_seed_propagations(masks,frame["valid"],cfg["sam2"]["minimum_agreement_fraction"])
        labels,conflicts=resolve_interaction(frame["valid"],frame["proposals"],frame_evidence[frame["source"]],propagated,conflict)
        labels_by_phase["interaction"][frame["source"]]=labels; save_assignment(output,"interaction",frame["source"],labels)
        propagation_rows.append({"phase":"interaction","original_frame_id":frame["source"],**stats,**conflicts})
    static_core=load_cloud(Path(cfg["inputs"]["v2_dir"])/"static_core.ply"); drawer_core=load_cloud(Path(cfg["inputs"]["v2_dir"])/"drawer_core.ply")
    closed_prior={"policy":"v2 conservative closed cores only","static_core_points":len(static_core),"drawer_core_points":len(drawer_core),
                  "v1_static_reused":False,"official_interaction_surface_reused":False,"static_is_remainder":False,"closed_static_prior":"v2_static_core"}
    (output/"closed_static_prior_audit.json").write_text(json.dumps(closed_prior,indent=2)+"\n")
    for phase in ("closed","open"):
        for frame in context["frames"][phase]:
            static=supported_template_pixels(static_core,frame,context["intrinsic"],cfg["projective_evidence"])
            if phase=="closed":
                drawer=supported_template_pixels(drawer_core,frame,context["intrinsic"],cfg["projective_evidence"]); conflict=np.zeros_like(frame["valid"])
                stats={"seed_count":0,"agreement_mean":0.0,"accepted_pixels":int(drawer.sum()),"conflict_pixels":0}
            else:
                masks=load_seed_masks(propagation["seed_dirs"],frame["source"]-interaction_start,frame["depth"].shape)
                drawer,conflict,stats=merge_seed_propagations(masks,frame["valid"],cfg["sam2"]["minimum_agreement_fraction"])
            labels=resolve_positive_support(frame["valid"],static,drawer,conflict); labels_by_phase[phase][frame["source"]]=labels; save_assignment(output,phase,frame["source"],labels)
            propagation_rows.append({"phase":phase,"original_frame_id":frame["source"],**stats,"static_positive_pixels":int(static.sum())})
    save_csv(output/"sam2_propagation_agreement.csv",propagation_rows)
    representatives=render_representative_frames(frames,frame_evidence,labels_by_phase["interaction"],context["moving_labels"],output)
    projection_dir=output/"visualization"/"accepted_drawer_projections"; projection_dir.mkdir(parents=True,exist_ok=True)
    accepted=[(frame,proposal,item) for frame in frames for proposal,item in zip(frame["proposals"],frame_evidence[frame["source"]]) if item["label"]=="drawer"]
    projection_sources=[]
    if accepted:
        accepted_q=np.asarray([triple[0]["q"] for triple in accepted])
        for quantile in (0.1,0.5,0.9):
            selected=accepted[int(np.argmin(np.abs(accepted_q-np.quantile(accepted_q,quantile))))]; source,proposal,item=selected
            target=max(frames,key=lambda frame:abs(frame["q"]-source["q"])); panel=projection_panel(source,item,target,context["intrinsic"],context["axis"],cfg["projective_evidence"])
            cv2.imwrite(str(projection_dir/f"source_{source['source']}_{proposal['proposal_id']}_target_{target['source']}.jpg"),panel); projection_sources.append({"source":source["source"],"target":target["source"],"proposal_id":proposal["proposal_id"]})
    category_images={name:[] for name in ("drawer","static","mixed","ambiguous")}
    for frame in frames:
        for proposal,item in zip(frame["proposals"],frame_evidence[frame["source"]]):
            category="mixed" if item["reason"]=="unknown_mixed_surface" else "ambiguous" if item["label"]=="unknown" else item["label"]
            if category in category_images and len(category_images[category])<12: category_images[category].append(region_preview(frame,proposal,item))
    contacts=output/"visualization"/"contact_sheets"; contacts.mkdir(parents=True,exist_ok=True)
    for category,images in category_images.items():
        sheet=make_contact_sheet(images);
        if sheet is not None: cv2.imwrite(str(contacts/f"{category}_regions.jpg"),sheet)
    near_threshold=float(cfg["near_contact"]["region_centroid_distance_m"]); static_items=[x for x in all_items if x["label"]=="static" and x["world_centroid"] is not None]; drawer_items=[x for x in all_items if x["label"]=="drawer" and x["canonical_centroid"] is not None]
    static_tree=cKDTree(np.asarray([x["world_centroid"] for x in static_items])) if static_items else None; near_ids=set(); pair_distances=[]
    if static_tree is not None:
        for item in drawer_items:
            distance,index=static_tree.query(np.asarray(item["canonical_centroid"])); pair_distances.append(float(distance))
            if distance<=near_threshold: near_ids.add((item["original_frame_id"],item["proposal_id"])); other=static_items[int(index)]; near_ids.add((other["original_frame_id"],other["proposal_id"]))
    difficult=[item for item in all_items if item["label"]!="invalid" and (item["reason"]=="unknown_mixed_surface" or abs(item["drawer"]["score"]-item["static"]["score"])<=float(cfg["near_contact"]["maximum_evidence_margin"]) or (item["original_frame_id"],item["proposal_id"]) in near_ids)]
    near_rows=[region_row(item) for item in difficult]; save_csv(output/"near_contact_region_evidence.csv",near_rows)
    frame_by_id={frame["source"]:frame for frame in frames}
    near_visual=output/"near_contact_visualizations"; near_visual.mkdir(exist_ok=True)
    for index,item in enumerate(difficult[:24]):
        frame=frame_by_id[item["original_frame_id"]]
        proposal=next(p for p in frame["proposals"] if p["proposal_id"]==item["proposal_id"]); cv2.imwrite(str(near_visual/f"{index:03d}_{item['original_frame_id']}_{item['source_layer']}.jpg"),region_preview(frame,proposal,item))
    near_points=[]; near_colors=[]
    for item in difficult:
        frame=frame_by_id[item["original_frame_id"]]; uv=item["sample_uv"]
        if not len(uv): continue
        world=unproject_pixels(uv,frame["depth"][uv[:,1],uv[:,0]],frame["pose"],context["intrinsic"]); pref=item["point_preference"]
        colors=np.zeros((len(uv),3),np.uint8); colors[pref==-1]=(40,120,255); colors[pref==1]=(255,50,40); colors[pref==0]=(255,210,30); near_points.append(world); near_colors.append(colors)
    write_ply(output/"near_contact_point_preference.ply",np.concatenate(near_points) if near_points else np.empty((0,3)),np.concatenate(near_colors) if near_colors else np.empty((0,3)))
    near_report={"automatic_definition":["unknown_mixed_surface","absolute evidence margin below configured threshold","accepted drawer/static canonical centroids within configured distance"],
                 "manual_roi_used":False,"difficult_region_count":len(difficult),"mixed_region_count":sum(x["reason"]=="unknown_mixed_surface" for x in difficult),
                 "close_accepted_region_count":len(near_ids),"accepted_pair_nearest_distance_m":quantiles(pair_distances),"unknown_ratio":sum(x["label"]=="unknown" for x in difficult)/max(len(difficult),1)}
    (output/"near_contact_region_report.json").write_text(json.dumps(near_report,indent=2)+"\n")
    point_parts={name:[] for name in ("static","drawer","unknown")}; color_parts={name:[] for name in point_parts}; phase_pixels={phase:{name:0 for name in ("static","drawer","unknown","invalid")} for phase in labels_by_phase}
    for phase in ("closed","interaction","open"):
        for labels in labels_by_phase[phase].values():
            for name,value in (("static",LABEL_STATIC),("drawer",LABEL_DRAWER),("unknown",LABEL_UNKNOWN),("invalid",LABEL_INVALID)):
                phase_pixels[phase][name]+=int((labels==value).sum())
    strides=cfg["sampling"]
    for phase in ("closed","interaction","open"):
        for frame in context["frames"][phase][::int(strides[f"{phase}_pointcloud_stride"])]:
            labels=labels_by_phase[phase][frame["source"]]
            for name,value in (("static",LABEL_STATIC),("drawer",LABEL_DRAWER),("unknown",LABEL_UNKNOWN)):
                mask=labels==value; yy,xx=np.nonzero(mask)
                if not len(xx): continue
                world=unproject_pixels(np.column_stack((xx,yy)),frame["depth"][yy,xx],frame["pose"],context["intrinsic"])
                if name=="drawer": world=world-frame["q"]*context["axis"]
                point_parts[name].append(world); color_parts[name].append(frame["rgb"][yy,xx][:,::-1])
    current={}
    for name,filename in (("static","static_v4_points.ply"),("drawer","drawer_v4_canonical_points.ply"),("unknown","unknown_v4_points.ply")):
        points=np.concatenate(point_parts[name]) if point_parts[name] else np.empty((0,3)); colors=np.concatenate(color_parts[name]) if color_parts[name] else np.empty((0,3)); points,colors=voxel_reduce(points,colors,float(cfg["assignment"]["output_voxel_m"])); current[name]=points; write_ply(output/filename,points,colors)
    v1=Path(cfg["inputs"]["v1_dir"]); v2=Path(cfg["inputs"]["v2_dir"]); threshold=float(cfg["comparison"]["nearest_assignment_distance_m"])
    comparison={"v1":{"method":"drawer template plus static remainder","static_point_count":len(load_cloud(v1/"cabinet_static_points.ply")),"drawer_point_count":len(load_cloud(v1/"drawer_canonical_closed_points.ply"))},
                "v2":{"method":"dual-close static bias with conservative cores","static_point_count":len(load_cloud(v2/"static_v2_points.ply")),"drawer_point_count":len(load_cloud(v2/"drawer_v2_canonical_points.ply"))},
                "v3":{"method":"point-track diagnostic","formal_assignment":False,"point_count_comparison":None,"failure":"track sparsity/correspondence failure"},
                "v4":{"method":"region-level direct projective geometry","static_point_count":len(current["static"]),"drawer_point_count":len(current["drawer"]),"unknown_point_count":len(current["unknown"])},
                "old_v1_static_fate":fate(load_cloud(v1/"cabinet_static_points.ply"),current,threshold),"old_v1_drawer_fate":fate(load_cloud(v1/"drawer_canonical_closed_points.ply"),current,threshold),
                "v2_static_fate":fate(load_cloud(v2/"static_v2_points.ply"),current,threshold),"near_contact_unknown_ratio":near_report["unknown_ratio"],
                "drawer_regions_without_initial_seed_overlap":sum(x["label"]=="drawer" and x["seed_overlap_ratio"]==0 for x in all_items),
                "open_only_unknown_ratio":phase_pixels["open"]["unknown"]/max(sum(phase_pixels["open"].values())-phase_pixels["open"]["invalid"],1)}
    (output/"assignment_v1_v2_v3_v4_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    region_counts=Counter(item["label"] for item in all_items); reason_counts=Counter(item["reason"] for item in all_items); prop_report=propagation["report"]; valid_items=[item for item in all_items if item["label"]!="invalid"]
    summary={"output_kind":OUTPUT_KIND,"proposal_count":proposal_audit["proposal_count"],"valid_region_count":sum(x["label"]!="invalid" for x in all_items),
             "static_region_count":region_counts["static"],"drawer_region_count":region_counts["drawer"],"unknown_region_count":region_counts["unknown"],"invalid_region_count":region_counts["invalid"],
             "mixed_region_count":reason_counts["unknown_mixed_surface"],"drawer_region_with_seed_count":sum(x["label"]=="drawer" and x["seed_overlap_ratio"]>0 for x in all_items),
             "drawer_region_without_seed_count":sum(x["label"]=="drawer" and x["seed_overlap_ratio"]==0 for x in all_items),"region_reasons":dict(reason_counts),"per_phase_pixels":phase_pixels,
             "interaction_drawer_support_distribution":quantiles([x["drawer"]["support_ratio_median"] for x in valid_items]),"interaction_static_support_distribution":quantiles([x["static"]["support_ratio_median"] for x in valid_items]),
             "evidence_margin_distribution":quantiles([x["drawer"]["score"]-x["static"]["score"] for x in valid_items]),"sam2_seed_count":prop_report["seed_count"],"sam2_propagation_ran":prop_report["ran"],
             "sam2_propagation_agreement":quantiles([x.get("agreement_mean",0) for x in propagation_rows if x["phase"] in ("interaction","open")]),
             "open":{"propagated_drawer_ratio":phase_pixels["open"]["drawer"]/max(sum(phase_pixels["open"].values())-phase_pixels["open"]["invalid"],1),
                     "static_positive_ratio":phase_pixels["open"]["static"]/max(sum(phase_pixels["open"].values())-phase_pixels["open"]["invalid"],1),
                     "unknown_open_only_ratio":comparison["open_only_unknown_ratio"]},"representative_interaction_frames":representatives,"accepted_drawer_projection_pairs":projection_sources}
    (output/"region_assignment_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    acceptance={"ready_for_dual_tsdf":False,"human_inspection_passed":False,"blocking_reasons":["explicit human review absent","no accepted drawer region" if region_counts["drawer"]==0 else "human validation of drawer regions absent","no accepted static region" if region_counts["static"]==0 else "human validation of static regions absent"],
                "tsdf_ran":False,"nksr_ran":False,"mesh_ran":False,"glb_ran":False,"urdf_ran":False,"pose_refinement_ran":False,"axis_refinement_ran":False,"q_refinement_ran":False,"moving_map_refinement_ran":False}
    (output/"acceptance_report.json").write_text(json.dumps(acceptance,indent=2)+"\n")
    report=f"""# Assignment v4: Region-level Articulation Consistency\n\nPoint tracking / LoFTR is not used as the primary ownership mechanism in Assignment v4.\n\n- Parent commit: `{cfg['audit']['parent_commit']}`\n- Frozen poses/axis/q/moving-map: verified\n- Regions: static={region_counts['static']}, drawer={region_counts['drawer']}, unknown={region_counts['unknown']}, invalid={region_counts['invalid']}\n- Revealed drawer regions (seed overlap 0): {summary['drawer_region_without_seed_count']}\n- Mixed regions: {summary['mixed_region_count']}\n- SAM2 propagation: {'ran' if prop_report['ran'] else 'not run'}; seeds={prop_report['seed_count']}\n- TSDF/NKSR/Mesh/GLB/URDF: not run\n- Ready for dual TSDF: **false** (explicit human inspection is absent)\n"""
    (output/"RUN_REPORT.md").write_text(report)
    outputs=[output/name for name in ("region_evidence.json","region_evidence.csv","region_assignment_summary.json","static_v4_points.ply","drawer_v4_canonical_points.ply","unknown_v4_points.ply")]
    (output/"core_output_hashes.json").write_text(json.dumps(sha_files(outputs),indent=2)+"\n")
    print(json.dumps({"output":str(output),"regions":dict(region_counts),"sam2":prop_report["ran"],"ready_for_dual_tsdf":False},indent=2))
