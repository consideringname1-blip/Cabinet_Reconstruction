"""Independent diagnostics for near-contact unknown Assignment v4 regions."""
from __future__ import annotations

from collections import Counter,defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .diagnostics import save_csv,write_ply
from .frame_data import load_and_validate,validate_registered_depth_scale
from .projective_models import (CONTRADICTION,OCCLUDED,STATUS_NAMES,SUPPORTED,UNOBSERVABLE,
                                evaluate_model,unproject_pixels)
from .proposals import attach_interaction_proposals
from .region_evidence import evaluate_region

CATEGORY_BOTH_SUPPORTED="static_drawer_both_supported"
CATEGORY_INSUFFICIENT="insufficient"
CATEGORY_BOTH_BAD="both_bad"
CATEGORY_COLORS={
    CATEGORY_BOTH_SUPPORTED:(45,190,70),
    CATEGORY_INSUFFICIENT:(230,185,35),
    CATEGORY_BOTH_BAD:(220,55,55),
}
STATUS_BGR={UNOBSERVABLE:(130,130,130),SUPPORTED:(40,220,40),OCCLUDED:(220,160,30),CONTRADICTION:(30,30,240)}


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def classify_ambiguity(item: dict,cfg: dict) -> tuple[str,list[str]]:
    reasons=[]
    insufficient_reasons=set(cfg["insufficient_reasons"])
    minimum_total=int(cfg["minimum_total_testable_points"])
    if item["reason"] in insufficient_reasons or min(item["static"]["testable_points"],item["drawer"]["testable_points"])<minimum_total:
        reasons.append("insufficient_model_evidence")
        return CATEGORY_INSUFFICIENT,reasons
    support=float(cfg["both_supported_minimum_median_support"])
    contradiction=float(cfg["both_supported_maximum_median_contradiction"])
    both_supported=(item["static"]["support_ratio_median"]>=support and item["drawer"]["support_ratio_median"]>=support
                    and item["static"]["contradiction_ratio_median"]<=contradiction
                    and item["drawer"]["contradiction_ratio_median"]<=contradiction)
    if both_supported:
        reasons.extend(["static_median_support_high","drawer_median_support_high","both_median_contradiction_low"])
        return CATEGORY_BOTH_SUPPORTED,reasons
    reasons.append("neither_an_indistinguishably_well_supported_pair")
    return CATEGORY_BOTH_BAD,reasons


def target_surface_hits(evidence: dict,proposals: list[dict]) -> list[dict]:
    uv=np.asarray(evidence["uv"],np.int64); status=np.asarray(evidence["status"],np.uint8)
    h,w=proposals[0]["mask"].shape if proposals else (0,0)
    inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h) if h and w else np.zeros(len(uv),bool)
    rows=[]
    for code,name in STATUS_NAMES.items():
        ids=np.flatnonzero(inside&(status==code))
        if not len(ids): continue
        assigned=np.zeros(len(ids),bool)
        for proposal in proposals:
            hit=proposal["mask"][uv[ids,1],uv[ids,0]]
            if hit.any():
                assigned|=hit
                rows.append({"status":name,"target_proposal_id":proposal["proposal_id"],"target_source_layer":proposal["source_layer"],"hit_count":int(hit.sum())})
        unmatched=int((~assigned).sum())
        if unmatched: rows.append({"status":name,"target_proposal_id":"no_autoseg_proposal","target_source_layer":-1,"hit_count":unmatched})
    return rows


def source_overlay(frame: dict,proposal: dict,item: dict,category: str) -> np.ndarray:
    image=frame["rgb"].copy(); mask=proposal["mask"]; rgb=tuple(int(v) for v in CATEGORY_COLORS[category][::-1])
    color=np.full_like(image,tuple(int(v) for v in CATEGORY_COLORS[category][::-1]))
    image[mask]=cv2.addWeighted(image,.35,color,.65,0)[mask]
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image,contours,-1,rgb,2)
    text=[f"{item['proposal_id']}  {category}",f"reason={item['reason']}  q={frame['q']:.4f}m",
          f"S={item['static']['support_ratio_median']:.3f} D={item['drawer']['support_ratio_median']:.3f}"]
    canvas=np.zeros((image.shape[0]+58,image.shape[1],3),np.uint8); canvas[58:]=image
    for index,line in enumerate(text): cv2.putText(canvas,line,(5,16+18*index),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
    return canvas


def q_visibility_plot(item: dict,path: Path) -> None:
    rows=item["per_target"]; q=np.asarray([r["target_q_m"] for r in rows]); delta=np.asarray([r["q_delta_m"] for r in rows])
    figure,axes=plt.subplots(2,1,figsize=(8,5.2),sharex=True)
    axes[0].plot(q,[r["static_observable_fraction"] for r in rows],"o-",label="static observable",color="#2878b5",markersize=3)
    axes[0].plot(q,[r["drawer_observable_fraction"] for r in rows],"o-",label="drawer observable",color="#d62728",markersize=3)
    axes[0].set_ylabel("observable fraction"); axes[0].set_ylim(-.03,1.03); axes[0].grid(alpha=.25); axes[0].legend(fontsize=8,ncol=2)
    axes[1].plot(q,[r["static_support_ratio"] for r in rows],"o-",label="static support",color="#17becf",markersize=3)
    axes[1].plot(q,[r["drawer_support_ratio"] for r in rows],"o-",label="drawer support",color="#e377c2",markersize=3)
    axes[1].plot(q,[r["static_contradiction_ratio"] for r in rows],"--",label="static contradiction",color="#1f4e79",linewidth=1)
    axes[1].plot(q,[r["drawer_contradiction_ratio"] for r in rows],"--",label="drawer contradiction",color="#8b1a1a",linewidth=1)
    axes[1].set_xlabel("target q (m)"); axes[1].set_ylabel("testable ratio"); axes[1].set_ylim(-.03,1.03); axes[1].grid(alpha=.25); axes[1].legend(fontsize=7,ncol=2)
    for axis in axes: axis.axvline(item["source_q_m"],color="black",alpha=.5,linewidth=1)
    figure.suptitle(f"{item['proposal_id']} | q span {delta.min():.3f}..{delta.max():.3f} m")
    figure.tight_layout(); figure.savefig(path,dpi=130); plt.close(figure)


def model_tile(target: dict,evidence: dict,model: str,surface_names: str,tile_size: tuple[int,int]) -> np.ndarray:
    image=target["rgb"].copy(); uv=np.asarray(evidence["uv"],np.int64); status=np.asarray(evidence["status"],np.uint8)
    h,w=image.shape[:2]; inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    for code in (UNOBSERVABLE,OCCLUDED,CONTRADICTION,SUPPORTED):
        ids=np.flatnonzero(inside&(status==code))
        for u,v in uv[ids]: cv2.circle(image,(int(u),int(v)),1,STATUS_BGR[code],-1)
    image=cv2.resize(image,tile_size,interpolation=cv2.INTER_AREA)
    header=np.zeros((38,tile_size[0],3),np.uint8)
    cv2.putText(header,f"{model} target={target['source']} q={target['q']:.3f}",(3,13),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(header,surface_names[:52],(3,29),cv2.FONT_HERSHEY_SIMPLEX,.27,(210,210,210),1,cv2.LINE_AA)
    return np.vstack((header,image))


def target_grid(cells: list[np.ndarray],columns: int) -> np.ndarray:
    if not cells: return np.zeros((1,1,3),np.uint8)
    h,w=cells[0].shape[:2]; rows=[]
    for start in range(0,len(cells),columns):
        row=cells[start:start+columns]+[np.zeros((h,w,3),np.uint8)]*(columns-len(cells[start:start+columns]))
        rows.append(np.hstack(row))
    legend=np.zeros((34,columns*w,3),np.uint8)
    cv2.putText(legend,"predicted hit: green=supported  orange=occluded  red=contradiction  gray=unobservable",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,255),1,cv2.LINE_AA)
    return np.vstack([legend,*rows])


def contact_sheet(paths: list[Path],output: Path,columns: int,thumb_width: int) -> None:
    images=[]
    for path in paths:
        image=cv2.imread(str(path))
        if image is None: continue
        height=max(1,int(image.shape[0]*thumb_width/image.shape[1])); images.append(cv2.resize(image,(thumb_width,height)))
    if not images: return
    h=max(image.shape[0] for image in images); padded=[]
    for image in images:
        canvas=np.zeros((h,thumb_width,3),np.uint8); canvas[:image.shape[0]]=image; padded.append(canvas)
    rows=[]
    for start in range(0,len(padded),columns):
        row=padded[start:start+columns]+[np.zeros((h,thumb_width,3),np.uint8)]*(columns-len(padded[start:start+columns])); rows.append(np.hstack(row))
    cv2.imwrite(str(output),np.vstack(rows))


def spatial_plot(rows: list[dict],output: Path) -> None:
    figure,axes=plt.subplots(1,3,figsize=(13,4))
    dimensions=((0,1,"world X","world Y"),(0,2,"world X","world Z"),(1,2,"world Y","world Z"))
    for axis,(a,b,xlabel,ylabel) in zip(axes,dimensions):
        for category,color in CATEGORY_COLORS.items():
            subset=[row for row in rows if row["audit_category"]==category and row["world_centroid"] is not None]
            if subset:
                points=np.asarray([row["world_centroid"] for row in subset]); axis.scatter(points[:,a],points[:,b],s=18,c=[np.asarray(color)/255],label=category,alpha=.8)
        axis.set_xlabel(xlabel); axis.set_ylabel(ylabel); axis.axis("equal"); axis.grid(alpha=.2)
    axes[0].legend(fontsize=7); figure.tight_layout(); figure.savefig(output,dpi=160); plt.close(figure)


def category_count_plot(counts: Counter,output: Path) -> None:
    names=list(CATEGORY_COLORS); values=[counts[name] for name in names]
    figure,axis=plt.subplots(figsize=(8,4))
    bars=axis.bar(range(len(names)),values,color=[np.asarray(CATEGORY_COLORS[name])/255 for name in names])
    axis.set_xticks(range(len(names)),names,rotation=15,ha="right"); axis.set_ylabel("unknown region count")
    axis.bar_label(bars); axis.set_title("Near-contact ambiguity taxonomy")
    figure.tight_layout(); figure.savefig(output,dpi=160); plt.close(figure)


def category_q_plot(rows: list[dict],category: str,output: Path) -> None:
    selected=[row for row in rows if row["audit_category"]==category]
    q_values=sorted({float(row["target_q_m"]) for row in selected})
    def median(key: str,q: float) -> float:
        values=[float(row[key]) for row in selected if float(row["target_q_m"])==q]
        return float(np.median(values)) if values else float("nan")
    figure,axes=plt.subplots(2,1,figsize=(8,5.2),sharex=True)
    for key,label,color in (("static_observable_fraction","static observable","#2878b5"),("drawer_observable_fraction","drawer observable","#d62728")):
        axes[0].plot(q_values,[median(key,q) for q in q_values],"o-",label=label,color=color,markersize=3)
    for key,label,color,style in (("static_support_ratio","static support","#17becf","-"),("drawer_support_ratio","drawer support","#e377c2","-"),
                                  ("static_contradiction_ratio","static contradiction","#1f4e79","--"),("drawer_contradiction_ratio","drawer contradiction","#8b1a1a","--")):
        axes[1].plot(q_values,[median(key,q) for q in q_values],style,label=label,color=color,markersize=3)
    for axis in axes: axis.set_ylim(-.03,1.03); axis.grid(alpha=.25); axis.legend(fontsize=7,ncol=2)
    axes[0].set_ylabel("median observable fraction"); axes[1].set_ylabel("median testable ratio"); axes[1].set_xlabel("target q (m)")
    figure.suptitle(f"{category}: q-dependent visibility and evidence"); figure.tight_layout(); figure.savefig(output,dpi=160); plt.close(figure)


def target_surface_matrix(rows: list[dict],category: str,output: Path) -> None:
    selected=[row for row in rows if row["audit_category"]==category]
    source_layers=sorted({int(row["source_layer"]) for row in selected})
    target_layers=sorted({int(row["target_source_layer"]) for row in selected})
    figure,axes=plt.subplots(2,3,figsize=(15,8),squeeze=False)
    for model_index,model in enumerate(("static","drawer")):
        for status_index,status in enumerate(("supported","occluded","contradiction")):
            matrix=np.zeros((len(source_layers),len(target_layers)),float)
            for row in selected:
                if row["model"]==model and row["status"]==status:
                    matrix[source_layers.index(int(row["source_layer"])),target_layers.index(int(row["target_source_layer"]))]+=int(row["hit_count"])
            axis=axes[model_index,status_index]; shown=axis.imshow(np.log1p(matrix),aspect="auto",cmap="magma")
            axis.set_title(f"{model} {status} | log(1+hits)"); axis.set_xticks(range(len(target_layers)),target_layers,fontsize=7)
            axis.set_yticks(range(len(source_layers)),source_layers,fontsize=7)
            axis.set_xlabel("target AutoSeg layer (-1=no proposal)"); axis.set_ylabel("source layer")
            figure.colorbar(shown,ax=axis,fraction=.046,pad=.04)
    figure.suptitle(f"{category}: predicted target-surface hits"); figure.tight_layout(); figure.savefig(output,dpi=150); plt.close(figure)


def run(config_path: Path) -> None:
    audit_cfg=yaml.safe_load(config_path.read_text()); base_path=Path(audit_cfg["inputs"]["assignment_v4_config"])
    cfg=yaml.safe_load(base_path.read_text()); v4=Path(audit_cfg["inputs"]["assignment_v4_output"]); output=Path(audit_cfg["output_dir"])
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"non-overwrite output exists: {output}")
    required=[v4/name for name in ("region_evidence.json","near_contact_region_evidence.csv","scale_consistency_gate_report.json","config_resolved.yaml")]
    missing=[str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(json.dumps({"code":"missing_v4_evidence","paths":missing},indent=2))
    frozen=yaml.safe_load((v4/"config_resolved.yaml").read_text())
    if float(frozen["validity"]["depth_scale_to_m"])!=float(cfg["validity"]["depth_scale_to_m"]): raise RuntimeError("resolved v4 config scale differs from audit base config")
    previous_gate=json.loads((v4/"scale_consistency_gate_report.json").read_text())
    if not previous_gate.get("passed"): raise RuntimeError("source Assignment v4 scale gate did not pass")
    repeated_gate=validate_registered_depth_scale(cfg)
    if not repeated_gate["passed"]: raise RuntimeError("near-contact audit scale gate did not pass")
    context=load_and_validate(cfg); frames=context["frames"]["interaction"]; attach_interaction_proposals(frames,cfg)
    by_source={frame["source"]:frame for frame in frames}; source_index={frame["source"]:index for index,frame in enumerate(frames)}
    evidence=json.loads((v4/"region_evidence.json").read_text()); by_key={(item["original_frame_id"],item["proposal_id"]):item for item in evidence}
    with (v4/"near_contact_region_evidence.csv").open() as stream: near_rows=list(csv.DictReader(stream))
    near_keys=[(int(row["original_frame_id"]),row["proposal_id"]) for row in near_rows]
    if len(near_keys)!=int(audit_cfg["audit"]["expected_unknown_region_count"]) or len(set(near_keys))!=len(near_keys): raise RuntimeError("near-contact region cardinality mismatch")
    selected=[by_key[key] for key in near_keys]
    if any(item["label"]!="unknown" for item in selected): raise RuntimeError("near-contact audit received non-unknown region")
    output.mkdir(parents=True); (output/"regions").mkdir(); (output/"summary_visualization").mkdir()
    (output/"config_resolved.yaml").write_text(yaml.safe_dump(audit_cfg,sort_keys=False))
    (output/"execution_manifest.json").write_text(json.dumps({"argv":sys.argv,"cwd":str(Path.cwd()),"entry_script":"tools/audit_assignment_v4_near_contact_ambiguity.py",
                                                               "source_v4_output":str(v4),"source_v4_hashes":{path.name:sha256(path) for path in required}},indent=2)+"\n")
    (output/"scale_consistency_gate_report.json").write_text(json.dumps(repeated_gate,indent=2)+"\n")
    taxonomy=audit_cfg["taxonomy"]; visual=audit_cfg["visualization"]; summary_rows=[]; visibility_rows=[]; surface_rows=[]; overview_by_category=defaultdict(list); category_points=defaultdict(list)
    maximum_recompute_difference=0.0
    for ordinal,item in enumerate(selected):
        source=int(item["original_frame_id"]); frame=by_source[source]; proposal=next(p for p in frame["proposals"] if p["proposal_id"]==item["proposal_id"])
        index=source_index[source]; seed=context["moving_labels"][index]==2
        recomputed=evaluate_region(index,frame,proposal["eroded_mask"],frames,context["q"],context["axis"],context["intrinsic"],context["travel"],cfg["region_evidence"],seed)
        for model in ("static","drawer"):
            for key in ("support_ratio_median","contradiction_ratio_median","score"):
                maximum_recompute_difference=max(maximum_recompute_difference,abs(float(item[model][key])-float(recomputed[model][key])))
        category,category_reasons=classify_ambiguity(item,taxonomy); safe=f"{ordinal:03d}_frame_{source}_layer_{item['source_layer']}"
        region_dir=output/"regions"/category/safe; region_dir.mkdir(parents=True)
        source_image=source_overlay(frame,proposal,item,category); cv2.imwrite(str(region_dir/"source_region.jpg"),source_image)
        item_for_plot={**item,"source_q_m":float(frame["q"]),"per_target":[]}
        uv=recomputed["sample_uv"]; depth=frame["depth"][uv[:,1],uv[:,0]]; world=unproject_pixels(uv,depth,frame["pose"],context["intrinsic"])
        cells=[]; per_region_surface=[]; per_region_visibility=[]
        for target_row in item["per_target"]:
            target=by_source[int(target_row["target_frame_id"])]; plot_row={**target_row,"target_q_m":float(target["q"])}
            item_for_plot["per_target"].append(plot_row)
            per_region_visibility.append({"region_ordinal":ordinal,"proposal_id":item["proposal_id"],"audit_category":category,"source_frame_id":source,
                                          "source_q_m":float(frame["q"]),"target_q_m":float(target["q"]),**target_row})
            model_tiles=[]
            for model in ("static","drawer"):
                ev=evaluate_model(world,float(frame["q"]),target,float(target["q"]),context["axis"],context["intrinsic"],model,cfg["region_evidence"])
                hits=target_surface_hits(ev,target["proposals"]); totals=Counter()
                for hit in hits:
                    row={"region_ordinal":ordinal,"proposal_id":item["proposal_id"],"audit_category":category,"source_frame_id":source,"source_layer":item["source_layer"],
                         "target_frame_id":target["source"],"source_q_m":float(frame["q"]),"target_q_m":float(target["q"]),"q_delta_m":float(target["q"]-frame["q"]),"model":model,**hit}
                    per_region_surface.append(row); totals[hit["target_proposal_id"]]+=hit["hit_count"]
                names=", ".join(f"{name}:{count}" for name,count in totals.most_common(2)) or "no visible hit"
                model_tiles.append(model_tile(target,ev,model,names,tuple(visual["target_tile_size"])))
            cells.append(np.hstack(model_tiles))
        visibility_rows.extend(per_region_visibility); surface_rows.extend(per_region_surface)
        save_csv(region_dir/"q_visibility.csv",per_region_visibility); save_csv(region_dir/"target_surface_hits.csv",per_region_surface)
        q_visibility_plot(item_for_plot,region_dir/"q_visibility.png"); cv2.imwrite(str(region_dir/"target_surfaces.jpg"),target_grid(cells,int(visual["target_grid_columns"])))
        q_image=cv2.imread(str(region_dir/"q_visibility.png")); source_scaled=cv2.resize(source_image,(int(source_image.shape[1]*q_image.shape[0]/source_image.shape[0]),q_image.shape[0]))
        overview=np.hstack((source_scaled,q_image)); cv2.imwrite(str(region_dir/"overview.jpg"),overview); overview_by_category[category].append(region_dir/"overview.jpg")
        if item["world_centroid"] is not None: category_points[category].append(np.asarray(item["world_centroid"],float))
        summary_rows.append({"region_ordinal":ordinal,"proposal_id":item["proposal_id"],"original_frame_id":source,"source_layer":item["source_layer"],"source_q_m":float(frame["q"]),
                             "audit_category":category,"audit_category_reasons":";".join(category_reasons),"original_reason":item["reason"],
                             "static_support_median":item["static"]["support_ratio_median"],"drawer_support_median":item["drawer"]["support_ratio_median"],
                             "static_contradiction_median":item["static"]["contradiction_ratio_median"],"drawer_contradiction_median":item["drawer"]["contradiction_ratio_median"],
                             "static_testable_points":item["static"]["testable_points"],"drawer_testable_points":item["drawer"]["testable_points"],
                             "target_frame_count":item["target_frame_count"],"world_centroid":item["world_centroid"],"canonical_centroid":item["canonical_centroid"],
                             "region_directory":str(region_dir)})
        print(f"[near-contact {ordinal+1:03d}/{len(selected):03d}] {item['proposal_id']} -> {category}",flush=True)
    if maximum_recompute_difference>float(audit_cfg["audit"]["maximum_recomputed_evidence_difference"]): raise RuntimeError(f"recomputed evidence mismatch: {maximum_recompute_difference}")
    save_csv(output/"ambiguity_regions.csv",[{**row,"world_centroid":json.dumps(row["world_centroid"]),"canonical_centroid":json.dumps(row["canonical_centroid"])} for row in summary_rows])
    (output/"ambiguity_regions.json").write_text(json.dumps(summary_rows,indent=2)+"\n"); save_csv(output/"q_dependent_visibility.csv",visibility_rows); save_csv(output/"target_surface_hits.csv",surface_rows)
    colors=[]; points=[]
    for row in summary_rows:
        if row["world_centroid"] is not None: points.append(row["world_centroid"]); colors.append(CATEGORY_COLORS[row["audit_category"]])
    write_ply(output/"ambiguity_region_centroids.ply",np.asarray(points),np.asarray(colors)); spatial_plot(summary_rows,output/"summary_visualization"/"spatial_distribution.png")
    for category,paths in overview_by_category.items(): contact_sheet(paths,output/"summary_visualization"/f"{category}_all_regions.jpg",int(visual["overview_contact_columns"]),int(visual["overview_thumbnail_width"]))
    category_counts=Counter(row["audit_category"] for row in summary_rows); status_counts=Counter()
    for row in surface_rows: status_counts[(row["audit_category"],row["model"],row["status"])]+=int(row["hit_count"])
    category_count_plot(category_counts,output/"summary_visualization"/"category_counts.png")
    for category in category_counts:
        category_q_plot(visibility_rows,category,output/"summary_visualization"/f"q_visibility_{category}.png")
        target_surface_matrix(surface_rows,category,output/"summary_visualization"/f"target_surface_matrix_{category}.png")
    visibility_summary={}
    for category in category_counts:
        rows=[row for row in visibility_rows if row["audit_category"]==category]
        visibility_summary[category]={key:float(np.median([float(row[key]) for row in rows])) for key in ("static_observable_fraction","drawer_observable_fraction","static_support_ratio","drawer_support_ratio")}
    top_surfaces={}
    for category in category_counts:
        for model in ("static","drawer"):
            for status in ("supported","occluded","contradiction"):
                counts=Counter()
                for row in surface_rows:
                    if row["audit_category"]==category and row["model"]==model and row["status"]==status:
                        counts[row["target_proposal_id"]]+=int(row["hit_count"])
                top_surfaces[f"{category}|{model}|{status}"]=counts.most_common(15)
    summary={"source_unknown_region_count":len(selected),"category_counts":dict(category_counts),"taxonomy":taxonomy,
             "world_centroid_bounds":{"minimum":np.min(np.asarray(points),axis=0).tolist(),"maximum":np.max(np.asarray(points),axis=0).tolist()},
             "q_visibility_medians":visibility_summary,
             "target_hit_status_counts":{"|".join(key):value for key,value in status_counts.items()},
             "top_target_surfaces_by_hit_count":top_surfaces,
             "maximum_recomputed_evidence_difference":maximum_recompute_difference,"original_assignment_modified":False,
             "camera_pose_axis_q_moving_map_modified":False,"tsdf_ran":False,"nksr_ran":False,"mesh_ran":False}
    (output/"ambiguity_audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    html_rows=[]
    for row in summary_rows:
        relative=Path(row["region_directory"]).relative_to(output)
        html_rows.append(f"<tr><td>{row['region_ordinal']}</td><td>{html.escape(row['proposal_id'])}</td><td>{html.escape(row['audit_category'])}</td><td>{row['source_q_m']:.4f}</td><td><a href='{relative}/overview.jpg'>overview</a></td><td><a href='{relative}/target_surfaces.jpg'>all targets</a></td></tr>")
    (output/"index.html").write_text("<html><body><h1>Near-contact ambiguity audit</h1><p>green=supported, orange=occluded, red=contradiction, gray=unobservable</p><table border='1'><tr><th>#</th><th>region</th><th>category</th><th>source q</th><th>overview</th><th>target surfaces</th></tr>"+"".join(html_rows)+"</table></body></html>")
    print(json.dumps({"output":str(output),"category_counts":dict(category_counts),"region_count":len(selected),"maximum_recompute_difference":maximum_recompute_difference},indent=2))
