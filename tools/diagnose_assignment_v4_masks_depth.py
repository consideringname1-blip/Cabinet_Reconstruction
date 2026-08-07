#!/usr/bin/env python3
"""Visual follow-up for Assignment v4 accepted masks, front residuals, and Long Throw registration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from itaco_region_assignment_v4.frame_data import load_and_validate
from itaco_region_assignment_v4.projective_models import (
    CONTRADICTION,
    OCCLUDED,
    SUPPORTED,
    evaluate_model,
    predicted_surface_front,
    project_world,
    transform_model,
    unproject_pixels,
)
from itaco_region_assignment_v4.proposals import attach_interaction_proposals


def title(image: np.ndarray, text: str) -> np.ndarray:
    shown=image.copy()
    cv2.rectangle(shown,(0,0),(shown.shape[1],24),(0,0,0),-1)
    cv2.putText(shown,text,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
    return shown


def mask_panel(mask: np.ndarray, text: str) -> np.ndarray:
    image=np.zeros((*mask.shape,3),np.uint8); image[mask]=(255,255,255)
    return title(image,text)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(30,40,245)) -> np.ndarray:
    shown=rgb.copy(); paint=np.full_like(shown,color); shown[mask]=cv2.addWeighted(shown,.25,paint,.75,0)[mask]
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(shown,contours,-1,(255,255,255),1)
    return shown


def fixed_depth_color(values: np.ndarray, valid: np.ndarray, low=.2, high=4.0) -> np.ndarray:
    normalized=np.clip((values-low)/(high-low),0,1)
    heat=cv2.applyColorMap(np.uint8(normalized*255),cv2.COLORMAP_TURBO)
    heat[~valid]=0
    return heat


def signed_color(values: np.ndarray, valid: np.ndarray, limit=.15) -> np.ndarray:
    normalized=np.clip((values+limit)/(2*limit),0,1)
    heat=cv2.applyColorMap(np.uint8(normalized*255),cv2.COLORMAP_TURBO)
    heat[~valid]=0
    return heat


def quantiles(values: np.ndarray) -> dict | None:
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    if not len(values): return None
    return {name:float(np.percentile(values,p)) for name,p in (("p10",10),("p25",25),("p50",50),("p75",75),("p90",90))}


def proposal(frame: dict, layer: int) -> dict:
    return next(item for item in frame["proposals"] if item["source_layer"]==layer)


def save_accepted_masks(frames: list[dict], output: Path) -> list[dict]:
    directory=output/"accepted_layer2_masks"; directory.mkdir()
    records=[]
    frame_map={frame["source"]:frame for frame in frames}
    for source in range(202,207):
        frame=frame_map[source]; item=proposal(frame,2); mask=item["mask"]
        rgb=title(frame["rgb"],f"frame {source} RGB")
        mask_view=mask_panel(mask,f"frame {source} layer2 mask | area={int(mask.sum())} px")
        overlay=title(overlay_mask(frame["rgb"],mask),f"frame {source} RGB + layer2")
        cv2.imwrite(str(directory/f"frame_{source}_layer2_rgb_mask.jpg"),np.hstack((rgb,mask_view,overlay)))
        yy,xx=np.nonzero(mask)
        records.append({"original_frame_id":source,"proposal_id":item["proposal_id"],"area_pixels":int(mask.sum()),
                        "valid_depth_pixels":int((mask&frame["valid"]).sum()),"bbox_xyxy":[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
                        "centroid_uv":[float(xx.mean()),float(yy.mean())]})
    return records


def projection_maps(ev: dict, shape: tuple[int,int]) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    predicted=np.zeros(shape,np.float32); measured=np.zeros(shape,np.float32); residual=np.zeros(shape,np.float32); valid=np.zeros(shape,bool)
    ids=np.flatnonzero(np.isfinite(ev["observed_depth_m"]))
    if len(ids):
        uv=ev["uv"][ids]; u,v=uv[:,0],uv[:,1]
        predicted[v,u]=ev["predicted_depth_m"][ids]; measured[v,u]=ev["observed_depth_m"][ids]
        residual[v,u]=ev["signed_depth_residual_m"][ids]; valid[v,u]=True
    return predicted,measured,residual,valid


def save_front_residuals(context: dict, cfg: dict, output: Path) -> list[dict]:
    directory=output/"moving_front_projection_residuals"; directory.mkdir()
    frames=context["frames"]["interaction"]; frame_map={f["source"]:f for f in frames}; rows=[]
    pairs={195:[177,202,213],202:[177,195,213]}
    for source_id,target_ids in pairs.items():
        source=frame_map[source_id]; item=proposal(source,25); region=item["eroded_mask"]&source["valid"]
        yy,xx=np.nonzero(region); uv=np.column_stack((xx,yy)); depth=source["depth"][yy,xx]
        world=unproject_pixels(uv,depth,source["pose"],context["intrinsic"])
        source_panels=[title(source["rgb"],f"source {source_id} RGB"),title(overlay_mask(source["rgb"],item["mask"]),f"layer25 raw mask area={int(item['mask'].sum())}"),title(overlay_mask(source["rgb"],region,(20,220,220)),f"formal valid samples={len(uv)}")]
        cv2.imwrite(str(directory/f"source_{source_id}_layer25_rgb_mask_valid.jpg"),np.hstack(source_panels))
        for target_id in target_ids:
            target=frame_map[target_id]
            drawer=evaluate_model(world,source["q"],target,target["q"],context["axis"],context["intrinsic"],"drawer",cfg["region_evidence"])
            static=evaluate_model(world,source["q"],target,target["q"],context["axis"],context["intrinsic"],"static",cfg["region_evidence"])
            inverse_world=world-(target["q"]-source["q"])*context["axis"]
            inverse=evaluate_model(inverse_world,0.0,target,0.0,context["axis"],context["intrinsic"],"static",cfg["region_evidence"])
            pred,measured,residual,valid=projection_maps(drawer,target["depth"].shape)
            status_view=target["rgb"].copy()
            for status,color in ((SUPPORTED,(40,220,40)),(OCCLUDED,(220,160,30)),(CONTRADICTION,(30,30,240))):
                ids=np.flatnonzero(drawer["status"]==status); points=drawer["uv"][ids]
                if len(points):
                    status_view[points[:,1],points[:,0]]=color
            delta=float(target["q"]-source["q"])
            panels=[title(status_view,f"target {target_id} predicted pixels | green support red contradiction"),
                    title(fixed_depth_color(pred,valid),f"predicted depth | dq={delta:+.4f} m"),
                    title(fixed_depth_color(measured,valid),"measured registered depth"),
                    title(signed_color(residual,valid),"residual measured-predicted | +/-15 cm")]
            cv2.imwrite(str(directory/f"source_{source_id}_target_{target_id}_layer25_drawer_projection.jpg"),np.hstack(panels))
            testable=(drawer["status"]==SUPPORTED)|(drawer["status"]==CONTRADICTION)
            residual_values=drawer["signed_depth_residual_m"][testable]
            rows.append({"source_frame_id":source_id,"target_frame_id":target_id,"source_q_m":float(source["q"]),"target_q_m":float(target["q"]),"delta_q_m":delta,
                         "source_raw_mask_pixels":int(item["mask"].sum()),"source_formal_valid_points":len(world),"formal_drawer":{key:drawer[key] for key in ("counts","testable_count","support_ratio","contradiction_ratio","observable_fraction")},
                         "formal_static":{key:static[key] for key in ("counts","testable_count","support_ratio","contradiction_ratio","observable_fraction")},
                         "inverse_axis_diagnostic":{key:inverse[key] for key in ("counts","testable_count","support_ratio","contradiction_ratio","observable_fraction")},
                         "formal_drawer_testable_residual_m":quantiles(residual_values)})
    return rows


def project_long_throw(world: np.ndarray, frame: dict, intrinsic: np.ndarray) -> np.ndarray:
    uv,z,inside=project_world(world,frame["pose"],intrinsic,frame["depth"].shape)
    ids=predicted_surface_front(uv,z,inside,frame["depth"].shape)
    result=np.zeros(frame["depth"].shape,np.float32)
    if len(ids): result[uv[ids,1],uv[ids,0]]=z[ids]
    return result


def save_depth_comparison(context: dict, raw_root: Path, output: Path) -> list[dict]:
    directory=output/"pinhole_vs_long_throw"; directory.mkdir()
    root=Path(raw_root)
    timestamps=[line.split()[0] for line in (root/"pinhole_projection/depth.txt").read_text().splitlines() if line.strip()]
    all_frames={frame["source"]:frame for phase in context["frames"].values() for frame in phase}
    rows=[]
    for source in (195,202,206):
        frame=all_frames[source]; ply=root/"Depth Long Throw"/f"{timestamps[source]}.ply"
        world=np.asarray(o3d.io.read_point_cloud(str(ply)).points)
        projected=project_long_throw(world,frame,context["intrinsic"])
        registered=frame["depth"]; pin_valid=(registered>=.2)&(registered<=4.0); ply_valid=(projected>=.2)&(projected<=4.0); common=pin_valid&ply_valid
        difference=np.zeros_like(registered); difference[common]=registered[common]-projected[common]
        rgb=title(frame["rgb"],f"frame {source} RGB")
        pin=mask_panel(pin_valid,f"pinhole registered valid {pin_valid.mean()*100:.2f}%")
        raw=mask_panel(ply_valid,f"Long Throw PLY -> RGB valid {ply_valid.mean()*100:.2f}%")
        diff=title(signed_color(difference,common,.05),"depth difference pinhole-PLY | +/-5 cm")
        cv2.imwrite(str(directory/f"frame_{source}_pinhole_vs_long_throw.jpg"),np.hstack((rgb,pin,raw,diff)))
        values=difference[common]
        rows.append({"original_frame_id":source,"long_throw_timestamp":timestamps[source],"long_throw_ply":str(ply),"image_pixels":int(registered.size),
                     "pinhole_valid_pixels":int(pin_valid.sum()),"pinhole_valid_ratio":float(pin_valid.mean()),"long_throw_projected_valid_pixels":int(ply_valid.sum()),"long_throw_projected_valid_ratio":float(ply_valid.mean()),
                     "valid_intersection_pixels":int(common.sum()),"valid_union_pixels":int((pin_valid|ply_valid).sum()),"valid_mask_iou":float(common.sum()/max((pin_valid|ply_valid).sum(),1)),
                     "pinhole_only_pixels":int((pin_valid&~ply_valid).sum()),"long_throw_only_pixels":int((ply_valid&~pin_valid).sum()),"signed_depth_difference_m":quantiles(values),
                     "absolute_depth_difference_m":quantiles(np.abs(values)),"within_1mm_ratio":float((np.abs(values)<=.001).mean()) if len(values) else 0.0,"within_1cm_ratio":float((np.abs(values)<=.01).mean()) if len(values) else 0.0,"within_3cm_ratio":float((np.abs(values)<=.03).mean()) if len(values) else 0.0})
    return rows


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cfg=yaml.safe_load(args.config.read_text()); context=load_and_validate(cfg); attach_interaction_proposals(context["frames"]["interaction"],cfg)
    report={"scope":"Assignment v4 follow-up visualization only; no assignment/model/refinement/reconstruction changes",
            "accepted_layer2_masks":save_accepted_masks(context["frames"]["interaction"],args.output),
            "moving_front_projection_residuals":save_front_residuals(context,cfg,args.output),
            "pinhole_vs_long_throw":save_depth_comparison(context,Path(cfg["inputs"]["raw_root"]),args.output),
            "frozen_axis_world":context["axis"].tolist(),"frozen_travel_m":context["travel"],
            "tsdf_ran":False,"nksr_ran":False,"mesh_ran":False,"parameter_optimization_ran":False}
    (args.output/"diagnostic_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"layer2_masks":len(report["accepted_layer2_masks"]),"front_pairs":len(report["moving_front_projection_residuals"]),"depth_frames":len(report["pinhole_vs_long_throw"])},indent=2))


if __name__=="__main__": main()
