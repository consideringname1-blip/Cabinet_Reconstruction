#!/usr/bin/env python3
"""Audit the discovered fivefold registered-depth scale against Long Throw PLY and v4 evidence."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from diagnose_assignment_v4_masks_depth import fixed_depth_color,mask_panel,overlay_mask,project_long_throw,projection_maps,proposal,quantiles,signed_color,title
from itaco_region_assignment_v4.frame_data import load_and_validate
from itaco_region_assignment_v4.projective_models import CONTRADICTION,OCCLUDED,SUPPORTED,evaluate_model,unproject_pixels
from itaco_region_assignment_v4.proposals import attach_interaction_proposals


def compact(ev: dict) -> dict:
    testable=(ev["status"]==SUPPORTED)|(ev["status"]==CONTRADICTION)
    return {"counts":ev["counts"],"testable_count":ev["testable_count"],"support_ratio":ev["support_ratio"],"contradiction_ratio":ev["contradiction_ratio"],"observable_fraction":ev["observable_fraction"],"testable_signed_residual_m":quantiles(ev["signed_depth_residual_m"][testable])}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cfg=yaml.safe_load(args.config.read_text()); formal=load_and_validate(cfg); attach_interaction_proposals(formal["frames"]["interaction"],cfg)
    corrected_cfg=copy.deepcopy(cfg); corrected_cfg["validity"]["depth_scale_to_m"]=float(cfg["validity"]["depth_scale_to_m"])/5.0
    corrected=load_and_validate(corrected_cfg); attach_interaction_proposals(corrected["frames"]["interaction"],corrected_cfg)
    formal_map={f["source"]:f for f in formal["frames"]["interaction"]}; corrected_map={f["source"]:f for f in corrected["frames"]["interaction"]}
    front_dir=args.output/"moving_front_projection_corrected_scale"; front_dir.mkdir(); front_rows=[]
    pairs={195:[177,202,213],202:[177,195,213]}
    for source_id,target_ids in pairs.items():
        source=corrected_map[source_id]; item=proposal(source,25); region=item["eroded_mask"]&source["valid"]
        yy,xx=np.nonzero(region); uv=np.column_stack((xx,yy)); world=unproject_pixels(uv,source["depth"][yy,xx],source["pose"],corrected["intrinsic"])
        cv2.imwrite(str(front_dir/f"source_{source_id}_layer25_corrected_valid.jpg"),np.hstack((title(source["rgb"],f"source {source_id} RGB"),title(overlay_mask(source["rgb"],item["mask"]),"layer25 raw mask"),title(overlay_mask(source["rgb"],region,(20,220,220)),f"valid after /5 scale = {len(world)}"))))
        for target_id in target_ids:
            target=corrected_map[target_id]
            ev=evaluate_model(world,source["q"],target,target["q"],corrected["axis"],corrected["intrinsic"],"drawer",corrected_cfg["region_evidence"])
            pred,measured,residual,valid=projection_maps(ev,target["depth"].shape)
            status_view=target["rgb"].copy()
            for status,color in ((SUPPORTED,(40,220,40)),(OCCLUDED,(220,160,30)),(CONTRADICTION,(30,30,240))):
                ids=np.flatnonzero(ev["status"]==status); points=ev["uv"][ids]
                if len(points): status_view[points[:,1],points[:,0]]=color
            delta=float(target["q"]-source["q"])
            panels=[title(status_view,f"target {target_id} corrected-scale predicted pixels"),title(fixed_depth_color(pred,valid),f"predicted depth | dq={delta:+.4f} m"),title(fixed_depth_color(measured,valid),"measured depth after /5"),title(signed_color(residual,valid),"residual measured-predicted | +/-15 cm")]
            cv2.imwrite(str(front_dir/f"source_{source_id}_target_{target_id}_layer25_corrected_scale.jpg"),np.hstack(panels))
            front_rows.append({"source_frame_id":source_id,"target_frame_id":target_id,"delta_q_m":delta,"source_valid_points":len(world),"corrected_scale_drawer":compact(ev)})
    depth_dir=args.output/"pinhole_vs_long_throw_corrected_scale"; depth_dir.mkdir(); depth_rows=[]
    root=Path(cfg["inputs"]["raw_root"]); timestamps=[line.split()[0] for line in (root/"pinhole_projection/depth.txt").read_text().splitlines() if line.strip()]
    all_formal={f["source"]:f for phase in formal["frames"].values() for f in phase}; all_corrected={f["source"]:f for phase in corrected["frames"].values() for f in phase}
    for source_id in (195,202,206):
        f0=all_formal[source_id]; fc=all_corrected[source_id]; ply=root/"Depth Long Throw"/f"{timestamps[source_id]}.ply"; world=np.asarray(o3d.io.read_point_cloud(str(ply)).points); projected=project_long_throw(world,fc,corrected["intrinsic"])
        file_support=f0["depth"]>0; formal_valid=(f0["depth"]>=.2)&(f0["depth"]<=4.0); corrected_valid=(fc["depth"]>=.2)&(fc["depth"]<=4.0); ply_valid=(projected>=.2)&(projected<=4.0); common=corrected_valid&ply_valid
        raw_common=file_support&(projected>0); ratio=f0["depth"][raw_common]/projected[raw_common]; difference=np.zeros_like(projected); difference[common]=fc["depth"][common]-projected[common]
        requested=np.hstack((title(fc["rgb"],f"frame {source_id} RGB"),mask_panel(corrected_valid,f"pinhole valid after /5: {corrected_valid.mean()*100:.2f}%"),mask_panel(ply_valid,f"Long Throw -> RGB valid: {ply_valid.mean()*100:.2f}%"),title(signed_color(difference,common,.02),"depth difference corrected-ply | +/-2 cm")))
        cv2.imwrite(str(depth_dir/f"frame_{source_id}_requested_four_panel.jpg"),requested)
        raw_difference=np.zeros_like(projected); raw_difference[raw_common]=f0["depth"][raw_common]-projected[raw_common]
        scale=np.zeros_like(projected); scale[raw_common]=f0["depth"][raw_common]/projected[raw_common]
        scale_heat=np.zeros((*scale.shape,3),np.uint8); scale_heat[raw_common]=cv2.applyColorMap(np.uint8(np.clip(scale[raw_common]/6,0,1)*255),cv2.COLORMAP_TURBO).reshape(-1,3)
        cv2.imwrite(str(depth_dir/f"frame_{source_id}_raw_scale_error.jpg"),np.hstack((mask_panel(file_support,f"stored nonzero support: {file_support.mean()*100:.2f}%"),mask_panel(formal_valid,f"v4 current 0.2-4m support: {formal_valid.mean()*100:.2f}%"),title(signed_color(raw_difference,raw_common,4.0),"raw configured depth - PLY | +/-4 m"),title(scale_heat,"stored depth / PLY depth | scale up to 6"))))
        depth_rows.append({"original_frame_id":source_id,"stored_nonzero_ratio":float(file_support.mean()),"formal_v4_valid_ratio":float(formal_valid.mean()),"corrected_valid_ratio":float(corrected_valid.mean()),"long_throw_projected_valid_ratio":float(ply_valid.mean()),"corrected_vs_ply_mask_iou":float(common.sum()/max((corrected_valid|ply_valid).sum(),1)),"stored_to_ply_depth_ratio":quantiles(ratio),"corrected_signed_difference_m":quantiles(difference[common]),"corrected_absolute_difference_m":quantiles(np.abs(difference[common])),"corrected_within_1mm_ratio":float((np.abs(difference[common])<=.001).mean()),"corrected_within_1cm_ratio":float((np.abs(difference[common])<=.01).mean()),"corrected_within_3cm_ratio":float((np.abs(difference[common])<=.03).mean())})
    report={"finding":"stored pinhole depth is approximately five times the corresponding Long Throw PLY virtual-pinhole Z projection","formal_config_depth_scale_to_m":cfg["validity"]["depth_scale_to_m"],"diagnostic_corrected_depth_scale_to_m":corrected_cfg["validity"]["depth_scale_to_m"],"correction_is_diagnostic_only":True,"formal_assignment_modified":False,"front":front_rows,"depth":depth_rows,"tsdf_ran":False,"reconstruction_ran":False}
    (args.output/"depth_scale_audit.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({"output":str(args.output),"front_pairs":len(front_rows),"depth_frames":len(depth_rows)},indent=2))


if __name__=="__main__": main()
