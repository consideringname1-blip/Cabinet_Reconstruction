"""Audited SAM2 identity propagation from geometry-accepted drawer regions."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def select_diverse_seeds(candidates: list[dict], maximum: int) -> list[dict]:
    """Deterministically retain strong front and revealed seeds without UID linking."""
    def rank(item): return (-(item["drawer_score"]-item["static_score"]),-item["drawer_support"],item["original_frame_id"],item["proposal_id"])
    groups={"front":sorted([x for x in candidates if x["seed_overlap_ratio"]>0],key=rank),
            "revealed":sorted([x for x in candidates if x["seed_overlap_ratio"]==0],key=rank)}
    selected=[]
    for kind in ("front","revealed"):
        if groups[kind]: selected.append(groups[kind].pop(0))
    remaining=sorted(groups["front"]+groups["revealed"],key=rank)
    for item in remaining:
        if len(selected)>=maximum: break
        duplicate=False
        for prior in selected:
            if prior["original_frame_id"]!=item["original_frame_id"]: continue
            intersection=np.logical_and(prior["mask"],item["mask"]).sum(); union=np.logical_or(prior["mask"],item["mask"]).sum()
            if intersection/max(union,1)>=0.80: duplicate=True; break
        if not duplicate: selected.append(item)
    return sorted(selected[:maximum],key=lambda x:(x["original_frame_id"],x["proposal_id"]))


def run_sam2(candidates: list[dict], context: dict, cfg: dict, output_dir: Path) -> dict:
    selected=select_diverse_seeds(candidates,int(cfg["sam2"]["maximum_seed_count"]))
    manifest_rows=[]; mask_dir=output_dir/"sam2_seed_masks"; mask_dir.mkdir(exist_ok=True)
    interaction_start=context["phases"]["interaction"][0]; combined=list(range(interaction_start,context["phases"]["open"][-1]+1))
    for index,item in enumerate(selected):
        path=mask_dir/f"seed_{index:03d}_{item['original_frame_id']}.npy"; np.save(path,item["mask"].astype(bool))
        manifest_rows.append({"seed_id":f"seed_{index:03d}_{item['original_frame_id']}","original_frame_id":item["original_frame_id"],
                              "proposal_id":item["proposal_id"],"seed_kind":"front" if item["seed_overlap_ratio"]>0 else "revealed",
                              "combined_index":item["original_frame_id"]-interaction_start,"mask_path":str(path.resolve()),
                              "area_px":int(item["mask"].sum()),"seed_overlap_ratio":item["seed_overlap_ratio"]})
    seed_manifest={"candidate_count":len(candidates),"selected_count":len(selected),"seeds":manifest_rows,
                   "prompt_source":"accepted drawer region masks; no point tracks or LoFTR prompts"}
    (output_dir/"sam2_seed_manifest.json").write_text(json.dumps(seed_manifest,indent=2)+"\n")
    if not selected:
        report={"ran":False,"reason":"no high-confidence drawer region","seed_count":0,"per_seed":[]}
        (output_dir/"sam2_propagation_report.json").write_text(json.dumps(report,indent=2)+"\n"); return {"report":report,"seed_dirs":[],"combined":combined}
    video=output_dir/"sam2_video_frames"; video.mkdir(exist_ok=True)
    for index,source in enumerate(combined):
        cv2.imwrite(str(video/f"{index:06d}.jpg"),cv2.imread(str(context["rgb_paths"][source])),[cv2.IMWRITE_JPEG_QUALITY,int(cfg["sam2"]["jpeg_quality"])])
    runtime={"sam2_repo":cfg["inputs"]["sam2_repo"],"model_config":cfg["inputs"]["sam2_model_config"],"checkpoint":cfg["inputs"]["sam2_checkpoint"],
             "video_dir":str(video),"output_dir":str(output_dir/"sam2_propagation_per_seed"),"seeds":manifest_rows}
    runtime_path=output_dir/"sam2_runtime_manifest.json"; runtime_path.write_text(json.dumps(runtime,indent=2)+"\n")
    command=[cfg["inputs"]["python_environment"],"-m","tools.itaco_funrec_assignment_v3.sam2_runner","--manifest",str(runtime_path)]
    subprocess.run(command,cwd=Path(__file__).resolve().parents[2],check=True)
    raw=json.loads((output_dir/"sam2_propagation_per_seed"/"sam2_raw_report.json").read_text())
    report={"ran":True,"seed_count":len(selected),"command":command,"per_seed":raw,"audited_runner":"tools.itaco_funrec_assignment_v3.sam2_runner",
            "point_tracker_prompt_used":False}
    (output_dir/"sam2_propagation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    return {"report":report,"seed_dirs":[output_dir/"sam2_propagation_per_seed"/row["seed_id"] for row in manifest_rows],"combined":combined}


def load_seed_masks(seed_dirs: list[Path], combined_index: int, shape: tuple[int,int]) -> list[np.ndarray]:
    masks=[]
    for directory in seed_dirs:
        path=directory/f"{combined_index:06d}.npy"; masks.append(np.load(path).astype(bool) if path.exists() else np.zeros(shape,bool))
    return masks
