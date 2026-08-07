#!/usr/bin/env python3
"""One-factor-at-a-time threshold diagnostics for Assignment v4; no propagation or reconstruction."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from .frame_data import load_and_validate
from .proposals import attach_interaction_proposals
from .region_evidence import evaluate_region


def quantiles(values):
    values=np.asarray(values,float)
    return {name:float(np.percentile(values,p)) for name,p in (("p10",10),("p25",25),("p50",50),("p75",75),("p90",90))} if len(values) else None


def main(config_path: Path, output_path: Path) -> None:
    cfg=yaml.safe_load(config_path.read_text())
    context=load_and_validate(cfg)
    frames=context["frames"]["interaction"]
    attach_interaction_proposals(frames,cfg)
    primary=cfg["region_evidence"]
    cases=[
        ("primary_3cm",0.03,0.03),
        ("support_2cm",0.02,0.03),
        ("support_4cm",0.04,0.03),
        ("occlusion_2cm",0.03,0.02),
    ]
    report={"policy":"one factor at a time; no combinatorial sweep","free_space_margin_m":float(primary["free_space_margin_m"]),"cases":[]}
    for name,support,occlusion in cases:
        case=copy.deepcopy(primary)
        case["depth_support_threshold_m"]=support
        case["occlusion_margin_m"]=occlusion
        rows=[]
        for source_index,frame in enumerate(frames):
            seed=context["moving_labels"][source_index]==2
            for proposal in frame["proposals"]:
                item=evaluate_region(source_index,frame,proposal["eroded_mask"],frames,context["q"],context["axis"],context["intrinsic"],context["travel"],case,seed)
                rows.append({"original_frame_id":frame["source"],"proposal_id":proposal["proposal_id"],"label":item["label"],"reason":item["reason"],"static_score":item["static"]["score"],"drawer_score":item["drawer"]["score"]})
        labels=Counter(row["label"] for row in rows)
        reasons=Counter(row["reason"] for row in rows)
        valid=[row for row in rows if row["label"]!="invalid"]
        report["cases"].append({"name":name,"depth_support_threshold_m":support,"occlusion_margin_m":occlusion,"region_counts":dict(labels),"reason_counts":dict(reasons),"evidence_margin_distribution":quantiles([row["drawer_score"]-row["static_score"] for row in valid]),"accepted_static_regions":[[row["original_frame_id"],row["proposal_id"]] for row in rows if row["label"]=="static"],"accepted_drawer_regions":[[row["original_frame_id"],row["proposal_id"]] for row in rows if row["label"]=="drawer"]})
        print(name,dict(labels),flush=True)
    output_path.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    main(args.config,args.output)
