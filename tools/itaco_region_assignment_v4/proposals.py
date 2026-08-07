"""Frame-local AutoSeg proposal adapter; layer IDs are provenance only."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_mapping(path: Path) -> dict[int,int]:
    rows=json.loads(path.read_text())
    mapping={int(row["source_index"]):int(row["local_index"]) for row in rows}
    if len(mapping)!=len(rows): raise ValueError("duplicate AutoSeg source IDs")
    return mapping


def load_frame_proposals(source: int, cfg: dict, mapping: dict[int,int]) -> list[dict]:
    if source not in mapping: return []
    path=Path(cfg["inputs"]["autoseg_npz_dir"])/f"mask_{mapping[source]:03d}.npz"
    layers=np.load(path)["a"]
    if layers.ndim==4: layers=layers[:,0]
    result=[]; erosion=int(cfg["proposals"]["region_erosion_pixels"])
    kernel=np.ones((2*erosion+1,2*erosion+1),np.uint8) if erosion else None
    for layer,raw in enumerate(layers):
        mask=np.asarray(raw,bool)
        if int(mask.sum())<int(cfg["proposals"]["minimum_area_pixels"]): continue
        eroded=cv2.erode(mask.astype(np.uint8),kernel).astype(bool) if erosion else mask.copy()
        result.append({"proposal_id":f"frame_{source}_layer_{layer}","original_frame_id":int(source),"source_layer":int(layer),
                       "source_uid":int(layer),"source_path":str(path),"mask":mask,"eroded_mask":eroded,
                       "area_pixels":int(mask.sum()),"eroded_area_pixels":int(eroded.sum()),"score":float(cfg["proposals"]["legacy_score"])})
    return result


def attach_interaction_proposals(frames: list[dict], cfg: dict) -> dict:
    mapping=load_mapping(Path(cfg["inputs"]["autoseg_reverse_mapping"])); count=0
    metadata=[]
    for frame in frames:
        proposals=load_frame_proposals(frame["source"],cfg,mapping); frame["proposals"]=proposals; count+=len(proposals)
        metadata.extend([{key:value for key,value in proposal.items() if key not in ("mask","eroded_mask")} for proposal in proposals])
    return {"proposal_count":count,"proposals":metadata,"identity_policy":"frame-local provenance only; no cross-frame UID identity"}
