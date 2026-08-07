"""Four-state region ownership and propagation conflict rules."""
from __future__ import annotations

import numpy as np

from . import LABEL_DRAWER, LABEL_INVALID, LABEL_STATIC, LABEL_UNKNOWN


def resolve_interaction(valid: np.ndarray, proposals: list[dict], evidence: list[dict],
                        propagated_drawer: np.ndarray | None = None,
                        propagation_conflict: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    valid=np.asarray(valid,bool); static=np.zeros_like(valid); drawer=np.zeros_like(valid)
    for proposal,item in zip(proposals,evidence):
        region=np.asarray(proposal["mask"],bool)&valid
        if item["label"]=="static": static|=region
        elif item["label"]=="drawer": drawer|=region
    overlap=static&drawer
    propagation_conflict=np.zeros_like(valid) if propagation_conflict is None else np.asarray(propagation_conflict,bool)&valid
    propagated=np.zeros_like(valid) if propagated_drawer is None else np.asarray(propagated_drawer,bool)&valid
    geometry_conflict=propagated&static
    conflict=overlap|propagation_conflict|geometry_conflict
    labels=np.full(valid.shape,LABEL_INVALID,np.uint8); labels[valid]=LABEL_UNKNOWN
    labels[static&~drawer&~conflict]=LABEL_STATIC; labels[drawer&~static&~conflict]=LABEL_DRAWER; labels[conflict]=LABEL_UNKNOWN
    return labels,{"static_drawer_overlap_pixels":int(overlap.sum()),"propagation_geometry_conflict_pixels":int(geometry_conflict.sum()),"propagation_conflict_pixels":int(propagation_conflict.sum())}


def resolve_positive_support(valid: np.ndarray, static_positive: np.ndarray,
                             drawer_positive: np.ndarray, conflict: np.ndarray | None = None) -> np.ndarray:
    valid=np.asarray(valid,bool); static=np.asarray(static_positive,bool)&valid; drawer=np.asarray(drawer_positive,bool)&valid
    conflict=np.zeros_like(valid) if conflict is None else np.asarray(conflict,bool)&valid
    ambiguous=(static&drawer)|conflict
    labels=np.full(valid.shape,LABEL_INVALID,np.uint8); labels[valid]=LABEL_UNKNOWN
    labels[static&~drawer&~ambiguous]=LABEL_STATIC; labels[drawer&~static&~ambiguous]=LABEL_DRAWER; labels[ambiguous]=LABEL_UNKNOWN
    return labels


def merge_seed_propagations(seed_masks: list[np.ndarray], valid: np.ndarray,
                            minimum_agreement_fraction: float) -> tuple[np.ndarray,np.ndarray,dict]:
    valid=np.asarray(valid,bool)
    if not seed_masks:
        return np.zeros_like(valid),np.zeros_like(valid),{"seed_count":0,"agreement_mean":0.0}
    stack=np.stack([np.asarray(mask,bool) for mask in seed_masks]); count=stack.sum(0); agreement=count/len(stack)
    union=count>0; drawer=valid&union&(agreement>=float(minimum_agreement_fraction)); conflict=valid&union&~drawer
    return drawer,conflict,{"seed_count":len(stack),"agreement_mean":float(agreement[union].mean()) if union.any() else 0.0,"union_pixels":int((valid&union).sum()),"accepted_pixels":int(drawer.sum()),"conflict_pixels":int(conflict.sum())}
