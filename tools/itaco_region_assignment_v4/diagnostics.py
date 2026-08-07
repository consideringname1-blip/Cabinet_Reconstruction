"""Assignment v4 reports, point clouds and visual diagnostics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import LABEL_DRAWER,LABEL_INVALID,LABEL_STATIC,LABEL_UNKNOWN
from .projective_models import CONTRADICTION,OCCLUDED,SUPPORTED,evaluate_model,unproject_pixels

PALETTE={LABEL_STATIC:(255,120,30),LABEL_DRAWER:(30,40,245),LABEL_UNKNOWN:(30,220,220),LABEL_INVALID:(0,0,0)}


def save_csv(path: Path,rows: list[dict]) -> None:
    if not rows: path.write_text(""); return
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def json_ready_region(item: dict) -> dict:
    result={}
    for key,value in item.items():
        if key in ("sample_uv","point_preference","mask","eroded_mask"): continue
        if isinstance(value,np.generic): result[key]=value.item()
        elif isinstance(value,np.ndarray): result[key]=value.tolist()
        else: result[key]=value
    return result


def write_ply(path: Path,points: np.ndarray,colors: np.ndarray) -> None:
    points=np.asarray(points,np.float32).reshape(-1,3); colors=np.clip(np.asarray(colors),0,255).astype(np.uint8).reshape(-1,3)
    dtype=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("red","u1"),("green","u1"),("blue","u1")]); data=np.empty(len(points),dtype=dtype)
    if len(points): data["x"],data["y"],data["z"]=points.T; data["red"],data["green"],data["blue"]=colors.T
    header=("ply\nformat binary_little_endian 1.0\n"+f"element vertex {len(points)}\n"+"property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as stream: stream.write(header.encode()); data.tofile(stream)


def voxel_reduce(points: np.ndarray,colors: np.ndarray,voxel: float) -> tuple[np.ndarray,np.ndarray]:
    if not len(points): return np.empty((0,3)),np.empty((0,3),np.uint8)
    keys=np.floor(points/voxel).astype(np.int64); _,index=np.unique(keys,axis=0,return_index=True); index=np.sort(index)
    return points[index],colors[index]


def assignment_overlay(rgb: np.ndarray,labels: np.ndarray) -> np.ndarray:
    color=np.zeros_like(rgb)
    for value,bgr in PALETTE.items(): color[labels==value]=bgr
    return cv2.addWeighted(rgb,.45,color,.55,0)


def proposal_overlay(rgb: np.ndarray,proposals: list[dict]) -> np.ndarray:
    image=rgb.copy()
    for index,proposal in enumerate(proposals):
        contours,_=cv2.findContours(proposal["mask"].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        color=((37*index)%255,(97*index+80)%255,(151*index+40)%255); cv2.drawContours(image,contours,-1,color,1)
    return image


def score_overlay(rgb: np.ndarray,proposals: list[dict],evidence: list[dict],key: str) -> np.ndarray:
    values=np.zeros(rgb.shape[:2],np.float32); counts=np.zeros(rgb.shape[:2],np.float32)
    for proposal,item in zip(proposals,evidence):
        if key=="margin": value=float(item["drawer"]["score"]-item["static"]["score"]); normalized=np.clip((value+1)/2,0,1)
        else: normalized=float(item[key]["support_ratio_median"])
        mask=proposal["mask"]; values[mask]+=normalized; counts[mask]+=1
    values/=np.maximum(counts,1); heat=cv2.applyColorMap(np.uint8(np.clip(values,0,1)*255),cv2.COLORMAP_TURBO); shown=rgb.copy(); occupied=counts>0
    shown[occupied]=cv2.addWeighted(rgb,.35,heat,.65,0)[occupied]; return shown


def point_preference_overlay(rgb: np.ndarray,evidence: list[dict]) -> np.ndarray:
    image=rgb.copy()
    for item in evidence:
        uv=item["sample_uv"]; pref=item["point_preference"]
        for value,color in ((-1,(255,120,30)),(1,(30,40,245)),(0,(30,220,220))):
            pts=uv[pref==value]
            if len(pts): image[pts[:,1],pts[:,0]]=color
    return image


def make_contact_sheet(images: list[np.ndarray],columns: int=3) -> np.ndarray | None:
    if not images:return None
    h,w=images[0].shape[:2]; rows=[]
    for start in range(0,len(images),columns):
        row=images[start:start+columns]+[np.zeros((h,w,3),np.uint8)]*(columns-len(images[start:start+columns])); rows.append(np.hstack(row))
    return np.vstack(rows)


def render_representative_frames(frames: list[dict],frame_evidence: dict[int,list[dict]],labels_by_frame: dict[int,np.ndarray],moving: np.ndarray,out: Path) -> list[int]:
    q=np.asarray([frame["q"] for frame in frames]); chosen=[]
    for quantile in (0.1,0.5,0.9):
        index=int(np.argmin(np.abs(q-np.quantile(q,quantile))))
        if index not in chosen: chosen.append(index)
    visual=out/"visualization"/"representative_interaction"; visual.mkdir(parents=True,exist_ok=True)
    for index in chosen:
        frame=frames[index]; evidence=frame_evidence[frame["source"]]; seed=moving[index]==2
        seed_view=frame["rgb"].copy(); seed_view[seed]=cv2.addWeighted(frame["rgb"],.3,np.full_like(frame["rgb"],(30,40,245)),.7,0)[seed]
        panels=[frame["rgb"],proposal_overlay(frame["rgb"],frame["proposals"]),seed_view,assignment_overlay(frame["rgb"],labels_by_frame[frame["source"]]),
                score_overlay(frame["rgb"],frame["proposals"],evidence,"static"),score_overlay(frame["rgb"],frame["proposals"],evidence,"drawer"),
                score_overlay(frame["rgb"],frame["proposals"],evidence,"margin"),point_preference_overlay(frame["rgb"],evidence)]
        cv2.imwrite(str(visual/f"frame_{frame['source']}_overview.jpg"),make_contact_sheet(panels,4))
    return [frames[index]["source"] for index in chosen]


def projection_panel(source: dict,item: dict,target: dict,intrinsic: np.ndarray,axis: np.ndarray,cfg: dict) -> np.ndarray:
    uv=item["sample_uv"]; depth=source["depth"][uv[:,1],uv[:,0]]; world=unproject_pixels(uv,depth,source["pose"],intrinsic)
    panels=[]; source_view=source["rgb"].copy(); source_view[uv[:,1],uv[:,0]]=(40,255,255); panels.append(source_view)
    for model in ("static","drawer"):
        ev=evaluate_model(world,source["q"],target,target["q"],axis,intrinsic,model,cfg); image=target["rgb"].copy()
        for status,color in ((SUPPORTED,(40,220,40)),(OCCLUDED,(220,160,30)),(CONTRADICTION,(30,30,240))):
            ids=np.flatnonzero(ev["status"]==status); projected=ev["uv"][ids]
            if len(projected): image[projected[:,1],projected[:,0]]=color
        panels.append(image)
    residual=np.zeros((*target["depth"].shape,3),np.uint8); valid=target["valid"]; depth_norm=np.clip(target["depth"]/4,0,1); residual[valid]=cv2.applyColorMap(np.uint8(depth_norm*255),cv2.COLORMAP_VIRIDIS)[valid]
    panels.append(residual); return np.hstack(panels)


def sha_files(paths: list[Path]) -> dict:
    return {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
