"""Add point-fate and track-coverage diagnostics to the Assignment v3 comparison."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, open3d as o3d, yaml
from scipy.spatial import cKDTree

def xyz(path): return np.asarray(o3d.io.read_point_cloud(str(path)).points)
def fate(points, targets, threshold):
    distances=np.stack([cKDTree(t).query(points,workers=-1)[0] if len(t) else np.full(len(points),np.inf) for t in targets])
    label=np.argmin(distances,axis=0); minimum=np.min(distances,axis=0); label[minimum>threshold]=3
    return {name:int((label==i).sum()) for i,name in enumerate(('static','drawer','unknown','invalid_or_unmatched'))}
def coverage(records):
    if not records: return {'observations':0,'unique_frames':0,'image_grid_cells':0,'spatial_voxels_5cm':0,'bbox_extent_m':[0,0,0]}
    p=np.asarray([x['point_world'] for x in records]); uv=np.asarray([x['pixel_uv'] for x in records]); frames={int(x['original_frame_id']) for x in records}
    return {'observations':len(records),'unique_frames':len(frames),'image_grid_cells':int(len(np.unique(np.floor(uv/32).astype(int),axis=0))),
            'spatial_voxels_5cm':int(len(np.unique(np.floor(p/.05).astype(int),axis=0))),'bbox_extent_m':np.ptp(p,axis=0).tolist()}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); a=ap.parse_args(); cfg=yaml.safe_load(a.config.read_text()); out=Path(cfg['inputs']['output_dir']); v1=Path(cfg['inputs']['v1_dir'])
    s,d,u=(xyz(out/name) for name in ('static_v3_points.ply','drawer_v3_canonical_points.ply','unknown_v3_points.ply')); old_s=xyz(v1/'cabinet_static_points.ply'); old_d=xyz(v1/'drawer_canonical_closed_points.ply')
    report=json.loads((out/'assignment_v1_v2_v3_comparison.json').read_text()); threshold=float(cfg['assignment']['static_core_projection_distance_m'])
    report['v3']['point_counts_after_voxel']={'static':len(s),'drawer_canonical':len(d),'unknown':len(u)}
    report['old_contaminated_static_points_fate']=fate(old_s,[s,d,u],threshold); report['old_drawer_points_fate']=fate(old_d,[s,d,u],threshold)
    labels={int(x['track_id']):x['label'] for x in json.loads((out/'track_labels.json').read_text())}; records=np.load(out/'tracks_filtered.npz',allow_pickle=True)['records'].tolist()
    report['track_spatial_coverage']={name:coverage([x for x in records if labels.get(int(x['track_id']))==label]) for name,label in (('moving','moving'),('static','static'),('unknown','unknown'))}
    report['drawer_side_coverage']={'value':None,'accepted':False,'reason':'automatic near-contact support set empty; no fixed ROI fallback'}
    report['cabinet_inner_wall_leakage']={'value':None,'accepted':False,'reason':'no independent inner-wall annotation or nonempty automatic near-contact set'}
    report['interpretation']='Counts and nearest-neighbor fate are diagnostics only; no improvement claim is accepted without region-level geometry review.'
    (out/'assignment_v1_v2_v3_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'v3_points':report['v3']['point_counts_after_voxel'],'coverage':report['track_spatial_coverage']},indent=2))
if __name__=='__main__': main()
