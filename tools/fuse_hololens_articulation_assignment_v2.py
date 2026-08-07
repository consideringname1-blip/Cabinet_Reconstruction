#!/usr/bin/env python3
"""Ownership assignment v2: clean closed core plus conservative revealed expansion.

This entry point intentionally contains no TSDF, NKSR, mesh, pose, axis, q, or
moving-map optimization code.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import cv2,matplotlib,numpy as np,open3d as o3d,yaml
from scipy.spatial import cKDTree
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent)); import fuse_hololens_articulation_dual_volume as dv

LABELS={'static_core':0,'drawer_core':1,'static_new_world_consistent':2,'drawer_new_canonical_consistent':3,'unknown_ambiguous_motion':4,'unknown_insufficient_support':5,'unknown_low_q_span':6,'unknown_visibility_failure':7,'invalid':8}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def keys(x,v): return np.floor(x/v).astype(np.int64)
def cloud(x,c=None,voxel=.006):
    p=o3d.geometry.PointCloud(); p.points=o3d.utility.Vector3dVector(x)
    if c is not None and len(c)==len(x): p.colors=o3d.utility.Vector3dVector(c)
    return p.voxel_down_sample(voxel)
def save_cloud(path,x,c,voxel): o3d.io.write_point_cloud(str(path),cloud(x,c,voxel))

def projected_visibility(points,pose,k,shape,rgb_support,hand,depth_min,depth_max,threshold,erosion):
    h,w=shape; cam=(points-pose[:3,3])@pose[:3,:3]; z=cam[:,2]; u=np.rint(k[0,0]*cam[:,0]/np.maximum(z,1e-12)+k[0,2]).astype(int); v=np.rint(k[1,1]*cam[:,1]/np.maximum(z,1e-12)+k[1,2]).astype(int)
    inside=(z>=depth_min)&(z<=depth_max)&(u>=0)&(u<w)&(v>=0)&(v<h)&np.isfinite(cam).all(1); ids=np.nonzero(inside)[0]; flat=v[ids]*w+u[ids]; order=np.lexsort((ids,z[ids])); ids=ids[order]; flat=flat[order]; _,first=np.unique(flat,return_index=True); front=ids[first]
    depth=np.zeros(h*w,np.float32); depth[flat[first]]=z[front]; depth=depth.reshape(h,w); valid_depth=depth>0
    if erosion: valid_depth=cv2.erode(valid_depth.astype(np.uint8),np.ones((2*erosion+1,)*2,np.uint8)).astype(bool)
    visible=np.zeros(len(points),bool); consistent=np.zeros(len(points),bool); base=inside.copy(); base[inside]&=rgb_support[v[inside],u[inside]]&~hand[v[inside],u[inside]]&valid_depth[v[inside],u[inside]]; visible[front]=base[front]; consistent[base]=np.abs(z[base]-depth[v[base],u[base]])<=threshold
    return u,v,inside,visible,consistent,depth

def voxel_stats(x,frame,q,phase,voxel):
    k=keys(x,voxel); uniq,inv=np.unique(k,axis=0,return_inverse=True); n=len(uniq); cent=np.zeros((n,3)); np.add.at(cent,inv,x); count=np.bincount(inv,minlength=n); cent/=count[:,None]; res=np.linalg.norm(x-cent[inv],axis=1)
    frames=np.zeros(n,int); qlo=np.full(n,np.inf); qhi=np.full(n,-np.inf); med=np.zeros(n); p90=np.zeros(n); phases=[]
    # Group once instead of scanning every observation for every voxel.
    order=np.argsort(inv,kind='stable'); sorted_inv=inv[order]; starts=np.r_[0,np.flatnonzero(np.diff(sorted_inv))+1]; ends=np.r_[starts[1:],len(order)]
    for start,end in zip(starts,ends):
        ids=order[start:end]; i=int(sorted_inv[start]); frames[i]=len(np.unique(frame[ids])); qlo[i]=q[ids].min(); qhi[i]=q[ids].max(); med[i]=np.median(res[ids]); p90[i]=np.percentile(res[ids],90); phases.append(sorted(set(phase[ids].tolist())))
    return {'keys':uniq,'inverse':inv,'centroid':cent,'count':count,'frames':frames,'qspan':qhi-qlo,'median':med,'p90':p90,'phases':phases}

def classify_motion(static,drawer,travel,cfg):
    si=static['inverse']; di=drawer['inverse']; sf=static['frames'][si]; df=drawer['frames'][di]; ss=static['p90'][si]; ds=drawer['p90'][di]; dq=drawer['qspan'][di]
    stable_s=(sf>=cfg['minimum_static_frames'])&(ss<=cfg['maximum_spread_m']); stable_d=(df>=cfg['minimum_drawer_frames'])&(ds<=cfg['maximum_spread_m']); qok=dq>=cfg['minimum_drawer_q_span_fraction']*travel
    accept_s=stable_s&((~stable_d)|(ss+cfg['evidence_spread_margin_m']<ds)|(sf>df)); accept_d=stable_d&qok&((~stable_s)|(ds+cfg['evidence_spread_margin_m']<ss)|(df>sf)); both=accept_s&accept_d; accept_s[both]=False; accept_d[both]=False
    label=np.full(len(si),LABELS['unknown_ambiguous_motion'],np.uint8); label[accept_s]=LABELS['static_new_world_consistent']; label[accept_d]=LABELS['drawer_new_canonical_consistent']; low=(~accept_s)&(~accept_d)&stable_d&(~qok); label[low]=LABELS['unknown_low_q_span']; insuff=(~accept_s)&(~accept_d)&(sf<cfg['minimum_static_frames'])&(df<cfg['minimum_drawer_frames']); label[insuff]=LABELS['unknown_insufficient_support']
    return label,{'static_support_frames':sf,'drawer_support_frames':df,'static_spread':ss,'drawer_spread':ds,'q_span':dq,'evidence_margin':ds-ss,'E_static':sf/np.maximum(ss,1e-6),'E_drawer':df*np.minimum(dq/max(travel,1e-9),1)/np.maximum(ds,1e-6)}

def render_clouds(groups,path,max_points,seed,title):
    rng=np.random.default_rng(seed); fig,ax=plt.subplots(1,3,figsize=(16,5),facecolor='#202124'); views=[(0,2,'XZ'),(0,1,'XY'),(2,1,'ZY')]
    for axes,(a,b,name) in zip(ax,views):
        axes.set_facecolor('#202124')
        for label,pts,color in groups:
            if len(pts)>max_points: pts=pts[np.sort(rng.choice(len(pts),max_points,replace=False))]
            axes.scatter(pts[:,a],pts[:,b],s=.35,c=color,label=label,rasterized=True)
        axes.set_aspect('equal','box'); axes.set_title(name,color='white'); axes.tick_params(colors='white')
    ax[0].legend(markerscale=8); fig.suptitle(title,color='white'); fig.tight_layout(); fig.savefig(path,dpi=180,facecolor=fig.get_facecolor()); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); a=ap.parse_args(); cfg=yaml.safe_load(a.config.read_text()); src=yaml.safe_load(Path(cfg['inputs']['source_config']).read_text()); inp=src['inputs']; th=src['thresholds']; out=Path(cfg['inputs']['output_dir'])
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'non-overwrite: {out}')
    out.mkdir(parents=True); (out/'visualization').mkdir(); (out/'config_resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    manifest=json.loads(Path(cfg['inputs']['reproduction_manifest']).read_text()); v1report=json.loads((Path(cfg['inputs']['v1_fusion_dir'])/'fusion_report.json').read_text()); raw=Path(inp['raw_root']); phases={p:dv.load_phase(Path(inp[f'{p}_manifest'])) for p in ('closed','interaction','open')}; poses=dv.load_odometry(raw/'pinhole_projection/odometry.log'); axis=np.load(inp['axis_path']).astype(float); axis/=np.linalg.norm(axis); q=np.load(inp['q_path']); travel=float(q.max())
    expected=[phases[p][i] for p,i in [('closed',0),('closed',-1),('interaction',0),('interaction',-1),('open',0),('open',-1)]]; recorded=[*manifest['frame_phases']['closed_original_frame_ids'],*manifest['frame_phases']['interaction_original_frame_ids'],*manifest['frame_phases']['open_original_frame_ids']]
    if expected!=recorded or not np.array_equal(axis,np.asarray(manifest['fixed_quantities']['opening_axis_world'])) or travel!=manifest['q_recomputation']['monotonic_range_m'] or not np.array_equal(axis,np.asarray(v1report['motion']['opening_axis_world'])): raise RuntimeError('frozen input conflict')
    audit={'official_compatible_surface':{'artifact_path':str(Path(cfg['inputs']['official_run_root'])/'official/view/surface/surface.ply'),'source_frames':'official interaction input; not closed-only','camera_policy':'HoloLens substitution in MonST3R-compatible slot','moving_map_source':'official refined moving map','partition_method':'first-frame moving-map threshold then 3cm surface distance','whether_reused':False,'reason':'validation explicitly warns background/cabinet over-selection'},'existing_closed_partition':{'artifact_path':None,'whether_reused':False,'reason':'no verified clean closed static/moving partition found'},'geometry_interior_v1':{'artifact_path':cfg['inputs']['v1_fusion_dir'],'source_frames':{p:[x[0],x[-1]] for p,x in phases.items()},'camera_policy':'fixed recorded HoloLens T_world_camera','moving_map_source':inp['moving_labels_path'],'partition_method':'drawer canonical template growth; static remainder inside cabinet bounds','whether_reused':False,'reason':'cabinet_static_points is known contaminated and static is remainder'},'v2_core_policy':'conservative rebuild from closed observations and q≈0 repaired moving seeds'}; (out/'closed_core_source_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    ts=[x.split()[0] for x in (raw/'pinhole_projection/depth.txt').read_text().splitlines() if x.strip()]; rgbpaths=[raw/'pinhole_projection'/x.split(maxsplit=1)[1].replace('\\','/') for x in (raw/'pinhole_projection/rgb.txt').read_text().splitlines() if x.strip()]; fx,fy,cx,cy=np.loadtxt(raw/'pinhole_projection/calibration.txt').reshape(-1)[:4]; k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]); moving=np.load(inp['moving_labels_path'])['a']; handdir=Path(inp['corrected_hand_dir']); closedmask=Path(inp['closed_cabinet_mask_dir'])
    selected=[]
    for phase,sources in phases.items():
        stride=src['sampling'][f'{phase}_stride']
        for local in dv.sampled(len(sources),stride): selected.append((phase,local,sources[local],0. if phase=='closed' else travel if phase=='open' else float(q[local])))
    obs=[]; support_rows=[]; depth_panels=[]
    for phase,local,s,state in selected:
        p=o3d.io.read_point_cloud(str(raw/'Depth Long Throw'/f'{ts[s]}.ply')); xyz,col=map(np.asarray,(p.points,p.colors)); bgr=cv2.imread(str(rgbpaths[s])); rgbmask=dv.rgb_support(bgr,th['rgb_close_radius_pixels'],th['rgb_erosion_radius_pixels']); hmask=np.zeros(rgbmask.shape,bool) if phase!='interaction' else np.load(handdir/f'{s}.npy').squeeze().astype(bool); u,v,inside,visible,consistent,depth=projected_visibility(xyz,poses[s],k,rgbmask.shape,rgbmask,hmask,cfg['validity']['depth_min_m'],cfg['validity']['depth_max_m'],cfg['validity']['depth_consistency_threshold_m'],cfg['validity']['depth_boundary_erosion_pixels']); valid=visible&consistent
        if phase=='closed': cm=np.load(closedmask/f'{local:06d}.npy').squeeze().astype(bool); valid[inside]&=cm[v[inside],u[inside]]
        strong=np.zeros(len(xyz),bool)
        if phase=='interaction': strong[inside]=moving[local,v[inside],u[inside]]==2
        obs.append({'phase':phase,'local':local,'frame':s,'q':state,'xyz':xyz,'color':col,'valid':valid,'visible':visible,'consistent':consistent,'strong':strong,'u':u,'v':v,'rgb':bgr})
        support_rows.append({'original_frame_id':s,'phase':phase,'visible_points':int(visible.sum()),'depth_consistent_points':int((visible&consistent).sum()),'depth_inconsistent_points':int((visible&~consistent).sum())})
        if len(depth_panels)<9:
            ov=bgr.copy(); pix=np.zeros(rgbmask.shape,bool); pix[v[visible],u[visible]]=True; bad=np.zeros(rgbmask.shape,bool); bad[v[visible&~consistent],u[visible&~consistent]]=True; ov[pix]=(ov[pix]*.5+np.array([40,220,40])*.5).astype(np.uint8); ov[bad]=(ov[bad]*.4+np.array([40,40,240])*.6).astype(np.uint8); depth_panels.append(ov)
    with (out/'closed_core_support_per_frame.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=support_rows[0]); w.writeheader(); w.writerows(support_rows)
    # Drawer core: only q≈0 high-confidence repaired moving observations.
    dparts=[]
    for o in obs:
        if o['phase']=='interaction' and o['q']<=cfg['core']['q_zero_max_fraction']*travel:
            sel=o['valid']&o['strong']; dparts.append((o['xyz'][sel]-o['q']*axis,o['color'][sel],np.full(sel.sum(),o['frame'])))
    dx=np.concatenate([x[0] for x in dparts]); dc=np.concatenate([x[1] for x in dparts]); df=np.concatenate([x[2] for x in dparts]); ds=voxel_stats(dx,df,np.zeros(len(df)),np.full(len(df),'interaction'),cfg['core']['voxel_size_m']); dkeep=ds['frames'][ds['inverse']]>=cfg['core']['minimum_unique_frames']; drawer_core=cloud(dx[dkeep],dc[dkeep],cfg['core']['output_voxel_m']); dtree=cKDTree(np.asarray(drawer_core.points))
    # Static core: repeated closed world geometry, positively supported and away from drawer ambiguity.
    sparts=[]; uparts=[]
    for o in obs:
        if o['phase']!='closed': continue
        sel=o['valid']; dist=dtree.query(o['xyz'][sel],workers=-1)[0]; safe=dist>=cfg['core']['drawer_seed_ambiguity_m']; x=o['xyz'][sel]; c=o['color'][sel]; sparts.append((x[safe],c[safe],np.full(safe.sum(),o['frame']))); uparts.append((x[~safe],c[~safe]))
    sx=np.concatenate([x[0] for x in sparts]); sc=np.concatenate([x[1] for x in sparts]); sf=np.concatenate([x[2] for x in sparts]); ss=voxel_stats(sx,sf,np.zeros(len(sf)),np.full(len(sf),'closed'),cfg['core']['voxel_size_m']); skeep=ss['frames'][ss['inverse']]>=cfg['core']['static_minimum_unique_frames']; static_core=cloud(sx[skeep],sc[skeep],cfg['core']['output_voxel_m']); static_tree=cKDTree(np.asarray(static_core.points)); unknown_core=cloud(np.concatenate([sx[~skeep]]+[x[0] for x in uparts]),np.concatenate([sc[~skeep]]+[x[1] for x in uparts]),cfg['core']['output_voxel_m'])
    for name,p in [('static_core',static_core),('drawer_core',drawer_core),('unknown_core',unknown_core)]: o3d.io.write_point_cloud(str(out/f'{name}.ply'),p)
    core_report={'source':'rebuilt conservatively; no existing partition reused','static_points':len(static_core.points),'drawer_points':len(drawer_core.points),'unknown_points':len(unknown_core.points),'drawer_q_zero_max_m':cfg['core']['q_zero_max_fraction']*travel,'static_is_remainder':False}; (out/'closed_core_assignment_report.json').write_text(json.dumps(core_report,indent=2)+'\n')
    # Revealed candidates are valid interaction/open observations not explained by either core.
    allx=[]; allc=[]; allf=[]; allq=[]; allphase=[]; allexps=[]; allexpd=[]; allds=[]; alldd=[]
    core_xyz=np.concatenate((np.asarray(static_core.points),np.asarray(drawer_core.points))); lo=core_xyz.min(0)-cfg['revealed']['region_margin_m']; hi=core_xyz.max(0)+cfg['revealed']['region_margin_m']
    for o in obs:
        if o['phase']=='closed': continue
        sel=o['valid']&np.all((o['xyz']>=lo)&(o['xyz']<=hi),1); x=o['xyz'][sel]; c=o['color'][sel]; can=x-o['q']*axis; dsw=static_tree.query(x,workers=-1)[0]; ddc=dtree.query(can,workers=-1)[0]; es=dsw<=cfg['revealed']['core_explained_distance_m']; ed=ddc<=cfg['revealed']['core_explained_distance_m']; rev=~es&~ed; n=rev.sum(); allx.append(x[rev]); allc.append(c[rev]); allf.append(np.full(n,o['frame'])); allq.append(np.full(n,o['q'])); allphase.append(np.full(n,o['phase'])); allexps.append(es[rev]); allexpd.append(ed[rev]); allds.append(dsw[rev]); alldd.append(ddc[rev])
    x=np.concatenate(allx); c=np.concatenate(allc); frame=np.concatenate(allf); qv=np.concatenate(allq); phase=np.concatenate(allphase); dstatic=np.concatenate(allds); ddrawer=np.concatenate(alldd); canonical=x-qv[:,None]*axis
    results={}
    for vox in cfg['revealed']['hypothesis_voxel_sizes_m']:
        st=voxel_stats(x,frame,qv,phase,vox); dr=voxel_stats(canonical,frame,qv,phase,vox); lab,ev=classify_motion(st,dr,travel,cfg['revealed']); results[f'{int(vox*1000)}mm']={'labels':lab,'evidence':ev,'static_stats':st,'drawer_stats':dr}
    primary=f"{int(cfg['revealed']['primary_voxel_size_m']*1000)}mm"; lab=results[primary]['labels']; ev=results[primary]['evidence']; static_new=lab==LABELS['static_new_world_consistent']; drawer_new=lab==LABELS['drawer_new_canonical_consistent']; unknown=~static_new&~drawer_new
    save_cloud(out/'static_new.ply',x[static_new],c[static_new],cfg['revealed']['output_voxel_m']); save_cloud(out/'drawer_new.ply',canonical[drawer_new],c[drawer_new],cfg['revealed']['output_voxel_m']); save_cloud(out/'unknown_new.ply',x[unknown],c[unknown],cfg['revealed']['output_voxel_m'])
    static_v2=static_core+o3d.io.read_point_cloud(str(out/'static_new.ply')); static_v2=static_v2.voxel_down_sample(cfg['revealed']['output_voxel_m']); drawer_v2=drawer_core+o3d.io.read_point_cloud(str(out/'drawer_new.ply')); drawer_v2=drawer_v2.voxel_down_sample(cfg['revealed']['output_voxel_m']); o3d.io.write_point_cloud(str(out/'static_v2_points.ply'),static_v2); o3d.io.write_point_cloud(str(out/'drawer_v2_canonical_points.ply'),drawer_v2); o3d.io.write_point_cloud(str(out/'unknown_v2_points.ply'),o3d.io.read_point_cloud(str(out/'unknown_new.ply'))+unknown_core)
    reason=np.asarray([next(k for k,v in LABELS.items() if v==int(z)) for z in lab]); dual=(dstatic<cfg['revealed']['dual_close_proximity_m'])&(ddrawer<cfg['revealed']['dual_close_proximity_m'])
    np.savez_compressed(out/'observation_provenance.npz',original_frame_id=frame,phase=phase,q_t=qv,world_xyz=x,canonical_xyz=canonical,visible=np.ones(len(x),bool),depth_consistent=np.ones(len(x),bool),explained_by_static_core=np.zeros(len(x),bool),explained_by_drawer_core=np.zeros(len(x),bool),static_support_frames=ev['static_support_frames'],drawer_support_frames=ev['drawer_support_frames'],static_spread=ev['static_spread'],drawer_spread=ev['drawer_spread'],q_span=ev['q_span'],E_static=ev['E_static'],E_drawer=ev['E_drawer'],classification_margin=ev['evidence_margin'],final_label=lab,classification_reason=reason,dual_close_region=dual)
    def summary(mask):
        n=max(mask.sum(),1); return {'observations':int(mask.sum()),'static_new_ratio':float((static_new&mask).sum()/n),'drawer_new_ratio':float((drawer_new&mask).sum()/n),'unknown_new_ratio':float((unknown&mask).sum()/n),'E_static_p50':float(np.median(ev['E_static'][mask])) if mask.any() else None,'E_drawer_p50':float(np.median(ev['E_drawer'][mask])) if mask.any() else None,'q_span_p50_m':float(np.median(ev['q_span'][mask])) if mask.any() else None,'support_frames_p50':{'static':float(np.median(ev['static_support_frames'][mask])) if mask.any() else None,'drawer':float(np.median(ev['drawer_support_frames'][mask])) if mask.any() else None},'classification_margin_p50_m':float(np.median(ev['evidence_margin'][mask])) if mask.any() else None}
    comparison={'v1':v1report['points'],'closed_core_only':{'static':len(static_core.points),'drawer_closed':len(drawer_core.points),'unknown':len(unknown_core.points)},'assignment_v2':{'static':len(static_v2.points),'drawer_closed':len(drawer_v2.points),'unknown':len((o3d.io.read_point_cloud(str(out/'unknown_v2_points.ply'))).points),'static_new':len(o3d.io.read_point_cloud(str(out/'static_new.ply')).points),'drawer_new':len(o3d.io.read_point_cloud(str(out/'drawer_new.ply')).points)},'per_phase':{p:{'static_new':int((static_new&(phase==p)).sum()),'drawer_new':int((drawer_new&(phase==p)).sum()),'unknown':int((unknown&(phase==p)).sum())} for p in ('interaction','open')},'dual_close_region':summary(dual),'ablations':{key:{'static_new':int((r['labels']==2).sum()),'drawer_new':int((r['labels']==3).sum()),'unknown':int(((r['labels']!=2)&(r['labels']!=3)).sum())} for key,r in results.items()}}
    old_s=o3d.io.read_point_cloud(str(Path(cfg['inputs']['v1_fusion_dir'])/'cabinet_static_points.ply')); old_d=o3d.io.read_point_cloud(str(Path(cfg['inputs']['v1_fusion_dir'])/'drawer_canonical_closed_points.ply')); vt=[cKDTree(np.asarray(static_v2.points)),cKDTree(np.asarray(drawer_v2.points)),cKDTree(np.asarray(o3d.io.read_point_cloud(str(out/'unknown_v2_points.ply')).points))]
    def fate(pc):
        z=np.asarray(pc.points); ds=[t.query(z,workers=-1)[0] for t in vt]; a=np.argmin(np.stack(ds),0); mind=np.min(np.stack(ds),0); a[mind>cfg['revealed']['core_explained_distance_m']]=3; return {'static':int((a==0).sum()),'drawer':int((a==1).sum()),'unknown':int((a==2).sum()),'invalid_or_unmatched':int((a==3).sum())}
    comparison['old_static_fate']=fate(old_s); comparison['old_drawer_fate']=fate(old_d); (out/'assignment_v2_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    cv2.imwrite(str(out/'visualization'/'depth_consistency_overlay_9grid.jpg'),np.vstack([np.hstack(depth_panels[i:i+3]) for i in range(0,9,3)]))
    mp=cfg['visualization']['max_points_per_cloud']; seed=cfg['visualization']['seed']; render_clouds([('static_core',np.asarray(static_core.points),'#3388ff'),('drawer_core',np.asarray(drawer_core.points),'#ff3344'),('unknown_core',np.asarray(unknown_core.points),'#ffcc22')],out/'visualization'/'01_closed_cores_xyz.png',mp,seed,'Closed clean cores'); render_clouds([('static_new',x[static_new],'#3388ff'),('drawer_new',canonical[drawer_new],'#ff3344'),('unknown_new',x[unknown],'#ffcc22')],out/'visualization'/'02_revealed_assignment_xyz.png',mp,seed,'Revealed motion-consistency assignment'); render_clouds([('static_v2',np.asarray(static_v2.points),'#3388ff'),('drawer_v2',np.asarray(drawer_v2.points),'#ff3344')],out/'visualization'/'03_final_v2_xyz.png',mp,seed,'Final assignment v2'); render_clouds([('dual_close',x[dual],'#cc44ff')],out/'visualization'/'04_dual_close_region_xyz.png',mp,seed,'Automatic dual-close region'); render_clouds([('old_static',np.asarray(old_s.points),'#999999'),('new_static',np.asarray(static_v2.points),'#3388ff')],out/'visualization'/'05_old_vs_new_static_xyz.png',mp,seed,'Old static vs v2 static'); render_clouds([('old_drawer',np.asarray(old_d.points),'#999999'),('new_drawer',np.asarray(drawer_v2.points),'#ff3344')],out/'visualization'/'06_old_vs_new_drawer_xyz.png',mp,seed,'Old drawer vs v2 drawer')
    hashes={p.name:sha(p) for p in sorted(out.glob('*points.ply'))}; deterministic=hashes=={p.name:sha(p) for p in sorted(out.glob('*points.ply'))}; final={'tsdf_ran':False,'nksr_ran':False,'mesh_ran':False,'frozen_inputs_verified':True,'primary_voxel':primary,'output_file_rehash_verified':deterministic,'full_deterministic_rerun_performed':False,'hashes':hashes,'ready_for_dual_tsdf':False,'readiness_reason':'requires human/geometric review; precision criteria not auto-asserted','comparison':comparison}; (out/'RUN_REPORT.json').write_text(json.dumps(final,indent=2)+'\n'); print(json.dumps(final,indent=2))
if __name__=='__main__': main()
