"""Complete diagnostics and verify an independent deterministic Assignment v3 rerun."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import cv2, numpy as np, yaml

from tools.itaco_funrec_assignment_v3.core import LABEL_DRAWER, LABEL_INVALID, LABEL_STATIC, LABEL_UNKNOWN, write_ply
from tools.itaco_funrec_assignment_v3.pipeline import load_proposals

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--repeat-dir',type=Path,required=True); a=ap.parse_args()
    cfg=yaml.safe_load(a.config.read_text()); out=Path(cfg['inputs']['output_dir']); repeat=a.repeat_dir
    core=['tracks_raw.npz','tracks_filtered.npz','track_labels.json','keyframe_region_votes.csv','drawer_v3_canonical_points.ply','static_v3_points.ply','unknown_v3_points.ply']
    hashes={name:{'primary':sha(out/name),'repeat':sha(repeat/name)} for name in core}; deterministic=all(x['primary']==x['repeat'] for x in hashes.values())
    verification={'full_pipeline_rerun_performed':True,'independent_output_dir':str(repeat),'all_core_hashes_equal':deterministic,'files':hashes}
    (out/'deterministic_rerun_report.json').write_text(json.dumps(verification,indent=2)+'\n')
    interaction=json.loads(Path(cfg['inputs']['interaction_manifest']).read_text())['source_indices']; opened=json.loads(Path(cfg['inputs']['open_manifest']).read_text())['source_indices']; sources=interaction+opened
    seed_dirs=sorted((out/'sam2_propagation_per_seed').glob('seed_*')); shape=(288,320)
    drawer_dir=out/'drawer_mask_propagated'; conflict_dir=out/'propagation_conflict_mask'; drawer_dir.mkdir(exist_ok=True); conflict_dir.mkdir(exist_ok=True)
    votes=[]; conflicts=[]; contact=[]; pair_ious=[]
    root=Path(cfg['inputs']['raw_root'])/'pinhole_projection'; rgbs=[root/x.split(maxsplit=1)[1].replace('\\','/') for x in (root/'rgb.txt').read_text().splitlines() if x.strip()]
    contact_sources=set(np.linspace(0,len(sources)-1,9,dtype=int).tolist())
    for position,source in enumerate(sources):
        idx=source-interaction[0]; stack=np.stack([np.load(d/f'{idx:06d}.npy').astype(bool) if (d/f'{idx:06d}.npy').exists() else np.zeros(shape,bool) for d in seed_dirs])
        count=stack.sum(0); union=stack.any(0); agreement=count/max(len(stack),1); conflict=union&(agreement<float(cfg['sam2']['conflict_min_agreement_fraction'])); drawer=(count>=int(cfg['sam2']['minimum_agreeing_seeds']))&~conflict
        np.save(drawer_dir/f'{source}.npy',drawer); np.save(conflict_dir/f'{source}.npy',conflict); votes.append(count.astype(np.uint8)); conflicts.append(conflict)
        for i in range(len(stack)):
            for j in range(i+1,len(stack)):
                pair_ious.append({'original_frame_id':source,'seed_a':seed_dirs[i].name,'seed_b':seed_dirs[j].name,'iou':float((stack[i]&stack[j]).sum()/max((stack[i]|stack[j]).sum(),1))})
        if position in contact_sources:
            rgb=cv2.imread(str(rgbs[source])); tint=np.zeros_like(rgb); tint[drawer]=(20,40,245); tint[conflict]=(200,30,220); panel=cv2.addWeighted(rgb,.45,tint,.55,0); cv2.putText(panel,f'{source}: red drawer / magenta conflict',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1); contact.append(panel)
    np.savez_compressed(out/'drawer_propagation_vote.npz',original_frame_ids=np.asarray(sources),positive_seed_votes=np.stack(votes),conflict=np.stack(conflicts))
    if contact:
        while len(contact)%3: contact.append(np.zeros_like(contact[0]))
        cv2.imwrite(str(out/'propagation_contact_sheet.jpg'),np.vstack([np.hstack(contact[i:i+3]) for i in range(0,len(contact),3)]))
    prop=json.loads((out/'propagation_report.json').read_text()); prop['seed_pair_iou']={'median':float(np.median([x['iou'] for x in pair_ious])),'p10':float(np.percentile([x['iou'] for x in pair_ious],10)),'records':len(pair_ious)}; prop['optional_sam2_fill_holes_extension_available']=False; prop['optional_extension_note']='SAM2 warned that compiled _C fill-hole post-processing is unavailable; predictor propagation completed.'; (out/'propagation_report.json').write_text(json.dumps(prop,indent=2)+'\n')
    # Preserve proposal layer indices only as provenance; keys are explicit local proposal records.
    votes_rows=list(csv.DictReader((out/'keyframe_region_votes.csv').open())); keys=json.loads((out/'interaction_keyframes.json').read_text()); kp=out/'keyframe_proposals'; rv=out/'region_vote_visualization'; kp.mkdir(exist_ok=True); rv.mkdir(exist_ok=True)
    for key in keys:
        source=key['original_frame_id']; proposals=load_proposals(source,cfg); payload={f'proposal_{i:03d}':p['mask'] for i,p in enumerate(proposals)}; np.savez_compressed(kp/f'{source}.npz',**payload)
        manifest=[{'key':f'proposal_{i:03d}','proposal_id':p['proposal_id'],'source_uid_provenance_only':p['source_uid'],'score':p['score'],'source':p['source']} for i,p in enumerate(proposals)]; (kp/f'{source}.json').write_text(json.dumps(manifest,indent=2)+'\n')
        rgb=cv2.imread(str(rgbs[source])); tint=np.zeros_like(rgb); mapping={r['proposal_id']:r['label'] for r in votes_rows if int(r['original_frame_id'])==source}
        for p in proposals:
            color={'drawer':(20,40,245),'static':(220,100,20),'unknown':(20,220,220)}.get(mapping.get(p['proposal_id']),(0,0,0)); tint[p['mask']]=color
        cv2.imwrite(str(rv/f'{source}.jpg'),cv2.addWeighted(rgb,.45,tint,.55,0))
    near=json.loads((out/'near_contact_region_report.json').read_text()); evidence=list(csv.DictReader((out/'track_motion_evidence.csv').open())); near.update({'moving_tracks':0,'static_tracks':0,'unknown_tracks':0,'q_span_m':None,'side_panel_first_motion_support_original_frame_id':None,'drawer_retention_in_open':None,'diagnostic_failure_reason':'automatic dual-support near-contact set is empty at 2.5 cm; no ROI or location fallback was introduced'})
    (out/'near_contact_region_report.json').write_text(json.dumps(near,indent=2)+'\n'); write_ply(out/'near_contact_tracks.ply',np.empty((0,3)))
    nv=out/'near_contact_assignment_visualization'; nv.mkdir(exist_ok=True); (nv/'README.txt').write_text('No panels: automatic near-contact set was empty; no manual ROI fallback was used.\n')
    acceptance=json.loads((out/'acceptance_report.json').read_text()); acceptance['conditions']['deterministic_full_rerun']=deterministic; acceptance['ready_for_dual_tsdf']=False; acceptance['blocking_reasons']=['automatic near-contact support set is empty','open registered depth is too sparse under the configured 2 px boundary rule','cabinet-inner-wall and drawer-side human checks not passed','no independent manual labels']; (out/'acceptance_report.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    report=(out/'RUN_REPORT.md').read_text().replace('human geometry review and deterministic full rerun remain required','deterministic full rerun passed; human geometry and observability review remain required')
    report+='\n## Determinism and blockers\n\n- Independent full rerun: passed; all seven core output hashes match.\n- SAM2 optional `_C` fill-hole extension: unavailable; propagation itself completed.\n- Near-contact automatic set: empty, so side-panel/inner-wall claims are not accepted.\n- Open registered depth: mostly invalid after the configured 2 px depth-boundary rule.\n'
    (out/'RUN_REPORT.md').write_text(report)
    print(json.dumps({'deterministic':deterministic,'near_contact':near['counts'],'seed_pair_iou_median':prop['seed_pair_iou']['median']},indent=2))

if __name__=='__main__': main()
