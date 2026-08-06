"""Post-run semantic evaluation; never imported by the optimizer."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

LABEL_NAMES={0:"static",1:"unknown",2:"moving",3:"invalid"}
MODES=("official","gate_only","gate_no_minmax")

def evaluate(root: Path, annotation_path: Path) -> dict:
    annotation=json.loads(annotation_path.read_text())
    start=annotation["frame_mapping"]["processing_index_0_source_frame"]
    data={}
    for mode in MODES:
        directory=root/mode
        data[mode]={"raw":np.load(directory/"moving_map_raw.npz")["a"],"score":np.load(directory/"moving_map_gated.npz")["a"],"labels":np.load(directory/"moving_map_labels.npz")["a"],"proposal_scores":np.load(directory/"proposal_scores_final.npy")}
    support=np.load(root/"gate_no_minmax"/"valid_support.npz")["sensor_support"]
    coverage=np.load(root/"gate_no_minmax"/"proposal_coverage.npz")["a"]
    records=[]
    for item in annotation["annotations"]:
        frame=item["original_frame_id"]-start; u,v=item["pixel_uv"]
        record={**item,"processing_index":frame,"sensor_support":bool(support[frame,v,u]),"proposal_covered":bool(coverage[frame,v,u]),"modes":{}}
        for mode,current in data.items():
            raw=float(current["raw"][frame,v,u]); ids=[]
            if record["proposal_covered"]:
                delta=np.abs(current["proposal_scores"]-raw); ids=np.flatnonzero(np.isclose(delta,delta.min(),atol=1e-10)).tolist()
            record["modes"][mode]={"raw_score":raw,"final_pixel_score":float(current["score"][frame,v,u]),"output_label":LABEL_NAMES[int(current["labels"][frame,v,u])],"candidate_proposal_ids":ids}
        records.append(record)
    summary={}
    for description in sorted({r["object_description"] for r in records}):
        subset=[r for r in records if r["object_description"]==description]; summary[description]={}
        for mode in MODES:
            summary[description][mode]={"count":len(subset),"raw_score_mean":float(np.mean([r["modes"][mode]["raw_score"] for r in subset])),"final_score_mean":float(np.mean([r["modes"][mode]["final_pixel_score"] for r in subset])),"labels":[r["modes"][mode]["output_label"] for r in subset],"proposal_ids":sorted({i for r in subset for i in r["modes"][mode]["candidate_proposal_ids"]})}
    report={"purpose":"post-run independent semantic evaluation; annotations were not loaded by optimization","annotation_file":str(annotation_path.resolve()),"records":records,"semantic_summary":summary}
    (root/"manual_evaluation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (root/"proposal_semantic_analysis.json").write_text(json.dumps(summary,indent=2)+"\n")
    return report

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",type=Path,required=True); parser.add_argument("--annotations",type=Path,required=True); args=parser.parse_args()
    print(json.dumps(evaluate(args.output_root,args.annotations)["semantic_summary"],indent=2))

if __name__=="__main__": main()
