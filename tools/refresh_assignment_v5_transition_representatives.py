#!/usr/bin/env python3
"""Refresh selected control visualizations without rerunning the control gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.itaco_ownership_v5.transition_diagnostic import (
    _find_proposal,
    _load_runtime,
    analyze_region,
    causal_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output = Path(cfg["output_dir"]); review = Path(cfg["review_bundle_dir"])
    gate = json.loads((output / "control_evaluation_report.json").read_text())
    summary = json.loads((output / "transition_event_summary.json").read_text())
    if gate["passed"] or summary["full_116_ran"]:
        raise RuntimeError("this refresh is restricted to a failed control gate before 116")
    annotations = json.loads(Path(cfg["inputs"]["control_annotations"]).read_text())["controls"]
    runtime = _load_runtime(cfg)
    first_by_group = {}
    for ordinal, annotation in enumerate(annotations):
        first_by_group.setdefault(annotation["control_group"], (ordinal, annotation))
    groups = ("drawer_front", "active_comoving_box", "documented_floor_false_positive", "plateau_only")
    for group in groups:
        ordinal, annotation = first_by_group[group]
        frame = runtime["frames_by_id"][int(annotation["original_frame_id"])]
        proposal = _find_proposal(frame, annotation["proposal_id"])
        result = analyze_region(frame, proposal, runtime, cfg,
                                annotation.get("evaluation_mode", "active_transition"))
        local = output / "controls" / group / f"{ordinal:03d}_{proposal['proposal_id']}" / "causal_reveal.jpg"
        tracked = review / "representatives" / group / "causal_reveal.jpg"
        causal_panel(frame, result, runtime, local)
        causal_panel(frame, result, runtime, tracked)
        print(f"refreshed {group}: {local} and {tracked}")
    (review / "representative_refresh.json").write_text(json.dumps({
        "control_gate_reran": False,
        "control_decisions_modified": False,
        "full_116_ran": False,
        "purpose": "add measured reveal-band/new-support overlays and legend",
        "groups": list(groups),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
