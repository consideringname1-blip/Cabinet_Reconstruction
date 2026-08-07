#!/usr/bin/env python3
"""Create diagnostic review sheets from an already completed v5 116 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.itaco_ownership_v5.real_diagnostic import contact_sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text()); output = Path(cfg["output_dir"])
    if not (output / "v5_116_summary.json").is_file():
        raise FileNotFoundError("completed v5 summary missing")
    raw_moving_blocked = []; non_tangent = []; other = []
    rows = json.loads((output / "per_region_summary.json").read_text())
    for row in rows:
        directory = Path(row["region_directory"])
        result = json.loads((directory / "v5_region_diagnostic.json").read_text())
        static = result["verified_active"]["static"]; drawer = result["verified_active"]["drawer"]
        moving_cfg = cfg["motion_evidence"]
        raw = (drawer["count"] >= int(moving_cfg["minimum_verified_active_observations"]) and
               drawer["q_span_m"] >= float(moving_cfg["minimum_verified_q_span_m"]) and
               drawer["fraction"] - static["fraction"] >= float(moving_cfg["minimum_verified_fraction_advantage"]))
        image = directory / "source_region.jpg"
        if raw and result["source_finite_surface_observability"]["tangent_motion_ambiguity"]:
            raw_moving_blocked.append(image)
        elif not result["source_finite_surface_observability"]["tangent_motion_ambiguity"]:
            non_tangent.append(image)
        else:
            other.append(image)
    viz = cfg["visualization"]; columns = int(viz["contact_sheet_columns"]); width = int(viz["contact_sheet_thumbnail_width"])
    target = output / "visualization"
    contact_sheet(raw_moving_blocked, target / "raw_moving_blocked_tangent_ambiguity.jpg", columns, width,
                  "Raw moving route rejected: tangent surface + unobservable finite boundary")
    contact_sheet(non_tangent, target / "non_tangent_without_positive_event.jpg", columns, width,
                  "Non-tangent regions: no positive motion or causal disocclusion event")
    contact_sheet(other, target / "other_unknown_regions.jpg", columns, width,
                  "Other unknown regions")
    payload = {"raw_moving_blocked_tangent_ambiguity": len(raw_moving_blocked),
               "non_tangent_without_positive_event": len(non_tangent),
               "other_unknown": len(other)}
    (output / "review_category_counts.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
