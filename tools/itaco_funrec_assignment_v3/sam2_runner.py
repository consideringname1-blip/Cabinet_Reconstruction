"""Run the audited local SAM2.1 video predictor for independent seed masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    sys.path.insert(0, manifest["sam2_repo"])
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(manifest["model_config"], manifest["checkpoint"], device="cuda")
    state = predictor.init_state(video_path=manifest["video_dir"])
    output = Path(manifest["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    report = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for seed in manifest["seeds"]:
            predictor.reset_state(state)
            mask = np.load(seed["mask_path"]).astype(bool)
            predictor.add_new_mask(state, frame_idx=int(seed["combined_index"]), obj_id=1, mask=mask)
            results: dict[int, np.ndarray] = {}
            for reverse in (False, True):
                for frame_idx, _, logits in predictor.propagate_in_video(state, reverse=reverse):
                    results[int(frame_idx)] = (logits[0, 0] > 0).cpu().numpy()
            seed_dir = output / seed["seed_id"]; seed_dir.mkdir(exist_ok=True)
            coverage = {}
            for index, mask_out in sorted(results.items()):
                np.save(seed_dir / f"{index:06d}.npy", mask_out)
                coverage[str(index)] = int(mask_out.sum())
            report.append({"seed_id": seed["seed_id"], "frames": len(results), "coverage_px": coverage})
    (output / "sam2_raw_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
