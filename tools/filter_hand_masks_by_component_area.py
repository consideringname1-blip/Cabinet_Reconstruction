#!/usr/bin/env python3
"""Preserve Grounded-SAM-2 masks while removing implausibly giant components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-component-fraction", type=float, default=0.08)
    args = parser.parse_args()
    mask_out = args.output_root / "mask"
    annotated_out = args.output_root / "annotated"
    mask_out.mkdir(parents=True, exist_ok=True)
    annotated_out.mkdir(parents=True, exist_ok=True)

    masks = sorted((args.raw_root / "mask").glob("*.npy"))
    images = sorted(args.image_dir.glob("*.jpg"))
    if len(masks) != len(images):
        raise ValueError((len(masks), len(images)))
    records = []
    for index, (mask_path, image_path) in enumerate(zip(masks, images)):
        raw = np.load(mask_path).squeeze().astype(bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            raw.astype(np.uint8), 8
        )
        corrected = np.zeros_like(raw)
        components = []
        for label in range(1, count):
            pixels = int(stats[label, cv2.CC_STAT_AREA])
            fraction = pixels / raw.size
            kept = fraction <= args.max_component_fraction
            if kept:
                corrected |= labels == label
            components.append(
                {"label": label, "pixels": pixels, "fraction": fraction, "kept": kept}
            )
        np.save(mask_out / f"{index:06d}.npy", corrected)
        cv2.imwrite(
            str(mask_out / f"{index:06d}.png"),
            corrected.astype(np.uint8) * 255,
        )
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        overlay = bgr.copy()
        overlay[raw & ~corrected] = (0, 0, 255)
        overlay[corrected] = (0, 200, 255)
        annotated = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 0)
        cv2.putText(
            annotated,
            f"yellow=kept red=rejected raw={raw.mean():.3f} fixed={corrected.mean():.3f}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(annotated_out / f"{index:06d}.jpg"), annotated)
        records.append(
            {
                "local_index": index,
                "raw_fraction": float(raw.mean()),
                "corrected_fraction": float(corrected.mean()),
                "components": components,
            }
        )
    report = {
        "method": "connected-component area gate on preserved Grounded-SAM-2 output",
        "raw_root": str(args.raw_root.resolve()),
        "max_component_fraction": args.max_component_fraction,
        "meaning": "Components larger than the plausible hand/arm area are rejected; no new hand pixels are invented.",
        "records": records,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "frames": len(records),
                "changed_frames": sum(
                    r["raw_fraction"] != r["corrected_fraction"] for r in records
                ),
                "raw_fraction_max": max(r["raw_fraction"] for r in records),
                "corrected_fraction_max": max(
                    r["corrected_fraction"] for r in records
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
