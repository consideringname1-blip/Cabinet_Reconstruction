#!/usr/bin/env python3
"""Prepare closed, interaction, reverse-AutoSeg, and open HoloLens sequences."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def rgb_paths(pinhole: Path) -> list[Path]:
    paths = []
    for line in (pinhole / "rgb.txt").read_text().splitlines():
        if line.strip():
            paths.append(pinhole / line.split(maxsplit=1)[1].replace("\\", "/"))
    return paths


def make_sequence(
    root: Path,
    name: str,
    source_indices: list[int],
    images: list[Path],
    source_root: Path,
) -> dict:
    sequence = root / "inputs" / name
    image_dir = sequence / "jpg"
    image_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    for local_index, source_index in enumerate(source_indices):
        source = images[source_index].resolve()
        target = image_dir / f"{local_index:06d}.jpg"
        if target.is_symlink():
            if target.resolve() != source:
                raise RuntimeError(f"Conflicting link: {target}")
        elif target.exists():
            raise FileExistsError(target)
        else:
            os.symlink(source, target)
        mapping.append(
            {
                "local_index": local_index,
                "source_index": source_index,
                "rgb": str(source),
                "linked_rgb": str(target.resolve()),
            }
        )
    report = {
        "name": name,
        "frame_count": len(mapping),
        "source_indices": source_indices,
        "source_root": str(source_root.resolve()),
        "frame_order": "ascending" if source_indices[0] < source_indices[-1] else "descending",
        "mapping": mapping,
    }
    (sequence / "frame_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    (sequence / "sequence_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--closed-start", type=int, default=5)
    parser.add_argument("--closed-end", type=int, default=165)
    parser.add_argument("--interaction-start", type=int, default=177)
    parser.add_argument("--interaction-end", type=int, default=213)
    parser.add_argument("--open-start", type=int, default=220)
    parser.add_argument("--open-end", type=int, default=360)
    args = parser.parse_args()

    pinhole = args.source_root / "pinhole_projection"
    images = rgb_paths(pinhole)
    phases = [
        (
            f"closed_forward_{args.closed_start:03d}_{args.closed_end:03d}",
            list(range(args.closed_start, args.closed_end + 1)),
        ),
        (
            f"interaction_forward_{args.interaction_start:03d}_{args.interaction_end:03d}",
            list(range(args.interaction_start, args.interaction_end + 1)),
        ),
        (
            f"autoseg_reverse_{args.interaction_end:03d}_{args.interaction_start:03d}",
            list(range(args.interaction_end, args.interaction_start - 1, -1)),
        ),
        (
            f"open_forward_{args.open_start:03d}_{args.open_end:03d}",
            list(range(args.open_start, args.open_end + 1)),
        ),
    ]
    reports = [
        make_sequence(args.run_root, name, indices, images, args.source_root)
        for name, indices in phases
    ]
    manifest = {
        "purpose": "HoloLens-GT iTACO plus articulation-aware closed/open dual-volume fusion",
        "source": str(args.source_root.resolve()),
        "substitution": "HoloLens odometry/world PLY depth replace MonST3R and PromptDA",
        "phases": reports,
        "known_rgb_depth_sync_exclusions_for_open_fusion": [
            248,
            265,
            314,
            318,
            328,
            331,
            342,
        ],
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "INPUT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({item["name"]: item["frame_count"] for item in reports}, indent=2))


if __name__ == "__main__":
    main()
