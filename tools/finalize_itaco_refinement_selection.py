#!/usr/bin/env python3
"""Select the lower-loss iTACO refinement and render its moving maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_report(result_dir: Path, joint_type: str) -> dict:
    report_path = result_dir / joint_type / "gt_native_refinement_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report["report_path"] = str(report_path)
    return report


def render_contact_sheet(rgb_dir: Path, moving_maps: np.ndarray, output: Path) -> None:
    rgb_paths = sorted(
        path for path in rgb_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(rgb_paths) != len(moving_maps):
        raise ValueError(f"RGB/moving-map count mismatch: {len(rgb_paths)} vs {len(moving_maps)}")

    cells = []
    for frame_index, (rgb_path, moving_map) in enumerate(zip(rgb_paths, moving_maps)):
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read {rgb_path}")
        if moving_map.shape != bgr.shape[:2]:
            moving_map = cv2.resize(
                moving_map.astype(np.float32),
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        weight = np.clip(moving_map, 0.0, 1.0)[..., None]
        heat = np.zeros_like(bgr, dtype=np.float32)
        heat[..., 2] = 255.0
        overlay = bgr.astype(np.float32) * (1.0 - 0.55 * weight) + heat * (0.55 * weight)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        cv2.putText(
            overlay,
            f"local {frame_index:02d} / source {97 + frame_index}",
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(overlay)

    cell_h, cell_w = cells[0].shape[:2]
    cols = 5
    rows = (len(cells) + cols - 1) // cols
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for index, cell in enumerate(cells):
        row, col = divmod(index, cols)
        canvas[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = cell
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    refinement_dir = (
        args.run_root
        / "official/prediction/refinement/monst3r/chamfer/0"
    )
    reports = {
        joint_type: load_report(refinement_dir, joint_type)
        for joint_type in ("revolute", "prismatic")
    }
    selected_type = min(reports, key=lambda name: float(reports[name]["best_loss"]))
    selected = reports[selected_type]
    selection = {
        "selection_rule": "minimum best Chamfer loss across the two official refinement hypotheses",
        "selected_joint_type": selected_type,
        "selected_best_loss": selected["best_loss"],
        "selected_joint_axis": selected["joint_axis"],
        "selected_joint_pos": selected["joint_pos"],
        "hypotheses": {
            joint_type: {
                "best_loss": report["best_loss"],
                "joint_axis": report["joint_axis"],
                "joint_pos": report["joint_pos"],
                "report_path": report["report_path"],
            }
            for joint_type, report in reports.items()
        },
        "loss_gap_revolute_minus_prismatic": (
            float(reports["revolute"]["best_loss"])
            - float(reports["prismatic"]["best_loss"])
        ),
    }
    selection_path = refinement_dir / "selected_hypothesis.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
        handle.write("\n")

    validation_dir = args.run_root / "validation"
    for joint_type in reports:
        moving_maps = np.load(refinement_dir / joint_type / "moving_map.npz")["a"]
        render_contact_sheet(
            args.run_root / "official/view/rgb",
            moving_maps,
            validation_dir / f"refined_{joint_type}_moving_map_all19.jpg",
        )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
