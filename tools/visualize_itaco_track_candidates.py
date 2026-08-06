#!/usr/bin/env python3
"""Visualize selected AutoSeg track IDs across forward interaction frames."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--autoseg-dir", type=Path, required=True)
    parser.add_argument("--hand-dir", type=Path, required=True)
    parser.add_argument("--uids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    uids = [int(value) for value in args.uids.split(",") if value.strip()]
    rgb_files = sorted(args.rgb_dir.glob("*.jpg"))
    mask_files = sorted(args.autoseg_dir.glob("mask_*.npz"))
    hand_files = sorted(args.hand_dir.glob("*.npy"))
    frame_ids = [0, 3, 6, 9, 12, 15, 18]
    width, height = Image.open(rgb_files[0]).size
    canvas = Image.new("RGB", (width * len(frame_ids), height * len(uids)), "black")

    for row, uid in enumerate(uids):
        for column, frame_id in enumerate(frame_ids):
            bgr = cv2.imread(str(rgb_files[frame_id]))
            masks = np.load(mask_files[len(mask_files) - 1 - frame_id])["a"][:, 0]
            hand = np.load(hand_files[frame_id]).squeeze().astype(bool)
            mask = masks[uid].astype(bool) & (~hand)
            overlay = bgr.copy()
            overlay[mask] = (0, 0, 255)
            annotated = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
            cv2.putText(
                annotated,
                f"uid {uid} src {97 + frame_id}",
                (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tile = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            canvas.paste(tile, (column * width, row * height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)


if __name__ == "__main__":
    main()
