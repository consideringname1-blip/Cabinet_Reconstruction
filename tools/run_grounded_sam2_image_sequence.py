#!/usr/bin/env python3
"""Run local GroundingDINO + SAM2 image segmentation on an image sequence.

This keeps the official Grounded-SAM-2 model calls, but re-detects the named
object in every frame so a large camera orbit cannot accumulate tracker drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-prompt", default="cabinet")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_dir.resolve()
    sys.path.insert(0, str(repo))

    from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    grounding_model = load_model(
        str(repo / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        str(repo / "gdino_checkpoints/groundingdino_swint_ogc.pth"),
        device=device,
    )
    sam_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(repo / "checkpoints/sam2.1_hiera_large.pt"),
        device=device,
    )
    sam_predictor = SAM2ImagePredictor(sam_model)

    images = sorted(
        [*args.image_dir.glob("*.jpg"), *args.image_dir.glob("*.png")],
        key=lambda path: int(path.stem),
    )
    if not images:
        raise FileNotFoundError(args.image_dir)

    mask_dir = args.output_dir / "mask"
    annotated_dir = args.output_dir / "annotated"
    mask_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for index, image_path in enumerate(images):
        image_source, image_tensor = load_image(str(image_path))
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image_tensor,
            caption=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=device,
        )
        if args.max_detections > 0 and len(boxes) > args.max_detections:
            keep = torch.argsort(confidences, descending=True)[: args.max_detections]
            boxes = boxes[keep]
            confidences = confidences[keep]
            labels = [labels[int(i)] for i in keep]
        height, width = image_source.shape[:2]
        if len(boxes):
            boxes_px = boxes * torch.tensor([width, height, width, height])
            boxes_xyxy = box_convert(boxes_px, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            sam_predictor.set_image(image_source)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                masks, scores, _ = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes_xyxy,
                    multimask_output=False,
                )
            masks = masks.squeeze(1) if masks.ndim == 4 else masks
            combined = np.any(masks.astype(bool), axis=0)
            sam_scores = np.asarray(scores).reshape(-1).astype(float).tolist()
        else:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            combined = np.zeros((height, width), dtype=bool)
            sam_scores = []

        np.save(mask_dir / f"{index:06d}.npy", combined)
        cv2.imwrite(str(mask_dir / f"{index:06d}.png"), combined.astype(np.uint8) * 255)

        bgr = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        overlay[combined] = (0, 180, 255)
        annotated = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 0)
        for box, label, confidence in zip(boxes_xyxy, labels, confidences.tolist()):
            x0, y0, x1, y1 = np.rint(box).astype(int)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 1)
            cv2.putText(
                annotated,
                f"{label} {confidence:.2f}",
                (max(0, x0), max(12, y0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(annotated_dir / f"{index:06d}.jpg"), annotated)
        records.append(
            {
                "local_index": index,
                "image": str(image_path.resolve()),
                "prompt": args.text_prompt,
                "labels": labels,
                "grounding_confidences": confidences.numpy().astype(float).tolist(),
                "sam_scores": sam_scores,
                "mask_pixels": int(combined.sum()),
                "mask_fraction": float(combined.mean()),
            }
        )
        print(
            f"[{index + 1:03d}/{len(images):03d}] boxes={len(boxes_xyxy)} "
            f"mask_fraction={combined.mean():.4f}"
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "implementation": "official local GroundingDINO + SAM2 image predictor",
                "prompt": args.text_prompt,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "max_detections": args.max_detections,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
