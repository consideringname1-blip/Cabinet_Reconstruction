#!/usr/bin/env python3
"""Generate SAM3 masks for a HoloLens pinhole_projection upload.

This is intentionally small and reproducible: it runs SAM3 image prompting on
each RGB frame and writes three mask streams that can be consumed by the local
iTACO/V2A experiments:

  - object: the whole articulated object / cabinet region
  - moving: the likely moving part
  - hand: hand pixels to exclude from geometry matching
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_INPUT = Path("/workspace_whz/data/upload/2026-07-27-175228/pinhole_projection")
DEFAULT_OUTPUT = Path("/workspace_whz/data/output/itaco_gt_upload/2026-07-27-175228_sam3_formal_masks")
SAM3_ROOT = Path("/workspace_whz/code/reconstruction/sam3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="0 means all frames")
    parser.add_argument("--object-prompt", default="cabinet")
    parser.add_argument("--moving-prompt", default="cabinet drawer")
    parser.add_argument("--hand-prompt", default="hand")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overlay-every", type=int, default=10)
    return parser.parse_args()


def sorted_rgb_files(input_dir: Path) -> list[Path]:
    rgb_dir = input_dir / "rgb"
    files = sorted(rgb_dir.glob("*.png"))
    if not files:
        files = sorted(rgb_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No RGB frames found under {rgb_dir}")
    return files


def tensor_to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def extract_instances(output: dict[str, Any], h: int, w: int, *, min_score: float, min_area: int, max_area_frac: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    masks = tensor_to_numpy(output.get("masks", np.zeros((0, 1, h, w), dtype=np.float32)))
    scores = tensor_to_numpy(output.get("scores", np.zeros((masks.shape[0],), dtype=np.float32))).reshape(-1)
    boxes = tensor_to_numpy(output.get("boxes", np.zeros((masks.shape[0], 4), dtype=np.float32))).reshape(-1, 4)

    if masks.ndim == 4:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None]

    union = np.zeros((h, w), dtype=bool)
    instances: list[dict[str, Any]] = []
    max_area = int(h * w * max_area_frac)
    for idx in range(masks.shape[0]):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        mask = masks[idx] > 0
        area = int(mask.sum())
        if score < min_score or area < min_area or area > max_area:
            continue
        union |= mask
        ys, xs = np.nonzero(mask)
        if len(xs):
            bbox_xyxy = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        else:
            bbox_xyxy = [0, 0, 0, 0]
        box = boxes[idx].tolist() if idx < len(boxes) else [0.0, 0.0, 0.0, 0.0]
        instances.append(
            {
                "score": score,
                "area": area,
                "bbox_xyxy": bbox_xyxy,
                "box_raw": [float(v) for v in box],
            }
        )
    return union, instances


def cleanup_mask(mask: np.ndarray, *, open_first: bool = False, close_iters: int = 1) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask_u8 = mask.astype(np.uint8)
    else:
        mask_u8 = mask.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    if open_first:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    if close_iters:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    return mask_u8.astype(bool)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def make_overlay(rgb: np.ndarray, object_mask: np.ndarray, moving_mask: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
    overlay = rgb.astype(np.float32).copy()
    # object green, moving red, hand blue
    colors = [
        (object_mask, np.array([40, 220, 60], dtype=np.float32), 0.22),
        (moving_mask, np.array([255, 50, 30], dtype=np.float32), 0.48),
        (hand_mask, np.array([50, 120, 255], dtype=np.float32), 0.50),
    ]
    for mask, color, alpha in colors:
        if mask.any():
            overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if not SAM3_ROOT.exists():
        raise FileNotFoundError(f"SAM3 checkout not found: {SAM3_ROOT}")
    sys.path.insert(0, str(SAM3_ROOT))

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    rgb_files = sorted_rgb_files(args.input_dir)
    if args.limit and args.limit > 0:
        rgb_files = rgb_files[: args.limit]

    for sub in ["object", "moving", "hand", "overlay"]:
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"[sam3] frames={len(rgb_files)} input={args.input_dir}")
    print(f"[sam3] output={args.output_dir}")
    print(
        "[sam3] prompts="
        f"object:{args.object_prompt!r}, moving:{args.moving_prompt!r}, hand:{args.hand_prompt!r}"
    )

    model = build_sam3_image_model()
    model.to(args.device)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=args.confidence)

    summary: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "prompts": {
            "object": args.object_prompt,
            "moving": args.moving_prompt,
            "hand": args.hand_prompt,
        },
        "confidence": args.confidence,
        "frames": [],
    }

    prompt_specs = [
        ("object", args.object_prompt, 300, 0.45),
        ("moving", args.moving_prompt, 80, 0.35),
        ("hand", args.hand_prompt, 50, 0.25),
    ]

    with torch.inference_mode():
        for idx, rgb_path in enumerate(rgb_files):
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to read {rgb_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            image = Image.fromarray(rgb)

            masks: dict[str, np.ndarray] = {}
            instances_by_kind: dict[str, list[dict[str, Any]]] = {}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(image)
                for kind, prompt, min_area, max_area_frac in prompt_specs:
                    output = processor.set_text_prompt(state=state, prompt=prompt)
                    mask, instances = extract_instances(
                        output,
                        h,
                        w,
                        min_score=args.confidence,
                        min_area=min_area,
                        max_area_frac=max_area_frac,
                    )
                    masks[kind] = mask
                    instances_by_kind[kind] = instances
                    processor.reset_all_prompts(state)

            object_mask = cleanup_mask(masks["object"], open_first=False, close_iters=2)
            moving_mask = cleanup_mask(masks["moving"], open_first=True, close_iters=1)
            hand_mask = cleanup_mask(masks["hand"], open_first=True, close_iters=1)

            # Keep the object stream inclusive enough for later filtering, and
            # remove hand pixels from object/moving geometry masks.
            object_mask = (object_mask | moving_mask) & (~hand_mask)
            moving_mask = moving_mask & (~hand_mask)

            name = f"{idx:06d}.png"
            save_mask(args.output_dir / "object" / name, object_mask)
            save_mask(args.output_dir / "moving" / name, moving_mask)
            save_mask(args.output_dir / "hand" / name, hand_mask)
            if args.overlay_every > 0 and (idx % args.overlay_every == 0 or idx == len(rgb_files) - 1):
                overlay = make_overlay(rgb, object_mask, moving_mask, hand_mask)
                cv2.imwrite(str(args.output_dir / "overlay" / name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            frame_summary = {
                "index": idx,
                "file": rgb_path.name,
                "size": [w, h],
                "area": {
                    "object": int(object_mask.sum()),
                    "moving": int(moving_mask.sum()),
                    "hand": int(hand_mask.sum()),
                },
                "instances": instances_by_kind,
            }
            summary["frames"].append(frame_summary)
            if idx % 10 == 0 or idx == len(rgb_files) - 1:
                print(
                    f"[sam3] {idx + 1:03d}/{len(rgb_files):03d} "
                    f"object={frame_summary['area']['object']} "
                    f"moving={frame_summary['area']['moving']} "
                    f"hand={frame_summary['area']['hand']}",
                    flush=True,
                )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    moving_areas = [f["area"]["moving"] for f in summary["frames"]]
    object_areas = [f["area"]["object"] for f in summary["frames"]]
    hand_areas = [f["area"]["hand"] for f in summary["frames"]]
    print(
        "[sam3] done "
        f"object_area_mean={np.mean(object_areas):.1f} "
        f"moving_area_mean={np.mean(moving_areas):.1f} "
        f"hand_area_mean={np.mean(hand_areas):.1f}"
    )
    print(f"[sam3] summary={args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
