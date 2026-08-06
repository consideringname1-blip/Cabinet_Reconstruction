#!/usr/bin/env python3
"""Export a deterministic AutoSeg/SAM2 UID overlay with a color legend."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True, help="AutoSeg NPZ containing key 'a'")
    parser.add_argument("--image", type=Path, required=True, help="RGB frame aligned with the mask")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-frame", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.48)
    return parser.parse_args()


def uid_color(uid: int) -> tuple[int, int, int]:
    # Golden-ratio hue stepping keeps neighboring integer UIDs visually distinct.
    hue = (0.08 + uid * 0.618033988749895) % 1.0
    saturation = 0.72 + 0.18 * ((uid % 3) / 2.0)
    value = 0.95
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, saturation, value))


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    with np.load(args.mask) as archive:
        if "a" not in archive.files:
            raise KeyError(f"{args.mask} does not contain key 'a'")
        masks = archive["a"]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected (D,H,W) or (D,1,H,W), got {masks.shape}")
    masks = masks.astype(bool)

    rgb = Image.open(args.image).convert("RGB")
    if (rgb.height, rgb.width) != masks.shape[1:]:
        raise ValueError(
            f"RGB size {(rgb.width, rgb.height)} does not match mask size "
            f"{(masks.shape[2], masks.shape[1])}"
        )

    base = np.asarray(rgb, dtype=np.float32)
    accum = base.copy()
    overlap_count = np.zeros(masks.shape[1:], dtype=np.uint16)
    colors = [uid_color(uid) for uid in range(masks.shape[0])]
    for uid, mask in enumerate(masks):
        if not mask.any():
            continue
        color = np.asarray(colors[uid], dtype=np.float32)
        accum[mask] = (1.0 - args.alpha) * accum[mask] + args.alpha * color
        overlap_count += mask

    overlay = Image.fromarray(np.clip(accum, 0, 255).astype(np.uint8))
    overlay_draw = ImageDraw.Draw(overlay)
    label_font = load_font(12)
    for uid, mask in enumerate(masks):
        yy, xx = np.nonzero(mask)
        if len(xx) == 0:
            continue
        x, y = int(np.median(xx)), int(np.median(yy))
        label = str(uid)
        bbox = overlay_draw.textbbox((x, y), label, font=label_font, anchor="mm", stroke_width=2)
        overlay_draw.rectangle(bbox, fill=(0, 0, 0))
        overlay_draw.text(
            (x, y), label, font=label_font, anchor="mm", fill=colors[uid], stroke_width=1, stroke_fill=(0, 0, 0)
        )

    title_font = load_font(18)
    legend_font = load_font(14)
    columns = 2
    rows = (masks.shape[0] + columns - 1) // columns
    legend_width = 300
    title_height = 52
    row_height = 24
    canvas_height = max(overlay.height + title_height, title_height + rows * row_height + 18)
    canvas = Image.new("RGB", (overlay.width + legend_width, canvas_height), (24, 24, 24))
    canvas.paste(overlay, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    source_text = f" / source frame {args.source_frame}" if args.source_frame is not None else ""
    draw.text((12, 12), f"AutoSeg/SAM2 UID overlay{source_text}", font=title_font, fill="white")
    draw.text((overlay.width + 16, 12), f"D = {masks.shape[0]} tracked UIDs", font=title_font, fill="white")

    column_width = (legend_width - 24) // columns
    for uid, color in enumerate(colors):
        column = uid // rows
        row = uid % rows
        x = overlay.width + 16 + column * column_width
        y = title_height + row * row_height
        draw.rectangle((x, y + 3, x + 17, y + 20), fill=color, outline="white")
        area = int(masks[uid].sum())
        draw.text((x + 25, y + 3), f"UID {uid:02d}  {area:,} px", font=legend_font, fill="white")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    manifest = {
        "kind": "non-official diagnostic visualization",
        "mask_path": str(args.mask.resolve()),
        "mask_key": "a",
        "mask_shape_original": [int(value) for value in np.load(args.mask)["a"].shape],
        "tracked_segment_count_D": int(masks.shape[0]),
        "uid_range": [0, int(masks.shape[0] - 1)],
        "nonempty_uid_count": int(np.count_nonzero(masks.reshape(masks.shape[0], -1).any(axis=1))),
        "rgb_path": str(args.image.resolve()),
        "source_frame": args.source_frame,
        "alpha": args.alpha,
        "overlap_pixel_count": int(np.count_nonzero(overlap_count > 1)),
        "output_path": str(args.output.resolve()),
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"D={masks.shape[0]}")
    print(f"overlay={args.output.resolve()}")
    print(f"manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
