from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from task_json import save_task_json  # noqa: E402


def _parse_rgb(value: str) -> tuple[int, int, int]:
    presets = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (127, 127, 127),
        "grey": (127, 127, 127),
        "lightgray": (235, 235, 235),
        "lightgrey": (235, 235, 235),
    }
    lowered = value.strip().lower()
    if lowered in presets:
        return presets[lowered]
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected color preset or R,G,B triplet, got {value!r}")
    rgb = tuple(int(p) for p in parts)
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError(f"RGB values must be in [0,255], got {value!r}")
    return rgb


def _fit_512(
    rgb: Image.Image,
    mask: np.ndarray,
    background_rgb: tuple[int, int, int],
    outline_rgb: tuple[int, int, int] | None,
    outline_width: int,
) -> tuple[Image.Image, dict[str, float]]:
    w, h = rgb.size
    scale = 512.0 / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_x = (512 - new_w) // 2
    pad_y = (512 - new_h) // 2

    rgba = rgb.convert("RGBA")
    alpha = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    rgba.putalpha(alpha)
    resized = rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (512, 512), background_rgb)
    canvas.paste(resized.convert("RGB"), (pad_x, pad_y), resized.split()[3])

    if outline_rgb is not None and outline_width > 0:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_resized = mask_img.resize((new_w, new_h), Image.Resampling.NEAREST)
        mask_canvas = Image.new("L", (512, 512), 0)
        mask_canvas.paste(mask_resized, (pad_x, pad_y))
        mask_arr = np.asarray(mask_canvas) > 0
        edge = ndi.binary_dilation(mask_arr, iterations=outline_width) & ~mask_arr
        arr = np.asarray(canvas).copy()
        arr[edge] = np.asarray(outline_rgb, dtype=np.uint8)
        canvas = Image.fromarray(arr, mode="RGB")

    return canvas, {"scale": scale, "pad_x": float(pad_x), "pad_y": float(pad_y)}


def _adjust_intrinsics(k: list[list[float]], fit: dict[str, float]) -> list[list[float]]:
    scale = float(fit["scale"])
    pad_x = float(fit["pad_x"])
    pad_y = float(fit["pad_y"])
    arr = np.asarray(k, dtype=np.float64).copy()
    arr[0, 0] *= scale
    arr[1, 1] *= scale
    arr[0, 2] = arr[0, 2] * scale + pad_x
    arr[1, 2] = arr[1, 2] * scale + pad_y
    return [[float(v) for v in row] for row in arr.tolist()]


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _sorted_input_items(inputs: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    out: list[tuple[str, str, dict[str, Any]]] = []
    for qtok in sorted(inputs.keys(), key=lambda x: float(x)):
        for frame_key in sorted(inputs[qtok].keys(), key=lambda x: int(x.rsplit("_", 1)[-1])):
            out.append((qtok, frame_key, inputs[qtok][frame_key]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--background", default="white")
    parser.add_argument("--outline-color", default="70,105,140")
    parser.add_argument("--outline-width", type=int, default=2)
    parser.add_argument("--no-outline", action="store_true")
    args = parser.parse_args()

    input_json = args.input_json.resolve()
    meta = json.loads(input_json.read_text(encoding="utf-8"))
    grouped = list(meta.get("grouped_captures") or [])
    items = _sorted_input_items(meta["inputs"])
    if len(grouped) != len(items):
        raise ValueError(f"Expected grouped_captures and inputs to have same length, got {len(grouped)} and {len(items)}")

    background_rgb = _parse_rgb(args.background)
    outline_rgb = None if args.no_outline else _parse_rgb(args.outline_color)

    out_root = (CODE_ROOT.parent / "data" / "upload" / "larm" / args.output_name).resolve()
    images_root = out_root / "images"
    sam3_root = out_root / "sam3"
    images_root.mkdir(parents=True, exist_ok=True)
    sam3_root.mkdir(parents=True, exist_ok=True)

    out_meta = dict(meta)
    out_meta["inputs"] = {qtok: {} for qtok in sorted(meta["inputs"].keys(), key=lambda x: float(x))}
    out_grouped: list[dict[str, Any]] = []
    adjusted_intrinsics = None

    for (qtok, frame_key, rec), capture in zip(items, grouped):
        source_image = Path(capture.get("source_image") or rec["image_path"]).resolve()
        mask_path = Path(capture.get("mask_path") or "").resolve()
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask_path for {frame_key}: {mask_path}")
        rgb = Image.open(source_image).convert("RGB")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        image, fit = _fit_512(rgb, mask, background_rgb, outline_rgb, args.outline_width)
        image_name = Path(rec["image_path"]).name
        image_path = images_root / image_name
        image.save(image_path)

        if adjusted_intrinsics is None:
            # The source metadata is already 512-adjusted; recompute only if it still has original-size K.
            adjusted_intrinsics = meta.get("intrinsics")

        out_meta["inputs"][qtok][frame_key] = {
            "transform_matrix": rec["transform_matrix"],
            "image_path": str(image_path),
            "qpos": rec["qpos"],
        }

        out_capture = dict(capture)
        out_capture["image_path"] = str(image_path)
        out_capture["background_rgb"] = list(background_rgb)
        out_capture["outline_rgb"] = None if outline_rgb is None else list(outline_rgb)
        out_capture["outline_width"] = int(args.outline_width)
        out_capture["fit"] = fit
        for key in ("mask_path", "overlay_path"):
            if out_capture.get(key):
                src = Path(out_capture[key]).resolve()
                dst = sam3_root / src.name
                _copy_or_link(src, dst)
                out_capture[key] = str(dst)
        out_grouped.append(out_capture)

    out_meta["intrinsics"] = adjusted_intrinsics
    out_meta["background_rgb"] = list(background_rgb)
    out_meta["outline_rgb"] = None if outline_rgb is None else list(outline_rgb)
    out_meta["outline_width"] = int(args.outline_width)
    out_meta["image_composition_source"] = str(input_json)
    out_meta["grouped_captures"] = out_grouped

    out_json = out_root / f"{args.output_name}.json"
    save_task_json(out_json, out_meta)
    datalist = out_root / "data.txt"
    datalist.write_text(str(out_json) + "\n", encoding="utf-8")
    print(json.dumps({"metadata_json": str(out_json), "datalist_path": str(datalist)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
