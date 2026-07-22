from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from task_json import save_task_json  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mask_to_numpy(mask: Any) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    arr = np.asarray(mask)
    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SAM3 mask shape: {arr.shape}")
    return arr > 0


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5), dtype=bool), iterations=1)
    mask = ndi.binary_fill_holes(mask)
    labeled, num = ndi.label(mask)
    if num > 1:
        sizes = ndi.sum(mask, labeled, index=np.arange(1, num + 1))
        mask = labeled == (int(np.argmax(sizes)) + 1)
    return mask.astype(bool)


def _score_mask(mask: np.ndarray, score: float | None) -> float:
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return -1.0
    area_ratio = len(xs) / float(h * w)
    cx = float(xs.mean()) / max(1.0, w - 1.0)
    cy = float(ys.mean()) / max(1.0, h - 1.0)
    center_penalty = abs(cx - 0.5) + abs(cy - 0.5)
    area_term = min(area_ratio, 0.65)
    score_term = float(score) if score is not None else 0.0
    return score_term + area_term - 0.35 * center_penalty


def _select_mask(masks: Any, scores: Any | None) -> tuple[np.ndarray, float | None]:
    mask_list = list(masks)
    if not mask_list:
        raise RuntimeError("SAM3 returned no masks")
    score_values: list[float | None] = [None] * len(mask_list)
    if scores is not None:
        score_arr = np.asarray(scores.detach().float().cpu().numpy() if hasattr(scores, "detach") else scores).reshape(-1)
        score_values = [float(v) for v in score_arr[: len(mask_list)]]
        score_values.extend([None] * (len(mask_list) - len(score_values)))

    best_idx = max(
        range(len(mask_list)),
        key=lambda i: _score_mask(_mask_to_numpy(mask_list[i]), score_values[i]),
    )
    return _refine_mask(_mask_to_numpy(mask_list[best_idx])), score_values[best_idx]


def _make_overlay(rgb: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    blended = np.where(mask[..., None], 0.55 * base + 0.45 * red, base)
    out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    ys, xs = np.nonzero(mask)
    if len(xs):
        draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(0, 255, 0), width=2)
    return out


def _fit_512_with_background(
    rgb: Image.Image,
    mask: np.ndarray,
    background_rgb: tuple[int, int, int],
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


def _parse_background(value: str) -> tuple[int, int, int]:
    presets = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (127, 127, 127),
        "grey": (127, 127, 127),
    }
    lowered = value.strip().lower()
    if lowered in presets:
        return presets[lowered]
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected background preset or R,G,B triplet, got {value!r}")
    rgb = tuple(int(p) for p in parts)
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError(f"Background RGB values must be in [0,255], got {value!r}")
    return rgb


def _estimate_object_points_world(
    depth_path: Path,
    mask: np.ndarray,
    intrinsics: list[list[float]],
    transform_matrix: list[list[float]],
    stride: int = 4,
) -> np.ndarray:
    depth = np.asarray(Image.open(depth_path), dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected single-channel depth image at {depth_path}, got {depth.shape}")

    h, w = mask.shape
    if depth.shape != mask.shape:
        depth_img = Image.fromarray(depth.astype(np.uint16), mode="I;16")
        depth = np.asarray(depth_img.resize((w, h), Image.Resampling.NEAREST), dtype=np.float32)

    valid = mask & np.isfinite(depth) & (depth > 0)
    ys, xs = np.nonzero(valid)
    if len(xs) < 64:
        return np.empty((0, 3), dtype=np.float64)
    if stride > 1:
        ys = ys[::stride]
        xs = xs[::stride]

    z = depth[ys, xs].astype(np.float64) / 1000.0
    k = np.asarray(intrinsics, dtype=np.float64)
    x = (xs.astype(np.float64) - k[0, 2]) / k[0, 0] * z
    y = (ys.astype(np.float64) - k[1, 2]) / k[1, 1] * z

    points_cv = np.stack([x, y, z], axis=1)
    points_cam = points_cv * np.asarray([1.0, -1.0, -1.0], dtype=np.float64)
    c2w = np.asarray(transform_matrix, dtype=np.float64)
    return points_cam @ c2w[:3, :3].T + c2w[:3, 3]


def _compute_depth_normalization(
    frames: list[dict[str, Any]],
    target_radius: float,
) -> dict[str, Any]:
    per_frame_centers: list[np.ndarray] = []
    sampled_points: list[np.ndarray] = []
    used_frames: list[str] = []
    for frame in frames:
        points = _estimate_object_points_world(
            Path(frame["depth_path"]),
            frame["mask"],
            frame["intrinsics"],
            frame["transform_matrix"],
        )
        if len(points) == 0:
            continue
        lo = np.percentile(points, 5.0, axis=0)
        hi = np.percentile(points, 95.0, axis=0)
        keep = np.all((points >= lo) & (points <= hi), axis=1)
        points = points[keep]
        if len(points) == 0:
            continue
        per_frame_centers.append(np.median(points, axis=0))
        sampled_points.append(points[:: max(1, len(points) // 2000)])
        used_frames.append(str(frame["capture"]))

    if not sampled_points:
        raise RuntimeError("Could not estimate object center from masked depth in any frame")

    center = np.median(np.stack(per_frame_centers, axis=0), axis=0)
    all_points = np.concatenate(sampled_points, axis=0)
    distances = np.linalg.norm(all_points - center[None, :], axis=1)
    distances = distances[np.isfinite(distances) & (distances > 1e-4)]
    if len(distances) == 0:
        raise RuntimeError("Depth normalization produced no valid object radius samples")
    observed_radius = float(np.percentile(distances, 90.0))
    scale = float(np.clip(target_radius / observed_radius, 0.25, 4.0))
    return {
        "center_world": [float(v) for v in center.tolist()],
        "observed_radius_p90": observed_radius,
        "target_radius": float(target_radius),
        "scale": scale,
        "used_frames": used_frames,
    }


def _normalize_transform_matrix(
    transform_matrix: list[list[float]],
    normalization: dict[str, Any],
) -> list[list[float]]:
    mat = np.asarray(transform_matrix, dtype=np.float64).copy()
    center = np.asarray(normalization["center_world"], dtype=np.float64)
    scale = float(normalization["scale"])
    mat[:3, 3] = (mat[:3, 3] - center) * scale
    return [[float(v) for v in row] for row in mat.tolist()]


def _load_sam3():
    import torch

    import config

    sam3_root = Path(config.SAM3_ROOT).resolve()
    if str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {
        "bpe_path": str(Path(config.SAM3_BEP).resolve()),
        "device": device,
        "enable_inst_interactivity": True,
        "compile": False,
    }
    model = build_sam3_image_model(**kwargs)
    return torch, Sam3Processor(model, device=device, confidence_threshold=0.15), device


def _run_sam3_prompt(processor: Any, state: dict[str, Any], prompts: list[str]) -> tuple[np.ndarray, float | None, str]:
    last_error: Exception | None = None
    for prompt in prompts:
        try:
            output = processor.set_text_prompt(state=state, prompt=prompt)
            mask, score = _select_mask(output["masks"], output.get("scores"))
            return mask, score, f"text:{prompt}"
        except Exception as exc:
            last_error = exc
            processor.reset_all_prompts(state)

    try:
        output = processor.add_geometric_prompt([0.5, 0.54, 0.82, 0.86], True, state)
        mask, score = _select_mask(output["masks"], output.get("scores"))
        return mask, score, "box:center_0.82x0.86"
    except Exception as exc:
        if last_error is not None:
            raise RuntimeError(
                f"SAM3 text prompts and box fallback failed; last text error: {last_error}; box error: {exc}"
            ) from exc
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-root", type=Path, default=Path("data/upload/larm_captures"))
    parser.add_argument("--output-name", default="20260622_joint_0")
    parser.add_argument("--prompt", default="cabinet,storage cabinet,cupboard,drawer cabinet,furniture")
    parser.add_argument("--joint-type", default="revolute")
    parser.add_argument("--background", default="white", help="white, black, gray, or R,G,B")
    parser.add_argument("--normalize-with-depth", action="store_true")
    parser.add_argument("--target-radius", type=float, default=0.45)
    args = parser.parse_args()

    root = args.captures_root.resolve()
    captures = sorted(p for p in root.iterdir() if p.is_dir() and (p / "capture.json").is_file())
    if len(captures) != 6:
        raise ValueError(f"Expected exactly 6 capture directories under {root}, found {len(captures)}")

    larm_root = (CODE_ROOT.parent / "data" / "upload" / "larm" / args.output_name).resolve()
    images_root = larm_root / "images"
    sam3_root = larm_root / "sam3"
    images_root.mkdir(parents=True, exist_ok=True)
    sam3_root.mkdir(parents=True, exist_ok=True)

    torch, processor, device = _load_sam3()
    prompts = [p.strip() for p in str(args.prompt).split(",") if p.strip()] or ["cabinet"]
    background_rgb = _parse_background(args.background)

    inputs: dict[str, dict[str, dict[str, Any]]] = {"0.00": {}, "1.00": {}}
    grouped: list[dict[str, Any]] = []
    pending_frames: list[dict[str, Any]] = []
    adjusted_intrinsics: list[list[float]] | None = None

    for idx, capture_dir in enumerate(captures):
        cap = _load_json(capture_dir / "capture.json")
        qtok = "0.00" if idx < 3 else "1.00"
        qpos = 0.0 if idx < 3 else 1.0
        frame_idx = idx if idx < 3 else idx - 3

        src = Path(cap["image_path"]).resolve()
        rgb = Image.open(src).convert("RGB")
        state = processor.set_image(rgb)
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    mask, score, prompt_used = _run_sam3_prompt(processor, state, prompts)
            else:
                mask, score, prompt_used = _run_sam3_prompt(processor, state, prompts)

        mask_path = sam3_root / f"{capture_dir.name}_sam3_cabinet_mask.png"
        overlay_path = sam3_root / f"{capture_dir.name}_sam3_cabinet_overlay.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        _make_overlay(rgb, mask).save(overlay_path)

        fitted, fit = _fit_512_with_background(rgb, mask, background_rgb)
        image_name = f"color_{qtok}_in_{frame_idx:03d}.png"
        image_path = images_root / image_name
        fitted.save(image_path)

        if adjusted_intrinsics is None:
            adjusted_intrinsics = _adjust_intrinsics(cap["intrinsics"], fit)

        frame_key = f"input_frame_{frame_idx}"
        depth_path = cap.get("depth_path", str(capture_dir / "depth.png"))
        pending_frames.append(
            {
                "qtok": qtok,
                "frame_key": frame_key,
                "transform_matrix": cap["transform_matrix"],
                "image_path": str(image_path),
                "qpos": qpos,
                "depth_path": depth_path,
                "mask": mask,
                "intrinsics": cap["intrinsics"],
                "capture": str(capture_dir),
            }
        )
        grouped.append(
            {
                "capture": str(capture_dir),
                "source_image": str(src),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "depth_path": depth_path,
                "qpos": qpos,
                "sam3_score": score,
                "sam3_prompt_used": prompt_used,
                "fit": fit,
            }
        )
        print(f"[OK] {capture_dir.name} qpos={qtok} {prompt_used} score={score} -> {image_path}", flush=True)

    normalization = None
    if args.normalize_with_depth:
        normalization = _compute_depth_normalization(pending_frames, args.target_radius)
        print(json.dumps({"depth_normalization": normalization}, indent=2), flush=True)

    for frame in pending_frames:
        transform_matrix = frame["transform_matrix"]
        if normalization is not None:
            transform_matrix = _normalize_transform_matrix(transform_matrix, normalization)
        inputs[frame["qtok"]][frame["frame_key"]] = {
            "transform_matrix": transform_matrix,
            "image_path": frame["image_path"],
            "qpos": frame["qpos"],
        }

    metadata_path = larm_root / f"{args.output_name}.json"
    metadata = {
        "intrinsics": adjusted_intrinsics,
        "joint_type": args.joint_type,
        "inputs": inputs,
        "source": "data/upload/larm_captures",
        "sam3_prompt": prompts,
        "background_rgb": list(background_rgb),
        "depth_normalization": normalization,
        "grouped_captures": grouped,
    }
    save_task_json(metadata_path, metadata)
    datalist_path = larm_root / "data.txt"
    datalist_path.write_text(str(metadata_path) + "\n", encoding="utf-8")
    print(json.dumps({"metadata_json": str(metadata_path), "datalist_path": str(datalist_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
