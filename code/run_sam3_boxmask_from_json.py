from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Tuple

import config
import numpy as np
from PIL import Image


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def require_attr(module: Any, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    joined = " / ".join(names)
    raise AttributeError(f"config.py is missing required attribute: {joined}")


def safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    text = text.strip("._")
    return text or "sam3_task"


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path: Path, data: dict) -> None:
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def read_image_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_image_array_preserve(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def build_box_from_selection(
    selection_box: dict,
    json_width: int,
    json_height: int,
) -> np.ndarray:
    tl = selection_box["top_left"]
    br = selection_box["bottom_right"]
    if len(tl) != 2 or len(br) != 2:
        raise ValueError("SelectionBox.top_left / bottom_right must be length-2 arrays")

    u0, v0 = float(tl[0]), float(tl[1])
    u1, v1 = float(br[0]), float(br[1])

    # Clamp ratios first.
    u0 = min(max(u0, 0.0), 1.0)
    v0 = min(max(v0, 0.0), 1.0)
    u1 = min(max(u1, 0.0), 1.0)
    v1 = min(max(v1, 0.0), 1.0)

    x0 = u0 * float(json_width)
    y0 = v0 * float(json_height)
    x1 = u1 * float(json_width)
    y1 = v1 * float(json_height)

    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])

    x0 = int(round(np.clip(x0, 0, max(json_width - 1, 0))))
    y0 = int(round(np.clip(y0, 0, max(json_height - 1, 0))))
    x1 = int(round(np.clip(x1, 0, max(json_width - 1, 0))))
    y1 = int(round(np.clip(y1, 0, max(json_height - 1, 0))))

    if x1 <= x0:
        x1 = min(json_width - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(json_height - 1, y0 + 1)

    return np.array([[x0, y0, x1, y1]], dtype=np.float32)


def squeeze_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    while mask.ndim > 2 and 1 in mask.shape:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    return mask > 0


def make_mask_png(mask_bool: np.ndarray) -> np.ndarray:
    return (mask_bool.astype(np.uint8) * 255)


def make_masked_rgba(color_rgb: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    alpha = (mask_bool.astype(np.uint8) * 255)
    return np.dstack([color_rgb, alpha])


def make_masked_depth(depth: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    if depth.ndim == 2:
        masked = depth.copy()
        masked[~mask_bool] = 0
        return masked
    if depth.ndim == 3:
        masked = depth.copy()
        masked[~mask_bool] = 0
        return masked
    raise ValueError(f"Unsupported depth image shape: {depth.shape}")


def save_array_png(arr: np.ndarray, path: Path) -> None:
    arr = np.asarray(arr)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    Image.fromarray(arr).save(path)


def main() -> int:
    if len(sys.argv) != 2:
        _eprint("Usage: python run_sam3_boxmask_from_json.py /path/to/task.json")
        return 2

    json_path = Path(sys.argv[1]).expanduser().resolve()
    ensure_file(json_path, "JSON file")

    upload_folder = Path(require_attr(config, "UPLOAD_FOLDER")).expanduser().resolve()
    depth_root = Path(require_attr(config, "HOLOLENS2_OUTPUT_DEPTH_IMAGES")).expanduser().resolve()
    output_root = Path(require_attr(config, "SAM3_OUTPUT_ROOT")).expanduser().resolve()
    bpe_path = Path(require_attr(config, "SAM3_BEP", "SAM3_BPE")).expanduser().resolve()
    checkpoint_path = Path(require_attr(config, "SAM3_CHECKPOINTS")).expanduser().resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    task = load_json(json_path)

    pv_info = task.get("PVCamera") or {}
    depth_info = task.get("DepthCamera") or {}
    selection = task.get("SelectionBox") or {}

    pv_name = pv_info.get("name")
    align_depth_name = depth_info.get("align_depth_name")
    json_width = int(pv_info.get("width"))
    json_height = int(pv_info.get("height"))

    if not pv_name:
        raise ValueError("PVCamera.name is missing")
    if not align_depth_name:
        raise ValueError("DepthCamera.align_depth_name is missing or null")
    if "top_left" not in selection or "bottom_right" not in selection:
        raise ValueError("SelectionBox.top_left / bottom_right is missing")

    color_path = ensure_file(upload_folder / pv_name, "PVCamera image")
    depth_path = ensure_file(depth_root / align_depth_name, "Aligned depth image")
    ensure_file(bpe_path, "SAM3_BEP/SAM3_BPE")
    ensure_file(checkpoint_path, "SAM3_CHECKPOINTS")

    device = "cuda"
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"Failed to import torch: {e}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this script, but torch.cuda.is_available() is False")

    # Import SAM3 only after cwd / env are ready in the caller.
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    color_pil = read_image_rgb(color_path)
    color_np = np.array(color_pil)
    depth_np = read_image_array_preserve(depth_path)

    input_box = build_box_from_selection(
        selection_box=selection,
        json_width=json_width,
        json_height=json_height,
    )

    print(f"[INFO] JSON       : {json_path}")
    print(f"[INFO] Color image: {color_path}")
    print(f"[INFO] Depth image: {depth_path}")
    print(f"[INFO] Output dir : {output_root}")
    print(f"[INFO] SAM3 BPE   : {bpe_path}")
    print(f"[INFO] Checkpoint : {checkpoint_path}")
    print(f"[INFO] Box (xyxy) : {input_box.tolist()}")

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=device,
        eval_mode=True,
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        compile=False,
    )

    processor = Sam3Processor(model)

    with torch.inference_mode():
        inference_state = processor.set_image(color_pil)
        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )

    if len(masks) < 1:
        raise RuntimeError("SAM3 returned no masks")

    mask_bool = squeeze_mask(np.asarray(masks[0]))
    mask_png = make_mask_png(mask_bool)
    masked_color_rgba = make_masked_rgba(color_np, mask_bool)
    masked_depth = make_masked_depth(depth_np, mask_bool)

    task_name = str(task.get("task_name") or "task")
    task_id = str(task.get("task_id") or json_path.stem)
    prefix = safe_name(f"{task_name}_{task_id[:8]}")

    mask_name = f"{prefix}_sam3_mask.png"
    color_name = f"{prefix}_sam3_color.png"
    depth_name = f"{prefix}_sam3_depth.png"

    mask_out = output_root / mask_name
    color_out = output_root / color_name
    depth_out = output_root / depth_name

    save_array_png(mask_png, mask_out)
    save_array_png(masked_color_rgba, color_out)
    save_array_png(masked_depth, depth_out)

    task["sam3Name"] = {
        "mask": mask_name,
        "color": color_name,
        "depth": depth_name,
    }
    save_json(json_path, task)

    best_score = None
    try:
        if len(scores) > 0:
            best_score = float(np.asarray(scores).reshape(-1)[0])
    except Exception:
        best_score = None

    print(f"[OK] mask  -> {mask_out}")
    print(f"[OK] color -> {color_out}")
    print(f"[OK] depth -> {depth_out}")
    if best_score is not None:
        print(f"[INFO] score -> {best_score:.6f}")
    print(f"[OK] JSON updated -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
