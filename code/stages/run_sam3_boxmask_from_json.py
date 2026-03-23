from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from _bootstrap import CODE_ROOT
import config
import numpy as np
from PIL import Image, ImageDraw
from task_json import load_task_json, resolve_task_json_path, save_task_json


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

    # 新版调用内部统一使用 shape=(4,)；推理时再 box[None, :]
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def squeeze_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    while mask.ndim > 2 and 1 in mask.shape:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    return mask > 0


def make_mask_png(mask_bool: np.ndarray) -> np.ndarray:
    return mask_bool.astype(np.uint8) * 255


def make_masked_rgba(color_rgb: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    alpha = mask_bool.astype(np.uint8) * 255
    return np.dstack([color_rgb, alpha])


def make_masked_depth(depth: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    masked = depth.copy()
    if depth.ndim == 2:
        masked[~mask_bool] = 0
        return masked
    if depth.ndim == 3:
        masked[~mask_bool] = 0
        return masked
    raise ValueError(f"Unsupported depth image shape: {depth.shape}")


def make_overlay_image(
    color_rgb: np.ndarray,
    mask_bool: np.ndarray,
    box_xyxy: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    mask_png = make_mask_png(mask_bool)
    mask_rgb = np.repeat(mask_png[..., None], 3, axis=2)

    base = color_rgb.astype(np.float32)
    overlay = mask_rgb.astype(np.float32)
    blended = np.clip((1.0 - alpha) * base + alpha * overlay, 0, 255).astype(np.uint8)

    out = Image.fromarray(blended, mode="RGB")
    draw = ImageDraw.Draw(out)

    x0, y0, x1, y1 = [int(round(v)) for v in np.asarray(box_xyxy).reshape(-1)[:4]]

    # 近似原 show_box 的绿色框，线宽 2
    draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
    return np.array(out)


def save_array_png(arr: np.ndarray, path: Path) -> None:
    arr = np.asarray(arr)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    Image.fromarray(arr).save(path)


def resolve_bpe_path() -> Path:
    if hasattr(config, "SAM3_BEP") or hasattr(config, "SAM3_BPE"):
        return ensure_file(
            Path(require_attr(config, "SAM3_BEP", "SAM3_BPE")).expanduser().resolve(),
            "SAM3_BEP/SAM3_BPE",
        )

    import sam3

    sam3_root = Path(sam3.__file__).resolve().parent.parent
    auto_bpe = sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    return ensure_file(auto_bpe, "SAM3 package BPE")


def resolve_device(torch_module: Any) -> Any:
    device_name = str(getattr(config, "SAM3_DEVICE", "auto")).strip().lower()

    if device_name == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")

    if device_name == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("config.SAM3_DEVICE='cuda' but CUDA is not available")
        return torch_module.device("cuda")

    if device_name == "mps":
        if not torch_module.backends.mps.is_available():
            raise RuntimeError("config.SAM3_DEVICE='mps' but MPS is not available")
        return torch_module.device("mps")

    if device_name == "cpu":
        return torch_module.device("cpu")

    raise ValueError("config.SAM3_DEVICE must be one of: auto / cuda / mps / cpu")


def main() -> int:
    if len(sys.argv) != 2:
        _eprint("Usage: python code/stages/run_sam3_boxmask_from_json.py <task_meta.json or filename>")
        return 2

    json_path = resolve_task_json_path(sys.argv[1])
    ensure_file(json_path, "JSON file")

    upload_folder = Path(require_attr(config, "UPLOAD_FOLDER")).expanduser().resolve()
    depth_root = Path(require_attr(config, "HOLOLENS2_OUTPUT_DEPTH_IMAGES")).expanduser().resolve()
    output_root = Path(require_attr(config, "SAM3_OUTPUT_ROOT")).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    task = load_task_json(json_path)

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

    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"Failed to import torch: {e}")

    sam3_root = Path(require_attr(config, "SAM3_ROOT")).expanduser().resolve()
    if str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))

    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe_path = resolve_bpe_path()
    device = resolve_device(torch)

    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    color_pil = read_image_rgb(color_path)
    color_np = np.array(color_pil)
    depth_np = read_image_array_preserve(depth_path)

    input_box = build_box_from_selection(
        selection_box=selection,
        json_width=json_width,
        json_height=json_height,
    )

    print(f"[INFO] JSON         : {json_path}")
    print(f"[INFO] Color image  : {color_path}")
    print(f"[INFO] Depth image  : {depth_path}")
    print(f"[INFO] Output dir   : {output_root}")
    print(f"[INFO] Device       : {device}")
    print(f"[INFO] SAM3 BPE     : {bpe_path}")
    print(f"[INFO] Box (xyxy)   : {input_box.tolist()}")

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=str(device),
        enable_inst_interactivity=True,
    )

    processor = Sam3Processor(model)
    inference_state = processor.set_image(color_pil)

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, _ = model.predict_inst(
                    inference_state,
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
        else:
            masks, scores, _ = model.predict_inst(
                inference_state,
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

    if len(masks) < 1:
        raise RuntimeError("SAM3 returned no masks")

    mask_bool = squeeze_mask(np.asarray(masks[0]))
    mask_png = make_mask_png(mask_bool)
    masked_color_rgba = make_masked_rgba(color_np, mask_bool)
    masked_depth = make_masked_depth(depth_np, mask_bool)
    overlay_rgb = make_overlay_image(color_np, mask_bool, input_box, alpha=0.5)

    task_name = str(task.get("task_name") or "task")
    prefix = safe_name(task_name)

    mask_name = f"{prefix}_sam3_mask.png"
    color_name = f"{prefix}_sam3_color.png"
    depth_name = f"{prefix}_sam3_depth.png"
    overlay_name = f"{prefix}_sam3_overlay.png"

    mask_out = output_root / mask_name
    color_out = output_root / color_name
    depth_out = output_root / depth_name
    overlay_out = output_root / overlay_name

    save_array_png(mask_png, mask_out)
    save_array_png(masked_color_rgba, color_out)
    save_array_png(masked_depth, depth_out)
    save_array_png(overlay_rgb, overlay_out)

    task["sam3Name"] = {
        "mask": mask_name,
        "color": color_name,
        "depth": depth_name,
        "overlay": overlay_name,
    }
    save_task_json(json_path, task)

    best_score = None
    try:
        if len(scores) > 0:
            best_score = float(np.asarray(scores).reshape(-1)[0])
    except Exception:
        best_score = None

    print(f"[OK] mask    -> {mask_out}")
    print(f"[OK] color   -> {color_out}")
    print(f"[OK] depth   -> {depth_out}")
    print(f"[OK] overlay -> {overlay_out}")
    if best_score is not None:
        print(f"[INFO] score -> {best_score:.6f}")
    print(f"[OK] JSON updated -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
