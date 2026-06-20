from __future__ import annotations

import json
import os
import re
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

from scipy import ndimage as ndi

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import _bootstrap
import config
import numpy as np
from PIL import Image, ImageDraw
from stage_common import ensure_file, load_stage_task
from task_json import load_task_json, resolve_task_json_path, save_task_json


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

def clip_mask_to_box(mask_bool: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_bool, dtype=bool)
    x0, y0, x1, y1 = [int(round(v)) for v in np.asarray(box_xyxy).reshape(-1)[:4]]

    h, w = mask.shape
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    clipped = np.zeros_like(mask, dtype=bool)
    clipped[y0:y1 + 1, x0:x1 + 1] = mask[y0:y1 + 1, x0:x1 + 1]
    return clipped


def refine_mask(mask_bool: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_bool, dtype=bool)

    mask = ndi.binary_opening(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    mask = ndi.binary_closing(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    mask = ndi.binary_fill_holes(mask)

    labeled, num = ndi.label(mask)
    if num > 1:
        sizes = ndi.sum(mask, labeled, index=np.arange(1, num + 1))
        largest_label = int(np.argmax(sizes)) + 1
        mask = labeled == largest_label

    return mask.astype(bool)

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




def _depth_raw_to_m(depth_raw: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_raw).astype(np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size and float(np.nanmedian(finite)) > 20.0:
        depth = depth / 1000.0
    return depth.astype(np.float32)


def _parse_vector3(value: Any, name: str) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if array.size < 3:
        return None
    return array[:3].astype(np.float64)


def _parse_quaternion_xyzw(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if array.size < 4:
        return None
    norm = float(np.linalg.norm(array[:4]))
    if norm <= 1e-9:
        return None
    return (array[:4] / norm).astype(np.float64)


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _camera_matrix_from_task(task: dict[str, Any]) -> np.ndarray | None:
    pvcamera = task.get("PVCamera") or {}
    raw = pvcamera.get("k") or pvcamera.get("K") or pvcamera.get("camera_matrix")
    if raw is None:
        return None
    try:
        matrix = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    except Exception:
        return None
    if not np.isfinite(matrix).all() or matrix[0, 0] == 0 or matrix[1, 1] == 0:
        return None
    return matrix


def compute_sam3_spatial_box(task: dict[str, Any], mask_bool: np.ndarray, depth_raw: np.ndarray) -> dict[str, Any]:
    camera_matrix = _camera_matrix_from_task(task)
    pvcamera = task.get("PVCamera") or {}
    camera_position = _parse_vector3(pvcamera.get("position"), "PVCamera.position")
    camera_rotation = _parse_quaternion_xyzw(pvcamera.get("rotation_quaternion_xyzw"))
    if camera_matrix is None or camera_position is None or camera_rotation is None:
        return {
            "status": "unavailable",
            "reason": "missing_pv_intrinsics_or_pose",
        }

    mask = np.asarray(mask_bool, dtype=bool)
    depth_m = _depth_raw_to_m(depth_raw)
    if depth_m.ndim == 3:
        depth_m = depth_m[:, :, 0]
    if depth_m.shape != mask.shape:
        return {
            "status": "unavailable",
            "reason": "depth_mask_shape_mismatch",
            "depth_shape": list(depth_m.shape),
            "mask_shape": list(mask.shape),
        }

    valid = mask & np.isfinite(depth_m) & (depth_m > 0.0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 32:
        return {
            "status": "unavailable",
            "reason": "not_enough_valid_depth_pixels",
            "valid_depth_pixels": valid_count,
        }

    ys, xs = np.nonzero(valid)
    zs = depth_m[valid].astype(np.float64)
    z_low, z_high = np.percentile(zs, [5.0, 95.0])
    keep = (zs >= z_low) & (zs <= z_high)
    if int(np.count_nonzero(keep)) < 32:
        keep = np.ones_like(zs, dtype=bool)
    xs = xs[keep].astype(np.float64)
    ys = ys[keep].astype(np.float64)
    zs = zs[keep].astype(np.float64)

    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    x_cv = (xs - cx) * zs / fx
    y_cv = (ys - cy) * zs / fy
    # OpenCV camera: +X right, +Y down, +Z forward. Unity camera: +X right, +Y up, +Z forward.
    points_camera_unity = np.stack([x_cv, -y_cv, zs], axis=1)
    rotation_world_from_camera = _quat_xyzw_to_matrix(camera_rotation)
    points_world = (rotation_world_from_camera @ points_camera_unity.T).T + camera_position.reshape(1, 3)

    if points_world.shape[0] >= 64:
        p_low = np.percentile(points_world, 2.0, axis=0)
        p_high = np.percentile(points_world, 98.0, axis=0)
    else:
        p_low = np.min(points_world, axis=0)
        p_high = np.max(points_world, axis=0)
    size = np.maximum(p_high - p_low, np.array([0.03, 0.03, 0.03], dtype=np.float64))
    center = (p_low + p_high) * 0.5
    p_low = center - size * 0.5
    p_high = center + size * 0.5

    x0 = y0 = x1 = y1 = 0
    mask_ys, mask_xs = np.nonzero(mask)
    if mask_xs.size:
        x0, x1 = int(mask_xs.min()), int(mask_xs.max())
        y0, y1 = int(mask_ys.min()), int(mask_ys.max())

    return {
        "status": "ready",
        "coordinate_space": "unity_world",
        "aabb_min_world": [float(v) for v in p_low],
        "aabb_max_world": [float(v) for v in p_high],
        "center_world": [float(v) for v in center],
        "size_world": [float(v) for v in size],
        "source": "sam3_mask_aligned_depth_percentile",
        "mask_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "mask_pixels": int(np.count_nonzero(mask)),
        "valid_depth_pixels": valid_count,
        "used_depth_pixels": int(zs.size),
        "depth_percentile_m": [float(z_low), float(z_high)],
    }

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


WORKER_RESPONSE_ENCODING = "utf-8"


class Sam3MaskRunner:
    def __init__(self) -> None:
        self._loaded = False
        self.torch = None
        self.device = None
        self.model = None
        self.processor_cls = None
        self.bpe_path: Path | None = None

    def _load_model(self) -> None:
        if self._loaded:
            return

        try:
            import torch
        except Exception as e:
            raise RuntimeError(f"Failed to import torch: {e}") from e

        sam3_root = Path(require_attr(config, "SAM3_ROOT")).expanduser().resolve()
        if str(sam3_root) not in sys.path:
            sys.path.insert(0, str(sam3_root))

        import sam3  # noqa: F401
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        bpe_path = resolve_bpe_path()
        device = resolve_device(torch)

        if device.type == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        print(f"[SAM3 worker] loading model on {device}; BPE={bpe_path}", flush=True)
        self.model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            device=str(device),
            enable_inst_interactivity=True,
            compile=False,
        )
        self.torch = torch
        self.device = device
        self.processor_cls = Sam3Processor
        self.bpe_path = bpe_path
        self._loaded = True
        print("[SAM3 worker] model ready", flush=True)

    def run_task(self, json_path: Path, task: dict[str, Any]) -> None:
        upload_folder = Path(require_attr(config, "UPLOAD_FOLDER")).expanduser().resolve()
        depth_root = Path(require_attr(config, "HOLOLENS2_OUTPUT_DEPTH_IMAGES")).expanduser().resolve()
        output_root = Path(require_attr(config, "SAM3_OUTPUT_ROOT")).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

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

        self._load_model()
        assert self.torch is not None
        assert self.device is not None
        assert self.model is not None
        assert self.processor_cls is not None
        assert self.bpe_path is not None

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
        print(f"[INFO] Device       : {self.device}")
        print(f"[INFO] SAM3 BPE     : {self.bpe_path}")
        print(f"[INFO] Box (xyxy)   : {input_box.tolist()}")

        processor = self.processor_cls(self.model)
        inference_state = processor.set_image(color_pil)

        with self.torch.inference_mode():
            if self.device.type == "cuda":
                with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                    masks, scores, _ = self.model.predict_inst(
                        inference_state,
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
            else:
                masks, scores, _ = self.model.predict_inst(
                    inference_state,
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )

        if len(masks) < 1:
            raise RuntimeError("SAM3 returned no masks")

        mask_bool = squeeze_mask(np.asarray(masks[0]))
        mask_bool = clip_mask_to_box(mask_bool, input_box)
        mask_bool = refine_mask(mask_bool)

        mask_png = make_mask_png(mask_bool)
        masked_color_rgba = make_masked_rgba(color_np, mask_bool)
        masked_depth = make_masked_depth(depth_np, mask_bool)
        overlay_rgb = make_overlay_image(color_np, mask_bool, input_box, alpha=0.5)
        spatial_box = compute_sam3_spatial_box(task, mask_bool, depth_np)

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
        task["Sam3SpatialBox"] = spatial_box
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
        print(f"[INFO] spatial box -> {spatial_box.get('status')}")
        print(f"[OK] JSON updated -> {json_path}")


def _read_socket_json(conn: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    raw = b"".join(chunks).splitlines()[0]
    return json.loads(raw.decode(WORKER_RESPONSE_ENCODING))


def _send_socket_json(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode(WORKER_RESPONSE_ENCODING))


def run_socket_server(socket_path: Path) -> None:
    socket_path = socket_path.expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass

    runner = Sam3MaskRunner()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(8)
    print(f"[SAM3 worker] listening: {socket_path}", flush=True)

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    request = _read_socket_json(conn)
                    if request.get("action") == "shutdown":
                        _send_socket_json(conn, {"ok": True, "shutdown": True})
                        break
                    json_path = ensure_file(resolve_task_json_path(request["json_path"]), "JSON file")
                    task = load_task_json(json_path)
                    runner.run_task(json_path, task)
                    _send_socket_json(conn, {"ok": True})
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    _send_socket_json(conn, {"ok": False, "error": str(exc)})
    finally:
        server.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--socket-server":
            run_socket_server(Path(sys.argv[2]))
            return 0

        json_path, task = load_stage_task(
            sys.argv,
            usage="Usage: python code/stages/hololens3d_reconstruction/run_sam3_boxmask_from_json.py <task_meta.json or filename>",
            stage_name="sam3mask",
        )
        Sam3MaskRunner().run_task(json_path, task)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
