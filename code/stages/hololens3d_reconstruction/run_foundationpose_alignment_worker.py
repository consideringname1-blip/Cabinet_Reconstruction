from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh


def _install_foundationpose_runtime_patches() -> None:
    import torch
    import Utils
    import learning.training.predict_pose_refine as predict_pose_refine
    import learning.training.predict_score as predict_score

    def compute_crop_window_tf_batch(
        pts=None,
        H=None,
        W=None,
        poses=None,
        K=None,
        crop_ratio=1.2,
        out_size=(160, 160),
        method="box_3d",
        mesh_diameter=None,
    ):
        def compute_tf_batch(left, right, top, bottom):
            B = len(left)
            left = torch.round(left)
            right = torch.round(right)
            top = torch.round(top)
            bottom = torch.round(bottom)

            tf = torch.eye(3, dtype=torch.float, device="cuda")[None].expand(B, -1, -1).contiguous()
            tf[:, 0, 2] = -left
            tf[:, 1, 2] = -top
            new_tf = torch.eye(3, dtype=torch.float, device="cuda")[None].expand(B, -1, -1).contiguous()
            new_tf[:, 0, 0] = out_size[0] / (right - left)
            new_tf[:, 1, 1] = out_size[1] / (bottom - top)
            return new_tf @ tf

        if method != "box_3d":
            raise RuntimeError
        if poses is None or K is None or mesh_diameter is None:
            raise ValueError("poses, K, and mesh_diameter are required")

        B = len(poses)
        poses_t = torch.as_tensor(poses, dtype=torch.float, device="cuda")
        k_t = torch.as_tensor(K, dtype=torch.float, device="cuda")
        radius = float(mesh_diameter) * float(crop_ratio) / 2.0
        offsets = torch.tensor(
            [
                0,
                0,
                0,
                radius,
                0,
                0,
                -radius,
                0,
                0,
                0,
                radius,
                0,
                0,
                -radius,
                0,
            ],
            dtype=torch.float,
            device="cuda",
        ).reshape(-1, 3)
        crop_pts = poses_t[:, :3, 3].reshape(-1, 1, 3) + offsets.reshape(1, -1, 3)
        projected = (k_t @ crop_pts.reshape(-1, 3).T).T
        uvs = projected[:, :2] / projected[:, 2:3]
        uvs = uvs.reshape(B, -1, 2)
        center = uvs[:, 0]
        radius_px = torch.abs(uvs - center.reshape(-1, 1, 2)).reshape(B, -1).max(axis=-1)[0].reshape(-1)
        return compute_tf_batch(
            center[:, 0] - radius_px,
            center[:, 0] + radius_px,
            center[:, 1] - radius_px,
            center[:, 1] + radius_px,
        )

    Utils.compute_crop_window_tf_batch = compute_crop_window_tf_batch
    predict_pose_refine.compute_crop_window_tf_batch = compute_crop_window_tf_batch
    predict_score.compute_crop_window_tf_batch = compute_crop_window_tf_batch


def _load_foundationpose_modules():
    foundationpose_root = Path(__file__).resolve().parents[2] / "reconstruction" / "FoundationPose"
    sys.path.insert(0, str(foundationpose_root))
    os.chdir(foundationpose_root)
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr, set_logging_format, set_seed

    return FoundationPose, PoseRefinePredictor, ScorePredictor, dr, set_logging_format, set_seed


def _read_color(path: Path) -> np.ndarray:
    color = imageio.imread(path)
    if color.ndim == 2:
        color = np.repeat(color[..., None], 3, axis=2)
    if color.shape[2] == 4:
        color = color[:, :, :3]
    return np.ascontiguousarray(color)


def _read_depth_m(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"failed to read depth image: {path}")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) * 1e-3
    else:
        depth = depth.astype(np.float32)
    depth[(depth < 0.001) | ~np.isfinite(depth)] = 0
    return np.ascontiguousarray(depth)


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask image: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return np.ascontiguousarray(mask > 0)


WORKER_RESPONSE_ENCODING = "utf-8"


class FoundationPoseAlignmentRunner:
    def __init__(self) -> None:
        (
            self.FoundationPose,
            self.PoseRefinePredictor,
            self.ScorePredictor,
            self.dr,
            self.set_logging_format,
            self.set_seed,
        ) = _load_foundationpose_modules()
        _install_foundationpose_runtime_patches()
        import torch

        self.torch = torch
        self.set_logging_format()
        self.set_seed(0)
        self.cuda_available = bool(torch.cuda.is_available())
        self.torch_device = "cuda" if self.cuda_available else "cpu"
        self.torch_device_name = torch.cuda.get_device_name(0) if self.cuda_available else ""
        self.scorer = self.ScorePredictor()
        self.refiner = self.PoseRefinePredictor()
        self.glctx = self.dr.RasterizeCudaContext()

    def run_alignment(self, request: dict[str, Any]) -> dict[str, Any]:
        mesh_file = Path(str(request["mesh_file"]))
        color_file = Path(str(request["color_file"]))
        depth_file = Path(str(request["depth_file"]))
        mask_file = Path(str(request["mask_file"]))
        model_scale = float(request["model_scale"])
        iteration = int(request.get("iteration") or 5)
        debug_dir = str(request.get("debug_dir") or "/tmp/foundationpose_alignment_debug")

        k_value = request.get("k")
        if k_value is None:
            k_value = json.loads(str(request["k_json"]))
        k = np.asarray(k_value, dtype=np.float32).reshape(3, 3)

        mesh = trimesh.load(mesh_file)
        mesh.apply_scale(model_scale)
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
            mesh.vertex_normals
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        color = _read_color(color_file)
        depth = _read_depth_m(depth_file)
        mask = _read_mask(mask_file)

        if hasattr(self.refiner, "last_trans_update"):
            self.refiner.last_trans_update = None
        if hasattr(self.refiner, "last_rot_update"):
            self.refiner.last_rot_update = None

        estimator = self.FoundationPose(
            model_pts=np.asarray(mesh.vertices, dtype=np.float32),
            model_normals=vertex_normals,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=debug_dir,
            debug=0,
            glctx=self.glctx,
        )
        pose = estimator.register(
            K=k,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=iteration,
        )
        return {
            "pose": np.asarray(pose, dtype=float).reshape(4, 4).tolist(),
            "backend": "foundationpose",
            "torch_device": self.torch_device,
            "torch_cuda_available": self.cuda_available,
            "torch_device_name": self.torch_device_name,
        }


def _request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mesh_file": args.mesh_file,
        "color_file": args.color_file,
        "depth_file": args.depth_file,
        "mask_file": args.mask_file,
        "k_json": args.k_json,
        "model_scale": args.model_scale,
        "iteration": args.iteration,
        "debug_dir": args.debug_dir,
    }


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

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(8)
    print(f"[FoundationPose worker] listening: {socket_path}", flush=True)
    runner: FoundationPoseAlignmentRunner | None = None

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    request = _read_socket_json(conn)
                    if request.get("action") == "shutdown":
                        _send_socket_json(conn, {"ok": True, "shutdown": True})
                        break
                    if runner is None:
                        print("[FoundationPose worker] loading models", flush=True)
                        runner = FoundationPoseAlignmentRunner()
                        print("[FoundationPose worker] models ready", flush=True)
                    payload = runner.run_alignment(request)
                    _send_socket_json(conn, {"ok": True, "result": payload})
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    _send_socket_json(conn, {"ok": False, "error": str(exc)})
    finally:
        server.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-server")
    parser.add_argument("--mesh-file")
    parser.add_argument("--color-file")
    parser.add_argument("--depth-file")
    parser.add_argument("--mask-file")
    parser.add_argument("--k-json")
    parser.add_argument("--model-scale", type=float)
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--debug-dir", default="/tmp/foundationpose_alignment_debug")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.socket_server:
        run_socket_server(Path(args.socket_server))
        return 0

    required = {
        "--mesh-file": args.mesh_file,
        "--color-file": args.color_file,
        "--depth-file": args.depth_file,
        "--mask-file": args.mask_file,
        "--k-json": args.k_json,
        "--model-scale": args.model_scale,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")

    runner = FoundationPoseAlignmentRunner()
    print(json.dumps(runner.run_alignment(_request_from_args(args))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
