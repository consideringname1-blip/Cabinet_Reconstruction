from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from collections import OrderedDict
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

    original_nvdiffrast_render = Utils.nvdiffrast_render

    def nvdiffrast_render_batched(
        K=None,
        H=None,
        W=None,
        ob_in_cams=None,
        glctx=None,
        context="cuda",
        get_normal=False,
        mesh_tensors=None,
        mesh=None,
        projection_mat=None,
        bbox2d=None,
        output_size=None,
        use_light=False,
        light_color=None,
        light_dir=np.array([0, 0, 1]),
        light_pos=np.array([0, 0, 0]),
        w_ambient=0.8,
        w_diffuse=0.5,
        extra=None,
    ):
        if extra is None:
            extra = {}
        render_batch_size = max(1, int(os.environ.get("FOUNDATIONPOSE_RENDER_BATCH_SIZE", "16")))
        if ob_in_cams is None or len(ob_in_cams) <= render_batch_size:
            return original_nvdiffrast_render(
                K=K, H=H, W=W, ob_in_cams=ob_in_cams, glctx=glctx, context=context,
                get_normal=get_normal, mesh_tensors=mesh_tensors, mesh=mesh, projection_mat=projection_mat,
                bbox2d=bbox2d, output_size=output_size, use_light=use_light, light_color=light_color,
                light_dir=light_dir, light_pos=light_pos, w_ambient=w_ambient, w_diffuse=w_diffuse, extra=extra,
            )

        rgb_chunks = []
        depth_chunks = []
        normal_chunks = []
        xyz_chunks = []
        for start in range(0, len(ob_in_cams), render_batch_size):
            end = min(start + render_batch_size, len(ob_in_cams))
            chunk_extra = {}
            chunk_bbox2d = bbox2d[start:end] if bbox2d is not None else None
            rgb_r, depth_r, normal_r = original_nvdiffrast_render(
                K=K, H=H, W=W, ob_in_cams=ob_in_cams[start:end], glctx=glctx, context=context,
                get_normal=get_normal, mesh_tensors=mesh_tensors, mesh=mesh, projection_mat=projection_mat,
                bbox2d=chunk_bbox2d, output_size=output_size, use_light=use_light, light_color=light_color,
                light_dir=light_dir, light_pos=light_pos, w_ambient=w_ambient, w_diffuse=w_diffuse, extra=chunk_extra,
            )
            rgb_chunks.append(rgb_r)
            depth_chunks.append(depth_r)
            if normal_r is not None:
                normal_chunks.append(normal_r)
            if "xyz_map" in chunk_extra:
                xyz_chunks.append(chunk_extra["xyz_map"])

        if xyz_chunks:
            extra["xyz_map"] = torch.cat(xyz_chunks, dim=0)
        normal = torch.cat(normal_chunks, dim=0) if normal_chunks else None
        return torch.cat(rgb_chunks, dim=0), torch.cat(depth_chunks, dim=0), normal

    Utils.nvdiffrast_render = nvdiffrast_render_batched
    predict_pose_refine.nvdiffrast_render = nvdiffrast_render_batched
    predict_score.nvdiffrast_render = nvdiffrast_render_batched

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
        import estimater as foundationpose_estimater
        import torch

        self.estimater_mod = foundationpose_estimater
        self.torch = torch
        self.set_logging_format()
        self.set_seed(0)
        self.cuda_available = bool(torch.cuda.is_available())
        self.torch_device = "cuda" if self.cuda_available else "cpu"
        self.torch_device_name = torch.cuda.get_device_name(0) if self.cuda_available else ""
        self.scorer = self.ScorePredictor()
        self.refiner = self.PoseRefinePredictor()
        self.glctx = self.dr.RasterizeCudaContext()
        self.estimator_cache_size = max(
            1, int(os.environ.get("FOUNDATIONPOSE_ESTIMATOR_CACHE_SIZE", "5"))
        )
        self._estimator_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    def _build_estimator(
        self,
        mesh_file: Path,
        model_scale: float,
        debug_dir: str,
        *,
        build_rotation_grid: bool = True,
    ):
        mesh_path = mesh_file.resolve()
        stat = mesh_path.stat()
        cache_key = (
            str(mesh_path),
            int(stat.st_mtime_ns),
            int(stat.st_size),
            float(model_scale),
            bool(build_rotation_grid),
        )
        cached = self._estimator_cache.get(cache_key)
        if cached is not None:
            cached.debug_dir = debug_dir
            self._estimator_cache.move_to_end(cache_key)
            return cached

        mesh = trimesh.load(mesh_file)
        mesh.apply_scale(float(model_scale))
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
            mesh.vertex_normals
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        kwargs = {
            "model_pts": np.asarray(mesh.vertices, dtype=np.float32),
            "model_normals": vertex_normals,
            "mesh": mesh,
            "scorer": self.scorer,
            "refiner": self.refiner,
            "debug_dir": debug_dir,
            "debug": 0,
            "glctx": self.glctx,
        }
        if build_rotation_grid:
            estimator = self.FoundationPose(**kwargs)
        else:
            original_make_rotation_grid = self.FoundationPose.make_rotation_grid

            def _skip_rotation_grid(instance, *args, **kwargs):
                instance.rot_grid = self.torch.empty(
                    (0, 4, 4),
                    dtype=self.torch.float,
                    device="cuda",
                )

            self.FoundationPose.make_rotation_grid = _skip_rotation_grid
            try:
                estimator = self.FoundationPose(**kwargs)
            finally:
                self.FoundationPose.make_rotation_grid = (
                    original_make_rotation_grid
                )

        self._estimator_cache[cache_key] = estimator
        self._estimator_cache.move_to_end(cache_key)
        while len(self._estimator_cache) > self.estimator_cache_size:
            self._estimator_cache.popitem(last=False)
        return estimator

    def _reset_refiner_state(self) -> None:
        if hasattr(self.refiner, "last_trans_update"):
            self.refiner.last_trans_update = None
        if hasattr(self.refiner, "last_rot_update"):
            self.refiner.last_rot_update = None

    def _tensor_to_numpy(self, value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        if hasattr(value, "data") and hasattr(value.data, "cpu"):
            return value.data.cpu().numpy()
        return np.asarray(value)

    def _run_initial_pose_candidates(
        self,
        *,
        estimator,
        candidates: list[dict[str, Any]],
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        k: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not candidates:
            raise ValueError("initial pose candidate list is empty")

        depth_work = self.estimater_mod.erode_depth(depth, radius=2, device="cuda")
        depth_work = self.estimater_mod.bilateral_filter_depth(depth_work, radius=2, device="cuda")
        valid = (depth_work >= 0.001) & (mask > 0)
        if valid.sum() < 4:
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = estimator.guess_translation(depth=depth_work, mask=mask, K=k)
            return pose, {"mode": "initial_candidates_fallback_valid_too_small", "candidate_count": len(candidates)}

        xyz_map = self.estimater_mod.depth2xyzmap(depth_work, k)
        tf_to_center = self._tensor_to_numpy(estimator.get_tf_to_centered_mesh()).reshape(4, 4)
        inv_tf_to_center = np.linalg.inv(tf_to_center)
        final_poses = np.asarray([candidate["pose_cv"] for candidate in candidates], dtype=np.float32).reshape(-1, 4, 4)
        centered_poses = final_poses @ inv_tf_to_center.reshape(1, 4, 4)

        self._reset_refiner_state()
        refined_poses, _vis = estimator.refiner.predict(
            mesh=estimator.mesh,
            mesh_tensors=estimator.mesh_tensors,
            rgb=rgb,
            depth=depth_work,
            K=k,
            ob_in_cams=centered_poses,
            normal_map=None,
            xyz_map=xyz_map,
            glctx=estimator.glctx,
            mesh_diameter=estimator.diameter,
            iteration=iteration,
            get_vis=False,
        )
        refined_poses_np = self._tensor_to_numpy(refined_poses).reshape(-1, 4, 4)
        scores, _vis = estimator.scorer.predict(
            mesh=estimator.mesh,
            rgb=rgb,
            depth=depth_work,
            K=k,
            ob_in_cams=refined_poses_np,
            normal_map=None,
            mesh_tensors=estimator.mesh_tensors,
            glctx=estimator.glctx,
            mesh_diameter=estimator.diameter,
            get_vis=False,
        )
        scores_np = self._tensor_to_numpy(scores).reshape(-1)
        best_index = int(np.argsort(scores_np)[::-1][0])
        pose = refined_poses_np[best_index] @ tf_to_center
        best_candidate = dict(candidates[best_index])
        return pose, {
            "mode": "initial_candidates",
            "candidate_count": len(candidates),
            "best_index": best_index,
            "best_score": float(scores_np[best_index]),
            "scores": [float(v) for v in scores_np],
            "best_candidate": {
                key: value
                for key, value in best_candidate.items()
                if key != "pose_cv"
            },
        }

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

        color = _read_color(color_file)
        depth = _read_depth_m(depth_file)
        mask = _read_mask(mask_file)
        initial_candidates = list(request.get("initial_pose_candidates_cv") or [])
        initial_search_info: dict[str, Any] | None = None

        if initial_candidates:
            grouped: dict[float, list[dict[str, Any]]] = {}
            for candidate in initial_candidates:
                grouped.setdefault(float(candidate.get("model_scale") or model_scale), []).append(candidate)
            scale_results: list[dict[str, Any]] = []
            best_payload: tuple[float, np.ndarray, dict[str, Any]] | None = None
            for scale, candidates in grouped.items():
                estimator = self._build_estimator(mesh_file, scale, debug_dir, build_rotation_grid=False)
                pose_candidate, info = self._run_initial_pose_candidates(
                    estimator=estimator,
                    candidates=candidates,
                    rgb=color,
                    depth=depth,
                    mask=mask,
                    k=k,
                    iteration=iteration,
                )
                raw_score = info.get("best_score")
                score = float(raw_score) if raw_score is not None else 0.0
                info["model_scale"] = float(scale)
                scale_results.append(dict(info))
                if best_payload is None or score > best_payload[0]:
                    best_payload = (score, pose_candidate, info)
            assert best_payload is not None
            _score, pose, initial_search_info = best_payload
            initial_search_info["total_candidate_count"] = len(initial_candidates)
            initial_search_info["scale_candidate_count"] = len(grouped)
            initial_search_info["scale_results"] = scale_results
            model_scale = float(initial_search_info.get("model_scale") or model_scale)
        else:
            estimator = self._build_estimator(mesh_file, model_scale, debug_dir)
            self._reset_refiner_state()
            pose = estimator.register(
                K=k,
                rgb=color,
                depth=depth,
                ob_mask=mask,
                iteration=iteration,
            )

        payload = {
            "pose": np.asarray(pose, dtype=float).reshape(4, 4).tolist(),
            "backend": "foundationpose",
            "model_scale": float(model_scale),
            "torch_device": self.torch_device,
            "torch_cuda_available": self.cuda_available,
            "torch_device_name": self.torch_device_name,
        }
        if initial_search_info is not None:
            payload["initial_search"] = initial_search_info
        return payload


def _request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    request = {
        "mesh_file": args.mesh_file,
        "color_file": args.color_file,
        "depth_file": args.depth_file,
        "mask_file": args.mask_file,
        "k_json": args.k_json,
        "model_scale": args.model_scale,
        "iteration": args.iteration,
        "debug_dir": args.debug_dir,
    }
    if args.initial_pose_candidates_json:
        request["initial_pose_candidates_cv"] = json.loads(args.initial_pose_candidates_json)
    return request


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
                    if request.get("action") == "prewarm":
                        _send_socket_json(
                            conn,
                            {
                                "ok": True,
                                "prewarmed": True,
                                "torch_device_name": runner.torch_device_name,
                            },
                        )
                        continue
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
    parser.add_argument("--initial-pose-candidates-json")
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
