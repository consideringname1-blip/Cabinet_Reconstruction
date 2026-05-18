from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-file", required=True)
    parser.add_argument("--color-file", required=True)
    parser.add_argument("--depth-file", required=True)
    parser.add_argument("--mask-file", required=True)
    parser.add_argument("--k-json", required=True)
    parser.add_argument("--model-scale", type=float, required=True)
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--debug-dir", default="/tmp/foundationpose_alignment_debug")
    args = parser.parse_args()

    FoundationPose, PoseRefinePredictor, ScorePredictor, dr, set_logging_format, set_seed = _load_foundationpose_modules()
    _install_foundationpose_runtime_patches()
    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(float(args.model_scale))
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.vertex_normals
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    color = _read_color(Path(args.color_file))
    depth = _read_depth_m(Path(args.depth_file))
    mask = _read_mask(Path(args.mask_file))
    k = np.asarray(json.loads(args.k_json), dtype=np.float32).reshape(3, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=np.asarray(mesh.vertices, dtype=np.float32),
        model_normals=vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=0,
        glctx=glctx,
    )
    pose = estimator.register(
        K=k,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=int(args.iteration),
    )
    print(json.dumps({"pose": np.asarray(pose, dtype=float).reshape(4, 4).tolist()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
