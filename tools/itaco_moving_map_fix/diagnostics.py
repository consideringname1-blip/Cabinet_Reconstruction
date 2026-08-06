from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


LABEL_CODES = {"static": 0, "unknown": 1, "moving": 2, "invalid": 3}
LABEL_COLORS = np.array(
    [[70, 130, 180], [160, 160, 160], [220, 50, 47], [20, 20, 20]],
    dtype=np.uint8,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_arrays(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_manifest(itaco_dir: Path, baseline_paths: list[Path]) -> dict:
    packages = {}
    for module_name in ("torch", "numpy", "cv2", "scipy", "yaml", "wandb"):
        try:
            module = __import__(module_name)
            packages[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            packages[module_name] = f"unavailable: {exc}"
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(itaco_dir), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        revision = "unknown"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": packages,
        "itaco_revision": revision,
        "official_source_hashes": {
            str(path.resolve()): sha256_file(path) for path in baseline_paths
        },
    }


def label_array(result: dict[str, torch.Tensor]) -> np.ndarray:
    shape = result["gated_score"].shape
    labels = np.empty(shape, dtype=np.uint8)
    for name, code in LABEL_CODES.items():
        labels[result[f"{name}_mask"].detach().cpu().numpy()] = code
    return labels


def comparison_metrics(
    result: dict[str, torch.Tensor],
    actual_sensor_support: np.ndarray,
    proposal_scores: np.ndarray,
    best_loss: float,
    moving_threshold: float,
) -> dict:
    score = result["gated_score"].detach().cpu().numpy()
    coverage = result["proposal_coverage"].detach().cpu().numpy()
    labels = label_array(result)
    high = score >= moving_threshold
    inside = actual_sensor_support
    outside = ~inside
    frame_min = score.reshape(score.shape[0], -1).min(axis=1)
    frame_max = score.reshape(score.shape[0], -1).max(axis=1)
    frame_mean = score.reshape(score.shape[0], -1).mean(axis=1)
    return {
        "total_high_score_area_ratio": float(high.mean()),
        "high_score_ratio_inside_sensor_support": float(high[inside].mean()),
        "high_score_ratio_outside_sensor_support": float(high[outside].mean()),
        "high_score_pixels_outside_sensor_support": int((high & outside).sum()),
        "invalid_area_ratio": float((~inside).mean()),
        "unknown_area_ratio": float((labels == LABEL_CODES["unknown"]).mean()),
        "proposal_covered_area_ratio": float(coverage.mean()),
        "per_frame_score_min": frame_min.tolist(),
        "per_frame_score_max": frame_max.tolist(),
        "per_frame_score_mean": frame_mean.tolist(),
        "per_proposal_score_variance_across_visible_frames": [
            0.0 for _ in proposal_scores
        ],
        "proposal_scores": proposal_scores.tolist(),
        "final_optimization_loss": float(best_loss),
    }


def proposal_evidence_table(adjustment, support: np.ndarray, scores: np.ndarray) -> list[dict]:
    """Compute deterministic per-proposal one-way nearest-surface evidence."""
    surface_tree = cKDTree(adjustment.surface_xyz.detach().cpu().numpy())
    xyz = adjustment.xyz.detach().cpu().numpy()
    parts = adjustment.part_segments_full.detach().cpu().numpy().astype(bool)
    cameras = adjustment.best_camera_poses
    axis = adjustment.best_joint_axis
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    values = adjustment.best_joint_state
    static_sum = np.zeros(parts.shape[1])
    dynamic_sum = np.zeros(parts.shape[1])
    counts = np.zeros(parts.shape[1], dtype=np.int64)
    frame_counts = np.zeros(parts.shape[1], dtype=np.int64)
    areas: list[list[int]] = [[] for _ in range(parts.shape[1])]
    for frame in range(parts.shape[0]):
        rotation = Rotation.from_quat(cameras[frame, :4], scalar_first=True).as_matrix()
        world = xyz[frame].reshape(-1, 3) @ rotation.T + cameras[frame, 4:]
        if adjustment.joint_type == "prismatic":
            dynamic = world + axis * values[frame]
        else:
            joint_rotation = Rotation.from_rotvec(axis * values[frame]).as_matrix()
            dynamic = world @ joint_rotation.T + (np.eye(3) - joint_rotation) @ adjustment.best_joint_pos
        flat_support = support[frame].reshape(-1)
        active = parts[frame].reshape(parts.shape[1], -1) & flat_support[None]
        union = active.any(axis=0)
        if not union.any():
            continue
        static_distance = surface_tree.query(world[union], workers=-1)[0]
        dynamic_distance = surface_tree.query(dynamic[union], workers=-1)[0]
        union_indices = np.flatnonzero(union)
        inverse = np.full(union.size, -1, dtype=np.int64)
        inverse[union_indices] = np.arange(union_indices.size)
        for proposal in range(parts.shape[1]):
            indices = np.flatnonzero(active[proposal])
            areas[proposal].append(int(indices.size))
            if indices.size:
                frame_counts[proposal] += 1
                local = inverse[indices]
                static_sum[proposal] += static_distance[local].sum()
                dynamic_sum[proposal] += dynamic_distance[local].sum()
                counts[proposal] += indices.size
    rank = np.empty_like(scores, dtype=np.int64)
    rank[np.argsort(-scores)] = np.arange(1, scores.size + 1)
    rows = []
    initial = adjustment.initial_occupation.detach().cpu().numpy()
    for proposal in range(scores.size):
        rows.append(
            {
                "proposal_id": proposal,
                "initial_score": float(initial[proposal]),
                "final_score": float(scores[proposal]),
                "visible_frame_count": int(frame_counts[proposal]),
                "valid_pixel_count": int(counts[proposal]),
                "mean_valid_area_per_frame": float(np.mean(areas[proposal]) if areas[proposal] else 0),
                "mean_static_chamfer": float(static_sum[proposal] / max(counts[proposal], 1)),
                "mean_dynamic_chamfer": float(dynamic_sum[proposal] / max(counts[proposal], 1)),
                "score_rank": int(rank[proposal]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def heatmap(score: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(np.clip(score * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def mask_panel(mask: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask] = color
    return image


def title_panel(image: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(canvas, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def create_comparison_visualizations(root: Path, config: dict) -> dict:
    modes = ("official", "gate_only", "gate_no_minmax")
    arrays = {}
    for mode in modes:
        mode_dir = root / mode
        arrays[mode] = {
            "score": np.load(mode_dir / "moving_map_gated.npz")["a"],
            "labels": np.load(mode_dir / "moving_map_labels.npz")["a"],
        }
    support = np.load(root / "gate_no_minmax" / "valid_support.npz")
    actual = support["sensor_support"]
    rgb_footprint = support["projected_rgb_footprint"]
    depth_valid = support["depth_valid"]
    hand = support["original_hand_mask"]
    coverage = np.load(root / "gate_no_minmax" / "proposal_coverage.npz")["a"]
    rgb_paths = sorted(Path(config["inputs"]["view_dir"]).joinpath("rgb").glob("*.jpg"))
    frames = list(dict.fromkeys(config["visualization"]["configured_frames"]))
    if config["visualization"].get("include_automatic_worst_offsupport_frame", False):
        official_high = arrays["official"]["score"] >= config["moving_map"]["moving_threshold"]
        counts = (official_high & (~actual)).reshape(official_high.shape[0], -1).sum(axis=1)
        frames.append(int(np.argmax(counts)))
        frames = list(dict.fromkeys(frames))
    vis_dir = root / "visualization"
    vis_dir.mkdir(exist_ok=True)
    written = []
    for frame in frames:
        rgb = cv2.imread(str(rgb_paths[frame]))
        labels = LABEL_COLORS[arrays["gate_no_minmax"]["labels"][frame]]
        panels = [
            title_panel(rgb, "RGB"),
            title_panel(mask_panel(rgb_footprint[frame]), "projected RGB footprint"),
            title_panel(mask_panel(depth_valid[frame]), "depth valid"),
            title_panel(mask_panel(hand[frame], (0, 180, 255)), "original hand mask"),
            title_panel(mask_panel(actual[frame]), "sensor support"),
            title_panel(mask_panel(coverage[frame]), "proposal coverage"),
            title_panel(heatmap(arrays["official"]["score"][frame]), "official min-max"),
            title_panel(heatmap(arrays["gate_only"]["score"][frame]), "gate_only min-max"),
            title_panel(heatmap(arrays["gate_no_minmax"]["score"][frame]), "bounded score"),
            title_panel(labels, "static/unknown/moving/invalid"),
            title_panel(mask_panel(arrays["gate_no_minmax"]["score"][frame] >= config["moving_map"]["moving_threshold"], (0, 0, 255)), "thresholded moving"),
        ]
        width = rgb.shape[1]
        blank = np.zeros_like(rgb)
        panels.append(blank)
        rows = [np.hstack(panels[i:i + 4]) for i in range(0, 12, 4)]
        sheet = np.vstack(rows)
        path = vis_dir / f"frame_{frame:06d}_comparison.jpg"
        cv2.imwrite(str(path), sheet)
        written.append(str(path))
    return {"frames": frames, "files": written}


def rotation_difference_deg(camera_a: np.ndarray, camera_b: np.ndarray) -> float:
    rotations_a = Rotation.from_quat(camera_a[:, :4], scalar_first=True)
    rotations_b = Rotation.from_quat(camera_b[:, :4], scalar_first=True)
    return float(np.degrees((rotations_a.inv() * rotations_b).magnitude()).mean())


def compare_modes(root: Path, config: dict) -> dict:
    modes = {}
    for name in ("official", "gate_only", "gate_no_minmax"):
        mode_dir = root / name
        modes[name] = {
            "metrics": json.loads((mode_dir / "comparison_metrics.json").read_text()),
            "axis": np.load(mode_dir / "joint_axis.npy"),
            "pos": np.load(mode_dir / "joint_pos.npy"),
            "q": np.load(mode_dir / "joint_value.npy"),
            "camera": np.load(mode_dir / "camera_poses.npy"),
        }
    reference = modes["official"]
    differences = {}
    for name, data in modes.items():
        a = data["axis"] / np.linalg.norm(data["axis"])
        b = reference["axis"] / np.linalg.norm(reference["axis"])
        axis_deg = np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1, 1)))
        differences[name] = {
            "joint_axis_difference_degrees": float(axis_deg),
            "joint_position_difference_m": float(np.linalg.norm(data["pos"] - reference["pos"])),
            "q_t_rmse": float(np.sqrt(np.mean((data["q"] - reference["q"]) ** 2))),
            "q_t_mean_absolute_difference": float(np.mean(np.abs(data["q"] - reference["q"]))),
            "camera_translation_mean_difference_m": float(
                np.linalg.norm(data["camera"][:, 4:] - reference["camera"][:, 4:], axis=1).mean()
            ),
            "camera_rotation_mean_difference_degrees": rotation_difference_deg(
                reference["camera"], data["camera"]
            ),
        }
    report = {
        "modes": {name: value["metrics"] for name, value in modes.items()},
        "differences_from_official": differences,
        "visualization": create_comparison_visualizations(root, config),
    }
    (root / "comparison_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
