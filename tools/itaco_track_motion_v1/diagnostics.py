"""Stage-1 raster diagnostics. Colors are configuration-only visualization aids."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np


def _blend(image: np.ndarray, mask: np.ndarray, color: list[int], alpha: float) -> np.ndarray:
    result = image.astype(np.float32).copy()
    bgr = np.asarray(color[::-1], dtype=np.float32)
    result[mask] = (1.0 - alpha) * result[mask] + alpha * bgr
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_validity_overlays(records: list[dict], payloads: list[dict], output_dir: Path, config: dict) -> None:
    target = output_dir / "hand_validity_overlay"; target.mkdir(parents=True, exist_ok=True)
    alpha = float(config["overlay_alpha"])
    colors = config["colors_rgb"]
    for record, payload in zip(records, payloads):
        image = payload["rgb"].copy()
        invalid = ~payload["valid_for_tracking"]
        image = _blend(image, invalid, colors["invalid"], alpha * 0.55)
        image = _blend(image, payload["hand"], colors["hand"], alpha)
        image = _blend(image, payload["valid_for_tracking"], colors["valid"], alpha * 0.35)
        cv2.putText(image, f"original={record['original_frame_id']} processing={record['processing_index']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(target / f"{record['original_frame_id']}.png"), image)


def draw_reference_scores(candidates: list[dict], selected_index: int, output_path: Path, config: dict) -> None:
    width = int(config["reference_plot_width_px"]); row_height = int(config["reference_plot_row_height_px"])
    height = max(row_height * len(candidates) + 40, 100)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scores = np.asarray([item["score"] for item in candidates], dtype=np.float64)
    lo, hi = float(scores.min()), float(scores.max())
    for row, item in enumerate(candidates):
        y = 25 + row * row_height
        fraction = 0.5 if hi - lo < 1e-12 else (item["score"] - lo) / (hi - lo)
        color_rgb = config["colors_rgb"]["reference_selected" if item["processing_index"] == selected_index else ("reference_accepted" if item["accepted"] else "reference_rejected")]
        cv2.rectangle(canvas, (180, y - 9), (180 + int((width - 200) * fraction), y + 4), tuple(color_rgb[::-1]), -1)
        cv2.putText(canvas, f"id={item['original_frame_id']} score={item['score']:.3f}", (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def draw_track_label_overlays(records: list[dict], tracks: list[dict], labels: list[dict], output_dir: Path, config: dict) -> None:
    target = output_dir / "track_label_projection"; target.mkdir(parents=True, exist_ok=True)
    label_by_id = {item["track_id"]: item["label"] for item in labels}
    by_frame = defaultdict(list)
    for track in tracks:
        for obs in track["observations"]:
            by_frame[obs["original_frame_id"]].append((obs, label_by_id[track["track_id"]]))
    colors = config["colors_rgb"]
    radius = int(config["track_point_radius_px"])
    for record in records:
        image = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
        for obs, label in by_frame[record["original_frame_id"]]:
            u, v = (int(round(value)) for value in obs["pixel_uv"])
            cv2.circle(image, (u, v), radius, tuple(colors[label][::-1]), -1, cv2.LINE_AA)
        cv2.putText(image, f"original={record['original_frame_id']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(target / f"{record['original_frame_id']}.png"), image)


def draw_split_merge_cases(records: list[dict], tracks: list[dict], labels: list[dict], diagnostics: dict, output_dir: Path, config: dict) -> None:
    label_by_id = {item["track_id"]: item for item in labels}
    split_dir, merge_dir = output_dir / "proposal_split_cases", output_dir / "proposal_merge_cases"
    split_dir.mkdir(parents=True, exist_ok=True); merge_dir.mkdir(parents=True, exist_ok=True)
    colors = config["colors_rgb"]
    for proposal_id, label_values in diagnostics["mixed_proposals_split"].items():
        observations = [(obs, label_by_id[track["track_id"]]["label"]) for track in tracks for obs in track["observations"] if obs["proposal_id"] == proposal_id]
        if not observations: continue
        frame_id = observations[len(observations) // 2][0]["original_frame_id"]
        record = next(item for item in records if item["original_frame_id"] == frame_id)
        image = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
        for obs, label in observations:
            if obs["original_frame_id"] != frame_id: continue
            u, v = (int(round(value)) for value in obs["pixel_uv"]); cv2.circle(image, (u, v), 2, tuple(colors[label][::-1]), -1)
        cv2.putText(image, f"split {proposal_id}: {','.join(label_values)}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(split_dir / f"{proposal_id}.png"), image)
    moving_tracks = [track for track in tracks if label_by_id[track["track_id"]]["label"] == "moving"]
    moving_proposals = diagnostics["moving_set_proposal_ids"]
    if len(moving_proposals) > 1 and moving_tracks:
        all_obs = [obs for track in moving_tracks for obs in track["observations"] if obs["proposal_id"]]
        if all_obs:
            frame_id = all_obs[len(all_obs) // 2]["original_frame_id"]
            record = next(item for item in records if item["original_frame_id"] == frame_id)
            image = cv2.imread(record["rgb_path"], cv2.IMREAD_COLOR)
            for obs in all_obs:
                if obs["original_frame_id"] != frame_id: continue
                u, v = (int(round(value)) for value in obs["pixel_uv"]); cv2.circle(image, (u, v), 2, tuple(colors["moving"][::-1]), -1)
            cv2.putText(image, f"moving set merges {len(moving_proposals)} proposals", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(merge_dir / "moving_set_multiple_proposals.png"), image)
