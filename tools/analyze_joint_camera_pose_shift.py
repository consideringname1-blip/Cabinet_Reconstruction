#!/usr/bin/env python3
"""Quantify and visualize camera-pose changes from a joint optimization run.

The iTACO adapter stores camera poses as [qw, qx, qy, qz, tx, ty, tz] with
T_world_camera semantics. This script compares the saved best poses with the
exact camera initialization stored in initial_parameters.npz.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


COLORS = {
    "navy": "#102A43",
    "blue": "#147D92",
    "cyan": "#2CB1BC",
    "red": "#D64545",
    "orange": "#F08C46",
    "green": "#2F855A",
    "purple": "#805AD5",
    "gray": "#829AB1",
    "light": "#E8EEF3",
    "ink": "#243B53",
    "white": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hololens-poses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="joint-camera C / gate_no_minmax")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quat_t_to_matrix(quat_t: np.ndarray) -> np.ndarray:
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(quat_t), axis=0)
    matrices[:, :3, :3] = Rotation.from_quat(
        quat_t[:, :4], scalar_first=True
    ).as_matrix()
    matrices[:, :3, 3] = quat_t[:, 4:]
    return matrices


def rotation_angle_deg(matrices: np.ndarray) -> np.ndarray:
    return np.degrees(Rotation.from_matrix(matrices).magnitude())


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def estimate_common_left_transform(
    initial: np.ndarray, optimized: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate G=(R,t) in T_optimized ~= G @ T_initial."""
    relative_rotations = optimized[:, :3, :3] @ np.transpose(
        initial[:, :3, :3], (0, 2, 1)
    )
    u, _, vt = np.linalg.svd(np.sum(relative_rotations, axis=0))
    common_rotation = u @ vt
    if np.linalg.det(common_rotation) < 0:
        u[:, -1] *= -1
        common_rotation = u @ vt
    common_translation = np.mean(
        optimized[:, :3, 3]
        - (common_rotation @ initial[:, :3, 3, None]).squeeze(-1),
        axis=0,
    )
    return common_rotation, common_translation


def apply_left_transform(
    poses: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    transformed = poses.copy()
    transformed[:, :3, :3] = rotation @ poses[:, :3, :3]
    transformed[:, :3, 3] = (
        rotation @ poses[:, :3, 3, None]
    ).squeeze(-1) + translation
    return transformed


def load_fonts() -> tuple[ImageFont.FreeTypeFont, ...]:
    regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return (
        ImageFont.truetype(str(bold), 40),
        ImageFont.truetype(str(bold), 25),
        ImageFont.truetype(str(regular), 20),
        ImageFont.truetype(str(regular), 17),
        ImageFont.truetype(str(bold), 18),
    )


def hex_color(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    title_font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=hex_color(COLORS["white"]))
    draw.text((x0 + 28, y0 + 22), title, fill=hex_color(COLORS["navy"]), font=title_font)
    draw.text((x0 + 28, y0 + 58), subtitle, fill=hex_color(COLORS["gray"]), font=small_font)
    return x0 + 78, y0 + 105, x1 - 32, y1 - 55


def plot_lines(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    x: np.ndarray,
    series: list[tuple[str, np.ndarray, str, int]],
    y_label: str,
    small_font: ImageFont.FreeTypeFont,
    label_font: ImageFont.FreeTypeFont,
    forced_range: tuple[float, float] | None = None,
) -> None:
    x0, y0, x1, y1 = area
    all_values = np.concatenate([values for _, values, _, _ in series])
    ymin = float(np.min(all_values)) if forced_range is None else forced_range[0]
    ymax = float(np.max(all_values)) if forced_range is None else forced_range[1]
    if math.isclose(ymin, ymax):
        ymin, ymax = ymin - 1.0, ymax + 1.0
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    def px(value: float) -> float:
        return x0 + (value - x[0]) / max(float(x[-1] - x[0]), 1.0) * (x1 - x0)

    def py(value: float) -> float:
        return y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)

    for fraction in np.linspace(0.0, 1.0, 5):
        value = ymin + fraction * (ymax - ymin)
        yy = py(value)
        draw.line((x0, yy, x1, yy), fill=hex_color(COLORS["light"]), width=2)
        label = f"{value:.1f}"
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text(
            (x0 - 12 - (bbox[2] - bbox[0]), yy - 10),
            label,
            fill=hex_color(COLORS["gray"]),
            font=small_font,
        )
    draw.line((x0, y0, x0, y1), fill=hex_color(COLORS["gray"]), width=2)
    draw.line((x0, y1, x1, y1), fill=hex_color(COLORS["gray"]), width=2)
    for tick in np.linspace(x[0], x[-1], 6):
        xx = px(float(tick))
        draw.line((xx, y1, xx, y1 + 7), fill=hex_color(COLORS["gray"]), width=2)
        label = str(int(round(tick)))
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text(
            (xx - (bbox[2] - bbox[0]) / 2, y1 + 10),
            label,
            fill=hex_color(COLORS["gray"]),
            font=small_font,
        )
    for name, values, color, width in series:
        points = [(px(float(xx)), py(float(yy))) for xx, yy in zip(x, values)]
        draw.line(points, fill=hex_color(color), width=width, joint="curve")
    draw.text((x0, y1 + 34), "Sampled frame", fill=hex_color(COLORS["ink"]), font=small_font)
    draw.text((x0, y0 - 28), y_label, fill=hex_color(COLORS["ink"]), font=small_font)
    legend_x = x1 - 200
    legend_y = y0 + 4
    for index, (name, _, color, width) in enumerate(series):
        yy = legend_y + index * 27
        draw.line((legend_x, yy + 9, legend_x + 30, yy + 9), fill=hex_color(color), width=width)
        draw.text((legend_x + 40, yy), name, fill=hex_color(COLORS["ink"]), font=label_font)


def plot_trajectory(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    initial_xyz: np.ndarray,
    optimized_xyz: np.ndarray,
    small_font: ImageFont.FreeTypeFont,
    label_font: ImageFont.FreeTypeFont,
) -> None:
    x0, y0, x1, y1 = area
    # Fixed isometric projection of HoloLens world XYZ to a 2D diagnostic view.
    projection = np.array([[0.84, -0.54, 0.0], [0.30, 0.47, -0.83]])
    initial_2d = initial_xyz @ projection.T
    optimized_2d = optimized_xyz @ projection.T
    combined = np.vstack((initial_2d, optimized_2d))
    low, high = np.min(combined, axis=0), np.max(combined, axis=0)
    span = np.maximum(high - low, 1e-9)
    scale = min((x1 - x0) / span[0], (y1 - y0) / span[1]) * 0.86
    center = (low + high) / 2.0

    def project(points: np.ndarray) -> list[tuple[float, float]]:
        values = (points - center) * scale
        return [
            ((x0 + x1) / 2 + point[0], (y0 + y1) / 2 - point[1])
            for point in values
        ]

    p0 = project(initial_2d)
    p1 = project(optimized_2d)
    for a, b in zip(p0, p1):
        draw.line((*a, *b), fill=hex_color(COLORS["light"]), width=2)
    draw.line(p0, fill=hex_color(COLORS["blue"]), width=5, joint="curve")
    draw.line(p1, fill=hex_color(COLORS["orange"]), width=5, joint="curve")
    for point in (p0[0], p0[-1]):
        draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=hex_color(COLORS["blue"]))
    for point in (p1[0], p1[-1]):
        draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=hex_color(COLORS["orange"]))
    legend = [
        ("HoloLens / initialization", COLORS["blue"]),
        ("Joint optimized", COLORS["orange"]),
        ("Per-frame correspondence", COLORS["light"]),
    ]
    for index, (name, color) in enumerate(legend):
        yy = y0 + index * 29
        draw.line((x0, yy + 10, x0 + 34, yy + 10), fill=hex_color(color), width=5)
        draw.text((x0 + 44, yy), name, fill=hex_color(COLORS["ink"]), font=label_font)
    draw.text(
        (x0, y1 + 12),
        "Isometric projection of camera centers (world XYZ)",
        fill=hex_color(COLORS["gray"]),
        font=small_font,
    )


def create_dashboard(
    output_path: Path,
    label: str,
    frame: np.ndarray,
    translation_mm: np.ndarray,
    translation_components_mm: np.ndarray,
    rotation_deg: np.ndarray,
    residual_translation_mm: np.ndarray,
    residual_rotation_deg: np.ndarray,
    initial_xyz: np.ndarray,
    optimized_xyz: np.ndarray,
    summary: dict,
) -> None:
    image = Image.new("RGB", (2200, 1540), hex_color("#F4F7FA"))
    draw = ImageDraw.Draw(image)
    title_font, panel_font, body_font, small_font, legend_font = load_fonts()
    draw.text(
        (70, 42),
        "Camera Pose Shift After Joint Optimization",
        fill=hex_color(COLORS["navy"]),
        font=title_font,
    )
    draw.text(
        (72, 96),
        f"{label}  |  37 sampled frames  |  T_world_camera",
        fill=hex_color(COLORS["gray"]),
        font=body_font,
    )
    metrics = [
        ("Mean translation", f"{summary['translation_offset_mm']['mean']:.1f} mm"),
        ("Max translation", f"{summary['translation_offset_mm']['max']:.1f} mm"),
        ("Mean rotation", f"{summary['rotation_offset_degrees']['mean']:.1f} deg"),
        ("Max rotation", f"{summary['rotation_offset_degrees']['max']:.1f} deg"),
    ]
    for index, (name, value) in enumerate(metrics):
        xx = 70 + index * 520
        draw.rounded_rectangle((xx, 145, xx + 470, 240), radius=14, fill=hex_color(COLORS["white"]))
        draw.text((xx + 22, 164), name, fill=hex_color(COLORS["gray"]), font=small_font)
        draw.text((xx + 22, 193), value, fill=hex_color(COLORS["navy"]), font=panel_font)

    panels = [
        (70, 280, 1070, 840),
        (1130, 280, 2130, 840),
        (70, 890, 1070, 1470),
        (1130, 890, 2130, 1470),
    ]
    area = draw_panel(
        draw,
        panels[0],
        "Translation offset by frame",
        "Direct difference in the shared HoloLens world frame",
        panel_font,
        small_font,
    )
    plot_lines(
        draw,
        area,
        frame,
        [
            ("norm", translation_mm, COLORS["navy"], 5),
            ("dx", translation_components_mm[:, 0], COLORS["red"], 3),
            ("dy", translation_components_mm[:, 1], COLORS["green"], 3),
            ("dz", translation_components_mm[:, 2], COLORS["purple"], 3),
        ],
        "millimeters",
        small_font,
        legend_font,
    )
    area = draw_panel(
        draw,
        panels[1],
        "Rotation offset by frame",
        "Geodesic angle between initialization and optimized orientation",
        panel_font,
        small_font,
    )
    plot_lines(
        draw,
        area,
        frame,
        [("angle", rotation_deg, COLORS["orange"], 5)],
        "degrees",
        small_font,
        legend_font,
        forced_range=(0.0, max(90.0, float(np.max(rotation_deg)))),
    )
    area = draw_panel(
        draw,
        panels[2],
        "Residual after one common SE(3) alignment",
        "Small residual means most of the change is a shared rigid drift",
        panel_font,
        small_font,
    )
    # Put both residuals on normalized percent scales while retaining units in legend.
    trans_scale = max(float(np.max(residual_translation_mm)), 1e-9)
    rot_scale = max(float(np.max(residual_rotation_deg)), 1e-9)
    plot_lines(
        draw,
        area,
        frame,
        [
            (
                f"translation / {trans_scale:.0f} mm",
                residual_translation_mm / trans_scale * 100.0,
                COLORS["blue"],
                5,
            ),
            (
                f"rotation / {rot_scale:.1f} deg",
                residual_rotation_deg / rot_scale * 100.0,
                COLORS["orange"],
                5,
            ),
        ],
        "percent of panel maximum",
        small_font,
        legend_font,
        forced_range=(0.0, 100.0),
    )
    area = draw_panel(
        draw,
        panels[3],
        "Camera-center trajectories",
        "Before and after joint optimization; pale lines pair the same frame",
        panel_font,
        small_font,
    )
    plot_trajectory(draw, area, initial_xyz, optimized_xyz, small_font, legend_font)
    image.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    hololens_path = args.hololens_poses.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    optimized_path = run_dir / "camera_poses.npy"
    initial_path = run_dir / "initial_parameters.npz"
    optimized_quat_t = np.load(optimized_path)
    with np.load(initial_path) as initial_npz:
        initial_quat_t = initial_npz["camera"].copy()
    hololens_raw = np.load(hololens_path)
    hololens = hololens_raw.copy()
    # The source 3x3 blocks contain small numerical non-orthogonality. Match
    # the original pipeline projection before comparing to saved quaternions.
    hololens[:, :3, :3] = Rotation.from_matrix(
        hololens_raw[:, :3, :3]
    ).as_matrix()
    initial = quat_t_to_matrix(initial_quat_t)
    optimized = quat_t_to_matrix(optimized_quat_t)
    if initial.shape != optimized.shape or initial.shape != hololens.shape:
        raise ValueError(
            f"pose shape mismatch: initial={initial.shape}, optimized={optimized.shape}, "
            f"hololens={hololens.shape}"
        )
    if not np.allclose(initial, hololens, atol=1e-12, rtol=0.0):
        raise ValueError("joint-optimization initialization is not identical to HoloLens poses")

    translation_world = optimized[:, :3, 3] - initial[:, :3, 3]
    translation_norm_m = np.linalg.norm(translation_world, axis=1)
    translation_local = np.einsum(
        "nij,nj->ni", np.transpose(initial[:, :3, :3], (0, 2, 1)), translation_world
    )
    rotation_delta_world = optimized[:, :3, :3] @ np.transpose(
        initial[:, :3, :3], (0, 2, 1)
    )
    rotation_delta_local = np.transpose(initial[:, :3, :3], (0, 2, 1)) @ optimized[:, :3, :3]
    rotation_angle = rotation_angle_deg(rotation_delta_local)
    rotation_vector_world_deg = np.degrees(
        Rotation.from_matrix(rotation_delta_world).as_rotvec()
    )
    rotation_vector_local_deg = np.degrees(
        Rotation.from_matrix(rotation_delta_local).as_rotvec()
    )

    common_r, common_t = estimate_common_left_transform(initial, optimized)
    common_aligned = apply_left_transform(initial, common_r, common_t)
    residual_translation_m = np.linalg.norm(
        optimized[:, :3, 3] - common_aligned[:, :3, 3], axis=1
    )
    residual_rotation = rotation_angle_deg(
        np.transpose(common_aligned[:, :3, :3], (0, 2, 1))
        @ optimized[:, :3, :3]
    )
    common_transform = np.eye(4)
    common_transform[:3, :3] = common_r
    common_transform[:3, 3] = common_t

    # Adjacent relative-pose change quantifies trajectory deformation independently
    # of any single world-frame left transform.
    relative_initial = np.linalg.inv(initial[:-1]) @ initial[1:]
    relative_optimized = np.linalg.inv(optimized[:-1]) @ optimized[1:]
    adjacent_translation_m = np.linalg.norm(
        relative_optimized[:, :3, 3] - relative_initial[:, :3, 3], axis=1
    )
    adjacent_rotation_deg = rotation_angle_deg(
        np.transpose(relative_initial[:, :3, :3], (0, 2, 1))
        @ relative_optimized[:, :3, :3]
    )

    translation_mm = translation_norm_m * 1000.0
    residual_translation_mm = residual_translation_m * 1000.0
    adjacent_translation_mm = adjacent_translation_m * 1000.0
    summary = {
        "comparison": args.label,
        "pose_semantics": "T_world_camera",
        "frame_count": int(len(initial)),
        "initial_equals_pipeline_projected_hololens_atol_1e-12": True,
        "hololens_raw_rotation_projection_max_abs_change": float(
            np.max(np.abs(hololens_raw[:, :3, :3] - hololens[:, :3, :3]))
        ),
        "translation_offset_mm": summarize(translation_mm),
        "rotation_offset_degrees": summarize(rotation_angle),
        "maximum_translation_frame": int(np.argmax(translation_mm)),
        "maximum_rotation_frame": int(np.argmax(rotation_angle)),
        "world_translation_component_mean_mm": {
            axis: float(value)
            for axis, value in zip("xyz", np.mean(translation_world * 1000.0, axis=0))
        },
        "initial_camera_local_translation_component_mean_mm": {
            axis: float(value)
            for axis, value in zip("xyz", np.mean(translation_local * 1000.0, axis=0))
        },
        "best_common_left_se3": {
            "matrix": common_transform.tolist(),
            "translation_mm": (common_t * 1000.0).tolist(),
            "translation_norm_mm": float(np.linalg.norm(common_t) * 1000.0),
            "rotation_degrees": float(rotation_angle_deg(common_r[None])[0]),
        },
        "residual_after_common_se3": {
            "translation_mm": summarize(residual_translation_mm),
            "rotation_degrees": summarize(residual_rotation),
        },
        "adjacent_relative_pose_change": {
            "translation_mm": summarize(adjacent_translation_mm),
            "rotation_degrees": summarize(adjacent_rotation_deg),
        },
        "source_files": {
            "optimized_camera_poses": {
                "path": str(optimized_path),
                "sha256": sha256(optimized_path),
            },
            "initial_parameters": {
                "path": str(initial_path),
                "sha256": sha256(initial_path),
            },
            "hololens_camera_poses": {
                "path": str(hololens_path),
                "sha256": sha256(hololens_path),
            },
        },
        "formulas": {
            "translation_world": "t_optimized - t_initial",
            "rotation_geodesic": "angle(R_initial^T R_optimized)",
            "common_alignment": "argmin_G sum pose residuals for T_optimized ~= G T_initial",
        },
    }

    csv_path = output_dir / "per_frame_pose_shift.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sampled_frame",
                "translation_mm",
                "rotation_deg",
                "world_dx_mm",
                "world_dy_mm",
                "world_dz_mm",
                "initial_camera_local_dx_mm",
                "initial_camera_local_dy_mm",
                "initial_camera_local_dz_mm",
                "world_rotvec_x_deg",
                "world_rotvec_y_deg",
                "world_rotvec_z_deg",
                "initial_camera_local_rotvec_x_deg",
                "initial_camera_local_rotvec_y_deg",
                "initial_camera_local_rotvec_z_deg",
                "common_se3_residual_translation_mm",
                "common_se3_residual_rotation_deg",
            ]
        )
        for index in range(len(initial)):
            writer.writerow(
                [
                    index,
                    translation_mm[index],
                    rotation_angle[index],
                    *(translation_world[index] * 1000.0),
                    *(translation_local[index] * 1000.0),
                    *rotation_vector_world_deg[index],
                    *rotation_vector_local_deg[index],
                    residual_translation_mm[index],
                    residual_rotation[index],
                ]
            )

    summary_path = output_dir / "pose_shift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.save(output_dir / "best_common_left_se3.npy", common_transform)
    dashboard_path = output_dir / "camera_pose_shift_dashboard.png"
    create_dashboard(
        dashboard_path,
        args.label,
        np.arange(len(initial)),
        translation_mm,
        translation_world * 1000.0,
        rotation_angle,
        residual_translation_mm,
        residual_rotation,
        initial[:, :3, 3],
        optimized[:, :3, 3],
        summary,
    )

    command = " ".join(shlex.quote(item) for item in sys.argv)
    report = f"""# Joint-optimized camera pose shift

## Scope and coordinate convention

- Comparison: `{args.label}`.
- Pose convention: `T_world_camera`.
- Frames: sampled frames `0..{len(initial) - 1}`, in saved array order.
- Baseline: the exact `camera` array in `initial_parameters.npz`.
- Validation: that baseline is elementwise equal at absolute tolerance `1e-12`
  after applying the pipeline rotation-matrix projection to the HoloLens 3x3
  blocks. The raw blocks have a maximum projection correction of
  `{summary["hololens_raw_rotation_projection_max_abs_change"]:.3e}`;
  translations are unchanged.
- This is a read-only derivative analysis. The official and joint-optimization
  result arrays were not changed and no optimization was rerun.

## Direct offset in the shared HoloLens world frame

- Translation mean / median / P95 / max:
  `{summary["translation_offset_mm"]["mean"]:.3f}` /
  `{summary["translation_offset_mm"]["median"]:.3f}` /
  `{summary["translation_offset_mm"]["p95"]:.3f}` /
  `{summary["translation_offset_mm"]["max"]:.3f} mm`.
- Rotation mean / median / P95 / max:
  `{summary["rotation_offset_degrees"]["mean"]:.3f}` /
  `{summary["rotation_offset_degrees"]["median"]:.3f}` /
  `{summary["rotation_offset_degrees"]["p95"]:.3f}` /
  `{summary["rotation_offset_degrees"]["max"]:.3f} degrees`.
- Maximum translation: sampled frame
  `{summary["maximum_translation_frame"]}`.
- Maximum rotation: sampled frame `{summary["maximum_rotation_frame"]}`.

## Common-drift decomposition

The best single left-multiplied transform `G` fitting
`T_optimized ~= G T_initial` has translation norm
`{summary["best_common_left_se3"]["translation_norm_mm"]:.3f} mm` and rotation
`{summary["best_common_left_se3"]["rotation_degrees"]:.3f} degrees`.

After removing this one shared rigid transform:

- Residual translation mean / P95 / max:
  `{summary["residual_after_common_se3"]["translation_mm"]["mean"]:.3f}` /
  `{summary["residual_after_common_se3"]["translation_mm"]["p95"]:.3f}` /
  `{summary["residual_after_common_se3"]["translation_mm"]["max"]:.3f} mm`.
- Residual rotation mean / P95 / max:
  `{summary["residual_after_common_se3"]["rotation_degrees"]["mean"]:.3f}` /
  `{summary["residual_after_common_se3"]["rotation_degrees"]["p95"]:.3f}` /
  `{summary["residual_after_common_se3"]["rotation_degrees"]["max"]:.3f} degrees`.

Adjacent-frame relative-pose changes remain non-zero, so the optimized
trajectory is not explained by only one global coordinate transform:

- Adjacent translation change mean / P95 / max:
  `{summary["adjacent_relative_pose_change"]["translation_mm"]["mean"]:.3f}` /
  `{summary["adjacent_relative_pose_change"]["translation_mm"]["p95"]:.3f}` /
  `{summary["adjacent_relative_pose_change"]["translation_mm"]["max"]:.3f} mm`.
- Adjacent rotation change mean / P95 / max:
  `{summary["adjacent_relative_pose_change"]["rotation_degrees"]["mean"]:.3f}` /
  `{summary["adjacent_relative_pose_change"]["rotation_degrees"]["p95"]:.3f}` /
  `{summary["adjacent_relative_pose_change"]["rotation_degrees"]["max"]:.3f} degrees`.

## Formula

- Translation: `delta_t = t_optimized - t_initial`, reported in HoloLens world
  XYZ and also resolved along each initial camera's local XYZ.
- Rotation: geodesic angle of `R_initial^T R_optimized`.
- Rotation-vector components in the CSV are given in both world and initial
  camera-local axes. Their norm equals the geodesic angle.

## Reproduction

```bash
{command}
```

Source paths and SHA-256 hashes are recorded in `pose_shift_summary.json`.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
