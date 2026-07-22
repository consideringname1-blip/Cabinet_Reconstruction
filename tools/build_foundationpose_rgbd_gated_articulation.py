from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/workspace_whz')
TOOLS = ROOT / 'tools'
sys.path.insert(0, str(TOOLS))
import align_qpos1_foundationpose_articulated as a

DEFAULT_FP_ITER1_REPORT = ROOT / 'data/output/geometric_joint_estimate_sam3door_sam3d_fitted/foundationpose_qpos1_articulated_rank01_iter1/foundationpose_qpos1_articulated_report.json'
DEFAULT_FP_ITER1_DRAWER = ROOT / 'data/output/geometric_joint_estimate_sam3door_sam3d_fitted/foundationpose_qpos1_articulated_rank01_iter1/drawer.glb'
DEFAULT_OUT = ROOT / 'data/output/geometric_joint_estimate_sam3door_sam3d_fitted/foundationpose_qpos1_articulated_rgbd_gated'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--foundationpose-iter1-report', default=str(DEFAULT_FP_ITER1_REPORT))
    parser.add_argument('--foundationpose-drawer-glb', default=str(DEFAULT_FP_ITER1_DRAWER))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUT))
    parser.add_argument('--clean', action='store_true', default=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out = Path(args.output_dir).expanduser()
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    fit = a.read_json(a.FIT_REPORT)
    joint_source = a.read_json(a.JOINT_JSON)
    fp_iter1_report_path = Path(args.foundationpose_iter1_report)
    fp_iter1 = a.read_json(fp_iter1_report_path)
    drawer_fp_path = Path(args.foundationpose_drawer_glb)
    if not drawer_fp_path.exists():
        raise FileNotFoundError(f'FoundationPose drawer GLB missing: {drawer_fp_path}')

    k = a.load_k()
    depth = a.load_depth_m(a.DEPTH_PATH)
    color = cv2.imread(str(a.COLOR_PATH), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(a.COLOR_PATH)

    base = a.load_mesh(Path(fit['parts']['base']['top_exports'][0]['glb']))
    drawer = a.load_mesh(drawer_fp_path)
    open_to_closed = np.asarray(joint_source['joint']['translation_open_to_closed_camera_m'], dtype=np.float64)
    closed = a.translate_mesh(drawer, open_to_closed)

    base_metrics = a.score_projection(base, k, a.load_mask(a.MASKS['base']), depth, 30000)
    drawer_metrics = a.score_projection(drawer, k, a.load_mask(a.MASKS['drawer']), depth, 30000)

    base_dir = out / 'base'
    drawer_dir = out / 'drawer'
    base_dir.mkdir(parents=True, exist_ok=True)
    drawer_dir.mkdir(parents=True, exist_ok=True)

    base_path = out / 'base.glb'
    drawer_path = out / 'drawer.glb'
    closed_path = drawer_dir / 'drawer_closed_from_selected_axis.glb'
    base.export(base_path)
    drawer.export(drawer_path)
    closed.export(closed_path)

    a.export_scene(out / 'cabinet_drawer_articulated_open.glb', {'base': base, 'drawer': drawer})
    a.export_scene(out / 'cabinet_drawer_articulated_closed.glb', {'base': base, 'drawer_closed': closed})
    a.export_scene(out / 'cabinet_drawer_articulated_open_closed_overlay.glb', {'base': base, 'drawer': drawer, 'drawer_closed': closed})
    a.draw_overlay(color, k, {'base': base}, {'base': a.MASKS['base']}, base_dir / 'base_qpos1_overlay.png')
    a.draw_overlay(color, k, {'drawer': drawer}, {'drawer': a.MASKS['drawer']}, drawer_dir / 'drawer_qpos1_overlay.png')
    a.draw_overlay(color, k, {'base': base, 'drawer': drawer}, {'base': a.MASKS['base'], 'drawer': a.MASKS['drawer']}, out / 'combined_qpos1_overlay.png')
    a.draw_overlay(color, k, {'base': base, 'drawer': drawer, 'drawer_closed': closed}, {'base': a.MASKS['base'], 'drawer': a.MASKS['drawer']}, out / 'combined_qpos1_open_closed_overlay.png')

    axis = np.asarray(joint_source['joint']['axis_camera_closed_to_open'], dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    report = {
        'method': 'RGB-D gated combination: keep selected_axis1d joint fixed; accept FoundationPose only where qpos1 mask/depth metrics improve.',
        'frame': joint_source.get('frame'),
        'camera_axes': joint_source.get('camera_axes'),
        'joint': {
            'type': 'prismatic',
            'axis_camera_closed_to_open': axis.astype(float).tolist(),
            'displacement_m': float(joint_source['joint']['displacement_m']),
            'translation_open_to_closed_camera_m': open_to_closed.astype(float).tolist(),
            'translation_closed_to_open_camera_m': np.asarray(joint_source['joint']['translation_closed_to_open_camera_m'], dtype=np.float64).astype(float).tolist(),
            'qpos_closed_m': 0.0,
            'qpos_open_m': float(joint_source['joint']['displacement_m']),
            'source_json': str(a.JOINT_JSON),
        },
        'alignment_selection': {
            'base': {
                'selected_source': 'sam3d_to_rgbd_surface_fit_rank01',
                'reason': 'FoundationPose conservative refine had worse qpos1 projection/depth metrics for base.',
                'source_glb': fit['parts']['base']['top_exports'][0]['glb'],
                'pose_cv_from_surface_fit': a.pose_from_fit_export(fit['parts']['base']['top_exports'][0]).astype(float).tolist(),
                'model_scale': float(fit['parts']['base']['top_exports'][0]['scale_uniform']),
                'projection_metrics_qpos1': base_metrics,
            },
            'drawer': {
                'selected_source': 'foundationpose_rank01_iter1',
                'reason': 'Conservative FoundationPose refine improved drawer qpos1 IoU/depth/leakage over rank01 surface fit.',
                'source_glb': str(drawer_fp_path),
                'pose_cv': fp_iter1['parts']['drawer']['foundationpose_pose_cv'],
                'model_scale': fp_iter1['parts']['drawer']['model_scale'],
                'projection_metrics_qpos1': drawer_metrics,
                'foundationpose_report': str(fp_iter1_report_path),
            },
        },
        'outputs': {
            'base_glb': str(base_path),
            'drawer_open_glb': str(drawer_path),
            'drawer_closed_glb': str(closed_path),
            'open_scene_glb': str(out / 'cabinet_drawer_articulated_open.glb'),
            'closed_scene_glb': str(out / 'cabinet_drawer_articulated_closed.glb'),
            'open_closed_overlay_glb': str(out / 'cabinet_drawer_articulated_open_closed_overlay.glb'),
            'combined_qpos1_overlay': str(out / 'combined_qpos1_overlay.png'),
            'combined_qpos1_open_closed_overlay': str(out / 'combined_qpos1_open_closed_overlay.png'),
        },
    }
    joint_path = out / 'joint.json'
    report_path = out / 'foundationpose_qpos1_articulated_rgbd_gated_report.json'
    joint_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'output_dir': str(out), 'joint_json': str(joint_path), 'report': str(report_path), 'open_scene_glb': report['outputs']['open_scene_glb'], 'closed_scene_glb': report['outputs']['closed_scene_glb']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
