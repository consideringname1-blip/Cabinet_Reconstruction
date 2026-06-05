from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image

from config import BLENDER_BIN, BLENDER_FBX_DIR
from coordinate_systems import UNITY_TO_OPENCV_CAMERA_BASIS, quat_xyzw_to_rotation_matrix

from . import settings
from .schemas import RosStamp, ShigureFrame, to_jsonable


OPENCV_TO_BLENDER_CAMERA_BASIS = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
FBX_IMPORTED_LOCAL_FROM_RUNTIME = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ModelDepthTemplate:
    front_depth_m: np.ndarray
    back_depth_m: np.ndarray
    mask: np.ndarray
    hit_count: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    output_dir: Path
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class DepthFrameDecision:
    timestamp: RosStamp
    status: str
    candidate: bool
    model_pixels: int
    valid_pixels: int
    evaluable_pixels: int
    present_pixels: int
    removed_pixels: int
    occluded_pixels: int
    support_pixels: int
    added_support_pixels: int
    evaluable_ratio: float
    removed_ratio: float
    present_ratio: float
    occluded_ratio: float
    median_removal_excess_m: float | None
    removed_mask: np.ndarray
    unoccluded_mask: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"removed_mask", "unoccluded_mask"}
        }
        return to_jsonable(payload)


def read_depth_image_m(path: str | Path) -> np.ndarray:
    depth = np.asarray(Image.open(path))
    return depth.astype(np.float64) * 0.001


def _rt_from_payload(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rotation = quat_xyzw_to_rotation_matrix(
        np.asarray(payload["rotation_quaternion_xyzw"], dtype=np.float64)
    )
    translation = np.asarray(payload["position"], dtype=np.float64).reshape(3)
    return rotation, translation


def object_world_to_current_opencv(
    task: Mapping[str, Any],
    marker_pose: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    aruco_reference = task.get("aruco_reference")
    object_world = task.get("object_world")
    if not isinstance(aruco_reference, Mapping) or not isinstance(object_world, Mapping):
        raise ValueError("task must contain aruco_reference and object_world")

    marker_world_rotation, marker_world_translation = _rt_from_payload(aruco_reference)
    object_world_rotation, object_world_translation = _rt_from_payload(object_world)
    object_marker_rotation = marker_world_rotation.T @ object_world_rotation
    object_marker_translation = marker_world_rotation.T @ (
        object_world_translation - marker_world_translation
    )

    unity_to_cv = np.asarray(UNITY_TO_OPENCV_CAMERA_BASIS, dtype=np.float64)
    object_marker_rotation_cv = unity_to_cv @ object_marker_rotation @ unity_to_cv.T
    object_marker_translation_cv = unity_to_cv @ object_marker_translation

    marker_cv = marker_pose.get("opencv_camera_pose")
    if not isinstance(marker_cv, Mapping):
        raise ValueError("marker pose does not contain opencv_camera_pose")
    marker_rotation_cv = np.asarray(marker_cv["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    marker_translation_cv = np.asarray(
        marker_cv.get("position", marker_cv.get("tvec_m")), dtype=np.float64
    ).reshape(3)
    return (
        marker_rotation_cv @ object_marker_rotation_cv,
        marker_rotation_cv @ object_marker_translation_cv + marker_translation_cv,
    )


def opencv_pose_to_blender_matrix(
    rotation_cv: np.ndarray,
    translation_cv: np.ndarray,
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = (
        OPENCV_TO_BLENDER_CAMERA_BASIS
        @ np.asarray(rotation_cv, dtype=np.float64).reshape(3, 3)
        @ FBX_IMPORTED_LOCAL_FROM_RUNTIME.T
    )
    matrix[:3, 3] = OPENCV_TO_BLENDER_CAMERA_BASIS @ np.asarray(
        translation_cv, dtype=np.float64
    ).reshape(3)
    return matrix


def _resolve_fbx(task: Mapping[str, Any]) -> Path:
    blender = task.get("Blender")
    fbx_name = str(blender.get("fbx") if isinstance(blender, Mapping) else "").strip()
    if not fbx_name:
        raise ValueError("Blender.fbx is missing")
    fbx_path = Path(BLENDER_FBX_DIR) / fbx_name
    if not fbx_path.is_file():
        raise FileNotFoundError(fbx_path)
    return fbx_path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, ensure_ascii=False, indent=2)
        file.write("\n")


def _blender_script_text() -> str:
    return r'''
import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def import_and_place(cfg):
    before = {obj.name for obj in bpy.context.scene.objects}
    bpy.ops.import_scene.fbx(filepath=cfg['fbx_path'])
    objects = [obj for obj in bpy.context.scene.objects if obj.name not in before]
    meshes = [obj for obj in objects if obj.type == 'MESH']
    if not meshes:
        raise RuntimeError('FBX contains no mesh objects')
    matrix = Matrix(cfg['matrix_world'])
    object_set = set(objects)
    roots = [obj for obj in objects if obj.parent not in object_set] or objects
    for obj in roots:
        obj.matrix_world = matrix @ obj.matrix_world
    return meshes


def render_depth_layers(cfg):
    width = int(cfg['width']); height = int(cfg['height'])
    fx = float(cfg['fx']); fy = float(cfg['fy'])
    cx = float(cfg['cx']); cy = float(cfg['cy'])
    x0, y0, x1, y1 = [int(value) for value in cfg['bbox_xyxy']]
    max_hits = int(cfg.get('max_ray_hits', 32))
    epsilon = float(cfg.get('ray_epsilon_m', 0.0002))
    fallback_back = float(cfg['fallback_back_depth_m'])

    front = np.zeros((height, width), dtype=np.float32)
    back = np.zeros((height, width), dtype=np.float32)
    hit_count = np.zeros((height, width), dtype=np.uint8)
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera_origin = Vector((0.0, 0.0, 0.0))

    for y in range(y0, y1):
        y_cam = (float(y) - cy) / fy
        for x in range(x0, x1):
            x_cam = (float(x) - cx) / fx
            direction = Vector((x_cam, -y_cam, -1.0)).normalized()
            origin = camera_origin
            depths = []
            for _ in range(max_hits):
                hit, location, _normal, _index, _obj, _matrix = scene.ray_cast(
                    depsgraph, origin, direction, distance=100.0
                )
                if not hit:
                    break
                z_m = -float(location.z)
                if z_m > 0.0:
                    depths.append(z_m)
                origin = location + direction * epsilon
            if depths:
                front[y, x] = float(depths[0])
                back[y, x] = float(depths[-1] if len(depths) > 1 else max(depths[0], fallback_back))
                hit_count[y, x] = min(255, len(depths))

    np.save(cfg['front_depth_npy'], front)
    np.save(cfg['back_depth_npy'], back)
    np.save(cfg['hit_count_npy'], hit_count)
    return {
        'mesh_count': len([obj for obj in bpy.context.scene.objects if obj.type == 'MESH']),
        'mask_pixels': int(np.count_nonzero(front)),
        'multi_hit_pixels': int(np.count_nonzero(hit_count > 1)),
        'single_hit_pixels': int(np.count_nonzero(hit_count == 1)),
    }


def main():
    cfg_path = Path(__file__).with_name('render_config.json')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    clean_scene()
    import_and_place(cfg)
    result = render_depth_layers(cfg)
    Path(cfg['debug_json']).write_text(json.dumps(result, indent=2), encoding='utf-8')


main()
'''


def render_model_depth_template(
    *,
    task: Mapping[str, Any],
    marker_pose_path: Path,
    camera_matrix: np.ndarray,
    image_size: tuple[int, int],
    bbox_xyxy: tuple[float, float, float, float],
    fallback_back_depth_m: float,
    output_dir: Path,
) -> ModelDepthTemplate:
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_pose = _load_json(marker_pose_path)
    rotation_cv, translation_cv = object_world_to_current_opencv(task, marker_pose)
    matrix_world = opencv_pose_to_blender_matrix(rotation_cv, translation_cv)
    width, height = image_size
    padding = max(0, int(settings.MODEL_DEPTH_RENDER_PADDING_PX))
    x0 = max(0, int(np.floor(bbox_xyxy[0])) - padding)
    y0 = max(0, int(np.floor(bbox_xyxy[1])) - padding)
    x1 = min(width, int(np.ceil(bbox_xyxy[2])) + padding + 1)
    y1 = min(height, int(np.ceil(bbox_xyxy[3])) + padding + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("projected model bbox is empty")

    front_path = output_dir / "front_depth_m.npy"
    back_path = output_dir / "back_depth_m.npy"
    hits_path = output_dir / "hit_count.npy"
    debug_path = output_dir / "blender_debug.json"
    config_path = output_dir / "render_config.json"
    script_path = output_dir / "render_depth_layers.py"
    config = {
        "fbx_path": str(_resolve_fbx(task)),
        "matrix_world": matrix_world.astype(float).tolist(),
        "width": int(width),
        "height": int(height),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "bbox_xyxy": [x0, y0, x1, y1],
        "fallback_back_depth_m": float(fallback_back_depth_m),
        "max_ray_hits": int(settings.MODEL_DEPTH_MAX_RAY_HITS),
        "ray_epsilon_m": float(settings.MODEL_DEPTH_RAY_EPSILON_M),
        "front_depth_npy": str(front_path),
        "back_depth_npy": str(back_path),
        "hit_count_npy": str(hits_path),
        "debug_json": str(debug_path),
    }
    _write_json(config_path, config)
    script_path.write_text(_blender_script_text(), encoding="utf-8")

    blender = Path(BLENDER_BIN)
    if not blender.is_file():
        found = shutil.which("blender")
        if not found:
            raise FileNotFoundError(f"Blender executable not found: {BLENDER_BIN}")
        blender = Path(found)
    subprocess.run(
        [str(blender), "--background", "--python", str(script_path)],
        cwd=str(output_dir),
        check=True,
    )

    front = np.load(front_path).astype(np.float64)
    back = np.load(back_path).astype(np.float64)
    hit_count = np.load(hits_path).astype(np.uint8)
    mask = front > 0.0
    erosion = max(0, int(settings.MODEL_DEPTH_MASK_ERODE_PX))
    if erosion > 0 and np.count_nonzero(mask) > 0:
        kernel = np.ones((erosion * 2 + 1, erosion * 2 + 1), dtype=np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
        if np.count_nonzero(eroded) >= settings.MODEL_DEPTH_MIN_MODEL_PIXELS:
            mask = eroded
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / "model_mask.png")

    metadata = _load_json(debug_path) if debug_path.is_file() else {}
    metadata = {
        **metadata,
        "bbox_xyxy": [x0, y0, x1, y1],
        "mask_pixels_after_erosion": int(np.count_nonzero(mask)),
        "coordinate_contract": "docs/coordinate-systems.md trusted FBX overlay chain",
    }
    _write_json(output_dir / "template.json", metadata)
    if np.count_nonzero(mask) < settings.MODEL_DEPTH_MIN_MODEL_PIXELS:
        raise RuntimeError("rendered model depth mask is too small")
    return ModelDepthTemplate(
        front_depth_m=front,
        back_depth_m=back,
        mask=mask,
        hit_count=hit_count,
        bbox_xyxy=(x0, y0, x1, y1),
        output_dir=output_dir,
        metadata=metadata,
    )


def calibrate_depth_bias(
    frames: list[ShigureFrame],
    template: ModelDepthTemplate,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    if not frames:
        return 0.0, template.mask.copy(), {"reason": "no calibration frames"}
    start_seconds = frames[0].stamp.seconds
    candidates: list[tuple[float, float, ShigureFrame, np.ndarray]] = []
    public_candidates: list[dict[str, Any]] = []
    mask = template.mask
    for frame in frames:
        if frame.stamp.seconds - start_seconds > settings.MODEL_DEPTH_CALIBRATION_SECONDS:
            break
        if frame.depth_path is None or not frame.depth_path.is_file():
            continue
        depth = read_depth_image_m(frame.depth_path)
        valid = mask & np.isfinite(depth) & (depth > 0.0)
        residual = depth - template.front_depth_m
        supported = (
            valid
            & (residual >= -settings.MODEL_DEPTH_OCCLUSION_MARGIN_M)
            & (residual <= settings.MODEL_DEPTH_CALIBRATION_MAX_RESIDUAL_M)
        )
        support_pixels = int(np.count_nonzero(supported))
        support_ratio = support_pixels / max(1, int(np.count_nonzero(mask)))
        if support_pixels < settings.MODEL_DEPTH_MIN_EVALUABLE_PIXELS:
            continue
        bias = float(np.median(residual[supported]))
        candidates.append((support_ratio, bias, frame, supported))
        public_candidates.append(
            {
                "timestamp": frame.stamp.to_dict(),
                "support_pixels": support_pixels,
                "support_ratio": support_ratio,
                "bias_m": bias,
            }
        )
    if not candidates:
        return 0.0, template.mask.copy(), {
            "reason": "no supported calibration frame",
            "candidates": [],
        }
    support_ratio, raw_bias, reference_frame, broad_support = max(
        candidates, key=lambda item: (item[0], -abs(item[1]))
    )
    bias = float(
        np.clip(
            raw_bias,
            -settings.MODEL_DEPTH_MAX_ABS_BIAS_M,
            settings.MODEL_DEPTH_MAX_ABS_BIAS_M,
        )
    )
    assert reference_frame.depth_path is not None
    reference_depth = read_depth_image_m(reference_frame.depth_path)
    reference_mask = (
        mask
        & np.isfinite(reference_depth)
        & (reference_depth > 0.0)
        & (
            np.abs(reference_depth - (template.front_depth_m + bias))
            <= settings.MODEL_DEPTH_PRESENT_TOLERANCE_M
        )
    )
    if np.count_nonzero(reference_mask) < settings.MODEL_DEPTH_MIN_EVALUABLE_PIXELS:
        reference_mask = broad_support
    selected = {
        "timestamp": reference_frame.stamp.to_dict(),
        "support_ratio": support_ratio,
        "bias_m": raw_bias,
        "reference_mask_pixels": int(np.count_nonzero(reference_mask)),
        "reference_mask_ratio": float(np.count_nonzero(reference_mask)) / max(1, int(np.count_nonzero(mask))),
    }
    return bias, reference_mask, {
        "selected": selected,
        "bias_m": bias,
        "present_tolerance_m": settings.MODEL_DEPTH_PRESENT_TOLERANCE_M,
        "candidates": public_candidates,
    }

class DynamicDepthMaskTracker:
    """Maintain observed model support while excluding current foreground occlusion."""

    def __init__(
        self,
        template: ModelDepthTemplate,
        *,
        initial_support_mask: np.ndarray,
        depth_bias_m: float,
    ) -> None:
        self.template = template
        self.depth_bias_m = float(depth_bias_m)
        self.support_mask = np.asarray(initial_support_mask, dtype=bool).copy()
        if self.support_mask.shape != template.mask.shape:
            raise ValueError("initial support mask shape does not match model template")
        self.support_mask &= template.mask
        self.current_unoccluded_mask = self.support_mask.copy()

    def update(
        self,
        frame: ShigureFrame,
        *,
        allow_support_update: bool = True,
    ) -> DepthFrameDecision:
        if frame.depth_path is None or not frame.depth_path.is_file():
            raise FileNotFoundError("tracking frame is missing depth")
        depth = read_depth_image_m(frame.depth_path)
        if depth.shape != self.template.mask.shape:
            raise ValueError(
                f"depth shape {depth.shape} does not match model template "
                f"{self.template.mask.shape}"
            )

        front = self.template.front_depth_m + self.depth_bias_m
        full_valid = self.template.mask & np.isfinite(depth) & (depth > 0.0)
        front_match = full_valid & (
            np.abs(depth - front) <= settings.MODEL_DEPTH_PRESENT_TOLERANCE_M
        )
        support_before = int(np.count_nonzero(self.support_mask))
        if allow_support_update:
            self.support_mask |= front_match
        support_pixels = int(np.count_nonzero(self.support_mask))
        added_support_pixels = max(0, support_pixels - support_before)

        valid = self.support_mask & full_valid
        occluded = valid & (depth < front - settings.MODEL_DEPTH_OCCLUSION_MARGIN_M)
        evaluable = valid & ~occluded
        self.current_unoccluded_mask = evaluable.copy()
        removed = evaluable & (
            depth > front + settings.MODEL_DEPTH_REMOVAL_MARGIN_M
        )
        present = evaluable & ~removed

        model_pixels = support_pixels
        valid_pixels = int(np.count_nonzero(valid))
        evaluable_pixels = int(np.count_nonzero(evaluable))
        removed_pixels = int(np.count_nonzero(removed))
        present_pixels = int(np.count_nonzero(present))
        occluded_pixels = int(np.count_nonzero(occluded))
        evaluable_ratio = evaluable_pixels / max(1, model_pixels)
        removed_ratio = removed_pixels / max(1, evaluable_pixels)
        present_ratio = present_pixels / max(1, evaluable_pixels)
        occluded_ratio = occluded_pixels / max(1, valid_pixels)
        candidate = (
            evaluable_pixels >= settings.MODEL_DEPTH_MIN_EVALUABLE_PIXELS
            and evaluable_ratio >= settings.MODEL_DEPTH_MIN_EVALUABLE_RATIO
            and removed_ratio >= settings.MODEL_DEPTH_REMOVED_RATIO
            and present_ratio <= settings.MODEL_DEPTH_MAX_PRESENT_RATIO
        )
        median_removal_excess = (
            float(np.median(depth[removed] - front[removed])) if removed_pixels else None
        )
        if candidate:
            status = "removed_candidate"
        elif evaluable_ratio < settings.MODEL_DEPTH_MIN_EVALUABLE_RATIO:
            status = "occluded_or_unknown"
        else:
            status = "present"
        return DepthFrameDecision(
            timestamp=frame.stamp,
            status=status,
            candidate=candidate,
            model_pixels=model_pixels,
            valid_pixels=valid_pixels,
            evaluable_pixels=evaluable_pixels,
            present_pixels=present_pixels,
            removed_pixels=removed_pixels,
            occluded_pixels=occluded_pixels,
            support_pixels=support_pixels,
            added_support_pixels=added_support_pixels,
            evaluable_ratio=evaluable_ratio,
            removed_ratio=removed_ratio,
            present_ratio=present_ratio,
            occluded_ratio=occluded_ratio,
            median_removal_excess_m=median_removal_excess,
            removed_mask=removed,
            unoccluded_mask=evaluable,
        )


def classify_depth_frame(
    frame: ShigureFrame,
    template: ModelDepthTemplate,
    *,
    tracking_mask: np.ndarray,
    depth_bias_m: float,
) -> DepthFrameDecision:
    tracker = DynamicDepthMaskTracker(
        template,
        initial_support_mask=tracking_mask,
        depth_bias_m=depth_bias_m,
    )
    return tracker.update(frame, allow_support_update=False)
