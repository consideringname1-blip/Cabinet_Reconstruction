import numpy as np
import hl2ss, hl2ss_3dcv
import os
import json
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TASK_DEBUG_OUTPUT_ENABLE
from path_config import CALIBRATION_DIR
from artifact_layout import model_debug_file, model_worker_file
from depth_camera_config import (
    DEPTH_SENSOR_AHAT,
    DEPTH_SENSOR_LONGTHROW,
    get_depth_sensor_limits,
    normalize_depth_sensor_name,
)


def depth_sensor_stream_port(sensor_name: str):
    sensor_name = normalize_depth_sensor_name(sensor_name)
    if sensor_name == DEPTH_SENSOR_AHAT:
        return hl2ss.StreamPort.RM_DEPTH_AHAT
    if sensor_name == DEPTH_SENSOR_LONGTHROW:
        return hl2ss.StreamPort.RM_DEPTH_LONGTHROW
    raise ValueError(f"Unsupported depth sensor: {sensor_name}")

def get_homogeneous_component(array):
    return array[..., -1, np.newaxis]

def get_inhomogeneous_component(array):
    return array[..., 0:-1]

def to_inhomogeneous(array):
    return get_inhomogeneous_component(array) / get_homogeneous_component(array)

def slice_to_block(slice):
    return slice[:, :, np.newaxis]

def block_to_list(points):
    return points.reshape((-1, points.shape[-1]))

#------------------------------------------------------------------------------

def DepthConvertToRGB(json_path, calibration_base_path=CALIBRATION_DIR):
    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    calibration_path = calibration_base_path / data["device"]["ip"]

    depth_camera_info = data.get("DepthCamera") or {}
    depth_sensor = normalize_depth_sensor_name(depth_camera_info.get("sensor"))
    depth_limits = get_depth_sensor_limits(depth_sensor)
    depth_stream_port = depth_sensor_stream_port(depth_sensor)

    calibration_depth = hl2ss_3dcv.get_calibration_rm2(calibration_path, depth_stream_port)
    uv2xy = calibration_depth.uv2xy
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_depth.scale)
    # print('xy1: %s, scale: %s' % (xy1, scale))

    xy1_o = xy1[:-1, :-1, :]
    xy1_d = xy1[1:, 1:, :]

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss_3dcv.pv_create_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    # Main Loop ---------------------------------------------------------------
    task_timestamp = str(data.get("task_timestamp") or "").strip()
    if not task_timestamp:
        raise ValueError("task_timestamp is required for depth alignment artifacts")
    depth_path = model_worker_file(task_timestamp, "input.depth")

    # Preprocess frames ---------------------------------------------------
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    z     = slice_to_block(depth / scale)

    color_intrinsics_data = data["PVCamera"]

    # Update PV intrinsics ------------------------------------------------
    # PV intrinsics may change between frames due to autofocus
    k = color_intrinsics_data['k']

    pv_intrinsics = hl2ss_3dcv.pv_update_intrinsics(
        pv_intrinsics,
        [k[0][0], k[1][1]],
        [k[0][2], k[1][2]]
    )
    
    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

    lt_to_world    = np.linalg.inv(calibration_depth.extrinsics) @ np.array(depth_camera_info["pose"]).reshape(4, 4)
    world_to_pv    = np.linalg.inv(np.array(data["PVCamera"]["pose"]).reshape(4, 4)) @ color_extrinsics
    pv_to_pv_image = color_intrinsics

    # 3D点群生成（深度カメラ座標系）
    lt_points_o    = xy1_o * z[:-1, :-1, :]
    lt_points_d    = xy1_d * z[:-1, :-1, :]

    # lt_points_o    = hl2ss_3dcv.rm_depth_to_points(xy1_o, z[:-1, :-1, :])
    world_points_o = hl2ss_3dcv.transform(lt_points_o, lt_to_world)
    pv_points_o    = hl2ss_3dcv.transform(world_points_o, world_to_pv)
    pv_uv_o        = hl2ss_3dcv.project(pv_points_o, pv_to_pv_image)

    # # カラーカメラ座標系へ変換
    pv_depth       = pv_points_o[:, :, 2:]

    world_points_d = hl2ss_3dcv.transform(lt_points_d, lt_to_world)
    pv_uv_d        = hl2ss_3dcv.project(world_points_d, world_to_pv @ pv_to_pv_image)

    pv_list_o     = block_to_list(pv_uv_o)
    pv_list_d     = block_to_list(pv_uv_d)
    pv_list_depth = block_to_list(pv_depth)

    mask = (depth[:-1,:-1].reshape((-1,)) > 0)

    pv_height = data["PVCamera"]["height"]
    pv_width = data["PVCamera"]["width"]

    pv_list = np.hstack((np.floor(pv_list_o[mask, :]), np.floor(pv_list_d[mask, :]) + 1, pv_list_depth[mask]))
    pv_z    = np.zeros((pv_height, pv_width), dtype=np.float32)

    for n in range(0, pv_list.shape[0]):
        u0 = int(pv_list[n, 0])
        v0 = int(pv_list[n, 1])
        u1 = int(pv_list[n, 2])
        v1 = int(pv_list[n, 3])

        if ((u0 < 0) or (u0 >= pv_width)):
            continue
        if ((u1 < 0) or (u1 > pv_width)):
            continue
        if ((v0 < 0) or (v0 >= pv_height)):
            continue
        if ((v1 < 0) or (v1 > pv_height)):
            continue

        pv_z[v0:v1, u0:u1] = pv_list[n, 4]

    print('Depth type: %s, shape : %s, max : %s' % (type(pv_z), str(pv_z.shape), np.max(pv_z)))

    align_depth = (pv_z * 1000).astype(np.uint16)
    valid_align_mask = (
        (align_depth >= depth_limits.min_depth_mm)
        & (align_depth <= depth_limits.max_reliable_depth_mm)
    )
    align_depth = np.where(valid_align_mask, align_depth, 0).astype(np.uint16)
    align_depth_path = model_worker_file(task_timestamp, "input.align_depth")
    align_depth_path.parent.mkdir(parents=True, exist_ok=True)

    align_depth_name = align_depth_path.name
    data["DepthCamera"]["align_depth_name"] = align_depth_name
    data["DepthCamera"]["align_depth_stats"] = {
        "sensor": depth_sensor,
        "min_depth_mm": int(depth_limits.min_depth_mm),
        "max_reliable_depth_mm": int(depth_limits.max_reliable_depth_mm),
        "valid_depth_pixels": int(valid_align_mask.sum()),
    }
    cv2.imwrite(str(align_depth_path), align_depth)

    if TASK_DEBUG_OUTPUT_ENABLE:
        align_depth_turbo = (pv_z * 256).astype(np.uint8)
        align_depth_turbo = cv2.applyColorMap(align_depth_turbo, cv2.COLORMAP_TURBO)
        align_depth_turbo_path = model_debug_file(task_timestamp, "depth.align_turbo")
        align_depth_turbo_path.parent.mkdir(parents=True, exist_ok=True)
        data["DepthCamera"]["align_depth_turbo_name"] = align_depth_turbo_path.name
        cv2.imwrite(str(align_depth_turbo_path), align_depth_turbo)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print('depth saved!')

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("meta_path", type=str)
    args = ap.parse_args()

    DepthConvertToRGB(args.meta_path)
