import numpy as np
from . import hl2ss, hl2ss_3dcv
import os
import json
import cv2
import json

from config import (
    CALIBRATION_DIR,
    UPLOAD_FOLDER,
    HOLOLENS2_OUTPUT_DEPTH_IMAGES,
)

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

def DepthConvertToRGB(json_path,calibration_base_path = CALIBRATION_DIR,data_base_path = UPLOAD_FOLDER):
    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    calibration_path = calibration_base_path / data["device"]["ip"]

    calibration_lt = hl2ss_3dcv.get_calibration_rm2(calibration_path, hl2ss.StreamPort.RM_DEPTH_AHAT)
    uv2xy = calibration_lt.uv2xy
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
    # print('xy1: %s, scale: %s' % (xy1, scale))

    xy1_o = xy1[:-1, :-1, :]
    xy1_d = xy1[1:, 1:, :]

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss_3dcv.pv_create_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    # Main Loop ---------------------------------------------------------------
    depth_path = data_base_path / data["DepthCamera"]["name"]

    # Preprocess frames ---------------------------------------------------
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    z     = slice_to_block(depth / scale)

    color_intrinsics_data = data["PVCamera"]

    # Update PV intrinsics ------------------------------------------------
    # PV intrinsics may change between frames due to autofocus
    pv_intrinsics = hl2ss_3dcv.pv_update_intrinsics(pv_intrinsics, [color_intrinsics_data['k'][0], color_intrinsics_data['k'][4]], [color_intrinsics_data['k'][2], color_intrinsics_data['k'][5]])
    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

    lt_to_world    = np.linalg.inv(calibration_lt.extrinsics) @ np.array(data["DepthCamera"]["pose"]).reshape(4, 4)
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
    align_depth_name = f"{data["task_name"]}_align_depth.png"
    data["DepthCamera"]["align_depth_name"] = align_depth_name
    cv2.imwrite(os.path.join(HOLOLENS2_OUTPUT_DEPTH_IMAGES, align_depth_name), align_depth)

    align_depth_turbo = (pv_z * 256).astype(np.uint8)
    align_depth_turbo = cv2.applyColorMap(align_depth_turbo, cv2.COLORMAP_TURBO)
    align_depth_turbo_name = f"{data["task_name"]}_align_depth_turbo.png"
    data["DepthCamera"]["align_depth_turbo_name"] = align_depth_turbo_name
    cv2.imwrite(os.path.join(HOLOLENS2_OUTPUT_DEPTH_IMAGES, align_depth_turbo_name), align_depth_turbo)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print('depth saved!')