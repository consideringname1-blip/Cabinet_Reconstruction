import numpy as np
import time
import hl2ss
import hl2ss_3dcv
import os
import json
import glob
import cv2

# Settings --------------------------------------------------------------------

# Calibration path (must exist but can be empty)
calibration_path = '/home/fumiya/ros2_ws/src/ros2_test_communicate/ros2_test_communicate/calibration/'
# data_path = '/home/fumiya/ros2_ws/hololens_dataset/'
base_data_path = '/home/fumiya/ros2_ws/hololens_dataset/20251003/rimokon'

# Camera parameters
pv_width = 640
pv_height = 360
pv_framerate = 30
pv_exposure_mode = hl2ss.PV_ExposureMode.Manual
pv_exposure = 5990

def get_homogeneous_component(array):
    return array[..., -1, np.newaxis]

def get_inhomogeneous_component(array):
    return array[..., 0:-1]

def to_inhomogeneous(array):
    return get_inhomogeneous_component(array) / get_homogeneous_component(array)

def slice_to_block(slice):
    return slice[:, :, np.newaxis]

def transform(points, transform4x4):
    return points @ transform4x4[:3, :3] + transform4x4[3, :3].reshape(([1] * (len(points.shape) - 1)).append(3))

def project(points, projection4x4):
    return to_inhomogeneous(transform(points, projection4x4))

def block_to_list(points):
    return points.reshape((-1, points.shape[-1]))

#------------------------------------------------------------------------------

if __name__ == '__main__':
    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    # calibration_lt = hl2ss_3dcv.get_calibration_rm(calibration_path, host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
    calibration_lt = hl2ss_3dcv.get_calibration_rm2(calibration_path, hl2ss.StreamPort.RM_DEPTH_AHAT)
    import pdb; pdb.set_trace()

    #uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    uv2xy = calibration_lt.uv2xy
    # print('uv2xy: %s' % uv2xy)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
    # print('xy1: %s, scale: %s' % (xy1, scale))

    xy1_o = xy1[:-1, :-1, :]
    xy1_d = xy1[1:, 1:, :]

    # Start PV and RM Depth Long Throw streams --------------------------------
    # sink_pv = hl2ss_mp.stream(hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate, decoded_format='rgb24'))
    # # sink_depth = hl2ss_mp.stream(hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    # sink_depth = hl2ss_mp.stream(hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT))

    # sink_pv.open()
    # sink_depth.open()

    last_fs = -1

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss_3dcv.pv_create_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    color_frames = []
    depth_frames = []
    fps_hist = []

    before_time = time.time()
 
    pv_pose_hist = []
    lt_pose_hist = []

    # frame_で始まるディレクトリを検索
    pattern = os.path.join(base_data_path, "frame_*")
    potential_dirs = glob.glob(pattern, recursive=False)

    # Main Loop ---------------------------------------------------------------
    for data_path in potential_dirs:
        print('Processing directory: %s' % data_path)
        color_path = os.path.join(data_path, 'color.png')
        depth_path = os.path.join(data_path, 'depth.png')
        color_intrinsics_path = os.path.join(data_path, 'color_intrinsics.json')
        pose_path = os.path.join(data_path, 'poses.json')

        # Preprocess frames ---------------------------------------------------
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        z     = slice_to_block(depth / scale)
        color = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_RGB2BGR)
        # print(np.mean(scale), np.min(scale), np.max(scale))
        # print('Original Depth type: %s, shape : %s, max : %s' % (type(depth), str(depth.shape), np.max(depth)))
        # print('Scaled Depth type: %s, shape : %s, max : %s' % (type(z), str(z.shape), np.max(z)))

        with open(color_intrinsics_path, 'r') as f:
            color_intrinsics_data = json.load(f)
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)

        # Update PV intrinsics ------------------------------------------------
        # PV intrinsics may change between frames due to autofocus
        pv_intrinsics = hl2ss_3dcv.pv_update_intrinsics(pv_intrinsics, [color_intrinsics_data['k'][0], color_intrinsics_data['k'][4]], [color_intrinsics_data['k'][2], color_intrinsics_data['k'][5]])
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

        lt_pose_zero = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        # print('focal length: %s, principal point: %s, lt_pose: %s, pv_pose: %s' % (data_pv.payload.focal_length, data_pv.payload.principal_point, data_lt.pose, data_pv.pose))
        # print('pv_intrinsics: %s, pv_extrinsics: %s, ahat_extrinsics: %s' % (color_intrinsics, color_extrinsics, calibration_lt.extrinsics))
        
        # Generate aligned RGBD image -----------------------------------------
        # pv_pose_hist.append(data_pv.pose)
        # lt_pose_hist.append(data_lt.pose)
        # print('pv_pose_mean: %s,\n lt_pose_mean: %s' % (np.mean(pv_pose_hist, axis=0), np.mean(lt_pose_hist, axis=0)))
        # print('pv_pose_var: %s,\n lt_pose_var: %s' % (np.var(pv_pose_hist, axis=0), np.var(lt_pose_hist, axis=0)))
        
        # lt_to_world    = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
        # world_to_pv    = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics)
        # pv_to_pv_image = hl2ss_3dcv.camera_to_image(color_intrinsics)
        lt_to_world    = np.linalg.inv(calibration_lt.extrinsics) @ np.array(pose_data['depth_pose']).reshape(4, 4)
        world_to_pv    = np.linalg.inv(np.array(pose_data['color_pose']).reshape(4, 4)) @ color_extrinsics
        pv_to_pv_image = color_intrinsics
        # print('data_lt.pose - data_pv.pose : ', data_lt.pose - data_pv.pose)

        # # 深度カメラ→リグノード
        # lt_to_rig    = np.linalg.inv(calibration_lt.extrinsics)
        # # カラーカメラ→リグノード
        # rig_to_pv    = color_extrinsics
        # pv_to_pv_image = color_intrinsics
        # lt_to_pv = lt_to_rig @ rig_to_pv

        # 3D点群生成（深度カメラ座標系）
        lt_points_o    = xy1_o * z[:-1, :-1, :]
        lt_points_d    = xy1_d * z[:-1, :-1, :]

        # lt_points_o    = hl2ss_3dcv.rm_depth_to_points(xy1_o, z[:-1, :-1, :])
        world_points_o = hl2ss_3dcv.transform(lt_points_o, lt_to_world)
        pv_points_o    = hl2ss_3dcv.transform(world_points_o, world_to_pv)
        pv_uv_o        = hl2ss_3dcv.project(pv_points_o, pv_to_pv_image)

        # # まずリグノード座標系へ
        # rig_points_o = transform(lt_points_o, lt_to_rig)
        # rig_points_d = transform(lt_points_d, lt_to_rig)

        # # 次にカラーカメラ座標系へ
        # pv_points_o    = transform(rig_points_o, rig_to_pv)
        # pv_points_d    = transform(rig_points_d, rig_to_pv)

        # pv_points_o    = transform(lt_points_o, lt_to_pv)
        # pv_points_d    = transform(lt_points_d, lt_to_pv)

        # # カラーカメラ座標系へ変換
        pv_depth       = pv_points_o[:, :, 2:]
        # pv_uv_o        = project(pv_points_o, pv_to_pv_image)
        # pv_uv_d        = project(pv_points_d, pv_to_pv_image)

        # lt_points_d    = hl2ss_3dcv.rm_depth_to_points(xy1_d, z[:-1, :-1, :])
        world_points_d = hl2ss_3dcv.transform(lt_points_d, lt_to_world)
        pv_uv_d        = hl2ss_3dcv.project(world_points_d, world_to_pv @ pv_to_pv_image)

        # pv_list_o     = hl2ss_3dcv.block_to_list(pv_uv_o)
        # pv_list_d     = hl2ss_3dcv.block_to_list(pv_uv_d)
        # pv_list_depth = hl2ss_3dcv.block_to_list(pv_depth)
        pv_list_o     = block_to_list(pv_uv_o)
        pv_list_d     = block_to_list(pv_uv_d)
        pv_list_depth = block_to_list(pv_depth)

        mask = (depth[:-1,:-1].reshape((-1,)) > 0)

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


        # print(pv_z)

        # fps = 1 / (time.time() - before_time)
        # before_time = time.time()
        # print('fps: %s' % fps)
        # fps_hist.append(fps)

        print('Color type: %s, shape : %s' % (type(color), str(color.shape)))
        cv2.imwrite(os.path.join(data_path, 'align_color.png'), cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        # cv2.imshow('color', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        # color_frames.append(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        print('color saved!')
        print('Depth type: %s, shape : %s, max : %s' % (type(pv_z), str(pv_z.shape), np.max(pv_z)))
        # enhance depth around object
        # align_depth = np.where(pv_z > 0.7, 0.7, pv_z)
        # align_depth = np.where(align_depth < 0.5, 0.5, align_depth)
        # align_depth = align_depth - 0.5
        # align_depth = (align_depth / 0.2 * 256).astype(np.uint8)

        align_depth = (pv_z * 1000).astype(np.uint16)
        # print('Align Depth type: %s, shape : %s, max : %s' % (type(align_depth), str(align_depth.shape), np.max(align_depth)))
        # align_depth = cv2.applyColorMap(align_depth, cv2.COLORMAP_TURBO)
        # cv2.imshow('depth', align_depth)
        cv2.imwrite(os.path.join(data_path, 'align_depth.png'), align_depth)

        align_depth_turbo = (pv_z * 256).astype(np.uint8)
        align_depth_turbo = cv2.applyColorMap(align_depth_turbo, cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(data_path, 'align_depth_turbo.png'), align_depth_turbo)
        # depth_frames.append(align_depth)
        print('depth saved!')