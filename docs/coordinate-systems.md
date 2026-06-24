# Coordinate Systems

All server-side reconstruction, alignment, and pose-composition math should use a
single internal coordinate system: `canonical_rh`. Other coordinate systems are
interface contracts only. Definitions and conversion matrices live in
`code/coordinate_systems.py`; do not add new hard-coded axis flips elsewhere.

The HoloLens depth-to-RGB registration code under
`code/Hololens2/DepthConvertToRGB` is intentionally left outside this refactor.
Treat its output as an input boundary to the server reconstruction stages.

## Internal System

### `canonical_rh`

This is the only coordinate system used for local server computation.

- Handedness: right-handed
- `+X`: camera right
- `+Y`: camera up
- `-Z`: camera/object forward
- Typical values: objects in front of the camera have negative `Z`
- Used by: depth point clouds after ingestion, model points during alignment,
  ICP/camera-refine/object-alignment math, camera-local solved object poses

Column-vector pose math is used after data enters the server stages:

```text
p_parent = R_parent_child * p_child + t_parent_child
```

## Boundary Systems

### `opencv_camera`

- Handedness: right-handed
- `+X`: image/camera right
- `+Y`: image/camera down
- `+Z`: camera forward
- Used by: OpenCV, ArUco detection, FoundationPose output, Shigurei/ROS image
  projection boundary
- Conversion to internal: `diag(1, -1, -1)`

### `unity`

- Handedness: left-handed
- `+X`: right
- `+Y`: up
- `+Z`: forward
- Used by: Unity/HoloLens runtime payloads, `PVCamera.position`,
  `PVCamera.rotation_quaternion_xyzw`, `object_world`, `object_aruco`, model
  bounds exposed to the app
- Conversion from internal: `diag(1, 1, -1)`

### `windows_spatial`

- Handedness: right-handed
- `+X`: right
- `+Y`: up
- `+Z`: backward relative to Unity forward
- Used by: raw HoloLens/hl2ss PV pose matrices at upload/API ingress
- Note: legacy PV matrices store translation in the last row and apply local
  points as row vectors. The boundary conversion transposes rotation before
  emitting Unity-style column-vector pose components.

### `model_input`

- Handedness: right-handed
- `-X`: model forward
- `+Y`: model right
- `+Z`: model up
- Used by: generated OBJ vertices before runtime mesh baking
- Conversion to internal: model `[x, y, z]` becomes canonical `[y, z, x]`

### `unity_runtime_local`

- RuntimeMesh preserves the generated OBJ/model-input local axes.
- Used by: RuntimeMesh OBJ/FBX assets consumed by Unity/TriLib.
- Conversion from `model_input`: `runtime_xyz = source_xyz`.
- Pose output applies `RUNTIME_LOCAL_TO_UNITY_POSE_ROTATION` so downloaded
  models keep the same HoloLens orientation as the pre-refactor pipeline.
- Face winding is not flipped at this stage because no handedness-changing
  vertex transform is applied.

### `blender_world`

- Handedness: right-handed
- `+X`: right
- `+Y`: forward
- `+Z`: up
- Used by: Blender preview renders, OBJ import/export staging, FBX wrapping
- Conversion from internal: canonical `[x, y, z]` becomes Blender `[x, -z, y]`

### `pointcloud_export`

- Handedness: right-handed export basis chosen for Blender import
- `+X`: forward
- `+Y`: up
- `+Z`: left
- Used by: binary PLY debug/export files
- Conversion to internal: export `[x, y, z]` becomes canonical `[-z, y, x]`

### `aruco`

- Runtime/storage coordinate system follows Unity-style pose components after
  ArUco synchronization.
- Used by: `object_aruco`, `ModelBounds.coordinate_space == "aruco"`, spatial
  ray queries against stored model bounds
- OpenCV marker detections are converted at the ArUco boundary before world or
  ArUco-local poses are composed.

## Pose And Asset Contracts

### Runtime Pose Output

`run_pose_from_json.py` emits `object_world` in Unity-style world coordinates.
The alignment stage solves a camera-local pose in `canonical_rh`; the pose stage
first converts that local pose to Unity camera coordinates, then composes it
with the PV camera world pose.

```text
R_local_unity, t_local_unity =
  model_pose_canonical_rh_to_unity_camera(R_local_canonical, t_local_canonical)

t_world_object = R_world_camera_raw * t_local_unity + t_world_camera
R_world_object = R_world_camera_filtered * R_local_unity * R_runtime_local_to_unity
```

`R_runtime_local_to_unity` currently resolves to:

```text
FBX_RUNTIME_TRANSFORM_COMPENSATION_TO_UNITY @ RUNTIME_LOCAL_TO_UNITY_POSE_ROTATION

RUNTIME_LOCAL_TO_UNITY_POSE_ROTATION =
[[0, 1, 0],
 [0, 0, 1],
 [1, 0, 0]]
```

For `OBJECT_ALIGNMENT_MODE == "off"`, `R_world_camera_filtered` can be reduced
by the `SKIP_ICP_POSE_USE_CAMERA_YAW`, `SKIP_ICP_POSE_USE_CAMERA_PITCH`, and
`SKIP_ICP_POSE_USE_CAMERA_ROLL` settings. For FoundationPose and camera-refine
alignment modes, the full PV camera rotation is used.

### Runtime Mesh And FBX Export

RuntimeMesh preserves generated model-input axes:

```text
runtime_xyz = source_xyz
MODEL_INPUT_TO_UNITY_RUNTIME_LOCAL = identity
```

FBX export is a Blender boundary. Runtime OBJ/MTL/PNG sources are imported with:

```text
forward_axis = NEGATIVE_Z
up_axis = Y
```

and exported as FBX with:

```text
axis_forward = -Z
axis_up = Y
bake_space_transform = True
```

The FBX stage applies `object_world.scale` to the exported mesh objects before
writing the FBX. Verification renders that import the FBX back into Blender
therefore treat FBX geometry as already in the runtime scale contract; they do
not multiply `object_world.scale` into the overlay placement matrix again.

## FBX On ROS RGB Overlay Verification

`code/.test/overlay_fbx_models_on_ros_rgb.py` renders completed task FBX files
onto a saved Shigurei/ROS RGB frame. The current trustworthy output relies on
the following transform chain.

1. `object_world` and `aruco_reference` are Unity-style world poses.
2. The object pose is first expressed relative to the reference marker:

```text
R_marker_object_unity = R_world_marker.T * R_world_object
t_marker_object_unity = R_world_marker.T * (t_world_object - t_world_marker)
```

3. The marker-local Unity pose is converted to OpenCV marker coordinates with:

```text
B_unity_to_opencv_camera = diag(1, -1, 1)

R_marker_object_cv = B_unity_to_opencv_camera * R_marker_object_unity * B_unity_to_opencv_camera.T
t_marker_object_cv = B_unity_to_opencv_camera * t_marker_object_unity
```

4. The current ROS snapshot marker pose comes from
   `marker_6d_pose.json["opencv_camera_pose"]` and is already in OpenCV camera
   coordinates. The object is placed in the current RGB camera frame by:

```text
R_camera_object_cv = R_camera_marker_cv * R_marker_object_cv
t_camera_object_cv = R_camera_marker_cv * t_marker_object_cv + t_camera_marker_cv
```

5. Blender's camera is configured at the world origin with its default optical
   direction (`-Z`) and the ROS RGB intrinsics from
   `rs_aligned_depth_to_color_cameraInfo.json`. The OpenCV camera pose is
   converted into Blender camera/world coordinates with:

```text
B_opencv_to_blender_camera = diag(1, -1, -1)
t_blender_object = B_opencv_to_blender_camera * t_camera_object_cv
```

6. Re-imported FBX local axes are compensated before assigning
   `matrix_world`. The overlay script currently uses:

```text
FBX_IMPORTED_LOCAL_FROM_RUNTIME =
[[-1,  0,  0],
 [ 0,  0, -1],
 [ 0, -1,  0]]

R_blender_imported_object =
  B_opencv_to_blender_camera * R_camera_object_cv * FBX_IMPORTED_LOCAL_FROM_RUNTIME.T
```

7. Optional depth occlusion uses
   `rs_aligned_depth_to_color_compressedDepth.png` and a Blender ray-cast depth
   buffer. Model pixels are hidden when:

```text
model_depth_m > observed_depth_m + occlusion_depth_margin_m
```

The default overlay margin is `0.05 m`.

Notes for this verification path:

- ROS RGB/depth frames are treated as the `opencv_camera` boundary:
  `+X` right, `+Y` down, `+Z` forward.
- Blender render projection uses the pinhole camera matrix `K`. Distortion
  coefficients are saved in the debug config, but the Blender render path does
  not apply lens distortion.
- The overlay script repeats a few boundary matrices locally because it is a
  test/debug utility. Production stage code should continue importing shared
  conversions from `code/coordinate_systems.py`.

## Taken Object Detection Projection Boundary

The current taken-object path lives in
`code/stages/taken_object_detection/run_taken_object_detection_from_json.py`. It
uses Shigurei RGB-D history as the OpenCV camera boundary (`+X` right, `+Y`
down, `+Z` forward). The default mode is `TAKEN_OBJECT_TRACKING_MODE=yolo_primary`:
ArUco/marker history projects the modeled object center into the Shigurei image,
YOLO `/tracking/active_objects` provides a candidate mask/id, and Shigurei depth
confirms occlusion or removal.

The legacy projected-mask path still exists behind explicit settings
(`TAKEN_OBJECT_TRACKING_MODE=legacy_projected_mask` and
`TAKEN_OBJECT_ENABLE_LEGACY_PROJECTED_MASK=true`) or as a fallback when
`TAKEN_OBJECT_ENABLE_LEGACY_FALLBACK=true`, but it is not the default coordinate
boundary.

After YOLO-first initialization, the trusted YOLO mask and its valid depth become
the reference. For each aligned depth frame, the implementation compares observed
depth against the cached trusted depth:

```text
delta[p] = current_depth[p] - cached_depth[p]
occluded = valid_depth AND delta <= TAKEN_OBJECT_OCCLUSION_DELTA_M
taken = valid_depth AND delta >= TAKEN_OBJECT_TAKEN_DELTA_M
```

A result is confirmed only after `TAKEN_OBJECT_TAKEN_RATIO` is satisfied for
`TAKEN_OBJECT_TAKEN_CONSECUTIVE_FRAMES` consecutive frames. Full foreground
occlusion is tracked separately with `TAKEN_OBJECT_FULL_OCCLUSION_RATIO` and
does not by itself mean the object was taken. RGB backtracking is used to choose
a better result/display frame, while depth remains the confirmation signal.

Current tuning lives in `code/stages/taken_object_detection/settings.py` and uses
`TAKEN_OBJECT_*` environment variables. The selected result frame is backed up
under the model task worker directory and copied into `result/07_taken_detection_*`
for `sam3d_body_mesh`, which then uses real Shigurei CameraInfo rather than
re-estimating camera intrinsics.

## Current Usage Map

- `code/coordinate_systems.py`: single source of truth for axis definitions,
  basis matrices, runtime-axis contract strings, and conversion helpers.
- `code/object_alignment_common.py`: alignment/image/mesh utilities; imports and
  re-exports coordinate helpers from `coordinate_systems.py`.
- `code/stages/hololens3d_reconstruction/run_object_alignment_from_json.py`:
  computes in `canonical_rh`; converts FoundationPose/OpenCV results at the
  boundary.
- `code/stages/hololens3d_reconstruction/run_pose_from_json.py`: converts the
  solved camera-local `canonical_rh` pose to Unity/HoloLens output.
- `code/stages/hololens3d_reconstruction/bake_runtime_mesh.py`: builds the
  runtime OBJ/MTL/texture while preserving generated model axes.
- `code/stages/hololens_aruco_reference/aruco_common.py`: converts OpenCV ArUco
  marker poses and HoloLens PV poses at the boundary.
- `code/server_api.py`: converts raw uploaded HoloLens PV matrices before
  responding to clients.
- `code/model_bounds.py`: reads runtime-local mesh vertices and stores bounds in
  ArUco/Unity-style pose space.
- `code/stages/taken_object_detection/run_taken_object_detection_from_json.py`:
  uses Shigurei OpenCV-camera RGB-D frames, YOLO-first object association, cached trusted depth, and current `TAKEN_OBJECT_*` thresholds for taken-object detection.
- `code/stages/sam3d_body_mesh/run_sam3d_body_mesh_from_json.py`: consumes the
  backed-up Shigurei result frame and CameraInfo produced by taken-object
  detection.
- `code/.test/overlay_fbx_models_on_ros_rgb.py`: debug verification renderer
  for placing exported FBX assets back onto saved ROS RGB frames through the
  ArUco/OpenCV/Blender chain above.
- `code/Hololens2/DepthConvertToRGB`: depth/RGB registration boundary; not
  changed by this coordinate-system cleanup.

## Related Settings

- `OBJECT_ALIGNMENT_MODE`: selects FoundationPose, camera-refine, or placement
  without ICP. This affects the solved camera-local pose and, for `off`, whether
  camera yaw/pitch/roll filtering is applied in the pose stage.
- `SKIP_ICP_POSE_USE_CAMERA_YAW`, `SKIP_ICP_POSE_USE_CAMERA_PITCH`,
  `SKIP_ICP_POSE_USE_CAMERA_ROLL`: only affect pose-stage camera rotation
  filtering when `OBJECT_ALIGNMENT_MODE == "off"`.
- `ARUCO_ANCHOR_MARKER_ID`: chooses the anchor marker whose world pose becomes
  the runtime ArUco reference.
- `ARUCO_SYNC_MARKER_REGISTRY_ON_START`: controls whether marker registry data
  is synced on server startup; pose conversion still uses the latest ArUco
  reference for the task's startup session.
- `RUNTIME_MESH_DECIMATE_RATIO`, `MODEL_FBX_DECIMATE_RATIO`, and
  `SAM3D_OBJECTS_FBX_DECIMATE_RATIO`: change mesh density only; they do not
  change axis contracts.
- `BLENDER_BIN` and `BLENDER_FBX_DIR`: select the Blender executable and FBX
  output location for the server stages. The overlay script also accepts a
  `--blender` override and writes debug artifacts under
  `code/.test/overlay_fbx_multi/`.

## Rules For New Code

1. Do local reconstruction/alignment math in `canonical_rh`.
2. Convert OpenCV/ROS/FoundationPose data into `canonical_rh` immediately after
   ingestion.
3. Convert to Unity/HoloLens only when emitting runtime/API/task JSON payloads.
4. Convert to Blender/FBX only inside Blender import/export stages.
5. Import matrices and helpers from `code/coordinate_systems.py`; do not create
   ad hoc `diag(...)` flips or manual `[x, -y, z]` transforms in stage code.
