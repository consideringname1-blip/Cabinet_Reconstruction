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

- Handedness: left-handed
- `+X`: right
- `+Y`: up
- `+Z`: forward
- Used by: RuntimeMesh OBJ/FBX assets consumed by Unity
- Conversion from `model_input`: `runtime_xyz = [source_y, source_z, -source_x]`
- RuntimeMesh flips face winding after this transform because the transform
  changes handedness.

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
- `code/stages/hololens3d_reconstruction/bake_runtime_mesh.py`: bakes generated
  model axes into `unity_runtime_local` before Unity loads the mesh.
- `code/stages/hololens_aruco_reference/aruco_common.py`: converts OpenCV ArUco
  marker poses and HoloLens PV poses at the boundary.
- `code/server_api.py`: converts raw uploaded HoloLens PV matrices before
  responding to clients.
- `code/model_bounds.py`: reads runtime-local mesh vertices and stores bounds in
  ArUco/Unity-style pose space.
- `code/Hololens2/DepthConvertToRGB`: depth/RGB registration boundary; not
  changed by this coordinate-system cleanup.

## Rules For New Code

1. Do local reconstruction/alignment math in `canonical_rh`.
2. Convert OpenCV/ROS/FoundationPose data into `canonical_rh` immediately after
   ingestion.
3. Convert to Unity/HoloLens only when emitting runtime/API/task JSON payloads.
4. Convert to Blender/FBX only inside Blender import/export stages.
5. Import matrices and helpers from `code/coordinate_systems.py`; do not create
   ad hoc `diag(...)` flips or manual `[x, -y, z]` transforms in stage code.
