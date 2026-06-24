# History Placement Coordinate Review

## Rotating polyhedron placement

HistoryPlacementRestorationDisplay.CreateDisplayForPayload first reads history_placement_restoration.display.polyhedron.pose_aruco. When that field is present, Unity does not use the sam3_spatial_box fallback. It transforms the ArUco-local pose into Unity world with:

world = current_aruco_world_position + current_aruco_world_rotation * pose_aruco.position

rotation = current_aruco_world_rotation * pose_aruco.rotation_quaternion_xyzw

The server builds that explicit pose in run_history_placement_restoration_from_json._polyhedron_pose. The old implementation incorrectly raised the pose by adding to ArUco-local Y. That is not generally Unity world-up: in the 20260624_182701_797036Z sample, ArUco-local +Y maps almost exactly to Unity world -Y, so adding local Y moved the cube downward in HoloLens world.

The current implementation first computes Unity world-up expressed in ArUco-local coordinates from task.aruco_reference.rotation:

local_up_aruco = aruco_reference_rotation.T * [0, 1, 0]

It then offsets either the original object pose or the matched current YOLO object pose along local_up_aruco by:

object_height_along_world_up * 0.5 + POLYHEDRON_ABOVE_MARGIN_M + POLYHEDRON_EDGE_LENGTH_M * 0.5

POLYHEDRON_ABOVE_MARGIN_M is intentionally small enough that the marker sits near the object top rather than floating high above it.

object_height_along_world_up is computed by projecting ModelBounds.corners_aruco onto local_up_aruco.

For MOVED, _build_display_payload attaches the cube to current_object, not to the original model. For MISSING, it attaches the polyhedron to original_model and sets show_model = true.

## Runtime object model pose

The generated model is solved in camera-local canonical RH coordinates by the alignment stage. run_pose_from_json.compute_world_pose converts that pose to Unity camera coordinates with model_pose_canonical_rh_to_unity_camera, then places it in HoloLens world:

world_position = pv_camera_rotation_raw * local_position + pv_camera_position

world_rotation = pv_camera_rotation_filtered * local_rotation * runtime_local_to_unity

The final object_world is also converted to object_aruco when an ArUco reference is available. Unity later prefers that ArUco-local pose so the same model can survive across app launches:

world = current_aruco_reference * object_aruco

## Runtime mesh and FBX axis contract

The normal object OBJ to FBX conversion uses Blender OBJ import forward_axis=NEGATIVE_Z, up_axis=Y and FBX export axis_forward=-Z, axis_up=Y, bake_space_transform=True. This import/export pair preserves the OBJ local axes in the FBX local asset space. The runtime pose then applies RUNTIME_LOCAL_TO_UNITY_POSE_ROTATION so the loaded asset matches the calibrated pose.

## SAM3D body mesh pose

The body mesh stage reconstructs vertices in Shigure/OpenCV camera coordinates, then converts them to ArUco/Unity coordinates:

marker_cv = R_marker.T * (point_camera - t_marker)

vertex_aruco = UNITY_TO_OPENCV_CAMERA_BASIS * marker_cv
vertex_aruco.z = -vertex_aruco.z

The Z mirror is intentionally applied for the HoloLens evidence-body display path because the reconstructed body otherwise appears left/right mirrored in runtime. Triangle winding is reversed when the mirror is enabled so the exported FBX keeps visible faces.

Those vertices are written directly into the selected body OBJ. Unity loads the body as an evidence overlay whose root object_aruco pose is the ArUco origin:

body_root_aruco = identity pose at [0, 0, 0]

Because the body OBJ already contains absolute ArUco/Unity coordinates, Unity must not translate it again by the object center. Its FBX export must preserve OBJ local axes exactly. The body exporter uses the same Blender import/export axis settings as the normal runtime model exporter.
