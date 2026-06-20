# Shigurei Core ROS Messages

This document records the ROS2 topics and message formats used by
`code/reconstruction/shigure_core`, plus the runtime state observed on
2026-06-19 from `/workspace_whs`.

## Environment

Load the project environment before checking topics:

```bash
cd /workspace_whs
source ./setup_env.sh
ros2 topic list -t
```

The root `setup_env.sh` loads ROS2 Humble and
`code/ros2/shigure_recv_ws/install/setup.bash`. The receive workspace only
contains local message definitions for receiving Shigurei topics; it does not
launch `shigure_core`.

## Current Runtime Snapshot

Observed Shigurei/YOLO-related topics:

```text
/rs/color/compressed                         sensor_msgs/msg/CompressedImage
/rs/aligned_depth_to_color/compressedDepth   sensor_msgs/msg/CompressedImage
/rs/aligned_depth_to_color/cameraInfo        sensor_msgs/msg/CameraInfo
/openpose/pose_key_points                    openpose_ros2_msgs/msg/PoseKeyPointsList
/shigure/people_detection                    shigure_core_msgs/msg/PoseKeyPointsList
/shigure/object_detection                    shigure_core_msgs/msg/DetectedObjectList
/shigure/object_tracking                     shigure_core_msgs/msg/TrackedObjectList
/shigure/contacted                           shigure_core_msgs/msg/ContactedList
/bounding_boxes                              bboxes_ex_msgs/msg/BoundingBoxes
/Segments                                    bboxes_ex_msgs/msg/Segments
/tracking/active_objects                     std_msgs/msg/String
/tracking/object_event                       std_msgs/msg/String
/tracking/bring_event_confirmed              std_msgs/msg/String
/tracking/feature_vector                     std_msgs/msg/String
/debug/combined_view/compressed              sensor_msgs/msg/CompressedImage
```

Important runtime notes:

- `tracking_node_ros2pub` publishes `/tracking/active_objects` and
  `/debug/combined_view/compressed`; `/tracking/active_objects` is the current
  usable segmentation-mask feed.
- `people_tracking_node` publishes `/shigure/people_detection`; a sample was
  received, but `pose_key_points_list` was empty at that moment.
- `yolox_object_detection_node` publishes `/shigure/object_detection`, but it
  depends on `/bounding_boxes`, `/Segments`, `/shigure/people_detection`, RGB
  and CameraInfo synchronization. During the check, `/bounding_boxes` and
  `/Segments` had no publishers, so `/shigure/object_detection` produced no
  received sample within 6 seconds.
- `/bounding_boxes` currently has subscribers but no publisher.
- `/Segments` currently has subscribers but no publisher, and the installed
  receive workspace cannot find `bboxes_ex_msgs/msg/Segments.idl`.

There is a source/runtime mismatch around `Segments`: source files exist at
`shigure_core_msgs/msg/Segments.msg` and `shigure_core_msgs/msg/Segment.msg`,
but `node_yolox_object_detection.py` imports `Segments` from
`bboxes_ex_msgs.msg`, while the installed receive workspace currently cannot
show either `bboxes_ex_msgs/msg/Segments` or `shigure_core_msgs/msg/Segments`.

## RGB-D Topics

These are the stable inputs used by the existing `shigure_history` recorder.

`/rs/color/compressed`
: `sensor_msgs/msg/CompressedImage`. Observed `format: jpg`,
`frame_id: room_camera1`, with JPEG bytes in `data`.

`/rs/aligned_depth_to_color/compressedDepth`
: `sensor_msgs/msg/CompressedImage`. Depth is aligned to the color camera.

`/rs/aligned_depth_to_color/cameraInfo`
: `sensor_msgs/msg/CameraInfo`. Observed `height: 720`, `width: 1280`,
`frame_id: room_camera1`, with `d[5]`, `k[9]`, `r[9]`, and `p[12]`.

## New Tracking JSON Topics

The currently usable segmentation result is a JSON string, not a typed
`shigure_core_msgs` message.

### `/tracking/active_objects`

Type: `std_msgs/msg/String`

Observed JSON shape:

```json
{
  "event": "active_objects",
  "timestamp": "2026-06-19T22:31:30.554588",
  "frame_w": 1280,
  "frame_h": 720,
  "objects": [
    {
      "object_id": 1,
      "x": 587,
      "y": 582,
      "bbox": [205, 422, 876, 718],
      "mask_b64": "<base64 encoded png mask>"
    }
  ]
}
```

Fields:

- `timestamp`: publisher-side ISO-like time string, not a ROS header stamp.
- `frame_w` / `frame_h`: image resolution. Observed as `1280x720`.
- `objects`: active object list for the frame.
- `object_id`: numeric tracking ID.
- `x` / `y`: object center in pixels.
- `bbox`: `[xmin, ymin, xmax, ymax]` in pixels.
- `mask_b64`: base64-encoded PNG. Observed masks are full-frame
  `1280x720` 8-bit grayscale PNGs, not bbox crops.

The 2026-06-19 sample contained 54 objects. Decoded debug files from that
sample are under `.test/shigure_active_objects_latest/`; the summary is
`.test/shigure_active_objects_latest/summary.json`.

Larger mask examples from that sample:

```text
object_id=1   bbox=[205, 422, 876, 718]    mask_nonzero=106115
object_id=2   bbox=[1021, 343, 1280, 720]  mask_nonzero=45030
object_id=3   bbox=[21, 106, 385, 334]     mask_nonzero=41534
object_id=39  bbox=[943, 0, 1252, 305]     mask_nonzero=42020
```

Integration guidance:

- For adding YOLO/segmentation data to Shigurei history, subscribe to
  `/tracking/active_objects` first.
- Store parsed sidecar data instead of embedding the whole JSON directly into
  every frame meta file. A practical layout is `<stamp>_active_objects.json`
  plus optional `<stamp>_object_<id>_mask.png` files; the RGB-D frame meta can
  keep only relative paths, object count and source topic.
- `mask_b64` is large, so use a script to parse/debug it. `ros2 topic echo`
  prints a huge single string.

Debug overlay utility:

```bash
cd /workspace_whs
python3 code/scripts/render_shigure_active_objects_overlay.py --count 1
```

This subscribes to `/rs/color/compressed`,
`/rs/aligned_depth_to_color/compressedDepth` and `/tracking/active_objects`,
then writes one timestamped `*_active_objects_overlay.png` plus a matching
JSON summary. By default it treats the largest bottom-center box as the table
ROI, estimates the table depth range from that object's mask, and redraws only
objects inside the table box whose median depth is closer than the far side of
that table-depth band. The table ROI is used only for selection; selected
objects keep their original full mask and bbox when drawn. Drawn masks are
blended onto the RGB frame, each bbox is
labeled with its `object_id`, and overlapping boxes are greedily assigned
visually distinct colors.

### Other `/tracking/*` Topics

These topics are present and use `std_msgs/msg/String`:

```text
/tracking/object_event
/tracking/bring_event_confirmed
/tracking/feature_vector
```

No sample was received from these topics during a short `echo --once` check, so
their JSON schema is not fixed here yet.

## Shigure Core Typed Messages

These messages come from `code/reconstruction/shigure_core/shigure_core_msgs/msg/`.

### `PoseKeyPointsList`

Topic: `/shigure/people_detection`

```text
std_msgs/Header header
PoseKeyPoints[] pose_key_points_list
```

`PoseKeyPoints`:

```text
string people_id
BoundingBox bounding_box
PointData[] point_data
```

`PointData`:

```text
string body_part_name
Point pixel_point
Point projection_point
float32 score
```

`Point`:

```text
float32 x
float32 y
float32 z
```

### `DetectedObjectList`

Topic: `/shigure/object_detection`

```text
std_msgs/Header header
DetectedObject[] object_list
```

`DetectedObject`:

```text
string action
BoundingBox bounding_box
sensor_msgs/CompressedImage mask
```

In the current source path, `node_yolox_object_detection.py` generates
bbox-local PNG masks from segmentation results.

### `TrackedObjectList`

Topic: `/shigure/object_tracking`

```text
std_msgs/Header header
TrackedObject[] tracked_object_list
```

`TrackedObject`:

```text
string object_id
string action
BoundingBox bounding_box
Cube collider
```

`Cube`:

```text
float32 x
float32 y
float32 z
float32 width
float32 height
float32 depth
```

`node_object_tracking.py` computes `collider` from aligned depth, CameraInfo
intrinsics and object mask/bbox. Internally the collider is min-corner plus
dimensions.

### `ContactedList`

Topics: `/shigure/contacted`, `/shigure/RaycastHit`

```text
std_msgs/Header header
Contacted[] contacted_list
```

`Contacted`:

```text
string event_id
string people_id
string object_id
string action
BoundingBox people_bounding_box
BoundingBox object_bounding_box
Cube object_cube
```

`node_contact_detection.py` publishes contact events from tracked objects and
people. Before publishing to HoloLens, it converts object cubes from min-corner
basis to center basis. `node_raycast_hit_detection.py` publishes `RAYCAST_HIT`
events to `/shigure/RaycastHit`.

## Legacy YOLO Box And Segment Messages

### `bboxes_ex_msgs/msg/BoundingBoxes`

Topic: `/bounding_boxes`

```text
std_msgs/Header header
std_msgs/Header image_header
BoundingBox[] bounding_boxes
```

`BoundingBox`:

```text
float32 probability
uint16 xmin
uint16 ymin
uint16 xmax
uint16 ymax
uint16 id
uint16 img_width
uint16 img_height
int32 center_dist
string class_id
```

Runtime note: current graph has subscribers but no publisher.

### `Segments`

Source files:

```text
code/reconstruction/shigure_core/shigure_core_msgs/msg/Segments.msg
code/reconstruction/shigure_core/shigure_core_msgs/msg/Segment.msg
```

Source schema:

```text
std_msgs/Header header
std_msgs/Header image_header
Segment[] segments
```

`Segment`:

```text
string class_id
float64 probability
int32 xmin
int32 ymin
int32 xmax
int32 ymax
int32[] x_masks
int32[] y_masks
```

Current runtime issue:

- ROS graph reports `/Segments [bboxes_ex_msgs/msg/Segments]`.
- `ros2 interface show bboxes_ex_msgs/msg/Segments` fails because the installed
  IDL is missing.
- `ros2 interface show shigure_core_msgs/msg/Segments` also fails in the current
  receive workspace install.
- `node_yolox_object_detection.py` imports `Segments` from `bboxes_ex_msgs.msg`,
  while the visible source definition lives under `shigure_core_msgs/msg`.

Until this is fixed and rebuilt, do not base server-side cache ingestion on
`/Segments`.

## Topic Flow

Intended old Shigure pipeline:

```text
/rs/* + /openpose/pose_key_points
  -> people_tracking_node
  -> /shigure/people_detection

/bounding_boxes + /Segments + /shigure/people_detection + /rs/color/compressed + CameraInfo
  -> yolox_object_detection_node
  -> /shigure/object_detection

/shigure/object_detection + /rs/aligned_depth_to_color/compressedDepth + CameraInfo
  -> object_tracking_node
  -> /shigure/object_tracking

/shigure/object_tracking + /shigure/people_detection + /rs/color/compressed + CameraInfo
  -> contact_detection_node
  -> /shigure/contacted
```

Current practical ingestion path for this project:

```text
/rs/color/compressed
/rs/aligned_depth_to_color/compressedDepth
/rs/aligned_depth_to_color/cameraInfo
/tracking/active_objects
```

Use the RGB-D topics for frame/cache timing, and use `/tracking/active_objects`
for segmentation masks until the old `/Segments -> /shigure/object_detection`
path is repaired.

