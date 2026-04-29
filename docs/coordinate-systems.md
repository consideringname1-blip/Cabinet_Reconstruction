# Coordinate Systems

This project uses one internal 3D basis for reconstruction math and converts to
runtime formats only at boundaries.

## Internal Canonical Basis

Reconstruction, depth point clouds, ICP alignment, and camera-local model poses
use a right-handed camera basis:

- `+X`: camera right
- `+Y`: camera up
- `-Z`: camera forward

Objects in front of the camera therefore have negative `Z` values.

## Input And Output Boundaries

- OpenCV camera input: `+X` right, `+Y` down, `+Z` forward.
  Convert to canonical with `diag(1, -1, -1)`.
- Marker local input: right-handed, origin at marker center, `+X` marker right,
  `+Y` marker up, `+Z` marker front normal.
- Blender preview world: `+X` right, `+Y` forward, `+Z` up.
  Convert canonical `[x, y, z]` to Blender `[x, -z, y]`.
- Unity runtime output: Unity-facing `+X` right, `+Y` up, `+Z` forward.
  Convert canonical `[x, y, z]` to Unity `[x, y, -z]`.

## Pose Conversion Rules

Use column-vector pose math for transforms:

```text
p_parent = R_parent_child * p_child + t_parent_child
```

If only the parent/camera basis changes, convert with:

```text
R' = B_parent * R
t' = B_parent * t
```

If both parent and child bases change, convert with:

```text
R' = B_parent * R * B_child^-1
t' = B_parent * t
```

For canonical-to-Unity local model poses in this repository, both the camera
parent basis and the internal model basis change by the same `diag(1, 1, -1)`
mapping before the runtime FBX compensation is applied.
