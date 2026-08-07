# Verified-depth Assignment v4 result bundle

This is the versioned review bundle for the full hard-gated
`0.0002 m/count` Assignment v4 run documented in
`../../verified_depth_rerun.md`.

Included:

- resolved configuration, execution manifest, frozen-input audit, and
  scale-consistency gate report;
- complete region evidence in JSON and CSV form;
- assignment summary, v1-v4 comparison, acceptance report, and core hashes;
- SAM2 seed, runtime, propagation, and agreement reports;
- static, drawer-canonical, unknown, and near-contact diagnostic PLYs;
- all six SAM2 seed masks;
- representative interaction views, accepted drawer projections, contact
  sheets, and near-contact visualizations.

Intentionally excluded because they are large reproducible intermediates:

- `per_frame_assignment/` (approximately 122 MB);
- `sam2_propagation_per_seed/` (approximately 100 MB);
- `sam2_video_frames/` (approximately 5 MB).

The full local output remains at:

`/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v4/region_assignment_v4_verified_depth_0002_001`

This result has `ready_for_dual_tsdf=false`; no TSDF, NKSR, Mesh, GLB, or
URDF output is part of this bundle.
