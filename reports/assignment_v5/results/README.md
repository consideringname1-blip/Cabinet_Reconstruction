# Assignment v5 curated results

This directory contains the review-sized result bundle committed to GitHub.
Full per-region outputs remain on the evaluation server.

## Primitive diagnostic

- `primitives/primitive_diagnostic.json`: synthetic primitive test and
  motion-transition output.

## 116 unknown-region diagnostic

- `unknown_116_physical_events/failed_001/`: preserved failed first adapter
  result (24 moving, 92 unknown). `moving_regions.jpg` exposes the tangent-floor
  false-positive failure.
- `unknown_116_physical_events/corrected_002/`: corrected hard-gated result
  (116 unknown), per-region summary, frozen motion states/transitions,
  scale-gate report, and the three review-category contact sheets.

Not included in Git are the complete 232 per-region source/timeline directories.
They remain at:

- `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v5/unknown_116_physical_events_001`
- `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v5/unknown_116_physical_events_002`

Neither result is reconstruction-ready. `ready_for_dual_tsdf=false`.
