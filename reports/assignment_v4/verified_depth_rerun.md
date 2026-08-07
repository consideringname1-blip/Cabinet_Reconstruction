# Assignment v4 verified-depth full rerun

Date: 2026-08-07

Status: completed corrected Assignment v4 run; **not accepted for dual TSDF**.

## Scope

The user explicitly authorized integrating a registered-depth
scale-consistency hard gate and fully rerunning Assignment v4 with the verified
`0.0002 m/count` contract. The old `0.001` outputs were not modified.

No camera pose, calibration, joint axis, joint type, `q_t`, moving map, or hand
mask was optimized. No TSDF, NKSR, Mesh, GLB, or URDF stage ran.

## Command and output

```bash
/workspace_whz/envs/video_articulation/bin/python \
  tools/fuse_hololens_articulation_assignment_v4.py \
  --config tools/itaco_moving_map_fix/configs/hololens_region_assignment_v4_verified_depth.yaml
```

Output:

`/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v4/region_assignment_v4_verified_depth_0002_001`

The configuration retains the original forward frame phases:

- closed: 5–165, 161 frames
- interaction: 177–213, 37 frames
- open: 220–360, 141 frames

## Hard-gate result

The gate ran before Assignment v4 loaded any frame for region evidence. It
compared nine registered uint16 PNGs with the corresponding Long Throw world
PLYs projected using the converter's rounded-pixel, last-write virtual-pinhole
optical-Z semantics.

| Metric | Result | Required |
|---|---:|---:|
| configured scale | 0.0002 m/count | 0.0002 m/count |
| median fitted scale | 0.000200012639 m/count | within 1% |
| relative scale drift | 0.00438% | <= 0.5% |
| minimum valid-mask IoU | 99.9857% | >= 98% |
| maximum frame median absolute error | 0.1007 mm | <= 1 mm |
| maximum frame p90 absolute error | 0.1807 mm | <= 5 mm |

Gate status: `passed=true`, with no failure codes.

The gate is mandatory in the Assignment v4 entry point. Missing, disabled,
wrong-scale, drifted, low-IoU, or excessive-error evidence stops the run with
`registered depth physical-unit mismatch` and writes
`scale_consistency_gate_report.json`.

## Corrected region result

| Region state | Old 0.001 run | Verified 0.0002 run | Change |
|---|---:|---:|---:|
| static | 1 | 78 | +77 |
| drawer | 5 | 76 | +71 |
| unknown | 154 | 192 | +38 |
| invalid | 530 | 344 | -186 |
| valid total | 160 | 346 | +186 |
| proposals | 690 | 690 | 0 |

Accepted drawer regions comprise 37 with repaired-moving seed overlap and 39
revealed regions without initial seed overlap. Six direct-geometry drawer seeds
were passed to SAM2, and propagation completed.

The output diagnostic clouds contain:

- static: 204,565 points
- drawer canonical: 14,864 points
- unknown: 205,128 points

Open-phase positive support is 60.12% static, 3.70% propagated drawer, and
36.18% unknown among valid pixels.

## Remaining blockers

`ready_for_dual_tsdf=false`.

1. The generated drawer/static ownership overlays have not been accepted by the
   user.
2. The automatically selected near-contact set remains 100% unknown, so the
   drawer-side versus cabinet-inner-wall boundary is not resolved.
3. More accepted regions do not by themselves prove correct object ownership;
   cabinet, floor, hand, and drawer-interior leakage still require visual review.
4. SAM2 completed, but its optional compiled `_C` fill-hole extension was
   unavailable; only that optional post-processing step was skipped.

## Validation

- Assignment v4 tests: 18/18 passed, including five hard-gate tests.
- Python compilation passed.
- `git diff --check` passed before documentation updates.
- Frozen pose/axis/`q_t`/moving-map/checkpoint hashes passed.
- The run wrote its resolved config, command manifest, scale gate report,
  complete region evidence, per-frame four-state assignments, SAM2 manifests,
  diagnostic point clouds, visualizations, acceptance report, and core hashes.

## Versioned result bundle

The GitHub review bundle is under
`reports/assignment_v4/results/verified_depth_rerun/`. It contains the
complete region evidence, gate and execution audits, result point clouds, SAM2
seed masks/reports, and all review visualizations. Large reproducible
per-frame assignments, SAM2 video frames, and per-seed propagation arrays remain
only in the documented local output directory.
