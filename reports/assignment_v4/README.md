# Assignment v4: Region-level Articulation Consistency

> **Critical follow-up (2026-08-07):** the formal run used `depth_scale_to_m=0.001`, but stored pinhole depth is approximately 5x the corresponding Long Throw PLY projection. The formal ownership result is retained as a failed diagnostic and must not be used for fusion. See `followup_depth_scale_diagnostic.md`.

> **Physical-unit audit:** the matched Microsoft StreamRecorderConverter explicitly uses 5000 counts per metre of virtual Long Throw pinhole optical-axis Z. Exact execution revision remains partial. See `registered_depth_provenance_audit/README.md`.

Status: extended research diagnostic; **not accepted for dual TSDF**.

## Scope and frozen inputs

Assignment v4 replaces point-track ownership with direct projective RGB-D evidence on frame-local AutoSeg surface regions. Point tracking and LoFTR are not the primary ownership mechanism and LoFTR, LK, and TAPIP3D were not called by the formal run.

The run froze the verified prismatic joint, HoloLens `T_world_camera`, calibration, moving map, hand masks, frame policy, axis, and monotonic `q_t`. The exact parent is `fd07a99c9efd821f4f1dfd477090d993e56b6b55`; the branch is `agent/region-assignment-v4`.

- closed: frames 5–165, forward, 161 frames
- interaction: frames 177–213, forward, 37 frames
- open: frames 220–360, forward, 141 frames
- axis: `[0.5380529878004112, 0.08855245285159109, 0.838246649508972]`
- travel: `0.3002295680213606 m`
- interaction pose-copy maximum absolute difference: `0.0`
- SAM2 checkpoint SHA-256: `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`

No camera, axis, `q_t`, calibration, hand-mask, or moving-map optimization ran.

## Formal result

Primary output:

`/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v4/region_assignment_v4_attempt_002_valid_discontinuities`

The earlier failed attempt is preserved separately as `region_assignment_v4` and was not overwritten. Its incorrect discontinuity construction invalidated 534/690 proposals and accepted no static or drawer region.

The corrected formal result evaluated every interaction frame at stride 1:

| Region state | Count |
|---|---:|
| static | 1 |
| drawer | 5 |
| unknown | 154 |
| invalid | 530 |
| total | 690 |

All five drawer regions are geometry-supported, zero-overlap revealed candidates at frames 202–206 (`frame_*_layer_2`); none was forced by the repaired moving seed. They have drawer median support from 0.854 to 1.0 and static median support 0.0. Their physical interpretation as drawer side/bottom remains unverified by a human.

Five mixed-surface proposals were correctly retained as `unknown_mixed_surface`. The automatic near-contact set contains 133 difficult regions and is 100% unknown. The only accepted static-to-drawer canonical centroid separations are about 0.238–0.249 m, so this run did not produce an accepted close-contact pair at the drawer-side/cabinet-wall boundary.

## SAM2 propagation

SAM2 ran only after the five direct-geometry drawer regions passed the seed gate. Each seed was propagated independently; no LoFTR/point-track prompt was used. Multi-seed agreement below 0.60 remained unknown and propagation could not override static geometry.

- interaction: active in 20/37 frames, 5,766 accepted pixels, 234 propagation-conflict pixels, active-frame mean agreement 0.892; 228 direct-static conflict pixels remained unknown
- open: active in 4/141 frames, 1,171 accepted drawer pixels, 49 conflict pixels, active-frame mean agreement 0.902
- open effective support: drawer 3.749%, positive static 0%, open-only unknown 96.251%

The local SAM2 predictor warned that the optional compiled `_C` fill-hole extension was unavailable, so fill-hole post-processing was skipped. Propagation itself completed.

## Threshold diagnostic

The diagnostic is one-factor-at-a-time; free-space margin stays 3 cm and no combinatorial grid was run.

| Case | static | drawer | unknown | invalid |
|---|---:|---:|---:|---:|
| primary: support 3 cm, occlusion 3 cm | 1 | 5 | 154 | 530 |
| support 2 cm | 1 | 3 | 156 | 530 |
| support 4 cm | 2 | 6 | 152 | 530 |
| occlusion 2 cm | 1 | 5 | 154 | 530 |

The accepted set is somewhat support-threshold-sensitive and insensitive to the tested occlusion-margin change. The 3 cm primary threshold remains the conservative registered-depth setting used by the formal result; the sweep is diagnostic, not a parameter-selection claim.

## Comparison and reproducibility

- v1: 57,155 static and 63,465 drawer points; static was a remainder class and contaminated.
- v2: 167,055 static and 17,270 drawer points; dual-close evidence introduced static bias.
- v3: no formal assignment; correspondence/track sparsity remained the failure.
- v4 diagnostic clouds: 26 static, 2,065 drawer, and 294,427 unknown points.

At a 2 cm nearest-assignment comparison radius, all old v1/v2 points are unmatched because v4 positive ownership is extremely sparse. This is not evidence that those old points are geometrically absent.

A complete repeat was written to `region_assignment_v4_attempt_002_repeat`. SHA-256 comparison covered 2,287 evidence, four-state mask, SAM2 mask, and diagnostic-cloud files: 2,287/2,287 were identical.

## Validation

- Assignment v4: 13/13 unit tests passed.
- Assignment v3 regression: 18/18 tests passed.
- Python compilation passed.
- `git diff --check` passed before this report update.
- Static dependency audit found no formal LoFTR/LK/TAPIP3D or reconstruction call.

## Acceptance and blockers

`ready_for_dual_tsdf=false`.

Blocking conditions:

1. No human inspection record; the execution environment could not open generated images because its filesystem viewer failed at bubblewrap namespace creation.
2. Drawer front was not accepted; all accepted drawer seeds have repaired-moving overlap zero.
3. The five revealed candidates have not been visually confirmed as drawer side/bottom rather than another moving-compatible surface.
4. Static support is nearly absent: only one interaction static region, no closed/open positive static pixels.
5. Open identity covers only four frames and 3.749% of valid open pixels.
6. No accepted near-contact drawer/static pair exists, so cabinet-inner-wall leakage at the critical interface is not demonstrated to be solved.
7. The optional SAM2 fill-hole extension is unavailable.

No TSDF, NKSR, Poisson, Mesh, GLB, URDF, pose refinement, axis refinement, `q_t` refinement, or moving-map refinement ran.

## Versioned result bundle

The GitHub report bundle under `reports/assignment_v4/results/` contains the
resolved configuration, input/method/checkpoint audits, full region evidence,
threshold and deterministic-repeat reports, v1-v4 comparison, SAM2 seed and
agreement reports, diagnostic point clouds, accepted projections, contact
sheets, representative-frame overviews, and near-contact visualizations.

The following large reproducible intermediates remain local and are intentionally
not versioned: per-frame four-state `.npy` masks, SAM2 video-frame cache,
per-seed propagation arrays, and the complete repeated-run directory. Their core
outputs are covered by `deterministic_rerun_report.json`, which records 2,287 of
2,287 compared files as byte-identical. The local output paths remain documented
above; the versioned result bundle is approximately 15 MB.
