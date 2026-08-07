# Cabinet Reconstruction Project Status

This file is the canonical handoff summary for `/workspace_whz`.

- Tracked workspace: `/workspace_whz`
- Excluded workspace: `/workspace` — do not inspect, modify, or include it in project status
- Last updated: 2026-08-07
- Verified project baseline before this status-only update: `7e394359d2b5a0412398fbd1743e5d8f47d43d4c`
- Cabinet publication target: `whz/main` at `consideringname1-blip/Cabinet_Reconstruction.git`

## 30-second handoff

The project reconstructs an articulated cabinet/drawer from real RGB-D video.
The best current result is a research-quality prismatic drawer model with fixed
HoloLens camera poses, an estimated 0.30023 m travel, separate static/drawer
geometry, and validated GLB/URDF packages.

The selected extended pipeline is:

```text
HoloLens RGB-D + recorded T_world_camera
-> RGB/depth/hand validity gating
-> AutoSeg-SAM2 proposals used only as soft regions
-> repaired moving map (no per-frame min-max)
-> fixed-camera moving-centroid prismatic axis
-> monotonic q_t
-> static/drawer dual-volume fusion
-> per-component NKSR reconstruction
-> articulated GLB/URDF packaging and validation
```

Important boundary: this best result is an extension around the official iTACO
baseline. It must not be described as an unmodified official iTACO result.
Official-compatible runs, failed runs, and corrected/extended runs remain
separate. The final reproduction manifest reports that the official core files
`joint_refinement.py` and `data.py` remained clean.

## Current best verified result

- Input recording: HoloLens `2026-07-30-002840`
- Joint type: prismatic
- Camera policy: recorded HoloLens `T_world_camera`; no joint camera optimization
- Opening axis in world coordinates:
  `[0.5380529878, 0.0885524529, 0.8382466495]`
- Joint convention: `q=0` is closed; positive `q` opens the drawer
- Monotonic travel: `0.3002295680 m`
- Fused points: 57,155 static; 63,465 canonical drawer
- Full NKSR meshes: 1,024,403 static triangles; 4,108,011 drawer triangles
- Preview meshes: 400,000 static triangles; 200,000 drawer triangles
- Package validation: `all_valid=true`
- Intended use: research demonstration and motion visualization
- Do not claim: production-grade geometry, a complete digital twin, or
  generalization beyond the tested cabinet sequence

Canonical local result directory:

`data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/geometry_interior_v1/articulated_model/`

Most useful artifacts:

- `reproduction_manifest.json`: inputs, frame policy, settings, commands,
  compatibility notes, metrics, and retained failed attempts
- `package_validation.json`: structural validation result
- `hololens_cabinet_with_interior_double_sided.glb`
- `hololens_cabinet_with_interior_double_sided_animated.glb`
- `hololens_cabinet_with_interior_double_sided.urdf`
- `render_closed_axis_corrected.png`
- `render_open_axis_corrected.png`

Final frame policy recorded by the weekly report:

- closed-state frames: 5–165
- interaction frames: 177–213
- open-state frames: 220–360
- frame order: forward in every phase

Do not infer a different frame range or order from a helper script name. Read the
reproduction manifest and exact run configuration before rerunning.

## Completed

### Pipeline reproduction and diagnosis

- Compared VGGT against HoloLens depth/poses. Even after scale alignment, the
  measured errors were too large to treat VGGT depth as sensor depth or ground
  truth on this recording.
- Reproduced the official-compatible MonST3R + AutoSeg-SAM2 path and retained its
  weak/failed results rather than replacing them with corrected output.
- Isolated camera-pose error with a controlled ARKit-like perturbation:
  planar thickness increased from 1.13 cm to 11.13 cm and normals with error
  above 30 degrees increased from 3.09% to 71.87%.
- Audited frame identity, validity masks, proposal identity, moving-map
  normalization, camera optimization, and initial-state assumptions.
- Completed the fixed-input Assignment v4 surface-identity/visibility follow-up.
  Drawer-front controls passed 4/4, static controls 6/8, and co-moving controls
  2/2; the 116 ambiguity regions yielded 1 experimental static preference, 16
  drawer preferences, and 99 unresolved. Formal labels remain unchanged.

### Tracks and moving map

- Implemented immutable frame manifests, automatic reference-frame selection,
  unified validity masks, explicit proposals, short 3D tracks, and
  static/moving/unknown classification in `tools/itaco_track_motion_v1/`.
- Implemented Stage 1.5 hand-mask recovery, gap reassociation, derived depth
  quality, manual evaluation hooks, coverage metrics, and observability checks.
- Stage 1.5 improved filtered tracks from 176 to 203 and static image coverage
  from 6.25% to 31.25%, but did not meet the evidence threshold for unconstrained
  joint optimization.
- Implemented the independent moving-map repair in
  `tools/itaco_moving_map_fix/`: invalid or unsupported areas cannot receive
  high moving scores, and per-frame min-max normalization is removed.
- The repaired moving map reduced high-score area from 58.52% to 3.19% and
  reduced high-score pixels outside valid support from 1,886,763 to zero.
- Selected the moving-centroid axis after 99.50% linear explanation,
  0.60-degree agreement with an earlier independent GT-depth centroid axis, and
  `q_t` correlation of 0.997.

### Geometry and packaging

- Recomputed monotonic `q_t` with no negative steps.
- Built articulation-aware static/drawer dual volumes.
- Recovered the drawer template using cross-frame canonical 3D consistency,
  not fixed UIDs, colors, 2D mask dilation, or hard-coded object position.
- Reconstructed static and drawer components separately with NKSR.
- Exported single- and double-sided GLB/URDF packages and an animated GLB.
- Verified the final package structure with `all_valid=true`.

### Repository and validation

- `4040b99d`: added RGB-D articulation integration and the first tracked tools.
- `9812ffcd`: tracked the remaining `tools/` and `scripts/` sources
  (110 files); Python cache files remain ignored.
- `7e394359`: tracked `AGENTS.md` and the weekly report.
- On 2026-08-06, the tracked tool packages passed:
  - Python syntax compilation for `tools/` and `scripts/`
  - moving-map tests: 11/11
  - track-motion tests: 10/10
  - Git whitespace checks except for one intentional Markdown hard break in the
    weekly report

Validation commands that passed in the existing `foundationpose` environment:

```bash
/opt/miniconda/envs/foundationpose/bin/python -m compileall -q tools scripts
/opt/miniconda/envs/foundationpose/bin/python tools/itaco_moving_map_fix/tests/run_tests.py
/opt/miniconda/envs/foundationpose/bin/python -m unittest tools.itaco_track_motion_v1.tests.test_phase1 tools.itaco_track_motion_v1.tests.test_stage1_5
```

## In progress

No official reproduction job or long-running reconstruction job is currently
known to be running.

The Assignment v4 near-contact ambiguity audit is complete. It decomposed all
131 unknown difficult regions into 116 regions simultaneously supported by the
static and drawer models, 14 with insufficient evidence, and one borderline
both-bad region. Per-region source overlays, q-dependent visibility, and target-
surface hit visualizations are available under the documented independent
output directory. The result remains a diagnostic and is not ready for dual
TSDF.

The surface-identity/visibility follow-up is also complete and is interpreted as
inconclusive. It is not an active reconstruction job and did not run propagation,
TSDF, NKSR, or Mesh.

The following work is incomplete and must not be presented as completed:

- Convert the final sequence of helper commands into a clean, deterministic
  one-command entry point with resolved paths instead of placeholders.
- Audit all 13 Stage 1.5 cross-gap track associations manually and improve the
  sparse moving/static labels.
- Pin and record exact source revisions for the local NKSR and VGGT source trees
  before treating them as reproducible dependencies.
- Improve point-cloud cleanup, normal consistency, occlusion completion, and
  mesh surface quality.
- Validate the approach on more recordings and on a revolute joint.

## Known issues

### Scientific and geometry limitations

- RGB and depth fields of view do not match. Depth outside the valid RGB footprint
  is not reliable semantic or motion evidence.
- HoloLens depth noise and occlusion boundaries still create rough surfaces,
  local holes, and incomplete drawer side/bottom geometry.
- Stage 1.5 has only three accepted moving tracks, 7.14% static bootstrap
  stability, and 5/21 manual classification accuracy. This is insufficient for
  releasing camera, axis, and labels into unconstrained joint optimization.
- Only one real prismatic drawer sequence has reached the current quality level.
- Verified-depth Assignment v4 still has 131 unresolved difficult regions. Of
  these, 116 are supported by both frozen motion hypotheses across cabinet,
  drawer, edge, and floor/background surfaces. This is a surface-identity and
  observability problem; threshold relaxation or region propagation is unsafe.
- Surface continuity separates the controls but leaves 85.34% of the 116
  both-supported regions unresolved. The 17 preferences are concentrated in
  repeated layer-6/11/28 observations, and no manually confirmed drawer-side /
  cabinet-inner-wall evaluation set exists.
- Local output artifacts are large. A path in this document does not imply that
  its artifact is tracked by Git.

### Reproduction and environment limitations

- The final `reproduction_manifest.json` records the real pipeline, but some
  commands still contain placeholders such as `<moving_map_labels.npz>`.
  Resolve them from the manifest/configuration before running.
- Tests require PyTorch, OpenCV, and PyYAML. The `foundationpose` environment
  passed all current tests; default Python and the `server` environment were
  each missing one or more dependencies.
- The weekly report contains absolute `/workspace_whz/data/output/...` links.
  They work only when the corresponding local artifacts exist.
- `data/config/conda/`, `data/config/docker/`, and
  `data/config/manifest-files.txt` are the tracked environment/configuration
  inventory; verify installed packages before reproduction.

### Git state at this update

- Working branch: `movable-parts`
- Substantive baseline before this status-only update:
  `7e394359d2b5a0412398fbd1743e5d8f47d43d4c`
- `whz/main` matched that baseline before this status commit.
- The local branch was `ahead 105, behind 2` relative to
  `origin/movable-parts`; do not merge or force-push `origin` implicitly.
- Untracked and intentionally excluded from this status commit:
  `code/reconstruction/NKSR/`, `code/reconstruction/vggt/`, and
  `use-proxy-auto.sh`.
- NKSR/VGGT directory names are not proof of which upstream model or revision
  ran. Record source URLs, revisions, local changes, environments, and licenses
  before tracking or publishing them.

A file cannot contain the hash of the commit that contains itself. After
checkout, run `git log -1 --oneline` to identify this status document's commit.

## Next steps

1. Record exact upstream URLs, revisions, local diffs, environments, and licenses
   for NKSR and VGGT; then decide between submodules, vendoring, or an external
   dependency manifest. Ask before substituting either official source.
2. Replace placeholder commands in the final reproduction manifest with a
   deterministic entry script and checked configuration while preserving the
   exact official step order.
3. Manually audit the 13 accepted gap associations and expand trustworthy
   moving/static/unknown annotations before Stage 2 joint optimization.
4. Rerun the selected extended pipeline from clean inputs and save command logs,
   environment capture, output hashes, and package validation beside the result.
5. Improve valid-depth registration, point cleanup, normal consistency, and
   canonical occlusion completion. Keep corrected output separate from the
   existing result.
6. Capture a cleaner closed-interaction-open RGB-D sequence and test at least one
   additional prismatic object before claiming generality.
7. Validate revolute support on a real sequence under the same evidence and
   reproduction standards.
8. Add a small manual drawer-side/cabinet-inner-wall evaluation set before using
   surface-identity preferences in any ownership update; keep dual TSDF blocked.

## Key entry points

- Workspace rules: `AGENTS.md`
- Architectural decisions: `DECISIONS.md`
- Detailed weekly evidence: `reports/itaco_weekly_summary_2026-07-30.md`
- Moving-map implementation/tests: `tools/itaco_moving_map_fix/`
- Track-motion implementation/tests: `tools/itaco_track_motion_v1/`
- HoloLens/iPhone fusion and reconstruction helpers: `tools/`
- Initial-value visualization: `scripts/visualize_itaco_initial_values.py`
- RGB-D articulation server integration:
  `code/scripts/reconstruct_articulation_rgbd.py`, `code/task_worker.py`,
  `code/server_api.py`, and `code/config.py`
- Environment reconstruction notes: `data/config/README_REBUILD.md`
- Final artifact manifest:
  `data/output/itaco_moving_map_fix/2026-07-30-002840_fixed_hololens_fixed_map_axis_trial_001/geometry_interior_v1/articulated_model/reproduction_manifest.json`

## Handoff rules for the next GPT

1. Read `AGENTS.md`, this file, and `DECISIONS.md` before changing the
   reconstruction pipeline.
2. Treat the weekly report and reproduction manifest as evidence, not permission
   to skip validation.
3. Never relabel an extended/corrected result as official-compatible.
4. Preserve failed official outputs and report their quality honestly.
5. Do not infer model provenance from directory names.
6. After material code changes, update Completed, In progress, Known issues, and
   Next steps.
7. Record exact commands/configs, frame ranges/order, model revisions,
   validations, and output paths for every new reproduction.


## Assignment v3 isolated-worktree update (2026-08-07)

### Completed and verified

- Created isolated branch `agent/funrec-inspired-assignment-v3` from exact commit
  `d5ef666310022604fcefbeac0df2034225e05c66`; the source worktree was not modified.
- Implemented `extended_non_official_funrec_inspired_assignment_v3`: periodic,
  spatially balanced LoFTR tracks; fixed HoloLens poses/axis/q; explicit AutoSeg
  proposals; robust track evidence; conservative region voting; SAM2 propagation
  and four-state assignment code paths; and structured safe-stop reporting.
- Audited Kornia LoFTR 0.8.2 with cached indoor checkpoint SHA-256
  `d73c54720370ba0690cd477e38404a626e881292cc0be09a6801da0cc6f53198`,
  AutoSeg-SAM2 revision `840a356f26bd31ac5f3033c05d3ba61a2ceaa38c`,
  SAM2 revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`, and SAM2 checkpoint
  SHA-256 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`.
- All 12 core tests pass. Two independent formal Phase B-D attempts produced
  identical hashes for raw/filtered tracks, labels, and region votes.
- Formal result: 882 tracks; 8 moving, 6 static, 868 unknown. Of the unknown
  tracks, 357 lack support, 152 lack excitation, and 359 fail the configured
  5 cm absolute residual gate.
- The formal run safely stopped in Phase D because no keyframe region had enough
  high-confidence moving support to seed SAM2.
- No formal SAM2 propagation, four-state assignment, TSDF, NKSR, mesh, GLB,
  URDF, or parameter refinement ran.

### In progress / not accepted

- `ready_for_dual_tsdf=false`. No v3 ownership point clouds are accepted.
- Drawer-side retention and cabinet-inner-wall leakage remain unobservable.

### Known issues

- LoFTR provides long-span correspondences, but most geometrically supported
  tracks have absolute world/canonical residual above 5 cm. Relative model
  preference alone produced unsafe drawer regions and is retained only as a
  diagnostic failure.
- The configured 2 px depth-boundary exclusion also makes open registered depth
  sparse, but formal execution stops before the open assignment stage.
- SAM2's optional compiled `_C` fill-hole extension is absent. This was observed
  only in preserved diagnostic propagation runs, not the formal stopped run.
- No independent manual ownership labels exist for this v3 attempt.

### Next steps

1. Diagnose RGB/depth correspondence and point-track geometric residuals before
   changing the 5 cm acceptance threshold.
2. Evaluate a repository-available tracker with depth-aware association or add
   explicit reprojection/occlusion checks without downloading a substitute.
3. Do not run SAM2 propagation or dual TSDF until interaction regions receive
   high-confidence moving support.

### Git and outputs

- Worktree: `/workspace_whz_worktrees/funrec-assignment-v3`
- Branch: `agent/funrec-inspired-assignment-v3`
- HEAD: `d5ef666310022604fcefbeac0df2034225e05c66`
- Formal stopped output: `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v3/funrec_assignment_v3`
- Formal repeat: `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v3/funrec_assignment_v3_repeat`
- Failed/diagnostic outputs are preserved with `lk_diagnostic`,
  `loftr_failed_association_v1`, and `loftr_relative_only_diagnostic` suffixes.
- Assignment v3 implementation/results commit: `07856bc0`; published to `whz/agent/funrec-inspired-assignment-v3`. No merge, stash, reset, or clean was performed.

## Depth-projective track diagnostic update (2026-08-07)

### Completed and verified

- Added a fixed-input, depth-aware projective LoFTR association mode. Every
  candidate observation is evaluated simultaneously under world-static and
  fixed-axis/fixed-`q_t` drawer hypotheses using reprojection error, registered
  depth residual, and occlusion checks. HoloLens poses, axis, and `q_t` remain
  frozen; no parameter optimization runs.
- Added per-observation residual decomposition for high-absolute-residual tracks:
  projective pixel error, signed/absolute registered-depth residual, axis error,
  perpendicular error, and total 3-D error under both hypotheses.
- Added a hard diagnostic stop before region voting and propagation. The run did
  not execute region voting, SAM2 propagation, assignment, TSDF, NKSR, Mesh, or
  any camera/axis/`q_t` optimization.
- All 18 Assignment v3 tests pass, including synthetic static/drawer association,
  invalid depth, double-occlusion rejection, axis/perpendicular decomposition,
  and the propagation hard gate.
- Real-sequence result: 882 tracks, 21 static, 12 moving, and 849 unknown. The
  previous formal-v3 moving count was 8; the configured clear-increase threshold
  is 13, so region propagation remains prohibited.
- The 12,360 candidate attempts yielded 1,531 accepted and 10,829 rejected. The
  main rejection counts are 4,125 forward-backward inconsistencies, 3,803 invalid
  target depth/mask observations, 1,266 excessive depth residuals, 752 excessive
  projective costs, 451 low-confidence matches, 402 double-occlusions, and 30
  reverse-association failures.
- Among accepted attempts, the selected-model metadata contains 371 static, 242
  drawer, and 918 ambiguous associations. This is association-level metadata;
  final track labels are determined from complete-track static/drawer evidence.

### In progress / not accepted

- `ready_for_region_propagation=false` and `ready_for_dual_tsdf=false`.
- The depth-projective result is a diagnostic, not an ownership assignment or a
  reconstruction result. Its output point clouds visualize classified track
  observations only.

### Known issues

- The reliable moving count improved from 8 to 12 but missed the configured
  threshold of 13. Most tracks remain unknown: 672 have insufficient support,
  149 lack motion excitation, 27 exceed the absolute residual gate, and one is
  ambiguous.
- High-residual tracks contain material perpendicular error as well as axial
  error (drawer-hypothesis p90 approximately 5.26 cm perpendicular and 6.98 cm
  axial). The remaining failures therefore cannot be attributed only to the
  fixed axis or `q_t`; correspondence/depth/pose inconsistency remains plausible.
- A candidate accepted under one hypothesis can legitimately have a large
  residual under the alternative hypothesis. Alternative-model maxima in the
  diagnostic report are not acceptance-threshold violations.

### Next steps

1. Manually inspect the 12 moving tracks and 27 high-residual tracks before
   changing thresholds.
2. Improve correspondence continuity and registered-depth validity while
   keeping poses, axis, and `q_t` frozen, then rerun the hard-stopped diagnostic.
3. Do not run region propagation until moving support clearly exceeds the gate
   and has credible spatial and temporal coverage.

### Git state and output

- Branch: `agent/funrec-inspired-assignment-v3`.
- Depth-projective implementation/results commit: `0422ba54`; published to
  `whz/agent/funrec-inspired-assignment-v3`.
- Output mirrored under `reports/assignment_v3_depth_projective_tracks/`.
- Draft PR creation was externally blocked at that time by the authenticated CLI user and GitHub App authorization.

## Assignment v4 region-level diagnostic update (2026-08-07)

### Completed and verified

- Added `tools/itaco_region_assignment_v4/` and the
  `tools/fuse_hololens_articulation_assignment_v4.py` entry point. Formal
  ownership uses direct static-versus-prismatic projective RGB-D evidence on
  frame-local AutoSeg regions; LoFTR, LK, and TAPIP3D are not formal classifiers.
- Frozen-input validation preserves the verified HoloLens poses, prismatic axis,
  monotonic `q_t`, calibration, frame policy, repaired moving labels, and hand
  masks. It also verifies the configured SAM2 checkpoint SHA-256 before use.
- The formal corrected run evaluates all 37 interaction frames and outputs four
  exclusive states. It classified 690 proposals as 1 static, 5 drawer, 154
  unknown, and 530 invalid. All five drawer regions are zero-moving-seed-overlap
  revealed candidates; five mixed proposals remain unknown.
- Audited SAM2 ran only after direct geometry accepted drawer seeds. It was active
  in 20/37 interaction frames and 4/141 open frames; open drawer coverage is
  3.749%, while 96.251% of effective open support remains unknown.
- A one-factor threshold diagnostic produced static/drawer counts of 1/3 at 2 cm
  support, 1/5 at the 3 cm primary setting, and 2/6 at 4 cm support. Changing only
  occlusion margin from 3 cm to 2 cm kept the primary 1/5 counts.
- A full deterministic repeat matched 2,287/2,287 evidence, per-frame assignment,
  SAM2-mask, and diagnostic-cloud artifact hashes.
- Assignment v4 tests pass 13/13; Assignment v3 regression tests pass 18/18;
  syntax compilation and whitespace checks pass.

### In progress / not accepted

- `ready_for_dual_tsdf=false`. Assignment v4 is an extended ownership diagnostic,
  not a reconstruction and not a new best verified result.
- Generated contact sheets and projection panels still require explicit human
  inspection. The current execution environment's image viewer failed because
  unprivileged bubblewrap namespaces are unavailable.

### Known issues

- No drawer-front region was accepted; every accepted drawer seed has repaired
  moving overlap zero and still needs visual confirmation as drawer side/bottom.
- Positive static support is nearly absent: one interaction static region and no
  closed/open positive static pixels. Consequently all old v1/v2 comparison
  points are unmatched at the 2 cm diagnostic radius.
- The 133 automatically selected near-contact difficult regions are all unknown,
  but no accepted drawer/static pair is spatially close. Inner-wall leakage at
  the critical interface is therefore not yet demonstrated to be solved.
- SAM2 propagation reaches only four open frames. The optional SAM2 `_C` fill-hole
  extension is unavailable, so that post-processing step was skipped.

### Next steps

1. Manually inspect drawer/static/mixed contact sheets and the accepted projection
   panels, explicitly checking drawer front, side/bottom, cabinet inner wall, and
   floor leakage.
2. Recover reliable positive static evidence without defining static as a
   remainder and without relaxing thresholds merely to increase counts.
3. Improve open identity continuity from geometry-validated interaction seeds;
   keep open-only and propagation-conflict pixels unknown.
4. Rerun the same fixed-input diagnostic after those evidence changes. Do not run
   dual TSDF until all acceptance conditions, including human review, pass.

### Git state and outputs

- Worktree: `/workspace_whz_worktrees/funrec-assignment-v3`
- Branch: `agent/region-assignment-v4`
- Exact parent before Assignment v4 changes: `fd07a99c9efd821f4f1dfd477090d993e56b6b55`
- Formal output: `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v4/region_assignment_v4_attempt_002_valid_discontinuities`
- Deterministic repeat: `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v4/region_assignment_v4_attempt_002_repeat`
- Preserved failed first attempt: `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v4/region_assignment_v4`
- Tracked report: `reports/assignment_v4/README.md`
- Assignment v4 code, curated results, and reports are published from this branch; no merge, stash, reset, or clean was performed.

## Assignment v4 registered-depth scale follow-up (2026-08-07)

### Completed and verified

- Added read-only diagnostics for the five accepted layer2 regions, layer25
  drawer-front projective residuals, and same-frame pinhole-depth versus Long
  Throw PLY projection.
- The five layer2 masks at frames 202–206 cover the same small brown vertical
  interior surface, not the large white drawer front. A diagnostic corrected
  depth scale continues to classify them as revealed drawer candidates, but
  drawer-rear-panel versus cabinet-inner-wall identity still requires human review.
- Identified a fivefold depth-scale mismatch: stored pinhole registered depth is
  4.9995x the corresponding Long Throw→virtual-pinhole optical-axis projection. The formal
  v4 config used 0.001 m/count; the diagnostic matching scale is 0.0002 m/count.
- After `/5` correction, pinhole/PLY valid-mask IoU is 99.984–99.996%, median
  absolute depth difference is about 0.11 mm, and 99.62% of common pixels agree
  within 1 cm.
- Layer25 changes from near-total free-space contradiction to 89–100% drawer
  support on the inspected frame pairs; residual medians return to millimetres.

### In progress / not accepted

- The original Assignment v4 formal output is retained as a failed diagnostic,
  not a valid ownership result. It must not enter dual TSDF.
- The `/5` result is a focused diagnostic only; a complete corrected v4 run has
  not been started.

### Known issues

- The provenance of the stored 0.2 mm/count encoding is not documented beside
  `pinhole_projection/depth`; audit the producer before making the scale a new
  frozen input.
- Corrected support does not by itself prove that the accepted brown interior
  surface belongs to the drawer rather than the cabinet; explicit visual and
  multi-view ownership review remains required.

### Next steps

1. Use `reports/assignment_v4/registered_depth_provenance_audit/` as the unit contract evidence.
2. Integrate the specified PLY/pinhole scale-consistency gate only before an approved rerun.
3. If the user authorizes it, rerun Assignment v4 into a new output directory,
   preserving the failed 0.001-scale output.
4. Do not run dual TSDF until corrected ownership and human review pass.

### Diagnostic outputs

- `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v4/region_assignment_v4_followup_masks_depth_001`
- `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v4/region_assignment_v4_followup_depth_scale_001`
- Report: `reports/assignment_v4/followup_depth_scale_diagnostic.md`
- GitHub-curated masks/projections: `reports/assignment_v4/results/followup_masks_depth/`
- GitHub-curated scale audit: `reports/assignment_v4/results/followup_depth_scale/`

## Registered-depth physical-unit and provenance audit (2026-08-07)

### Completed and verified

- Identified the `pinhole_projection` producer family as Microsoft
  HoloLens2ForCV StreamRecorderConverter. Its exact output signature matches the
  recording and its source explicitly encodes virtual-pinhole optical-axis Z as
  `uint16(Z_m * 5000)`.
- Audited all 382 registered PNGs: every image is 288x320 uint16, zero is the
  invalid value, no 65535 saturation occurs, and the nonzero raw range is
  996–30,403 counts.
- Evaluated 11 uniformly distributed frames: closed 5/85/165, interaction
  177/186/195/204/213, and open 220/290/360. Robust through-origin scale is
  0.000200013845 m/count; per-frame peak-to-peak variation is 0.00441%.
- Producer-algorithm re-encoding agrees exactly on 98.86–99.85% of common
  pixels. Source PGM radial range and PLY range agree at sub-micrometre median
  error. Optical-Z residual median is 0.052 mm, versus 60.1 mm for radial range
  and 85.0 mm for true PV forward depth.
- Added the independent audit tool, nine synthetic/unit tests, consumer audit,
  historical impact audit, unit contract and a non-integrated gate specification.

### In progress / not accepted

- The exact HoloLens2ForCV checkout and literal conversion command used on the
  recording were not preserved. Producer family/formula are verified, but exact
  execution-revision provenance remains partial.
- The provenance audit itself did not authorize a rerun. The user subsequently
  authorized and completed the separate corrected v4 run documented below.

### Known issues

- Assignment v3, v3 depth-projective and formal v4 directly decoded the
  registered PNG with 0.001 m/count and require separate reruns.
- The earlier phrase “PV depth” was imprecise: RGB is PV color resampled into a
  virtual Long Throw pinhole view; the stored depth is not true PV-camera Z.

### Next steps

1. Review the corrected v4 ownership visualizations described below.
2. Keep all failed 0.001-scale outputs intact and separate.
3. Do not run dual TSDF until corrected near-contact ownership passes review.

### Git state and outputs

- Branch: `agent/region-assignment-v4`
- Audit-start HEAD: `052094187789b1d78f1032e2572d1af015bcc434`
- Output: `reports/assignment_v4/registered_depth_provenance_audit/`
- No commit, push, merge, Assignment v4, SAM2, TSDF, NKSR or Mesh run occurred.

## Assignment v4 verified-depth hard-gated rerun (2026-08-07)

### Completed and verified

- Integrated a mandatory pre-assignment scale-consistency gate using matching
  registered PNG and Long Throw world PLY evidence. The gate checks configured
  and fitted scale, per-frame drift, valid-mask IoU, and median/p90 depth error.
- Added five hard-gate tests. Assignment v4 now passes 18/18 unit tests; the
  explicit `0.001` case fails with `configured_scale_mismatch`.
- Created a separate configuration with verified `0.0002 m/count` and a new,
  non-overwriting output directory.
- The real nine-frame gate passed: median fitted scale
  0.000200012639 m/count, minimum IoU 99.9857%, maximum frame median/p90 error
  0.101/0.181 mm, and 0.00438% relative scale drift.
- Completed all Assignment v4 region evidence and conditional SAM2 propagation.
  Results are 78 static, 76 drawer, 192 unknown, and 344 invalid regions from
  690 proposals. The old invalid-scale result was 1/5/154/530.
- SAM2 ran from six geometry-accepted drawer seeds. No LoFTR, LK, TAPIP3D,
  camera/axis/`q_t`/moving-map optimization, TSDF, NKSR, Mesh, GLB, or URDF ran.

### In progress / not accepted

- `ready_for_dual_tsdf=false`: user visual ownership review is still absent.
- The automatically defined near-contact set remains 100% unknown, so the
  drawer-side versus cabinet-inner-wall interface is not resolved.

### Known issues

- Correct scale restores evidence coverage but cannot by itself rule out cabinet,
  floor, hand, or drawer-interior ownership leakage.
- SAM2 reported the known optional `_C` fill-hole extension warning.
  Propagation completed; optional fill-hole post-processing did not run.
- Exact original producer checkout/command provenance remains partial.

### Next steps

1. Inspect corrected drawer/static contact sheets and frames 181/195/207.
2. Review the six seed masks and near-contact unknown regions.
3. Keep dual TSDF blocked unless drawer-side/cabinet-wall ownership is accepted.

### Git state and outputs

- Implementation base commit: `f72ade23e0de56b25b4cb45e5bd5a692e82001cc`
- Config: `tools/itaco_moving_map_fix/configs/hololens_region_assignment_v4_verified_depth.yaml`
- Report: `reports/assignment_v4/verified_depth_rerun.md`
- Output: `/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v4/region_assignment_v4_verified_depth_0002_001`
- Previous `0.001` outputs remain preserved and unchanged.
