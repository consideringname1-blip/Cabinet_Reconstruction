# Cabinet Reconstruction Project Status

This file is the canonical handoff summary for `/workspace_whz`.

- Tracked workspace: `/workspace_whz`
- Excluded workspace: `/workspace` — do not inspect, modify, or include it in project status
- Last updated: 2026-08-06
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
