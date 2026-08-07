# Assignment v4 follow-up: surface identity and explained visibility

Date: 2026-08-07

Interpretation: **inconclusive**. `ready_for_dual_tsdf=false`.

## Scope and frozen quantities

This is a read-only diagnostic over the verified-depth Assignment v4 result. It
tests whether model-frame surface continuity, target-surface switching,
explained occlusion/disocclusion, and coherent contradiction runs add identity
evidence beyond model-compatible occupancy.

The registered-depth scale (`0.0002 m/count`), camera poses, calibration,
prismatic axis, `q_t`, joint type, moving map, hand masks, frame ranges, AutoSeg
proposals, SAM2 checkpoint, and all Assignment v4 labels remained frozen. The
existing registered-depth hard gate ran again before frame loading and passed.
No Assignment v4 rerun, SAM2 inference, LoFTR, LK, TAPIP3D, BA, propagation,
TSDF, NKSR, Poisson, Mesh, GLB, or URDF ran.

All 116 experiment regions retain the formal Assignment v4 label `unknown`.
The preferences below are experimental diagnostics, not ownership labels.

The input set is an evidence-margin difficult/ambiguity set. It is not a literal
drawer-side/cabinet-wall near-contact boundary, and this experiment does not
claim that boundary is solved.

## Command and outputs

```bash
/workspace_whz/envs/video_articulation/bin/python \
  tools/audit_assignment_v4_surface_identity_visibility.py \
  --config tools/itaco_moving_map_fix/configs/hololens_surface_identity_visibility_audit.yaml
```

Full local output:

`/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v4/surface_identity_visibility_audit_001`

Tracked review bundle:

`reports/assignment_v4/results/surface_identity_visibility_audit/`

The local output contains 1,182 files and is approximately 134 MB. The curated
review bundle is approximately 17 MB and contains no checkpoint, environment,
raw recording, complete per-region image tree, or reconstruction artifact.

## Method

### Existing projective evidence

For source points `P_t^W`, the audit reuses the exact v4 projective-depth
semantics:

```text
STATIC: P_j = P_t^W
DRAWER: P_j = P_t^W + (q_j - q_t) axis
```

`SUPPORTED` means only model-compatible occupancy. `CONTRADICTION` remains
negative evidence. `OCCLUDED` is sent to explicit occluder attribution, and
`UNOBSERVABLE` remains unknown.

### Trusted occluders

The primary trusted drawer occluder is the per-pixel 3/4 agreement of the four
existing front-seed SAM2 propagations at frames 195–198. A 4/4 mask is also
recorded. The two revealed/payload seeds were excluded. SAM2 was not rerun.

The conservative static occluder is the v2 `static_core` projected into each
registered-depth frame with a strict 1.5 cm support threshold. It is used only
to explain foreground occlusion, never to label a source region static.

### Surface descriptors

Each current-frame proposal is unprojected with valid registered RGB-D. Its
deterministic descriptor records point count, robust centroid, PCA eigenvalues,
canonicalized plane normal, planarity, tangent axes, in-plane extents, 3-D
bounding box, and world/drawer-canonical voxel occupancy at 5, 10, and 20 mm.
The primary voxel size is 10 mm.

Normals are sign-canonicalized by forcing the largest-magnitude component to be
positive. Proposal layer/UID is retained only as provenance metadata.

### Continuity chain

For source descriptor `s` and a current-frame target proposal `p`, normalized
compatibility cost is:

```text
C(s,p) = 0.15 min(normal_error / 20 deg, 1)
       + 0.20 min(plane_offset / 0.04 m, 1)
       + 0.25 min(centroid_distance / 0.12 m, 1)
       + 0.10 (1 - extent_compatibility)
       + 0.30 (1 - symmetric_partial_voxel_overlap)
```

The emission adds `0.15 * (1 - supported_fraction)`. A deterministic dynamic
program chooses one compatible target proposal per target frame:

```text
D_j(p) = emission_j(p)
       + min_r [D_(j-1)(r) + 0.35 C(r,p)]
```

Neither emission nor transition cost uses proposal UID. A switch is recorded
only when adjacent selected descriptors cross a configured 3-D normal, plane,
or centroid discontinuity threshold. Overlapping AutoSeg memberships are saved,
but each projected point contributes total fractional membership one rather than
multiple independent votes.

### Experimental preference

A preference requires two independent aligned cues:

1. continuity advantage;
2. explained static disocclusion or drawer attachment;
3. a coherent contradiction run under the opposite hypothesis.

Conflicting cue sets yield `conflicting`; insufficient aligned cues yield
`unresolved`. Thresholds were fixed before the 116-region run and were not tuned
to increase the resolved count.

## Control gate

The control gate passed, but the static controls passed only at the configured
boundary and therefore warrant caution.

| Control | Count | Expected preference accuracy |
|---|---:|---:|
| drawer-front, existing front seeds | 4 | 4/4 = 100% |
| strong world-static geometry | 8 | 6/8 = 75% |
| revealed layer2 co-moving payload | 2 | 2/2 = 100% |

The actual drawer-front controls were frames 195–198, proposal layer 25. The
co-moving controls were frames 211 and 213, proposal layer 2. These identities
came from the audited seed manifest and formal evidence selection; no layer was
hard-coded as an algorithm rule. Layer2 is a box moving with the drawer, not a
drawer structural surface.

The two unresolved static controls were `frame_198_layer_1` and
`frame_198_layer_3`. Both had a large static continuity advantage but lacked a
second independent cue. This is intended conservative behavior.

## Results on 116 ambiguity regions

| Experimental preference | Count | Fraction |
|---|---:|---:|
| static preferred | 1 | 0.86% |
| drawer preferred | 16 | 13.79% |
| unresolved | 99 | 85.34% |
| conflicting | 0 | 0% |

The only static preference is `frame_195_layer_28`. The 16 drawer preferences
are ten layer-6 regions and six layer-11 regions between frames 189 and 198.
Because these are repeated observations of a small number of surfaces, 17/116
preferences are not 17 independent physical parts.

### Surface continuity

- static continuity advantage: 60 regions;
- drawer continuity advantage: 16 regions;
- indistinguishable under the 0.12 advantage threshold: 40 regions.

Median continuity scores are 0.711 static and 0.552 drawer. A static continuity
advantage alone is not ownership: 59/60 such regions lacked a second aligned
cue and remained unknown. This is the main reason the method does not simply
turn the ambiguity set into static.

### Target-surface switching

- STATIC chain: 34 regions switch, 51 adjacent 3-D discontinuity events, maximum
  two events in one region;
- DRAWER chain: five regions switch, five events, maximum one event per region.

The repeated STATIC switches are concentrated around transitions involving
frame-local layer-3 surfaces and earlier layer-25/11/12 surfaces. DRAWER switches
occur only for several layer-0/28 sources. These layer numbers are provenance;
the switch decision comes from descriptor discontinuity, not equal/different UID.

The smaller DRAWER switch count does not itself imply drawer ownership. DRAWER
still has lower median continuity and many selected candidates are incompatible
with the source descriptor even when the selected target sequence does not jump
between adjacent descriptors.

### Explained occlusion

Aggregate occluded projected-point attribution is:

| Hypothesis | trusted drawer | conservative static | unknown occluder |
|---|---:|---:|---:|
| STATIC | 147,973 | 120 | 16,018 |
| DRAWER | 23,651 | 105,690 | 285,471 |

At least one trusted-drawer occlusion occurs in 36/116 regions; at least one
static-core and unknown-occluder event occurs in all 116. These any-hit region
counts are not evidence strength. The point totals show that DRAWER predictions
are more often blocked by unknown/static-core foreground, whereas STATIC
predictions are commonly behind the trusted front.

The median depth gap for STATIC occlusions is 0.731 m, versus 0.043 m for DRAWER.
The very large STATIC gap indicates that some projections are far behind the
observed front and should not be interpreted as a clean local cabinet-wall
disocclusion merely because the front pixel is trusted drawer.

### q-dependent visibility and disocclusion

STATIC same-surface support rises across early/middle/late q bins from 0.561 to
0.766 to 0.990. Trusted-drawer occlusion falls from 0.163 to 0.084 to 0.063.
DRAWER same-surface support rises from 0.351 to 0.598 to 0.832, while unknown
occlusion falls from 0.297 to 0.117 to 0.026.

Eleven regions pass the configured static-disocclusion pattern, but six of them
are layer-11 regions that receive drawer preference from stronger drawer
continuity plus attachment cues. This shows that explained occlusion alone can
be physically non-unique. Fifty-eight regions pass the drawer-attachment cue,
but most lack a drawer continuity advantage and remain unresolved.

### Coherent contradiction runs

No region has a coherent STATIC-hypothesis contradiction run. Two regions have
coherent DRAWER-hypothesis contradiction:

- `frame_187_layer_0`: target frames 204–213, q span 0.01577 m;
- `frame_195_layer_28`: target frames 200–213, q span 0.02679 m.

The latter combines static continuity advantage with the opposite-hypothesis
contradiction run and is the sole `static_preferred` result. This confirms that
run analysis can expose evidence hidden by median support, but it affects only
2/116 regions here.

## Interpretation

The result is **inconclusive**, not failed: drawer-front, static, and co-moving
controls are mostly/fully separated, so the cues contain real signal. It is not
promising enough to change ownership because 99/116 regions remain unresolved,
the 17 preferences are spatially repetitive, static controls only just meet the
75% gate, and no manually confirmed drawer-side/cabinet-inner-wall region exists.

Surface continuity helps explain why occupancy was ambiguous, but target
surface identity is still not sufficiently observable in this recording. In
particular, partial planar overlap and occluder identity do not provide enough
independent evidence for most regions.

`ready_for_dual_tsdf=false`. Remaining blockers are manual side/wall ground
truth, stronger finite-surface identity across q, and trustworthy attribution
for the large unknown/static-core occlusion population. No Assignment v5,
propagation, or reconstruction should start from these preferences.

## Outputs and validation

Every one of the 116 ambiguity regions stores:

- source metadata and original v4 evidence;
- STATIC and DRAWER per-target depth evidence;
- all/dominant proposal memberships and split entropy;
- selected surface descriptor and continuity metrics;
- UID-independent switch metrics;
- occluder attribution and depth gaps;
- contradiction runs and temporal cues;
- experimental preference, cues, and unresolved reason;
- titled source, q-time strip, two model-frame continuity plots, occlusion panel,
  surface-switch plot, and overview.

Representative cases are selected automatically for strongest static/drawer
preference, most unresolved, highest switching, strongest static disocclusion,
and strongest contradiction run. Drawer-front and layer2 controls are included.

Validation:

- 34/34 Assignment v4 and new surface-identity tests passed;
- registered-depth hard gate passed before formal frame loading;
- 116/116 ambiguity regions and 14/14 controls completed;
- no cross-frame UID identity is used;
- overlapping proposal membership is not duplicated as independent evidence;
- original v4 labels and point clouds are unchanged;
- no SAM2 rerun or reconstruction/refinement stage ran;
- `git diff --check` passed before documentation updates.
