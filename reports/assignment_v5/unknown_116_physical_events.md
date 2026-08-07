# Assignment v5 diagnostic on 116 v4 unknown regions

Date: 2026-08-07

## Scope

This diagnostic connects only the reviewed Assignment v5 physical-event
primitives to the existing 116 `static_drawer_both_supported` Assignment v4
regions. Camera poses, registered-depth scale, prismatic axis, frozen `q_t`,
moving map, hand masks, AutoSeg proposals, and formal v4 labels remain frozen.

The run does not execute SAM2, dense propagation, camera/articulation
optimization, TSDF, NKSR, Mesh, GLB, or URDF. It does not modify the formal v4
labels.

## Method

- Raw recording timestamps and frozen `q_t` define `CLOSED_PLATEAU`,
  `ACTIVE_MOTION`, and `OPEN_PLATEAU` states.
- Only frame pairs that traverse ACTIVE_MOTION and exceed the configured
  `delta q` can create ownership.
- Every target candidate is compared directly to the immutable source finite
  surface. Target observations never become new anchors. If multiple target
  surfaces match, identity is ambiguous rather than selected by score.
- Positive moving evidence requires a unique finite source anchor to follow the
  frozen articulated transform over multiple active observations and sufficient
  q span, while outperforming the static explanation.
- Positive static evidence requires a local trusted-drawer occlusion, nearby
  silhouette departure, temporally adjacent reveal, and persistent world-static
  support.
- Free-space contradiction vetoes an otherwise positive route.

## Failed first diagnostic preserved

Output:

`/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v5/unknown_116_physical_events_001`

The first implementation returned 24 moving and 92 unknown. Direct visual
review showed that 19/24 were repeated layer6 floor masks; the other accepted
regions included two early layer0 floor observations and three small layer11
interior observations.

The failure was physical, not UID-specific: translation tangent to a large
plane can keep finding compatible occupied geometry. The truncated RGB/depth
footprint made a large floor plane look like a finite proposal. This violated
the required infinite-plane ambiguity case, so the 24 labels are rejected and
the failed output remains preserved.

## Corrective hard gate

A positive moving route now checks whether articulated motion is tangent to the
source plane. When it is tangent, the proposal must have a reliably observable
finite boundary. Proposal boundary coincident with the RGB observation
footprint is treated as observation truncation, not as a physical surface
anchor.

This gate is generic: it does not use UID, color, floor height, object position,
or fixed region/frame rules. Thresholds are stored in YAML.

## Corrected result

Output:

`/workspace_whz_worktrees/funrec-assignment-v3_outputs/geometry_interior_v5/unknown_116_physical_events_002`

| Decision | Regions |
|---|---:|
| STATIC | 0 |
| MOVING | 0 |
| UNKNOWN | 116 |
| CONFLICTING | 0 |

Additional diagnostics:

- 24 regions satisfied the pre-gate raw moving-route test; all 24 were rejected
  by tangent-motion/finite-boundary ambiguity.
- 102/116 source regions are tangent-motion ambiguous under the observation
  footprint test.
- Median finite-boundary confidence is 0.0122; range is 0.0–0.8529.
- Median absolute source-normal/axis alignment is 0.0900.
- 14 non-tangent regions remain unknown. They are repeated layer3 observations:
  some have 1–5 static anchor-compatible active observations, but none has a
  valid local causal disocclusion event.
- Across all source-target hypotheses, static has 864 verified source-anchor
  observations and 1,926 identity-lost observations; drawer has 558 verified
  and 2,232 lost. Compatibility counts alone are not ownership evidence.
- No accepted local causal static-disocclusion event was found.

The corrected result is conservative but expected: this 116-region set does not
contain sufficient trustworthy finite-surface identity or causal reveal evidence
to release ownership under the v5 architecture.

## Visual review

- `visualization/raw_moving_blocked_tangent_ambiguity.jpg`: the 24 raw-moving
  candidates rejected by the hard gate; visual review shows the dominant floor
  failure mode.
- `visualization/non_tangent_without_positive_event.jpg`: 14 non-tangent regions
  that still lack a positive physical event.
- `visualization/other_unknown_regions.jpg`: remaining 78 unknown regions.
- Each region stores `source_region.jpg`, `identity_timeline.png`, and
  `v5_region_diagnostic.json`.

## Validation

- Assignment v5 tests: 23/23 passed.
- Assignment v4 regression tests: 34/34 passed.
- Verified-depth scale gate passed before frame loading.
- Formal v4 labels remain unchanged.
- Failed `001` and corrected `002` results are separate and preserved.

## Conclusion and missing observation

`ready_for_dual_tsdf=false`.

The missing evidence is a trustworthy temporal finite-surface identity signal,
such as feature/track correspondence that follows the same drawer-side point or
boundary across a sufficiently large ACTIVE_MOTION transition. For positive
static ownership, the recording must also expose a local drawer-occluder reveal
with persistent world-static support. AutoSeg occupancy compatibility alone
cannot provide either identity.
