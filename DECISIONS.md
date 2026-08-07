# Workspace A Architectural Decisions

Tracked workspace: workspace_whz/

Last updated: 2026-08-06

## Decision log

## 2026-08-07 — FunREC-inspired Assignment v3 remains a diagnostic extension

- Status: accepted for implementation; not accepted for TSDF input.
- Decision: infer motion identity only from periodic interaction tracks under the
  frozen HoloLens pose/prismatic motion model, vote on explicit AutoSeg regions,
  and propagate drawer identity with the audited local SAM2.1 predictor.
- Decision: retain static/drawer/unknown/invalid as mutually exclusive states;
  static is positive evidence, never `valid & ~drawer`.
- Decision: open-only and conflicting propagation remains unknown. No UID, color,
  floor height, fixed ROI, or object-position ownership rule is permitted.
- Decision: preserve the strict 2 px depth-boundary run even though it exposes
  sparse open-depth support; do not hide the failure by changing thresholds in
  place.
- Consequence: `ready_for_dual_tsdf=false` until near-contact observability,
  drawer-side retention, inner-wall leakage, and independent review pass.

## 2026-08-07 — Assignment v4 uses region-level direct projective evidence

- Status: accepted as an extended diagnostic design; not accepted for fusion.
- Decision: use frame-local AutoSeg surface regions as the ownership unit and
  compare world-static against the frozen prismatic drawer model directly in
  registered target RGB-D. Predicted-surface z-buffering is required; occlusion
  is neutral, free space is contradictory, and unobservable samples do not enter
  evidence denominators.
- Decision: moving labels are seed priors only. Static is never a remainder,
  proposal layers are provenance only, and mixed regions remain unknown.
- Decision: run audited SAM2 only after direct geometry accepts drawer seeds;
  propagation cannot override geometry and open-only surfaces remain unknown.
- Consequence: current positive static and open propagation support are too
  sparse. Human review is absent, so `ready_for_dual_tsdf=false` and no
  reconstruction stage may run from this result.

## 2026-08-07 — Assignment v4 formal 0.001 depth scale is invalidated

- Status: accepted diagnostic finding; corrected full assignment not yet run.
- Evidence: stored pinhole registered depth is approximately 4.9995x the same
  Long Throw world PLY reprojected into the recorded virtual pinhole camera. A diagnostic 0.0002 m/count
  scale restores >99.98% valid-mask IoU and millimetre-scale depth agreement.
- Decision: preserve the 0.001-scale v4 output as a failed diagnostic and prohibit
  it from dual TSDF. Do not silently overwrite its config or artifacts.
- Decision: require an explicit physical-unit/provenance audit and a hard
  pinhole-versus-PLY scale-consistency gate before the corrected scale becomes a
  frozen pipeline input.
- Consequence: the earlier near-total layer25 contradiction is attributed mainly
  to depth scale, not to the fixed axis or `q_t`; corrected ownership still needs
  a separate full rerun and human review.

## 2026-08-07 — Registered-depth unit contract verified; execution revision remains partial

- Status: provenance/unit diagnostic complete; corrected Assignment v4 not run.
- Evidence: the recording exactly matches Microsoft HoloLens2ForCV
  StreamRecorderConverter output (`*_proj.png`, fixed K=200/200/160/144,
  depth/rgb indexes, trajectory and odometry). The matched source explicitly
  writes `uint16(virtual_pinhole_Z_m * 5000)`. Across 11 closed/interaction/open
  frames, the robust fitted scale is 0.000200013845 m/count with no drift.
- Decision: treat 0.0002 m/count as the verified unit contract for this artifact
  family. The quantity is virtual Long Throw pinhole optical-axis Z, not radial
  range and not true PV-camera Z.
- Limitation: the exact converter checkout and literal command used for this
  recording were not preserved, so producer execution provenance is partial.
- Consequence: Assignment v3 and v4 registered-PNG results require rerun; PLY-
  based official preprocess, moving-map, axis/q, v1/v2 geometry and GLB/URDF are
  not invalidated by this unit issue. Corrected v4 required separate user approval.

## 2026-08-07 — Verified-depth v4 passes hard gate but remains blocked from fusion

- Status: corrected Assignment v4 completed; not accepted for dual TSDF.
- Decision: every Assignment v4 run must pass a pre-assignment comparison of
  registered uint16 depth against matching Long Throw world PLY projections.
  Missing or disabled gate configuration is a hard failure.
- Evidence: the authorized `0.0002 m/count` run passed all nine reference
  frames with 99.9857% minimum valid-mask IoU and 0.181 mm worst-frame p90
  absolute error. It recovered 78 static and 76 drawer regions, versus 1 and 5
  under the invalid `0.001` scale.
- Consequence: the scale failure is resolved for this run, but near-contact
  ownership remains completely unknown and human review is absent.
  `ready_for_dual_tsdf=false`; no reconstruction stage may consume it yet.

## 2026-08-07 — Near-contact ambiguity is primarily dual-model non-identifiability

- Status: audit complete; all audited regions remain unknown.
- Evidence: of 131 verified-depth difficult regions, 116 have median support at
  least 0.90 and contradiction at most 0.10 under both the world-static and
  frozen-drawer hypotheses; 14 are insufficient and one is borderline both-bad.
- Decision: do not propagate, threshold-relax, or force either ownership label
  for this set. The automatically selected set spans cabinet, drawer, edge, and
  floor/background surfaces and is not a literal near-contact-only subset.
- Decision: preserve proposal IDs only as target-surface provenance. Overlapping
  proposal hits and nearby planar surfaces require stronger cross-frame surface
  identity and negative evidence before ownership can be released.
- Consequence: `ready_for_dual_tsdf=false`; pose, axis, `q_t`, moving map, and
  reconstruction remain frozen.

## 2026-08-07 — Surface identity continuity is informative but inconclusive

- Status: diagnostic complete; formal Assignment v4 labels remain unchanged.
- Evidence: drawer-front controls pass 4/4, static controls 6/8, and layer2
  co-moving controls 2/2. Among 116 both-supported ambiguity regions, the
  experimental preference is 1 static, 16 drawer, 99 unresolved, 0 conflicting.
- Decision: require two independent identity cues and retain all 116 formal
  regions as unknown. Continuity advantage, visibility fraction, explained
  occlusion, proposal UID, or a single contradiction interval cannot assign
  ownership alone.
- Decision: treat the result as inconclusive. Do not claim drawer-side versus
  cabinet-inner-wall resolution without manually confirmed regions.
- Consequence: `ready_for_dual_tsdf=false`; no Assignment v5, propagation, TSDF,
  NKSR, Mesh, or parameter refinement follows from this audit.

## 2026-08-07 — Assignment v5 begins with physical-event primitives only

- Status: primitive layer implemented and synthetically verified; no real-data classifier or reconstruction.
- Decision: ownership creation requires an ACTIVE_MOTION transition and immutable source-anchored finite-surface identity. Target observations never become new anchors; failed identity is unknown rather than a nearby-surface substitution.
- Decision: positive static disocclusion requires a local causal reveal chain. Large same-ray depth gaps, plateau-only support, occupancy compatibility, and propagation without an established identity cannot create ownership.
- Decision: correlated metrics from one physical observation remain one evidence family. Independent positive static and moving families produce `CONFLICTING`, not a score-based winner.
- Consequence: AutoSeg/SAM2 integration, dense ownership, object-scope filtering, TSDF, NKSR, and Mesh remain out of scope. `ready_for_dual_tsdf=false`.

## 2026-08-07 — Tangent-plane occupancy cannot establish v5 moving ownership

- Status: accepted correction from the 116-region real-data diagnostic; ownership remains unresolved.
- Evidence: the first v5 adapter labeled 24 regions moving, including 19 repeated layer6 floor masks. Translation tangent to these planar surfaces found compatible occupied geometry without establishing finite physical identity.
- Decision: when articulated motion is tangent to a source plane, positive moving ownership requires a reliably observable finite boundary. Proposal edges coincident with the RGB/depth observation footprint are truncation boundaries, not physical anchors.
- Decision: preserve failed `001`; use corrected `002`, where all 116 regions remain unknown. The gate is geometry/observability based and must not use UID, color, floor height, object position, or fixed frame rules.
- Consequence: compatibility counts remain diagnostic only. No valid causal static disocclusion was observed, and `ready_for_dual_tsdf=false` remains mandatory.
