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
