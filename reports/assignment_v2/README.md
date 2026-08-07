# Ownership assignment v2 results

This package publishes the assignment-only `clean closed core + conservative
revealed-surface expansion` experiment. It is an extended, non-official run
with fixed HoloLens poses, repaired labels, axis, monotonic `q_t`, frame policy,
masks, and calibration. No TSDF, NKSR, mesh, GLB/URDF, or refinement ran.

## Status

The run completed but did not pass ownership-quality acceptance, so
`ready_for_dual_tsdf` is `false`. The q≈0 evidence produced only 2,878 drawer
core points versus 101,097 static core points. The automatic dual-close region
was 85.75% static, 12.42% drawer, and 1.84% unknown, which is not conservative
enough near drawer/cabinet contact.

## Reproduction

```bash
/workspace_whz/envs/video_articulation/bin/python \
  tools/fuse_hololens_articulation_assignment_v2.py \
  --config tools/itaco_moving_map_fix/configs/hololens_assignment_v2.yaml
```

See `config_resolved.yaml` for exact settings, `assignment_v2_comparison.json`
for quantitative results, and `RUN_REPORT.json` for hashes and readiness.
Binary point clouds and the 542,516-observation provenance file are stored with
Git LFS. Six synthetic/validity tests passed. Output file rehashing passed, but
a complete deterministic rerun was not performed. The first O(N×V) performance
failure remains preserved locally but is not included in this package.
