# Registered-depth Physical-Unit & Provenance Audit

This audit is read-only. It did not rerun Assignment v4, SAM2, TSDF, NKSR, Mesh,
or any camera/axis/q/moving-map optimization.

## Plain-language answer

One nonzero integer in `pinhole_projection/depth/*.png` is **not one millimetre**.
The matched Microsoft StreamRecorderConverter first constructs a 3-D Long Throw
point in metres, projects that point into a fixed 320x288 virtual pinhole camera,
takes the point's optical-axis `Z`, multiplies it by 5000, truncates it to
`uint16`, and writes the PNG. Therefore the producer contract is
`stored = uint16(Z_m * 5000)`. The nominal decoder is `Z_m = stored / 5000`,
with a quantization interval of 0.2 mm per count.

`0.001 m/count` was wrong because local consumers treated any uint16 depth PNG
as millimetres. Git history first records this assumption without producer
evidence. `0.0002` comes directly from the matched producer's explicit factor
5000, and the independent robust fit is `0.000200013844825 m/count`.

The original Long Throw PGM stores radial range in millimetres. Its LUT rays have
unit norm. The registered PNG is different: it stores optical-axis Z in a virtual
pinhole frame aligned to the Long Throw point cloud. It is not radial range and
it is not true PV-camera Z. The paired RGB is PV color resampled onto that virtual
depth view, which caused earlier reports to call the frame “PV” too loosely.

## Provenance level

- Producer family and formula: verified by the exact output signature and the
  Microsoft source at `207205596840ae6e7b8c2a795d35c3c4e7bec22e`.
- Exact checkout/command used on 2026-07-30: not recorded locally; revision-level
  execution provenance remains partial.

Reference source at `207205596840ae6e7b8c2a795d35c3c4e7bec22e`:

- [save_pclouds.py](https://github.com/microsoft/HoloLens2ForCV/blob/207205596840ae6e7b8c2a795d35c3c4e7bec22e/Samples/StreamRecorder/StreamRecorderConverter/save_pclouds.py)
- [utils.py](https://github.com/microsoft/HoloLens2ForCV/blob/207205596840ae6e7b8c2a795d35c3c4e7bec22e/Samples/StreamRecorder/StreamRecorderConverter/utils.py)
- Unit contract: verified for the matched converter lineage; the factor 5000 is
  explicit and has been present since the initial public converter history.
- Empirical scale: consistent across 11 representative closed/interaction/open frames.

The repository-local `code/Hololens2/DepthConvertToRGB/align_pv_depth.py` is not
this artifact's producer: it writes a differently named PV-sized alignment using
`pv_z * 1000`, whereas this recording has the Microsoft converter's fixed
320x288 calibration, `*_proj.png` names, and `depth.txt/rgb.txt/trajectory.xyz/odometry.log` bundle.

## Historical impact

Affected and requiring rerun: Assignment v3, v3 depth-projective, Assignment v4 formal.

Not affected by this unit issue: official-compatible iTACO baseline, moving-map fix, centroid axis, monotonic q_t, geometry_interior_v1, Assignment v2, static TSDF diagnostics, pose-error TSDF diagnostics, articulated GLB/URDF.

This does not mean every affected scientific result is otherwise wrong, and it
does not retroactively invalidate PLY-based reconstruction paths.

## Can corrected v4 run now?

`corrected_v4_rerun_allowed = false`.
The physical contract is now supported, but this audit was not user authorization
to execute a corrected research run. The next step is user approval for a new,
separate corrected-v4 output plus a loader scale-consistency gate.
