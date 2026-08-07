# Assignment v4: Region-level Articulation Consistency

Point tracking / LoFTR is not used as the primary ownership mechanism in Assignment v4.

- Parent commit: `f72ade23e0de56b25b4cb45e5bd5a692e82001cc`
- Frozen poses/axis/q/moving-map: verified
- Regions: static=78, drawer=76, unknown=192, invalid=344
- Revealed drawer regions (seed overlap 0): 39
- Mixed regions: 0
- SAM2 propagation: ran; seeds=6
- TSDF/NKSR/Mesh/GLB/URDF: not run
- Ready for dual TSDF: **false** (explicit human inspection is absent)
