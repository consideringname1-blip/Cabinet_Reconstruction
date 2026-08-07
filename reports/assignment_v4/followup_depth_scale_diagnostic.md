# Assignment v4 follow-up: accepted masks and registered-depth scale

Status: diagnostic correction identified; the prior v4 formal result must not be
used for fusion or treated as valid ownership evidence until rerun with the
correct depth scale.

## Accepted layer2 masks

The five accepted masks at frames 202–206 cover the same small brown vertical
surface inside the open drawer/cabinet aperture. They do not cover the large
white drawer front. Visually this surface is consistent with an interior rear
panel/back wall, although the present view alone cannot conclusively distinguish
drawer rear panel from cabinet inner wall.

Individual RGB, binary-mask, and RGB-overlay images are saved under:

`results/followup_masks_depth/accepted_layer2_masks/`

The masks contain 977–1,011 pixels, with bounding boxes around x=135–181 and
y=31–83. Under the diagnostic corrected scale they remain revealed-drawer
regions at frames 202–206 with drawer median support 1.0 and zero initial moving
seed overlap.

## Why layer25 was almost entirely contradictory

Layer25 is the large white drawer-front mask and overlaps repaired moving-front
seed evidence by 1.0. The original v4 configuration interpreted stored uint16
registered depth with `depth_scale_to_m=0.001`. Direct comparison with the
corresponding Long Throw world PLY projection shows that stored registered depth
is almost exactly five times PV optical-axis depth:

- frame 195 median stored/PLY ratio: 4.99952
- frame 202 median stored/PLY ratio: 4.99956
- frame 206 median stored/PLY ratio: 4.99954
- correlation before correcting scale: approximately 0.99998

Thus the source front was unprojected approximately five times too far from the
camera, and target measured depth was also interpreted five times too large.
Camera changes make this scale error non-cancelling in world coordinates. The
result was large positive `measured - predicted` residual, classified as
free-space contradiction. Examples under the formal scale include:

- source 202 → target 177: 100% contradiction; median residual +0.514 m
- source 202 → target 213: 99.94% contradiction; median residual +0.237 m
- source 195 → target 177: 100% contradiction; median residual +0.540 m

A diagnostic-only `depth_scale_to_m=0.0002` comparison changes layer25 to strong
drawer evidence:

- source 202 → target 177: 89.04% support; median residual -0.19 mm
- source 202 → target 195: 96.43% support; median residual +0.53 mm
- source 202 → target 213: 100% support; median residual -1.56 mm
- source 195 → targets 177/202/213: 94.88%/100%/99.95% support

Focused full-region aggregation changes layer25 at frames 195 and 202–206 from
mixed/seed-conflict unknown to `drawer_region_geometry_with_seed`.

Formal-scale and corrected-scale projection panels are under:

- `results/followup_masks_depth/moving_front_projection_residuals/`
- `results/followup_depth_scale/moving_front_projection_corrected_scale/`

## Pinhole depth sparsity and Long Throw consistency

The stored nonzero support is not as sparse as v4 reported:

| Frame | stored nonzero | v4 current 0.2–4 m | corrected 0.2–4 m | PLY projected | corrected mask IoU |
|---|---:|---:|---:|---:|---:|
| 195 | 61.67% | 21.94% | 61.67% | 61.67% | 99.996% |
| 202 | 61.79% | 16.90% | 61.79% | 61.79% | 99.996% |
| 206 | 61.91% | 17.16% | 61.91% | 61.91% | 99.984% |

The apparent 17–22% depth sparsity was therefore mostly artificial: values
physically inside 0.2–4 m were multiplied by five and rejected by v4's 4 m gate.
After `/5` correction, pinhole depth and Long Throw→RGB projection agree closely:

- median absolute difference: 0.109–0.111 mm
- within 1 mm: 93.39–94.33%
- within 1 cm: 99.62–99.63%
- within 3 cm: 99.96–99.98%

Requested four-panel images and explicit raw-scale-error images are under:

`results/followup_depth_scale/pinhole_vs_long_throw_corrected_scale/`

## Consequence

This is a diagnostic result only. The original v4 output used the wrong depth
scale and is not valid input to dual TSDF. The code/config of the original run
has not been silently changed and no TSDF, NKSR, Mesh, or parameter optimization
ran. A corrected v4 rerun must use an explicitly audited depth scale and must be
kept in a new output directory.
