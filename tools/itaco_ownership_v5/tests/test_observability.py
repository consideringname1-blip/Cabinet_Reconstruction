import unittest

import numpy as np

from tools.itaco_ownership_v5.observability import source_finite_surface_observability


CFG = {"observation_boundary_margin_pixels": 2,
       "minimum_normal_axis_alignment_for_non_tangent_motion": .3,
       "minimum_finite_boundary_confidence": .5}


def points(normal_axis=False):
    u, v = np.meshgrid(np.linspace(-.1, .1, 20), np.linspace(-.1, .1, 20))
    if normal_axis:
        return np.column_stack((np.zeros(u.size), u.ravel(), v.ravel()))
    return np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size)))


def inputs(clipped):
    mask = np.zeros((40, 40), bool); mask[10:30, 10:30] = True
    footprint = np.ones_like(mask)
    if clipped:
        footprint[:12] = False; footprint[28:] = False; footprint[:, :12] = False; footprint[:, 28:] = False
    frame = {"rgb_footprint": footprint}
    return frame, {"eroded_mask": mask}


class ObservabilityTests(unittest.TestCase):
    def test_tangent_plane_clipped_boundary_is_ambiguous(self):
        frame, proposal = inputs(True)
        result = source_finite_surface_observability(frame, proposal, points(False), [1, 0, 0], CFG)
        self.assertTrue(result["tangent_motion_ambiguity"])

    def test_tangent_plane_with_finite_boundary_is_allowed(self):
        frame, proposal = inputs(False)
        result = source_finite_surface_observability(frame, proposal, points(False), [1, 0, 0], CFG)
        self.assertFalse(result["tangent_motion_ambiguity"])

    def test_motion_normal_to_plane_does_not_need_tangent_boundary_gate(self):
        frame, proposal = inputs(True)
        result = source_finite_surface_observability(frame, proposal, points(True), [1, 0, 0], CFG)
        self.assertFalse(result["tangent_motion_ambiguity"])


if __name__ == "__main__":
    unittest.main()
