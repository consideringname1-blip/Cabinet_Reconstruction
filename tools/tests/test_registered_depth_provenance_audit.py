import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit_hololens_registered_depth_provenance import (
    candidate_metrics, detect_scale_drift, fit_scale_offset,
    fit_through_origin, parse_consumer_hits, quantity_fit, saturation_count, write_json,
)


class RegisteredDepthProvenanceAuditTests(unittest.TestCase):
    def test_scale_fit_synthetic(self):
        scale = 0.0002
        z = np.linspace(0.4, 3.0, 2000)
        raw = np.floor(z / scale).astype(np.uint16)
        self.assertAlmostEqual(fit_through_origin(raw, z)["scale"], scale, delta=2e-8)

    def test_scale_and_offset_synthetic(self):
        raw = np.arange(1000, 10000, 7)
        fit = fit_scale_offset(raw, raw * 0.0002 + 0.012)
        self.assertAlmostEqual(fit["scale"], 0.0002, places=10)
        self.assertAlmostEqual(fit["offset_m"], 0.012, places=8)

    def test_radial_is_not_optical_z(self):
        z = np.full(1000, 1.2); x = np.linspace(-0.7, 0.7, len(z))
        radial = np.sqrt(x * x + z * z)
        result = quantity_fit(np.floor(z * 5000).astype(np.uint16), {"optical_z": z, "radial": radial})
        self.assertEqual(result["best_quantity"], "optical_z")
        self.assertGreater(result["candidates"]["radial"]["median_abs_residual_m"], 0.02)

    def test_factor_five_mismatch(self):
        z = np.linspace(0.3, 3.5, 2000).reshape(20, 100)
        raw = np.floor(z * 5000).astype(np.uint16); valid = np.ones_like(raw, dtype=bool)
        wrong = candidate_metrics(raw, z, valid, 0.001)
        correct = candidate_metrics(raw, z, valid, 0.0002)
        self.assertLess(wrong["valid_mask_iou"], 0.3)
        self.assertGreater(correct["valid_mask_iou"], 0.99)
        self.assertLess(correct["median_abs_depth_error_m"], 0.0002)

    def test_invalid_zeros_ignored(self):
        raw = np.asarray([0, 0, 5000, 10000], dtype=np.uint16)
        fit = fit_through_origin(raw, np.asarray([99, -5, 1, 2], dtype=float))
        self.assertEqual(fit["sample_count"], 2)
        self.assertAlmostEqual(fit["scale"], 0.0002, places=10)

    def test_uint16_saturation_detection(self):
        raw = np.asarray([0, 100, 65535, 65535], dtype=np.uint16)
        self.assertEqual(saturation_count(raw), 2)

    def test_per_frame_scale_drift_detection(self):
        self.assertFalse(detect_scale_drift([0.0001999, 0.0002, 0.0002001], .005)["drift_detected"])
        self.assertTrue(detect_scale_drift([0.0002, 0.000205], .005)["drift_detected"])

    def test_consumer_audit_path_parsing(self):
        rows = parse_consumer_hits("tools/a.py:12:depth = raw * 0.001\ncode/b.py:9:open(depth_path)\nnoise")
        self.assertEqual([r["consumer_path"] for r in rows], ["tools/a.py", "code/b.py"])
        self.assertEqual(rows[0]["line"], 12)

    def test_deterministic_audit_output(self):
        value = {"z": np.asarray([2, 1]), "a": {"v": np.float64(0.0002)}}
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.json", Path(directory) / "b.json"
            write_json(a, value); write_json(b, value)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(json.loads(a.read_text())["a"]["v"], 0.0002)


if __name__ == "__main__":
    unittest.main()
