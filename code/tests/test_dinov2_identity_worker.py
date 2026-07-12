from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = CODE_ROOT / "stages" / "hololens3d_reconstruction"
for path in (STAGE_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_dinov2_identity_from_json as dino_worker  # noqa: E402


class _InspectRunner(dino_worker.Dinov2IdentityRunner):
    def __init__(self) -> None:
        super().__init__()
        self.received = None

    def _embed_rgb_mask(self, rgb, mask, *, source, start_total=None):
        self.received = {
            "rgb": np.asarray(rgb).copy(),
            "mask": np.asarray(mask).copy(),
            "source": dict(source),
        }
        return {"embedding": [1.0, 0.0], "source": dict(source)}

    def embed_task(self, json_path):
        return {"legacy_json_path": str(json_path)}


class Dinov2EmbedFilesTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> tuple[Path, Path]:
        color_path = root / "color.png"
        mask_path = root / "mask.png"
        color = np.zeros((8, 10, 3), dtype=np.uint8)
        color[2:6, 3:8] = (10, 20, 200)
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:6, 3:8] = 255
        self.assertTrue(cv2.imwrite(str(color_path), color))
        self.assertTrue(cv2.imwrite(str(mask_path), mask))
        return color_path, mask_path

    def test_embed_files_uses_shared_loaded_rgb_mask_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            color_path, mask_path = self._write_inputs(Path(tmp))
            runner = _InspectRunner()
            result = runner.embed_files(color_path, mask_path)

        self.assertEqual([1.0, 0.0], result["embedding"])
        self.assertIsNotNone(runner.received)
        self.assertEqual((8, 10, 3), runner.received["rgb"].shape)
        self.assertEqual((8, 10), runner.received["mask"].shape)
        self.assertEqual(20, int(np.count_nonzero(runner.received["mask"])))
        self.assertIn("color_path", runner.received["source"])
        self.assertIn("mask_path", runner.received["source"])

    def test_socket_dispatch_accepts_embed_files_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            color_path, mask_path = self._write_inputs(Path(tmp))
            runner = _InspectRunner()
            result = dino_worker.dispatch_embedding_request(
                runner,
                {
                    "action": "embed_files",
                    "color_file": str(color_path),
                    "mask_file": str(mask_path),
                },
            )

        self.assertEqual([1.0, 0.0], result["embedding"])
        self.assertIsNotNone(runner.received)

    def test_socket_dispatch_rejects_unknown_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported DINOv2 identity action"):
            dino_worker.dispatch_embedding_request(_InspectRunner(), {"action": "unknown"})

    def test_socket_dispatch_keeps_legacy_json_path_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "task.json"
            json_path.write_text("{}", encoding="utf-8")
            result = dino_worker.dispatch_embedding_request(
                _InspectRunner(),
                {"json_path": str(json_path)},
            )

        self.assertEqual(str(json_path.resolve()), result["legacy_json_path"])


if __name__ == "__main__":
    unittest.main()
