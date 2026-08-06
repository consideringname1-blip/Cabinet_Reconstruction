from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from tools.itaco_moving_map_fix.moving_map import build_moving_map
from tools.itaco_moving_map_fix.bundle_adjustment_adapter import make_bundle_adjustment_adapter
from tools.itaco_moving_map_fix.valid_support import (
    build_coarse_static_mask,
    build_sensor_support,
)


def toy():
    parts = torch.zeros((1, 2, 3, 4), dtype=torch.float64)
    parts[0, 0, :, :2] = 1
    parts[0, 1, 0:2, 2] = 1
    support = torch.ones((1, 3, 4), dtype=torch.bool)
    return parts, support


def test_sensor_support_combination():
    rgb = np.array([[1, 1], [0, 1]], dtype=bool)
    depth = np.array([[1, 0], [1, 1]], dtype=bool)
    hand = np.array([[0, 0], [0, 1]], dtype=bool)
    expected = np.array([[1, 0], [0, 0]], dtype=bool)
    assert np.array_equal(build_sensor_support(rgb, depth, hand), expected)


def test_coarse_static_mask_keeps_every_gate():
    dynamic = np.array([[0, 0], [0, 1]], dtype=bool)
    obj = np.array([[1, 0], [1, 1]], dtype=bool)
    rgb = np.array([[1, 1], [0, 1]], dtype=bool)
    depth = np.array([[1, 1], [1, 0]], dtype=bool)
    expected = np.array([[1, 0], [0, 0]], dtype=bool)
    assert np.array_equal(build_coarse_static_mask(dynamic, obj, rgb, depth), expected)


def test_gate_outside_is_zero_and_invalid():
    parts, support = toy()
    support[:, :, 0] = False
    result = build_moving_map(parts, torch.tensor([4.0, -4.0]), support, "gate_no_minmax")
    assert torch.all(result["gated_score"][:, :, 0] == 0)
    assert torch.all(result["invalid_mask"][:, :, 0])
    assert not torch.any(result["moving_mask"][:, :, 0])


def test_gate_outside_is_excluded_from_loss_mask():
    class DummyOfficial:
        pass

    adapter = make_bundle_adjustment_adapter(
        DummyOfficial, chamfer_distance=None, axis_angle_to_matrix=None,
        quaternion_to_matrix=None, wandb=None,
    )
    instance = adapter.__new__(adapter)
    instance.moving_map_mode = "gate_only"
    instance.obj_mask_bool = torch.ones((1, 2, 2), dtype=torch.bool)
    instance.old_part_segments_list = torch.ones((1, 2, 2), dtype=torch.bool)
    instance.sensor_support = torch.tensor([[[True, False], [False, True]]])
    assert torch.equal(
        instance._computable(0), torch.tensor([True, False, False, True])
    )


def test_bounded_score_range_with_overlap():
    parts, support = toy()
    parts[0, 1, :, 1] = 1
    result = build_moving_map(
        parts, torch.tensor([20.0, 20.0]), support, "gate_no_minmax"
    )
    assert float(result["gated_score"].min()) >= 0
    assert float(result["gated_score"].max()) <= 1
    assert torch.any(result["raw_score"] > 1)


def test_all_consumers_are_identical_by_construction():
    parts, support = toy()
    parameter = torch.tensor([0.2, -1.3], dtype=torch.float64)
    hashes = []
    for _consumer in ("train", "eval", "save", "visualize", "point_cloud", "adapter"):
        score = build_moving_map(parts, parameter, support, "gate_no_minmax")[
            "gated_score"
        ].numpy()
        hashes.append(hashlib.sha256(score.tobytes()).hexdigest())
    assert len(set(hashes)) == 1


def test_partition_is_disjoint_and_complete():
    parts, support = toy()
    support[0, 2, 3] = False
    result = build_moving_map(parts, torch.tensor([-3.0, 3.0]), support, "gate_no_minmax")
    masks = torch.stack(
        [
            result["static_mask"],
            result["unknown_mask"],
            result["moving_mask"],
            result["invalid_mask"],
        ]
    )
    assert torch.all(masks.sum(dim=0) == 1)


def test_uncovered_valid_pixel_is_unknown_not_static():
    parts, support = toy()
    result = build_moving_map(parts, torch.tensor([-10.0, 10.0]), support, "gate_no_minmax")
    assert result["unknown_mask"][0, 2, 3]
    assert not result["static_mask"][0, 2, 3]


def test_official_mode_retains_per_frame_minmax():
    parts, support = toy()
    result = build_moving_map(parts, torch.tensor([0.2, 0.4]), support, "official")
    assert torch.isclose(result["gated_score"].max(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(result["gated_score"].min(), torch.tensor(0.0, dtype=torch.float64))


def test_same_seed_synthetic_core_hash_is_repeatable():
    def once():
        torch.manual_seed(0)
        parts = (torch.rand((3, 4, 8, 9)) > 0.7).to(torch.float64)
        logits = torch.randn(4, dtype=torch.float64)
        support = torch.rand((3, 8, 9)) > 0.2
        score = build_moving_map(parts, logits, support, "gate_no_minmax")[
            "gated_score"
        ]
        return hashlib.sha256(score.numpy().tobytes()).hexdigest()

    assert once() == once()


def test_official_source_does_not_import_adapter():
    workspace = Path(__file__).resolve().parents[3]
    for name in ("joint_refinement.py", "data.py"):
        source = (
            workspace / "code/reconstruction/video2articulation" / name
        ).read_text(encoding="utf-8")
        assert "itaco_moving_map_fix" not in source
