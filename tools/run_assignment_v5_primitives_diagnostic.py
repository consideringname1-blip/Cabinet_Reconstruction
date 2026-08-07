#!/usr/bin/env python3
"""Run only the pre-integration synthetic diagnostic required by Assignment v5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.itaco_ownership_v5.motion import build_active_transitions, decompose_motion_states


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("ready_for_dual_tsdf") is not False:
        raise RuntimeError("v5 primitive diagnostic must keep ready_for_dual_tsdf=false")
    prohibited = set(cfg.get("prohibited_stages", []))
    required = {"autoseg_integration", "sam2_dense_propagation", "dense_ownership", "geometry_fusion",
                "tsdf", "nksr", "mesh"}
    if not required.issubset(prohibited):
        raise RuntimeError("configuration does not explicitly prohibit all post-primitive stages")

    suite = unittest.defaultTestLoader.loadTestsFromName("tools.itaco_ownership_v5.tests.test_primitives")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    states = decompose_motion_states(range(6), [0, 1, 2, 3, 4, 5], [0, 0, .02, .05, .08, .08], cfg["motion_state"])
    transitions = build_active_transitions(states, cfg["active_transition"])
    payload = {
        "stage": cfg["stage"],
        "scope": "synthetic primitives only; no real-data ownership labels created",
        "tests_run": result.testsRun,
        "tests_passed": result.testsRun - len(result.failures) - len(result.errors),
        "tests_failed": len(result.failures) + len(result.errors),
        "motion_states": [row.to_dict() for row in states],
        "active_transitions": [row.to_dict() for row in transitions],
        "prohibited_stages_not_run": sorted(prohibited),
        "ready_for_dual_tsdf": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "primitive_diagnostic.json").write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
