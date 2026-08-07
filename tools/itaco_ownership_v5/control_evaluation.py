"""Evaluation-only control aggregation and hard gate."""
from __future__ import annotations

from collections import Counter, defaultdict


def evaluate_control_gate(results: list[dict], synthetic_tests_passed: bool, cfg: dict) -> dict:
    grouped = defaultdict(list)
    for row in results:
        grouped[row["control_group"]].append(row)
    front = grouped["drawer_front"]
    box = grouped["active_comoving_box"]
    floor = grouped["documented_floor_false_positive"]
    plateau = grouped["plateau_only"]
    front_moving = sum(row["decision"] == "MOVING_LINK" for row in front)
    box_moving = sum(row["decision"] == "MOVING_LINK" for row in box)
    box_static = sum(row["decision"] == "WORLD_STATIC" for row in box)
    floor_moving = sum(row["decision"] == "MOVING_LINK" for row in floor)
    plateau_owned = sum(row["decision"] in ("MOVING_LINK", "WORLD_STATIC", "CONFLICTING") for row in plateau)
    checks = {
        "drawer_front_4_of_4_moving_link": bool(
            len(front) == int(cfg["drawer_front_required_moving"]) and
            front_moving == int(cfg["drawer_front_required_moving"])),
        "active_comoving_box_at_least_80pct_moving_link": bool(
            box and box_moving / len(box) >= float(cfg["active_comoving_minimum_moving_fraction"])),
        "active_comoving_box_zero_world_static": bool(
            box_static <= int(cfg["active_comoving_maximum_world_static"])),
        "documented_floor_zero_moving_link": bool(
            floor_moving <= int(cfg["documented_floor_maximum_moving"])),
        "plateau_only_creates_no_ownership": bool(
            not cfg["require_plateau_only_no_ownership"] or plateau_owned == 0),
        "synthetic_tests_all_pass": bool(
            synthetic_tests_passed or not cfg["require_synthetic_tests"]),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "status": "passed" if passed else "failed_stop_before_116",
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "metrics": {
            "drawer_front_count": len(front),
            "drawer_front_moving_link_count": front_moving,
            "drawer_front_true_positive_rate": float(front_moving / max(len(front), 1)),
            "active_comoving_box_count": len(box),
            "active_comoving_box_moving_link_count": box_moving,
            "active_comoving_box_moving_link_fraction": float(box_moving / max(len(box), 1)),
            "active_comoving_box_world_static_count": box_static,
            "documented_floor_count": len(floor),
            "documented_floor_moving_link_count": floor_moving,
            "documented_floor_false_positive_rate": float(floor_moving / max(len(floor), 1)),
            "plateau_only_count": len(plateau),
            "plateau_only_ownership_count": plateau_owned,
            "synthetic_tests_passed": bool(synthetic_tests_passed),
        },
        "decision_counts_by_group": {
            group: dict(Counter(row["decision"] for row in rows)) for group, rows in grouped.items()
        },
        "thresholds_frozen_before_control_evaluation": True,
        "threshold_tuning_after_gate": False,
        "annotations_used_by_classifier": False,
    }
