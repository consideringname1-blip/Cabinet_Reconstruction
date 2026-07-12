from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shigure_identity import (  # noqa: E402
    AMBIGUOUS,
    MATCHED,
    UNBOUND,
    BindingConflictError,
    EphemeralBindingRegistry,
    match_display_identity,
)


class MatchDisplayIdentityTests(unittest.TestCase):
    def test_multiview_candidate_uses_best_reference(self) -> None:
        result = match_display_identity(
            [1.0, 0.0, 0.0],
            [
                {
                    "display_object_id": "display-a",
                    "references": [
                        {"capture_instance_id": "side", "embedding": [0.0, 1.0, 0.0]},
                        {"capture_instance_id": "front", "embedding": [1.0, 0.0, 0.0]},
                    ],
                },
                {
                    "display_object_id": "display-b",
                    "embedding": [0.7, 0.7, 0.0],
                },
            ],
        )

        self.assertEqual(MATCHED, result["status"])
        self.assertEqual("display-a", result["display_object_id"])
        self.assertEqual("front", result["selected_reference_id"])
        self.assertEqual(2, result["candidate_scores"][0]["valid_reference_count"])

    def test_close_second_candidate_is_ambiguous(self) -> None:
        result = match_display_identity(
            [1.0, 0.0],
            [
                {"display_object_id": "display-a", "embedding": [1.0, 0.0]},
                {"display_object_id": "display-b", "embedding": [0.999, 0.01]},
            ],
            distance_threshold=0.2,
            second_margin=0.05,
            require_second_margin=True,
        )

        self.assertEqual(AMBIGUOUS, result["status"])
        self.assertIsNone(result["display_object_id"])
        self.assertEqual("second_candidate_too_close", result["reason"])

    def test_candidate_above_threshold_is_unbound(self) -> None:
        result = match_display_identity(
            [1.0, 0.0],
            [{"display_object_id": "display-a", "embedding": [0.0, 1.0]}],
            distance_threshold=0.2,
        )

        self.assertEqual(UNBOUND, result["status"])
        self.assertIsNone(result["display_object_id"])

    def test_optional_geometry_score_changes_ranking(self) -> None:
        result = match_display_identity(
            [1.0, 0.0],
            [
                {
                    "display_object_id": "appearance-only",
                    "embedding": [1.0, 0.0],
                    "geometry_score": 0.0,
                },
                {
                    "display_object_id": "geometry-consistent",
                    "embedding": [0.995, 0.1],
                    "geometry_score": 1.0,
                },
            ],
            distance_threshold=0.2,
            require_second_margin=False,
            geometry_weight=0.1,
        )

        self.assertEqual(MATCHED, result["status"])
        self.assertEqual("geometry-consistent", result["display_object_id"])

    def test_never_considers_more_than_five_display_objects(self) -> None:
        candidates = [
            {"display_object_id": f"display-{index}", "embedding": [0.0, 1.0]}
            for index in range(5)
        ]
        candidates.append({"display_object_id": "display-six", "embedding": [1.0, 0.0]})

        result = match_display_identity(
            [1.0, 0.0],
            candidates,
            max_candidates=100,
            require_second_margin=False,
        )

        self.assertEqual(5, result["considered_display_object_count"])
        self.assertEqual(1, result["truncated_display_object_count"])
        self.assertEqual(UNBOUND, result["status"])


class EphemeralBindingRegistryTests(unittest.TestCase):
    def test_one_to_one_binding_and_explicit_rebind(self) -> None:
        registry = EphemeralBindingRegistry(ingress_session_uuid="ingress", max_display_objects=5)
        first = registry.bind("startup", "shigure-1", "display-1")
        self.assertEqual(MATCHED, first.status)

        with self.assertRaises(BindingConflictError):
            registry.bind("startup", "shigure-2", "display-1")
        with self.assertRaises(BindingConflictError):
            registry.bind("startup", "shigure-1", "display-2")

        rebound = registry.bind(
            "startup",
            "shigure-1",
            "display-2",
            allow_rebind=True,
        )
        self.assertEqual("display-2", rebound.display_object_id)
        self.assertIsNone(registry.get_by_display_object("startup", "display-1"))

    def test_epoch_and_reset_invalidate_old_tokens(self) -> None:
        registry = EphemeralBindingRegistry(ingress_session_uuid="ingress")
        first = registry.bind("startup", "shigure-1", "display-1")
        advanced = registry.advance_epoch("startup", "shigure-1")

        self.assertEqual(first.epoch + 1, advanced.epoch)
        self.assertFalse(registry.is_current(first))
        self.assertTrue(registry.is_current(advanced))

        removed = registry.reset(startup_session_id="startup")
        self.assertEqual(1, removed)
        self.assertFalse(registry.is_current(advanced))
        self.assertIsNone(registry.get("startup", "shigure-1"))

    def test_capacity_evicts_least_recent_bound_display(self) -> None:
        registry = EphemeralBindingRegistry(ingress_session_uuid="ingress", max_display_objects=2)
        registry.bind("startup", "shigure-1", "display-1")
        registry.bind("startup", "shigure-2", "display-2")
        registry.advance_epoch("startup", "shigure-1")
        registry.bind("startup", "shigure-3", "display-3")

        evicted = registry.get("startup", "shigure-2")
        self.assertIsNotNone(evicted)
        self.assertEqual(UNBOUND, evicted.status)
        self.assertEqual("capacity_evicted", evicted.reason)
        self.assertIsNotNone(registry.get_by_display_object("startup", "display-1"))
        self.assertIsNotNone(registry.get_by_display_object("startup", "display-3"))

    def test_unbound_and_ambiguous_records_never_create_display_ids(self) -> None:
        registry = EphemeralBindingRegistry(ingress_session_uuid="ingress")
        unbound = registry.mark_unbound(
            "startup",
            "shigure-1",
            reason="distance_threshold",
        )
        ambiguous = registry.mark_ambiguous(
            "startup",
            "shigure-2",
            candidate_display_object_ids=["display-1", "display-2"],
            reason="second_candidate_too_close",
        )

        self.assertEqual(UNBOUND, unbound.status)
        self.assertIsNone(unbound.display_object_id)
        self.assertEqual(AMBIGUOUS, ambiguous.status)
        self.assertIsNone(ambiguous.display_object_id)
        self.assertEqual(("display-1", "display-2"), ambiguous.candidate_display_object_ids)


if __name__ == "__main__":
    unittest.main()
