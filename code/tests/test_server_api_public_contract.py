from __future__ import annotations

import ast
import unittest
from pathlib import Path


SERVER_API_PATH = Path(__file__).resolve().parents[1] / "server_api.py"


def _server_api_tree() -> ast.Module:
    return ast.parse(SERVER_API_PATH.read_text(encoding="utf-8"), filename=str(SERVER_API_PATH))


def _function_node(name: str) -> ast.FunctionDef:
    for node in _server_api_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"server_api function not found: {name}")


class ServerApiPublicContractTests(unittest.TestCase):
    def test_tracking_snapshot_does_not_reactivate_persisted_objects(self) -> None:
        function = _function_node("_tracking_mode_items")
        called_attributes = {
            node.func.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        self.assertNotIn("activate_display_object", called_attributes)

        response_keys = {
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue(
            {
                "latest_body_revision",
                "latest_body_task_id",
                "body_evidence",
                "latest_body",
                "evidence",
                "tracking_status",
                "tracking_reason",
                "tracking_event_sequence",
            }.issubset(response_keys)
        )

    def test_spatial_box_sanitizer_keeps_only_public_preview_coordinates(self) -> None:
        function = _function_node("_public_sam3_spatial_box")
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace: dict = {}
        exec(compile(module, str(SERVER_API_PATH), "exec"), namespace)

        result = namespace["_public_sam3_spatial_box"](
            {
                "status": "ready",
                "coordinate_space": "unity_world",
                "center_world": [1.0, 2.0, 3.0],
                "aabb_min_world": [0.0, 0.0, 0.0],
                "canonical_coordinate_space": "aruco_local",
                "center_aruco": [4.0, 5.0, 6.0],
                "corners_aruco": [[0.0, 0.0, 0.0]],
                "aruco_sync_source": "test",
                "diagonal_m": 1.0,
            }
        )

        self.assertEqual("unity_world", result["coordinate_space"])
        self.assertEqual([1.0, 2.0, 3.0], result["center_world"])
        self.assertNotIn("canonical_coordinate_space", result)
        self.assertNotIn("center_aruco", result)
        self.assertNotIn("corners_aruco", result)
        self.assertNotIn("aruco_sync_source", result)
        self.assertNotIn("diagonal_m", result)


if __name__ == "__main__":
    unittest.main()
