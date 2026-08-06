#!/usr/bin/env python3
"""Dependency-free runner for the intentionally simple regression tests."""

from __future__ import annotations

import inspect
import sys
import traceback
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))

import tools.itaco_moving_map_fix.tests.test_moving_map as suite


def main() -> int:
    tests = [
        (name, function)
        for name, function in inspect.getmembers(suite, inspect.isfunction)
        if name.startswith("test_")
    ]
    failures = 0
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failures} passed, {failures} failed")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
