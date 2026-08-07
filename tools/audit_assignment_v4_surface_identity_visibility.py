#!/usr/bin/env python3
"""Run the fixed-input Assignment v4 surface identity/visibility audit."""
from __future__ import annotations

import argparse
from pathlib import Path

from itaco_region_assignment_v4.surface_identity_visibility import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
