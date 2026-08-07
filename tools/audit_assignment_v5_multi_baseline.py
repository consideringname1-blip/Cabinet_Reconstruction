#!/usr/bin/env python3
"""Run frozen-noise Assignment-v5 multi-baseline discrimination."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.itaco_ownership_v5.multi_baseline_diagnostic import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("tools/itaco_moving_map_fix/configs/hololens_ownership_v5_multi_baseline.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
