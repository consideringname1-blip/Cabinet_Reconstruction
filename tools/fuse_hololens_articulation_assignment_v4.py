#!/usr/bin/env python3
"""Run Assignment v4 region-level articulation consistency."""
from __future__ import annotations

import argparse
from pathlib import Path

from itaco_region_assignment_v4.pipeline import main

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); args=parser.parse_args(); main(args.config)
