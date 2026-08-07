#!/usr/bin/env python3
"""Run the independent Assignment v4 near-contact ambiguity audit."""
from __future__ import annotations

import argparse
from pathlib import Path

from itaco_region_assignment_v4.near_contact_audit import run

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args()
    run(args.config)
