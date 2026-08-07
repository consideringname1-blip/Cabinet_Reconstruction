#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.itaco_funrec_assignment_v3.pipeline import main

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); args=parser.parse_args(); main(args.config)
