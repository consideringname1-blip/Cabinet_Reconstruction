"""CLI for stage 1.5 observation recovery and observability validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline_stage1_5 import run


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.config,args.output),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
