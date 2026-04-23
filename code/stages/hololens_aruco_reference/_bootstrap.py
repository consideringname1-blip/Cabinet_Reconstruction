from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
STAGES_ROOT = PACKAGE_ROOT.parent
CODE_ROOT = STAGES_ROOT.parent
PROJECT_ROOT = CODE_ROOT.parent

for path in (PACKAGE_ROOT, STAGES_ROOT, CODE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
