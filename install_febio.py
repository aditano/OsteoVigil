#!/usr/bin/env python3
"""Install or build a repo-local FEBio executable for OsteoVigil."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cpt_predictor.utils.febio_manager import cli


if __name__ == "__main__":
    raise SystemExit(cli())
