#!/usr/bin/env python3
"""
Quick demo script — runs the pipeline on a sample project.

Usage:
    python scripts/run_demo.py /path/to/project /path/to/mask.npy
"""

import sys
from pathlib import Path

# Add src to path for development (before pip install -e .)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rebar_algorithm.cli import main

if __name__ == "__main__":
    sys.exit(main())
