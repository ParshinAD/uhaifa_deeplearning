"""Pytest configuration: make ``src/`` and the repo root importable.

This lets ``import mfas`` and ``import eval.harness`` resolve without an editable
install, both under pytest and ``python -m eval.harness``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
