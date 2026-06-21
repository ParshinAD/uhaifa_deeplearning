"""mfas — Minimum Feedback Arc Set thesis infrastructure.

Public API for the evaluation oracle and the Rocket baseline.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import metrics  # noqa: F401  (the oracle — always available)

__all__ = ["metrics", "__version__"]
