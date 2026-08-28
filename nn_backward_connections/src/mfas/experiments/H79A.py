"""Variant H79A - H79 arm: the champion stack started from ``reverse_greedy_fas``.

Greedy-FAS on the TRANSPOSED graph, rank reversed (tau 0.089885 vs the champion's construction - the FARTHEST of the three).

This module is three lines of configuration on top of :mod:`mfas.experiments.H79`, which
holds the constructions, the shared pipeline and the full rationale. Read that module's
docstring first - in particular the novelty argument against H48 (which blocks
``ratio_greedy`` as an arm) and M9/M10 (which set the bar this study is measured against).

The deliverable of the H79 family is the SPREAD across arms, not this arm's delta.
"""
from __future__ import annotations

from typing import Optional

from ..baseline.rocket import RocketResult
from ..io import GraphData
from .H79 import HYPOTHESIS as _H79_HYPOTHESIS
from .H79 import run_with_construction

ID = "H79A"
CONSTRUCTION = "reverse_greedy_fas"
HYPOTHESIS = f"[arm {CONSTRUCTION}] {_H79_HYPOTHESIS}"


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """The H64 champion pipeline, started from ``reverse_greedy_fas`` instead of greedy-FAS."""
    return run_with_construction(g, seed, device, time_limit, CONSTRUCTION)
