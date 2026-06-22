"""Discrete order-refinement post-phases (leakage-safe).

This package holds local-search refiners that run *after* a continuous optimizer
(e.g. Rocket) has produced an ordering. They operate on the input graph's edge
weights and the current integer-rank vector only; the frozen discrete oracle is
used solely to accept/reject a whole candidate vector (best-by-oracle), never to
choose a move. See :mod:`mfas.refine.insertion`.
"""
from .insertion import (
    build_sift_edges,
    jacobi_best_gaps,
    jacobi_rebuild,
    sift,
)
from .lns import (
    apply_victim_reinsertions,
    back_edge_weight,
    ils_lns,
)

__all__ = [
    "build_sift_edges",
    "jacobi_best_gaps",
    "jacobi_rebuild",
    "sift",
    "back_edge_weight",
    "apply_victim_reinsertions",
    "ils_lns",
]
