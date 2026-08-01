"""Answer-free random directed-weighted-graph generators (Track C null models).

Each generator returns a :class:`mfas.io.GraphData` with contiguous node indices and NO
planted ordering. Weights stay in a native integer/float dtype (never float32) so the frozen
scorer stays exact. All randomness flows from an explicit ``seed`` (reproducibility invariant).

SKELETON: signatures are fixed; bodies raise ``NotImplementedError`` until Track C is built.
See ``experiments/randomgraph.md`` for the estimator/matching contract and null-model list.

Design notes for the implementer (do not delete — these are the matching decisions):
- "*_like(g, ...)" generators take a real :class:`GraphData` and MATCH a chosen invariant of
  it (n + edge count, exact degree sequence, block structure), so the null is comparable.
- Preserve the weight *distribution* explicitly (e.g. resample the real weight multiset) rather
  than inventing weights, or the feedback comparison confounds topology with weight scale.
- Return self-loop-free graphs (self-loops are neither feed-forward nor feedback and would skew
  ``total_weight``); document any multi-edge policy.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..io import GraphData

__all__ = [
    "erdos_renyi_like",
    "configuration_model_like",
    "stochastic_block_model",
    "degree_preserving_rewire",
]


def erdos_renyi_like(g: GraphData, seed: int, name: Optional[str] = None) -> GraphData:
    """Erdős–Rényi digraph matched to ``g`` on node count and edge count (naïve null).

    Draw ``g.n_edges`` directed edges uniformly among ordered node pairs (no self-loops),
    assigning weights resampled from ``g.weight``. This is the *structure-free* floor.
    """
    raise NotImplementedError("Track C skeleton — see experiments/randomgraph.md G01")


def configuration_model_like(g: GraphData, seed: int, name: Optional[str] = None) -> GraphData:
    """Directed configuration model matched to ``g``'s exact in/out degree sequence.

    Preserves every node's in- and out-degree (the degree-heterogeneity null); randomises
    which stubs connect. Weights resampled from ``g.weight``.
    """
    raise NotImplementedError("Track C skeleton — see experiments/randomgraph.md G01")


def stochastic_block_model(
    block_sizes: Sequence[int],
    block_prob: np.ndarray,
    seed: int,
    weight_pool: Optional[np.ndarray] = None,
    name: str = "sbm",
) -> GraphData:
    """Directed stochastic block model — the *structured* null.

    ``block_prob[i, j]`` = edge probability from a node in block ``i`` to one in block ``j``
    (asymmetric ⇒ directed community structure). Fit ``block_sizes``/``block_prob`` to a real
    graph's communities (e.g. SBM MLE / spectral communities) to get a structure-matched null.
    Weights drawn from ``weight_pool`` (default: resample the real weight multiset).
    """
    raise NotImplementedError("Track C skeleton — see experiments/randomgraph.md G01")


def degree_preserving_rewire(
    g: GraphData, seed: int, n_swaps: Optional[int] = None, name: Optional[str] = None
) -> GraphData:
    """Degree-preserving directed edge rewiring (double-edge swap null).

    Randomise wiring while keeping every node's in/out degree *and* the exact weight multiset;
    the closest null to ``g`` that destroys only higher-order structure. ``n_swaps`` defaults to
    a few × ``g.n_edges``.
    """
    raise NotImplementedError("Track C skeleton — see experiments/randomgraph.md G01")
