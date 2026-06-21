"""Exact feedforward-weight scorer — the ground-truth discrete metric (the "oracle").

This module defines the ONE canonical scoring function used everywhere in the
project. An edge ``(u, v)`` is *feedforward* iff ``pos[u] < pos[v]`` (equivalently
``pos[v] > pos[u]``). The score is the total weight of all feedforward edges; the
feedback (= remaining) weight is ``total_weight - score``.

This is the discrete ground-truth metric, NOT the sigmoid surrogate used to train
Rocket. It must be exact, vectorized and deterministic.

Determinism invariant
----------------------
Weights are summed in their **native integer / float64 dtype**, never float32.
Summing the fly connectome's integer weights in float32 loses precision and yields
a value that is off by a handful of units (this is the source of the historical
``34,751,904`` vs ``34,751,902`` discrepancy). ``_ff_weight`` upcasts any float32
input to float64 before summing so the result is bit-stable.

GOVERNANCE: once ``tests/test_metrics.py`` passes, this scorer defines truth and
must not be modified. See CLAUDE.md.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "score_from_positions",
    "score_from_order",
    "positions_to_order",
    "pct",
    "feedback_weight",
]


def _as_sum_dtype(weights: np.ndarray) -> np.ndarray:
    """Return ``weights`` in a dtype safe for exact summation.

    Integer weights keep their (≥int64) dtype; floating weights are upcast to
    float64. float32 is never summed directly (it loses precision on large graphs).
    """
    weights = np.asarray(weights)
    if np.issubdtype(weights.dtype, np.integer):
        # int64 accumulation is exact for the connectome's synapse counts.
        return weights.astype(np.int64, copy=False)
    return weights.astype(np.float64, copy=False)


def _ff_weight(arr: np.ndarray, src: np.ndarray, tgt: np.ndarray,
               weights: np.ndarray) -> float:
    """Core kernel: total weight of edges with ``arr[tgt] > arr[src]``.

    ``arr`` may be real-valued positions or an integer permutation (ranks); the
    feedforward test ``arr[tgt] > arr[src]`` is identical in both cases. Ties
    (``arr[tgt] == arr[src]``) are counted as NOT feedforward (strict ``>``).
    """
    arr = np.asarray(arr)
    src = np.asarray(src)
    tgt = np.asarray(tgt)
    w = _as_sum_dtype(weights)
    ff = arr[tgt] > arr[src]
    return float(w[ff].sum())


def score_from_positions(positions: np.ndarray, src: np.ndarray, tgt: np.ndarray,
                         weights: np.ndarray) -> float:
    """Feedforward weight for real-valued node positions.

    Parameters
    ----------
    positions : array of shape (n_nodes,)
        ``positions[i]`` is the continuous coordinate of node ``i``.
    src, tgt : int arrays of shape (n_edges,)
        Contiguous node indices for each edge's source / target.
    weights : array of shape (n_edges,)
        Edge weights (int or float). Summed in int64 / float64.

    Returns
    -------
    float
        Total weight of edges where ``positions[tgt] > positions[src]``.
    """
    return _ff_weight(positions, src, tgt, weights)


def score_from_order(order: np.ndarray, src: np.ndarray, tgt: np.ndarray,
                     weights: np.ndarray) -> float:
    """Feedforward weight for a permutation (``order[i]`` = rank of node ``i``)."""
    return _ff_weight(order, src, tgt, weights)


def positions_to_order(positions: np.ndarray) -> np.ndarray:
    """Convert continuous positions to a rank permutation (stable argsort).

    The node with the smallest position gets rank 0. ``score_from_positions`` does
    not require this conversion (it scores positions directly); it is provided for
    callers that need an explicit ordering.
    """
    return np.argsort(np.asarray(positions), kind="stable")


def pct(score: float, total_weight: float) -> float:
    """Feedforward score as a percentage of total edge weight."""
    return 100.0 * score / total_weight


def feedback_weight(score: float, total_weight: float) -> float:
    """Feedback (non-feedforward) weight = ``total_weight - score``."""
    return total_weight - score
