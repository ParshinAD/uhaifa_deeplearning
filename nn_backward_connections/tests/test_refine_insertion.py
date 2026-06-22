"""Unit tests for the full-range exact-gain sift refiner (mfas.refine.insertion).

Covers:
  (1) jacobi_best_gaps gains == brute-force per-node optimum on a small random graph
      WITH an isolated node + a self-loop (0 mismatches);
  (2) jacobi_best_gaps gains == best_gap_for_node_ref gains on a small graph AND on mouse;
  (3) sift best_score >= initial score on mouse (monotone);
  (4) score_from_positions(best_rank.astype(float32)) == score_from_order(best_rank) on mouse.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfas import io
from mfas.metrics import score_from_order, score_from_positions
from mfas.refine import build_sift_edges, jacobi_best_gaps, sift
from mfas.refine.insertion import best_gap_for_node_ref, build_adj
from mfas.io import GraphData


def _make_small_graph(seed: int = 0, n: int = 30, m: int = 90) -> GraphData:
    """Random directed weighted graph WITH an isolated node and a self-loop."""
    rng = np.random.RandomState(seed)
    # Confine edges to nodes [0, n-2] so node n-1 is ISOLATED.
    src = rng.randint(0, n - 1, size=m).astype(np.int64)
    tgt = rng.randint(0, n - 1, size=m).astype(np.int64)
    w = (rng.randint(1, 10, size=m)).astype(np.int64)
    # Force at least one SELF-LOOP.
    src[0] = 5
    tgt[0] = 5
    node_ids = np.arange(n, dtype=np.int64)
    return GraphData(src=src, tgt=tgt, weight=w, node_ids=node_ids, name="small_test")


def _brute_force_best_gain(u: int, rank: np.ndarray, src: np.ndarray,
                           tgt: np.ndarray, w: np.ndarray, n: int) -> float:
    """Brute-force exact best insertion gain for node u (O(n*deg)).

    For every gap g, build the full rank vector with u re-inserted at gap g and score
    ONLY u's incident edges; return max_g(value) - value(current gap).
    """
    p = int(rank[u])
    # order without u: nodes sorted by rank, drop u
    order = np.argsort(rank, kind="stable")          # order[r] = node at rank r
    order_wo = order[order != u]                     # length n-1
    incid = (src == u) | (tgt == u)
    su, tu, wu = src[incid], tgt[incid], w[incid]

    def value_at_gap(g: int) -> float:
        new_order = np.insert(order_wo, g, u)
        rk = np.empty(n, dtype=np.int64)
        rk[new_order] = np.arange(n, dtype=np.int64)
        ff = rk[tu] > rk[su]
        # drop self-loops (never feedforward, never feedback)
        ff = ff & (su != tu)
        return float(w[incid][ff].sum())

    vals = [value_at_gap(g) for g in range(n)]
    cur = value_at_gap(p)
    return max(vals) - cur


def test_jacobi_gains_match_brute_force_small():
    """(1) Vectorized gains == brute-force per-node optimum (isolated node + self-loop)."""
    g = _make_small_graph(seed=3)
    n = g.n_nodes
    rng = np.random.RandomState(7)
    rank = rng.permutation(n).astype(np.int64)

    src, tgt, w = build_sift_edges(g)
    best_gap, gain = jacobi_best_gaps(rank, src, tgt, w, n)

    # brute force uses the FULL edge arrays (it drops self-loops internally)
    fsrc = np.asarray(g.src, dtype=np.int64)
    ftgt = np.asarray(g.tgt, dtype=np.int64)
    fw = np.asarray(g.weight, dtype=np.float64)

    mism = 0
    for u in range(n):
        bf = _brute_force_best_gain(u, rank, fsrc, ftgt, fw, n)
        if abs(bf - gain[u]) > 1e-7:
            mism += 1
    assert mism == 0, f"{mism} nodes disagree with brute force"
    # isolated node (n-1) must have gain 0
    assert gain[n - 1] == 0.0


def test_jacobi_gains_match_reference_small():
    """(2a) jacobi_best_gaps gains == best_gap_for_node_ref gains on a small graph."""
    g = _make_small_graph(seed=11)
    n = g.n_nodes
    rng = np.random.RandomState(5)
    rank = rng.permutation(n).astype(np.int64)

    src, tgt, w = build_sift_edges(g)
    _, gain = jacobi_best_gaps(rank, src, tgt, w, n)

    adj = build_adj(g)
    ref_gain = np.array([best_gap_for_node_ref(u, rank, adj)[1] for u in range(n)])
    assert np.allclose(gain, ref_gain, atol=1e-7)


def test_jacobi_gains_match_reference_mouse():
    """(2b) jacobi_best_gaps gains == best_gap_for_node_ref gains on mouse."""
    g = io.load_dataset("mouse")
    n = g.n_nodes
    rng = np.random.RandomState(99)
    rank = rng.permutation(n).astype(np.int64)

    src, tgt, w = build_sift_edges(g)
    _, gain = jacobi_best_gaps(rank, src, tgt, w, n)

    adj = build_adj(g)
    ref_gain = np.array([best_gap_for_node_ref(u, rank, adj)[1] for u in range(n)])
    assert np.allclose(gain, ref_gain, atol=1e-6)


def test_sift_monotone_mouse():
    """(3) sift best_score >= initial score on mouse (monotone)."""
    g = io.load_dataset("mouse")
    n = g.n_nodes
    rng = np.random.RandomState(123)
    rank0 = rng.permutation(n).astype(np.int64)
    init_score = score_from_order(rank0, np.asarray(g.src), np.asarray(g.tgt), g.weight)

    best_rank, best_score, sweep_log = sift(g, rank0, max_sweeps=30)
    assert best_score >= init_score
    assert len(sweep_log) >= 1


def test_float32_rank_oracle_exact_mouse():
    """(4) score_from_positions(rank.float32) == score_from_order(rank) on mouse."""
    g = io.load_dataset("mouse")
    n = g.n_nodes
    rng = np.random.RandomState(321)
    rank0 = rng.permutation(n).astype(np.int64)
    best_rank, best_score, _ = sift(g, rank0, max_sweeps=30)

    s_order = score_from_order(best_rank, np.asarray(g.src), np.asarray(g.tgt), g.weight)
    s_pos = score_from_positions(best_rank.astype(np.float32),
                                 np.asarray(g.src), np.asarray(g.tgt), g.weight)
    assert s_pos == s_order == best_score


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
