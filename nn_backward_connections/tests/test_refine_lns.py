"""Unit tests for the ILS/LNS refiner (mfas.refine.lns) — the H31 wrapper.

Covers:
  (1) back_edge_weight matches a brute-force per-node feedback-weight count on a small
      graph (target-blind: input weights + ranks only);
  (2) apply_victim_reinsertions returns a valid permutation and never decreases the score
      below the seed (each victim moved to an exact-optimal gap; monotone is NOT guaranteed
      for the perturbation alone — we only assert it stays a permutation);
  (3) ils_lns best_score >= sift best_score >= initial score on mouse (monotone,
      oracle-gated best-by-oracle);
  (4) ils_lns "kick" mode is also monotone (random perturbation, oracle-gated);
  (5) leakage guard: the perturb/repair selectors take only (rank, src, tgt, w, n) — no
      GraphData / oracle handle — so a move can never read the discrete score.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from mfas import io
from mfas.metrics import score_from_order
from mfas.refine import sift
from mfas.refine.insertion import build_sift_edges
from mfas.refine.lns import (
    apply_victim_reinsertions,
    back_edge_weight,
    ils_lns,
)
from mfas.io import GraphData


def _make_small_graph(seed: int = 0, n: int = 30, m: int = 90) -> GraphData:
    """Random directed weighted graph WITH an isolated node and a self-loop."""
    rng = np.random.RandomState(seed)
    src = rng.randint(0, n - 1, size=m).astype(np.int64)
    tgt = rng.randint(0, n - 1, size=m).astype(np.int64)
    w = (rng.randint(1, 10, size=m)).astype(np.int64)
    src[0] = 5
    tgt[0] = 5  # self-loop
    node_ids = np.arange(n, dtype=np.int64)
    return GraphData(src=src, tgt=tgt, weight=w, node_ids=node_ids, name="small_test")


def test_back_edge_weight_matches_bruteforce_small():
    """(1) back_edge_weight == brute-force per-node feedback-weight count."""
    g = _make_small_graph(seed=4)
    n = g.n_nodes
    rng = np.random.RandomState(9)
    rank = rng.permutation(n).astype(np.int64)
    src, tgt, w = build_sift_edges(g)

    back = back_edge_weight(rank, src, tgt, w, n)

    # Brute force: an edge (u,v) is back iff rank[u] > rank[v]; charge both endpoints.
    bf = np.zeros(n, dtype=np.float64)
    for u, v, ww in zip(src, tgt, w):
        if rank[u] > rank[v]:
            bf[u] += ww
            bf[v] += ww
    assert np.allclose(back, bf)
    # Total back-edge weight charged twice == 2 * sum of feedback edge weights.
    fb = float(w[rank[src] > rank[tgt]].sum())
    assert abs(back.sum() - 2.0 * fb) < 1e-9


def test_apply_victim_reinsertions_is_permutation_small():
    """(2) Sequential victim re-insertion yields a valid permutation."""
    g = _make_small_graph(seed=8)
    n = g.n_nodes
    rng = np.random.RandomState(2)
    rank = rng.permutation(n).astype(np.int64)
    src, tgt, w = build_sift_edges(g)

    victims = rng.choice(n, size=5, replace=False)
    new_rank = apply_victim_reinsertions(rank, victims, src, tgt, w, n)
    assert np.array_equal(np.sort(new_rank), np.arange(n))


def test_ils_lns_monotone_ruin_mouse():
    """(3) ils_lns best >= sift best >= initial on mouse (ruin mode, oracle-gated)."""
    g = io.load_dataset("mouse")
    n = g.n_nodes
    rng = np.random.RandomState(123)
    rank0 = rng.permutation(n).astype(np.int64)
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    init_score = score_from_order(rank0, src, tgt, g.weight)

    sift_rank, sift_score, _ = sift(g, rank0, max_sweeps=30)
    assert sift_score >= init_score

    lns_rank, lns_score, rlog = ils_lns(
        g, sift_rank, time_budget_s=2.0, k=8, mode="ruin",
        inner_sweeps=2, sequential_victims=True, seed=123)
    assert lns_score >= sift_score
    assert np.array_equal(np.sort(lns_rank), np.arange(n))
    # The LNS must actually run rounds within the budget.
    assert len(rlog) >= 1


def test_ils_lns_monotone_kick_mouse():
    """(4) ils_lns "kick" (random perturbation) is also monotone on mouse."""
    g = io.load_dataset("mouse")
    n = g.n_nodes
    rng = np.random.RandomState(999)
    rank0 = rng.permutation(n).astype(np.int64)
    sift_rank, sift_score, _ = sift(g, rank0, max_sweeps=30)

    lns_rank, lns_score, _ = ils_lns(
        g, sift_rank, time_budget_s=1.0, k=6, mode="kick",
        inner_sweeps=1, sequential_victims=False, seed=999)
    assert lns_score >= sift_score
    assert np.array_equal(np.sort(lns_rank), np.arange(n))


def test_lns_selectors_are_leakage_safe():
    """(5) Perturb/repair selectors take no GraphData / oracle handle (leakage guard).

    The victim selector and the repair both operate on (rank, src, tgt, w, n) only — plain
    arrays — so a move can never read the discrete oracle or data/best_solution.
    """
    for fn in (back_edge_weight, apply_victim_reinsertions):
        params = set(inspect.signature(fn).parameters)
        assert "rank" in params and "src" in params and "tgt" in params
        # No GraphData / oracle argument is accepted.
        assert "g" not in params and "graph" not in params
        assert "oracle" not in params and "best_solution" not in params


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
