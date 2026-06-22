"""Unit tests for the under-relaxed two-phase sift refiner (mfas.refine.underrelax).

Covers:
  (1) at alpha=1.0 sift_underrelaxed reproduces the production Jacobi sift EXACTLY
      (same best_score) on mouse and on a small random graph -> the win is purely the
      alpha<1 dynamics, not a kernel change;
  (2) sift_underrelaxed best_score >= initial score on mouse (monotone, best-by-oracle);
  (3) score_from_positions(best_rank.astype(float32)) == score_from_order(best_rank) on
      mouse (the runner's re-score assertion holds for the returned positions).
"""
from __future__ import annotations

import numpy as np
import pytest

from mfas import io
from mfas.metrics import score_from_order, score_from_positions
from mfas.refine import sift
from mfas.refine.underrelax import sift_underrelaxed
from mfas.io import GraphData


def _make_small_graph(seed: int = 0, n: int = 30, m: int = 90) -> GraphData:
    """Random directed weighted graph WITH an isolated node and a self-loop."""
    rng = np.random.RandomState(seed)
    src = rng.randint(0, n - 1, size=m).astype(np.int64)
    tgt = rng.randint(0, n - 1, size=m).astype(np.int64)
    w = (rng.randint(1, 10, size=m)).astype(np.int64)
    src[0] = 5
    tgt[0] = 5
    node_ids = np.arange(n, dtype=np.int64)
    return GraphData(src=src, tgt=tgt, weight=w, node_ids=node_ids, name="small_test")


def test_alpha1_matches_jacobi_sift_small():
    """(1a) alpha=1.0 reproduces insertion.sift exactly on a small graph."""
    g = _make_small_graph(seed=3)
    n = g.n_nodes
    rng = np.random.RandomState(7)
    rank0 = rng.permutation(n).astype(np.int64)

    _, s_jac, _ = sift(g, rank0, max_sweeps=20)
    _, s_ur, _ = sift_underrelaxed(g, rank0, k_full=6, alpha=1.0, max_sweeps=20)
    assert s_ur == s_jac


def test_alpha1_matches_jacobi_sift_mouse():
    """(1b) alpha=1.0 reproduces insertion.sift exactly on mouse (any k_full)."""
    g = io.load_dataset("mouse")
    n = g.n_nodes
    rng = np.random.RandomState(123)
    rank0 = rng.permutation(n).astype(np.int64)

    _, s_jac, _ = sift(g, rank0, max_sweeps=40)
    for k_full in (0, 6, 40):
        _, s_ur, _ = sift_underrelaxed(g, rank0, k_full=k_full, alpha=1.0, max_sweeps=40)
        assert s_ur == s_jac, f"k_full={k_full} diverged from Jacobi"


def test_underrelaxed_monotone_mouse():
    """(2) sift_underrelaxed best_score >= initial score on mouse (best-by-oracle)."""
    g = io.load_dataset("mouse")
    n = g.n_nodes
    rng = np.random.RandomState(321)
    rank0 = rng.permutation(n).astype(np.int64)
    init_score = score_from_order(rank0, np.asarray(g.src), np.asarray(g.tgt), g.weight)

    best_rank, best_score, sweep_log = sift_underrelaxed(
        g, rank0, k_full=6, alpha=0.7, max_sweeps=40)
    assert best_score >= init_score
    assert len(sweep_log) >= 1
    # the two-phase schedule must never beat plain Jacobi by being worse than it:
    _, s_jac, _ = sift(g, rank0, max_sweeps=40)
    assert best_score >= s_jac


def test_float32_rank_oracle_exact_mouse():
    """(3) score_from_positions(rank.float32) == score_from_order(rank) on mouse."""
    g = io.load_dataset("mouse")
    n = g.n_nodes
    rng = np.random.RandomState(321)
    rank0 = rng.permutation(n).astype(np.int64)
    best_rank, best_score, _ = sift_underrelaxed(g, rank0, k_full=6, alpha=0.7, max_sweeps=40)

    s_order = score_from_order(best_rank, np.asarray(g.src), np.asarray(g.tgt), g.weight)
    s_pos = score_from_positions(best_rank.astype(np.float32),
                                 np.asarray(g.src), np.asarray(g.tgt), g.weight)
    assert s_pos == s_order == best_score


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
