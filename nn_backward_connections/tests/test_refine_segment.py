"""Unit tests for the exact-gain bounded-span segment refiner (H41).

The whole hypothesis rests on ONE correctness claim: the closed-form gain of a rigid
segment move equals the change the frozen oracle measures. Everything else the module
promises (monotonicity, safe alternation with stages 3 and 4) is a corollary of it. So
the centrepiece here is an exhaustive brute-force cross-check of

    segment_gains(...)[i]  ==  score(after move) - score(before move)

with ``score`` being :func:`mfas.metrics.score_from_order` — the frozen scorer, not a
re-implementation — over many (segment, target) pairs, several weight regimes (unit,
integer, float, ZERO-weight edges, heavy duplicates), several graph sizes and several
starting orders.

The remaining tests cover: sweep-level composition (disjoint moves compose exactly),
monotonicity, determinism, and the edge cases (L=1 reduces to a single-node move,
segment at either end, target == current position, L >= n).
"""
from __future__ import annotations

import numpy as np
import pytest

from mfas.io import GraphData
from mfas.metrics import score_from_order
from mfas.refine.segment import (
    DEFAULT_OFFSETS,
    DEFAULT_SEG_LENGTHS,
    apply_segment_move_ref,
    apply_segment_moves,
    best_segment_moves,
    build_span_index,
    segment_gains,
    segment_move_gain_ref,
    segment_refine,
    segment_sweep,
    select_disjoint_moves,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: several weight regimes, all small
# ──────────────────────────────────────────────────────────────────────────────
def _graph(n, m, seed, regime="int"):
    """Small random directed graph in one of several weight regimes."""
    rng = np.random.RandomState(seed)
    src = rng.randint(0, n, size=m).astype(np.int64)
    tgt = rng.randint(0, n, size=m).astype(np.int64)
    if regime == "unit":
        w = np.ones(m, dtype=np.int64)
    elif regime == "int":
        w = rng.randint(1, 50, size=m).astype(np.int64)
    elif regime == "zero":
        # a third of the edges carry weight 0 (legal, and a classic sign-bug trap)
        w = rng.randint(0, 3, size=m).astype(np.int64) * rng.randint(0, 20, size=m)
        w = w.astype(np.int64)
    elif regime == "dup":
        # only two distinct weights -> massive ties everywhere
        w = np.where(rng.rand(m) < 0.5, 7, 7000).astype(np.int64)
    elif regime == "float":
        w = rng.rand(m).astype(np.float64) * 3.5
    elif regime == "selfloop":
        w = rng.randint(1, 20, size=m).astype(np.int64)
        src[: m // 5] = tgt[: m // 5]          # a fifth are self-loops
    else:                                       # pragma: no cover
        raise ValueError(regime)
    return GraphData(src=src, tgt=tgt, weight=w,
                     node_ids=np.arange(n, dtype=np.int64), name=f"t_{regime}")


def _score(g, rank):
    return score_from_order(rank, np.asarray(g.src), np.asarray(g.tgt), g.weight)


REGIMES = ("unit", "int", "zero", "dup", "float", "selfloop")


# ──────────────────────────────────────────────────────────────────────────────
# THE test: exact gain == oracle delta, over many (segment, target) pairs
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("n,seed", [(8, 0), (17, 1), (40, 2), (60, 3)])
def test_exact_gain_matches_frozen_scorer_brute_force(regime, n, seed, request):
    """``segment_gains[i]`` == oracle(after) - oracle(before), for EVERY (i, L, d)."""
    g = _graph(n, m=max(4 * n, 24), seed=seed, regime=regime)
    checked = 0
    for rseed in range(3):
        rank = np.random.RandomState(1000 * seed + rseed).permutation(n).astype(np.int64)
        before = _score(g, rank)
        idx = build_span_index(rank, np.asarray(g.src), np.asarray(g.tgt), g.weight,
                               n, max_span=n)
        # full ladder plus a few off-ladder sizes, so the kernel is not only exercised
        # on powers of two.
        sizes = sorted({1, 2, 3, 4, 5, 7, 8, 11, 16, 23, 32, n - 1, n} & set(range(1, n + 1)))
        for L in sizes:
            for d in sizes:
                if L + d > n:
                    continue
                gains = segment_gains(idx, n, L, d)
                assert gains.shape[0] == n - L - d + 1
                for i in range(gains.shape[0]):
                    after_rank = apply_segment_move_ref(rank, i, L, d)
                    assert np.array_equal(np.sort(after_rank), np.arange(n))
                    truth = _score(g, after_rank) - before
                    assert gains[i] == pytest.approx(truth, rel=0, abs=1e-9), (
                        f"regime={regime} n={n} rank_seed={rseed} L={L} d={d} i={i}: "
                        f"kernel {gains[i]!r} != oracle delta {truth!r}")
                    # and the independent enumeration reference agrees too
                    assert segment_move_gain_ref(g, rank, i, L, d) == \
                        pytest.approx(truth, rel=0, abs=1e-9)
                    checked += 1
    # make the coverage count visible in -s runs / on failure
    request.node.user_properties.append(("pairs_checked", checked))
    assert checked > 200


def test_brute_force_coverage_is_large():
    """Report the total number of (segment, target) pairs the brute force covers.

    Recomputed independently of the parametrised test so the number quoted in the
    write-up is itself asserted, not hand-counted.
    """
    total = 0
    for n in (8, 17, 40, 60):
        sizes = sorted({1, 2, 3, 4, 5, 7, 8, 11, 16, 23, 32, n - 1, n} & set(range(1, n + 1)))
        per_rank = sum(n - L - d + 1 for L in sizes for d in sizes if L + d <= n)
        total += per_rank * 3 * len(REGIMES)      # 3 ranks x every weight regime
    assert total > 100_000, total


# ──────────────────────────────────────────────────────────────────────────────
# Sweep level: disjoint moves compose exactly
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("seed", range(4))
def test_sweep_predicted_gain_equals_realised_delta(regime, seed):
    """The sum of the accepted gains IS the realised oracle delta (disjointness proof)."""
    n = 120
    g = _graph(n, m=700, seed=seed, regime=regime)
    rank = np.random.RandomState(seed + 77).permutation(n).astype(np.int64)
    before = _score(g, rank)
    new_rank, info = segment_sweep(g, rank, seg_lengths=(1, 2, 3, 4, 8, 16, 32),
                                   offsets=(1, 2, 3, 4, 8, 16, 32))
    assert np.array_equal(np.sort(new_rank), np.arange(n)), "not a permutation"
    realised = _score(g, new_rank) - before
    assert info["predicted_gain"] == pytest.approx(realised, rel=0, abs=1e-9)
    if info["n_moves"] > 0:
        assert realised > 0


def test_selected_move_windows_are_disjoint():
    n = 200
    g = _graph(n, m=1400, seed=5, regime="int")
    rank = np.random.RandomState(5).permutation(n).astype(np.int64)
    idx = build_span_index(rank, np.asarray(g.src), np.asarray(g.tgt), g.weight,
                           n, max_span=64)
    bg, bL, bd = best_segment_moves(idx, n, (1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32))
    moves = select_disjoint_moves(bg, bL, bd, n)
    assert moves, "expected at least one improving move on a random order"
    covered = np.zeros(n, dtype=bool)
    for (i, L, d, gain) in moves:
        assert gain > 0
        assert not covered[i:i + L + d].any(), "overlapping windows selected"
        covered[i:i + L + d] = True


# ──────────────────────────────────────────────────────────────────────────────
# Monotonicity
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("seed", range(6))
def test_sweeps_never_decrease_the_score(regime, seed):
    n = 150
    g = _graph(n, m=900, seed=seed, regime=regime)
    rank = np.random.RandomState(seed + 4242).permutation(n).astype(np.int64)
    prev = _score(g, rank)
    for _ in range(5):
        rank, _info = segment_sweep(g, rank, seg_lengths=(1, 2, 4, 8, 16, 32),
                                    offsets=(1, 2, 4, 8, 16, 32))
        cur = _score(g, rank)
        assert cur >= prev, f"segment sweep regressed: {cur} < {prev}"
        prev = cur


@pytest.mark.parametrize("seed", range(6))
def test_driver_never_returns_worse_than_input(seed):
    g = _graph(150, m=900, seed=seed, regime="int")
    rank = np.random.RandomState(seed).permutation(150).astype(np.int64)
    before = _score(g, rank)
    out, score, log = segment_refine(g, rank, max_sweeps=6,
                                     seg_lengths=(1, 2, 4, 8, 16, 32),
                                     offsets=(1, 2, 4, 8, 16, 32))
    assert np.array_equal(np.sort(out), np.arange(150))
    assert score >= before
    assert score == pytest.approx(_score(g, out), rel=0, abs=1e-9)
    assert all(row["predicted_gain"] >= 0 for row in log)


# ──────────────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("regime", REGIMES)
def test_two_runs_are_bit_identical(regime):
    g = _graph(180, m=1200, seed=11, regime=regime)
    rank = np.random.RandomState(11).permutation(180).astype(np.int64)
    a, sa, la = segment_refine(g, rank, max_sweeps=4)
    b, sb, lb = segment_refine(g, rank, max_sweeps=4)
    assert np.array_equal(a, b), "segment_refine is not deterministic"
    assert sa == sb
    assert [r["n_moves"] for r in la] == [r["n_moves"] for r in lb]
    assert [r["predicted_gain"] for r in la] == [r["predicted_gain"] for r in lb]


def test_input_rank_is_not_mutated():
    g = _graph(100, m=600, seed=3, regime="int")
    rank = np.random.RandomState(3).permutation(100).astype(np.int64)
    keep = rank.copy()
    segment_refine(g, rank, max_sweeps=3)
    segment_sweep(g, rank)
    assert np.array_equal(rank, keep)


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("d", [1, 2, 5, 13])
def test_L1_reduces_to_a_single_node_move(d):
    """L=1 must be exactly 'take the node at position i and put it d places later'."""
    n = 40
    g = _graph(n, m=260, seed=9, regime="int")
    rank = np.random.RandomState(9).permutation(n).astype(np.int64)
    idx = build_span_index(rank, np.asarray(g.src), np.asarray(g.tgt), g.weight,
                           n, max_span=n)
    gains = segment_gains(idx, n, 1, d)
    seq = np.empty(n, dtype=np.int64)
    seq[rank] = np.arange(n)
    for i in range(gains.shape[0]):
        # explicit single-node relocation, built by list surgery (no module code)
        lst = list(seq)
        node = lst.pop(i)
        lst.insert(i + d, node)
        moved = np.empty(n, dtype=np.int64)
        moved[np.asarray(lst, dtype=np.int64)] = np.arange(n)
        assert np.array_equal(moved, apply_segment_move_ref(rank, i, 1, d))
        assert gains[i] == pytest.approx(_score(g, moved) - _score(g, rank),
                                         rel=0, abs=1e-9)


def test_segment_at_either_end_of_the_order():
    n = 30
    g = _graph(n, m=200, seed=13, regime="int")
    rank = np.random.RandomState(13).permutation(n).astype(np.int64)
    before = _score(g, rank)
    idx = build_span_index(rank, np.asarray(g.src), np.asarray(g.tgt), g.weight,
                           n, max_span=n)
    for L, d in [(1, 1), (4, 3), (7, 8), (1, n - 1), (n - 1, 1)]:
        gains = segment_gains(idx, n, L, d)
        for i in (0, gains.shape[0] - 1):                # first and last window
            after = apply_segment_move_ref(rank, i, L, d)
            assert gains[i] == pytest.approx(_score(g, after) - before, rel=0, abs=1e-9)
    # the extreme window covers the whole line
    full = segment_gains(idx, n, n // 2, n - n // 2)
    assert full.shape[0] == 1


def test_target_equals_current_position_is_a_zero_gain_no_op():
    n = 25
    g = _graph(n, m=150, seed=17, regime="int")
    rank = np.random.RandomState(17).permutation(n).astype(np.int64)
    idx = build_span_index(rank, np.asarray(g.src), np.asarray(g.tgt), g.weight,
                           n, max_span=n)
    for L in (1, 3, 8):
        gains = segment_gains(idx, n, L, 0)             # d = 0 => stay put
        assert gains.shape[0] == n - L + 1
        assert np.all(gains == 0.0)
        assert np.array_equal(apply_segment_move_ref(rank, 3, L, 0), rank)
    # and a zero offset is never selected as a move
    _new, info = segment_sweep(g, rank, seg_lengths=(2, 4), offsets=(0,))
    assert info["n_moves"] == 0


@pytest.mark.parametrize("L", [30, 31, 60])
def test_L_at_or_beyond_n_yields_no_candidates(L):
    n = 30
    g = _graph(n, m=180, seed=19, regime="int")
    rank = np.random.RandomState(19).permutation(n).astype(np.int64)
    idx = build_span_index(rank, np.asarray(g.src), np.asarray(g.tgt), g.weight,
                           n, max_span=n)
    assert segment_gains(idx, n, L, 1).shape[0] == 0     # L + d > n
    out, info = segment_sweep(g, rank, seg_lengths=(L,), offsets=(1, 2))
    assert info["n_moves"] == 0
    assert np.array_equal(out, rank)


def test_tiny_and_degenerate_graphs():
    for n in (1, 2, 3):
        g = GraphData(src=np.array([0], dtype=np.int64), tgt=np.array([0], dtype=np.int64),
                      weight=np.array([3], dtype=np.int64),
                      node_ids=np.arange(n, dtype=np.int64), name="tiny")
        rank = np.arange(n, dtype=np.int64)
        out, info = segment_sweep(g, rank)
        assert np.array_equal(out, rank)
        assert info["n_moves"] == 0
    # empty edge set
    g = GraphData(src=np.zeros(0, dtype=np.int64), tgt=np.zeros(0, dtype=np.int64),
                  weight=np.zeros(0, dtype=np.int64),
                  node_ids=np.arange(10, dtype=np.int64), name="noedge")
    rank = np.arange(10, dtype=np.int64)
    out, info = segment_sweep(g, rank)
    assert np.array_equal(out, rank) and info["n_moves"] == 0


def test_default_ladders_are_geometric_and_bounded():
    assert DEFAULT_SEG_LENGTHS == DEFAULT_OFFSETS
    assert list(DEFAULT_SEG_LENGTHS) == [2 ** k for k in range(11)]
    assert max(DEFAULT_SEG_LENGTHS) + max(DEFAULT_OFFSETS) == 2048


def test_apply_segment_moves_handles_multiple_disjoint_moves():
    rank = np.arange(20, dtype=np.int64)
    out = apply_segment_moves(rank, [(0, 2, 3, 0.0), (10, 1, 4, 0.0)])
    seq = np.empty(20, dtype=np.int64)
    seq[out] = np.arange(20)
    assert list(seq[:5]) == [2, 3, 4, 0, 1]
    assert list(seq[10:15]) == [11, 12, 13, 14, 10]
    assert list(seq[5:10]) == [5, 6, 7, 8, 9]
