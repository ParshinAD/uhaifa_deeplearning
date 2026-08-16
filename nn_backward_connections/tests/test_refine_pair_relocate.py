"""Correctness tests for the sequential pair-relocation refiner.

The campaign's accounting discipline is **predicted gain == oracle-realised delta**. This move
class earned that scrutiny the hard way: the first prototype of the same closed form matched the
frozen oracle on 1 of 18 moves, because it aggregated contributions per EDGE instead of per
POSITION and so evaluated splits that cannot be realised. The closed form itself was never wrong
— a brute-force sweep over every split reproduced it with zero mismatches. These tests pin both
halves so that distinction cannot be lost again.

Everything runs on mouse (148 nodes) and on small synthetic graphs, on the CPU, in seconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfas.io import GraphData, load_dataset
from mfas.metrics import score_from_order
from mfas.refine.pair_relocate import (build_direction_csr, pair_move_gain,
                                       pair_relocate)


@pytest.fixture(scope="module")
def mouse():
    return load_dataset("mouse")


def _csr(g):
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    keep = src != tgt
    n = g.n_nodes
    return (build_direction_csr(src[keep], tgt[keep], w[keep], n),
            build_direction_csr(tgt[keep], src[keep], w[keep], n))


def _apply(seq, rank, u, v, cut, lo, hi):
    block = [x for x in seq[lo:hi + 1] if x != u and x != v]
    out, placed = [], False
    for x in block:
        out.append(x)
        if rank[x] == cut:
            out.extend((u, v))
            placed = True
    if not placed:
        out = [u, v] + out
    new_seq = seq.copy()
    new_seq[lo:hi + 1] = np.asarray(out, dtype=np.int64)
    new_rank = np.empty(seq.shape[0], dtype=np.int64)
    new_rank[new_seq] = np.arange(seq.shape[0], dtype=np.int64)
    return new_rank


# ── the closed form equals the oracle, move by move ──────────────────────────────────
def test_predicted_gain_equals_oracle_on_mouse(mouse):
    """Every candidate pair's predicted gain must equal the realised delta exactly."""
    g = mouse
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    n = g.n_nodes
    (out_ptr, out_idx, out_w), (in_ptr, in_idx, in_w) = _csr(g)

    rank = np.arange(n, dtype=np.int64)           # a deliberately bad order: plenty of backward
    seq = np.empty(n, dtype=np.int64)
    seq[rank] = np.arange(n, dtype=np.int64)
    base = score_from_order(rank, src, tgt, g.weight)

    back = np.flatnonzero(rank[np.asarray(src)] > rank[np.asarray(tgt)])
    checked = 0
    for ei in back[:200]:
        u, v = int(src[ei]), int(tgt[ei])
        gain, cut, lo, hi = pair_move_gain(rank, u, v, out_ptr, out_idx, out_w,
                                           in_ptr, in_idx, in_w)
        new_rank = _apply(seq, rank, u, v, cut, lo, hi)
        realised = score_from_order(new_rank, src, tgt, g.weight) - base
        assert realised == pytest.approx(gain, abs=1e-9), (
            f"pair ({u},{v}): predicted {gain!r}, realised {realised!r}")
        checked += 1
    assert checked > 20, "fixture produced too few backward edges to be a real test"


def test_gain_matches_brute_force_over_every_split():
    """The closed form must agree with brute force at EVERY split, not just the argmax.

    This is the assertion that separates 'the formula is right' from 'the scan is right' — the
    distinction that took a 1-of-18 failure to notice.
    """
    rng = np.random.default_rng(0)
    n, m = 40, 260
    s = rng.integers(0, n, m)
    t = rng.integers(0, n, m)
    keep = s != t
    s, t = s[keep], t[keep]
    w = rng.integers(1, 9, s.shape[0]).astype(np.float64)
    # GraphData derives n_nodes from node_ids; it has no n_nodes field of its own.
    g = GraphData(src=s.astype(np.int64), tgt=t.astype(np.int64), weight=w,
                  node_ids=np.arange(n), name="synth")
    (out_ptr, out_idx, out_w), (in_ptr, in_idx, in_w) = _csr(g)

    rank = np.arange(n, dtype=np.int64)
    seq = np.empty(n, dtype=np.int64)
    seq[rank] = np.arange(n, dtype=np.int64)
    base = score_from_order(rank, s, t, w)

    back = np.flatnonzero(rank[s] > rank[t])
    assert back.size > 0
    tested = 0
    for ei in back[:15]:
        u, v = int(s[ei]), int(t[ei])
        lo, hi = int(rank[v]), int(rank[u])
        if hi - lo < 2:
            continue
        gain, cut, _, _ = pair_move_gain(rank, u, v, out_ptr, out_idx, out_w,
                                         in_ptr, in_idx, in_w)
        block = [int(x) for x in seq[lo:hi + 1] if x != u and x != v]
        best = None
        for r in range(len(block) + 1):
            cand = block[:r] + [u, v] + block[r:]
            ns = seq.copy()
            ns[lo:hi + 1] = np.asarray(cand, dtype=np.int64)
            nr = np.empty(n, dtype=np.int64)
            nr[ns] = np.arange(n, dtype=np.int64)
            d = score_from_order(nr, s, t, w) - base
            best = d if best is None else max(best, d)
        assert gain == pytest.approx(best, abs=1e-9)
        tested += 1
    assert tested > 3


# ── the driver ───────────────────────────────────────────────────────────────────────
def test_pair_relocate_is_monotone(mouse):
    """Only strictly-positive exact gains are applied, so the score can never fall."""
    g = mouse
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    rank0 = np.arange(g.n_nodes, dtype=np.int64)
    before = score_from_order(rank0, src, tgt, g.weight)
    _, after, log = pair_relocate(g, rank0, n_passes=3)
    assert after >= before
    assert log and sum(r["n_applied"] for r in log) > 0


def test_pass_log_predicted_matches_realised(mouse):
    """Per pass, the summed predicted gain must equal the measured pass delta."""
    g = mouse
    rank0 = np.arange(g.n_nodes, dtype=np.int64)
    _, _, log = pair_relocate(g, rank0, n_passes=3)
    for row in log:
        if row["n_applied"]:
            assert row["gain_pp"] == pytest.approx(row["predicted_gain_pp"], abs=1e-9)


def test_pop_budget_is_respected_and_deterministic(mouse):
    """The budget is a POP count, not a clock — so two runs must be bit-identical."""
    g = mouse
    rank0 = np.arange(g.n_nodes, dtype=np.int64)
    a_rank, a_score, a_log = pair_relocate(g, rank0, n_passes=3, max_pops=50)
    b_rank, b_score, b_log = pair_relocate(g, rank0, n_passes=3, max_pops=50)
    assert np.array_equal(a_rank, b_rank)
    assert a_score == b_score
    assert a_log[-1]["n_popped"] <= 50


def test_returns_input_when_no_improving_move_exists(mouse):
    """A converged order must come back unchanged, not merely equal-scoring."""
    g = mouse
    rank0 = np.arange(g.n_nodes, dtype=np.int64)
    r1, s1, _ = pair_relocate(g, rank0, n_passes=6)
    r2, s2, log2 = pair_relocate(g, r1, n_passes=6)
    assert s2 == s1
    assert np.array_equal(r2, r1)
    assert sum(row["n_applied"] for row in log2) == 0
