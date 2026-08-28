"""H73: the ``tie_break`` keyword on the exact-gain sift kernel.

Two things must hold, and the first is the important one:

1. **The default is bit-identical to the pre-H73 code.** Every champion through H64 was
   measured with the first-argmax rule, so ``tie_break="first"`` must reproduce it
   exactly -- otherwise adding this keyword silently invalidates ``sota.json``.
2. ``tie_break="mindisp"`` selects a DIFFERENT point of the SAME argmax set: identical
   gain, a maximizing gap, and never farther from the current rank than the default.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfas.io import load_dataset
from mfas.refine.insertion import build_sift_edges, jacobi_best_gaps
from mfas.refine.underrelax import sift_underrelaxed


def _profile_ref(u, rank, src, tgt, w, n):
    """Exact total_u(g) for all gaps, straight from the definition (no shared code)."""
    p = int(rank[u])
    q_of = np.where(rank < p, rank, rank - 1)
    vals = np.full(n, 0.0)
    g = np.arange(n)
    for e in range(src.shape[0]):
        a, b, ww = int(src[e]), int(tgt[e]), float(w[e])
        if a == u:
            vals += ww * (g <= q_of[b])
        elif b == u:
            vals += ww * (g > q_of[a])
    return vals, p


@pytest.fixture(scope="module")
def mouse():
    g = load_dataset("mouse")
    return g, *build_sift_edges(g)


def test_default_is_unchanged(mouse):
    """The keyword's default must reproduce the exact pre-H73 kernel output."""
    g, src, tgt, w = mouse
    n = g.n_nodes
    rng = np.random.default_rng(7)
    for _ in range(5):
        rank = rng.permutation(n).astype(np.int64)
        bg_a, gain_a = jacobi_best_gaps(rank, src, tgt, w, n)
        bg_b, gain_b = jacobi_best_gaps(rank, src, tgt, w, n, tie_break="first")
        assert np.array_equal(bg_a, bg_b)
        assert np.array_equal(gain_a, gain_b)


def test_default_pipeline_is_bit_identical(mouse):
    """A whole stage-3 sift must be bit-identical with the keyword left at its default."""
    g, _, _, _ = mouse
    n = g.n_nodes
    rng = np.random.default_rng(11)
    init = rng.permutation(n).astype(np.int64)
    r_a, s_a, _ = sift_underrelaxed(g, init, k_full=2, alpha=0.7, max_sweeps=8)
    r_b, s_b, _ = sift_underrelaxed(g, init, k_full=2, alpha=0.7, max_sweeps=8,
                                    tie_break="first")
    assert np.array_equal(r_a, r_b)
    assert s_a == s_b


def test_mindisp_has_identical_gain_and_is_never_farther(mouse):
    """mindisp picks another point of the SAME argmax set, never farther from ``rank``."""
    g, src, tgt, w = mouse
    n = g.n_nodes
    rng = np.random.default_rng(13)
    for _ in range(5):
        rank = rng.permutation(n).astype(np.int64)
        bg_f, gain_f = jacobi_best_gaps(rank, src, tgt, w, n)
        bg_m, gain_m = jacobi_best_gaps(rank, src, tgt, w, n, tie_break="mindisp")
        # same exact gain: the rules differ only in WHICH maximizer is returned
        assert np.array_equal(gain_f, gain_m)
        mv = gain_f > 1e-9
        assert mv.any()
        d_f = np.abs(bg_f[mv] - rank[mv])
        d_m = np.abs(bg_m[mv] - rank[mv])
        assert np.all(d_m <= d_f)


def test_mindisp_returns_a_true_maximizer(mouse):
    """Brute-force the profile: the returned gap must attain the profile maximum."""
    g, src, tgt, w = mouse
    n = g.n_nodes
    rng = np.random.default_rng(17)
    rank = rng.permutation(n).astype(np.int64)
    bg_f, gain = jacobi_best_gaps(rank, src, tgt, w, n)
    bg_m, _ = jacobi_best_gaps(rank, src, tgt, w, n, tie_break="mindisp")
    movers = np.flatnonzero(gain > 1e-9)
    for u in rng.choice(movers, size=min(15, movers.shape[0]), replace=False):
        vals, p = _profile_ref(int(u), rank, src, tgt, w, n)
        mx = vals.max()
        assert vals[int(bg_f[u])] == pytest.approx(mx)
        assert vals[int(bg_m[u])] == pytest.approx(mx)
        # mindisp is THE nearest maximizer, not merely a nearer one
        argset = np.flatnonzero(vals == mx)
        assert abs(int(bg_m[u]) - p) == int(np.abs(argset - p).min())
        assert (mx - vals[p]) == pytest.approx(gain[u])


def test_bad_tie_break_rejected(mouse):
    g, src, tgt, w = mouse
    rank = np.arange(g.n_nodes, dtype=np.int64)
    with pytest.raises(ValueError):
        jacobi_best_gaps(rank, src, tgt, w, g.n_nodes, tie_break="nearest-ish")
