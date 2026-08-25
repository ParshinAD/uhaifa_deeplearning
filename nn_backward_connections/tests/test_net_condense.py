"""Correctness tests for the net digraph D+ (H60).

Two claims carry the whole variant, and both are theorems rather than measurements, so
both are asserted here rather than left to a screen to notice:

1. **The reduction is exact.** ``score(order) == C + net_score(order)`` for EVERY order,
   where ``C = sum over unordered pairs of min(w_uv, w_vu)``. If that identity holds, then
   maximising raw feedforward weight and maximising net feedforward weight are the same
   problem, which is the entire justification for substituting the structure graph.
2. **D+ strictly refines G.** ``D+``'s arcs are a subset of ``G``'s, so every ``D+`` SCC is
   contained in a ``G`` SCC. The substitution can therefore only expose decompositions,
   never hide one.

Plus the property that makes the substitution safe to ship: the refiner run on ``D+``
is still monotone non-decreasing in the RAW score.

Everything runs on small random graphs and on mouse (148 nodes). Connectome and MICrONS
are NEVER touched by a unit test.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mfas.io import load_dataset
from mfas.metrics import score_from_order
from mfas.refine.net_condense import build_net_arcs, net_structure_graph
from mfas.refine.scc_recursive import scc_recursive_refine


def _random_digraph(rng, n, m, wmax=10):
    """A random weighted digraph WITH parallel edges, self-loops and reciprocal pairs."""
    s = rng.integers(0, n, size=m)
    t = rng.integers(0, n, size=m)
    w = rng.integers(1, wmax + 1, size=m).astype(np.int64)
    return s.astype(np.int64), t.astype(np.int64), w


def _net_score(order, nsrc, ntgt, nw):
    return float(nw[order[ntgt] > order[nsrc]].sum())


# ──────────────────────────────────────────────────────────────────────────────
# 1. The reduction is exact, for every order
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("trial", range(8))
def test_raw_score_equals_constant_plus_net_score(trial):
    rng = np.random.default_rng(1000 + trial)
    n, m = 24, 160
    s, t, w = _random_digraph(rng, n, m)
    nsrc, ntgt, nw, const_c = build_net_arcs(s, t, w, n)

    # Self-loops never count as feedforward under any ordering, so they are outside the
    # identity; drop them from the raw side exactly as build_net_arcs drops them.
    keep = s != t
    for _ in range(25):
        order = rng.permutation(n).astype(np.int64)
        raw = score_from_order(order, s[keep], t[keep], w[keep])
        assert raw == pytest.approx(const_c + _net_score(order, nsrc, ntgt, nw),
                                    rel=0, abs=1e-9)


def test_reduction_is_exact_on_mouse():
    g = load_dataset("mouse")
    n = g.n_nodes
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    nsrc, ntgt, nw, const_c = build_net_arcs(src, tgt, g.weight, n)
    keep = src != tgt
    rng = np.random.default_rng(7)
    for _ in range(10):
        order = rng.permutation(n).astype(np.int64)
        raw = score_from_order(order, src[keep], tgt[keep], g.weight[keep])
        assert raw == pytest.approx(const_c + _net_score(order, nsrc, ntgt, nw),
                                    rel=0, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Shape of D+: one arc per non-tied pair, heavier direction, positive weight
# ──────────────────────────────────────────────────────────────────────────────
def test_net_arcs_are_positive_simple_and_loopless():
    rng = np.random.default_rng(3)
    n, m = 30, 300
    s, t, w = _random_digraph(rng, n, m)
    nsrc, ntgt, nw, _ = build_net_arcs(s, t, w, n)
    assert np.all(nw > 0), "a net arc with non-positive weight would constrain nothing"
    assert np.all(nsrc != ntgt), "self-loop survived into D+"
    keys = nsrc * n + ntgt
    assert keys.size == np.unique(keys).size, "D+ has parallel arcs"
    rev = ntgt * n + nsrc
    assert not np.intersect1d(keys, rev).size, "D+ contains a reciprocal pair"


def test_tied_reciprocal_pair_produces_no_arc():
    """A pair with equal weights both ways scores the same either way — it must not weld."""
    s = np.array([0, 1], dtype=np.int64)
    t = np.array([1, 0], dtype=np.int64)
    w = np.array([5, 5], dtype=np.int64)
    nsrc, ntgt, nw, const_c = build_net_arcs(s, t, w, 2)
    assert nsrc.size == 0
    assert const_c == 5.0


def test_lopsided_reciprocal_pair_keeps_the_heavier_direction():
    s = np.array([0, 1], dtype=np.int64)
    t = np.array([1, 0], dtype=np.int64)
    w = np.array([100, 1], dtype=np.int64)
    nsrc, ntgt, nw, const_c = build_net_arcs(s, t, w, 2)
    assert (nsrc.tolist(), ntgt.tolist(), nw.tolist()) == ([0], [1], [99.0])
    assert const_c == 1.0


def test_parallel_edges_are_aggregated_before_the_comparison():
    """Two 3-weight edges u->v must beat one 5-weight edge v->u."""
    s = np.array([0, 0, 1], dtype=np.int64)
    t = np.array([1, 1, 0], dtype=np.int64)
    w = np.array([3, 3, 5], dtype=np.int64)
    nsrc, ntgt, nw, const_c = build_net_arcs(s, t, w, 2)
    assert (nsrc.tolist(), ntgt.tolist(), nw.tolist()) == ([0], [1], [1.0])
    assert const_c == 5.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. D+ strictly refines G
# ──────────────────────────────────────────────────────────────────────────────
def _sccs(src, tgt, n):
    mat = coo_matrix((np.ones(src.size, dtype=np.int8), (src, tgt)), shape=(n, n))
    return connected_components(mat, directed=True, connection="strong")


@pytest.mark.parametrize("trial", range(5))
def test_net_components_refine_raw_components(trial):
    rng = np.random.default_rng(50 + trial)
    n, m = 40, 200
    s, t, w = _random_digraph(rng, n, m)
    nsrc, ntgt, nw, _ = build_net_arcs(s, t, w, n)
    keep = s != t
    n_raw, lab_raw = _sccs(s[keep], t[keep], n)
    n_net, lab_net = _sccs(nsrc, ntgt, n)
    assert n_net >= n_raw, "D+ produced FEWER components than G — refinement is violated"
    # every net component is contained in exactly one raw component
    for c in range(n_net):
        members = np.flatnonzero(lab_net == c)
        assert np.unique(lab_raw[members]).size == 1, (
            "a D+ component spans two G components, which is impossible for an arc subset")


def test_net_components_refine_raw_components_on_mouse():
    g = load_dataset("mouse")
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    keep = src != tgt
    nsrc, ntgt, _, _ = build_net_arcs(src, tgt, g.weight, g.n_nodes)
    n_raw, lab_raw = _sccs(src[keep], tgt[keep], g.n_nodes)
    n_net, lab_net = _sccs(nsrc, ntgt, g.n_nodes)
    assert n_net > n_raw, (
        "mouse's net digraph does not refine at all; the mechanism has no footprint here")
    for c in range(n_net):
        members = np.flatnonzero(lab_net == c)
        assert np.unique(lab_raw[members]).size == 1


# ──────────────────────────────────────────────────────────────────────────────
# 4. The refiner stays monotone in the RAW score when driven by D+
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("trial", range(6))
def test_refining_on_the_net_graph_never_lowers_the_raw_score(trial):
    rng = np.random.default_rng(200 + trial)
    n, m = 120, 1400
    s, t, w = _random_digraph(rng, n, m)
    g = type("G", (), {"src": s, "tgt": t, "weight": w, "n_nodes": n})()
    g_net = net_structure_graph(g)

    rank = rng.permutation(n).astype(np.int64)
    prev = score_from_order(rank, s, t, w)
    for r, sf in enumerate((0.5, 0.382, 0.618, 0.25, 0.75)):
        rank = scc_recursive_refine(g_net, rank, min_block=4, split_frac=sf)
        cur = score_from_order(rank, s, t, w)
        assert cur >= prev, f"round {r}: net-driven refinement LOWERED the raw score"
        prev = cur


def test_net_structure_graph_carries_what_the_refiner_reads():
    g = load_dataset("mouse")
    gs = net_structure_graph(g)
    assert gs.n_nodes == g.n_nodes
    assert gs.src.shape == gs.tgt.shape == gs.weight.shape
    assert gs.const_c > 0
    assert gs.name.endswith("-net")
