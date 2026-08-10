"""Unit tests for the recursive SCC-topological block refiner (H36).

The refiner's whole justification is a correctness claim — that the move is monotone
non-decreasing in the exact score *by construction* — so these tests check the claim
itself, not just that the code runs:

* it returns a valid permutation;
* the exact score never decreases, over many random graphs and starting orders;
* after one pass every edge crossing two distinct SCCs of a decomposed block is
  feedforward (the mechanism that makes the move monotone);
* nodes inside one SCC keep their relative order (the other half of the argument);
* the topological sort is a genuine topological order and respects its tie-break;
* the alternation driver never returns worse than its input.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfas.io import GraphData
from mfas.metrics import score_from_order
from mfas.refine import (
    SccRecursiveRefiner,
    alternate_scc_sift,
    scc_recursive_refine,
    topo_order_labels,
)


def _random_graph(n=120, m=600, seed=0, name="synthetic"):
    rng = np.random.RandomState(seed)
    src = rng.randint(0, n, size=m).astype(np.int64)
    tgt = rng.randint(0, n, size=m).astype(np.int64)
    w = rng.randint(1, 50, size=m).astype(np.int64)
    return GraphData(src=src, tgt=tgt, weight=w,
                     node_ids=np.arange(n, dtype=np.int64), name=name)


def _score(g, rank):
    return score_from_order(rank, np.asarray(g.src), np.asarray(g.tgt), g.weight)


# ──────────────────────────────────────────────────────────────────────────────
# Permutation validity + the monotonicity claim
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", range(12))
def test_refine_returns_permutation_and_never_regresses(seed):
    g = _random_graph(n=150, m=900, seed=seed)
    rng = np.random.RandomState(seed + 1000)
    rank = rng.permutation(g.n_nodes).astype(np.int64)
    before = _score(g, rank)

    out = scc_recursive_refine(g, rank, min_block=4,
                               split_frac=[0.5, 0.382, 0.618][seed % 3])

    assert np.array_equal(np.sort(out), np.arange(g.n_nodes)), "not a permutation"
    assert _score(g, out) >= before, "the refiner regressed the exact score"


@pytest.mark.parametrize("seed", range(6))
def test_repeated_passes_are_monotone(seed):
    """Every pass, not just the first, is non-decreasing."""
    g = _random_graph(n=200, m=1600, seed=seed)
    rank = np.random.RandomState(seed).permutation(g.n_nodes).astype(np.int64)
    prev = _score(g, rank)
    for k in range(5):
        rank = scc_recursive_refine(g, rank, min_block=4,
                                    split_frac=[0.5, 0.382, 0.618][k % 3])
        cur = _score(g, rank)
        assert cur >= prev, f"pass {k} regressed: {cur} < {prev}"
        prev = cur


def test_input_rank_is_not_mutated():
    g = _random_graph(n=80, m=400, seed=3)
    rank = np.random.RandomState(0).permutation(g.n_nodes).astype(np.int64)
    keep = rank.copy()
    scc_recursive_refine(g, rank, min_block=4)
    assert np.array_equal(rank, keep), "the refiner mutated its input"


# ──────────────────────────────────────────────────────────────────────────────
# The two halves of the monotonicity argument, checked directly
# ──────────────────────────────────────────────────────────────────────────────
def test_top_level_inter_scc_edges_all_feedforward():
    """After one pass, every edge between two distinct top-level SCCs is forward."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    g = _random_graph(n=140, m=380, seed=7)     # sparse enough to have many SCCs
    n = g.n_nodes
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    n_lab, labels = connected_components(
        coo_matrix((np.ones(src.size, np.int8), (src, tgt)), shape=(n, n)),
        directed=True, connection="strong")
    assert n_lab > 1, "fixture must decompose for this test to mean anything"

    rank = np.random.RandomState(1).permutation(n).astype(np.int64)
    out = scc_recursive_refine(g, rank, min_block=4)

    cross = labels[src] != labels[tgt]
    assert cross.any(), "fixture must have inter-SCC edges"
    assert np.all(out[src[cross]] < out[tgt[cross]]), \
        "an inter-SCC edge is still feedback after a pass"


def test_nodes_within_one_scc_keep_relative_order_when_block_is_topo_sorted():
    """A 2-SCC graph: the SCCs get ordered, their internal order is untouched."""
    # Two triangles A={0,1,2}, B={3,4,5}, with a single A->B bridge, laid out
    # initially with B entirely before A so the bridge starts as feedback.
    src = np.array([0, 1, 2, 3, 4, 5, 0], dtype=np.int64)
    tgt = np.array([1, 2, 0, 4, 5, 3, 3], dtype=np.int64)
    w = np.array([1, 1, 1, 1, 1, 1, 100], dtype=np.int64)
    g = GraphData(src=src, tgt=tgt, weight=w,
                  node_ids=np.arange(6, dtype=np.int64), name="synthetic")
    # order: 3,4,5,0,1,2  -> rank[node] = position
    rank = np.array([3, 4, 5, 0, 1, 2], dtype=np.int64)

    out = scc_recursive_refine(g, rank, min_block=1)

    assert out[0] < out[3], "the A->B bridge should have become feedforward"
    # relative order inside each SCC is preserved
    assert (out[0] < out[1]) and (out[1] < out[2])
    assert (out[3] < out[4]) and (out[4] < out[5])
    assert _score(g, out) > _score(g, rank)


# ──────────────────────────────────────────────────────────────────────────────
# Topological sort
# ──────────────────────────────────────────────────────────────────────────────
def test_topo_order_is_topological_and_respects_tie_break():
    # DAG: 0->2, 1->2. Labels 0 and 1 are incomparable; the key decides their order.
    ls = np.array([0, 1], dtype=np.int64)
    lt = np.array([2, 2], dtype=np.int64)
    out = topo_order_labels(3, ls, lt, np.array([5, 1, 0], dtype=np.int64))
    pos = {int(l): i for i, l in enumerate(out)}
    assert pos[0] < pos[2] and pos[1] < pos[2], "not a topological order"
    assert pos[1] < pos[0], "tie-break key ignored (label 1 has the smaller key)"


def test_topo_order_rejects_a_cycle():
    ls = np.array([0, 1], dtype=np.int64)
    lt = np.array([1, 0], dtype=np.int64)
    with pytest.raises(RuntimeError):
        topo_order_labels(2, ls, lt, np.zeros(2, dtype=np.int64))


def test_topo_order_handles_a_dag_with_no_edges():
    out = topo_order_labels(3, np.empty(0, np.int64), np.empty(0, np.int64),
                            np.array([2, 0, 1], dtype=np.int64))
    assert list(out) == [1, 2, 0], "with no constraints the key alone should order"


# ──────────────────────────────────────────────────────────────────────────────
# Refiner bookkeeping + the alternation driver
# ──────────────────────────────────────────────────────────────────────────────
def test_min_block_is_respected():
    """With min_block >= n the refiner is a no-op."""
    g = _random_graph(n=60, m=300, seed=5)
    rank = np.random.RandomState(2).permutation(g.n_nodes).astype(np.int64)
    ref = SccRecursiveRefiner(np.asarray(g.src), np.asarray(g.tgt), g.n_nodes,
                              min_block=g.n_nodes)
    out = ref.run(rank)
    assert np.array_equal(out, rank)
    assert ref.n_scc_calls == 0


def test_self_loops_are_ignored():
    g = _random_graph(n=60, m=200, seed=11)
    src = np.concatenate([np.asarray(g.src), np.arange(60, dtype=np.int64)])
    tgt = np.concatenate([np.asarray(g.tgt), np.arange(60, dtype=np.int64)])
    w = np.concatenate([g.weight, np.full(60, 999, dtype=g.weight.dtype)])
    gl = GraphData(src=src, tgt=tgt, weight=w,
                   node_ids=np.arange(60, dtype=np.int64), name="synthetic")
    rank = np.random.RandomState(4).permutation(60).astype(np.int64)
    out = scc_recursive_refine(gl, rank, min_block=4)
    assert np.array_equal(np.sort(out), np.arange(60))
    assert _score(gl, out) >= _score(gl, rank)


@pytest.mark.parametrize("seed", range(4))
def test_alternation_never_regresses_and_logs_each_cycle(seed):
    g = _random_graph(n=180, m=1400, seed=seed + 20)
    rank = np.random.RandomState(seed).permutation(g.n_nodes).astype(np.int64)
    before = _score(g, rank)

    best_rank, best_score, log = alternate_scc_sift(
        g, rank, n_cycles=3, sift_sweeps=4, min_block=4)

    assert np.array_equal(np.sort(best_rank), np.arange(g.n_nodes))
    assert best_score >= before
    assert _score(g, best_rank) == best_score, "returned score is not the returned order's"
    assert len(log) == 3
    assert all("after_scc_pct" in row and "after_sift_pct" in row for row in log)


def test_alternation_zero_cycles_is_identity():
    g = _random_graph(n=50, m=200, seed=9)
    rank = np.random.RandomState(1).permutation(50).astype(np.int64)
    best_rank, best_score, log = alternate_scc_sift(g, rank, n_cycles=0)
    assert np.array_equal(best_rank, rank)
    assert best_score == _score(g, rank)
    assert log == []
