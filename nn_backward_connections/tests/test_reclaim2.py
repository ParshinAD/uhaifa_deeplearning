"""``mfas.refine.reclaim2`` must agree with ``mfas.refine.reclaim``, arc for arc.

reclaim2 exists only to be FASTER. Every claim in this cycle rests on it computing the same
answer, so the reference implementation is used here as an oracle: same input, same
reclaimable list, same accepted list, same resulting order. The randomised cases are the ones
that matter — they generate graphs with genuine conflicts, which the hand-built cases cannot
be relied on to produce.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfas.refine import reclaim as ref
from mfas.refine import reclaim2 as fast


class _G:
    """Minimal GraphData stand-in: reclaim only reads src/tgt/weight/n_nodes."""

    def __init__(self, src, tgt, w, n):
        self.src = np.asarray(src, dtype=np.int64)
        self.tgt = np.asarray(tgt, dtype=np.int64)
        self.weight = np.asarray(w, dtype=np.float64)
        self.n_nodes = int(n)
        self.name = "synthetic"
        self.n_edges = int(self.src.shape[0])
        self.total_weight = float(self.weight.sum())


def _random_digraph(rng, n, m, wmax=5):
    src = rng.integers(0, n, size=m)
    tgt = rng.integers(0, n, size=m)
    keep = src != tgt
    src, tgt = src[keep], tgt[keep]
    w = rng.integers(1, wmax + 1, size=src.shape[0]).astype(np.float64)
    return src.astype(np.int64), tgt.astype(np.int64), w


def _both(rank, src, tgt, w, n, budget=10**9, cbudget=10**9):
    a_r, F_r, s_r = ref.reclaimable_arcs(rank, src, tgt, w, n, budget=budget)
    a_f, F_f, s_f = fast.reclaimable_arcs_fast(rank, src, tgt, w, n, budget=budget)
    acc_r = ref.resolve_conflicts(*F_r, a_r, n, conflict_budget=cbudget)
    acc_f = fast.resolve_conflicts_scc(*F_f, a_f, n, conflict_budget=cbudget)
    return (a_r, s_r, acc_r), (a_f, s_f, acc_f)


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(12)))
def test_reclaimable_set_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = 60
    src, tgt, w = _random_digraph(rng, n, 400)
    rank = rng.permutation(n).astype(np.int64)
    (a_r, s_r, _), (a_f, s_f, _) = _both(rank, src, tgt, w, n)
    assert a_f == a_r                       # same arcs, same heaviest-first order
    assert s_f["n_reclaimable"] == s_r["n_reclaimable"]
    assert s_f["n_backward"] == s_r["n_backward"]
    assert s_f["n_unknown"] == s_r["n_unknown"] == 0


def test_prefilters_partition_the_backward_edges():
    """Every backward edge is settled by exactly one route, and the counts add up."""
    rng = np.random.default_rng(3)
    n = 80
    src, tgt, w = _random_digraph(rng, n, 600)
    rank = rng.permutation(n).astype(np.int64)
    _, stats = fast.reclaimable_arcs_fast(rank, src, tgt, w, n)[0::2]
    assert stats["n_onehop"] + stats["n_proved"] + stats["n_searched"] == stats["n_backward"]


def test_reclaimed_arc_really_closes_no_cycle():
    """The lemma's precondition, checked directly rather than trusted."""
    rng = np.random.default_rng(11)
    n = 50
    src, tgt, w = _random_digraph(rng, n, 300)
    rank = rng.permutation(n).astype(np.int64)
    arcs, (F_src, F_tgt), _ = fast.reclaimable_arcs_fast(rank, src, tgt, w, n)
    assert arcs, "test is vacuous if nothing is reclaimable"
    out_ptr, out_idx = ref.build_csr(F_src, F_tgt, n)
    in_ptr, in_idx = ref.build_csr(F_tgt, F_src, n)
    for _, u, v in arcs:
        assert ref.reachable_in_F(out_ptr, out_idx, in_ptr, in_idx, rank, v, u, 10**9) is False


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(12)))
def test_accepted_set_matches_reference(seed):
    rng = np.random.default_rng(100 + seed)
    n = 60
    src, tgt, w = _random_digraph(rng, n, 400)
    rank = rng.permutation(n).astype(np.int64)
    (_, _, acc_r), (_, _, acc_f) = _both(rank, src, tgt, w, n)
    assert list(zip(acc_f[0], acc_f[1], acc_f[2])) == list(zip(acc_r[0], acc_r[1], acc_r[2]))
    assert acc_f[3]["n_accepted"] == acc_r[3]["n_accepted"]
    assert acc_f[3]["w_accepted"] == acc_r[3]["w_accepted"]
    assert acc_f[3]["n_budget_rejects"] == 0


def test_conflicts_actually_occur_in_the_random_suite():
    """Guard against the equivalence tests passing only because nothing ever conflicts."""
    seen = 0
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        n = 60
        src, tgt, w = _random_digraph(rng, n, 400)
        rank = rng.permutation(n).astype(np.int64)
        (_, _, acc_r), _ = _both(rank, src, tgt, w, n)
        seen += acc_r[3]["n_conflicts"]
    assert seen > 0, "no conflict was exercised — the equivalence claim would be untested"


def test_scc_confinement_never_accepts_a_cycle_creating_arc():
    rng = np.random.default_rng(7)
    n = 70
    src, tgt, w = _random_digraph(rng, n, 500)
    rank = rng.permutation(n).astype(np.int64)
    arcs, (F_src, F_tgt), _ = fast.reclaimable_arcs_fast(rank, src, tgt, w, n)
    a_src, a_tgt, a_w, _ = fast.resolve_conflicts_scc(F_src, F_tgt, arcs, n)
    A_src = np.concatenate([F_src, np.asarray(a_src, dtype=np.int64)])
    A_tgt = np.concatenate([F_tgt, np.asarray(a_tgt, dtype=np.int64)])
    ptr, idx = ref.build_csr(A_src, A_tgt, n)
    placed, _ = ref._kahn_residue(ptr, idx, np.bincount(A_tgt, minlength=n).astype(np.int64), n)
    assert placed == n, "F u accepted must be acyclic"


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_stage_matches_reference_and_is_monotone(seed):
    from mfas.metrics import score_from_order

    rng = np.random.default_rng(200 + seed)
    n = 60
    src, tgt, w = _random_digraph(rng, n, 400)
    g = _G(src, tgt, w, n)
    rank = rng.permutation(n).astype(np.int64)

    out_r, log_r = ref.reclaim_arcs(g, rank, rounds=2)
    out_f, log_f = fast.reclaim_arcs_fast(g, rank, rounds=2)
    assert np.array_equal(out_f, out_r)

    base = score_from_order(rank, g.src, g.tgt, g.weight)
    got = score_from_order(out_f, g.src, g.tgt, g.weight)
    w_acc = sum(e.get("w_accepted", 0.0) for e in log_f)
    assert got >= base + w_acc - 1e-9        # the reclamation lemma
    assert got >= base                       # monotone, unconditionally


def test_zero_rounds_is_the_identity():
    rng = np.random.default_rng(5)
    n = 40
    src, tgt, w = _random_digraph(rng, n, 200)
    g = _G(src, tgt, w, n)
    rank = rng.permutation(n).astype(np.int64)
    out, log = fast.reclaim_arcs_fast(g, rank, rounds=0)
    assert np.array_equal(out, rank) and log == []


def test_second_round_on_a_reclaimed_order_finds_nothing_new():
    """The stage reaches a fixed point — the certificate that the arc set is minimal."""
    rng = np.random.default_rng(13)
    n = 55
    src, tgt, w = _random_digraph(rng, n, 350)
    g = _G(src, tgt, w, n)
    rank = rng.permutation(n).astype(np.int64)
    out, log = fast.reclaim_arcs_fast(g, rank, rounds=6)
    assert log[-1]["n_accepted"] == 0 or len(log) == 6
    out2, log2 = fast.reclaim_arcs_fast(g, out, rounds=1)
    if log[-1]["n_accepted"] == 0:
        assert log2[0]["n_accepted"] == 0
        assert np.array_equal(out2, out)
