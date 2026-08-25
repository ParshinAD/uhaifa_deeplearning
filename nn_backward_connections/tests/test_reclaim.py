"""Correctness tests for minimal-FAS arc reclamation (H59).

The whole variant rests on one theorem, so the theorem is asserted here rather than left to a
screen to notice:

    If ``u`` is unreachable from ``v`` in the forward DAG ``F``, then ``F u {(u,v)}`` is
    acyclic and EVERY topological order of it scores at least ``base + w_uv``.

The tests below check the three things that could each break it independently:

1. **The reachability oracle is right.** ``reachable_in_F`` agrees with a brute-force
   transitive closure on random digraphs, including the budget-exhaustion path, which must
   answer ``None`` and never a wrong boolean.
2. **The accepted arc set is jointly acyclic.** Individually-safe arcs can conflict in pairs;
   ``resolve_conflicts`` must never emit a set that closes a cycle.
3. **The stage is monotone and it converges.** ``reclaim_arcs`` never lowers the exact score,
   it gains at least the reclaimed weight, and a second round finds nothing.

Everything runs on small random graphs and on mouse (148 nodes). Connectome and MICrONS are
NEVER touched by a unit test.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfas.io import load_dataset
from mfas.metrics import score_from_order
from mfas.refine.reclaim import (build_csr, reachable_in_F, reclaim_arcs,
                                 reclaimable_arcs, resolve_conflicts, topo_rank_stable)


def _random_digraph(rng, n, m, wmax=10):
    s = rng.integers(0, n, size=m).astype(np.int64)
    t = rng.integers(0, n, size=m).astype(np.int64)
    w = rng.integers(1, wmax + 1, size=m).astype(np.int64)
    return s, t, w


def _closure(F_src, F_tgt, n):
    """Brute-force reachability matrix (Floyd-Warshall on booleans). O(n^3), tiny n only."""
    R = np.zeros((n, n), dtype=bool)
    R[F_src, F_tgt] = True
    for k in range(n):
        R |= np.outer(R[:, k], R[k, :])
    return R


def _has_cycle(src, tgt, n):
    """Kahn: True iff the digraph has a cycle."""
    indeg = np.bincount(tgt, minlength=n).astype(np.int64)
    ptr, idx = build_csr(np.asarray(src, dtype=np.int64),
                         np.asarray(tgt, dtype=np.int64), n)
    stack = [int(x) for x in np.flatnonzero(indeg == 0)]
    placed = 0
    while stack:
        x = stack.pop()
        placed += 1
        for y in idx[ptr[x]:ptr[x + 1]]:
            indeg[y] -= 1
            if indeg[y] == 0:
                stack.append(int(y))
    return placed != n


class _G:
    """Minimal GraphData stand-in - reclaim_arcs only reads these four attributes."""
    def __init__(self, src, tgt, w, n, name="synthetic"):
        self.src, self.tgt, self.weight, self.name = src, tgt, w, name
        self._n = n

    @property
    def n_nodes(self):
        return self._n

    @property
    def total_weight(self):
        return float(self.weight.sum())


# ---------------------------------------------------------------------------
# 1. the reachability oracle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("trial", range(12))
def test_reachable_in_F_matches_brute_force_closure(trial):
    rng = np.random.default_rng(1000 + trial)
    n = 24
    s, t, _ = _random_digraph(rng, n, 90)
    rank = rng.permutation(n).astype(np.int64)
    fwd = rank[s] < rank[t]
    F_src, F_tgt = s[fwd], t[fwd]
    out_ptr, out_idx = build_csr(F_src, F_tgt, n)
    in_ptr, in_idx = build_csr(F_tgt, F_src, n)
    R = _closure(F_src, F_tgt, n)

    checked = 0
    for u, v in zip(s[~fwd].tolist(), t[~fwd].tolist()):
        if u == v:
            continue
        got = reachable_in_F(out_ptr, out_idx, in_ptr, in_idx, rank, v, u, 10 ** 9)
        assert got is not None
        assert got == bool(R[v, u]), f"v={v} u={u}"
        checked += 1
    assert checked > 0


def test_exhausted_budget_answers_none_never_a_wrong_boolean():
    """A budget of 0 must degrade to ``None``, which callers read as 'not reclaimable'."""
    rng = np.random.default_rng(7)
    n = 60
    s, t, _ = _random_digraph(rng, n, 600)
    rank = rng.permutation(n).astype(np.int64)
    fwd = rank[s] < rank[t]
    out_ptr, out_idx = build_csr(s[fwd], t[fwd], n)
    in_ptr, in_idx = build_csr(t[fwd], s[fwd], n)
    R = _closure(s[fwd], t[fwd], n)

    seen_none = False
    for u, v in zip(s[~fwd].tolist(), t[~fwd].tolist()):
        if u == v:
            continue
        got = reachable_in_F(out_ptr, out_idx, in_ptr, in_idx, rank, v, u, 0)
        if got is None:
            seen_none = True
        else:
            # Whatever it DOES answer under a starved budget must still be correct: the
            # early-exit paths (1-hop, empty window) are exact and must stay exact.
            assert got == bool(R[v, u])
    assert seen_none, "a zero budget should have starved at least one query"


def test_adjacent_backward_edge_is_reclaimable_iff_no_reverse_edge():
    """The smallest case, and the one the single-node sift already covers - so it must agree."""
    n = 2
    rank = np.array([1, 0], dtype=np.int64)          # node 1 sits before node 0
    # Only the backward edge 0 -> 1 exists: F is empty, so nothing can reach anything.
    ptr, idx = build_csr(np.array([], dtype=np.int64), np.array([], dtype=np.int64), n)
    assert reachable_in_F(ptr, idx, ptr, idx, rank, 1, 0, 10 ** 6) is False
    # Now add the reverse edge 1 -> 0, which IS forward: the pair is reciprocal and stuck.
    ptr2, idx2 = build_csr(np.array([1], dtype=np.int64), np.array([0], dtype=np.int64), n)
    iptr2, iidx2 = build_csr(np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), n)
    assert reachable_in_F(ptr2, idx2, iptr2, iidx2, rank, 1, 0, 10 ** 6) is True


# ---------------------------------------------------------------------------
# 2. the accepted set is jointly acyclic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("trial", range(10))
def test_accepted_arcs_are_jointly_acyclic(trial):
    rng = np.random.default_rng(2000 + trial)
    n = 40
    s, t, w = _random_digraph(rng, n, 260)
    keep = s != t
    s, t, w = s[keep], t[keep], w[keep]
    rank = rng.permutation(n).astype(np.int64)

    arcs, (F_src, F_tgt), _ = reclaimable_arcs(rank, s, t, w, n)
    acc_src, acc_tgt, _, st = resolve_conflicts(F_src, F_tgt, arcs, n)
    if not acc_src:
        pytest.skip("no reclaimable arcs on this draw")
    all_src = np.concatenate([F_src, np.asarray(acc_src, dtype=np.int64)])
    all_tgt = np.concatenate([F_tgt, np.asarray(acc_tgt, dtype=np.int64)])
    assert not _has_cycle(all_src, all_tgt, n)
    assert st["n_accepted"] == len(acc_src)


@pytest.mark.parametrize("trial", range(10))
def test_every_reclaimable_arc_is_individually_safe(trial):
    """Each arc alone must close no cycle - that is the lemma's precondition."""
    rng = np.random.default_rng(3000 + trial)
    n = 30
    s, t, w = _random_digraph(rng, n, 150)
    keep = s != t
    s, t, w = s[keep], t[keep], w[keep]
    rank = rng.permutation(n).astype(np.int64)
    arcs, (F_src, F_tgt), _ = reclaimable_arcs(rank, s, t, w, n)
    for _, u, v in arcs:
        one_src = np.concatenate([F_src, np.array([u], dtype=np.int64)])
        one_tgt = np.concatenate([F_tgt, np.array([v], dtype=np.int64)])
        assert not _has_cycle(one_src, one_tgt, n), f"arc ({u},{v}) closes a cycle alone"


def test_conflicting_pair_is_caught():
    """Two arcs that are each safe alone but form a cycle together: exactly one survives.

    This is the case the whole ``resolve_conflicts`` stage exists for, so it is built by hand
    rather than hoped for on a random draw. Ranks put the nodes in the order 0, 2, 3, 1::

        F      = {0 -> 1, 2 -> 3}          both forward
        arc e1 = (3 -> 0), weight 5        backward, and acyclic alone
        arc e2 = (1 -> 2), weight 3        backward, and acyclic alone
        e1+e2  ->  0 -> 1 -> 2 -> 3 -> 0   a cycle

    Reclaimability is a per-arc property, so both arcs pass the scan; only the joint check
    can catch this, and heaviest-first says the weight-5 arc is the one to keep.
    """
    n = 4
    rank = np.array([0, 3, 1, 2], dtype=np.int64)
    F_src = np.array([0, 2], dtype=np.int64)
    F_tgt = np.array([1, 3], dtype=np.int64)
    arcs = [(5.0, 3, 0), (3.0, 1, 2)]
    for _, u, v in arcs:
        assert rank[u] > rank[v], "candidate must be a backward edge"
        assert not _has_cycle(np.concatenate([F_src, np.array([u])]),
                              np.concatenate([F_tgt, np.array([v])]), n)
    assert _has_cycle(np.concatenate([F_src, np.array([3, 1])]),
                      np.concatenate([F_tgt, np.array([0, 2])]), n)
    acc_src, acc_tgt, acc_w, st = resolve_conflicts(F_src, F_tgt, arcs, n)
    assert not _has_cycle(np.concatenate([F_src, np.asarray(acc_src, dtype=np.int64)]),
                          np.concatenate([F_tgt, np.asarray(acc_tgt, dtype=np.int64)]), n)
    # heaviest-first: the weight-5 arc is the one that must survive
    assert acc_w and max(acc_w) == 5.0


# ---------------------------------------------------------------------------
# 3. the stage: monotone, at least the reclaimed weight, and it converges
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("trial", range(12))
def test_reclaim_never_lowers_the_score_and_gains_at_least_the_reclaimed_weight(trial):
    rng = np.random.default_rng(4000 + trial)
    n = 45
    s, t, w = _random_digraph(rng, n, 300)
    g = _G(s, t, w, n)
    rank = rng.permutation(n).astype(np.int64)
    before = score_from_order(rank, s, t, w)

    new_rank, log = reclaim_arcs(g, rank, rounds=1)
    after = score_from_order(new_rank, s, t, w)
    gained = float(sum(e["w_accepted"] for e in log))

    assert sorted(new_rank.tolist()) == list(range(n))     # still a permutation
    assert after >= before
    assert after >= before + gained - 1e-9                 # THE LEMMA


@pytest.mark.parametrize("trial", range(8))
def test_a_second_round_finds_nothing(trial):
    """The stage runs to a fixed point: after one round the arc set is minimal."""
    rng = np.random.default_rng(5000 + trial)
    n = 35
    s, t, w = _random_digraph(rng, n, 200)
    g = _G(s, t, w, n)
    rank = rng.permutation(n).astype(np.int64)
    once, _ = reclaim_arcs(g, rank, rounds=1)
    keep = s != t
    again, _, st = reclaimable_arcs(once, s[keep], t[keep], w[keep], n)
    assert st["n_reclaimable"] == 0, f"{st['n_reclaimable']} arcs still reclaimable"


def test_zero_rounds_is_a_no_op():
    rng = np.random.default_rng(11)
    n = 20
    s, t, w = _random_digraph(rng, n, 80)
    g = _G(s, t, w, n)
    rank = rng.permutation(n).astype(np.int64)
    out, log = reclaim_arcs(g, rank, rounds=0)
    assert log == []
    assert np.array_equal(out, rank)


def test_deterministic_across_repeated_calls():
    rng = np.random.default_rng(12)
    n = 50
    s, t, w = _random_digraph(rng, n, 340)
    g = _G(s, t, w, n)
    rank = rng.permutation(n).astype(np.int64)
    a, la = reclaim_arcs(g, rank, rounds=1)
    b, lb = reclaim_arcs(g, rank, rounds=1)
    assert np.array_equal(a, b)
    assert [e["w_accepted"] for e in la] == [e["w_accepted"] for e in lb]


def test_topo_rank_stable_returns_none_on_a_cycle():
    n = 3
    src = np.array([0, 1, 2], dtype=np.int64)
    tgt = np.array([1, 2, 0], dtype=np.int64)
    ptr, idx = build_csr(src, tgt, n)
    indeg = np.bincount(tgt, minlength=n).astype(np.int64)
    assert topo_rank_stable(ptr, idx, indeg, n, np.arange(n)) is None


def test_topo_rank_stable_keeps_the_incoming_order_when_it_is_already_topological():
    """No arcs added -> the least-disturbing topological order IS the incoming one."""
    n = 6
    src = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    tgt = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    ptr, idx = build_csr(src, tgt, n)
    indeg = np.bincount(tgt, minlength=n).astype(np.int64)
    order = topo_rank_stable(ptr, idx, indeg, n, np.arange(n))
    assert order.tolist() == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# mouse - a real graph, still tiny
# ---------------------------------------------------------------------------
def test_mouse_reclamation_is_monotone_and_converges():
    g = load_dataset("mouse")
    rng = np.random.default_rng(99)
    rank = rng.permutation(g.n_nodes).astype(np.int64)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    before = score_from_order(rank, src, tgt, g.weight)
    new_rank, log = reclaim_arcs(g, rank, rounds=2)
    after = score_from_order(new_rank, src, tgt, g.weight)
    gained = float(sum(e["w_accepted"] for e in log))
    assert after >= before + gained - 1e-9
    # the last round of a converged run reclaims nothing
    assert log[-1]["n_accepted"] == 0 or len(log) == 1
