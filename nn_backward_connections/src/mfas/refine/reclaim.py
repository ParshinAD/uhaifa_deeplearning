"""Minimal-FAS arc reclamation (H59) — re-add backward edges that close no cycle.

The move class
--------------
Every other refiner in this campaign works in POSITION space: it moves nodes and asks what
that does to the edges. This one works in ARC space. It asks a different question —

    is the campaign's feedback arc set *minimal*?

A feedback arc set is minimal when no deleted arc can be put back without creating a cycle.
Nothing in the pipeline has ever enforced that, and it turns out not to hold.

The lemma this rests on
-----------------------
An order induces the forward edge set ``F = {(u,v) : rank[u] < rank[v]}``. ``F`` is a DAG and
the order is one of its topological orders. Let ``(u, v)`` be BACKWARD, i.e. ``rank[v] <
rank[u]``, so it currently counts as feedback.

    **Reclamation lemma.** If ``u`` is not reachable from ``v`` in ``F``, then ``F u {(u,v)}``
    is acyclic, and EVERY topological order of it scores at least ``base + w_uv``.

    *Proof.* Adding an arc to a DAG creates a cycle iff its head already reaches its tail, so
    ``F u {(u,v)}`` is acyclic exactly under the stated condition. A topological order of a
    DAG makes every arc of that DAG feedforward; it therefore retains all of ``F`` — the
    ``base`` score — and gains ``(u,v)``. Any edge outside ``F u {(u,v)}`` can only add
    more. ∎

So reclamation is **monotone by construction** and its realised gain is bounded BELOW by the
reclaimed weight. That lower bound is asserted at runtime by the caller
(:mod:`mfas.experiments.H59`) and by ``tests/test_reclaim.py``.

Why no existing stage can find these moves
------------------------------------------
Reclaiming one arc generally requires re-sorting thousands of nodes, none of which profits on
its own, so the single-node sift (H35) and the exact pair move (H45) are blind to it by
construction. The SCC refiner (H36/H42) condenses ``G``, where the two endpoints sit in one
giant SCC — and H60 established that even the exact net-digraph condensation ``D+`` does not
separate them. The move here never asks where a node should go; it edits the arc set and
re-derives an order from it.

Why the search is affordable
----------------------------
Every ``F``-path moves strictly rightward in rank, so a ``v -> u`` path is confined to the
open rank interval ``(rank[v], rank[u])``. Reachability is therefore a bidirectional BFS
inside a window, never a global search, and the overwhelmingly common answer ("yes, in one or
two hops") costs almost nothing.

Leakage-safety: only edge weights and the current ranks are ever read. The frozen oracle
scores the result; it never selects an arc. ``data/best_solution`` is never opened.
"""
from __future__ import annotations

import heapq
import time
from typing import Optional

import numpy as np

__all__ = ["build_csr", "reachable_in_F", "reclaimable_arcs", "resolve_conflicts",
           "topo_rank_stable", "reclaim_arcs"]


# ---------------------------------------------------------------------------
# CSR
# ---------------------------------------------------------------------------
def build_csr(row: np.ndarray, col: np.ndarray, n: int):
    """CSR of a directed edge list, each row block sorted ascending by column.

    The sort is what lets the 1-hop test be a ``searchsorted`` instead of a scan.
    """
    order = np.lexsort((col, row))
    row_s, col_s = row[order], col[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(row_s, minlength=n), out=indptr[1:])
    return indptr, np.ascontiguousarray(col_s, dtype=np.int64)


# ---------------------------------------------------------------------------
# Reachability inside the rank window
# ---------------------------------------------------------------------------
def reachable_in_F(out_ptr, out_idx, in_ptr, in_idx, rank, v: int, u: int,
                   budget: int) -> Optional[bool]:
    """Is ``u`` reachable from ``v`` in ``F``?  ``True`` / ``False`` / ``None``.

    ``None`` means the expansion budget ran out before the question was settled. Callers
    MUST treat ``None`` as "reachable" — i.e. not reclaimable — because that is the only
    direction that cannot manufacture a gain.

    Bidirectional BFS confined to the open rank interval ``(rank[v], rank[u])``, expanding
    whichever frontier is currently smaller.
    """
    rv = int(rank[v])
    ru = int(rank[u])
    if ru <= rv:
        return True                       # not a backward pair; nothing to reclaim

    out_v = out_idx[out_ptr[v]:out_ptr[v + 1]]
    j = int(np.searchsorted(out_v, u))
    if j < out_v.shape[0] and int(out_v[j]) == u:
        return True                       # 1-hop: a direct v -> u edge already in F
    if ru == rv + 1:
        return False                      # empty window, and no direct edge

    fwd = {v}
    bwd = {u}
    f_frontier = [v]
    b_frontier = [u]
    spent = 0

    while f_frontier and b_frontier:
        if len(f_frontier) <= len(b_frontier):
            nxt = []
            for x in f_frontier:
                row = out_idx[out_ptr[x]:out_ptr[x + 1]]
                spent += row.shape[0]
                for y in row:
                    y = int(y)
                    yr = int(rank[y])
                    if yr >= ru:
                        if y == u:
                            return True
                        continue
                    if yr <= rv or y in fwd:
                        continue
                    if y in bwd:
                        return True
                    fwd.add(y)
                    nxt.append(y)
            f_frontier = nxt
        else:
            nxt = []
            for x in b_frontier:
                row = in_idx[in_ptr[x]:in_ptr[x + 1]]
                spent += row.shape[0]
                for y in row:
                    y = int(y)
                    yr = int(rank[y])
                    if yr <= rv:
                        if y == v:
                            return True
                        continue
                    if yr >= ru or y in bwd:
                        continue
                    if y in fwd:
                        return True
                    bwd.add(y)
                    nxt.append(y)
            b_frontier = nxt
        if spent > budget:
            return None
    return False


def reclaimable_arcs(rank, e_src, e_tgt, e_w, n, budget=1_000_000,
                     deadline: Optional[float] = None, max_candidates: Optional[int] = None):
    """Backward edges that close no cycle in ``F``, heaviest first.

    Returns ``(arcs, stats)`` where ``arcs`` is a list of ``(weight, u, v)``. Candidates are
    tested in weight-descending order (ties broken by a stable sort on edge index, so the
    result is deterministic), which front-loads the weight that could actually matter and
    makes a truncated scan degrade gracefully.
    """
    fwd_mask = rank[e_src] < rank[e_tgt]
    F_src, F_tgt = e_src[fwd_mask], e_tgt[fwd_mask]
    B_src, B_tgt, B_w = e_src[~fwd_mask], e_tgt[~fwd_mask], e_w[~fwd_mask]

    out_ptr, out_idx = build_csr(F_src, F_tgt, n)
    in_ptr, in_idx = build_csr(F_tgt, F_src, n)

    order_b = np.argsort(-B_w.astype(np.float64), kind="stable")
    if max_candidates is not None:
        order_b = order_b[:max_candidates]

    arcs = []
    n_unknown = 0
    w_unknown = 0.0
    n_tested = 0
    truncated = False
    for i in order_b:
        u, v, wt = int(B_src[i]), int(B_tgt[i]), float(B_w[i])
        r = reachable_in_F(out_ptr, out_idx, in_ptr, in_idx, rank, v, u, budget)
        n_tested += 1
        if r is None:
            n_unknown += 1
            w_unknown += wt
        elif r is False:
            arcs.append((wt, u, v))
        if deadline is not None and (n_tested & 1023) == 0 and time.time() > deadline:
            truncated = True
            break
    stats = dict(n_backward=int(B_w.shape[0]), w_backward=float(B_w.sum()),
                 n_tested=n_tested, scan_truncated=truncated,
                 n_reclaimable=len(arcs), w_reclaimable=float(sum(a[0] for a in arcs)),
                 n_unknown=n_unknown, w_unknown=w_unknown)
    return arcs, (F_src, F_tgt), stats


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------
def _kahn_residue(out_ptr, out_idx, indeg, n):
    """Nodes Kahn's algorithm cannot place — every cycle is inside this set."""
    deg = indeg.copy()
    stack = [int(x) for x in np.flatnonzero(deg == 0)]
    placed = 0
    alive = np.ones(n, dtype=bool)
    while stack:
        x = stack.pop()
        alive[x] = False
        placed += 1
        for y in out_idx[out_ptr[x]:out_ptr[x + 1]]:
            y = int(y)
            deg[y] -= 1
            if deg[y] == 0:
                stack.append(y)
    return placed, alive


def _reachable_small(adj, v, u, budget):
    """Bounded DFS in a dict-of-lists digraph. ``None`` on budget exhaustion."""
    seen = {v}
    stack = [v]
    spent = 0
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y == u:
                return True
            if y not in seen:
                seen.add(y)
                stack.append(y)
        spent += 1
        if spent > budget:
            return None
    return False


def resolve_conflicts(F_src, F_tgt, arcs, n, conflict_budget=200_000):
    """Greedy heaviest-first re-add of ``arcs``, exact where it can be, safe where it cannot.

    Reclaimable arcs are pairwise independent with respect to ``F`` — each alone closes no
    cycle — but two together can, when ``F`` has a path from one's head to the other's tail.
    So:

    1. Try the whole batch. One Kahn pass over ``F u arcs``; if it places every node the
       batch is jointly acyclic and everything is accepted. This is the cheap common case.
    2. Otherwise Kahn's residue contains every cycle. Arcs with an endpoint outside it cannot
       lie on one and are accepted outright; the rest are re-added greedily, heaviest first,
       against the induced subgraph on the residue.

    A conflict test that exhausts ``conflict_budget`` REJECTS its arc. That only ever forgoes
    gain — it can never accept an arc that would create a cycle — and the caller's final Kahn
    would refuse to produce an order at all if it did.
    """
    R = sorted(arcs, key=lambda a: (-a[0], a[1], a[2]))
    if not R:
        return [], [], [], dict(n_accepted=0, w_accepted=0.0, n_conflicts=0,
                                n_budget_rejects=0, n_cycle_residue=0, batch_acyclic=True)
    r_u = np.asarray([a[1] for a in R], dtype=np.int64)
    r_v = np.asarray([a[2] for a in R], dtype=np.int64)
    r_w = [a[0] for a in R]

    A_tgt = np.concatenate([F_tgt, r_v])
    a_ptr, a_idx = build_csr(np.concatenate([F_src, r_u]), A_tgt, n)
    placed, alive = _kahn_residue(a_ptr, a_idx,
                                  np.bincount(A_tgt, minlength=n).astype(np.int64), n)
    if placed == n:
        return (list(r_u), list(r_v), r_w,
                dict(n_accepted=len(r_w), w_accepted=float(sum(r_w)), n_conflicts=0,
                     n_budget_rejects=0, n_cycle_residue=0, batch_acyclic=True))

    m = alive[F_src] & alive[F_tgt]
    adj = {}
    for a, b in zip(F_src[m].tolist(), F_tgt[m].tolist()):
        adj.setdefault(a, []).append(b)

    acc_src, acc_tgt, acc_w = [], [], []
    n_conflicts = 0
    n_budget = 0
    for wt, u, v in R:
        if not (alive[u] and alive[v]):
            acc_src.append(u); acc_tgt.append(v); acc_w.append(wt)
            continue
        r = _reachable_small(adj, v, u, conflict_budget)
        if r is None:
            n_budget += 1
            continue
        if r:
            n_conflicts += 1
            continue
        acc_src.append(u); acc_tgt.append(v); acc_w.append(wt)
        adj.setdefault(u, []).append(v)
    return (acc_src, acc_tgt, acc_w,
            dict(n_accepted=len(acc_w), w_accepted=float(sum(acc_w)),
                 n_conflicts=n_conflicts, n_budget_rejects=n_budget,
                 n_cycle_residue=int(alive.sum()), batch_acyclic=False))


# ---------------------------------------------------------------------------
# Rank-stable topological sort
# ---------------------------------------------------------------------------
def topo_rank_stable(out_ptr, out_idx, indeg, n, key):
    """Kahn with a min-heap on ``key``: the topological order closest to the incoming one.

    The lemma holds for ANY topological order of ``F u S``; this one is chosen because it
    disturbs the incoming order least, which keeps the backward edges that were NOT reclaimed
    roughly where they were. Returns ``None`` if a cycle survives, which can only mean the
    conflict bookkeeping is wrong.
    """
    deg = indeg.copy()
    heap = [(int(key[x]), int(x)) for x in np.flatnonzero(deg == 0)]
    heapq.heapify(heap)
    order = np.empty(n, dtype=np.int64)
    filled = 0
    while heap:
        _, x = heapq.heappop(heap)
        order[filled] = x
        filled += 1
        for y in out_idx[out_ptr[x]:out_ptr[x + 1]]:
            y = int(y)
            deg[y] -= 1
            if deg[y] == 0:
                heapq.heappush(heap, (int(key[y]), y))
    return None if filled != n else order


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def reclaim_arcs(g, rank, rounds=1, budget=1_000_000, conflict_budget=200_000,
                 time_budget_s: Optional[float] = None, max_candidates=None):
    """One or more rounds of arc reclamation. Returns ``(new_rank, log)``.

    Monotone: the returned order is never worse than the incoming one, because every round
    either produces a topological order of ``F u S`` (which by the lemma scores at least
    ``base + w(S)``) or returns the incoming order untouched.

    ``time_budget_s`` is an ABORT, not a sizing rule: it can only cut a round short, and a
    round that is cut short before accepting anything leaves the order exactly as it was. The
    number of rounds is a fixed constant, so a machine fast enough to finish the configured
    work always produces the same answer.
    """
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight)
    n = int(g.n_nodes)
    keep = src != tgt
    e_src, e_tgt, e_w = src[keep], tgt[keep], w[keep]

    t0 = time.time()
    deadline = None if time_budget_s is None else t0 + float(time_budget_s)
    cur = np.asarray(rank, dtype=np.int64).copy()
    log = []

    for rnd in range(int(rounds)):
        if deadline is not None and time.time() > deadline:
            break
        t_r = time.time()
        arcs, (F_src, F_tgt), st = reclaimable_arcs(
            cur, e_src, e_tgt, e_w, n, budget=budget, deadline=deadline,
            max_candidates=max_candidates)
        acc_src, acc_tgt, acc_w, cst = resolve_conflicts(
            F_src, F_tgt, arcs, n, conflict_budget=conflict_budget)

        entry = dict(round=rnd, **st, **cst)
        if acc_w:
            A_tgt = np.concatenate([F_tgt, np.asarray(acc_tgt, dtype=np.int64)])
            a_ptr, a_idx = build_csr(
                np.concatenate([F_src, np.asarray(acc_src, dtype=np.int64)]), A_tgt, n)
            new_order = topo_rank_stable(
                a_ptr, a_idx, np.bincount(A_tgt, minlength=n).astype(np.int64), n, cur)
            if new_order is None:
                entry["fatal"] = "cycle in F u S"
                log.append(entry)
                break
            nr = np.empty(n, dtype=np.int64)
            nr[new_order] = np.arange(n, dtype=np.int64)
            cur = nr
        entry["wall_s"] = time.time() - t_r
        log.append(entry)
        if not acc_w or st["scan_truncated"]:
            break
    return cur, log
