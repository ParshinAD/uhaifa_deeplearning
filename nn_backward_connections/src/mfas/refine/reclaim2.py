"""Minimal-FAS arc reclamation, made cheap (P13) — same move class as :mod:`.reclaim`.

Why a second module rather than an edit
---------------------------------------
:mod:`mfas.refine.reclaim` is the reference implementation. It is the one that produced the
cycle-14 prototype numbers and H59's committed connectome screen, so it is left byte-for-byte
alone: it is the ORACLE this module is checked against, and a rewrite that silently changed
it would destroy the only independent check available.

This module computes the SAME answer by a cheaper route. "Same" is not asserted, it is
tested: ``tests/test_reclaim2.py`` requires the accepted arc set here to be identical to
:mod:`.reclaim`'s on random digraphs and on mouse, and ``experiments/proto_H63_fastreclaim.py``
requires it on connectome and microns against the stored cycle-14 artefacts.

Why the reference is slow, measured (``experiments/outputs/proto_H59_microns.json``)
------------------------------------------------------------------------------------
On microns one round costs 1075.3 s, of which

* **scan 222.0 s** — one Python call to ``reachable_in_F`` for each of 2,084,531 backward
  edges. The overwhelming majority are settled by the first two lines of that function (a
  ``searchsorted`` for a 1-hop edge, or an empty rank window), so almost all of that time is
  per-edge interpreter and NumPy-call overhead, not search.
* **conflict resolution 843.2 s** — 4,249 unbounded DFS queries over a 65,948-node Kahn
  residue holding ~10 M edges, i.e. ~198 ms each, essentially a full traversal per arc.

Both are addressed here without changing which arcs are accepted.

1. The scan's cheap cases are done for ALL backward edges at once, in NumPy
   (:func:`reclaimable_arcs_fast`). Three of the four prefilters are *proofs of
   unreachability*, so they answer positively as well as negatively:

   ============================  ==================================================
   1-hop        ``(v,u) in F``   reachable  -> NOT reclaimable
   empty window ``r_u = r_v+1``  unreachable -> reclaimable
   no exit      ``minOut[v]>=r_u``  every F-successor of ``v`` sits at or beyond ``u``;
                                 F-paths only move rightward, so none returns -> reclaimable
   no entry     ``maxIn[u]<=r_v``  every F-predecessor of ``u`` sits at or before ``v``
                                 -> reclaimable
   ============================  ==================================================

   Only what survives all four gets a bidirectional BFS, and that BFS is rank-space and
   slices each (ascending) adjacency row at ``u`` instead of scanning it.

2. Conflict resolution confines each reachability query to the **strongly connected
   component of** ``F u S``  (:func:`resolve_conflicts_scc`) instead of to Kahn's residue.
   This is exact, and it is much sharper: Kahn's residue is every node with an ancestor on a
   cycle, which on these graphs is nearly the whole vertex set (65,948 of 67,534 on microns),
   whereas the SCC is only what actually lies on cycles.

   *Confinement lemma.* Let ``S`` be the reclaimable set and ``A ⊆ S`` what has been accepted
   so far. For a candidate ``(u,v) ∈ S``, any ``F u A`` path ``v -> u`` closes a cycle
   together with the arc ``(u,v) ∈ S``, so every node on it is mutually reachable with ``u``
   in ``F u S`` and therefore lies in ``u``'s SCC of ``F u S``. Hence searching only inside
   that component can miss no path — and if the two endpoints lie in DIFFERENT components,
   no such path can exist at all and the arc is safe without any search. ∎

Determinism, and the direction of every approximation
-----------------------------------------------------
Candidates are tested in ``(-weight, u, v)`` order exactly as in the reference, so the
accepted set is a deterministic function of the input. Both budgets keep the reference's
fail-safe direction: a reachability query that exhausts its budget is treated as REACHABLE
(the arc is dropped), which can only forgo gain and can never create a cycle.

Leakage-safety: identical to the reference — only edge weights and current ranks are read,
and the frozen oracle scores the result without ever selecting an arc.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .reclaim import build_csr, topo_rank_stable

__all__ = ["reclaimable_arcs_fast", "resolve_conflicts_scc", "reclaim_arcs_fast"]

_INF = np.iinfo(np.int64).max


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
def _reach_rank(out_ptr, out_idx, in_ptr, in_idx, v: int, u: int, budget: int):
    """Is ``u`` reachable from ``v``, in RANK space?  ``True`` / ``False`` / ``None``.

    Rank space means node label == rank, so every edge of ``F`` goes from a smaller label to
    a larger one and a ``v -> u`` path is confined to the open interval ``(v, u)``. Callers
    guarantee ``v + 1 < u`` and that no direct ``v -> u`` edge exists.

    ``None`` = budget exhausted, which callers MUST read as "reachable" (see module
    docstring). Bidirectional, expanding whichever frontier is smaller; adjacency rows are
    ascending, so each is sliced at the far endpoint rather than scanned.
    """
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
                j = int(np.searchsorted(row, u, side="right"))
                spent += j
                for y in row[:j].tolist():
                    if y == u:
                        return True
                    if y in fwd:
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
                j = int(np.searchsorted(row, v, side="left"))
                spent += row.shape[0] - j
                for y in row[j:].tolist():
                    if y == v:
                        return True
                    if y in bwd:
                        continue
                    if y in fwd:
                        return True
                    bwd.add(y)
                    nxt.append(y)
            b_frontier = nxt
        if spent > budget:
            return None
    return False


def reclaimable_arcs_fast(rank, e_src, e_tgt, e_w, n, budget=1_000_000,
                          deadline: Optional[float] = None, max_candidates=None):
    """Backward edges that close no cycle in ``F``, heaviest first.

    Drop-in replacement for :func:`mfas.refine.reclaim.reclaimable_arcs`: same signature,
    same return shape ``(arcs, (F_src, F_tgt), stats)``, same arc set, same order.
    """
    ra = rank[e_src]
    rb = rank[e_tgt]
    fwd_mask = ra < rb

    # F in RANK space (labels are ranks, so every row is a set of larger labels).
    out_ptr, out_idx = build_csr(ra[fwd_mask], rb[fwd_mask], n)
    in_ptr, in_idx = build_csr(rb[fwd_mask], ra[fwd_mask], n)

    # ...and in NODE space, because that is what the caller re-sorts with.
    F_src, F_tgt = e_src[fwd_mask], e_tgt[fwd_mask]

    bmask = ~fwd_mask
    B_src, B_tgt, B_w = e_src[bmask], e_tgt[bmask], e_w[bmask]
    B_ru, B_rv = ra[bmask], rb[bmask]          # rank of u (tail) and of v (head); ru > rv

    order_b = np.argsort(-B_w.astype(np.float64), kind="stable")
    if max_candidates is not None:
        order_b = order_b[:max_candidates]

    # ── Prefilter 1: a direct v -> u edge already in F (1-hop) => reachable ──────────
    key_f = np.sort(ra[fwd_mask] * np.int64(n) + rb[fwd_mask])
    q = B_rv * np.int64(n) + B_ru
    pos = np.searchsorted(key_f, q)
    onehop = np.zeros(q.shape[0], dtype=bool)
    ok = pos < key_f.shape[0]
    if ok.any():
        onehop[ok] = key_f[pos[ok]] == q[ok]

    # ── Prefilters 2-4: proofs that no F-path v -> u exists ─────────────────────────
    has_out = out_ptr[1:] > out_ptr[:-1]
    min_out = np.full(n, _INF, dtype=np.int64)
    min_out[has_out] = out_idx[out_ptr[:-1][has_out]]
    has_in = in_ptr[1:] > in_ptr[:-1]
    max_in = np.full(n, -1, dtype=np.int64)
    max_in[has_in] = in_idx[in_ptr[1:][has_in] - 1]

    proved_unreach = (~onehop) & ((B_ru == B_rv + 1)          # empty rank window
                                  | (min_out[B_rv] >= B_ru)   # v has no exit into the window
                                  | (max_in[B_ru] <= B_rv))   # u has no entry from the window
    undecided = (~onehop) & (~proved_unreach)

    # ── Only the survivors get a search, in weight order so a cut scan degrades well ──
    verdict = np.zeros(q.shape[0], dtype=np.int8)             # 0 reachable, 1 reclaimable, 2 unknown
    verdict[proved_unreach] = 1
    todo = order_b[undecided[order_b]]
    n_unknown = 0
    w_unknown = 0.0
    truncated = False
    n_searched = 0
    t_pre = time.time()
    for i in todo:
        r = _reach_rank(out_ptr, out_idx, in_ptr, in_idx,
                        int(B_rv[i]), int(B_ru[i]), budget)
        n_searched += 1
        if r is None:
            verdict[i] = 2
            n_unknown += 1
            w_unknown += float(B_w[i])
        elif r is False:
            verdict[i] = 1
        if deadline is not None and (n_searched & 255) == 0 and time.time() > deadline:
            truncated = True
            break

    sel = order_b[verdict[order_b] == 1]
    arcs = [(float(B_w[i]), int(B_src[i]), int(B_tgt[i])) for i in sel]
    stats = dict(n_backward=int(B_w.shape[0]), w_backward=float(B_w.sum()),
                 n_tested=int(order_b.shape[0]), scan_truncated=truncated,
                 n_reclaimable=len(arcs), w_reclaimable=float(sum(a[0] for a in arcs)),
                 n_unknown=n_unknown, w_unknown=w_unknown,
                 n_onehop=int(onehop.sum()), n_proved=int(proved_unreach.sum()),
                 n_searched=n_searched, t_search_s=time.time() - t_pre)
    return arcs, (F_src, F_tgt), stats


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------
def _dfs_within(adj_ptr, adj_idx, extra, comp, k, v: int, u: int, budget: int):
    """Bounded DFS for ``v -> u`` confined to component ``k``. ``None`` on exhaustion.

    ``extra`` is the small dict-of-lists overlay carrying arcs accepted so far; the CSR is
    the static forward-edge part.
    """
    seen = {v}
    stack = [v]
    spent = 0
    while stack:
        x = stack.pop()
        row = adj_idx[adj_ptr[x]:adj_ptr[x + 1]].tolist()
        ex = extra.get(x)
        if ex:
            row = row + ex
        for y in row:
            if y == u:
                return True
            if y in seen or comp[y] != k:
                continue
            seen.add(y)
            stack.append(y)
        spent += 1
        if spent > budget:
            return None
    return False


def resolve_conflicts_scc(F_src, F_tgt, arcs, n, conflict_budget=200_000):
    """Greedy heaviest-first re-add of ``arcs``, with SCC-confined exact conflict tests.

    Drop-in replacement for :func:`mfas.refine.reclaim.resolve_conflicts`, returning the same
    ``(acc_src, acc_tgt, acc_w, stats)``. See the module docstring for the confinement lemma.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    R = sorted(arcs, key=lambda a: (-a[0], a[1], a[2]))
    base_stats = dict(n_accepted=0, w_accepted=0.0, n_conflicts=0, n_budget_rejects=0,
                      n_cycle_residue=0, batch_acyclic=True, n_scc_nontrivial=0,
                      n_scc_max=0, n_arcs_in_scc=0)
    if not R:
        return [], [], [], base_stats

    r_u = np.asarray([a[1] for a in R], dtype=np.int64)
    r_v = np.asarray([a[2] for a in R], dtype=np.int64)
    r_w = [a[0] for a in R]

    # SCCs of F u S. Anything strongly connected here is exactly what can carry a cycle.
    A_src = np.concatenate([F_src, r_u])
    A_tgt = np.concatenate([F_tgt, r_v])
    m = csr_matrix((np.ones(A_src.shape[0], dtype=np.int8), (A_src, A_tgt)), shape=(n, n))
    _, comp = connected_components(m, directed=True, connection="strong")
    comp = comp.astype(np.int64)
    sizes = np.bincount(comp, minlength=int(comp.max()) + 1)
    in_scc = sizes[comp] > 1

    conflicted = in_scc[r_u] & in_scc[r_v] & (comp[r_u] == comp[r_v])
    n_arcs_in_scc = int(conflicted.sum())
    base_stats.update(n_scc_nontrivial=int((sizes > 1).sum()),
                      n_scc_max=int(sizes.max()),
                      n_arcs_in_scc=n_arcs_in_scc,
                      n_cycle_residue=int(in_scc.sum()))
    if n_arcs_in_scc == 0:
        base_stats.update(n_accepted=len(r_w), w_accepted=float(sum(r_w)))
        return list(r_u), list(r_v), r_w, base_stats
    base_stats["batch_acyclic"] = False

    # Static forward-edge CSR restricted to the union of the nontrivial components. Edges
    # crossing components can never lie on a cycle, so dropping them costs nothing.
    keep = in_scc[F_src] & in_scc[F_tgt] & (comp[F_src] == comp[F_tgt])
    adj_ptr, adj_idx = build_csr(F_src[keep], F_tgt[keep], n)

    extra: dict = {}
    acc_src, acc_tgt, acc_w = [], [], []
    n_conflicts = 0
    n_budget = 0
    for j in range(len(R)):
        wt, u, v = r_w[j], int(r_u[j]), int(r_v[j])
        if not conflicted[j]:
            acc_src.append(u); acc_tgt.append(v); acc_w.append(wt)
            continue
        k = int(comp[u])
        r = _dfs_within(adj_ptr, adj_idx, extra, comp, k, v, u, conflict_budget)
        if r is None:
            n_budget += 1
            continue
        if r:
            n_conflicts += 1
            continue
        acc_src.append(u); acc_tgt.append(v); acc_w.append(wt)
        extra.setdefault(u, []).append(v)
    base_stats.update(n_accepted=len(acc_w), w_accepted=float(sum(acc_w)),
                      n_conflicts=n_conflicts, n_budget_rejects=n_budget)
    return acc_src, acc_tgt, acc_w, base_stats


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def reclaim_arcs_fast(g, rank, rounds=1, budget=1_000_000, conflict_budget=200_000,
                      time_budget_s: Optional[float] = None, max_candidates=None):
    """One or more rounds of arc reclamation. Same contract as
    :func:`mfas.refine.reclaim.reclaim_arcs`, and the same monotonicity guarantee: a round
    either yields a topological order of ``F u S`` — worth at least ``base + w(S)`` by the
    reclamation lemma — or leaves the incoming order untouched.
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
        arcs, (F_src, F_tgt), st = reclaimable_arcs_fast(
            cur, e_src, e_tgt, e_w, n, budget=budget, deadline=deadline,
            max_candidates=max_candidates)
        t_scan = time.time() - t_r
        t_c = time.time()
        acc_src, acc_tgt, acc_w, cst = resolve_conflicts_scc(
            F_src, F_tgt, arcs, n, conflict_budget=conflict_budget)
        t_conf = time.time() - t_c

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
        entry["t_scan_s"] = t_scan
        entry["t_conflict_s"] = t_conf
        entry["wall_s"] = time.time() - t_r
        log.append(entry)
        if not acc_w or st["scan_truncated"]:
            break
    return cur, log
