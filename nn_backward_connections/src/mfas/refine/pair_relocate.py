"""Exact-gain SEQUENTIAL pair relocation — a move class the single-node sift cannot see.

The move
--------
Take a backward edge ``(u, v)`` — an edge ``u -> v`` whose endpoints sit in the wrong order,
``rank[v] < rank[u]``, so it currently counts as feedback. Extract BOTH endpoints and re-insert
them adjacent, as ``u, v``, at some split ``r`` inside the interval between their old positions.

Why the champion's sift is blind to it
--------------------------------------
:func:`mfas.refine.insertion.jacobi_best_gaps` gives every node its own exact-optimal gap,
computed with all other nodes held fixed. A pair whose endpoints only profit **jointly** — where
neither ``u`` nor ``v`` improves alone but both improve by meeting — is invisible to that
kernel by construction. Measured on the champion's connectome order
(``experiments/outputs/proto_H45_pair_connectome.json``): of 3,498 pairs with a strictly positive
exact gain, **989 (28.3 %) have zero solo gain at both endpoints**, and the single best move spans
92,958 positions — 68 % of the line. Nothing else in the pipeline reaches that range.

The exact gain
--------------
Let ``n_1 .. n_t`` be the nodes strictly between ``v`` and ``u``. After the move the interval
reads ``n_1 .. n_r, u, v, n_{r+1} .. n_t``, and

    Delta(r) = (w_uv - w_vu)
             + sum_{j<=r} [ w(n_j->v) - w(v->n_j) ]
             + sum_{j>r}  [ w(u->n_j) - w(n_j->u) ]

**Proof.** The interval is a contiguous position range and every node that moves stays inside it,
so by the contiguous-block lemma (see :mod:`mfas.refine.segment`) no edge with an endpoint
outside can flip. The ``n_j`` keep their relative order, so nothing among them flips. Only edges
incident to ``u`` or ``v`` flip, each on exactly one side of the split: ``v`` passes to the right
of ``n_1..n_r`` and ``u`` to the left of ``n_{r+1}..n_t``. The ``u``-``v`` edge itself becomes
feedforward (``+w_uv``) and any ``v -> u`` edge becomes feedback (``-w_vu``).

``Delta`` is piecewise constant in ``r`` and changes only where ``n_j`` is a neighbour of ``u``
or ``v``, so the optimal split costs ``O(d(u) + d(v))`` **independent of the interval length**.

Why SEQUENTIAL, and why that is the whole point
------------------------------------------------
Every other refiner in this package computes all gains and then applies a maximal set of moves
with **disjoint** windows. For single-node moves, whose windows are one position wide, that is
correct and fast. For this move class it is catastrophic: the interval spans the distance between
a backward edge's endpoints — a mean of ~20,536 positions on connectome — so disjoint intervals
barely fit on the line. Measured (``proto_H50_connectome.json``): over 40 batched sweeps,
**4,018,567 positive-gain candidates were found and 4,193 applied — 0.104 %**.

Applying moves one at a time, each recomputed against the updated order, removes that constraint
entirely. On the same start, sequential realised **4.13x the gain per move** and ~11x the total
gain in under half the wall clock (``proto_H51_connectome.json``).

Determinism and sizing
----------------------
No RNG anywhere. Candidates are drawn from a max-heap keyed by ``(-weight, edge_index)``, so ties
break on the lowest edge index and the whole refiner is a pure function of the graph and the
incoming order.

The budget is a **pop count**, never a wall-clock. A time-sized budget would make the result
machine-dependent and break the bit-reproducibility that ``sota.json``'s std = 0 and the campaign's
1-seed screen policy rest on — the same reasoning that made P05's runtime guard an abort at a
stage boundary rather than a sizing rule.

Leakage-safety
--------------
Every move is chosen from the input edge weights and the current ranks alone, via the closed form
above. The frozen oracle scores whole candidate rank vectors only; it never enters a move choice,
and ``data/best_solution`` is never read.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..io import GraphData
from ..metrics import pct, score_from_order

__all__ = ["build_direction_csr", "pair_move_gain", "pair_relocate"]


def build_direction_csr(row: np.ndarray, col: np.ndarray, val: np.ndarray, n: int):
    """CSR adjacency in one direction: ``ptr``, ``idx``, ``w``."""
    cnt = np.bincount(row, minlength=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(cnt, out=ptr[1:])
    o = np.argsort(row, kind="stable")
    return ptr, col[o].copy(), val[o].copy()


def pair_move_gain(rank, u, v, out_ptr, out_idx, out_w, in_ptr, in_idx, in_w
                   ) -> Tuple[float, int, int, int]:
    """Exact gain and best split for relocating ``u, v`` adjacent. ``O(d(u) + d(v))``.

    Assumes ``rank[v] < rank[u]`` (the edge ``u -> v`` is currently feedback).
    Returns ``(gain, cut_position, lo, hi)``.

    Contributions are aggregated **per POSITION**, not per edge. A single interval node may be a
    neighbour of both ``u`` and ``v`` (and may carry parallel edges); emitting one event per edge
    and walking them individually manufactures intermediate states where a node sits in the prefix
    for its ``v``-edges and the suffix for its ``u``-edges. Those are not achievable splits and
    their gain is inflated — that bug scored 1 of 18 against the frozen oracle before it was found.
    """
    lo = int(rank[v])
    hi = int(rank[u])

    a, b = out_ptr[u], out_ptr[u + 1]
    seg = out_idx[a:b]
    hit = np.flatnonzero(seg == v)
    w_uv = float(out_w[a:b][hit].sum()) if hit.size else 0.0
    a, b = out_ptr[v], out_ptr[v + 1]
    seg = out_idx[a:b]
    hit = np.flatnonzero(seg == u)
    w_vu = float(out_w[a:b][hit].sum()) if hit.size else 0.0

    acc_v: Dict[int, float] = {}
    acc_u: Dict[int, float] = {}
    for k in range(in_ptr[v], in_ptr[v + 1]):
        p = int(rank[in_idx[k]])
        if lo < p < hi:
            acc_v[p] = acc_v.get(p, 0.0) + float(in_w[k])
    for k in range(out_ptr[v], out_ptr[v + 1]):
        p = int(rank[out_idx[k]])
        if lo < p < hi:
            acc_v[p] = acc_v.get(p, 0.0) - float(out_w[k])
    for k in range(out_ptr[u], out_ptr[u + 1]):
        p = int(rank[out_idx[k]])
        if lo < p < hi:
            acc_u[p] = acc_u.get(p, 0.0) + float(out_w[k])
    for k in range(in_ptr[u], in_ptr[u + 1]):
        p = int(rank[in_idx[k]])
        if lo < p < hi:
            acc_u[p] = acc_u.get(p, 0.0) - float(in_w[k])

    base = w_uv - w_vu
    if not acc_v and not acc_u:
        return base, lo, lo, hi

    positions = sorted(set(acc_v) | set(acc_u))
    suffix = 0.0
    for val in acc_u.values():
        suffix += val
    gain, cut = base + suffix, lo
    prefix = 0.0
    for p in positions:
        prefix += acc_v.get(p, 0.0)
        suffix -= acc_u.get(p, 0.0)
        g = base + prefix + suffix
        if g > gain:
            gain, cut = g, p
    return gain, cut, lo, hi


def pair_relocate(g: GraphData, init_rank: np.ndarray, *, n_passes: int = 2,
                  max_pops: Optional[int] = None, tol: float = 1e-9,
                  time_budget_s: Optional[float] = None
                  ) -> Tuple[np.ndarray, float, List[Dict]]:
    """Sequential exact-gain pair relocation, best-by-oracle.

    Same signature register as :func:`mfas.refine.underrelax.sift_underrelaxed` so it drops into
    a pipeline stage unchanged.

    Parameters
    ----------
    n_passes
        Heap refills. Each pass rebuilds the candidate heap from the CURRENT backward edges,
        which is how newly-created backward edges re-enter consideration.
    max_pops
        TOTAL pop budget across all passes — the deterministic sizing knob. ``None`` means drain.
    time_budget_s
        Checked only at a PASS boundary, so the campaign's run-level wall-clock guard can
        interrupt this stage without making the result machine-dependent. It is an abort, never
        a sizing rule.

    Returns ``(best_rank, best_score, pass_log)``. The move is monotone by construction (only
    strictly-positive exact gains are applied), so the returned order is never worse than
    ``init_rank``.
    """
    import time as _time

    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    n, total = g.n_nodes, g.total_weight

    keep = src != tgt
    out_ptr, out_idx, out_w = build_direction_csr(src[keep], tgt[keep], w[keep], n)
    in_ptr, in_idx, in_w = build_direction_csr(tgt[keep], src[keep], w[keep], n)

    rank = np.asarray(init_rank, dtype=np.int64).copy()
    seq = np.empty(n, dtype=np.int64)
    seq[rank] = np.arange(n, dtype=np.int64)
    best_score = score_from_order(rank, src, tgt, g.weight)
    best_rank = rank.copy()

    popped_total = 0
    applied_total = 0
    log: List[Dict] = []
    t0 = _time.time()

    for p_i in range(n_passes):
        if time_budget_s is not None and (_time.time() - t0) > time_budget_s:
            break
        if max_pops is not None and popped_total >= max_pops:
            break

        back = np.flatnonzero(rank[src] > rank[tgt])
        heap = [(-float(w[e]), int(e)) for e in back]
        heapq.heapify(heap)
        p_start = score_from_order(rank, src, tgt, g.weight)
        p_applied = 0
        p_predicted = 0.0
        t_pass = _time.time()

        while heap:
            if max_pops is not None and popped_total >= max_pops:
                break
            _, ei = heapq.heappop(heap)
            popped_total += 1
            u = int(src[ei])
            v = int(tgt[ei])
            if rank[u] <= rank[v]:
                continue                       # no longer a backward edge
            gain, cut, lo, hi = pair_move_gain(rank, u, v, out_ptr, out_idx, out_w,
                                               in_ptr, in_idx, in_w)
            if gain <= tol:
                continue
            block = [x for x in seq[lo:hi + 1] if x != u and x != v]
            new_block: List[int] = []
            placed = False
            for x in block:
                new_block.append(x)
                if rank[x] == cut:
                    new_block.append(u)
                    new_block.append(v)
                    placed = True
            if not placed:
                new_block = [u, v] + new_block
            seq[lo:hi + 1] = np.asarray(new_block, dtype=np.int64)
            rank[seq[lo:hi + 1]] = np.arange(lo, hi + 1, dtype=np.int64)
            p_applied += 1
            applied_total += 1
            p_predicted += gain

        p_end = score_from_order(rank, src, tgt, g.weight)
        if p_end > best_score:
            best_score, best_rank = p_end, rank.copy()
        log.append(dict(pass_i=p_i, n_applied=p_applied, n_popped=popped_total,
                        before_pct=pct(p_start, total), after_pct=pct(p_end, total),
                        gain_pp=pct(p_end, total) - pct(p_start, total),
                        predicted_gain_pp=100.0 * p_predicted / total,
                        wall=_time.time() - t_pass,
                        cum_wall_s=_time.time() - t0))
        if p_applied == 0:
            break

    return best_rank, float(best_score), log
