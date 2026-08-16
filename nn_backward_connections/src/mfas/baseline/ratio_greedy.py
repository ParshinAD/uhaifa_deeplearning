"""Ratio-greedy initial ordering — a cheap peel by ``(out_w + 1) / (in_w + 1)``.

This is the reference family's Algorithm 1 (Vahidi 2025, arXiv:2506.13799; see
``autoresearch/lit/vahidi-2025.md``), reimplemented here from its description and measured
independently on our own frozen scorer.

The rule
--------
Repeatedly take the remaining node with the largest ``(out_w + 1) / (in_w + 1)``, where the
weights are sums over edges to and from nodes that are still remaining, place it at the next
position from the FRONT, remove it, and update its neighbours' remaining sums. A node with much
out-weight and little in-weight is the one that most wants to be early: placing it next turns
many of its edges feedforward. The ``+1`` terms are the paper's, and they keep the ratio finite
for pure sources and pure sinks.

It differs from :func:`mfas.experiments.H02.greedy_fas_order` (Eades-Lin-Smyth / GreedyAbs) in
the key it maximises — a RATIO rather than the difference ``out_w - in_w`` — and in having no
separate source/sink special-casing.

Measured, not assumed (``experiments/outputs/proto_H48_connectome.json``,
``proto_H48_mouse.json``, 2026-08-16, both scored by the frozen oracle):

===========  ==========================  ==========================  ==========
dataset      greedy-FAS (H02)            ratio greedy (this)         init delta
===========  ==========================  ==========================  ==========
connectome   68.91343 %   (17.4 s)       **74.61730 %**  (20.0 s)    **+5.70387 pp**
mouse        90.12630 %                  88.38256 %                  -1.74374 pp
===========  ==========================  ==========================  ==========

So it is a much better start on the mission graph and a worse one on the tiny graph, at
essentially identical cost. What survives refinement is a separate question and is NOT settled
by this table — after the champion's stage 3 the connectome advantage collapses from +5.70 pp to
+0.03456 pp, i.e. the sift absorbs 99.4% of it.

RETURN CONVENTION — read this before using it
----------------------------------------------
Returns a **RANK vector**: ``rank[u]`` is the position of node ``u`` in ``[0, n)``, smaller =
earlier = source side. This deliberately matches :func:`mfas.experiments.H02.greedy_fas_order`,
which returns a rank despite being named "order".

The two conventions are NOT interchangeable and confusing them is silent, not loud: during this
function's prototype a true-order return was fed through a rank-expecting path and scored our
mouse greedy at 43.797% instead of the 90.126% the champion's own run records. It looked like a
plausible bad number rather than a crash. Hence: rank in, rank out, everywhere.

Determinism
-----------
No RNG. Ties in the heap break on the lowest node id via the ``(key, node)`` tuple ordering, so
the result is a pure function of the graph. The campaign's 1-seed screen policy and
``sota.json``'s std = 0 depend on this holding all the way down the pipeline.

Leakage-safety
--------------
Uses only ``g.src`` / ``g.tgt`` / ``g.weight`` and the remaining-subgraph degree sums. The
discrete oracle is never consulted, and ``data/best_solution`` is never read.
"""
from __future__ import annotations

import heapq

import numpy as np

from ..io import GraphData

__all__ = ["ratio_greedy_rank"]


def ratio_greedy_rank(g: GraphData) -> np.ndarray:
    """Greedy peel by ``(out_w + 1) / (in_w + 1)``; returns a RANK vector, int64, shape (n,).

    Complexity is ``O((n + m) log n)`` — one lazy-deletion max-heap, each edge relaxed once
    when its far endpoint is removed. On connectome (n = 136,648, m = 5,657,719) this measured
    20.0 s, against 17.4 s for the incumbent greedy-FAS.
    """
    n = g.n_nodes
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)

    keep = src != tgt                      # self-loops can never be feedforward or feedback
    src, tgt, w = src[keep], tgt[keep], w[keep]

    out_w = np.zeros(n, dtype=np.float64)
    in_w = np.zeros(n, dtype=np.float64)
    np.add.at(out_w, src, w)
    np.add.at(in_w, tgt, w)

    def _csr(row, col, val):
        cnt = np.bincount(row, minlength=n)
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(cnt, out=ptr[1:])
        o = np.argsort(row, kind="stable")
        return ptr, col[o].copy(), val[o].copy()

    op, oi, ow = _csr(src, tgt, w)         # u -> out-neighbours
    ip, ii, iw = _csr(tgt, src, w)         # v <- in-neighbours

    alive = np.ones(n, dtype=bool)
    # Negated key: heapq is a min-heap, so -ratio pops the largest ratio first. The node id is
    # the tie-break, which is what makes this deterministic.
    heap = [(-(out_w[u] + 1.0) / (in_w[u] + 1.0), int(u)) for u in range(n)]
    heapq.heapify(heap)

    order = np.empty(n, dtype=np.int64)    # order[i] = node placed at position i
    filled = 0
    while filled < n:
        key, u = heapq.heappop(heap)
        if not alive[u]:
            continue
        live_key = -(out_w[u] + 1.0) / (in_w[u] + 1.0)
        if live_key != key:                # stale entry: its neighbours moved since it was pushed
            heapq.heappush(heap, (live_key, u))
            continue
        alive[u] = False
        order[filled] = u
        filled += 1
        # u has left the remaining subgraph: it no longer contributes in-weight to its
        # out-neighbours, nor out-weight to its in-neighbours.
        for k in range(op[u], op[u + 1]):
            v = int(oi[k])
            if alive[v]:
                in_w[v] -= ow[k]
                heapq.heappush(heap, (-(out_w[v] + 1.0) / (in_w[v] + 1.0), v))
        for k in range(ip[u], ip[u + 1]):
            v = int(ii[k])
            if alive[v]:
                out_w[v] -= iw[k]
                heapq.heappush(heap, (-(out_w[v] + 1.0) / (in_w[v] + 1.0), v))

    rank = np.empty(n, dtype=np.int64)     # invert order -> rank, per the convention above
    rank[order] = np.arange(n, dtype=np.int64)
    return rank
