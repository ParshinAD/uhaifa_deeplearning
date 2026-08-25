"""The NET digraph as the refiner's structure graph (H60).

The defect this fixes
---------------------
:class:`mfas.refine.scc_recursive.SccRecursiveRefiner` builds its condensation matrix
over the RAW arc set, every edge counting the same::

    mat = coo_matrix((np.ones(eidx.size, dtype=np.int8), (ls_pos, lt_pos)), ...)

So a reciprocal pair ``u <-> v`` is an unbreakable 2-cycle: ``u`` and ``v`` always land
in one strongly connected component, and the refiner then preserves each component's
internal relative order by construction. This holds even when ``w_uv = 100`` and
``w_vu = 1`` — a pair whose orientation is worth 99 units is welded shut by an edge
worth 1.

The reduction
-------------
For an unordered pair ``{u, v}`` the ordering contributes ``w_uv`` if ``u`` precedes
``v`` and ``w_vu`` otherwise, which is

    ``min(w_uv, w_vu)``  +  ``|w_uv - w_vu|`` if the heavier direction is forward.

The first term does not depend on the ordering at all. So, with

    ``C = sum over unordered pairs of min(w_uv, w_vu)``           (a constant)
    ``D+ = {(u, v) : w_uv > w_vu}``, ``d_uv = w_uv - w_vu``       (the NET digraph)

maximising raw feedforward weight is *identical* to maximising feedforward weight on
``D+``. The rank-aggregation literature names the resulting partition: the Extended
Condorcet Criterion's finest block structure is the SCC condensation of the net
(majority) digraph, not of the raw one (arXiv:2506.15097).

Why substituting ``D+`` is still exactly monotone
-------------------------------------------------
The contiguous-block lemma is untouched — blocks are still contiguous position ranges,
so no edge with an endpoint outside a block can flip. Inside a block, laying the
``D+``-components out topologically makes every inter-component net arc forward (a true
net gain: the reciprocal half sits in the constant ``C``), and every intra-component
pair keeps its relative order (net change 0). Net-forward weight is therefore
non-decreasing, hence RAW feedforward weight is non-decreasing.

``D+``'s arcs are a subset of ``G``'s, so ``D+``'s SCCs **strictly refine** ``G``'s: the
substitution can only expose decompositions, never hide one.

Leakage-safety
--------------
``D+`` is built from the input edge arrays alone. No ordering, no score and no oracle
call enters its construction; ``data/best_solution`` is never read.

Measured (``experiments/outputs/proto_H60_alt_*.json``, from each dataset's stored
champion order at matched cycles and matched constants)::

    connectome  +0.027477 pp   (control arm +0.005039, net arm +0.032516)
    microns     see the JSON
    mouse       +0.085531 pp   (control arm exactly +0.000000)
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import numpy as np

__all__ = ["build_net_arcs", "net_structure_graph"]


def build_net_arcs(src: np.ndarray, tgt: np.ndarray, weight: np.ndarray, n: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Reduce a weighted digraph to its net digraph ``D+``.

    Parallel edges are aggregated first, then each unordered pair ``{u, v}`` is
    collapsed to the single arc carrying ``|w_uv - w_vu|`` in the heavier direction.
    Pairs that tie (``w_uv == w_vu``) and self-loops produce NO arc: their contribution
    to the objective is the same under either ordering, so they must not constrain the
    decomposition.

    Parameters
    ----------
    src, tgt : int arrays of shape (n_edges,)
        Contiguous node indices for each edge's source / target.
    weight : array of shape (n_edges,)
        Edge weights.
    n : int
        Number of nodes.

    Returns
    -------
    nsrc, ntgt : int64 arrays
        Endpoints of the net arcs (one per non-tied unordered pair).
    nw : float64 array
        Net weights ``|w_uv - w_vu|``, all strictly positive.
    const_c : float
        ``sum over unordered pairs of min(w_uv, w_vu)`` — the order-independent part of
        the objective. Reported for diagnosis; it is NOT recoverable by any ordering.

    Notes
    -----
    Aggregation runs in float64, which is exact for every weight total in this campaign
    (the largest is 4.2e7, far below 2^53). Nothing computed here reaches the frozen
    scorer, which always re-reads the original integer weights.
    """
    src = np.asarray(src, dtype=np.int64)
    tgt = np.asarray(tgt, dtype=np.int64)
    keep = src != tgt
    s, t = src[keep], tgt[keep]
    w = np.asarray(weight)[keep].astype(np.float64)

    # One weight per ORDERED pair (collapses parallel edges).
    okey = s * np.int64(n) + t
    uo, inv = np.unique(okey, return_inverse=True)
    ow = np.bincount(inv, weights=w, minlength=uo.size)
    ou, ov = uo // np.int64(n), uo % np.int64(n)

    # One (w_lo_hi, w_hi_lo) pair per UNORDERED pair.
    lo, hi = np.minimum(ou, ov), np.maximum(ou, ov)
    pkey = lo * np.int64(n) + hi
    is_lo_hi = ou == lo
    up, pinv = np.unique(pkey, return_inverse=True)
    w_lo_hi = np.bincount(pinv[is_lo_hi], weights=ow[is_lo_hi], minlength=up.size)
    w_hi_lo = np.bincount(pinv[~is_lo_hi], weights=ow[~is_lo_hi], minlength=up.size)

    d = w_lo_hi - w_hi_lo
    const_c = float(np.minimum(w_lo_hi, w_hi_lo).sum())

    plo, phi = up // np.int64(n), up % np.int64(n)
    pos, neg = d > 0, d < 0
    nsrc = np.concatenate([plo[pos], phi[neg]]).astype(np.int64)
    ntgt = np.concatenate([phi[pos], plo[neg]]).astype(np.int64)
    nw = np.concatenate([d[pos], -d[neg]]).astype(np.float64)
    return nsrc, ntgt, nw, const_c


def net_structure_graph(g) -> SimpleNamespace:
    """``g``'s net digraph, shaped so the block refiner can consume it directly.

    :func:`mfas.refine.scc_recursive.scc_recursive_refine` reads only ``src``, ``tgt``
    and ``n_nodes`` from its graph argument — it never looks at weights, because the
    condensation is purely structural. So a light namespace is enough, and using one
    keeps the returned object obviously distinct from a real
    :class:`~mfas.io.GraphData` (which still carries the weights every score is
    computed from).
    """
    nsrc, ntgt, nw, const_c = build_net_arcs(g.src, g.tgt, g.weight, g.n_nodes)
    return SimpleNamespace(src=nsrc, tgt=ntgt, weight=nw, n_nodes=g.n_nodes,
                           const_c=const_c, name=f"{getattr(g, 'name', '?')}-net")
