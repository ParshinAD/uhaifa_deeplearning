"""Recursive SCC-topological BLOCK refinement — a structural move class (H36).

Motivation
----------
Every discrete refiner in this campaign so far moves ONE node at a time
(:mod:`mfas.refine.insertion`, :mod:`mfas.refine.underrelax`). Meta-rule **M4**
(``autoresearch/killed.json``) records that bounded *rank-window* neighbourhoods
recover nothing — the recoverable weight is long-range — and closes H22. The
revival condition M4 states is explicit: *the window must be replaced by a
STRUCTURAL neighbourhood (SCC / block), not a rank window*. This module is that
neighbourhood.

The contiguous-block lemma
--------------------------
If a set of nodes occupies a **contiguous range of positions** ``[lo, hi)`` on the
line, then permuting those nodes among themselves cannot change the orientation of
any edge with an endpoint outside the range: the outside endpoint sits either
before every position in ``[lo, hi)`` or after every one of them. Hence each
contiguous block is an **independent sub-problem** — its intra-block feedforward
weight can be maximised without any regard for the rest of the line.

The move
--------
For a block ``B`` with induced subgraph ``G[B]``:

1. decompose ``G[B]`` into strongly connected components;
2. lay the SCCs out in a **topological order of the condensation DAG** (ties broken
   by the incoming order, so we stay as close to the current solution as possible);
3. keep every SCC's nodes in their **current relative order**, and recurse into each.

Two facts make step 2 safe. Every edge between two distinct SCCs ``A`` and ``B``
runs one way only (an edge back from ``B`` to ``A`` would merge them into a single
SCC), so a topological layout makes **every** inter-SCC edge feedforward. And step 3
leaves every intra-SCC edge's orientation untouched. Therefore

    **the block's contribution is non-decreasing, and the whole procedure is
    monotone non-decreasing in the exact score, by construction.**

A block that is a single SCC has no internal decomposition, so there we split it in
half by current position and recurse into each half — a *strict* subgraph, whose
strong connectivity generally breaks, so the halves decompose again. That is what
makes the scheme recursive rather than one-shot.

Why one-shot SCC condensation is NOT this
-----------------------------------------
``dr_tmp/FINDINGS_underrelaxation.md`` E2 measured the top-level condensation on the
fly connectome: 9,626 SCCs but a giant one holding 92.82% of the nodes, inter-SCC
weight only 1.50% of the total and already 99.99% feedforward in H30's order.
Forcing all inter-SCC edges feedforward gains **+0.00013 pp** — nothing. The whole
residual lives *inside* the giant SCC. The recursion attacks exactly that: it
re-decomposes strict subgraphs of the giant SCC, where strong connectivity is far
weaker, and the decomposition it finds there is not visible to the top-level split.

Leakage-safety
--------------
Every decision comes from the input edge arrays and the current order alone. The
frozen oracle is used only to score whole candidate vectors (best-by-oracle) in
:func:`alternate_scc_sift`; it never enters a move choice, a loss, or a
dataset-specific branch, and ``data/best_solution`` is never read.
"""
from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..io import GraphData
from ..metrics import pct, score_from_order
from .underrelax import sift_underrelaxed

__all__ = [
    "topo_order_labels",
    "SccRecursiveRefiner",
    "scc_recursive_refine",
    "alternate_scc_sift",
    "DEFAULT_SPLIT_FRACS",
]

# Split fractions cycled across rounds. Varying where a single SCC is cut moves the
# block boundaries between rounds, so a later round sees decompositions the earlier
# one could not (an edge cut by a boundary is invisible to both sides). Fixed,
# ordered and deterministic — this is NOT randomisation and consumes no RNG.
DEFAULT_SPLIT_FRACS: Tuple[float, ...] = (0.5, 0.382, 0.618, 0.25, 0.75)


# ──────────────────────────────────────────────────────────────────────────────
# Condensation topological order, tie-broken by the CURRENT order
# ──────────────────────────────────────────────────────────────────────────────
def topo_order_labels(n_lab: int, ls: np.ndarray, lt: np.ndarray,
                      lab_key: np.ndarray) -> np.ndarray:
    """Kahn topological order of a condensation DAG, tie-broken by ``lab_key``.

    Parameters
    ----------
    n_lab : int
        Number of SCC labels.
    ls, lt : int arrays
        Label of the source / target endpoint of each edge that crosses two
        distinct labels. Duplicates are fine; they are deduplicated here.
    lab_key : array, shape (n_lab,)
        Tie-break key per label — the label's smallest current position. Among the
        many valid topological orders this picks one close to the incoming order,
        which keeps the move conservative.

    Returns
    -------
    int64 array, shape (n_lab,) — the label sequence, first to last.

    Raises
    ------
    RuntimeError
        If the graph over labels is not acyclic. That cannot happen for a genuine
        condensation, so it is a guard against a corrupted label array, not an
        expected condition.
    """
    if ls.size:
        pair = np.unique(ls.astype(np.int64) * n_lab + lt.astype(np.int64))
        us = pair // n_lab
        vs = pair % n_lab
    else:
        us = vs = np.empty(0, dtype=np.int64)

    indeg = np.zeros(n_lab, dtype=np.int64)
    np.add.at(indeg, vs, 1)
    o = np.argsort(us, kind="stable")
    us_s, vs_s = us[o], vs[o]
    start = np.searchsorted(us_s, np.arange(n_lab + 1))

    heap = [(float(lab_key[i]), int(i)) for i in np.flatnonzero(indeg == 0)]
    heapq.heapify(heap)
    out = np.empty(n_lab, dtype=np.int64)
    k = 0
    while heap:
        _, u = heapq.heappop(heap)
        out[k] = u
        k += 1
        for j in range(start[u], start[u + 1]):
            v = int(vs_s[j])
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (float(lab_key[v]), v))
    if k != n_lab:
        raise RuntimeError(
            f"condensation is not acyclic: ordered {k} of {n_lab} labels")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# The recursive refiner
# ──────────────────────────────────────────────────────────────────────────────
class SccRecursiveRefiner:
    """Recursive SCC-topological block refinement over a rank vector.

    One call to :meth:`run` performs a single full pass over the line. The pass is
    monotone non-decreasing in the exact feedforward score (see the module
    docstring), so no accept/reject is needed inside it.

    Parameters
    ----------
    src, tgt : arrays
        Edge endpoint arrays of the graph (self-loops are dropped internally).
    n : int
        Number of nodes.
    min_block : int
        Blocks of at most this size are left alone. Below a few tens of nodes the
        scipy call costs more than the decomposition can return.
    split_frac : float
        Where to cut a block that is a single SCC, as a fraction of its size.

    Attributes set by :meth:`run` (diagnostics only)
    ------------------------------------------------
    n_scc_calls : number of strong-decomposition calls made.
    n_topo_blocks : blocks that decomposed into more than one SCC.
    n_reordered : blocks whose layout actually changed.
    """

    def __init__(self, src: np.ndarray, tgt: np.ndarray, n: int,
                 min_block: int = 32, split_frac: float = 0.5) -> None:
        self.src = np.asarray(src, dtype=np.int64)
        self.tgt = np.asarray(tgt, dtype=np.int64)
        self.n = int(n)
        self.min_block = int(min_block)
        self.split_frac = float(split_frac)
        self.n_scc_calls = 0
        self.n_topo_blocks = 0
        self.n_reordered = 0

    def run(self, rank: np.ndarray) -> np.ndarray:
        """Return a refined rank vector. Never scores worse than ``rank``."""
        rank = np.asarray(rank, dtype=np.int64).copy()
        self.seq = np.empty(self.n, dtype=np.int64)
        self.seq[rank] = np.arange(self.n, dtype=np.int64)
        self.pos = rank
        self.n_scc_calls = self.n_topo_blocks = self.n_reordered = 0
        eidx = np.flatnonzero(self.src != self.tgt).astype(np.int64)
        self._refine(0, self.n, eidx, is_scc=False)
        return self.pos.copy()

    # -- internals ------------------------------------------------------------
    def _refine(self, lo: int, hi: int, eidx: np.ndarray, is_scc: bool) -> None:
        """Refine the contiguous block of positions ``[lo, hi)``.

        ``eidx`` indexes exactly the edges with BOTH endpoints inside the block.
        ``is_scc`` records that the caller already knows the block is strongly
        connected (it came out of a condensation), so re-deciding that is waste.
        """
        nb = hi - lo
        if nb <= self.min_block or eidx.size == 0:
            return
        if is_scc:
            self._split(lo, hi, eidx)
            return

        ls_pos = self.pos[self.src[eidx]] - lo      # local positions in [0, nb)
        lt_pos = self.pos[self.tgt[eidx]] - lo

        self.n_scc_calls += 1
        mat = coo_matrix((np.ones(eidx.size, dtype=np.int8), (ls_pos, lt_pos)),
                         shape=(nb, nb))
        n_lab, labels = connected_components(mat, directed=True,
                                             connection="strong")
        if n_lab > 1:
            self._apply_topo(lo, hi, eidx, labels, n_lab, ls_pos, lt_pos)
        else:
            self._split(lo, hi, eidx)

    def _apply_topo(self, lo: int, hi: int, eidx: np.ndarray,
                    labels: np.ndarray, n_lab: int,
                    ls_pos: np.ndarray, lt_pos: np.ndarray) -> None:
        """Lay the block's SCCs out topologically, then recurse into each."""
        nb = hi - lo
        ls, lt = labels[ls_pos], labels[lt_pos]
        cross = ls != lt

        lab_key = np.full(n_lab, nb, dtype=np.int64)
        np.minimum.at(lab_key, labels, np.arange(nb, dtype=np.int64))
        order_lab = topo_order_labels(n_lab, ls[cross], lt[cross], lab_key)
        self.n_topo_blocks += 1

        lab_rank = np.empty(n_lab, dtype=np.int64)
        lab_rank[order_lab] = np.arange(n_lab, dtype=np.int64)
        # Stable sort by the label's topological rank => labels in topological
        # order, and inside a label the nodes keep their current relative order.
        new_local = np.argsort(lab_rank[labels], kind="stable")

        if not np.array_equal(new_local, np.arange(nb, dtype=np.int64)):
            self.n_reordered += 1
            self.seq[lo:hi] = self.seq[lo:hi][new_local]
            self.pos[self.seq[lo:hi]] = np.arange(lo, hi, dtype=np.int64)

        # Recurse into each (now contiguous) SCC block with its INTRA edges only.
        sizes = np.bincount(labels, minlength=n_lab)
        starts = lo + np.concatenate(
            [[0], np.cumsum(sizes[order_lab])[:-1]]).astype(np.int64)
        intra = eidx[~cross]
        if intra.size == 0:
            return
        intra_rank = lab_rank[ls[~cross]]
        o = np.argsort(intra_rank, kind="stable")
        intra_sorted = intra[o]
        bounds = np.searchsorted(intra_rank[o], np.arange(n_lab + 1))
        for r, lab in enumerate(order_lab):
            if sizes[lab] <= self.min_block:
                continue
            self._refine(int(starts[r]), int(starts[r] + sizes[lab]),
                         intra_sorted[bounds[r]:bounds[r + 1]], is_scc=True)

    def _split(self, lo: int, hi: int, eidx: np.ndarray) -> None:
        """Cut a single-SCC block in two and recurse; the halves decompose again."""
        nb = hi - lo
        mid = lo + max(1, min(nb - 1, int(round(nb * self.split_frac))))
        ps = self.pos[self.src[eidx]]
        pt = self.pos[self.tgt[eidx]]
        left = (ps < mid) & (pt < mid)
        right = (ps >= mid) & (pt >= mid)
        self._refine(lo, mid, eidx[left], is_scc=False)
        self._refine(mid, hi, eidx[right], is_scc=False)


def scc_recursive_refine(g: GraphData, rank: np.ndarray, *, min_block: int = 32,
                         split_frac: float = 0.5) -> np.ndarray:
    """One monotone pass of recursive SCC-topological block refinement."""
    ref = SccRecursiveRefiner(np.asarray(g.src), np.asarray(g.tgt), g.n_nodes,
                              min_block=min_block, split_frac=split_frac)
    return ref.run(rank)


# ──────────────────────────────────────────────────────────────────────────────
# Alternation driver: block moves <-> single-node moves
# ──────────────────────────────────────────────────────────────────────────────
def alternate_scc_sift(g: GraphData, init_rank: np.ndarray, *, n_cycles: int,
                       sift_sweeps: int = 8, k_full: int = 2, alpha: float = 0.7,
                       min_block: int = 32,
                       split_fracs: Sequence[float] = DEFAULT_SPLIT_FRACS,
                       time_budget_s: Optional[float] = None,
                       g_struct=None, tie_break: str = "first"
                       ) -> Tuple[np.ndarray, float, List[Dict]]:
    """Alternate the block refiner with the champion's single-node sift.

    The two move classes are disjoint: the sift relocates ONE node at a time to its
    exact-optimal gap and is blind to a rearrangement that only pays off jointly;
    the block refiner rearranges whole SCCs at once but never moves a node relative
    to its own component. Each therefore re-opens moves the other has exhausted,
    and the alternation reaches a joint fixed point neither reaches alone.

    ``g_struct`` (H60) supplies the graph the BLOCK REFINER condenses, while the sift
    and every score still read ``g``. It defaults to ``None``, meaning ``g`` itself --
    so the champions H36/H42/H52 are bit-identical to before this parameter existed.
    Passing :func:`mfas.refine.net_condense.net_structure_graph(g) <net_structure_graph>`
    condenses the NET digraph instead, which strictly refines ``g``'s components and is
    still exactly monotone (see that module's docstring for the proof).

    ``tie_break`` is forwarded to the inner sift (H73); it defaults to ``"first"``, the
    rule every champion through H64 was measured with.

    Returns ``(best_rank, best_score, log)``; ``log`` has one row per cycle. The
    returned order is the best the frozen oracle has scored, so it can never be
    worse than ``init_rank``.
    """
    g_struct = g if g_struct is None else g_struct
    src_o, tgt_o = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight
    rank = np.asarray(init_rank, dtype=np.int64).copy()
    best_score = score_from_order(rank, src_o, tgt_o, g.weight)
    best_rank = rank.copy()

    log: List[Dict] = []
    t0 = time.time()
    for c in range(n_cycles):
        if time_budget_s is not None and (time.time() - t0) > time_budget_s:
            break
        ta = time.time()
        rank = scc_recursive_refine(g_struct, rank, min_block=min_block,
                                    split_frac=split_fracs[c % len(split_fracs)])
        s_scc = score_from_order(rank, src_o, tgt_o, g.weight)
        t_scc = time.time() - ta
        if s_scc > best_score:
            best_score, best_rank = s_scc, rank.copy()

        tb = time.time()
        left = None if time_budget_s is None else max(
            0.0, time_budget_s - (time.time() - t0))
        rank, s_sift, _ = sift_underrelaxed(g, rank, k_full=k_full, alpha=alpha,
                                            max_sweeps=sift_sweeps,
                                            time_budget_s=left, tie_break=tie_break)
        t_sift = time.time() - tb
        if s_sift > best_score:
            best_score, best_rank = s_sift, rank.copy()

        log.append(dict(cycle=c, split_frac=split_fracs[c % len(split_fracs)],
                        after_scc_pct=pct(s_scc, total),
                        after_sift_pct=pct(s_sift, total),
                        best_pct=pct(best_score, total),
                        t_scc_s=t_scc, t_sift_s=t_sift,
                        cum_wall_s=time.time() - t0))
    return best_rank, float(best_score), log
