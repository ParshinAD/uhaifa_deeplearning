"""Exact-gain bounded-span SEGMENT moves — a rigid-block relocation move class (H41).

Motivation
----------
The champion (:mod:`mfas.experiments.H42`) alternates exactly two move classes:

* stage 3 — :func:`mfas.refine.underrelax.sift_underrelaxed`: relocate ONE node to its
  exact-optimal gap. Blind to any rearrangement that only pays off *jointly*: if a run
  of 300 nodes belongs 800 positions earlier but no single one of them profits from
  moving alone, the sift is at a fixed point and sees nothing.
* stage 4 — :func:`mfas.refine.scc_recursive.scc_recursive_refine`: rearrange whole
  SCCs of a contiguous block. It can only relocate a group when that group happens to
  come out of a strong decomposition; a group that is *not* SCC-separable from its
  neighbours is invisible to it, and it never moves a node relative to its own SCC.

``experiments/diagnosis.md`` reports Kendall-tau 0.610 and a median rank-distance of
~22,580 against the 84.6147% reference, with the disagreement spread uniformly over
weight buckets and sitting on median-degree endpoints. That is the signature of whole
REGIONS sitting in the wrong place, which is precisely the hole between the two
existing move classes. This module is the third move class: take a contiguous run of
positions and move it, as a rigid unit, to a nearby target position.

The move, and why it is really an adjacent-block swap
-----------------------------------------------------
Take the segment ``S`` occupying positions ``[i, i+L)`` and move it right by ``d``
positions. The ``d`` nodes it jumps over are exactly the contiguous block
``J = [i+L, i+L+d)``, and afterwards ``J`` occupies ``[i, i+d)`` and ``S`` occupies
``[i+d, i+L+d)``. So a right segment move **is** the swap of the two adjacent blocks
``S`` and ``J``. A *left* move of a segment ``S'`` of length ``L'`` by ``d'`` positions
is the swap of the adjacent blocks ``J' = [i'-d', i')`` and ``S'`` — i.e. the same
primitive with ``(L, d) = (d', L')`` and start ``i'-d'``. **Enumerating right moves over
a symmetric (L, d) grid therefore covers left moves too**, which is why nothing below
ever special-cases direction.

The exact-gain lemma
--------------------
This is the same contiguous-block lemma stage 4 relies on. Let ``A = [a, b)`` and
``B = [b, c)`` be adjacent contiguous position ranges and let the move swap them
(``B`` first, then ``A``), keeping the internal order of each rigid. Then

(a) an edge with both endpoints in ``A`` keeps its orientation (``A`` is rigid);
(b) an edge with both endpoints in ``B`` keeps its orientation (``B`` is rigid);
(c) an edge with at most one endpoint in ``A ∪ B`` keeps its orientation — the outside
    endpoint lies either before every position in ``[a, c)`` or after every one of
    them, both before and after the move;
(d) an edge with one endpoint in ``A`` and one in ``B`` **flips**.

Hence, writing ``F = Σ w over edges A -> B`` (feedforward before the swap) and
``R = Σ w over edges B -> A`` (feedback before the swap),

    **delta = R − F**

exactly, with no rescoring of the graph. For a right move of ``S`` by ``d`` that is
``delta = W(J→S) − W(S→J)``; for a left move of ``S`` by ``d`` it is
``delta = W(S→J) − W(J→S)`` with ``J`` the block on the left — the same formula under
the reparametrisation above.

Vectorising it: one difference array per (L, d)
------------------------------------------------
Fix ``L`` and ``d`` and let ``W = L + d``. For an edge with endpoint positions
``p = rank[src]``, ``q = rank[tgt]``, put ``x = min(p, q)``, ``y = max(p, q)`` and

    s = −w  if the edge runs x → y (feedforward now, so the swap costs it)
    s = +w  if the edge runs y → x (feedback now, so the swap wins it)

The edge is an ``A``–``B`` cross edge of the window starting at ``i`` iff
``i ≤ x ≤ i+L−1`` and ``i+L ≤ y ≤ i+W−1``, i.e. iff

    i ∈ [max(x−L+1, y−W+1), min(x, y−L)]

which is a **contiguous interval of window starts**. So the gain of every start ``i``
at this ``(L, d)`` is a single prefix sum over a difference array of length ``n−W+2``,
built with one ``bincount`` over the edges. Two consequences:

* only edges with ``y − x ≤ W − 1`` can contribute at all (both endpoints must fit in a
  window of ``W`` positions), so the edge list is sorted **once per sweep** by span and
  each ``(L, d)`` reads only a prefix of it;
* the whole ``(L, d)`` grid costs ``O(Σ_{L,d} (m_{L+d} + n))``, never ``O(n²)``.

Bounded span
------------
``seg_lengths`` and ``offsets`` are geometric ladders (default ``1, 2, …, 1024``). The
ladder is what keeps the search bounded: an unbounded enumeration of (segment, target)
pairs is ``O(n²)`` windows. The defaults cover a span of at most 2048 positions, which
is ~1.5% of the connectome's line — chosen because a geometric ladder covers six orders
of magnitude of segment size in 11 rungs, so a region that belongs "somewhere near here"
is reachable in one move at *some* rung even though no single rung is dense.

Applying a sweep: disjoint windows compose exactly
--------------------------------------------------
Two accepted moves whose windows ``[i₁, i₁+W₁)`` and ``[i₂, i₂+W₂)`` are **disjoint** are
independent: each permutes only nodes inside its own window, so by (c) above every edge
between the two windows, and every edge from either window to the rest of the line,
keeps its orientation. Their exact gains therefore ADD. A sweep collects, for every start
``i``, the best ``(L, d)``; sorts the strictly-positive candidates by gain; and greedily
accepts those whose windows are still free. The realised score delta equals the sum of
the accepted gains exactly — asserted against the frozen oracle in
``tests/test_refine_segment.py``.

Monotone
--------
Only strictly-positive exact gains are applied, so a sweep can never lower the score.
That is what lets this alternate safely with stages 3 and 4.

Deterministic
-------------
No RNG anywhere, not even a tie-break. Ties in the per-start argmax over the grid go to
the FIRST ``(L, d)`` in the fixed iteration order (``L`` ascending, then ``d``
ascending); ties in the greedy selection go to the lowest start index. The campaign's
seed policy and its bit-reproducibility rest on the pipeline never drawing from ``seed``.

Numerics
--------
Gains are accumulated in float64. For integer weights this is EXACT provided the graph's
total weight stays below ``2**53`` (connectome: 4.19e7, i.e. 2e8× of margin), because
every difference-array entry and every prefix sum is an integer bounded in absolute value
by the total weight, and float64 represents such integers exactly.

Leakage-safety
--------------
Every move is decided from the input edge weights and the current ranks alone (the
closed-form gain above). The frozen oracle (:func:`mfas.metrics.score_from_order`) is
consulted only to score whole candidate rank vectors in the drivers (best-by-oracle);
it never enters a move choice, a loss, or a dataset-specific branch, and
``data/best_solution`` is never read.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..io import GraphData
from ..metrics import pct, score_from_order
from .scc_recursive import DEFAULT_SPLIT_FRACS, scc_recursive_refine
from .underrelax import sift_underrelaxed

__all__ = [
    "DEFAULT_SEG_LENGTHS",
    "DEFAULT_OFFSETS",
    "build_span_index",
    "segment_gains",
    "best_segment_moves",
    "select_disjoint_moves",
    "apply_segment_moves",
    "segment_sweep",
    "segment_refine",
    "alternate_scc_sift_segment",
    # small reference kernels for unit tests only
    "segment_move_gain_ref",
    "apply_segment_move_ref",
]


# Geometric ladders. ``seg_lengths`` is the rigid segment size L; ``offsets`` is how far
# it travels, d. Both are needed: (L, d) and (d, L) are the same primitive read in the
# two travel directions, so a SYMMETRIC pair of ladders makes the move class direction
# symmetric for free. 11 rungs -> 121 (L, d) pairs, max window span 2048 positions.
DEFAULT_SEG_LENGTHS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
DEFAULT_OFFSETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


# ──────────────────────────────────────────────────────────────────────────────
# One-time-per-sweep span index
# ──────────────────────────────────────────────────────────────────────────────
def build_span_index(rank: np.ndarray, src: np.ndarray, tgt: np.ndarray,
                     w: np.ndarray, n: int, max_span: int) -> Dict:
    """Sort the edges by position span once, so every ``(L, d)`` reads a prefix.

    Parameters
    ----------
    rank : int64 array, shape (n,)
        ``rank[u]`` = current position of node ``u`` in ``[0, n)``.
    src, tgt, w : arrays, shape (m,)
        Edge arrays. Self-loops are dropped here (they can never flip).
    n : int
        Number of nodes.
    max_span : int
        Largest window width ``W = L + d`` any ``(L, d)`` in the grid will use. Edges
        whose endpoints are further than ``max_span - 1`` positions apart can never have
        both endpoints inside a window and are dropped once, here.

    Returns
    -------
    dict with keys ``x``, ``y`` (int64, the lower / upper endpoint POSITION of each kept
    edge), ``s`` (float64, ``+w`` if the edge currently runs backward along the line and
    ``-w`` if it runs forward — the signed contribution of a swap), and ``span``
    (``y - x``), all sorted by ``span`` ascending.

    Notes
    -----
    ``s`` is the *gain* contribution: swapping the two blocks that separate ``x`` from
    ``y`` turns a currently-feedback edge feedforward (``+w``) and a currently-feedforward
    edge feedback (``-w``).
    """
    rank = np.asarray(rank, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    tgt = np.asarray(tgt, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)

    keep = src != tgt
    src, tgt, w = src[keep], tgt[keep], w[keep]

    p = rank[src]
    q = rank[tgt]
    x = np.minimum(p, q)
    y = np.maximum(p, q)
    span = y - x
    # backward edge (p > q) -> the swap wins it (+w); forward edge -> the swap loses it.
    s = np.where(p > q, w, -w)

    lim = int(min(max_span - 1, max(n - 1, 0)))
    sel = (span >= 1) & (span <= lim)
    x, y, s, span = x[sel], y[sel], s[sel], span[sel]

    o = np.argsort(span, kind="stable")
    return dict(x=x[o].copy(), y=y[o].copy(), s=s[o].copy(), span=span[o].copy())


# ──────────────────────────────────────────────────────────────────────────────
# Exact gain of every window start, for one (L, d)
# ──────────────────────────────────────────────────────────────────────────────
def segment_gains(idx: Dict, n: int, L: int, d: int) -> np.ndarray:
    """Exact gain of the adjacent-block swap ``[i, i+L) <-> [i+L, i+L+d)``, for EVERY ``i``.

    Returns
    -------
    float64 array, shape ``(n - L - d + 1,)``
        ``out[i]`` is the exact change in feedforward weight caused by moving the
        segment at positions ``[i, i+L)`` right by ``d`` positions (equivalently, moving
        the segment at ``[i+L, i+L+d)`` left by ``L`` positions). Empty if ``L + d > n``.

    ``L`` or ``d`` equal to 0 is a no-op and yields an all-zero array — a move to the
    segment's current position has gain 0 by definition, never a move.
    """
    L = int(L)
    d = int(d)
    W = L + d
    if L < 0 or d < 0:
        raise ValueError("L and d must be non-negative")
    if W > n or W <= 0:
        return np.zeros(0, dtype=np.float64)
    n_i = n - W + 1
    if L == 0 or d == 0:
        return np.zeros(n_i, dtype=np.float64)

    span = idx["span"]
    k = int(np.searchsorted(span, W - 1, side="right"))
    if k == 0:
        return np.zeros(n_i, dtype=np.float64)

    x = idx["x"][:k]
    y = idx["y"][:k]
    s = idx["s"][:k]

    lo = np.maximum(x - L + 1, y - W + 1)
    np.maximum(lo, 0, out=lo)
    hi = np.minimum(x, y - L)
    np.minimum(hi, n_i - 1, out=hi)
    ok = lo <= hi
    if not ok.any():
        return np.zeros(n_i, dtype=np.float64)
    lo, hi, s = lo[ok], hi[ok], s[ok]

    # difference array: +s at lo, -s at hi+1; prefix sum gives the per-start gain.
    pos = np.concatenate([lo, hi + 1])
    val = np.concatenate([s, -s])
    diff = np.bincount(pos, weights=val, minlength=n_i + 1)
    return np.cumsum(diff[:n_i])


def best_segment_moves(idx: Dict, n: int,
                       seg_lengths: Sequence[int] = DEFAULT_SEG_LENGTHS,
                       offsets: Sequence[int] = DEFAULT_OFFSETS
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best ``(L, d)`` and its exact gain for every window start ``i``.

    Sweeps the whole grid ``seg_lengths x offsets`` and keeps, per start position, the
    single best move. Restricting each start to its own best move keeps the candidate
    list at ``O(n)`` instead of ``O(n·|grid|)``; a move that loses a start to a better
    one is simply re-offered on the next sweep.

    Ties are broken by the FIXED iteration order (``L`` ascending, then ``d`` ascending):
    a later ``(L, d)`` must be STRICTLY better to displace an earlier one, so the
    smallest segment and the shortest travel win. No RNG.

    Returns ``(best_gain float64[n], best_L int64[n], best_d int64[n])``; entries for
    starts that admit no move keep gain 0 and ``L = d = 0``.
    """
    best_gain = np.zeros(n, dtype=np.float64)
    best_L = np.zeros(n, dtype=np.int64)
    best_d = np.zeros(n, dtype=np.int64)
    for L in sorted({int(v) for v in seg_lengths}):
        if L <= 0:
            continue
        for d in sorted({int(v) for v in offsets}):
            if d <= 0 or L + d > n:
                continue
            gains = segment_gains(idx, n, L, d)
            n_i = gains.shape[0]
            better = gains > best_gain[:n_i]
            if not better.any():
                continue
            where = np.flatnonzero(better)
            best_gain[where] = gains[where]
            best_L[where] = L
            best_d[where] = d
    return best_gain, best_L, best_d


# ──────────────────────────────────────────────────────────────────────────────
# Greedy disjoint selection + application
# ──────────────────────────────────────────────────────────────────────────────
def select_disjoint_moves(best_gain: np.ndarray, best_L: np.ndarray,
                          best_d: np.ndarray, n: int, tol: float = 1e-9
                          ) -> List[Tuple[int, int, int, float]]:
    """Greedily pick strictly-improving moves with pairwise DISJOINT windows.

    Disjoint windows compose exactly (see the module docstring), so the realised score
    delta of the selected set is the sum of their gains. Candidates are taken in
    decreasing gain, ties broken by the lowest start index — a fixed rule, no RNG.

    Returns a list of ``(i, L, d, gain)``.
    """
    cand = np.flatnonzero(best_gain > tol)
    if cand.size == 0:
        return []
    # primary key = -gain (ascending => gain descending); secondary = start index.
    order = np.lexsort((cand, -best_gain[cand]))
    taken = np.zeros(n, dtype=bool)
    moves: List[Tuple[int, int, int, float]] = []
    for j in order:
        i = int(cand[j])
        L = int(best_L[i])
        d = int(best_d[i])
        hi = i + L + d
        if taken[i:hi].any():
            continue
        taken[i:hi] = True
        moves.append((i, L, d, float(best_gain[i])))
    return moves


def apply_segment_moves(rank: np.ndarray,
                        moves: Sequence[Tuple[int, int, int, float]]) -> np.ndarray:
    """Apply a set of DISJOINT adjacent-block swaps to a rank vector."""
    rank = np.asarray(rank, dtype=np.int64)
    n = rank.shape[0]
    seq = np.empty(n, dtype=np.int64)          # seq[p] = node at position p
    seq[rank] = np.arange(n, dtype=np.int64)
    new_seq = seq.copy()
    for (i, L, d, _g) in moves:
        new_seq[i:i + d] = seq[i + L:i + L + d]
        new_seq[i + d:i + L + d] = seq[i:i + L]
    new_rank = np.empty(n, dtype=np.int64)
    new_rank[new_seq] = np.arange(n, dtype=np.int64)
    return new_rank


# ──────────────────────────────────────────────────────────────────────────────
# One sweep
# ──────────────────────────────────────────────────────────────────────────────
def segment_sweep(g: GraphData, rank: np.ndarray, *,
                  seg_lengths: Sequence[int] = DEFAULT_SEG_LENGTHS,
                  offsets: Sequence[int] = DEFAULT_OFFSETS,
                  tol: float = 1e-9) -> Tuple[np.ndarray, Dict]:
    """One monotone pass of exact-gain bounded-span segment moves.

    Builds the span index once, sweeps the whole ``(L, d)`` grid, then applies a greedy
    set of disjoint strictly-positive moves. The score cannot decrease.

    Returns ``(new_rank int64[n], info)`` where ``info`` carries
    ``{n_moves, predicted_gain, n_candidates, max_gain}``. ``predicted_gain`` is the EXACT
    score delta (verified against the frozen oracle in the unit tests), so a caller may
    use it without rescoring.
    """
    n = g.n_nodes
    rank = np.asarray(rank, dtype=np.int64)
    lengths = sorted({int(v) for v in seg_lengths if int(v) > 0})
    offs = sorted({int(v) for v in offsets if int(v) > 0})
    if not lengths or not offs:
        return rank.copy(), dict(n_moves=0, predicted_gain=0.0, n_candidates=0,
                                 max_gain=0.0)
    max_span = min(max(lengths) + max(offs), n)
    if max_span < 2:
        return rank.copy(), dict(n_moves=0, predicted_gain=0.0, n_candidates=0,
                                 max_gain=0.0)

    idx = build_span_index(rank, np.asarray(g.src), np.asarray(g.tgt), g.weight,
                           n, max_span)
    best_gain, best_L, best_d = best_segment_moves(idx, n, lengths, offs)
    n_candidates = int((best_gain > tol).sum())
    moves = select_disjoint_moves(best_gain, best_L, best_d, n, tol=tol)
    predicted = float(sum(m[3] for m in moves))
    if not moves:
        return rank.copy(), dict(n_moves=0, predicted_gain=0.0,
                                 n_candidates=n_candidates, max_gain=0.0)
    new_rank = apply_segment_moves(rank, moves)
    return new_rank, dict(n_moves=len(moves), predicted_gain=predicted,
                          n_candidates=n_candidates,
                          max_gain=float(best_gain.max()))


# ──────────────────────────────────────────────────────────────────────────────
# Driver — same register as sift_underrelaxed
# ──────────────────────────────────────────────────────────────────────────────
def segment_refine(g: GraphData, init_rank: np.ndarray, *, max_sweeps: int = 8,
                   seg_lengths: Sequence[int] = DEFAULT_SEG_LENGTHS,
                   offsets: Sequence[int] = DEFAULT_OFFSETS,
                   time_budget_s: Optional[float] = None, tol: float = 1e-9
                   ) -> Tuple[np.ndarray, float, List[Dict]]:
    """Repeated monotone segment sweeps, best-by-oracle.

    Same signature register and return shape as
    :func:`mfas.refine.underrelax.sift_underrelaxed`, so this drops into an alternation
    unchanged. Unlike the sift there is no accept/reject subtlety: each sweep is monotone
    by construction, so the working order IS the best order; the oracle is still consulted
    once per sweep for the log and as a cross-check of the closed-form gain.

    ``time_budget_s`` is checked at a SWEEP boundary only (as in ``sift_underrelaxed`` and
    ``alternate_scc_sift``), so the campaign's run-level wall-clock guard can interrupt
    this stage. A sweep is the atomic unit: the ``(L, d)`` grid is always swept in full,
    so which moves are found never depends on machine speed — only how many sweeps run.

    Returns ``(best_rank int64[n], best_score float, sweep_log)`` where each row is
    ``{sweep, n_moves, n_candidates, predicted_gain, score_pct, wall}``.
    """
    src_o, tgt_o = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight
    rank = np.asarray(init_rank, dtype=np.int64).copy()
    best_score = score_from_order(rank, src_o, tgt_o, g.weight)
    best_rank = rank.copy()

    log: List[Dict] = []
    t_start = time.time()
    for s in range(max_sweeps):
        if time_budget_s is not None and (time.time() - t_start) > time_budget_s:
            break
        t0 = time.time()
        cand_rank, info = segment_sweep(g, rank, seg_lengths=seg_lengths,
                                        offsets=offsets, tol=tol)
        wall = time.time() - t0
        if info["n_moves"] == 0:
            log.append(dict(sweep=s, n_moves=0, n_candidates=info["n_candidates"],
                            predicted_gain=0.0, score_pct=pct(best_score, total),
                            wall=wall))
            break
        rank = cand_rank
        sc = score_from_order(rank, src_o, tgt_o, g.weight)
        if sc > best_score:
            best_score, best_rank = sc, rank.copy()
        log.append(dict(sweep=s, n_moves=info["n_moves"],
                        n_candidates=info["n_candidates"],
                        predicted_gain=info["predicted_gain"],
                        score_pct=pct(sc, total), wall=wall))
    return best_rank, float(best_score), log


def alternate_scc_sift_segment(g: GraphData, init_rank: np.ndarray, *, n_cycles: int,
                               sift_sweeps: int = 2, k_full: int = 2, alpha: float = 0.7,
                               min_block: int = 32,
                               split_fracs: Sequence[float] = DEFAULT_SPLIT_FRACS,
                               seg_sweeps: int = 1,
                               seg_lengths: Sequence[int] = DEFAULT_SEG_LENGTHS,
                               offsets: Sequence[int] = DEFAULT_OFFSETS,
                               time_budget_s: Optional[float] = None
                               ) -> Tuple[np.ndarray, float, List[Dict]]:
    """Champion stage 4 (block refine + short sift) with the segment sweep interleaved.

    Mirrors :func:`mfas.refine.scc_recursive.alternate_scc_sift` exactly, adding a third
    stage per cycle. The log records the score after EACH of the three stages, so each
    move class's increment stays separately attributable and no credit is double-counted.

    Returns ``(best_rank, best_score, log)``; the returned order is the best the frozen
    oracle has scored, so it can never be worse than ``init_rank``.

    Oracle-call parity with ``alternate_scc_sift``: the cycle's starting score ``prev`` is
    CARRIED from the previous cycle's post-segment score rather than re-measured, because
    ``rank`` is not touched between the two points. Re-measuring would add one full rescore
    per cycle (5.66 M edge comparisons on connectome) of pure bookkeeping compute, and this
    variant's whole claim is that its gain is a move class and not extra CPU.
    """
    src_o, tgt_o = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight
    rank = np.asarray(init_rank, dtype=np.int64).copy()
    best_score = score_from_order(rank, src_o, tgt_o, g.weight)
    best_rank = rank.copy()
    prev = best_score

    log: List[Dict] = []
    t0 = time.time()
    for c in range(n_cycles):
        if time_budget_s is not None and (time.time() - t0) > time_budget_s:
            break

        ta = time.time()
        rank = scc_recursive_refine(g, rank, min_block=min_block,
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
                                            time_budget_s=left)
        t_sift = time.time() - tb
        if s_sift > best_score:
            best_score, best_rank = s_sift, rank.copy()

        tc = time.time()
        left = None if time_budget_s is None else max(
            0.0, time_budget_s - (time.time() - t0))
        rank, s_seg, seg_log = segment_refine(g, rank, max_sweeps=seg_sweeps,
                                              seg_lengths=seg_lengths, offsets=offsets,
                                              time_budget_s=left)
        t_seg = time.time() - tc
        if s_seg > best_score:
            best_score, best_rank = s_seg, rank.copy()

        log.append(dict(cycle=c, split_frac=split_fracs[c % len(split_fracs)],
                        before_pct=pct(prev, total),
                        after_scc_pct=pct(s_scc, total),
                        after_sift_pct=pct(s_sift, total),
                        after_seg_pct=pct(s_seg, total),
                        d_scc_pp=pct(s_scc, total) - pct(prev, total),
                        d_sift_pp=pct(s_sift, total) - pct(s_scc, total),
                        d_seg_pp=pct(s_seg, total) - pct(s_sift, total),
                        seg_moves=int(sum(r["n_moves"] for r in seg_log)),
                        best_pct=pct(best_score, total),
                        t_scc_s=t_scc, t_sift_s=t_sift, t_seg_s=t_seg,
                        cum_wall_s=time.time() - t0))
        # ``rank`` is not touched between here and the next cycle's first stage, so the
        # next cycle's starting score is exactly this one's post-segment score.
        prev = s_seg
    return best_rank, float(best_score), log


# ──────────────────────────────────────────────────────────────────────────────
# Small O(m) reference kernels — UNIT TESTS ONLY
# ──────────────────────────────────────────────────────────────────────────────
def apply_segment_move_ref(rank: np.ndarray, i: int, L: int, d: int) -> np.ndarray:
    """Reference application of ONE right segment move (tests only)."""
    return apply_segment_moves(rank, [(int(i), int(L), int(d), 0.0)])


def segment_move_gain_ref(g: GraphData, rank: np.ndarray, i: int, L: int, d: int
                          ) -> float:
    """Reference exact gain of one move, by direct enumeration of cross edges (tests only).

    Independent of the vectorised difference-array kernel: it filters the edge list for
    ``S``–``J`` cross edges and sums the signed weights directly.
    """
    rank = np.asarray(rank, dtype=np.int64)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    p, q = rank[src], rank[tgt]
    in_s = lambda z: (z >= i) & (z < i + L)          # noqa: E731
    in_j = lambda z: (z >= i + L) & (z < i + L + d)  # noqa: E731
    lose = in_s(p) & in_j(q)      # S -> J was feedforward, becomes feedback
    win = in_j(p) & in_s(q)       # J -> S was feedback, becomes feedforward
    return float(w[win].sum() - w[lose].sum())
