"""Full-range exact-gain single-node re-insertion ("sift") — pure vectorized NumPy.

This is the leakage-safe discrete refiner used by experiment H30. After a continuous
optimizer (Rocket) produces an ordering, ``sift`` repeatedly moves nodes to their
EXACT feedforward-weight-maximising rank (sweeping the WHOLE line, not a bounded
window) and accepts a whole candidate vector only if the frozen oracle says it
strictly improves the best score (best-by-oracle).

Exact-gain insertion math (validated by brute force)
----------------------------------------------------
Consider node ``u`` at current rank ``p`` in a rank vector ``rank`` over ``n`` nodes.
Remove ``u`` (the remaining ``n-1`` nodes keep their relative order) and consider
re-inserting ``u`` at *gap* ``g`` (``g`` in ``[0, n-1]``: gap 0 = before everything,
gap ``n-1`` = after everything). For each other node ``x`` its *reduced rank* in the
order-without-``u`` is

    q(x) = rank[x]      if rank[x] < p
    q(x) = rank[x] - 1  if rank[x] > p

The feedforward weight contributed by ``u``'s incident edges, as a function of ``g``:

* an out-edge ``u -> v`` (``u`` wants to come BEFORE ``v``) is feedforward iff
  ``g <= q(v)``  -> contributes ``w`` for ``g in [0, q(v)]``;
* an in-edge ``x -> u`` (``u`` wants to come AFTER ``x``) is feedforward iff
  ``g > q(x)``   -> contributes ``w`` for ``g in [q(x)+1, n-1]``.

So ``total_u(0) = Σ_{u->v} w`` (= ``base[u]`` = total out-weight), and at each
breakpoint ``b = q+1`` the value jumps by ``delta``: an out-edge contributes
``delta = -w`` at ``b = q(v)+1`` (it stops being feedforward), an in-edge contributes
``delta = +w`` at ``b = q(x)+1`` (it starts being feedforward). ``total_u`` is
piecewise-constant in ``g`` (cumulative sum of base + deltas at breakpoints), so its
maximum is attained either at ``g = 0`` or at one of the breakpoints. Self-loops
(``u == v``) contribute nothing and are dropped once up front.

``jacobi_best_gaps`` computes, for EVERY node at once, the best gap and the gain over
its current value, fully vectorized: it builds one event per incident edge
``(node=u, b=q+1, delta)``, sorts by the combined key ``u*(n+2)+b`` (a single
``argsort``; safe for ``u*(n+2)+b < 2^63`` at our ``n``), AGGREGATES deltas per
``(u, b)`` before the per-node segmented maximum (mandatory: a partial running sum
inside a tied-``b`` group must never be selected), then runs a segmented prefix sum
(over base + aggregated deltas) and a segmented maximum to get each node's best value.

Leakage-safety
--------------
Every move is decided purely from the input edge weights and the current ranks
(the closed-form gain above). The frozen oracle (:func:`mfas.metrics.score_from_order`)
is consulted only to accept/reject a whole candidate rank vector after it is built
(exactly as the baseline tracks best-by-oracle); it never enters a move choice, a
loss, or any dataset-specific branch, and never reads ``data/best_solution``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..io import GraphData
from ..metrics import pct, score_from_order

__all__ = [
    "build_sift_edges",
    "jacobi_best_gaps",
    "jacobi_rebuild",
    "sift",
    # small references for unit tests only
    "build_adj",
    "best_gap_for_node_ref",
    "sift_gauss_seidel_ref",
]


# ──────────────────────────────────────────────────────────────────────────────
# One-time edge preparation
# ──────────────────────────────────────────────────────────────────────────────
def build_sift_edges(g: GraphData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(src, tgt, w)`` as ``int64/int64/float64`` with self-loops dropped.

    Self-loops (``src == tgt``) contribute nothing to any insertion gain and are
    removed once here so the vectorized kernel never has to special-case them.
    """
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    keep = src != tgt
    return src[keep].copy(), tgt[keep].copy(), w[keep].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized exact best-gap / gain for every node (Jacobi)
# ──────────────────────────────────────────────────────────────────────────────
def jacobi_best_gaps(rank: np.ndarray, src: np.ndarray, tgt: np.ndarray,
                     w: np.ndarray, n: int, *, tie_break: str = "first"
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Exact best insertion gap and gain for EVERY node, from the FIXED ``rank``.

    Parameters
    ----------
    rank : int64 array, shape (n,)
        ``rank[u]`` = current rank of node ``u`` in ``[0, n)``.
    src, tgt, w : arrays, shape (m,)
        Edge arrays with self-loops already dropped (see :func:`build_sift_edges`).
    n : int
        Number of nodes.
    tie_break : {"first", "mindisp"}, keyword-only
        Which gap to return when the profile's maximum is attained on an INTERVAL
        (a plateau), which it usually is: the profile has at most ``deg(u)``
        breakpoints over ``n`` gaps. Both choices have the SAME exact gain -- this
        selects a point of the same argmax set and never changes ``gain``.

        * ``"first"`` (default, and the behaviour every champion through H64 was
          measured with): the smallest maximizing breakpoint, with gap 0 winning any
          tie against it. Both are LEFTWARD, so a node whose plateau lies to its left
          is transported to the FAR edge while one whose plateau lies to its right
          lands on the NEAR edge.
        * ``"mindisp"`` (H73): the maximizing gap closest to the node's current rank,
          ties broken toward the smaller gap. Deterministic, RNG-free, same asymptotic
          cost (one extra segmented min).

        The default MUST stay ``"first"`` -- ``tests/test_H73_tiebreak.py`` pins
        H42/H64 bit-reproducibility on it.

    Returns
    -------
    best_gap : int64 array, shape (n,)
        The gap that maximises ``total_u(g)`` for each node.
    gain : float64 array, shape (n,)
        ``best_val - cur_val`` for each node (>= 0 by construction; the current gap
        is always a candidate). Isolated nodes get gain 0.
    """
    rank = np.asarray(rank, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    tgt = np.asarray(tgt, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    m = src.shape[0]

    # base[u] = Σ out-weight = total_u(0).
    base = np.zeros(n, dtype=np.float64)
    np.add.at(base, src, w)

    best_gap = np.zeros(n, dtype=np.int64)   # gap 0 is the default candidate
    best_val = base.copy()                   # value at gap 0
    # cur_val (value at the node's CURRENT gap p) starts from base too; updated below
    # as the last breakpoint with b <= p.
    cur_val = base.copy()

    if m == 0:
        return best_gap, np.zeros(n, dtype=np.float64)

    p = rank  # current rank == current gap of each node in the order-without-itself

    # ── Build one event per incident edge: (node u, breakpoint b = q+1, delta) ──
    # out-edge u->v: node = u (=src), neighbour = v (=tgt), delta = -w, breakpoint q(v)+1.
    out_u = src
    out_nb = rank[tgt]
    out_p = p[src]
    out_q = np.where(out_nb < out_p, out_nb, out_nb - 1)
    out_b = out_q + 1
    out_delta = -w

    # in-edge x->u: node = u (=tgt), neighbour = x (=src), delta = +w, breakpoint q(x)+1.
    in_u = tgt
    in_nb = rank[src]
    in_p = p[tgt]
    in_q = np.where(in_nb < in_p, in_nb, in_nb - 1)
    in_b = in_q + 1
    in_delta = w

    ev_u = np.concatenate([out_u, in_u])
    ev_b = np.concatenate([out_b, in_b])
    ev_delta = np.concatenate([out_delta, in_delta])

    # Breakpoints b are in [1, n-1] (q in [0, n-2]); clip defensively.
    np.clip(ev_b, 0, n, out=ev_b)

    # ── Single argsort on combined key u*(n+2)+b (safe < 2^63 at our n). ─────────
    key = ev_u * np.int64(n + 2) + ev_b
    order = np.argsort(key, kind="stable")
    su = ev_u[order]
    sb = ev_b[order]
    sd = ev_delta[order]

    # ── Aggregate deltas per (u, b) BEFORE any segmented max (mandatory). ────────
    # A new (u, b) group starts where (su, sb) changes.
    new_grp = np.empty(su.shape[0], dtype=bool)
    new_grp[0] = True
    new_grp[1:] = (su[1:] != su[:-1]) | (sb[1:] != sb[:-1])
    grp_idx = np.cumsum(new_grp) - 1            # dense group id per event
    n_grp = int(grp_idx[-1]) + 1
    grp_u = su[new_grp]                          # node of each group
    grp_b = sb[new_grp]                          # breakpoint of each group
    grp_delta = np.zeros(n_grp, dtype=np.float64)
    np.add.at(grp_delta, grp_idx, sd)           # aggregated delta per (u, b)

    # ── Per-node segmented prefix sum of (base + aggregated deltas). ─────────────
    # Within a node's groups (already sorted by b ascending), value at breakpoint b_k
    # is base[u] + Σ_{j<=k} grp_delta[j]. Take running cumsum then subtract the
    # node's prefix offset so each node restarts at its own base.
    node_start = np.empty(n_grp, dtype=bool)
    node_start[0] = True
    node_start[1:] = grp_u[1:] != grp_u[:-1]
    seg_id = np.cumsum(node_start) - 1           # dense per-NODE segment id (over groups)

    csum = np.cumsum(grp_delta)
    # prefix value just before each node's first group (0 for the first node).
    seg_first_pos = np.flatnonzero(node_start)
    pre = np.zeros(n_grp, dtype=np.float64)
    # value before the segment = csum at (first_pos - 1); first node -> 0.
    pre_per_seg = np.empty(seg_first_pos.shape[0], dtype=np.float64)
    pre_per_seg[0] = 0.0
    pre_per_seg[1:] = csum[seg_first_pos[1:] - 1]
    pre = pre_per_seg[seg_id]
    seg_base = base[grp_u]                        # base value of each group's node
    grp_val = seg_base + (csum - pre)             # total_u at breakpoint grp_b

    # ── Segmented maximum over each node's groups -> best breakpoint value. ──────
    seg_max = np.maximum.reduceat(grp_val, seg_first_pos)
    # argmax within each segment for best_gap. reduceat gives max only; find the
    # FIRST position achieving the segment max (ties -> smallest b, fine).
    seg_max_per_grp = seg_max[seg_id]
    is_max = grp_val >= seg_max_per_grp - 0.0     # exact equality on the segment max
    # first index (in group space) per segment where grp_val == seg_max.
    # Build candidate positions; take min position per segment among maxima.
    big = np.int64(n_grp + 1)
    pos = np.arange(n_grp, dtype=np.int64)
    cand = np.where(is_max, pos, big)
    seg_argmax_pos = np.minimum.reduceat(cand, seg_first_pos)  # first max position
    seg_best_b = grp_b[seg_argmax_pos]
    seg_node = grp_u[seg_first_pos]

    # ── cur_val: value at the CURRENT gap p[u] = last breakpoint with b <= p. ────
    # Within each node's sorted-by-b groups, the value just below/at p is the cumsum
    # through the last group with grp_b <= p[u]; if none, it's base.
    node_p = p[grp_u]                             # current gap for each group's node
    le = grp_b <= node_p                          # groups whose breakpoint is <= p
    # value AT current gap = base + Σ deltas of groups with b <= p (per node).
    cur_contrib_grp = np.where(le, grp_delta, 0.0)
    csum_le = np.cumsum(cur_contrib_grp)
    pre_le = np.empty(seg_first_pos.shape[0], dtype=np.float64)
    pre_le[0] = 0.0
    pre_le[1:] = csum_le[seg_first_pos[1:] - 1]
    pre_le_b = pre_le[seg_id]
    node_cur = base[grp_u] + (csum_le - pre_le_b)
    # the per-node current value is the value after the LAST group of the node.
    seg_last_pos = np.empty(seg_first_pos.shape[0], dtype=np.int64)
    seg_last_pos[:-1] = seg_first_pos[1:] - 1
    seg_last_pos[-1] = n_grp - 1
    seg_cur = node_cur[seg_last_pos]

    # ── Write per-node results; nodes with no events keep best_gap=0, gain=0. ────
    # For nodes that have events, compare gap-0 (base) vs the best breakpoint.
    nodes = seg_node
    # candidate at gap 0:
    gap0_val = base[nodes]
    use_bp = seg_max > gap0_val
    chosen_val = np.where(use_bp, seg_max, gap0_val)
    chosen_gap = np.where(use_bp, seg_best_b, 0).astype(np.int64)

    if tie_break == "mindisp":
        # ── H73: pick the maximizing gap NEAREST the node's current rank. ───────────
        # Materialise the plateau as a union of gap intervals. Breakpoint group k owns
        # the gaps [grp_b[k], next_b - 1]; the last group of a node owns [grp_b, n-1];
        # and the gap-0 interval [0, b_first - 1] carries the value `base`.
        is_last = np.empty(n_grp, dtype=bool)
        is_last[:-1] = node_start[1:]
        is_last[-1] = True
        nxt = np.empty(n_grp, dtype=np.int64)
        nxt[:-1] = grp_b[1:]
        nxt[-1] = n
        lo_i = grp_b
        hi_i = np.where(is_last, n - 1, nxt - 1)

        # `chosen_val` is the OVERALL max per node (breakpoints and gap 0 together).
        overall = chosen_val[seg_id]
        p_grp = p[grp_u]
        closest = np.minimum(np.maximum(p_grp, lo_i), hi_i)
        # Lexicographic key: minimise |displacement| first, then the gap index. Both
        # fit int64 comfortably (dist, gap <= n).
        sent = np.int64(1) << np.int64(62)
        key = np.where(grp_val == overall,
                       np.abs(closest - p_grp) * np.int64(n + 1) + closest, sent)
        seg_key = np.minimum.reduceat(key, seg_first_pos)

        b_first = grp_b[seg_first_pos]
        p_seg = p[nodes]
        base_is_max = (b_first >= 1) & (gap0_val == chosen_val)
        c_base = np.minimum(p_seg, np.maximum(b_first - 1, 0))
        seg_key = np.minimum(seg_key,
                             np.where(base_is_max,
                                      np.abs(c_base - p_seg) * np.int64(n + 1) + c_base,
                                      sent))
        chosen_gap = (seg_key % np.int64(n + 1)).astype(np.int64)
    elif tie_break != "first":
        raise ValueError("tie_break must be 'first' or 'mindisp', got %r" % (tie_break,))

    best_val[nodes] = chosen_val
    best_gap[nodes] = chosen_gap
    cur_val[nodes] = seg_cur

    gain = best_val - cur_val
    # Numerical guard: current gap is always a candidate, so gain >= 0; clamp tiny
    # negative values from float arithmetic.
    gain[gain < 0] = 0.0
    return best_gap, gain


def jacobi_rebuild(rank: np.ndarray, best_gap: np.ndarray, gain: np.ndarray,
                   n: int, tol: float = 1e-9) -> np.ndarray:
    """Rebuild a rank vector after a Jacobi sweep (all movers applied at once).

    Movers (``gain > tol``) are given the fractional key ``best_gap - 0.5`` so they
    sort into the requested gap; non-movers keep their current ``rank``. The new
    rank vector is ``argsort(argsort(key))`` (stable), an O(m log m + n log n)
    operation. Because many nodes move simultaneously (Jacobi), the realised order
    is an approximation of every node's individual optimum; the oracle accept/reject
    in :func:`sift` guarantees the kept order never regresses.
    """
    rank = np.asarray(rank, dtype=np.int64)
    best_gap = np.asarray(best_gap, dtype=np.int64)
    gain = np.asarray(gain, dtype=np.float64)
    key = rank.astype(np.float64).copy()
    movers = gain > tol
    key[movers] = best_gap[movers].astype(np.float64) - 0.5
    new_rank = np.argsort(np.argsort(key, kind="stable"), kind="stable").astype(np.int64)
    return new_rank


# ──────────────────────────────────────────────────────────────────────────────
# Sift driver (Jacobi loop, best-by-oracle)
# ──────────────────────────────────────────────────────────────────────────────
def sift(g: GraphData, init_rank: np.ndarray, *, max_sweeps: int = 12,
         time_budget_s: Optional[float] = None, tol: float = 1e-9
         ) -> Tuple[np.ndarray, float, List[Dict]]:
    """Full-range exact-gain Jacobi node re-insertion, best-by-oracle.

    The *working* order advances unconditionally each sweep (Jacobi dynamics: every
    node moves toward its exact-optimal gap simultaneously), exactly as the Rocket
    optimizer's iterate advances unconditionally. A SEPARATE best-by-oracle tracker
    keeps the strictly-best candidate the oracle has ever scored and is what is
    returned -- so the result can never regress below the initial order. Decoupling
    the working iterate from the tracked best is essential for Jacobi: on dense cyclic
    cores a single simultaneous sweep can transiently OVERSHOOT (dip then climb), so
    stopping on the first non-improving sweep (the Gauss-Seidel convention) would
    abandon the gain. We instead run to ``max_sweeps`` / ``time_budget_s`` and stop
    early only when the working order reaches a stationary point (no movers, a true
    fixed point) -- never on a transient dip.

    Each move is chosen from input edge weights + current ranks (the closed-form gain);
    the frozen oracle is consulted ONLY to accept/reject whole candidate vectors.

    Returns ``(best_rank int64[n], best_score float, sweep_log)`` where each
    ``sweep_log`` row is ``{sweep, candidate_pct, accepted, n_movers, wall}``.
    """
    import time

    n = g.n_nodes
    src_o = np.asarray(g.src)
    tgt_o = np.asarray(g.tgt)
    src, tgt, w = build_sift_edges(g)

    work_rank = np.asarray(init_rank, dtype=np.int64).copy()
    best_rank = work_rank.copy()
    best_score = score_from_order(best_rank, src_o, tgt_o, g.weight)
    total = g.total_weight

    sweep_log: List[Dict] = []
    t_start = time.time()
    for s in range(max_sweeps):
        if time_budget_s is not None and (time.time() - t_start) > time_budget_s:
            break
        t0 = time.time()
        best_gap, gain = jacobi_best_gaps(work_rank, src, tgt, w, n)
        n_movers = int((gain > tol).sum())
        cand_rank = jacobi_rebuild(work_rank, best_gap, gain, n, tol=tol)
        cand_score = score_from_order(cand_rank, src_o, tgt_o, g.weight)
        # Working iterate advances unconditionally (Jacobi dynamics).
        work_rank = cand_rank
        # Best-by-oracle: keep the strictly-best candidate ever seen.
        accepted = cand_score > best_score
        if accepted:
            best_rank = cand_rank.copy()
            best_score = cand_score
        wall = time.time() - t0
        sweep_log.append(dict(sweep=s, candidate_pct=pct(cand_score, total),
                              accepted=bool(accepted), n_movers=n_movers, wall=wall))
        # Stop only at a true fixed point (no node wants to move).
        if n_movers == 0:
            break
    return best_rank, float(best_score), sweep_log


# ──────────────────────────────────────────────────────────────────────────────
# Small O(n)/O(deg) reference kernels — UNIT TESTS ONLY (re-homed prototype logic)
# ──────────────────────────────────────────────────────────────────────────────
def build_adj(g: GraphData) -> Dict:
    """CSR adjacency for the O(deg) reference kernels (tests only)."""
    n = g.n_nodes
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    o = np.argsort(src, kind="stable")
    out_start = np.searchsorted(src[o], np.arange(n + 1))
    out_nbr, out_w = tgt[o], w[o]
    i = np.argsort(tgt, kind="stable")
    in_start = np.searchsorted(tgt[i], np.arange(n + 1))
    in_nbr, in_w = src[i], w[i]
    return dict(n=n, out_start=out_start, out_nbr=out_nbr, out_w=out_w,
                in_start=in_start, in_nbr=in_nbr, in_w=in_w)


def best_gap_for_node_ref(u: int, rank: np.ndarray, adj: Dict
                          ) -> Tuple[int, float]:
    """Reference O(deg) exact best gap + gain for node ``u`` (difference array).

    Re-homed from the ``dr_tmp`` prototype kernel; for unit-test cross-checks at
    small ``n`` only. Self-loops are skipped.
    """
    n = adj["n"]
    p = int(rank[u])
    D = np.zeros(n + 1, dtype=np.float64)
    os_, oe = adj["out_start"][u], adj["out_start"][u + 1]
    for k in range(os_, oe):
        v = int(adj["out_nbr"][k])
        if v == u:
            continue
        qv = rank[v] if rank[v] < p else rank[v] - 1
        D[0] += adj["out_w"][k]
        D[qv + 1] -= adj["out_w"][k]
    is_, ie = adj["in_start"][u], adj["in_start"][u + 1]
    for k in range(is_, ie):
        x = int(adj["in_nbr"][k])
        if x == u:
            continue
        qx = rank[x] if rank[x] < p else rank[x] - 1
        D[qx + 1] += adj["in_w"][k]
    total = np.cumsum(D[:n])
    cur = total[p]
    g = int(np.argmax(total))
    return g, float(total[g] - cur)


def _order_from_rank(rank: np.ndarray) -> np.ndarray:
    n = rank.shape[0]
    seq = np.empty(n, dtype=np.int64)
    seq[rank] = np.arange(n, dtype=np.int64)
    return seq


def _move_node(seq: np.ndarray, rank: np.ndarray, u: int, new_gap: int
               ) -> np.ndarray:
    p = int(rank[u])
    seq = np.delete(seq, p)
    seq = np.insert(seq, new_gap, u)
    rank[seq] = np.arange(seq.shape[0], dtype=np.int64)
    return seq


def sift_gauss_seidel_ref(g: GraphData, init_rank: np.ndarray, adj: Dict,
                          max_sweeps: int = 12, seed: int = 0
                          ) -> Tuple[np.ndarray, float]:
    """Reference Gauss-Seidel sift (one node moved at a time, O(n)/move).

    Re-homed prototype kernel for cross-checks / the prototype script. Returns
    ``(rank, pct)``. Leakage-safe: moves use input weights + current ranks only.
    """
    rng = np.random.RandomState(seed)
    rank = np.asarray(init_rank, dtype=np.int64).copy()
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    n = adj["n"]
    for _ in range(max_sweeps):
        seq = _order_from_rank(rank)
        nodes = np.arange(n)
        rng.shuffle(nodes)
        moved = 0
        for u in nodes:
            bg, gain = best_gap_for_node_ref(int(u), rank, adj)
            if gain > 1e-9 and bg != rank[u]:
                seq = _move_node(seq, rank, int(u), bg)
                moved += 1
        if moved == 0:
            break
    sc = score_from_order(rank, src, tgt, g.weight)
    return rank, pct(sc, g.total_weight)
