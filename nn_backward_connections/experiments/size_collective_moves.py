"""S1 + S2 opportunity-sizing — can COLLECTIVE (multi-node) discrete moves beat the
1-opt fixed point that H35's sift converges to?

DIAGNOSTIC / SIZING ONLY. This is the go/no-go gate for the roadmap's ``A-SCC`` item,
following the H09 and H22 precedent (size the opportunity *before* building a variant).
It runs no registered variant and writes NOTHING to ``results/``; its output goes to
``experiments/outputs/collective_moves_sizing.json``.

Leakage: NONE. This script never reads ``data/best_solution``. It reads only (a) the
input graph and (b) our own logged H35 orderings (``results/*-H35-*_positions.npy``),
and scores every candidate ordering with the frozen oracle ``mfas.metrics``.

Background
----------
H35's under-relaxed sift converges to a **1-opt fixed point**: every node already sits
at the rank that maximizes the feedforward weight of its incident edges *given all
others fixed*. Whatever remains of the ~0.70 pp gap to the challenge SOTA therefore
requires a move class that relocates **more than one node at a time**. Vahidi 2025
(arXiv:2506.13799) reaches 84.61% with exactly such moves. This script sizes the two
that matter, both of them exact-gain and leakage-safe:

  **S1 — paired heavy-backward-edge relocation (Vahidi Algorithm 2).**
      For a backward edge ``(u, v)`` (rank[u] > rank[v]) let the span be the nodes
      strictly between them, ``[v, n_1, ..., n_t, u]``. Try every split ``r`` of
      ``[n_1..n_r, u, v, n_{r+1}..n_t]`` and take the best. This makes ``u`` and ``v``
      adjacent (fixing the heavy edge) while placing the *pair* optimally — a move no
      sequence of single-node sift steps can reach without first going downhill.

  **S2 — block-SCC condensation refinement (Vahidi Algorithm 3).**
      Cut the current order into consecutive rank blocks of size ``s``; the induced
      subgraph of a block is NOT strongly connected even though the whole graph is
      92.8% one giant SCC. Topologically sort the block's condensation (turns every
      inter-sub-SCC edge feedforward) and exactly re-solve sub-SCCs of size <= 10.

Why the gains are exactly computable (the contiguous-interval lemma)
--------------------------------------------------------------------
Both moves permute nodes **within a contiguous interval of ranks**. Any edge with at
least one endpoint outside such an interval keeps its orientation, because every node
of the interval stays inside the same set of rank slots. So each move's gain is a
purely intra-interval quantity — no global rescoring is needed to evaluate it. Every
number below is nevertheless cross-checked against the frozen oracle on the whole
graph (``assert accumulated_gain == oracle(after) - oracle(before)``); if that
assertion fails the script has a bug and no conclusion may be drawn.

Attribution control (``--sift-first``, IMPORTANT)
--------------------------------------------------
H35's returned order is **not** a single-node (1-opt) fixed point — its sweep cap can
bite before convergence (163 movers remain on connectome, 608 on microns, 0 on mouse).
Without a control, part of any "collective move" gain is really just *more of H35's own
sift*. ``--sift-first`` therefore runs :func:`mfas.refine.underrelax.sift_underrelaxed`
to a true fixed point (or the sweep cap) **before** sizing, and reports that recovery
separately, so the number credited to S1/S2 is genuinely outside the single-node class.
Cite the ``--sift-first`` numbers for attribution claims.

Deliberate conservatism (disclose when citing)
----------------------------------------------
* S1 processes only the top-``K`` backward edges by weight and does **not** re-insert
  newly created backward edges into the queue (Vahidi's Alg-2 line 29), nor does it
  implement his fallback greedy moves. It therefore **under**-states Alg 2: on the
  connectome, ``K=200,000`` covers only ~50% of the backward weight and a full-K pass
  gains ~2.5x more. Treat ``--top-k`` as a declared budget knob, not a property of Alg 2.
* S2 uses contiguous rank blocks over *all* nodes rather than blocks of the giant
  SCC's node list. This is the formulation for which the interval lemma is exactly
  true; with a 92.82% giant SCC the two are nearly identical.
* The iteration's block size is the argmax of the single-pass S2 grid measured on the
  *same* order it then iterates — mildly optimistic, disclose it.

Input dependency (re-running from a clean checkout)
-----------------------------------------------------
``results/*_positions.npy`` is **gitignored**, so this probe cannot run on a fresh clone
until the H35 orders are regenerated::

    for S in 42 123 999; do for D in connectome microns mouse; do \
      python -m eval.run_variant --exp H35 --dataset $D --seed $S --out results/ \
        --role implement --device auto; done; done

Usage
-----
    /opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python \
        experiments/size_collective_moves.py [--datasets connectome,microns,mouse]
        [--seed 42] [--top-k N] [--rounds N] [--sift-first] [--out NAME.json]
"""
from __future__ import annotations

import argparse
import glob
import heapq
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mfas import io                                      # noqa: E402
from mfas.metrics import pct, score_from_positions       # noqa: E402

RESULTS = _ROOT / "results"
OUT_DIR = _ROOT / "experiments" / "outputs"

# ── sizing configuration (no magic numbers below this point) ────────────────────
CONFIG = {
    "seed": 42,                       # which logged H35 order to size from
    "s1_top_k": 20000,                # backward edges (by weight) examined per dataset
    "s1_checkpoints": [100, 500, 1000, 2000, 5000, 10000, 20000,
                       50000, 100000, 200000, 500000, 1200000],
    "sift_first_max_sweeps": 60,      # --sift-first: cap for the 1-opt attribution control
    "sift_first_time_budget_s": 420,  # ...and its wall-clock bound (microns sweeps are slow)
    "s2_block_sizes": [64, 256, 1024, 4096, 16384],
    "s2_small_scc_max": 10,           # exact bitmask DP for sub-SCCs up to this size
    "verify_every": 2000,             # oracle cross-check cadence inside the S1 loop
    "iter_rounds": 6,                 # alternating S1<->S2 rounds (does the gain compound?)
}


# ── helpers ─────────────────────────────────────────────────────────────────────
def git_commit() -> str:
    """Current HEAD (with a ``+dirty`` marker), for provenance."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT).decode().strip()
        return h + ("+dirty" if dirty else "")
    except Exception:                                     # pragma: no cover
        return "unknown"


def latest_h35_positions(dataset: str, seed: int) -> Path:
    """Newest logged H35 positions file for ``dataset``/``seed``."""
    hits = sorted(glob.glob(str(RESULTS / f"*-H35-{dataset}-s{seed}-*_positions.npy")))
    if not hits:
        raise FileNotFoundError(f"no H35 positions for {dataset} seed {seed}")
    return Path(hits[-1])


def rank_of(positions: np.ndarray) -> np.ndarray:
    """``rank[node]`` = 0-based slot on the line (stable double argsort)."""
    return np.argsort(np.argsort(positions, kind="stable"), kind="stable").astype(np.int64)


def group_by_key(key: np.ndarray, n_keys: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(order, starts)`` so that group ``k`` is ``order[starts[k]:starts[k+1]]``."""
    order = np.argsort(key, kind="stable")
    starts = np.zeros(n_keys + 1, dtype=np.int64)
    np.cumsum(np.bincount(key, minlength=n_keys), out=starts[1:])
    return order, starts


def build_adjacency(src: np.ndarray, tgt: np.ndarray, w: np.ndarray, n: int
                    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """CSR-style out- and in-adjacency (self-loops already dropped by the caller)."""
    def csr(key: np.ndarray, other: np.ndarray) -> Dict[str, np.ndarray]:
        order, ptr = group_by_key(key, n)
        return {"ptr": ptr, "idx": other[order], "w": w[order].astype(np.float64)}
    return csr(src, tgt), csr(tgt, src)


# ── S1 — paired heavy-backward-edge relocation (Vahidi Algorithm 2) ─────────────
def _pair_gain_curve(u: int, v: int, lo: int, hi: int,
                     out_adj: Dict, in_adj: Dict, rank: np.ndarray) -> np.ndarray:
    """Exact gain of ``[n_1..n_r, u, v, n_{r+1}..n_t]`` for every split ``r``.

    Derivation (edges not incident to ``u`` or ``v`` keep their orientation, because
    the nodes ``n_i`` keep their relative order and the interval's rank slots are
    unchanged; edges leaving the interval are covered by the contiguous-interval
    lemma in the module docstring)::

        gain(r) = [w(u->v) - w(v->u)]
                + sum_{i >  r} [ w(u->n_i) - w(n_i->u) ]      # u lands before n_{r+1}
                + sum_{i <= r} [ w(n_i->v) - w(v->n_i) ]      # v lands after  n_r

    Returns ``gain`` indexed by ``r = 0..t``.
    """
    t = hi - lo - 1
    width = t + 2
    a = np.zeros(width, dtype=np.float64)   # a[i] = w(u->n_i) - w(n_i->u), i = rank - lo
    b = np.zeros(width, dtype=np.float64)   # b[i] = w(n_i->v) - w(v->n_i)
    partner_w = {"u_to_v": 0.0, "v_to_u": 0.0}

    def scatter(node: int, partner: int, into: np.ndarray,
                sign_out: float, sign_in: float) -> None:
        for adj, sign, is_out in ((out_adj, sign_out, True), (in_adj, sign_in, False)):
            s, e = adj["ptr"][node], adj["ptr"][node + 1]
            if s == e:
                continue
            nb, wt = adj["idx"][s:e], adj["w"][s:e]
            r_nb = rank[nb]
            inside = (r_nb > lo) & (r_nb < hi)
            if inside.any():
                into += np.bincount(r_nb[inside] - lo,
                                    weights=sign * wt[inside], minlength=width)[:width]
            hit = nb == partner
            if hit.any():
                key = "u_to_v" if (is_out and node == u) or (not is_out and node == v) \
                    else "v_to_u"
                partner_w[key] = float(wt[hit].sum())

    scatter(u, v, a, +1.0, -1.0)
    scatter(v, u, b, -1.0, +1.0)
    C = partner_w["u_to_v"] - partner_w["v_to_u"]

    cum_a = np.concatenate(([0.0], np.cumsum(a[1:t + 1])))   # cum_a[r] = sum_{i<=r} a_i
    cum_b = np.concatenate(([0.0], np.cumsum(b[1:t + 1])))
    return C + (cum_a[-1] - cum_a) + cum_b


def s1_pass(cur: np.ndarray, seq: np.ndarray, src: np.ndarray, tgt: np.ndarray,
            w: np.ndarray, out_adj: Dict, in_adj: Dict, top_k: int
            ) -> Tuple[float, int, List[int]]:
    """One greedy sweep of the paired move; mutates ``cur``/``seq`` in place.

    Candidates are the top-``top_k`` **currently** backward edges by weight, so calling
    this repeatedly re-derives the queue against the latest order.
    Returns ``(gain, moves_applied, spans)``.
    """
    bw_idx = np.flatnonzero(cur[src] > cur[tgt])
    if bw_idx.size == 0:
        return 0.0, 0, []
    order_by_w = bw_idx[np.argsort(-w[bw_idx], kind="stable")][:top_k]
    gain, applied, spans = 0.0, 0, []
    for e in order_by_w:
        u, v = int(src[e]), int(tgt[e])
        if cur[u] <= cur[v]:
            continue
        lo, hi = int(cur[v]), int(cur[u])
        gains = _pair_gain_curve(u, v, lo, hi, out_adj, in_adj, cur)
        r = int(np.argmax(gains))
        if gains[r] <= 0:
            continue
        mid = seq[lo + 1:hi]
        new_seq = np.concatenate((mid[:r], np.array([u, v], dtype=seq.dtype), mid[r:]))
        seq[lo:hi + 1] = new_seq
        cur[new_seq] = np.arange(lo, hi + 1, dtype=cur.dtype)
        gain += float(gains[r])
        applied += 1
        spans.append(hi - lo)
    return gain, applied, spans


def size_s1(g, rank: np.ndarray, src: np.ndarray, tgt: np.ndarray, w: np.ndarray,
            top_k: int, checkpoints: List[int], verify_every: int) -> Dict:
    """Sequential greedy application of the paired move over the top-K backward edges."""
    n = g.n_nodes
    out_adj, in_adj = build_adjacency(src, tgt, w, n)
    base_score = score_from_positions(rank, src, tgt, w)

    bw_idx = np.flatnonzero(rank[src] > rank[tgt])
    order_by_w = bw_idx[np.argsort(-w[bw_idx], kind="stable")][:top_k]

    stats = {
        "n_backward_edges": int(bw_idx.size),
        "backward_weight": float(np.asarray(w)[bw_idx].astype(np.float64).sum()),
        "candidates_examined": int(order_by_w.size),
        "min_candidate_weight": float(w[order_by_w[-1]]) if order_by_w.size else 0.0,
    }

    # ---- pass 1: every candidate evaluated INDEPENDENTLY at the UNCHANGED order ----
    # NOTE: this sum is NOT an upper bound on the realized gain. Moves interact both
    # ways — they can cannibalise each other (over-counting) but applying one can also
    # expose new gain for a later candidate (under-counting). It is reported only as
    # "how much signal each candidate sees in isolation", next to the realized number.
    t0 = time.time()
    indep, indep_positive = 0.0, 0
    for e in order_by_w:
        u, v = int(src[e]), int(tgt[e])
        best = float(_pair_gain_curve(u, v, int(rank[v]), int(rank[u]),
                                      out_adj, in_adj, rank).max())
        if best > 0:
            indep += best
            indep_positive += 1
    stats["independent_sum_at_initial_order_weight"] = indep
    stats["independent_sum_at_initial_order_pp"] = 100.0 * indep / g.total_weight
    stats["independent_positive_candidates"] = indep_positive
    stats["pass1_seconds"] = round(time.time() - t0, 1)

    # ---- pass 2: sequential greedy application (the realizable single-pass gain) ----
    t0 = time.time()
    cur = rank.copy()
    seq = np.argsort(cur, kind="stable")            # seq[r] = node at rank r
    realized, applied = 0.0, 0
    spans: List[int] = []
    curve, ckpts = [], set(checkpoints)
    for i, e in enumerate(order_by_w, start=1):
        u, v = int(src[e]), int(tgt[e])
        if cur[u] > cur[v]:                          # still backward -> try the move
            lo, hi = int(cur[v]), int(cur[u])
            gains = _pair_gain_curve(u, v, lo, hi, out_adj, in_adj, cur)
            r = int(np.argmax(gains))
            if gains[r] > 0:
                mid = seq[lo + 1:hi]                 # n_1..n_t
                new_seq = np.concatenate(
                    (mid[:r], np.array([u, v], dtype=seq.dtype), mid[r:]))
                seq[lo:hi + 1] = new_seq
                cur[new_seq] = np.arange(lo, hi + 1, dtype=cur.dtype)
                realized += float(gains[r])
                applied += 1
                spans.append(hi - lo)
        if verify_every and i % verify_every == 0:
            got = score_from_positions(cur, src, tgt, w) - base_score
            if abs(got - realized) > 1e-6:
                raise AssertionError(
                    f"S1 gain algebra broken at candidate {i}: "
                    f"oracle delta={got} vs accumulated {realized}")
        if i in ckpts:
            curve.append({"candidates": i, "applied": applied,
                          "gain_weight": realized,
                          "gain_pp": 100.0 * realized / g.total_weight})

    final_score = score_from_positions(cur, src, tgt, w)
    if abs((final_score - base_score) - realized) > 1e-6:
        raise AssertionError(
            f"S1 final mismatch: oracle {final_score - base_score} vs {realized}")

    stats.update({
        "moves_applied": applied,
        "realized_gain_weight": realized,
        "realized_gain_pp": 100.0 * realized / g.total_weight,
        "oracle_before_pct": pct(base_score, g.total_weight),
        "oracle_after_pct": pct(final_score, g.total_weight),
        "oracle_verified": True,
        "move_span_ranks": {
            "mean": float(np.mean(spans)) if spans else 0.0,
            "median": float(np.median(spans)) if spans else 0.0,
            "p90": float(np.percentile(spans, 90)) if spans else 0.0,
            "max": int(np.max(spans)) if spans else 0,
        },
        "gain_curve": curve,
        "pass2_seconds": round(time.time() - t0, 1),
    })
    return stats


# ── S2 — block-SCC condensation refinement (Vahidi Algorithm 3) ─────────────────
def _exact_small_scc_order(W: np.ndarray) -> Tuple[np.ndarray, float]:
    """Exact max-feedforward internal order of a tiny SCC via bitmask DP.

    ``W[i, j]`` = weight of edges from member ``i`` to member ``j`` (dense, k <= 10).
    ``f[S]`` = best feedforward weight when the members of ``S`` occupy the first
    ``|S|`` slots; placing ``x`` last in ``S`` earns every edge ``S\\{x} -> x``.
    Returns ``(order_as_member_indices, best_weight)``.
    """
    k = W.shape[0]
    full = 1 << k
    idx = np.arange(full)
    inw = np.zeros((k, full), dtype=np.float64)          # inw[x][S] = w(S -> x)
    for x in range(k):
        acc = np.zeros(full, dtype=np.float64)
        for i in range(k):
            if W[i, x]:
                acc[(idx & (1 << i)) != 0] += W[i, x]
        inw[x] = acc
    f = np.full(full, -np.inf)
    choice = np.zeros(full, dtype=np.int64)
    f[0] = 0.0
    for S in range(1, full):
        best, arg = -np.inf, -1
        for x in range(k):
            bit = 1 << x
            if not (S & bit):
                continue
            cand = f[S ^ bit] + inw[x][S ^ bit]
            if cand > best:
                best, arg = cand, x
        f[S], choice[S] = best, arg
    seq: List[int] = []
    S = full - 1
    while S:
        x = int(choice[S])
        seq.append(x)
        S ^= (1 << x)
    seq.reverse()
    return np.asarray(seq, dtype=np.int64), float(f[full - 1])


def _topo_order_condensation(nc: int, cs: np.ndarray, ct: np.ndarray,
                             min_loc: np.ndarray) -> List[int]:
    """Kahn topological order of the condensation, tie-broken by current position.

    Any valid topological order makes **every** inter-SCC edge feedforward, so the
    gain is independent of this choice; the ``min_loc`` tie-break only keeps the new
    order as close as possible to the current one.
    """
    indeg = np.bincount(ct, minlength=nc) if ct.size else np.zeros(nc, dtype=np.int64)
    succ_order, succ_starts = group_by_key(cs.astype(np.int64), nc) if cs.size \
        else (np.zeros(0, dtype=np.int64), np.zeros(nc + 1, dtype=np.int64))
    succ_targets = ct[succ_order] if cs.size else np.zeros(0, dtype=np.int64)

    heap = [(int(min_loc[c]), int(c)) for c in range(nc) if indeg[c] == 0]
    heapq.heapify(heap)
    out: List[int] = []
    while heap:
        _, c = heapq.heappop(heap)
        out.append(c)
        for d in succ_targets[succ_starts[c]:succ_starts[c + 1]].tolist():
            indeg[d] -= 1
            if indeg[d] == 0:
                heapq.heappush(heap, (int(min_loc[d]), int(d)))
    if len(out) != nc:                                    # pragma: no cover
        raise AssertionError("condensation is not acyclic — SCC labelling is wrong")
    return out


def s2_pass(n: int, rank: np.ndarray, src: np.ndarray, tgt: np.ndarray,
            wf: np.ndarray, s: int, offset: int, small_max: int) -> Tuple[np.ndarray, Dict]:
    """One block-SCC sweep at block size ``s`` and grid ``offset``.

    ``offset`` shifts the block boundaries (Vahidi Alg-3 line 4's "optional bias"), so
    that repeated sweeps see structure that straddled a previous boundary. Returns the
    new rank vector and a stats dict; the caller verifies the gain against the oracle.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    seq0 = np.argsort(rank, kind="stable")
    bid_s, bid_t = (rank[src] + offset) // s, (rank[tgt] + offset) // s
    e_idx = np.flatnonzero(bid_s == bid_t)
    # ceiling for ANY within-block reordering at this granularity
    ceiling = float(wf[e_idx][rank[tgt][e_idx] <= rank[src][e_idx]].sum())

    n_blocks = int(np.ceil((n + offset) / s))
    eb_order, eb_starts = group_by_key(bid_s[e_idx].astype(np.int64), n_blocks)
    e_sorted = e_idx[eb_order]

    new_rank = rank.copy()
    gain_topo = gain_small = ub_small = 0.0
    n_blocks_improved = n_small_sccs = 0

    for b in range(n_blocks):
        ee = e_sorted[eb_starts[b]:eb_starts[b + 1]]
        if ee.size == 0:
            continue
        lo, hi = max(0, b * s - offset), min((b + 1) * s - offset, n)
        sb = hi - lo
        lsrc = (rank[src[ee]] - lo).astype(np.int64)   # local slot 0..sb-1
        ltgt = (rank[tgt[ee]] - lo).astype(np.int64)
        ew = wf[ee]

        A = csr_matrix((np.ones(ee.size, dtype=np.int8), (lsrc, ltgt)), shape=(sb, sb))
        nc, labels = connected_components(A, directed=True, connection="strong")
        labels = labels.astype(np.int64)
        memb_order, memb_starts = group_by_key(labels, nc)

        diff = labels[lsrc] != labels[ltgt]
        g_topo = float(ew[diff].sum()) - float(ew[diff & (ltgt > lsrc)].sum())

        # exact re-solve of tiny sub-SCCs: bound first, compute only if non-zero
        sizes = memb_starts[1:] - memb_starts[:-1]
        is_small = (sizes >= 2) & (sizes <= small_max)
        n_small_sccs += int(is_small.sum())
        g_small = 0.0
        member_order: Dict[int, np.ndarray] = {}
        same_small = (~diff) & is_small[labels[lsrc]]
        bound = float(ew[same_small & (ltgt <= lsrc)].sum())
        ub_small += bound
        if bound > 0:
            by_comp: Dict[int, List[int]] = {}
            for pos in np.flatnonzero(same_small).tolist():
                by_comp.setdefault(int(labels[lsrc[pos]]), []).append(pos)
            for c, plist in by_comp.items():
                members = np.sort(memb_order[memb_starts[c]:memb_starts[c + 1]])
                remap = {int(m): i for i, m in enumerate(members)}
                W = np.zeros((members.size, members.size), dtype=np.float64)
                cur_ff = 0.0
                for pos in plist:
                    i, j = remap[int(lsrc[pos])], remap[int(ltgt[pos])]
                    W[i, j] += ew[pos]
                    if ltgt[pos] > lsrc[pos]:
                        cur_ff += ew[pos]
                best_seq, best_ff = _exact_small_scc_order(W)
                if best_ff > cur_ff + 1e-9:
                    g_small += best_ff - cur_ff
                    member_order[c] = members[best_seq]

        if g_topo + g_small <= 0:
            continue
        n_blocks_improved += 1
        gain_topo += g_topo
        gain_small += g_small

        # materialize: SCCs in topological order, small ones internally optimised
        cs_all, ct_all = labels[lsrc][diff], labels[ltgt][diff]
        if cs_all.size:
            uniq = np.unique(cs_all * nc + ct_all)
            cs_all, ct_all = uniq // nc, uniq % nc
        min_loc = np.full(nc, sb, dtype=np.int64)
        np.minimum.at(min_loc, labels, np.arange(sb, dtype=np.int64))
        pieces = [member_order[c] if c in member_order
                  else np.sort(memb_order[memb_starts[c]:memb_starts[c + 1]])
                  for c in _topo_order_condensation(nc, cs_all, ct_all, min_loc)]
        new_loc = np.concatenate(pieces)
        new_rank[seq0[lo:hi][new_loc]] = np.arange(lo, hi, dtype=new_rank.dtype)

    stats = {
        "block_size": s,
        "offset": offset,
        "n_blocks": n_blocks,
        "n_blocks_improved": n_blocks_improved,
        "intra_block_feedback_ceiling_weight": ceiling,
        "gain_topo_weight": gain_topo,
        "gain_small_scc_weight": gain_small,
        "small_scc_upper_bound_weight": ub_small,
        "n_small_sccs_examined": n_small_sccs,
        "total_gain_weight": gain_topo + gain_small,
    }
    return new_rank, stats


def size_s2(g, rank: np.ndarray, src: np.ndarray, tgt: np.ndarray, w: np.ndarray,
            block_sizes: List[int], small_max: int) -> List[Dict]:
    """One pass of the block-SCC move for each block size; exact, oracle-verified."""
    base_score = score_from_positions(rank, src, tgt, w)
    wf = np.asarray(w).astype(np.float64)
    rows = []
    for s in block_sizes:
        if s >= g.n_nodes:
            continue
        t0 = time.time()
        new_rank, st = s2_pass(g.n_nodes, rank, src, tgt, wf, s, 0, small_max)
        new_score = score_from_positions(new_rank, src, tgt, w)
        if abs((new_score - base_score) - st["total_gain_weight"]) > 1e-6:
            raise AssertionError(
                f"S2 gain algebra broken at s={s}: oracle delta="
                f"{new_score - base_score} vs accumulated {st['total_gain_weight']}")
        st.update({
            "intra_block_feedback_ceiling_pp":
                100.0 * st["intra_block_feedback_ceiling_weight"] / g.total_weight,
            "total_gain_pp": 100.0 * st["total_gain_weight"] / g.total_weight,
            "oracle_before_pct": pct(base_score, g.total_weight),
            "oracle_after_pct": pct(new_score, g.total_weight),
            "oracle_verified": True,
            "seconds": round(time.time() - t0, 1),
        })
        rows.append(st)
        print(f"  [S2] s={s:>6}: gain {st['total_gain_pp']:+.5f} pp "
              f"(topo {st['gain_topo_weight']:.4g} + small "
              f"{st['gain_small_scc_weight']:.4g}); ceiling "
              f"{st['intra_block_feedback_ceiling_pp']:.4f} pp; "
              f"{st['seconds']}s", flush=True)
    return rows


# ── S1 + S2 alternated to a fixed point (does the gain compound?) ───────────────
def size_iterated(g, rank: np.ndarray, src: np.ndarray, tgt: np.ndarray, w: np.ndarray,
                  rounds: int, top_k: int, block_size: int, small_max: int) -> Dict:
    """Alternate one S1 sweep and one S2 sweep per round; report the cumulative curve.

    The S2 grid offset alternates between ``0`` and ``s/2`` so that consecutive sweeps
    do not re-solve identical blocks (Vahidi Alg-3's "optional bias"). Each round is
    re-scored by the frozen oracle and the accumulated gain must match exactly.
    """
    n = g.n_nodes
    wf = np.asarray(w).astype(np.float64)
    out_adj, in_adj = build_adjacency(src, tgt, w, n)
    base_score = score_from_positions(rank, src, tgt, w)

    cur = rank.copy()
    seq = np.argsort(cur, kind="stable")
    total = 0.0
    curve = []
    for r in range(1, rounds + 1):
        t0 = time.time()
        g1, applied, _ = s1_pass(cur, seq, src, tgt, w, out_adj, in_adj, top_k)
        total += g1
        got = score_from_positions(cur, src, tgt, w) - base_score
        if abs(got - total) > 1e-6:
            raise AssertionError(f"iterated S1 mismatch (round {r}): {got} vs {total}")

        offset = 0 if r % 2 == 1 else block_size // 2
        cur, st = s2_pass(n, cur, src, tgt, wf, block_size, offset, small_max)
        seq = np.argsort(cur, kind="stable")
        total += st["total_gain_weight"]
        got = score_from_positions(cur, src, tgt, w) - base_score
        if abs(got - total) > 1e-6:
            raise AssertionError(f"iterated S2 mismatch (round {r}): {got} vs {total}")

        curve.append({
            "round": r, "s1_gain_weight": g1, "s1_moves": applied,
            "s2_offset": offset, "s2_gain_weight": st["total_gain_weight"],
            "cumulative_gain_weight": total,
            "cumulative_gain_pp": 100.0 * total / g.total_weight,
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  [ITER] round {r}: S1 {100.0*g1/g.total_weight:+.5f} pp "
              f"({applied} moves) + S2@{offset} "
              f"{100.0*st['total_gain_weight']/g.total_weight:+.5f} pp "
              f"-> cumulative {curve[-1]['cumulative_gain_pp']:+.5f} pp "
              f"({curve[-1]['seconds']}s)", flush=True)

    final = score_from_positions(cur, src, tgt, w)
    return {
        "rounds": rounds, "top_k": top_k, "block_size": block_size,
        "curve": curve,
        "total_gain_pp": 100.0 * total / g.total_weight,
        "oracle_before_pct": pct(base_score, g.total_weight),
        "oracle_after_pct": pct(final, g.total_weight),
        "oracle_verified": True,
    }


# ── driver ──────────────────────────────────────────────────────────────────────
def run_dataset(name: str, seed: int, top_k: int, rounds: int,
                sift_first: bool) -> Dict:
    t0 = time.time()
    g = io.load_dataset(name)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight)
    keep = src != tgt                              # self-loops are always feedback
    n_self = int((~keep).sum())
    if n_self:
        src, tgt, w = src[keep], tgt[keep], w[keep]

    pos_file = latest_h35_positions(name, seed)
    rank = rank_of(np.load(pos_file))
    h35_score = score_from_positions(rank, src, tgt, w)
    print(f"[{name}] seed={seed} n={g.n_nodes:,} m={len(src):,} self_loops={n_self} "
          f"H35 order = {pct(h35_score, g.total_weight):.5f}%  ({pos_file.name})", flush=True)

    # ---- attribution control: exhaust the SINGLE-NODE move class first ----
    control = None
    if sift_first:
        from mfas.refine.underrelax import sift_underrelaxed
        t_c = time.time()
        rank, sift_score, log = sift_underrelaxed(
            g, rank, max_sweeps=CONFIG["sift_first_max_sweeps"],
            time_budget_s=CONFIG["sift_first_time_budget_s"])
        rank = np.asarray(rank, dtype=np.int64)
        movers_left = int(log[-1]["n_movers"]) if log else 0
        control = {
            "sweeps_run": len(log),
            "movers_left_at_stop": movers_left,
            "reached_fixed_point": movers_left == 0,
            "single_node_recovery_weight": float(sift_score) - h35_score,
            "single_node_recovery_pp": 100.0 * (float(sift_score) - h35_score) / g.total_weight,
            "pct_after": pct(float(sift_score), g.total_weight),
            "seconds": round(time.time() - t_c, 1),
        }
        print(f"  [1-OPT CONTROL] {control['sweeps_run']} sweeps, "
              f"{movers_left} movers left, single-node recovery "
              f"{control['single_node_recovery_pp']:+.5f} pp -> "
              f"{control['pct_after']:.5f}%  ({control['seconds']}s)", flush=True)

    base = score_from_positions(rank, src, tgt, w)

    s1 = size_s1(g, rank, src, tgt, w, top_k=min(top_k, len(src)),
                 checkpoints=CONFIG["s1_checkpoints"],
                 verify_every=CONFIG["verify_every"])
    print(f"  [S1] realized {s1['realized_gain_pp']:+.5f} pp from {s1['moves_applied']} "
          f"moves (independent sum {s1['independent_sum_at_initial_order_pp']:+.5f} pp); "
          f"median span {s1['move_span_ranks']['median']:.0f} ranks; "
          f"{s1['pass1_seconds']}s + {s1['pass2_seconds']}s", flush=True)

    s2 = size_s2(g, rank, src, tgt, w,
                 block_sizes=CONFIG["s2_block_sizes"],
                 small_max=CONFIG["s2_small_scc_max"])

    # best single-pass block size drives the alternating S1<->S2 iteration
    best_s = max(s2, key=lambda r: r["total_gain_weight"])["block_size"] if s2 \
        else CONFIG["s2_block_sizes"][0]
    it = size_iterated(g, rank, src, tgt, w, rounds=rounds,
                       top_k=min(top_k, len(src)), block_size=best_s,
                       small_max=CONFIG["s2_small_scc_max"]) if rounds > 0 else None

    return {
        "dataset": name,
        "seed": seed,
        "n_nodes": int(g.n_nodes),
        "n_edges": int(len(src)),
        "n_self_loops_dropped": n_self,
        "total_weight": float(g.total_weight),
        "h35_positions_file": pos_file.name,
        "h35_pct": pct(h35_score, g.total_weight),
        "sift_first_control": control,
        "sizing_baseline_pct": pct(base, g.total_weight),
        "S1_paired_backward_move": s1,
        "S2_block_scc": s2,
        "iterated_S1_S2": it,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="S1/S2 collective-move sizing")
    ap.add_argument("--datasets", default="connectome,microns,mouse")
    ap.add_argument("--seed", type=int, default=CONFIG["seed"],
                    help="which logged H35 order to size from")
    ap.add_argument("--top-k", type=int, default=CONFIG["s1_top_k"])
    ap.add_argument("--rounds", type=int, default=CONFIG["iter_rounds"],
                    help="alternating S1<->S2 rounds (0 disables the iteration probe)")
    ap.add_argument("--sift-first", action="store_true",
                    help="exhaust the single-node sift first, so the gain credited to "
                         "S1/S2 is genuinely outside the 1-opt move class")
    ap.add_argument("--out", default="collective_moves_sizing.json")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(CONFIG, seed=args.seed, s1_top_k=args.top_k, iter_rounds=args.rounds,
               sift_first=args.sift_first)
    out = {
        "probe": "S1/S2 collective-move sizing (roadmap A-SCC / A-PAIR gate)",
        "git_commit": git_commit(),
        "config": cfg,
        "datasets": {},
    }
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        out["datasets"][name] = run_dataset(name, args.seed, args.top_k, args.rounds,
                                            args.sift_first)

    dest = Path(args.out) if Path(args.out).is_absolute() else OUT_DIR / args.out
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {dest}")


if __name__ == "__main__":
    main()
