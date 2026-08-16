"""H45 prototype gate — exact-gain JOINT PAIR relocation, measured on the champion's order.

The question this answers
-------------------------
The champion's stage 3 (:func:`mfas.refine.underrelax.sift_underrelaxed`) relocates nodes
INDEPENDENTLY: :func:`mfas.refine.insertion.jacobi_best_gaps` takes each node's own argmax
over its exact insertion profile. It is therefore blind by construction to a move where
neither endpoint of a heavy backward edge profits alone but both profit from MEETING.

H45 is that move. Take a backward edge ``(u, v)`` — an edge ``u -> v`` with
``rank[v] < rank[u]``, so it currently counts as feedback — extract BOTH endpoints, and
re-insert them adjacent as ``u, v`` at some split ``r`` inside the interval between them.

The exact gain
--------------
Let ``n_1 .. n_t`` be the nodes strictly between ``v`` and ``u``. After the move the line
reads ``... n_1 .. n_r, u, v, n_{r+1} .. n_t ...``. Then

    Delta(r) = (w_uv - w_vu)
             + sum_{j<=r} [ w(n_j -> v) - w(v -> n_j) ]
             + sum_{j>r}  [ w(u -> n_j) - w(n_j -> u) ]

**Proof.** The interval ``[rank[v], rank[u]]`` is a contiguous position range and every node
that moves stays inside it, so by the contiguous-block lemma (proved in
``mfas/refine/segment.py``) no edge with an endpoint outside the interval can flip. The
``n_j`` keep their relative order, so no edge among them flips. Only edges incident to ``u``
or ``v`` flip, and each flips on exactly one side of the split: ``v`` passes to the right of
``n_1..n_r`` (so those edges reverse) and ``u`` passes to the left of ``n_{r+1}..n_t``.
The ``u``-``v`` edge itself becomes feedforward, contributing ``+w_uv``, and any ``v -> u``
edge becomes feedback, contributing ``-w_vu``.

Why it is cheap
---------------
``Delta(r)`` is PIECEWISE CONSTANT in ``r`` and changes only at those ``j`` where ``n_j`` is a
neighbour of ``u`` or of ``v``. So the optimal split is found in
``O(d(u) + d(v))`` **independent of the interval length** — the interval can be 100,000
positions wide and cost nothing extra. Mean total degree on connectome is
``2 * 5,657,719 / 136,648 = 83``.

This is what makes H45 different from every window-bounded move class the campaign has tried:
its range is set by the DATA (where the backward edge's endpoints actually are) rather than by
a parameter, and it is affordable at unbounded range.

What this script measures, and what it deliberately does NOT
------------------------------------------------------------
It is a GATE, not a variant. It reports, on the champion's own connectome order:

1. **Distinctness** — the fraction of candidate pairs whose joint optimum is a move that
   neither endpoint's independent argmax would make. If that is near zero the class is not
   distinct from the existing sift and H45 dies here, for free. (Meta-rule M8: a class must be
   judged by what it adds, not by its own credit.)
2. **Exactness** — predicted ``Delta`` versus the realised delta from the FROZEN oracle, on
   every applied move. The campaign's whole accounting discipline is
   predicted == realised; a prototype that cannot show that is worthless.
3. **Reachable gain** — the total exact gain of a greedy disjoint-interval batch, i.e. what
   one sweep would actually realise.

It does NOT screen the variant. Per M8 the number that decides H45 is the composed
champion-vs-variant delta, which requires a variant module and the gate ladder.

Leakage-safety: every quantity is computed from the edge weights and the current ranks. The
frozen oracle is called only to CHECK a predicted gain after the fact, never to choose a move.
``data/best_solution`` is never read.

Run:  PYTHONPATH=src python experiments/proto_H45_pair.py [dataset] [top_k]
Out:  experiments/outputs/proto_H45_pair_<dataset>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io                                          # noqa: E402
from mfas.metrics import pct, score_from_order               # noqa: E402
from mfas.refine.insertion import jacobi_best_gaps           # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# The exact pair move
# ──────────────────────────────────────────────────────────────────────────────
def pair_move_best_split(rank, adj_out, adj_in, u, v, w_uv, w_vu):
    """Best split ``r`` and its EXACT gain for relocating ``u, v`` adjacent.

    ``rank[v] < rank[u]`` is assumed (the edge ``u -> v`` is currently feedback).

    Returns ``(best_gain, best_cut_pos)`` where ``best_cut_pos`` is the POSITION after
    which the pair is inserted (``rank[v]`` means "immediately after v's old slot", i.e.
    r = 0). Cost is O(d(u) + d(v)); the interval length never enters.
    """
    lo = rank[v]
    hi = rank[u]
    base = w_uv - w_vu                      # the u-v edge itself flips, always

    # Breakpoints: only neighbours of u or v that lie strictly inside the interval can
    # change Delta.
    #
    # CRITICAL: aggregate by POSITION before scanning. A single interval node may be a
    # neighbour of BOTH u and v (and may carry parallel edges), and a split is defined by a
    # NODE being wholly on one side. Emitting one event per edge and scanning them
    # individually manufactures intermediate states where a node is counted in the prefix
    # for its v-edges while still counted in the suffix for its u-edges. Those states are
    # not splits, they are not achievable, and their gain is inflated. That bug produced a
    # 1-of-18 exactness result on mouse before it was caught by the oracle cross-check.
    acc_v = {}     # position -> c_v = sum[ w(n_j -> v) - w(v -> n_j) ]
    acc_u = {}     # position -> c_u = sum[ w(u -> n_j) - w(n_j -> u) ]

    for nb, wt in adj_in[v]:               # n_j -> v
        p = rank[nb]
        if lo < p < hi:
            acc_v[p] = acc_v.get(p, 0.0) + wt
    for nb, wt in adj_out[v]:              # v -> n_j
        p = rank[nb]
        if lo < p < hi:
            acc_v[p] = acc_v.get(p, 0.0) - wt
    for nb, wt in adj_out[u]:              # u -> n_j
        p = rank[nb]
        if lo < p < hi:
            acc_u[p] = acc_u.get(p, 0.0) + wt
    for nb, wt in adj_in[u]:               # n_j -> u
        p = rank[nb]
        if lo < p < hi:
            acc_u[p] = acc_u.get(p, 0.0) - wt

    if not acc_v and not acc_u:
        # No neighbour inside the interval: Delta is constant = base for every split.
        return float(base), int(lo)

    pos = np.array(sorted(set(acc_v) | set(acc_u)), dtype=np.int64)
    cv = np.array([acc_v.get(int(p), 0.0) for p in pos], dtype=np.float64)
    cu = np.array([acc_u.get(int(p), 0.0) for p in pos], dtype=np.float64)

    suffix_u_total = float(cu.sum())
    # r = 0 (pair goes immediately after v's slot): prefix empty, suffix is everything.
    best_gain = base + suffix_u_total
    best_cut = int(lo)

    prefix_v = 0.0
    suffix_u = suffix_u_total
    for k in range(pos.shape[0]):
        prefix_v += cv[k]
        suffix_u -= cu[k]
        g = base + prefix_v + suffix_u
        if g > best_gain:
            best_gain = g
            best_cut = int(pos[k])
    return float(best_gain), best_cut


def apply_pair_move(seq, rank, u, v, cut_pos):
    """Rebuild the position sequence with ``u, v`` adjacent just after ``cut_pos``."""
    n = seq.shape[0]
    lo, hi = rank[v], rank[u]
    block = [x for x in seq[lo:hi + 1] if x != u and x != v]
    out = []
    placed = False
    for x in block:
        out.append(x)
        if rank[x] == cut_pos:
            out.extend((u, v)); placed = True
    if not placed:
        out = [u, v] + out
    new_seq = seq.copy()
    new_seq[lo:hi + 1] = np.asarray(out, dtype=np.int64)
    return new_seq


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "connectome"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

    g = io.load_dataset(dataset)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    n = g.n_nodes
    total = g.total_weight

    # The champion's order, straight off a logged run (never recomputed here).
    import glob
    cands = sorted(glob.glob(f"results/*-H4*-{dataset}-s42-*.json"))
    pos_path = None
    for f in reversed(cands):
        d = json.load(open(f))
        p = d.get("best_positions_path")
        if p and Path(p).exists():
            pos_path = p
            champ_pct = d["pct"]
            break
    if pos_path is None:
        print("no champion position vector found on disk")
        return 2
    positions = np.load(pos_path)
    rank = np.argsort(np.argsort(positions, kind="stable"), kind="stable").astype(np.int64)
    base_score = score_from_order(rank, src, tgt, g.weight)
    print(f"champion order: {pos_path}  pct={pct(base_score, total):.6f} (record {champ_pct})",
          flush=True)

    # Adjacency as CSR. Python lists of tuples would be ~11 M objects and several GB at
    # connectome scale; two int64/float64 CSR pairs are ~140 MB and index far faster.
    keep = src != tgt
    ks, kt, kw = src[keep], tgt[keep], w[keep]

    def build_csr(row, col, val):
        cnt = np.bincount(row, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(cnt, out=indptr[1:])
        order_ = np.argsort(row, kind="stable")
        return indptr, col[order_].copy(), val[order_].copy()

    out_ptr, out_idx, out_w = build_csr(ks, kt, kw)   # u -> out-neighbours
    in_ptr, in_idx, in_w = build_csr(kt, ks, kw)      # v <- in-neighbours

    class _Adj:
        """Minimal sequence view so pair_move_best_split keeps its (nb, wt) contract."""
        __slots__ = ("ptr", "idx", "wt")

        def __init__(self, ptr, idx, wt):
            self.ptr, self.idx, self.wt = ptr, idx, wt

        def __getitem__(self, node):
            a, b = self.ptr[node], self.ptr[node + 1]
            return zip(self.idx[a:b], self.wt[a:b])

    adj_out = _Adj(out_ptr, out_idx, out_w)
    adj_in = _Adj(in_ptr, in_idx, in_w)

    def reverse_weight(a, b):
        """Total weight of edges a -> b, read from the CSR (0.0 if none)."""
        p, q = out_ptr[a], out_ptr[a + 1]
        seg = out_idx[p:q]
        hits = np.flatnonzero(seg == b)
        return float(out_w[p:q][hits].sum()) if hits.size else 0.0

    # backward edges = current feedback, heaviest first
    back = np.flatnonzero(rank[src] > rank[tgt])
    order = back[np.argsort(-w[back], kind="stable")][:top_k]
    print(f"backward edges: {back.size:,}  examining top {order.size:,} by weight", flush=True)

    # independent argmaxes, for the DISTINCTNESS test
    t0 = time.time()
    best_gap, best_gain_single = jacobi_best_gaps(rank, src, tgt, w, n)
    t_jacobi = time.time() - t0

    rows = []
    t0 = time.time()
    for ei in order:
        u = int(src[ei]); v = int(tgt[ei])
        # Parallel edges are summed, which is what the scorer does too.
        w_uv = reverse_weight(u, v)
        w_vu = reverse_weight(v, u)
        gain, cut = pair_move_best_split(rank, adj_out, adj_in, u, v, w_uv, w_vu)
        if gain > 0:
            rows.append(dict(u=u, v=v, gain=gain, cut=cut,
                             lo=int(rank[v]), hi=int(rank[u]),
                             span=int(rank[u] - rank[v]),
                             single_u=float(best_gain_single[u]),
                             single_v=float(best_gain_single[v])))
    t_scan = time.time() - t0

    rows.sort(key=lambda r: -r["gain"])
    n_pos = len(rows)
    # DISTINCTNESS: a pair move that neither endpoint would make alone
    n_distinct = sum(1 for r in rows
                     if r["single_u"] <= 1e-9 and r["single_v"] <= 1e-9)

    print(f"jacobi profiles: {t_jacobi:.1f}s | pair scan: {t_scan:.1f}s for {order.size:,} edges",
          flush=True)
    print(f"positive-gain pairs: {n_pos:,} of {order.size:,}", flush=True)
    print(f"  of which NEITHER endpoint profits alone (the distinct class): "
          f"{n_distinct:,} ({100.0*n_distinct/max(n_pos,1):.1f}% of positive)", flush=True)

    # EXACTNESS: verify the top moves against the frozen oracle, one at a time
    seq = np.empty(n, dtype=np.int64)
    seq[rank] = np.arange(n, dtype=np.int64)
    checks = []
    for r in rows[:25]:
        new_seq = apply_pair_move(seq, rank, r["u"], r["v"], r["cut"])
        new_rank = np.empty(n, dtype=np.int64)
        new_rank[new_seq] = np.arange(n, dtype=np.int64)
        realised = score_from_order(new_rank, src, tgt, g.weight) - base_score
        checks.append(dict(u=r["u"], v=r["v"], predicted=r["gain"],
                           realised=float(realised),
                           match=bool(abs(realised - r["gain"]) < 1e-6)))
    n_match = sum(c["match"] for c in checks)
    print(f"EXACTNESS: {n_match}/{len(checks)} top moves match the frozen oracle exactly",
          flush=True)
    if n_match != len(checks):
        for c in checks:
            if not c["match"]:
                print(f"   MISMATCH u={c['u']} v={c['v']} "
                      f"predicted={c['predicted']} realised={c['realised']}", flush=True)

    # REACHABLE GAIN — two batchings, because the packing rule is not a detail here.
    #
    # Disjoint intervals compose ADDITIVELY (contiguous-block lemma), so a batch's realised
    # delta is exactly the sum of its gains. But these intervals are long — a heavy backward
    # edge can span most of the line — so which disjoint SET you pick matters much more than
    # it did for H41's bounded segments.
    #
    # (a) greedy by gain, the rule select_disjoint_moves uses today;
    # (b) weighted interval scheduling, which is OPTIMAL. Sort by right endpoint, then
    #     dp[i] = max(dp[i-1], gain_i + dp[p(i)]) where p(i) is the last interval ending
    #     before interval i starts. O(k log k). The 2026 Vahidi-Koutis abstract names exactly
    #     this ("a dynamic program to select large sets of non-overlapping backward-edge
    #     intervals, then refining them in parallel").
    taken = np.zeros(n, dtype=bool)
    greedy = []
    for r in rows:
        if taken[r["lo"]:r["hi"] + 1].any():
            continue
        taken[r["lo"]:r["hi"] + 1] = True
        greedy.append(r)
    greedy_gain = sum(r["gain"] for r in greedy)
    print(f"greedy disjoint batch:  {len(greedy):,} moves, "
          f"predicted +{100.0*greedy_gain/total:.5f} pp", flush=True)

    iv = sorted(rows, key=lambda r: r["hi"])
    ends = [r["hi"] for r in iv]
    import bisect
    dp = [0.0] * (len(iv) + 1)
    choice = [False] * (len(iv) + 1)
    for i, r in enumerate(iv, start=1):
        j = bisect.bisect_right(ends, r["lo"] - 1, 0, i - 1)   # last interval ending < lo
        take = r["gain"] + dp[j]
        if take > dp[i - 1]:
            dp[i] = take
            choice[i] = True
        else:
            dp[i] = dp[i - 1]
    # reconstruct
    opt = []
    i = len(iv)
    while i > 0:
        if choice[i]:
            r = iv[i - 1]
            opt.append(r)
            i = bisect.bisect_right(ends, r["lo"] - 1, 0, i - 1)
        else:
            i -= 1
    opt_gain = dp[len(iv)]
    print(f"OPTIMAL disjoint batch: {len(opt):,} moves, "
          f"predicted +{100.0*opt_gain/total:.5f} pp", flush=True)

    out = dict(
        item="H45", dataset=dataset, top_k=int(order.size),
        champion_positions=pos_path, champion_pct=pct(base_score, total),
        n_backward_edges=int(back.size),
        n_positive_pairs=n_pos,
        n_distinct_pairs=n_distinct,
        distinct_frac_of_positive=(n_distinct / n_pos) if n_pos else 0.0,
        t_jacobi_s=t_jacobi, t_pair_scan_s=t_scan,
        exactness_checks=checks,
        exactness_all_match=bool(n_match == len(checks)),
        greedy_batch_moves=len(greedy),
        greedy_batch_gain_pp=100.0 * greedy_gain / total,
        optimal_batch_moves=len(opt),
        optimal_batch_gain_pp=100.0 * opt_gain / total,
        top_moves=rows[:50],
    )
    op = _ROOT / "experiments" / "outputs" / f"proto_H45_pair_{dataset}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
