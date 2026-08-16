"""H50 — the reference ROUTE, run standalone. Divergent-mode item.

Why this exists
---------------
Six axes were measured shut on 2026-08-16 (Rocket epochs on both primaries, the initial order,
the stage-3 sweep caps, segment moves, pair relocation bolted onto a converged order, and coarse
block permutation). Five consecutive science kills followed the one score move, which forces
divergent mode: a structurally different approach rather than another increment.

The evidence points at one place. H45's gate established that the reference family's move class
is **real and distinct** — 989 improving pair moves that our sift cannot see by construction, the
best spanning 92,958 positions, 68% of the line — but that it has almost nothing left to do on
OUR converged order (only 0.30% of backward edges admit any improving pair move there). In the
reference family that same move is not a bolt-on: it is the **primary refiner**, applied from a
cheap greedy order, and it is credited with the climb from 75.24% to 84.61%.

So this script does not add anything to the champion. It builds the reference route on its own
terms and measures where it lands:

    ratio-greedy init  ->  iterated exact-gain pair relocation  ->  (report)

with no Rocket, no under-relaxed sift, and no SCC block refinement.

What it is measured against
---------------------------
NOT the champion — that would repeat H45's mistake of judging a route by a yardstick built for a
different one. The honest comparators, both measured on this machine with the frozen scorer:

* ``74.61730 %`` — the ratio-greedy start itself (``proto_H48_connectome.json``);
* ``83.87975 %`` — our own **Rocket-free** pipeline, i.e. the ``epochs=0`` arm of the connectome
  sizing grid: greedy-FAS + under-relaxed sift + SCC stage 4 (``proto_S01_connectome.json``).
  That arm's stage 3 hit its 40-sweep cap without converging, so it is a LOWER bound on what our
  refiner does from a cheap start;
* ``84.15410 %`` — the champion, quoted for orientation only.

The question is whether a refiner built from ONE move class, applied globally and iterated, gets
into that company. If it lands far below 83.88 the reference route does not transfer and the
external 84.61 figure must be explained some other way (its provenance is already unverified —
see ``autoresearch/lit/vahidi-2025.md``). If it lands near or above it, the campaign has a second
refiner worth developing.

The move and its exact gain
---------------------------
For a backward edge ``(u, v)`` with ``rank[v] < rank[u]``, extract both endpoints and re-insert
them adjacent as ``u, v`` at the best split ``r`` inside the interval between them:

    Delta(r) = (w_uv - w_vu) + sum_{j<=r}[w(n_j->v) - w(v->n_j)] + sum_{j>r}[w(u->n_j) - w(n_j->u)]

exact by the contiguous-block lemma, and computable in ``O(d(u) + d(v))`` **independent of the
interval length** because ``Delta`` is piecewise constant and changes only at neighbours of ``u``
or ``v``. Verified 25/25 against the frozen oracle in ``proto_H45_pair.py``.

Batching: accepted moves must occupy DISJOINT intervals, which compose additively by the same
lemma. Selection is the optimal weighted-interval-scheduling DP (sort by right endpoint,
``dp[i] = max(dp[i-1], gain_i + dp[p(i)])``), which measured 1.93x better than greedy packing on
the same candidate set.

Leakage-safety: every move is chosen from edge weights and current ranks alone. The frozen oracle
scores whole rank vectors between sweeps, as a check and for the log; it never enters a move
choice. ``data/best_solution`` is never read.

Run:  PYTHONPATH=src python experiments/proto_H50_reference_route.py [dataset] [max_sweeps] [top_k]
Out:  experiments/outputs/proto_H50_<dataset>.json
"""
from __future__ import annotations

import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io                                            # noqa: E402
from mfas.metrics import pct, score_from_order                 # noqa: E402
from mfas.baseline.ratio_greedy import ratio_greedy_rank       # noqa: E402

# Comparators, all measured on this machine with the frozen scorer.
REF = {
    "connectome": dict(ratio_init=74.61730241840903, rocket_free=83.87974978419737,
                       champion=84.15409511053134),
    "mouse": dict(ratio_init=88.38256063787692, rocket_free=93.08288021668459,
                  champion=93.08288021668459),
}


def build_csr(row, col, val, n):
    cnt = np.bincount(row, minlength=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(cnt, out=ptr[1:])
    o = np.argsort(row, kind="stable")
    return ptr, col[o].copy(), val[o].copy()


def pair_sweep(rank, src, tgt, w, n, out_ptr, out_idx, out_w, in_ptr, in_idx, in_w, top_k):
    """One sweep: scan the heaviest backward edges, take the optimal disjoint batch.

    Returns ``(new_rank, info)``. ``info['predicted_gain']`` is exact by the lemma, so the
    caller can cross-check it against the oracle rather than trusting it.
    """
    back = np.flatnonzero(rank[src] > rank[tgt])
    if back.size == 0:
        return rank, dict(n_candidates=0, n_positive=0, n_applied=0, predicted_gain=0.0)
    order = back[np.argsort(-w[back], kind="stable")][:top_k]

    rows = []
    for ei in order:
        u = int(src[ei]); v = int(tgt[ei])
        lo = int(rank[v]); hi = int(rank[u])
        # w(u->v) and w(v->u), parallel edges summed (the scorer sums them too)
        a, b = out_ptr[u], out_ptr[u + 1]
        seg = out_idx[a:b]; hits = np.flatnonzero(seg == v)
        w_uv = float(out_w[a:b][hits].sum()) if hits.size else 0.0
        a, b = out_ptr[v], out_ptr[v + 1]
        seg = out_idx[a:b]; hits = np.flatnonzero(seg == u)
        w_vu = float(out_w[a:b][hits].sum()) if hits.size else 0.0

        # Breakpoints, aggregated PER POSITION (per-edge events would manufacture splits that
        # are not achievable - the bug that scored 1/18 in the H45 prototype).
        acc_v, acc_u = {}, {}
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
            gain, cut = base, lo
        else:
            pos = sorted(set(acc_v) | set(acc_u))
            suffix = sum(acc_u.values())
            gain, cut = base + suffix, lo
            pref = 0.0
            for p in pos:
                pref += acc_v.get(p, 0.0)
                suffix -= acc_u.get(p, 0.0)
                g = base + pref + suffix
                if g > gain:
                    gain, cut = g, p
        if gain > 1e-9:
            rows.append((lo, hi, gain, u, v, cut))

    if not rows:
        return rank, dict(n_candidates=int(order.size), n_positive=0, n_applied=0,
                          predicted_gain=0.0)

    # optimal disjoint selection: weighted interval scheduling on [lo, hi]
    rows.sort(key=lambda r: r[1])
    ends = [r[1] for r in rows]
    dp = [0.0] * (len(rows) + 1)
    take = [False] * (len(rows) + 1)
    for i, r in enumerate(rows, start=1):
        j = bisect.bisect_right(ends, r[0] - 1, 0, i - 1)
        if r[2] + dp[j] > dp[i - 1]:
            dp[i], take[i] = r[2] + dp[j], True
        else:
            dp[i] = dp[i - 1]
    chosen, i = [], len(rows)
    while i > 0:
        if take[i]:
            chosen.append(rows[i - 1])
            i = bisect.bisect_right(ends, rows[i - 1][0] - 1, 0, i - 1)
        else:
            i -= 1

    seq = np.empty(n, dtype=np.int64)
    seq[rank] = np.arange(n, dtype=np.int64)
    new_seq = seq.copy()
    for (lo, hi, gain, u, v, cut) in chosen:
        block = [x for x in seq[lo:hi + 1] if x != u and x != v]
        out, placed = [], False
        for x in block:
            out.append(x)
            if rank[x] == cut:
                out.extend((u, v)); placed = True
        if not placed:
            out = [u, v] + out
        new_seq[lo:hi + 1] = np.asarray(out, dtype=np.int64)
    new_rank = np.empty(n, dtype=np.int64)
    new_rank[new_seq] = np.arange(n, dtype=np.int64)
    return new_rank, dict(n_candidates=int(order.size), n_positive=len(rows),
                          n_applied=len(chosen),
                          predicted_gain=float(sum(r[2] for r in chosen)))


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "connectome"
    max_sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000

    g = io.load_dataset(dataset)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    n, total = g.n_nodes, g.total_weight
    ref = REF.get(dataset, {})

    keep = src != tgt
    ks, kt, kw = src[keep], tgt[keep], w[keep]
    out_ptr, out_idx, out_w = build_csr(ks, kt, kw, n)
    in_ptr, in_idx, in_w = build_csr(kt, ks, kw, n)

    t0 = time.time()
    rank = ratio_greedy_rank(g)
    t_init = time.time() - t0
    score = score_from_order(rank, src, tgt, g.weight)
    print(f"init (ratio greedy): {pct(score, total):.5f}%  {t_init:.1f}s", flush=True)

    rows, t_start = [], time.time()
    for s in range(max_sweeps):
        t1 = time.time()
        new_rank, info = pair_sweep(rank, src, tgt, w, n, out_ptr, out_idx, out_w,
                                    in_ptr, in_idx, in_w, top_k)
        new_score = score_from_order(new_rank, src, tgt, g.weight)
        realised = new_score - score
        exact = abs(realised - info["predicted_gain"]) < 1e-6
        rows.append(dict(sweep=s, pct=pct(new_score, total),
                         n_positive=info["n_positive"], n_applied=info["n_applied"],
                         predicted_gain_pp=100.0 * info["predicted_gain"] / total,
                         realised_gain_pp=100.0 * realised / total,
                         exact=bool(exact), wall=time.time() - t1,
                         cum_wall=time.time() - t_start + t_init))
        print(f"  sweep {s:>2}: {pct(new_score, total):.5f}%  "
              f"applied={info['n_applied']:>5}/{info['n_positive']:<6} "
              f"gain={100.0*realised/total:+.5f} pp  exact={exact}  "
              f"{time.time()-t1:.0f}s", flush=True)
        if info["n_applied"] == 0 or realised <= 0:
            break
        rank, score = new_rank, new_score

    final = pct(score, total)
    print(f"\nFINAL {final:.5f}%", flush=True)
    for k, v in ref.items():
        print(f"  vs {k:<12} {v:.5f}  ->  {final - v:+.5f} pp", flush=True)

    out = dict(item="H50", dataset=dataset, top_k=top_k, max_sweeps=max_sweeps,
               init_pct=pct(score_from_order(ratio_greedy_rank(g), src, tgt, g.weight), total),
               t_init_s=t_init, sweeps=rows, final_pct=final, comparators=ref,
               deltas={k: final - v for k, v in ref.items()},
               all_sweeps_exact=bool(all(r["exact"] for r in rows)))
    op = _ROOT / "experiments" / "outputs" / f"proto_H50_{dataset}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
