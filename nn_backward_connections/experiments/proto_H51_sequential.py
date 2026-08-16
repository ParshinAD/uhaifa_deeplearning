"""H51 — SEQUENTIAL exact-gain pair relocation. Gate 1, and the champion-order test.

What H50 established
--------------------
Iterated pair relocation finds ~100,000 improving moves per sweep on connectome and, under the
campaign's batched-disjoint architecture, applies **0.104 %** of them — because a pair move's
interval averages ~20,536 positions and disjoint intervals of that size barely fit on a
136,648-position line. The proposed explanation was that the *architecture*, not the move class,
is what fails.

What gate 1 measured (``--start ratio``, same order H50 used)
--------------------------------------------------------------

===========================  ==========  ==============================
arm                          moves       gain
===========================  ==========  ==============================
H50 batched, sweep 0         846         +0.10931 pp
sequential                   1,000       **+0.45127 pp**  (4.13x per move)
sequential                   20,000      +3.02241 pp in 131 s
sequential, full heap drain  159,671     **+7.74682 pp** (74.61730 -> 82.36412)
===========================  ==========  ==============================

Against H50's batched arm — 40 sweeps, 2,150 s, +0.67573 pp — sequential is ~11x the gain in
about half the time. The throughput diagnosis is confirmed, and every checkpoint was exact
(predicted == oracle-realised).

A second finding, contrary to H51's own filing: the naive ``O(interval)`` rebuild is **cheap
enough**. 20,000 moves at a mean span of 22,686 positions cost 131 s. The Dietz-Sleator
order-maintenance structure this item was filed as needing is probably unnecessary, which
downgrades its cost from "high" to "medium".

The two starting orders
-----------------------
``ratio``     the reference route's own start (74.61730 %) — a ROUTE study.
``champion``  the champion's order from ``sota.json``. This is the score-relevant test: H45
              measured this same move class here with BATCHED application and got +0.00193 pp,
              applying 185 of 3,498 positive-gain candidates. Sequential application should
              realise nearly all of them. A gain above the 0.012 pp minimum effect size here
              would be a champion-beating variant, not a route study.

``n_passes`` refills the heap from the CURRENT backward edges between passes, which is what the
reference's Algorithm 2 does by pushing newly-created backward edges. One pass is a lower bound.

Leakage-safety: moves are chosen from edge weights and current ranks alone; the frozen oracle is
called only to score whole rank vectors at checkpoints and pass boundaries. ``data/best_solution``
is never read.

Run:  PYTHONPATH=src python experiments/proto_H51_sequential.py <dataset> <target> <start> <passes>
Out:  experiments/outputs/proto_H51_<dataset>_<start>.json
"""
from __future__ import annotations

import glob
import heapq
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

CHECKPOINTS = (1, 10, 100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
               100_000, 200_000, 300_000, 500_000, 750_000, 1_000_000)
BATCH_REF = {"connectome": dict(moves=846, gain_pp=0.10931)}


def build_csr(row, col, val, n):
    cnt = np.bincount(row, minlength=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(cnt, out=ptr[1:])
    o = np.argsort(row, kind="stable")
    return ptr, col[o].copy(), val[o].copy()


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "connectome"
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    start_mode = sys.argv[3] if len(sys.argv) > 3 else "ratio"
    n_passes = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    g = io.load_dataset(dataset)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    n, total = g.n_nodes, g.total_weight

    keep = src != tgt
    ks, kt, kw = src[keep], tgt[keep], w[keep]
    out_ptr, out_idx, out_w = build_csr(ks, kt, kw, n)
    in_ptr, in_idx, in_w = build_csr(kt, ks, kw, n)

    champ_pct = None
    if start_mode == "champion":
        sota = json.loads((_ROOT / "autoresearch" / "sota.json").read_text())
        champ = sota["datasets"][dataset]["champion"]
        pos_path = None
        for f in reversed(sorted(glob.glob(f"results/*-{champ}-{dataset}-*.json"))):
            d0 = json.load(open(f))
            pth = d0.get("best_positions_path")
            if pth and Path(pth).exists():
                pos_path, champ_pct = pth, d0["pct"]
                break
        if pos_path is None:
            print(f"no position vector for champion {champ} on {dataset}")
            return 2
        positions = np.load(pos_path)
        rank = np.argsort(np.argsort(positions, kind="stable"),
                          kind="stable").astype(np.int64)
        print(f"start = CHAMPION {champ}  ({Path(pos_path).name})", flush=True)
    else:
        rank = ratio_greedy_rank(g)
        print("start = ratio greedy", flush=True)

    seq = np.empty(n, dtype=np.int64)
    seq[rank] = np.arange(n, dtype=np.int64)
    start_score = score_from_order(rank, src, tgt, g.weight)
    print(f"start score: {pct(start_score, total):.6f}%", flush=True)
    if champ_pct is not None:
        print(f"  (champion record {champ_pct})", flush=True)

    def pair_gain(u, v):
        """Exact gain and best split for relocating u, v adjacent. O(d(u)+d(v))."""
        lo, hi = int(rank[v]), int(rank[u])
        a, b = out_ptr[u], out_ptr[u + 1]
        s_ = out_idx[a:b]
        h = np.flatnonzero(s_ == v)
        w_uv = float(out_w[a:b][h].sum()) if h.size else 0.0
        a, b = out_ptr[v], out_ptr[v + 1]
        s_ = out_idx[a:b]
        h = np.flatnonzero(s_ == u)
        w_vu = float(out_w[a:b][h].sum()) if h.size else 0.0
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
            return base, lo, lo, hi
        pos = sorted(set(acc_v) | set(acc_u))
        suffix = sum(acc_u.values())
        gain, cut = base + suffix, lo
        pref = 0.0
        for p in pos:
            pref += acc_v.get(p, 0.0)
            suffix -= acc_u.get(p, 0.0)
            gv = base + pref + suffix
            if gv > gain:
                gain, cut = gv, p
        return gain, cut, lo, hi

    def refill():
        b = np.flatnonzero(rank[src] > rank[tgt])
        h = [(-float(w[e]), int(e)) for e in b]
        heapq.heapify(h)
        return h

    applied = popped = 0
    predicted_total = 0.0
    span_total = 0
    rows, pass_rows = [], []
    t0 = time.time()

    for pass_i in range(n_passes):
        heap = refill()
        p_start_score = score_from_order(rank, src, tgt, g.weight)
        p_applied0 = applied
        print(f"-- pass {pass_i}: {len(heap):,} backward edges, "
              f"at {pct(p_start_score, total):.6f}%", flush=True)

        while heap and applied < target:
            _, ei = heapq.heappop(heap)
            popped += 1
            u, v = int(src[ei]), int(tgt[ei])
            if rank[u] <= rank[v]:
                continue
            gain, cut, lo, hi = pair_gain(u, v)
            if gain <= 1e-9:
                continue
            block = [x for x in seq[lo:hi + 1] if x != u and x != v]
            out_l, placed = [], False
            for x in block:
                out_l.append(x)
                if rank[x] == cut:
                    out_l.extend((u, v))
                    placed = True
            if not placed:
                out_l = [u, v] + out_l
            seq[lo:hi + 1] = np.asarray(out_l, dtype=np.int64)
            rank[seq[lo:hi + 1]] = np.arange(lo, hi + 1, dtype=np.int64)
            applied += 1
            predicted_total += gain
            span_total += (hi - lo)
            if applied in CHECKPOINTS:
                sc = score_from_order(rank, src, tgt, g.weight)
                realised = sc - start_score
                rows.append(dict(applied=applied, popped=popped, pct=pct(sc, total),
                                 realised_gain_pp=100.0 * realised / total,
                                 predicted_gain_pp=100.0 * predicted_total / total,
                                 exact=bool(abs(realised - predicted_total) < 1e-6),
                                 mean_span=span_total / applied,
                                 wall=time.time() - t0))
                r = rows[-1]
                print(f"  applied={applied:>7} popped={popped:>8} {r['pct']:.6f}%  "
                      f"gain={r['realised_gain_pp']:+.5f} pp  exact={r['exact']}  "
                      f"{r['wall']:.0f}s", flush=True)

        p_end = score_from_order(rank, src, tgt, g.weight)
        pass_rows.append(dict(pass_i=pass_i, applied=applied - p_applied0,
                              start_pct=pct(p_start_score, total),
                              end_pct=pct(p_end, total),
                              gain_pp=100.0 * (p_end - p_start_score) / total,
                              wall=time.time() - t0))
        pr = pass_rows[-1]
        print(f"-- pass {pass_i} END: {pr['end_pct']:.6f}%  "
              f"(+{pr['gain_pp']:.5f} pp, {pr['applied']:,} moves, {pr['wall']:.0f}s)",
              flush=True)
        if pr["applied"] == 0 or pr["gain_pp"] <= 0.0:
            print("   no further improvement - stopping", flush=True)
            break
        if applied >= target:
            break

    final = score_from_order(rank, src, tgt, g.weight)
    total_gain = 100.0 * (final - start_score) / total
    print(f"\nFINAL {pct(final, total):.6f}%   total gain {total_gain:+.5f} pp   "
          f"applied={applied:,}  popped={popped:,}", flush=True)
    if champ_pct is not None:
        print(f"vs CHAMPION {champ_pct}: {pct(final, total) - champ_pct:+.5f} pp", flush=True)

    out = dict(item="H51", dataset=dataset, start_mode=start_mode, n_passes=n_passes,
               start_pct=pct(start_score, total), champion_pct=champ_pct,
               final_pct=pct(final, total), total_gain_pp=total_gain,
               applied=applied, popped=popped, checkpoints=rows, passes=pass_rows,
               batch_reference=BATCH_REF.get(dataset),
               delta_vs_champion_pp=((pct(final, total) - champ_pct)
                                     if champ_pct is not None else None))
    op = _ROOT / "experiments" / "outputs" / f"proto_H51_{dataset}_{start_mode}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
