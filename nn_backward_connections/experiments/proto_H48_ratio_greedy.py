"""H48 prototype gate — ratio-greedy initial order vs our greedy-FAS.

The question
------------
The reference family's Algorithm 1 is a greedy peel by ``score(u) = (out_w(u)+1)/(in_w(u)+1)``
with a max-heap and scores updated as nodes are removed. It is reported to reach **75.24%** on
this exact connectome graph. Our ``mfas.experiments.H02.greedy_fas_order`` reaches
**68.91343%** — measured 2026-08-16 as the ``epochs=0`` arm of the connectome sizing grid
(``experiments/outputs/proto_S01_connectome.json``, ``arms[0].pure_pct``).

If that +6.33 pp of starting quality is real, it is the cheapest unexploited thing the campaign
has found, because both inits are the same asymptotic cost — one pass with a heap.

Why this is not blocked by the kill index
------------------------------------------
* ``M6-init-is-flat`` says the init -> plateau curve is flat on connectome and better-init-alone
  has a <=0.06 pp ceiling. That was measured **through Rocket**, whose plateau washes the init
  out — which is precisely what M6 describes. The 2026-08-16 grids measured the Rocket-free and
  Rocket-reduced regime directly for the first time, so the question is re-opened on evidence.
* ``M7-expensive-warmstarts-lose`` forbids EXPENSIVE global linear-algebra warm starts (trophic,
  magnetic Laplacian; 304-507 s). A ratio greedy is the same heap peel we already run.

What this measures
------------------
Three numbers, all from the frozen scorer:

1. our ``greedy_fas_order`` score (re-derived here, not quoted);
2. the ratio-greedy score;
3. the ratio-greedy score after the champion's stage 3 (the under-relaxed sift), so the
   comparison is init-and-refiner rather than init alone — per meta-rule M8, what matters is the
   composed result, not the intermediate.

Leakage-safety: the peel uses only edge weights and the remaining-subgraph degrees. The frozen
oracle scores whole candidate orders afterwards. ``data/best_solution`` is never read.

Run:  PYTHONPATH=src python experiments/proto_H48_ratio_greedy.py [dataset]
Out:  experiments/outputs/proto_H48_<dataset>.json
"""
from __future__ import annotations

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

from mfas import io                                          # noqa: E402
from mfas.metrics import pct, score_from_order               # noqa: E402
from mfas.experiments.H02 import greedy_fas_order            # noqa: E402
from mfas.refine.underrelax import sift_underrelaxed         # noqa: E402


def ratio_greedy_order(g) -> np.ndarray:
    """Greedy peel by ``(out_w+1)/(in_w+1)``, highest first (reference Algorithm 1).

    The node with the largest ratio is the one that most wants to be EARLY: it carries much
    out-weight and little in-weight, so placing it next makes many of its edges feedforward.
    It is appended to the order and removed, and its neighbours' remaining-degree sums are
    updated, which is what makes this different from a static sort by ratio.

    Lazy-deletion heap: a node may sit in the heap several times with stale keys; the first
    time it is popped while still alive is the one that counts. Ties break on the lowest node
    id, so the order is deterministic and draws no random numbers.

    Returns the node order (position 0 first), int64, shape (n,).
    """
    n = g.n_nodes
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    keep = src != tgt
    src, tgt, w = src[keep], tgt[keep], w[keep]

    out_w = np.zeros(n, dtype=np.float64)
    in_w = np.zeros(n, dtype=np.float64)
    np.add.at(out_w, src, w)
    np.add.at(in_w, tgt, w)

    # CSR both ways, so removing a node is O(deg).
    def csr(row, col, val):
        cnt = np.bincount(row, minlength=n)
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(cnt, out=ptr[1:])
        o = np.argsort(row, kind="stable")
        return ptr, col[o].copy(), val[o].copy()

    op, oi, ow = csr(src, tgt, w)     # u -> (v, w)
    ip, ii, iw = csr(tgt, src, w)     # v -> (u, w)

    alive = np.ones(n, dtype=bool)
    heap = [(-(out_w[u] + 1.0) / (in_w[u] + 1.0), u) for u in range(n)]
    heapq.heapify(heap)

    order = np.empty(n, dtype=np.int64)
    filled = 0
    while filled < n:
        key, u = heapq.heappop(heap)
        if not alive[u]:
            continue
        cur = -(out_w[u] + 1.0) / (in_w[u] + 1.0)
        if cur != key:                      # stale entry, re-push with the live key
            heapq.heappush(heap, (cur, u))
            continue
        alive[u] = False
        order[filled] = u
        filled += 1
        # u leaves the remaining subgraph: its out-edges stop counting toward targets' in_w,
        # and its in-edges stop counting toward sources' out_w.
        for k in range(op[u], op[u + 1]):
            v = oi[k]
            if alive[v]:
                in_w[v] -= ow[k]
                heapq.heappush(heap, (-(out_w[v] + 1.0) / (in_w[v] + 1.0), v))
        for k in range(ip[u], ip[u + 1]):
            v = ii[k]
            if alive[v]:
                out_w[v] -= iw[k]
                heapq.heappush(heap, (-(out_w[v] + 1.0) / (in_w[v] + 1.0), v))
    return order


def order_to_rank(order: np.ndarray, n: int) -> np.ndarray:
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    return rank


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "connectome"
    g = io.load_dataset(dataset)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    n, total = g.n_nodes, g.total_weight
    out = dict(item="H48", dataset=dataset, n_nodes=int(n), n_edges=int(src.size))

    # ── ours ────────────────────────────────────────────────────────────────
    t0 = time.time()
    ours = greedy_fas_order(g)
    t_ours = time.time() - t0
    # CAREFUL: despite its name, greedy_fas_order returns a RANK vector - H02's own docstring
    # says "order[u] is the rank/position of node u" - while ratio_greedy_order below returns a
    # true ORDER (order[i] = the node at position i). Converting both the same way silently
    # scrambles one of them: it showed our mouse greedy at 43.797% against the 90.126% the H44
    # run records, which is what caught it. The two conventions are NOT interchangeable.
    rank_ours = np.asarray(ours, dtype=np.int64)
    s_ours = score_from_order(rank_ours, src, tgt, g.weight)
    print(f"greedy-FAS (ours)   : {pct(s_ours, total):.5f}%   {t_ours:.1f}s", flush=True)

    # ── theirs ──────────────────────────────────────────────────────────────
    t0 = time.time()
    theirs = ratio_greedy_order(g)
    t_theirs = time.time() - t0
    rank_theirs = order_to_rank(theirs, n)
    s_theirs = score_from_order(rank_theirs, src, tgt, g.weight)
    print(f"ratio greedy (H48)  : {pct(s_theirs, total):.5f}%   {t_theirs:.1f}s", flush=True)
    print(f"  init delta        : {pct(s_theirs, total) - pct(s_ours, total):+.5f} pp", flush=True)

    out.update(
        ours_pct=pct(s_ours, total), ours_wall_s=t_ours,
        ratio_pct=pct(s_theirs, total), ratio_wall_s=t_theirs,
        init_delta_pp=pct(s_theirs, total) - pct(s_ours, total),
    )

    # ── composed: both inits through the champion's stage 3 (M8: judge the composition) ──
    sweeps = {"connectome": 40, "mouse": 40, "microns": 12}.get(dataset, 40)
    for tag, rank0 in (("ours", rank_ours), ("ratio", rank_theirs)):
        t0 = time.time()
        r, s, log = sift_underrelaxed(g, rank0, k_full=6, alpha=0.7,
                                      max_sweeps=sweeps, time_budget_s=None)
        dt = time.time() - t0
        conv = bool(log and log[-1]["n_movers"] == 0)
        print(f"  {tag:>5} + stage3   : {pct(s, total):.5f}%   {dt:.1f}s  "
              f"sweeps={len(log)}/{sweeps} converged={conv}", flush=True)
        out[f"{tag}_sift_pct"] = pct(s, total)
        out[f"{tag}_sift_wall_s"] = dt
        out[f"{tag}_sift_sweeps"] = len(log)
        out[f"{tag}_sift_converged"] = conv
    out["composed_delta_pp"] = out["ratio_sift_pct"] - out["ours_sift_pct"]
    print(f"  COMPOSED delta    : {out['composed_delta_pp']:+.5f} pp", flush=True)

    op = _ROOT / "experiments" / "outputs" / f"proto_H48_{dataset}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
