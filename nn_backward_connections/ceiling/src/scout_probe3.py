"""Scout 3: re-run of experiments/diagnostics/probe_cyclepacking_upper_bound.py logic with LB checkpoints.

Same algorithm (exact 2-cycle packing, then greedy 3-cycles over shuffled seed edges, seed 0),
but loads via src.mfas.io.load_dataset and prints LB at time checkpoints."""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mfas.io import load_dataset

CONFIG = dict(budget_s=900.0, report_every_s=30.0, checkpoints=[60, 120, 420, 900], seed=0)

g = load_dataset("connectome")
si, ti, w = g.src.astype(np.int64), g.tgt.astype(np.int64), g.weight.astype(np.int64)
n, m, W = g.n_nodes, g.n_edges, int(w.sum())

key = si * n + ti
order = np.argsort(key)
ks, ws = key[order], w[order]
pos = np.clip(np.searchsorted(ks, ti * n + si), 0, m - 1)
has_rev = ks[pos] == (ti * n + si)
w_rev = np.where(has_rev, ws[pos], 0)
pack2 = np.where(has_rev, np.minimum(w, w_rev), 0)
lb2 = int(pack2[has_rev].sum() // 2)
print("n", n, "m", m, "W", W, "stage1 LB", lb2, "ceiling", round(100 * (W - lb2) / W, 4), flush=True)

cap = (w - pack2).astype(np.int64)
keep = cap > 0
A = sp.csr_matrix((np.ones(int(keep.sum()), np.int8), (si[keep], ti[keep])), shape=(n, n))
A.sum_duplicates()
indptr, indices = A.indptr, A.indices
ordk = order
kks = ks

rng = np.random.default_rng(CONFIG["seed"])
cand = np.flatnonzero(keep)
rng.shuffle(cand)
t0, gain, tried, triangles_seen, packed = time.time(), 0, 0, 0, 0
next_rep = CONFIG["report_every_s"]
cps = list(CONFIG["checkpoints"])
done = True
for e in cand:
    el = time.time() - t0
    if el > CONFIG["budget_s"]:
        done = False
        break
    if el > next_rep:
        lb = lb2 + gain
        print("t", round(el, 1), "seeds", tried, "of", cand.size, "triangles_seen", triangles_seen,
              "packed_triangles", packed, "LB", lb, "ceiling", round(100 * (W - lb) / W, 4), flush=True)
        next_rep += CONFIG["report_every_s"]
    while cps and el > cps[0]:
        lb = lb2 + gain
        print("CHECKPOINT", cps.pop(0), "s LB", lb, "ceiling", round(100 * (W - lb) / W, 4), "seeds", tried, flush=True)
    tried += 1
    if cap[e] <= 0:
        continue
    u, v = int(si[e]), int(ti[e])
    out_v = indices[indptr[v]:indptr[v + 1]]
    if out_v.size == 0:
        continue
    k_back = out_v.astype(np.int64) * n + u
    p = np.clip(np.searchsorted(kks, k_back), 0, m - 1)
    ok = kks[p] == k_back
    if not ok.any():
        continue
    id_back = ordk[p[ok]]
    p2 = np.clip(np.searchsorted(kks, np.int64(v) * n + out_v[ok]), 0, m - 1)
    id_fwd = ordk[p2]
    triangles_seen += int(ok.sum())
    for a, b in zip(id_fwd, id_back):
        c = min(cap[e], cap[a], cap[b])
        if c > 0:
            cap[e] -= c
            cap[a] -= c
            cap[b] -= c
            gain += int(c)
            packed += 1
            if cap[e] <= 0:
                break
el = time.time() - t0
lb = lb2 + gain
print("FINAL completed" if done else "FINAL stopped at budget", "t", round(el, 1), "seeds", tried, "of", cand.size,
      "triangles_seen", triangles_seen, "packed", packed, "LB", lb, "ceiling", round(100 * (W - lb) / W, 4), flush=True)
print("seed edges per s", round(tried / el, 1), "triangles seen per s", round(triangles_seen / el, 1))
print("residual weight left", int(cap[keep].sum()))
