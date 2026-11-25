"""Fractional packing of explicit cycle columns (Garg-Konemann style multiplicative weights,
all near-shortest columns routed per round). Tracks a feasible primal (each y_C divided by the max congestion on its edges)
and the dual upper estimate sum w l / min_C l(C) over the same column set.
Usage: m_short_mwu.py <cols.npz> <out.npz> [eps] [max_rounds]"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mfas.io import load_dataset

CONFIG = dict(eps=0.05, max_rounds=4000, report_every=100, time_limit_s=3000)
if len(sys.argv) > 3:
    CONFIG["eps"] = float(sys.argv[3])
if len(sys.argv) > 4:
    CONFIG["max_rounds"] = int(sys.argv[4])

t0 = time.time()
g = load_dataset("connectome")
w = g.weight.astype(np.float64)
m = g.n_edges
z = np.load(sys.argv[1])
cols = [z[k] for k in z.files if k.startswith("c")]
ptr = np.cumsum([0] + [c.shape[0] for c in cols])
K = int(ptr[-1])
print("columns", K, "by length", [c.shape for c in cols], flush=True)
eps = CONFIG["eps"]
l = 1.0 / w
y = [np.zeros(c.shape[0]) for c in cols]


def lengths(l):
    return [l[c].sum(axis=1) for c in cols]


best_lb, best_ub = 0.0, np.inf
hist = []
for r in range(CONFIG["max_rounds"]):
    L = lengths(l)
    alpha = min(x.min() for x in L)
    ub = float((w * l).sum() / alpha)
    best_ub = min(best_ub, ub)
    sel = [np.flatnonzero(x <= alpha * (1 + eps)) for x in L]
    load = np.zeros(m)
    for c, s in zip(cols, sel):
        load += np.bincount(c[s].ravel(), minlength=m)
    inc = np.zeros(m)
    for i, (c, s) in enumerate(zip(cols, sel)):
        f = (w[c[s]] / load[c[s]]).min(axis=1)
        y[i][s] += f
        inc += np.bincount(c[s].ravel(), weights=np.repeat(f, c.shape[1]), minlength=m)
    l *= 1 + eps * inc / w
    l /= l.max()
    if r % CONFIG["report_every"] == 0 or r == CONFIG["max_rounds"] - 1:
        tot = np.zeros(m)
        for c, yy in zip(cols, y):
            tot += np.bincount(c.ravel(), weights=np.repeat(yy, c.shape[1]), minlength=m)
        cong = tot / w
        lb = float(sum((yy / cong[c].max(axis=1).clip(1e-12)).sum() for c, yy in zip(cols, y)))
        best_lb = max(best_lb, lb)
        hist.append([r, round(time.time() - t0, 1), lb, ub])
        print("round", r, "sel", sum(s.size for s in sel), "primal", round(lb, 1), "dual", round(ub, 1),
              "best", round(best_lb, 1), round(best_ub, 1), "gap%", round(100 * (best_ub - best_lb) / best_ub, 3),
              "s", round(time.time() - t0, 1), flush=True)
        if time.time() - t0 > CONFIG["time_limit_s"]:
            break
tot = np.zeros(m)
for c, yy in zip(cols, y):
    tot += np.bincount(c.ravel(), weights=np.repeat(yy, c.shape[1]), minlength=m)
cong = tot / w
yall = np.concatenate([yy / cong[c].max(axis=1).clip(1e-12) for c, yy in zip(cols, y)])
yall[~np.isfinite(yall)] = 0
np.savez(sys.argv[2], y=yall, l=l, ptr=ptr)
out = dict(config=CONFIG, columns=K, final_primal=float(yall.sum()), best_primal=best_lb, best_dual=best_ub,
           runtime_s=round(time.time() - t0, 1), hist=hist)
Path(sys.argv[2]).with_suffix(".json").write_text(json.dumps(out))
print("final primal", round(float(yall.sum()), 1), "best dual", round(best_ub, 1), "s", round(time.time() - t0, 1))
