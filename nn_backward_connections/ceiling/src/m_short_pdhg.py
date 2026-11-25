"""Packing LP max sum y s.t. sum_{C ni e} y_C <= w_e over explicit cycle columns, by diagonally
preconditioned PDHG (Pock-Chambolle) with restarts to the average. Each check gives
primal: y_C / max(1, max congestion on C)   (feasible packing)
dual:   Lagrangian bound sum_e x_e + sum_C min_e w_e * max(0, 1 - sum_{e in C} x_e / w_e)  (upper estimate).
Usage: m_short_pdhg.py <cols.npz> <out.npz> [omega] [iters] [warm.npz]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mfas.io import load_dataset

CONFIG = dict(omega=1.0, iters=20000, check_every=200, time_limit_s=3300)
if len(sys.argv) > 3:
    CONFIG["omega"] = float(sys.argv[3])
if len(sys.argv) > 4:
    CONFIG["iters"] = int(sys.argv[4])

t0 = time.time()
g = load_dataset("connectome")
w = g.weight.astype(np.float64)
m = g.n_edges
z = np.load(sys.argv[1])
keys = [k for k in z.files if k.startswith("c")]
cols = [z[k].astype(np.int64) for k in keys]
K = sum(c.shape[0] for c in cols)
rows = np.concatenate([c.ravel() for c in cols])
colid = np.concatenate([np.repeat(np.arange(c.shape[0]) + o, c.shape[1]) for c, o in
                        zip(cols, np.cumsum([0] + [c.shape[0] for c in cols])[:-1])])
A = sp.csr_matrix((1.0 / w[rows], (rows, colid)), shape=(m, K))
AT = A.T.tocsr()
del rows, colid
ybar = np.concatenate([w[c].min(axis=1) for c in cols])
T = CONFIG["omega"] / np.asarray(AT.sum(axis=1)).ravel()
rs = np.asarray(A.sum(axis=1)).ravel()
S = np.where(rs > 0, 1.0 / np.where(rs > 0, rs, 1) / CONFIG["omega"], 0.0)
print("columns", K, "rows used", int((rs > 0).sum()), "build s", round(time.time() - t0, 1), flush=True)

y = np.zeros(K)
x = np.zeros(m)
if len(sys.argv) > 5:
    wz = np.load(sys.argv[5])
    x = wz["x"].copy()
    y[:wz["ylast"].size] = wz["ylast"]


def bounds(y, x):
    Ay = A @ y
    cong = np.maximum(Ay, 1.0)
    mx = np.concatenate([cong[c].max(axis=1) for c in cols])
    yp = y / mx
    red = 1.0 - AT @ x
    ub = float(x[rs > 0].sum() + (ybar * np.maximum(red, 0)).sum())
    return float(yp.sum()), ub, yp


ysum, xsum, cnt = np.zeros(K), np.zeros(m), 0
best_lb, best_ub, best_y = 0.0, np.inf, None
hist = []
for it in range(1, CONFIG["iters"] + 1):
    yn = np.maximum(0.0, y + T * (1.0 - AT @ x))
    x = np.maximum(0.0, x + S * (A @ (2 * yn - y) - 1.0))
    y = yn
    ysum += y
    xsum += x
    cnt += 1
    if it % CONFIG["check_every"] == 0:
        lb, ub, yp = bounds(y, x)
        lba, uba, ypa = bounds(ysum / cnt, xsum / cnt)
        if lb > best_lb:
            best_lb, best_y = lb, yp
        if lba > best_lb:
            best_lb, best_y = lba, ypa
        best_ub = min(best_ub, ub, uba)
        if uba - lba < ub - lb:
            y, x = ysum / cnt, xsum / cnt
        ysum[:] = 0
        xsum[:] = 0
        cnt = 0
        hist.append([it, round(time.time() - t0, 1), best_lb, best_ub])
        print("it", it, "cur", round(lb, 1), round(ub, 1), "avg", round(lba, 1), round(uba, 1), "best", round(best_lb, 1),
              round(best_ub, 1), "gap%", round(100 * (best_ub - best_lb) / best_ub, 4), "s", round(time.time() - t0, 1), flush=True)
        if time.time() - t0 > CONFIG["time_limit_s"]:
            break
np.savez(sys.argv[2], y=best_y, x=x, ylast=y)
out = dict(config=CONFIG, columns=K, best_primal=best_lb, best_dual=best_ub, runtime_s=round(time.time() - t0, 1), hist=hist)
Path(sys.argv[2]).with_suffix(".json").write_text(json.dumps(out))
print("best primal", round(best_lb, 1), "best dual", round(best_ub, 1), "s", round(time.time() - t0, 1))
