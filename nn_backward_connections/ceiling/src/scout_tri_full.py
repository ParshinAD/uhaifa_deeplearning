"""Scout 3b: exact enumeration of all directed 3-cycles of the residual graph (after exact 2-cycle
packing) by chunked sparse matmul, then one greedy integral packing pass over them in random order.
Same LB construction as the old probe, but vectorized enumeration; a bound only, no certificate written."""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mfas.io import load_dataset

CONFIG = dict(seed=0, chunk=4000, out="/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/tri3_residual.npz")

t0 = time.time()
g = load_dataset("connectome")
si, ti, w = g.src.astype(np.int64), g.tgt.astype(np.int64), g.weight.astype(np.int64)
n, m, W = g.n_nodes, g.n_edges, int(w.sum())
key = si * n + ti
order = np.argsort(key)
ks = key[order]
rk = ti * n + si
pos = np.clip(np.searchsorted(ks, rk), 0, m - 1)
has = ks[pos] == rk
pack2 = np.where(has, np.minimum(w, w[order[pos]]), 0)
lb2 = int(pack2.sum() // 2)
cap = w - pack2
k = cap > 0
A = sp.csr_matrix((np.ones(int(k.sum()), np.float32), (si[k], ti[k])), shape=(n, n))
AT = A.T.tocsr()


def eid(a, b):
    kk = a * n + b
    p = np.searchsorted(ks, kk)
    assert np.all(ks[p] == kk)
    return order[p]


tris = []
t1 = time.time()
for r0 in range(0, n, CONFIG["chunk"]):
    rows = np.arange(r0, min(n, r0 + CONFIG["chunk"]))
    P = (A[rows] @ A).multiply(AT[rows]).tocoo()
    u = rows[P.row]
    x = P.col.astype(np.int64)
    keepu = (u < x)
    u, x = u[keepu], x[keepu]
    if u.size == 0:
        continue
    tris.append(np.column_stack([u, x]))
pairs = np.vstack(tris)
print("closing pairs (u<x, x->u, path u->?->x) count", pairs.shape[0], "s", round(time.time() - t1, 1), flush=True)

# expand middle vertices: v in out(u) & in(x)
t1 = time.time()
ip, ix = A.indptr, A.indices
ipT, ixT = AT.indptr, AT.indices
U, V, X = [], [], []
for uu, xx in pairs:
    mid = np.intersect1d(ix[ip[uu]:ip[uu + 1]], ixT[ipT[xx]:ipT[xx + 1]], assume_unique=True)
    U.append(np.full(mid.size, uu))
    V.append(mid)
    X.append(np.full(mid.size, xx))
U, V, X = np.concatenate(U), np.concatenate(V).astype(np.int64), np.concatenate(X)
# keep u = smallest node of the cycle (u<x already; need u<v) -> each directed 3-cycle once
sel = U < V
U, V, X = U[sel], V[sel], X[sel]
print("directed 3-cycles in residual graph", U.size, "expand s", round(time.time() - t1, 1), flush=True)

e1, e2, e3 = eid(U, V), eid(V, X), eid(X, U)
mincap = np.minimum(np.minimum(cap[e1], cap[e2]), cap[e3])
np.savez(CONFIG["out"], e1=e1, e2=e2, e3=e3, cap=cap, lb2=lb2)
print("sum of min residual cap over 3-cycles (no sharing)", int(mincap.sum()))
rng = np.random.default_rng(CONFIG["seed"])
perm = rng.permutation(U.size)
t1 = time.time()
c = cap.copy()
gain = 0
for i in perm:
    a, b, d = e1[i], e2[i], e3[i]
    y = min(c[a], c[b], c[d])
    if y > 0:
        c[a] -= y
        c[b] -= y
        c[d] -= y
        gain += int(y)
print("greedy random-order packing of all 3-cycles: gain", gain, "s", round(time.time() - t1, 1))
lb = lb2 + gain
print("LB 2-cycle + all-3-cycle greedy", lb, "ceiling", round(100 * (W - lb) / W, 4))
perm = np.argsort(-mincap, kind="stable")
c = cap.copy()
gain = 0
for i in perm:
    a, b, d = e1[i], e2[i], e3[i]
    y = min(c[a], c[b], c[d])
    if y > 0:
        c[a] -= y
        c[b] -= y
        c[d] -= y
        gain += int(y)
lb = lb2 + gain
print("greedy max-mincap-first packing: gain", gain, "LB", lb, "ceiling", round(100 * (W - lb) / W, 4))
print("total s", round(time.time() - t0, 1))
