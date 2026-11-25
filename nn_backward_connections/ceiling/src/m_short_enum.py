"""Enumerate all directed 2-cycles and 3-cycles of the FULL connectome graph (no 2-cycle removal),
as edge-id columns for the joint packing LP. Output: npz with c2 (k,2) and c3 (k,3) edge ids."""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mfas.io import load_dataset

CONFIG = dict(chunk=4000, out="/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/mshort/cols23_full.npz")

t0 = time.time()
g = load_dataset("connectome")
si, ti, w = g.src.astype(np.int64), g.tgt.astype(np.int64), g.weight.astype(np.int64)
n, m = g.n_nodes, g.n_edges
key = si * n + ti
order = np.argsort(key)
ks = key[order]


def eid(a, b):
    kk = a * n + b
    p = np.searchsorted(ks, kk)
    assert np.all(ks[p] == kk)
    return order[p]


fwd = np.flatnonzero(si < ti)
rk = ti[fwd] * n + si[fwd]
p = np.clip(np.searchsorted(ks, rk), 0, m - 1)
hit = ks[p] == rk
c2 = np.column_stack([fwd[hit], order[p[hit]]])
print("2-cycles", c2.shape[0], flush=True)

A = sp.csr_matrix((np.ones(m, np.float32), (si, ti)), shape=(n, n))
AT = A.T.tocsr()
pairs = []
t1 = time.time()
for r0 in range(0, n, CONFIG["chunk"]):
    rows = np.arange(r0, min(n, r0 + CONFIG["chunk"]))
    P = (A[rows] @ A).multiply(AT[rows]).tocoo()
    u = rows[P.row]
    x = P.col.astype(np.int64)
    keep = u < x
    pairs.append(np.column_stack([u[keep], x[keep]]))
pairs = np.vstack(pairs)
print("closing pairs", pairs.shape[0], "s", round(time.time() - t1, 1), flush=True)
t1 = time.time()
ip, ix = A.indptr, A.indices
ipT, ixT = AT.indptr, AT.indices
U, V, X = [], [], []
for uu, xx in pairs:
    mid = np.intersect1d(ix[ip[uu]:ip[uu + 1]], ixT[ipT[xx]:ipT[xx + 1]], assume_unique=True)
    mid = mid[mid > uu]
    U.append(np.full(mid.size, uu))
    V.append(mid)
    X.append(np.full(mid.size, xx))
U, V, X = np.concatenate(U), np.concatenate(V).astype(np.int64), np.concatenate(X)
print("3-cycles", U.size, "expand s", round(time.time() - t1, 1), flush=True)
c3 = np.column_stack([eid(U, V), eid(V, X), eid(X, U)])
np.savez(CONFIG["out"], c2=c2, c3=c3)
print("saved", CONFIG["out"], "total s", round(time.time() - t0, 1))
