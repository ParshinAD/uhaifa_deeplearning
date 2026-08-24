"""Certified UPPER BOUND on the achievable feedforward share (connectome).

Weak LP duality for weighted MFAS: any feasible *fractional cycle packing*
(x_c >= 0 with sum_{c ∋ e} x_c <= w_e) lower-bounds the minimum feedback weight,
because every feedback arc set must hit every cycle.  We build an *integral*
packing greedily, so the bound is valid but not tight:

  stage 1 (exact, O(m log m)): every reciprocal pair (u,v),(v,u) is a 2-cycle;
          pack min(w_uv, w_vu) copies.  Disjoint across pairs -> feasible.
  stage 2 (greedy, time-boxed): on the residual capacities, list 3-cycles
          u->v->x->u and pack min residual capacity.

  UPPER BOUND on feedforward share = (W - packed) / W.

Measured 2026-08-19 on data/connectome_graph.csv.gz (n=136,648, m=5,657,719,
W=41,912,141):
    stage 1 alone              -> 2,815,303 feedback  => <= 93.2828 %
    stage 1 + 420 s of stage 2 -> 4,537,757 feedback  => <= 89.1732 %
(best known solution on this instance = 84.6147 %)

Run:  python experiments/diagnostics/probe_cyclepacking_upper_bound.py [seconds]
"""
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp


def main(budget_s: float = 420.0) -> None:
    df = pd.read_csv("data/connectome_graph.csv.gz")
    s = df.iloc[:, 0].to_numpy()
    t = df.iloc[:, 1].to_numpy()
    w = df.iloc[:, 2].to_numpy().astype(np.int64)
    nodes = np.unique(np.concatenate([s, t]))
    si = np.searchsorted(nodes, s).astype(np.int64)
    ti = np.searchsorted(nodes, t).astype(np.int64)
    n, m, W = len(nodes), len(si), int(w.sum())

    key = si * n + ti
    order = np.argsort(key)
    ks, ws = key[order], w[order]
    pos = np.clip(np.searchsorted(ks, ti * n + si), 0, m - 1)
    has_rev = ks[pos] == (ti * n + si)
    w_rev = np.where(has_rev, ws[pos], 0)
    pack2 = np.where(has_rev, np.minimum(w, w_rev), 0)
    lb2 = int(pack2[has_rev].sum() // 2)
    print(f"n={n} m={m} W={W}")
    print(f"stage 1 (2-cycles): {lb2}  => upper bound {100 * (W - lb2) / W:.4f} %")

    cap = (w - pack2).astype(np.int64)
    keep = cap > 0
    A = sp.csr_matrix(
        (np.ones(int(keep.sum()), np.int8), (si[keep], ti[keep])), shape=(n, n)
    )
    A.sum_duplicates()
    indptr, indices = A.indptr, A.indices
    kk = si * n + ti
    ordk = np.argsort(kk)
    kks = kk[ordk]

    rng = np.random.default_rng(0)
    cand = np.flatnonzero(keep)
    rng.shuffle(cand)
    t0, gain, tried = time.time(), 0, 0
    for e in cand:
        if time.time() - t0 > budget_s:
            break
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
        for a, b in zip(id_fwd, id_back):
            c = min(cap[e], cap[a], cap[b])
            if c > 0:
                cap[e] -= c
                cap[a] -= c
                cap[b] -= c
                gain += int(c)
                if cap[e] <= 0:
                    break
    tot = lb2 + gain
    print(f"stage 2 (3-cycles): +{gain} after {tried}/{cand.size} seed edges")
    print(f"TOTAL feedback lower bound {tot}"
          f"  => UPPER BOUND on feedforward = {100 * (W - tot) / W:.4f} %")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 420.0)
