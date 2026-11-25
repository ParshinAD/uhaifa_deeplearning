"""Pricing for the cycle packing LP: under edge lengths l_e = x_e / w_e (x = PDHG dual), find
cycles with l(C) < 1 - margin (violated dual constraints). For each source s: Dijkstra from s
(limit 1), close with every in-edge t -> s, keep the best k cycles of length >= min_len.
Usage: m_short_price.py <pool.npz> <dual.npz> <out.npz> <tag> [n_sources] [workers] [seed]"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mfas.io import load_dataset

CONFIG = dict(seed=int(sys.argv[7]) if len(sys.argv) > 7 else 0, n_sources=int(sys.argv[5]) if len(sys.argv) > 5 else 2000,
              workers=int(sys.argv[6]) if len(sys.argv) > 6 else 5, k_per_source=20, min_len=4,
              margin=1e-4, floor_len=1e-3, batch=20)

G = {}


def init():
    g = load_dataset("connectome")
    si, ti, w = g.src.astype(np.int64), g.tgt.astype(np.int64), g.weight.astype(np.float64)
    n, m = g.n_nodes, g.n_edges
    x = np.load(sys.argv[2])["x"]
    l = np.maximum(x / w, 0) + CONFIG["floor_len"]
    order = np.lexsort((ti, si))
    A = sp.csr_matrix((l[order], (si[order], ti[order])), shape=(n, n))
    A.sort_indices()
    key = si * n + ti
    ko = np.argsort(key)
    inp = np.argsort(ti, kind="stable")
    inptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(ti, minlength=n), out=inptr[1:])
    G.update(A=A, l=l, si=si, ti=ti, n=n, m=m, ks=key[ko], ko=ko, inp=inp, inptr=inptr)


def eid(a, b):
    kk = a * G["n"] + b
    p = np.searchsorted(G["ks"], kk)
    return G["ko"][p]


def work(srcs):
    out = []
    l, si, n = G["l"], G["si"], G["n"]
    d, pred = dijkstra(G["A"], directed=True, indices=srcs, return_predecessors=True, limit=1.0)
    for r, s in enumerate(srcs):
        ein = G["inp"][G["inptr"][s]:G["inptr"][s + 1]]
        t = si[ein]
        cl = d[r, t] + l[ein]
        ok = np.flatnonzero(cl < 1 - CONFIG["margin"])
        found = 0
        for j in ok[np.argsort(cl[ok])]:
            path = [int(t[j])]
            while path[-1] != s and len(path) <= n:
                path.append(int(pred[r, path[-1]]))
            if len(path) < CONFIG["min_len"]:
                continue
            nodes = np.array(path[::-1], np.int64)
            e = np.concatenate([eid(nodes[:-1], nodes[1:]), [ein[j]]])
            out.append((float(cl[j]), e))
            found += 1
            if found >= CONFIG["k_per_source"]:
                break
    return out


if __name__ == "__main__":
    t0 = time.time()
    init()
    n = G["n"]
    rng = np.random.default_rng(CONFIG["seed"])
    deg_in = np.diff(G["inptr"])
    cand = np.flatnonzero(deg_in > 0)
    srcs = rng.permutation(cand)[:CONFIG["n_sources"]]
    chunks = [srcs[i:i + CONFIG["batch"]] for i in range(0, srcs.size, CONFIG["batch"])]
    ctx = mp.get_context("fork")
    res = []
    with ctx.Pool(CONFIG["workers"]) as pool:
        for i, r in enumerate(pool.imap_unordered(work, chunks)):
            res.extend(r)
            if i % 50 == 0:
                print("chunks", i, "of", len(chunks), "cycles", len(res), "s", round(time.time() - t0, 1), flush=True)
    seen, groups, lens_found = set(), {}, []
    for cl, e in res:
        r = int(np.argmin(e))
        key = tuple(np.roll(e, -r).tolist())
        if key in seen:
            continue
        seen.add(key)
        groups.setdefault(len(key), []).append(key)
        lens_found.append(cl)
    tag = sys.argv[4]
    arrs = {"c" + str(k) + "_" + tag: np.array(v, np.int64) for k, v in sorted(groups.items())}
    np.savez(sys.argv[3], **arrs)
    info = dict(config=CONFIG, sources=int(srcs.size), raw=len(res), unique=len(seen),
                by_len={k: len(v) for k, v in sorted(groups.items())},
                mean_dual_len=float(np.mean(lens_found)) if lens_found else None, runtime_s=round(time.time() - t0, 1))
    Path(sys.argv[3]).with_suffix(".json").write_text(json.dumps(info))
    print(info)
