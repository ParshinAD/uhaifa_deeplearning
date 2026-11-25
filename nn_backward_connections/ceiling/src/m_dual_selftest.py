"""Self-test of the dual checker (run from repo root, prints PASS/FAIL lines).

1. Random small digraphs, random int lengths with many zeros: min_cycle, pdist and cyc_through vs
   Floyd-Warshall (min over edges X_e + d(v, u)), for 1 and 6 threads.
2. Brute force on tiny graphs: tau* from the LP over ALL simple cycles (networkx), UB from the
   scaled LP solution must be >= tau* and close to it.
3. Calibration instances with known cycle LP (mouse, fly pieces from ceiling/results/m_calib_*.json):
   lazy cycle LP x (m_calib_core), X = ceil(S x), exact UB must be >= cycLP of the JSON and within
   1e-4 relative of it."""
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import m_dual_core as core
import m_dual_check as chk
import m_calib_core as cal

CONFIG = dict(seed=11, graphs=300, S=10 ** 6, pieces_file="ceiling/results/m_calib_pieces_rc.json",
              n_pieces=12, rel_tol=1e-4)
rng = np.random.default_rng(CONFIG["seed"])
fails = 0


def floyd_int(k, s, t, X):
    INF = 2 ** 60
    D = np.full((k, k), INF, dtype=object)
    for i in range(k):
        D[i, i] = 0
    for a, b, x in zip(s, t, X):
        D[a, b] = min(D[a, b], int(x))
    for z in range(k):
        for i in range(k):
            for j in range(k):
                if D[i, z] + D[z, j] < D[i, j]:
                    D[i, j] = D[i, z] + D[z, j]
    return D


def rand_graph(k, p):
    A = rng.random((k, k)) < p
    np.fill_diagonal(A, False)
    s, t = np.nonzero(A)
    return s.astype(np.int64), t.astype(np.int64)


# 1. kernels vs Floyd
t0 = time.time()
bad = 0
for it in range(CONFIG["graphs"]):
    k = int(rng.integers(2, 12))
    s, t = rand_graph(k, float(rng.uniform(0.1, 0.6)))
    if s.size == 0:
        continue
    X = rng.integers(0, 6, s.size) * (rng.random(s.size) < 0.6)
    D = floyd_int(k, s, t, X)
    ref = min(int(X[e]) + D[t[e], s[e]] for e in range(s.size))
    G = core.Graph(s, t, k)
    for th in (1, 6):
        L = G.min_cycle(X, threads=th)[0]
        if ref >= 2 ** 60:
            ok = L >= np.iinfo(np.int64).max // 4
        else:
            ok = L == ref
        bad += not ok
    a = rng.integers(0, k, 40)
    b = rng.integers(0, k, 40)
    cap = rng.integers(0, 20, 40)
    got = G.pdist(X, a, b, cap, threads=3)
    exp = np.array([min(D[i, j], c) for i, j, c in zip(a, b, cap)])
    bad += int((got != exp).sum())
    srcs = np.arange(k)
    caps = np.full(k, 10 ** 9)
    ln, cyc = G.cyc_through(X, srcs, caps, threads=2)
    for v in range(k):
        into = [e for e in range(s.size) if t[e] == v]
        r = min([int(X[e]) + D[v, s[e]] for e in into], default=2 ** 60)
        r = min(r, 10 ** 9)
        if ln[v] != r:
            bad += 1
        if r < 10 ** 9:
            c = cyc[v]
            closed = all(t[c[i]] == s[c[(i + 1) % len(c)]] for i in range(len(c))) and s[c[0]] == v
            bad += (not closed) or int(X[c].sum()) != r
    nk = 3
    ln2, cyc2 = G.cyc_multi(X, srcs, caps, nk=nk, threads=2)
    for v in range(k):
        into = [e for e in range(s.size) if t[e] == v]
        exp = sorted(int(X[e]) + D[v, s[e]] for e in into if int(X[e]) + D[v, s[e]] < 10 ** 9)[:nk]
        got = [int(ln2[v * nk + j]) for j in range(nk) if cyc2[v * nk + j].size]
        if got != exp:
            bad += 1
        for j in range(len(got)):
            c = cyc2[v * nk + j]
            closed = all(t[c[i]] == s[c[(i + 1) % len(c)]] for i in range(len(c))) and s[c[0]] == v
            bad += (not closed) or int(X[c].sum()) != got[j]
    F = np.flatnonzero(rng.random(s.size) < 0.5)
    if F.size:
        Sx = 7
        got = G.raise_heads(X, F, Sx, threads=2)
        exp = X.copy()
        for b in F:
            exp[b] = max(int(X[b]), Sx - min(int(D[t[b], s[b]]), Sx))
        bad += int((got != exp).sum())
print("kernels vs Floyd:", "PASS" if bad == 0 else "FAIL", "mismatches", bad, "s", round(time.time() - t0, 1))
fails += bad > 0

# 2. tiny graphs, LP over all simple cycles
t0 = time.time()
bad = 0
for it in range(60):
    k = int(rng.integers(3, 7))
    s, t = rand_graph(k, 0.5)
    if s.size == 0:
        continue
    w = rng.integers(1, 10, s.size)
    eid = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(s, t))}
    cyc = [[eid[(c[i], c[(i + 1) % len(c)])] for i in range(len(c))] for c in nx.simple_cycles(nx.DiGraph(list(eid)))]
    G = core.Graph(s, t, k)
    if not cyc:
        bad += G.min_cycle(np.zeros(s.size, np.int64))[0] < np.iinfo(np.int64).max // 4
        continue
    M = np.zeros((len(cyc), s.size))
    for i, c in enumerate(cyc):
        M[i, c] = 1
    res = linprog(w, A_ub=-M, b_ub=-np.ones(len(cyc)), bounds=(0, None), method="highs")
    X = chk.to_int_lengths(res.x, CONFIG["S"])
    out = chk.check(G, w, X, threads=2)
    ub = Fraction(out["UB"])
    Lb = min(int(X[c].sum()) for c in cyc)
    ok = out["L"] == Lb and ub >= Fraction(res.fun) - Fraction(1, 10 ** 9) and float(ub) <= res.fun * (1 + 1e-4) + 1e-9
    bad += not ok
print("tiny graphs vs all-cycle LP:", "PASS" if bad == 0 else "FAIL", "bad", bad, "s", round(time.time() - t0, 1))
fails += bad > 0


def calib_case(label, k, s, t, w, cyc_lp_ref):
    m = s.size
    val, cycles, _, _ = cal.cycle_lp(k, s, t, w)
    if not cycles:
        return True
    M = cal.incidence(cycles, m)
    res = linprog(w, A_ub=-M, b_ub=-np.ones(len(cycles)), bounds=(0, None), method="highs")
    X = chk.to_int_lengths(res.x, CONFIG["S"])
    G = core.Graph(s, t, k)
    out = chk.check(G, w, X, threads=2)
    ub = out["UB_float"]
    ok = ub >= cyc_lp_ref * (1 - 1e-9) and ub <= cyc_lp_ref * (1 + CONFIG["rel_tol"]) + 1e-9
    print(" ", label, "nodes", k, "m", m, "cycLP(json)", round(cyc_lp_ref, 6), "UB", round(ub, 6), "L", out["L"],
          "ok", ok, flush=True)
    return ok


# 3. calibration
t0 = time.time()
bad = 0
from src.mfas.io import load_dataset
gm = load_dataset("mouse")
mj = json.loads((ROOT / "ceiling/results/m_calib_mouse.json").read_text())
bad += not calib_case("mouse", gm.n_nodes, gm.src.astype(np.int64), gm.tgt.astype(np.int64),
                      gm.weight.astype(np.float64), mj["cycle_lp"])
gf = load_dataset("connectome")
pj = json.loads((ROOT / CONFIG["pieces_file"]).read_text())["instances"]
for inst in sorted(pj, key=lambda i: -i["m"])[:CONFIG["n_pieces"]]:
    nodes = np.array(inst["nodes"], np.int64)
    loc = -np.ones(gf.n_nodes, np.int64)
    loc[nodes] = np.arange(nodes.size)
    keep = (loc[gf.src] >= 0) & (loc[gf.tgt] >= 0)
    s, t, w = loc[gf.src[keep]], loc[gf.tgt[keep]], gf.weight[keep].astype(np.int64)
    bad += not calib_case(inst["kind"] + str(inst["size"]) + "_" + str(inst["seed"]), nodes.size, s, t, w,
                          inst["cycle_lp"])
print("calibration:", "PASS" if bad == 0 else "FAIL", "bad", bad, "s", round(time.time() - t0, 1))
fails += bad > 0
print("ALL", "PASS" if fails == 0 else "FAIL")
