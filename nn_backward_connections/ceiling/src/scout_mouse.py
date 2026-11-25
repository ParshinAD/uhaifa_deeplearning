"""Scout 5: mouse meso graph stats and exact-MFAS feasibility with scipy milp (HiGHS).

Usage: python scout_mouse.py lazy | order
lazy  = binary y_e per edge, cycle constraints added lazily (shortest cycle through every remaining edge)
order = x_ij binary for i<j (i before j), 3-cycle transitivity constraints, on the nontrivial SCCs"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse.csgraph import connected_components, shortest_path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from layering.io_utils import load_graph_and_champion_order
from src.mfas.metrics import score_from_order, pct

CONFIG = dict(time_limit=280.0, max_iter=200)

g, order, rank, ff = load_graph_and_champion_order("mouse")
si, ti, w = g.src.astype(np.int64), g.tgt.astype(np.int64), g.weight.astype(np.float64)
n, m, W = g.n_nodes, g.n_edges, float(w.sum())
key = si * n + ti
d = {int(k): i for i, k in enumerate(key)}
rev = np.array([d.get(int(b) * n + int(a), -1) for a, b in zip(si, ti)])
has = rev >= 0
lb2 = float(np.minimum(w, np.where(has, w[np.maximum(rev, 0)], 0))[has].sum() / 2)
print("mouse n", n, "m", m, "W", round(W, 6), "self loops", int((si == ti).sum()), "dups", m - np.unique(key).size)
print("weight min/median/max", round(float(w.min()), 6), round(float(np.median(w)), 6), round(float(w.max()), 6))
print("reciprocal pairs", int(has.sum() // 2), "edge share", round(float(has.mean()), 4), "weight share", round(float(w[has].sum() / W), 4))
print("2-cycle LB", round(lb2, 6), "ceiling", round(100 * (W - lb2) / W, 4))
back = rank[si] >= rank[ti]
fb = float(w[back].sum())
print("champion ff%", round(ff, 6), "feedback", round(fb, 6), "backward edges", int(back.sum()))
A = sp.csr_matrix((np.ones(m), (si, ti)), shape=(n, n))
nc, lab = connected_components(A, directed=True, connection="strong")
sz = np.bincount(lab)
print("SCC count", nc, "sizes>1", sorted(sz[sz > 1].tolist(), reverse=True))
inside = lab[si] == lab[ti]
print("edges inside SCCs", int(inside.sum()), "weight between SCCs", round(float(w[~inside].sum()), 6))


def find_cycles(removed):
    keep = inside & ~removed
    idx = np.flatnonzero(keep)
    G = sp.csr_matrix((np.ones(idx.size), (si[idx], ti[idx])), shape=(n, n))
    _, pred = shortest_path(G, directed=True, unweighted=True, return_predecessors=True)
    eid = {(int(si[e]), int(ti[e])): int(e) for e in idx}
    cycles = set()
    for e in idx:
        u, v = int(si[e]), int(ti[e])
        if pred[v, u] < 0 and v != u:
            continue
        path = [u]
        x = u
        while x != v:
            x = int(pred[v, x])
            path.append(x)
        path = path[::-1]
        cyc = [e] + [eid[(path[i], path[i + 1])] for i in range(len(path) - 1)]
        cycles.add(tuple(sorted(cyc)))
    return cycles


def run_lazy():
    t0 = time.time()
    cyc = list(find_cycles(np.zeros(m, bool)))
    removed = np.zeros(m, bool)
    for it in range(CONFIG["max_iter"]):
        rows = np.repeat(np.arange(len(cyc)), [len(c) for c in cyc])
        cols = np.concatenate([np.array(c) for c in cyc])
        M = sp.csr_matrix((np.ones(cols.size), (rows, cols)), shape=(len(cyc), m))
        left = CONFIG["time_limit"] - (time.time() - t0)
        if left <= 0:
            print("time limit reached before iteration", it)
            return
        r = milp(w * inside, constraints=LinearConstraint(M, lb=1, ub=np.inf), integrality=np.ones(m),
                 bounds=Bounds(0, 1), options=dict(time_limit=left, mip_rel_gap=0.0))
        y = np.round(r.x).astype(bool) if r.x is not None else np.zeros(m, bool)
        lb = getattr(r, "mip_dual_bound", None)
        new = find_cycles(y)
        print("iter", it, "cycles", len(cyc), "status", r.status, "obj", round(float(r.fun), 6) if r.fun is not None else None,
              "dual bound", round(float(lb), 6) if lb is not None else None, "new cycles", len(new),
              "t", round(time.time() - t0, 1), flush=True)
        if r.status != 0:
            print("milp not optimal:", r.message)
            return
        if not new:
            opt = float(r.fun)
            print("OPTIMAL feedback", round(opt, 6), "OPT ff%", round(100 * (W - opt) / W, 6),
                  "champion gap pp", round(100 * (W - opt) / W - ff, 6), "t", round(time.time() - t0, 3))
            keep = ~y
            G = sp.csr_matrix((np.ones(int(keep.sum())), (si[keep], ti[keep])), shape=(n, n))
            import networkx as nx
            topo = list(nx.topological_sort(nx.from_scipy_sparse_array(G, create_using=nx.DiGraph)))
            rk = np.empty(n, np.int64)
            rk[np.array(topo)] = np.arange(n)
            print("oracle check: order from topo sort of G minus y scores ff%",
                  round(pct(score_from_order(rk, g.src, g.tgt, g.weight), g.total_weight), 6))
            return
        cyc = list(set(cyc) | new)


def run_order():
    t0 = time.time()
    nodes = np.flatnonzero(sz[lab] > 1)
    k = nodes.size
    loc = -np.ones(n, np.int64)
    loc[nodes] = np.arange(k)
    vid = -np.ones((k, k), np.int64)
    iu, ju = np.triu_indices(k, 1)
    vid[iu, ju] = np.arange(iu.size)
    nv = iu.size
    c = np.zeros(nv)
    const = 0.0
    for e in np.flatnonzero(inside):
        a, b = loc[si[e]], loc[ti[e]]
        if a < b:
            c[vid[a, b]] -= w[e]
            const += w[e]
        else:
            c[vid[b, a]] += w[e]
    i, j, l = [x.ravel() for x in np.meshgrid(np.arange(k), np.arange(k), np.arange(k), indexing="ij")]
    s = (i < j) & (j < l)
    i, j, l = i[s], j[s], l[s]
    nt = i.size
    rows = np.repeat(np.arange(nt), 3)
    cols = np.column_stack([vid[i, j], vid[j, l], vid[i, l]]).ravel()
    vals = np.tile([1.0, 1.0, -1.0], nt)
    M = sp.csr_matrix((vals, (rows, cols)), shape=(nt, nv))
    print("order formulation: nodes", k, "vars", nv, "transitivity rows", nt, "build s", round(time.time() - t0, 1), flush=True)
    r = milp(c, constraints=LinearConstraint(M, lb=0, ub=1), integrality=np.ones(nv), bounds=Bounds(0, 1),
             options=dict(time_limit=CONFIG["time_limit"], mip_rel_gap=0.0))
    lb = getattr(r, "mip_dual_bound", None)
    print("status", r.status, r.message[:80])
    if r.fun is not None:
        opt = float(r.fun) + const
        print("best feedback", round(opt, 6), "ff%", round(100 * (W - opt) / W, 6))
    if lb is not None:
        print("dual bound feedback", round(float(lb) + const, 6), "ceiling", round(100 * (W - float(lb) - const) / W, 6))
    print("t", round(time.time() - t0, 1))


if __name__ == "__main__":
    run_lazy() if sys.argv[1] == "lazy" else run_order()
