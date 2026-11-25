"""Calibration on induced subgraphs of the fly connectome with known optimum.

Selection kinds (seed per instance = base_seed + index, pinned):
  recip    dense reciprocal cluster: seed node from the top nodes by reciprocal weight, grow by max
           reciprocal weight to the set (total weight once reciprocal runs out)
  celltype a random cell type with few members in the giant SCC, plus partners by total weight to the set
  bfs      undirected BFS ball in the giant SCC from a random root, neighbors in random order
  window   consecutive nodes of the champion order H64
Usage (repo root): python -u ceiling/src/m_calib_pieces.py <kinds comma-separated> <tag> [sizes comma-separated] [reps]"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ceiling" / "src"))
from layering.io_utils import load_graph_and_champion_order
import m_calib_core as core

CONFIG = dict(sizes=[15, 18, 20, 30, 45, 60, 80], reps=2, base_seed=20261125, recip_top=2000,
              celltype_members=(3, 10), milp_time=120.0, tri_max_nodes=120,
              ann=ROOT / "data" / "external" / "flywire_neuron_annotations.tsv.gz",
              out_dir=ROOT / "ceiling" / "results")
KIND_OFFSET = dict(recip=0, celltype=1000, bfs=2000, window=3000)

kinds, tag = sys.argv[1].split(","), sys.argv[2]
if len(sys.argv) > 3:
    CONFIG["sizes"] = [int(a) for a in sys.argv[3].split(",")]
if len(sys.argv) > 4:
    CONFIG["reps"] = int(sys.argv[4])
t00 = time.time()
g, order, rank, ff = load_graph_and_champion_order("connectome")
n = g.n_nodes
A = sp.csr_matrix((g.weight.astype(np.float64), (g.src, g.tgt)), shape=(n, n))
S = (A + A.T).tocsr()
R = A.minimum(A.T).tocsr()
_, lab = connected_components(A, directed=True, connection="strong")
giant = lab == np.bincount(lab).argmax()
print("loaded connectome n", n, "giant SCC", int(giant.sum()), "t", round(time.time() - t00, 1), flush=True)

ann = pd.read_csv(CONFIG["ann"], sep="\t", usecols=["root_id", "cell_type"], low_memory=False)
ct_map = dict(zip(ann.root_id.astype(np.int64), ann.cell_type))
cell_type = np.array([ct_map.get(int(r), None) for r in g.node_ids], dtype=object)


def grow(start, k, M, fallback=None):
    H = list(start)[:k]
    inH = np.zeros(n, bool)
    inH[H] = True
    conn = np.asarray(M[H].sum(axis=0)).ravel()
    conn2 = np.asarray(fallback[H].sum(axis=0)).ravel() if fallback is not None else None
    while len(H) < k:
        sc = np.where(inH, -1.0, conn)
        if sc.max() <= 0 and conn2 is not None:
            sc = np.where(inH, -1.0, conn2)
        v = int(sc.argmax())
        H.append(v)
        inH[v] = True
        conn += M[v].toarray().ravel()
        if conn2 is not None:
            conn2 += fallback[v].toarray().ravel()
    return np.array(H)


def select(kind, k, rng):
    if kind == "recip":
        rs = np.where(giant, np.asarray(R.sum(axis=1)).ravel(), -1)
        top = np.argsort(-rs, kind="stable")[:CONFIG["recip_top"]]
        seed = int(rng.choice(top))
        return grow([seed], k, R, S), dict(seed_node=seed)
    if kind == "celltype":
        lo, hi = CONFIG["celltype_members"]
        ok = pd.Series(cell_type[giant & (cell_type != None)]).value_counts()
        types = sorted(ok[(ok >= lo) & (ok <= hi)].index.tolist())
        ct = types[int(rng.integers(len(types)))]
        mem = np.flatnonzero((cell_type == ct) & giant)
        return grow(mem.tolist(), k, S), dict(cell_type=str(ct), members=int(mem.size))
    if kind == "bfs":
        root = int(rng.choice(np.flatnonzero(giant)))
        seen, H, q = {root}, [root], deque([root])
        while q and len(H) < k:
            u = q.popleft()
            nb = S.indices[S.indptr[u]:S.indptr[u + 1]]
            for v in rng.permutation(nb[giant[nb]]):
                if int(v) not in seen and len(H) < k:
                    seen.add(int(v))
                    H.append(int(v))
                    q.append(int(v))
        return np.array(H), dict(root=root)
    start = int(rng.integers(0, n - k + 1))
    return order[start:start + k].copy(), dict(window_start=start)


def run_instance(kind, k, seed):
    rng = np.random.default_rng(seed)
    H, info = select(kind, k, rng)
    sub = A[H][:, H].tocoo()
    s, t, w = sub.row.astype(np.int64), sub.col.astype(np.int64), sub.data.astype(np.float64)
    rec = dict(kind=kind, size=k, seed=seed, nodes=H.tolist(), m=int(s.size), W=float(w.sum()), **info)
    blocks = core.scc_blocks(k, s, t)[0]
    rec["scc_sizes"] = sorted([int(b.size) for b in blocks], reverse=True)
    tm = {}
    t0 = time.time(); rec["lb_2cycle"] = core.lb_2cycle(k, s, t, w); tm["lb_2cycle"] = time.time() - t0
    t0 = time.time(); rec["greedy_23"], rec["n_triangles"] = core.greedy_23(k, s, t, w); tm["greedy_23"] = time.time() - t0
    t0 = time.time()
    rec["cycle_lp"], cyc, rec["cycle_lp_rounds"], rec["cycle_lp_dual_sum"] = core.cycle_lp(k, s, t, w)
    rec["cycle_lp_ncycles"] = len(cyc)
    tm["cycle_lp"] = time.time() - t0
    t0 = time.time()
    e1, y, ok1, rec["lazy_milp_rounds"] = core.lazy_cycle_milp(k, s, t, w, cyc, CONFIG["milp_time"])
    tm["exact_lazy_cycle_milp"] = time.time() - t0
    rec["exact_lazy_cycle_milp"] = e1
    if ok1:
        rec["exact_order_feedback"] = core.feedback_of_rank(core.order_from_removed(k, s, t, y), s, t, w)
    big = max(rec["scc_sizes"] or [0]) > CONFIG["tri_max_nodes"]
    e2 = None
    if not big:
        t0 = time.time(); rec["triangle_lp"], _, rec["triangle_lp_info"] = core.ordering_bound(k, s, t, w, False); tm["triangle_lp"] = time.time() - t0
        t0 = time.time()
        e2, ok2, _ = core.ordering_bound(k, s, t, w, True, CONFIG["milp_time"])
        tm["exact_ordering_milp"] = time.time() - t0
    elif ok1 and e1 - rec["cycle_lp"] < 1e-6:
        rec["triangle_lp"], rec["triangle_lp_implied"] = rec["cycle_lp"], True
    else:
        rec["triangle_lp"] = None
    rec["exact_ordering_milp"] = e2
    if max(rec["scc_sizes"] or [0]) <= core.CONFIG["dp_max_nodes"]:
        t0 = time.time(); rec["exact_dp"] = core.dp_exact(k, s, t, w); tm["exact_dp"] = time.time() - t0
    vals = [v for v in (e1, rec.get("exact_order_feedback"), e2, rec.get("exact_dp")) if v is not None]
    rec["exact_methods_agree"] = bool(vals) and max(vals) - min(vals) < 1e-6
    rec["opt"] = vals[0] if vals and rec["exact_methods_agree"] else None
    rec["times_s"] = tm
    opt, lb2 = rec["opt"], rec["lb_2cycle"]
    if opt is not None and opt > 0:
        for b in ("lb_2cycle", "greedy_23", "cycle_lp", "triangle_lp"):
            if rec[b] is None:
                continue
            rec[b + "_share_opt"] = rec[b] / opt
            rec[b + "_gap_closed"] = (rec[b] - lb2) / (opt - lb2) if opt - lb2 > 1e-9 else None
    print(kind, "k", k, "seed", seed, "m", rec["m"], "W", int(rec["W"]), "scc", rec["scc_sizes"][:3],
          "| lb2", round(lb2, 1), "g23", round(rec["greedy_23"], 1), "cycLP", round(rec["cycle_lp"], 3),
          "triLP", rec["triangle_lp"] if rec["triangle_lp"] is None else round(rec["triangle_lp"], 3), "opt", opt, "| exact", e1, e2, rec.get("exact_dp"),
          "| t", {a: round(b, 2) for a, b in tm.items()}, flush=True)
    return rec


out = CONFIG["out_dir"] / ("m_calib_pieces_" + tag + ".json")
recs = []
for kind in kinds:
    for i, k in enumerate(CONFIG["sizes"]):
        for r in range(CONFIG["reps"]):
            seed = CONFIG["base_seed"] + KIND_OFFSET[kind] + i * CONFIG["reps"] + r
            recs.append(run_instance(kind, k, seed))
            out.write_text(json.dumps(dict(config={a: str(b) for a, b in CONFIG.items()}, kinds=kinds,
                                           runtime_s=time.time() - t00, instances=recs), indent=1))
print("done", len(recs), "instances, total s", round(time.time() - t00, 1), "->", out)
