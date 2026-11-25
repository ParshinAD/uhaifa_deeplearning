"""Lowering pass on a feasible integer length vector X (min cycle >= S): set each non-FAS edge e = (u, v) with X_e > 0
to max(0, S - dist(v, u)) (heads in `chunks` sequential groups, m_close_core mode 1), then re-raise the H64 backward
edges (chunked raise) and run the exact check. Heuristic; the bound rests only on the final exact check.
Usage (repo root): m_close_lower.py <X.npz> <name> <chunks_lower> <chunks_raise> [passes]"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import m_close_core as cc
import m_dual_check as chk
import m_dual_core as core

SCR = Path("/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/close")
CONFIG = dict(S=10 ** 6, threads=6)

if __name__ == "__main__":
    from layering.io_utils import load_graph_and_champion_order
    t00 = time.time()
    path, name, chl, chr_ = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    passes = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    g, order, rank, ff = load_graph_and_champion_order("connectome")
    w = g.weight.astype(np.int64)
    S = CONFIG["S"]
    inscc = core.scc_internal(g.src, g.tgt, g.n_nodes)
    back = rank[g.src] >= rank[g.tgt]
    G = core.Graph(g.src, g.tgt, g.n_nodes)
    X = np.load(path)["X"].astype(np.int64)
    rc = chk.check(G, w, X, threads=CONFIG["threads"])
    res = dict(command=" ".join([sys.executable] + sys.argv), git_commit=core.git_commit(), input=path, S=S,
               start_L=rc["L"], start_UB=rc["UB"], start_UB_float=round(rc["UB_float"], 2), passes=[])
    print(json.dumps(res), flush=True)
    best = (rc["UB_float"] if rc["L"] >= S else float("inf"), X)
    for p in range(passes):
        t0 = time.time()
        E = np.flatnonzero(~back & inscc & (X > 0))
        X1, _, _, _ = cc.raise_cycles(G, X, E, S, S, chunks=chl, threads=CONFIG["threads"], mode=1)
        low = float((w * (X - X1)).sum() / S)
        F = np.flatnonzero(back & inscc & (X1 < S))
        X2, _, _, _ = cc.raise_cycles(G, X1, F, S, S, chunks=chr_, threads=CONFIG["threads"])
        rais = float((w * (X2 - X1)).sum() / S)
        r = chk.check(G, w, X2, threads=CONFIG["threads"])
        row = dict(lowered_edges=int((X1 < X).sum()), lower_gain=round(low, 1), raise_cost=round(rais, 1), L=r["L"],
                   UB=r["UB"], UB_float=round(r["UB_float"], 2), UB_status="PASS L>=S" if r["L"] >= S else "FAIL L<S",
                   seconds=round(time.time() - t0, 1))
        res["passes"].append(row)
        print(json.dumps(row), flush=True)
        if r["L"] >= S and r["UB_float"] < best[0]:
            best = (r["UB_float"], X2)
            np.savez(SCR / ("X_lower_" + name + ".npz"), X=X2, UB=np.array(r["UB"]))
        X = X2
    res["runtime_s"] = round(time.time() - t00, 1)
    (ROOT / "ceiling" / "results" / ("m_close_lower_" + name + ".json")).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
