"""Repair of float duals x (edge-indexed, length x_e / w_e) into an exact upper bound on tau*, with chunked raise.

X0 = ceil(S x / w) on SCC edges; raise the H64 backward edges (m_close_core.raise_cycles, `chunks` sequential
head groups); exact integer check (m_dual_check.check). Writes ceiling/results/m_close_repair_<name>.json and
the lengths to scratchpad close/X_<name>.npz.
Usage (repo root): m_close_repair.py <duals.npz> <key> <name> <chunks> [x_key_is_X]"""
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
CONFIG = dict(S=10 ** 6, threads=6, thr=0)


def repair(G, w, inscc, back, x, chunks, S, threads, thr=0):
    """Float duals x -> (X0, X1 feasible after the check, cycles of the raise). x is edge-indexed (length x/w)."""
    X0 = chk.to_int_lengths(x / w, S)
    X0[~inscc] = 0
    F = np.flatnonzero(back & inscc & (X0 < S))
    X1, cp, ce, cb = cc.raise_cycles(G, X0, F, S, thr, chunks=chunks, threads=threads)
    return X0, X1, cp, ce, cb


if __name__ == "__main__":
    from layering.io_utils import load_graph_and_champion_order
    t00 = time.time()
    path, key, name, chunks = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    g, order, rank, ff = load_graph_and_champion_order("connectome")
    w = g.weight.astype(np.int64)
    S = CONFIG["S"]
    inscc = core.scc_internal(g.src, g.tgt, g.n_nodes)
    back = rank[g.src] >= rank[g.tgt]
    G = core.Graph(g.src, g.tgt, g.n_nodes)
    x = np.load(path)[key].astype(np.float64)
    t0 = time.time()
    X0, X1, cp, ce, cb = repair(G, w, inscc, back, x, chunks, S, CONFIG["threads"], CONFIG["thr"])
    res = dict(command=" ".join([sys.executable] + sys.argv), git_commit=core.git_commit(), input=path, key=key,
               S=S, chunks=chunks, sum_wX0_over_S=round(float((w * X0).sum() / S), 1),
               raise_cost=round(float((w * (X1 - X0)).sum() / S), 1), raised_edges=int((X1 > X0).sum()),
               raise_cycles=int(cb.size), raise_s=round(time.time() - t0, 1))
    print(json.dumps(res), flush=True)
    SCR.mkdir(parents=True, exist_ok=True)
    np.savez(SCR / ("X_" + name + ".npz"), X=X1)
    res.update(chk.check(G, w, X1, threads=CONFIG["threads"]))
    res["UB_status"] = "PASS L>=S" if res["L"] >= S else "FAIL L<S"
    res["runtime_s"] = round(time.time() - t00, 1)
    (ROOT / "ceiling" / "results" / ("m_close_repair_" + name + ".json")).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
