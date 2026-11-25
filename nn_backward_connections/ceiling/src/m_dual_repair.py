"""Repair of LP duals (edge lengths for a restricted cycle pool) into a fractional FAS, and estimates.

X0 = ceil(S * x / w) on edges inside SCCs (x = PDHG dual, length l_e = x_e / w_e), 0 elsewhere.
Repair rule (valid): process the edges of a FAS F (champion backward edges); set
X_b = max(X_b, S - dist_X(v, u)) for b = (u, v). Lengths only grow, so after all of F is processed every
cycle (it contains an F edge) has length >= S. Distances computed in parallel batches are stale (too
short), which only overpays. Edges of F with X_b >= S need no query.

Mode 'heads': one fully stale raise of all F edges, one Dijkstra per head node (m_dual_core.raise_heads),
then the exact check. Mode 'estimate': sample n_sample sources (shortest cycle through each) and n_sample F edges; reports
the violated share and the cost of the fully parallel raise extrapolated from the sample (an estimate,
not a bound). Mode 'full': runs the raise in batches and the exact check.
Usage (repo root): m_dual_repair.py <duals.npz> <key> <name> <mode> [batch]"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import m_dual_core as core
import m_dual_check as chk

CONFIG = dict(S=10 ** 6, threads=6, n_sample=2000, seed=0, batch=6000, time_limit_s=3300,
              out_dir="/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/"
                      "380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/mdual")

if __name__ == "__main__":
    from layering.io_utils import load_graph_and_champion_order
    t00 = time.time()
    path, key, name, mode = sys.argv[1:5]
    if len(sys.argv) > 5:
        CONFIG["batch"] = int(sys.argv[5])
    g, order, rank, ff = load_graph_and_champion_order("connectome")
    w = g.weight.astype(np.int64)
    S = CONFIG["S"]
    inscc = core.scc_internal(g.src, g.tgt, g.n_nodes)
    x = np.load(path)[key].astype(np.float64)
    X = chk.to_int_lengths(x / w, S)
    X[~inscc] = 0
    G = core.Graph(g.src, g.tgt, g.n_nodes)
    back = rank[g.src] >= rank[g.tgt]
    F = np.flatnonzero(back & inscc & (X < S))
    rng = np.random.default_rng(CONFIG["seed"])
    res = dict(command=" ".join([sys.executable] + sys.argv), git_commit=core.git_commit(), input=path, key=key,
               S=S, mode=mode, sum_wX0_over_S=float((w * X).sum() / S), zero_share_in_scc=float((X[inscc] == 0).mean()),
               F_queries=int(F.size), F_weight=int(w[F].sum()))
    print(json.dumps(res), flush=True)
    srcs = rng.choice(g.n_nodes, CONFIG["n_sample"], replace=False)
    t0 = time.time()
    ln, _ = G.cyc_through(X, srcs, np.full(srcs.size, S), threads=CONFIG["threads"])
    res["sample_sources_violated_share"] = float((ln < S).mean())
    res["sample_sources_zero_cycle_share"] = float((ln == 0).mean())
    res["sample_shortest_cycle_quantiles_over_S"] = [float(q) for q in np.quantile(ln / S, [0.1, 0.5, 0.9])]
    qs = rng.choice(F, min(CONFIG["n_sample"], F.size), replace=False)
    d = G.pdist(X, g.tgt[qs], g.src[qs], S - X[qs], threads=CONFIG["threads"])
    defi = np.maximum(0, S - X[qs] - d)
    res["sample_F_violated_share"] = float((defi > 0).mean())
    res["est_parallel_raise_cost"] = float((w[qs] * defi).sum() / S * F.size / qs.size)
    res["est_parallel_UB"] = res["sum_wX0_over_S"] + res["est_parallel_raise_cost"]
    res["sample_s"] = round(time.time() - t0, 1)
    res["est_full_query_cpu_s"] = float((time.time() - t0) * CONFIG["threads"] / (srcs.size + qs.size) * F.size)
    print(json.dumps(res), flush=True)
    if mode == "full":
        hist = []
        perm = F[np.argsort(w[F], kind="stable")]
        for p in range(0, perm.size, CONFIG["batch"]):
            if time.time() - t00 > CONFIG["time_limit_s"]:
                res["stopped_at"] = p
                break
            b = perm[p:p + CONFIG["batch"]]
            d = G.pdist(X, g.tgt[b], g.src[b], S - X[b], threads=CONFIG["threads"])
            X[b] = np.maximum(X[b], S - d)
            if (p // CONFIG["batch"]) % 10 == 0:
                hist.append([p + b.size, round(time.time() - t00, 1), float((w * X).sum() / S)])
                print("processed", p + b.size, "of", perm.size, "sum wX/S", round(hist[-1][2], 1), "s", hist[-1][1], flush=True)
        res["hist"] = hist
        if "stopped_at" not in res:
            np.savez(Path(CONFIG["out_dir"]) / ("repair_" + name + ".npz"), X=X)
            res.update(chk.check(G, w, X))
    if mode == "heads":
        t0 = time.time()
        X1 = G.raise_heads(X, F, S, threads=CONFIG["threads"])
        res["raise_s"] = round(time.time() - t0, 1)
        res["raise_cost"] = float((w * (X1 - X)).sum() / S)
        res["raised_edges"] = int((X1 > X).sum())
        print("raise done s", res["raise_s"], "cost", round(res["raise_cost"], 1), flush=True)
        np.savez(Path(CONFIG["out_dir"]) / ("repair_" + name + ".npz"), X=X1)
        res.update(chk.check(G, w, X1))
    res["runtime_s"] = round(time.time() - t00, 1)
    (ROOT / "ceiling" / "results" / ("m_dual_repair_" + name + ".json")).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
