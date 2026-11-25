"""Block upper bound on tau*: cut the champion order into rank windows H_1 < H_2 < ... .
X = S on edges going backward between windows, X = S * (cycle-LP solution of H_i) inside window i.
Feasible: a cycle inside one window is covered by that window's LP; a cycle that leaves its window must
come back to a lower window through an inter-window backward edge (length S).
So tau* <= F_between + sum_i tau*(H_i), and the gain over the order is sum_i F_pi(H_i) - tau*(H_i).

Window LP: lazy cycle generation (HiGHS), separation = shortest cycle through every node (m_dual_core,
lengths round(x * S_sep)). Each window's X = ceil(S * x) is rescaled to have min cycle >= S exactly;
the assembled global X is checked by m_dual_check.

Usage (repo root): m_dual_block.py <k> <first window> <last window, exclusive> <name> [time_limit_s]
  [base_rank.npz] [offset]   (windows of the base order, shifted by offset ranks; default champion H64)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import m_dual_core as core
import heapq

CONFIG = dict(S=10 ** 6, S_sep=10 ** 9, sep_tol=10 ** -4, threads=6, max_rounds=400, nk=10, drop_slack=1e-6, window_time_cap_s=400, repair_batch=32, time_limit_s=3300,
              out_dir="/private/tmp/claude-501/-Users-abed359-IdeaProjects-university/"
                      "380c1a69-674f-4f4c-be10-ecf084bbd9db/scratchpad/mdual")


def block_lp(k, s, t, w, log, time_cap=None):
    """Cycle LP on a local graph; returns x (float), value, rounds, rows in the last LP.
    Rows that are slack after a solve are dropped; convergence is decided by exact separation only."""
    G = core.Graph(s, t, k)
    m = s.size
    nodes = np.arange(k)
    rows = []
    seen = set()
    seen_dropped, sticky = set(), set()
    X = np.ones(m, np.int64)
    x = np.zeros(m)
    t0 = time.time()
    for r in range(CONFIG["max_rounds"]):
        cap = np.full(k, 10 ** 15) if r == 0 else np.full(k, int(CONFIG["S_sep"] * (1 - CONFIG["sep_tol"])))
        ln, cyc = G.cyc_multi(X, nodes, cap, nk=CONFIG["nk"], threads=CONFIG["threads"], maxlen=500)
        new = {tuple(sorted(c.tolist())) for c, l in zip(cyc, ln) if c.size and l < cap[0]}
        new -= set(rows)
        if not new or (time_cap is not None and time.time() - t0 > time_cap and r > 0):
            global LAST_ROWS
            LAST_ROWS = rows
            return x, float(w @ x), r, len(rows) if not new else -len(rows)
        seen |= new
        rows = rows + sorted(new)
        ri = np.repeat(np.arange(len(rows)), [len(c) for c in rows])
        ci = np.concatenate([np.array(c, np.int64) for c in rows])
        M = sp.csr_matrix((np.ones(ci.size), (ri, ci)), shape=(len(rows), m))
        res = linprog(w.astype(np.float64), A_ub=-M, b_ub=-np.ones(len(rows)), bounds=(0, None), method="highs")
        assert res.status == 0, res.message
        x = res.x
        X = np.round(np.maximum(x, 0) * CONFIG["S_sep"]).astype(np.int64)
        tight = ((M @ x) < 1 + CONFIG["drop_slack"]) | np.array([c in sticky for c in rows])
        sticky |= {c for c, keep in zip(rows, tight) if not keep and c in seen_dropped}
        seen_dropped |= {c for c, keep in zip(rows, tight) if not keep}
        n_all = len(rows)
        rows = [c for c, keep in zip(rows, tight) if keep]
        log("  round", r, "rows", n_all, "kept", len(rows), "new", len(new), "lp", round(float(res.fun), 3),
            "s", round(time.time() - t0, 1))
    raise RuntimeError("block LP did not converge")


def repair_raise(G, s, t, X, F, S, batch, threads):
    """Raise X on the FAS edges F (b = (u, v)) to S - dist_X(v, u), in batches with stale distances.
    Lengths only grow, so afterwards every cycle (it has an F edge) has length >= S."""
    X = X.copy()
    F = F[X[F] < S]
    for p in range(0, F.size, batch):
        b = F[p:p + batch]
        d = G.pdist(X, t[b], s[b], S - X[b], threads=threads)
        X[b] = np.maximum(X[b], S - d)
    return X


def round_order(k, s, t, cut):
    """Kahn topological sort of (s, t) without the cut edges, ties and cycles broken by local index."""
    keep = ~cut
    indeg = np.bincount(t[keep], minlength=k)
    adj = [[] for _ in range(k)]
    for a, b in zip(s[keep].tolist(), t[keep].tolist()):
        adj[a].append(b)
    alive = np.ones(k, bool)
    ready = [v for v in range(k) if indeg[v] == 0]
    heapq.heapify(ready)
    rest = list(range(k))
    out = []
    p = 0
    while len(out) < k:
        if not ready:
            while not alive[rest[p]]:
                p += 1
            ready = [rest[p]]
        v = heapq.heappop(ready)
        if not alive[v]:
            continue
        alive[v] = False
        out.append(v)
        for u in adj[v]:
            indeg[u] -= 1
            if indeg[u] == 0 and alive[u]:
                heapq.heappush(ready, u)
    return np.array(out, np.int64)


if __name__ == "__main__":
    from layering.io_utils import load_graph_and_champion_order
    t00 = time.time()
    kwin, a, b, name = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    tl = float(sys.argv[5]) if len(sys.argv) > 5 else CONFIG["time_limit_s"]
    g, order, rank, ff = load_graph_and_champion_order("connectome")
    if len(sys.argv) > 6:
        rank = np.load(sys.argv[6])["rank"].astype(np.int64)
        order = np.argsort(rank, kind="stable")
        assert np.array_equal(np.sort(rank), np.arange(g.n_nodes))
    off = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    w = g.weight.astype(np.int64)
    S = CONFIG["S"]
    rs, rt = rank[g.src].astype(np.int64), rank[g.tgt].astype(np.int64)
    win_s, win_t = (rs + off) // kwin, (rt + off) // kwin

    def log(*x):
        print(*x, flush=True)

    out = []
    new_rank = rank.astype(np.int64).copy()
    Xg = np.zeros(g.n_edges, np.int64)
    done = []
    for i in range(a, b):
        if time.time() - t00 > tl:
            log("time limit, stop before window", i)
            break
        t0 = time.time()
        eid = np.flatnonzero((win_s == i) & (win_t == i))
        eall = eid
        Fpi = int(w[eid][rs[eid] >= rt[eid]].sum())
        nodes = order[max(0, i * kwin - off):max(0, (i + 1) * kwin - off)]
        loc = -np.ones(g.n_nodes, np.int64)
        loc[nodes] = np.arange(nodes.size)
        s, t, we = loc[g.src[eid]], loc[g.tgt[eid]], w[eid]
        inner = core.scc_internal(s, t, nodes.size)
        eid, s, t, we = eid[inner], s[inner], t[inner], we[inner]
        rec = dict(window=i, nodes=int(nodes.size), m_scc=int(eid.size), F_pi=Fpi)
        if eid.size == 0:
            rec.update(lp=0.0, UB_block=0, L=None, seconds=round(time.time() - t0, 1))
        else:
            x, val, rounds, nrows = block_lp(nodes.size, s, t, we, log, CONFIG["window_time_cap_s"])
            X = np.ceil(np.maximum(x, 0) * S).astype(np.int64)
            Gb = core.Graph(s, t, nodes.size)
            L = Gb.min_cycle(X, threads=CONFIG["threads"])[0]
            rec["converged"] = nrows >= 0
            if L < S * (1 - 10 * CONFIG["sep_tol"]):
                back = np.flatnonzero(rs[eid] >= rt[eid])
                back = back[np.argsort(we[back], kind="stable")]
                X0 = X
                X = repair_raise(Gb, s, t, X, back, S, CONFIG["repair_batch"], CONFIG["threads"])
                rec["repaired"] = dict(L_before=L, cost_before=float((we * X0).sum() / S), cost_after=float((we * X).sum() / S))
                L = Gb.min_cycle(X, threads=CONFIG["threads"])[0]
            if L > 0:
                X = -((-X * S) // L)
            if L == 0 or (we * X).sum() > Fpi * S:
                back = rs[eid] >= rt[eid]
                X = np.where(back, S, 0).astype(np.int64)
                rec["fallback_to_order"] = True
            L2 = Gb.min_cycle(X, threads=CONFIG["threads"])[0]
            assert L2 >= S
            Xg[eid] = X
            rec.update(lp=val, rounds=rounds, rows=nrows, L_before_rescale=L, L=L2,
                       UB_block=float((we * X).sum() / S), seconds=round(time.time() - t0, 1))
            sa, ta = loc[g.src[eall]], loc[g.tgt[eall]]
            cut = np.zeros(eall.size, bool)
            cut[np.searchsorted(eall, eid)] = X >= S // 2
            perm = round_order(nodes.size, sa, ta, cut)
            pos = np.empty(nodes.size, np.int64)
            pos[perm] = np.arange(nodes.size)
            Fr = int(w[eall][pos[sa] >= pos[ta]].sum())
            rec["F_rounded"] = Fr
            if Fr < Fpi:
                new_rank[nodes] = max(0, i * kwin - off) + pos
        rec["gain"] = rec["F_pi"] - rec["UB_block"]
        done.append(i)
        out.append(rec)
        log("window", i, json.dumps(rec))
    between = (win_s != win_t) & (rs >= rt)
    Xg[between] = S
    mask_done = np.isin(win_s, done) & (win_s == win_t)
    rest = ~mask_done & ~between & (rs >= rt)
    Xg[rest] = S
    tot_gain = sum(r["gain"] for r in out)
    res = dict(command=" ".join([sys.executable] + sys.argv), git_commit=core.git_commit(), k=kwin, offset=off,
               windows_done=done, S=S, F_base=int(w[rs >= rt].sum()), total_gain_float=tot_gain,
               assembled_sum_wX_over_S=float((w * Xg).sum() / S), windows=out,
               runtime_s=round(time.time() - t00, 1))
    np.savez(Path(CONFIG["out_dir"]) / ("block_" + name + ".npz"), X=Xg, rank=new_rank)
    from src.mfas.metrics import score_from_order, pct
    sc = score_from_order(new_rank, g.src, g.tgt, g.weight)
    res["rounded_order_feedback"] = float(g.total_weight - sc)
    res["rounded_order_ff_pct"] = pct(sc, g.total_weight)
    (ROOT / "ceiling" / "results" / ("m_dual_block_" + name + ".json")).write_text(json.dumps(res, indent=1))
    log("assembled sum wX / S", round(res["assembled_sum_wX_over_S"], 3), "gain", round(tot_gain, 3),
        "runtime", res["runtime_s"], "rounded order feedback", round(res["rounded_order_feedback"], 1))
