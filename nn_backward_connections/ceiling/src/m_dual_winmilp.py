"""Exact optimum of champion-order windows by lazy cycle MILP, to measure the integrality gap of the cycle LP
on windows far larger than the calibration pieces (opt(H) vs tau*(H) from m_dual_block).

Window = induced subgraph on the k nodes of base-order ranks [i k, (i + 1) k), SCC-internal edges only
(edges between SCCs of the window add 0 to both). MILP: binary y_e, rows = cycles; start from the rows of
the window cycle LP; separation: cycles of the window graph without the y = 1 edges (multi-cycle kernel,
hop lengths). Not a certificate: HiGHS floats; opt is checked by the feedback of a topological order.
Usage (repo root): m_dual_winmilp.py <k> <window ids comma-separated> <name> [time per window s]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import m_dual_core as core
import m_dual_block as blk

CONFIG = dict(threads=6, nk=10, max_rounds=200, time_per_window_s=600, lp_cap_s=300)


def rows_matrix(rows, m):
    ri = np.repeat(np.arange(len(rows)), [len(c) for c in rows])
    ci = np.concatenate([np.array(c, np.int64) for c in rows])
    return sp.csr_matrix((np.ones(ci.size), (ri, ci)), shape=(len(rows), m))


def separate(G, k, cut, nk):
    BIG = 10 ** 12
    X = np.where(cut, BIG, 1).astype(np.int64)
    ln, cyc = G.cyc_multi(X, np.arange(k), np.full(k, BIG), nk=nk, threads=CONFIG["threads"], maxlen=2000)
    return {tuple(sorted(c.tolist())) for c, l in zip(cyc, ln) if c.size and l < BIG}


if __name__ == "__main__":
    from layering.io_utils import load_graph_and_champion_order
    t00 = time.time()
    kwin, ids, name = int(sys.argv[1]), [int(a) for a in sys.argv[2].split(",")], sys.argv[3]
    tl = float(sys.argv[4]) if len(sys.argv) > 4 else CONFIG["time_per_window_s"]
    g, order, rank, ff = load_graph_and_champion_order("connectome")
    w = g.weight.astype(np.int64)
    rs, rt = rank[g.src].astype(np.int64), rank[g.tgt].astype(np.int64)
    out = []
    for i in ids:
        t0 = time.time()
        nodes = order[i * kwin:(i + 1) * kwin]
        eid = np.flatnonzero((rs // kwin == i) & (rt // kwin == i))
        Fpi = int(w[eid][rs[eid] >= rt[eid]].sum())
        loc = -np.ones(g.n_nodes, np.int64)
        loc[nodes] = np.arange(nodes.size)
        s, t, we = loc[g.src[eid]], loc[g.tgt[eid]], w[eid]
        inner = core.scc_internal(s, t, nodes.size)
        s, t, we = s[inner], t[inner], we[inner]
        k, m = nodes.size, s.size
        x, lp, _, nrows = blk.block_lp(k, s, t, we, lambda *a: None, CONFIG["lp_cap_s"])
        G = core.Graph(s, t, k)
        rows = sorted(blk.LAST_ROWS)
        rec = dict(window=i, nodes=int(k), m_scc=int(m), F_pi_window=Fpi, lp_rows_value=lp, lp_converged=nrows >= 0)
        opt, status = None, "time"
        for r in range(CONFIG["max_rounds"]):
            left = tl - (time.time() - t0)
            if left <= 0:
                break
            res = milp(we.astype(np.float64), constraints=LinearConstraint(rows_matrix(rows, m), lb=1, ub=np.inf),
                       integrality=np.ones(m), bounds=Bounds(0, 1),
                       options=dict(time_limit=left, mip_rel_gap=0.0))
            if res.status != 0 or res.x is None:
                rec["milp_message"] = str(res.message)
                db = getattr(res, "mip_dual_bound", None)
                rec["milp_dual_bound_rows"] = None if db is None else float(db)
                break
            y = np.round(res.x).astype(bool)
            new = separate(G, k, y, CONFIG["nk"]) - set(rows)
            print(" window", i, "round", r, "rows", len(rows), "milp", round(float(res.fun), 3), "new", len(new),
                  "s", round(time.time() - t0, 1), flush=True)
            if not new:
                perm = blk.round_order(k, s, t, y)
                pos = np.empty(k, np.int64)
                pos[perm] = np.arange(k)
                opt = int(we[pos[s] >= pos[t]].sum())
                assert opt == int(we[y].sum()), (opt, int(we[y].sum()))
                status = "optimal"
                break
            rows = sorted(set(rows) | new)
        rec.update(opt=opt, status=status, rounds=r + 1, seconds=round(time.time() - t0, 1),
                   opt_minus_lp_rows_value=None if opt is None else opt - lp,
                   champion_excess=None if opt is None else Fpi - opt)
        print("window", i, json.dumps(rec), flush=True)
        out.append(rec)
    res = dict(command=" ".join([sys.executable] + sys.argv), git_commit=core.git_commit(), k=kwin, windows=out,
               runtime_s=round(time.time() - t00, 1))
    (ROOT / "ceiling" / "results" / ("m_dual_winmilp_" + name + ".json")).write_text(json.dumps(res, indent=1))
