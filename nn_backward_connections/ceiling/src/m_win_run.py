"""H8: exact optimum of H64 windows and dense subgraphs vs their cycle LP, the H64 order and a local search.

Instance = induced subgraph of the connectome on a node set H:
  win:<k>:<part>:<rep>   k consecutive H64 ranks, start drawn uniformly (seeded) inside the rank range of <part>
  at:<k>:<start>         k consecutive H64 ranks from rank <start> (no randomness; aligned M_DUAL windows)
  recip:<k>:<rep>        dense reciprocal cluster: seed node drawn from the top giant-SCC nodes by reciprocal
                         weight, grown by max reciprocal weight to the set (as in m_calib_pieces.py)
Per instance:
  F_h64   feedback of H64 restricted to H (all edges of H)
  F_ls    H64 restricted to H, then node insertion local search (best slot per node, passes until no gain)
  F_lpround_ls  topological order of H minus the LP edges with x >= 0.5, then the same local search
  lp      cycle LP over SCC-internal edges (m_dual_block.block_lp: lazy rows, separation tol 1e-4)
  opt     lazy cycle MILP (HiGHS, mip_rel_gap 0) started from the LP rows, separation = up to nk shortest-hop
          cycles through every node of H minus the chosen edges; optimal when none is left (checked by the
          feedback of a topological order). On time limit: lower = max(lp, MILP dual bound over the current
          rows), upper = best of F_h64, F_ls and local search on the rounded MILP incumbents.
Not a certificate (HiGHS floats). Usage (repo root): m_win_run.py <name> <spec,spec,...> <time per instance s> [threads] [lp cap s]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import m_dual_core as core
import m_dual_block as blk

CONFIG = dict(base_seed=20261125, threads=1, nk=10, max_rounds=500, lp_cap_s=600, ls_max_passes=50,
              recip_top=2000,
              parts=dict(early=(9000, 21000), optic=(21000, 99000), central=(117000, 129000),
                         late=(129000, 136648)))
blk.CONFIG["threads"] = CONFIG["threads"]
KIND = dict(win=1, recip=2)
PART = dict(early=1, optic=2, central=3, late=4)


def feedback(pos, s, t, w):
    return int(w[pos[s] >= pos[t]].sum())


def insertion_ls(perm, s, t, w, max_passes):
    """Move single nodes to their best slot until a full pass gives no gain. perm[i] = node at slot i."""
    k = perm.size
    perm = perm.copy()
    out_ptr = np.zeros(k + 1, np.int64)
    o = np.argsort(s, kind="stable")
    np.cumsum(np.bincount(s, minlength=k), out=out_ptr[1:])
    in_ptr = np.zeros(k + 1, np.int64)
    i_ = np.argsort(t, kind="stable")
    np.cumsum(np.bincount(t, minlength=k), out=in_ptr[1:])
    passes = 0
    for passes in range(1, max_passes + 1):
        moved = 0
        pos = np.empty(k, np.int64)
        pos[perm] = np.arange(k)
        for v in perm.copy().tolist():
            oe = o[out_ptr[v]:out_ptr[v + 1]]
            ie = i_[in_ptr[v]:in_ptr[v + 1]]
            if oe.size == 0 and ie.size == 0:
                continue
            p = pos[v]
            iu = pos[t[oe]]
            iu = iu - (iu > p)
            ji = pos[s[ie]]
            ji = ji - (ji > p)
            a = np.zeros(k + 1)
            np.add.at(a, iu + 1, w[oe])
            cost_out = np.cumsum(a)[:k]
            b = np.bincount(ji, weights=w[ie], minlength=k)
            cost_in = w[ie].sum() - np.concatenate([[0.0], np.cumsum(b)[:k - 1]])
            c = cost_out + cost_in
            q = int(c.argmin())
            if c[q] < c[p] - 0.5:
                perm = np.insert(np.delete(perm, p), q, v)
                pos[perm] = np.arange(k)
                moved += 1
        if moved == 0:
            break
    return perm, passes


def rows_matrix(rows, m):
    ri = np.repeat(np.arange(len(rows)), [len(c) for c in rows])
    ci = np.concatenate([np.array(c, np.int64) for c in rows])
    return sp.csr_matrix((np.ones(ci.size), (ri, ci)), shape=(len(rows), m))


def separate(G, k, cut):
    BIG = 10 ** 12
    X = np.where(cut, BIG, 1).astype(np.int64)
    ln, cyc = G.cyc_multi(X, np.arange(k), np.full(k, BIG), nk=CONFIG["nk"], threads=CONFIG["threads"], maxlen=5000)
    return {tuple(sorted(c.tolist())) for c, l in zip(cyc, ln) if c.size and l < BIG}


def select(spec, g, order, R, Ssym, giant):
    f = spec.split(":")
    kind, k = f[0], int(f[1])
    if kind == "at":
        start = int(f[2])
        return order[start:start + k].copy(), dict(kind=kind, size=k, seed=None, window_start=start)
    if kind == "win":
        part, rep = f[2], int(f[3])
        seed = [CONFIG["base_seed"], KIND[kind], k, PART[part], rep]
        lo, hi = CONFIG["parts"][part]
        start = int(np.random.default_rng(seed).integers(lo, hi - k + 1))
        return order[start:start + k].copy(), dict(kind=kind, size=k, part=part, rep=rep, seed=seed, window_start=start)
    rep = int(f[2])
    seed = [CONFIG["base_seed"], KIND[kind], k, 0, rep]
    rng = np.random.default_rng(seed)
    rs = np.where(giant, np.asarray(R.sum(axis=1)).ravel(), -1)
    top = np.argsort(-rs, kind="stable")[:CONFIG["recip_top"]]
    v0 = int(rng.choice(top))
    H = [v0]
    inH = np.zeros(g.n_nodes, bool)
    inH[v0] = True
    conn = R[v0].toarray().ravel()
    conn2 = Ssym[v0].toarray().ravel()
    while len(H) < k:
        sc = np.where(inH, -1.0, conn)
        if sc.max() <= 0:
            sc = np.where(inH, -1.0, conn2)
        v = int(sc.argmax())
        H.append(v)
        inH[v] = True
        conn += R[v].toarray().ravel()
        conn2 += Ssym[v].toarray().ravel()
    return np.array(H), dict(kind=kind, size=k, rep=rep, seed=seed, seed_node=v0)


def run_instance(spec, g, order, rank, A, R, Ssym, giant, tl):
    t0 = time.time()
    H, rec = select(spec, g, order, R, Ssym, giant)
    rec["spec"] = spec
    H = H[np.argsort(rank[H], kind="stable")]
    k = H.size
    sub = A[H][:, H].tocoo()
    s, t, w = sub.row.astype(np.int64), sub.col.astype(np.int64), np.rint(sub.data).astype(np.int64)
    rec.update(m=int(s.size), W=int(w.sum()))
    h64 = np.arange(k)
    rec["F_h64"] = feedback(h64, s, t, w)
    tt = time.time()
    perm_ls, passes = insertion_ls(np.arange(k), s, t, w, CONFIG["ls_max_passes"])
    pos = np.empty(k, np.int64)
    pos[perm_ls] = np.arange(k)
    rec.update(F_ls=feedback(pos, s, t, w), ls_passes=passes, ls_s=round(time.time() - tt, 1))
    inner = core.scc_internal(s, t, k)
    si, ti, wi = s[inner], t[inner], w[inner]
    _, lab = connected_components(sp.csr_matrix((np.ones(s.size), (s, t)), shape=(k, k)), directed=True,
                                  connection="strong")
    rec["scc_max"] = int(np.bincount(lab).max())
    rec["m_scc"] = int(si.size)
    rec["F_h64_cross_scc"] = int(w[~inner][(h64[s] >= h64[t])[~inner]].sum())
    tt = time.time()
    x, lp, lp_rounds, nrows = blk.block_lp(k, si, ti, wi.astype(np.int64), lambda *a: None, CONFIG["lp_cap_s"])
    rows = sorted(blk.LAST_ROWS)
    rec.update(lp=float(lp), lp_converged=bool(nrows >= 0), lp_rounds=lp_rounds, lp_rows=len(rows),
               lp_s=round(time.time() - tt, 1))
    cut = np.zeros(s.size, bool)
    cut[np.flatnonzero(inner)[x >= 0.5]] = True
    perm_lp, _ = insertion_ls(blk.round_order(k, s, t, cut), s, t, w, CONFIG["ls_max_passes"])
    pos = np.empty(k, np.int64)
    pos[perm_lp] = np.arange(k)
    rec["F_lpround_ls"] = feedback(pos, s, t, w)
    G = core.Graph(si, ti, k)
    best_up = min(rec["F_h64"], rec["F_ls"], rec["F_lpround_ls"])
    best_round, best_perm = None, None
    lower = float(lp)
    status, opt, r = "time", None, 0
    tt = time.time()
    hist = []
    for r in range(CONFIG["max_rounds"]):
        left = tl - (time.time() - t0)
        if left <= 5:
            break
        res = milp(wi.astype(np.float64), constraints=LinearConstraint(rows_matrix(rows, si.size), lb=1, ub=np.inf),
                   integrality=np.ones(si.size), bounds=Bounds(0, 1),
                   options=dict(time_limit=left, mip_rel_gap=0.0))
        db = getattr(res, "mip_dual_bound", None)
        if db is not None and np.isfinite(db):
            lower = max(lower, float(db))
        if res.x is not None:
            y = np.round(res.x).astype(bool)
            cut = np.zeros(s.size, bool)
            cut[np.flatnonzero(inner)[y]] = True
            perm = blk.round_order(k, s, t, cut)
            pos = np.empty(k, np.int64)
            pos[perm] = np.arange(k)
            fr = feedback(pos, s, t, w)
            if best_round is None or fr < best_round:
                best_round, best_perm = fr, perm
        if res.status != 0 or res.x is None:
            rec["milp_message"] = str(res.message)
            hist.append(dict(round=r, rows=len(rows), status=int(res.status), dual_bound=db))
            break
        lower = max(lower, float(res.fun))
        new = separate(G, k, y) - set(rows)
        hist.append(dict(round=r, rows=len(rows), milp=float(res.fun), new=len(new), s=round(time.time() - t0, 1)))
        print(" ", spec, "round", r, "rows", len(rows), "milp", round(float(res.fun), 1), "new", len(new),
              "s", round(time.time() - t0, 1), flush=True)
        if not new:
            assert fr == int(round(float(wi[y].sum()))), (fr, float(wi[y].sum()))
            opt, status = fr, "optimal"
            break
        rows = sorted(set(rows) | new)
    if status != "optimal" and best_perm is not None:
        perm2, _ = insertion_ls(best_perm, s, t, w, CONFIG["ls_max_passes"])
        pos = np.empty(k, np.int64)
        pos[perm2] = np.arange(k)
        rec.update(F_milp_round=best_round, F_milp_round_ls=feedback(pos, s, t, w))
        best_up = min(best_up, rec["F_milp_round_ls"])
    rec.update(status=status, opt=opt, opt_lower=lower if opt is None else opt, opt_upper=best_up if opt is None else opt,
               milp_rounds=r + 1, milp_rows_final=len(rows), milp_s=round(time.time() - tt, 1), milp_hist=hist,
               seconds=round(time.time() - t0, 1), nodes=H.tolist())
    lo, up = rec["opt_lower"], rec["opt_upper"]
    print(spec, "k", k, "m", rec["m"], "m_scc", rec["m_scc"], "| lp", round(lp, 1), "conv", rec["lp_converged"],
          "| opt", opt, "bracket", round(lo, 1), up, "| h64", rec["F_h64"], "ls", rec["F_ls"], "lpls", rec["F_lpround_ls"],
          "| lp/up", round(lp / up, 5) if up else None, "h64/up", round(rec["F_h64"] / up, 5) if up else None,
          "| s lp", rec["lp_s"], "milp", rec["milp_s"], "total", rec["seconds"], flush=True)
    return rec


if __name__ == "__main__":
    from layering.io_utils import load_graph_and_champion_order
    t00 = time.time()
    name, specs, tl = sys.argv[1], sys.argv[2].split(","), float(sys.argv[3])
    if len(sys.argv) > 4:
        CONFIG["threads"] = blk.CONFIG["threads"] = int(sys.argv[4])
    if len(sys.argv) > 5:
        CONFIG["lp_cap_s"] = float(sys.argv[5])
    g, order, rank, ff = load_graph_and_champion_order("connectome")
    n = g.n_nodes
    A = sp.csr_matrix((g.weight.astype(np.float64), (g.src, g.tgt)), shape=(n, n))
    Ssym = (A + A.T).tocsr()
    R = A.minimum(A.T).tocsr()
    _, lab = connected_components(A, directed=True, connection="strong")
    giant = lab == np.bincount(lab).argmax()
    out = ROOT / "ceiling" / "results" / ("m_win_" + name + ".json")
    recs = []
    for spec in specs:
        recs.append(run_instance(spec, g, order, rank, A, R, Ssym, giant, tl))
        out.write_text(json.dumps(dict(command=" ".join([sys.executable] + sys.argv), git_commit=core.git_commit(),
                                       config={a: str(b) for a, b in CONFIG.items()}, time_per_instance_s=tl,
                                       runtime_s=round(time.time() - t00, 1), instances=recs), indent=1))
    print("done", len(recs), "instances, total s", round(time.time() - t00, 1), "->", out, flush=True)
