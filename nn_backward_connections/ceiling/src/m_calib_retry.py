"""Retry the exact optimum of pieces left unsolved in m_calib_pieces_big.json, with a longer MILP limit.
The piece is rebuilt from its stored node list; the lazy-cycle MILP starts from the cycles of the
final cycle LP that carry positive packing weight. If the MILP still times out, the optimum is
bracketed: lower = max(cycle LP, MILP dual bound over the current cycle rows; that MILP is a relaxation),
upper = feedback of the H64 champion order restricted to the piece.

Usage (repo root): python -u ceiling/src/m_calib_retry.py [source tag, default big]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ceiling" / "src"))
from layering.io_utils import load_graph_and_champion_order
import m_calib_core as core

tag = sys.argv[1] if len(sys.argv) > 1 else "big"
CONFIG = dict(src=ROOT / "ceiling" / "results" / ("m_calib_pieces_" + tag + ".json"),
              out=ROOT / "ceiling" / "results" / ("m_calib_retry.json" if tag == "big" else "m_calib_retry_" + tag + ".json"),
              milp_time=300.0, dual_tol=1e-9)

t00 = time.time()
g, order, rank, ff = load_graph_and_champion_order("connectome")
A = sp.csr_matrix((g.weight.astype(np.float64), (g.src, g.tgt)), shape=(g.n_nodes, g.n_nodes))
todo = [r for r in json.loads(CONFIG["src"].read_text())["instances"] if r["opt"] is None]
print("unsolved instances", len(todo), flush=True)
out = []
for r in todo:
    H = np.array(r["nodes"])
    k = H.size
    sub = A[H][:, H].tocoo()
    s, t, w = sub.row.astype(np.int64), sub.col.astype(np.int64), sub.data.astype(np.float64)
    m = s.size
    eid = core.eid_matrix(k, s, t)
    t0 = time.time()
    lp, cyc, rounds, dsum = core.cycle_lp(k, s, t, w)
    res = linprog(w, A_ub=-core.incidence(cyc, m), b_ub=-np.ones(len(cyc)), bounds=(0, None), method="highs")
    cycles = [c for c, y in zip(cyc, -res.ineqlin.marginals) if y > CONFIG["dual_tol"]]
    n_tight = len(cycles)
    t_lp = time.time() - t0
    t0 = time.time()
    opt, dual, mr, status = None, None, 0, None
    while True:
        left = CONFIG["milp_time"] - (time.time() - t0)
        if left <= 0:
            break
        mr += 1
        q = milp(w, constraints=LinearConstraint(core.incidence(cycles, m), lb=1, ub=np.inf), integrality=np.ones(m),
                 bounds=Bounds(0, 1), options=dict(time_limit=left, mip_rel_gap=0.0))
        status = int(q.status)
        db = getattr(q, "mip_dual_bound", None)
        if db is not None and np.isfinite(db):
            dual = float(db)
        if q.status != 0:
            break
        y = np.round(q.x).astype(bool)
        new = core.violated_cycles(k, s, t, np.zeros(m), eid, keep=~y, tol=0.5)
        if not new:
            opt = float(w[y].sum())
            fb = core.feedback_of_rank(core.order_from_removed(k, s, t, y), s, t, w)
            assert abs(fb - opt) < 1e-6
            break
        cycles = cycles + sorted(new - set(cycles))
    t_milp = time.time() - t0
    ub_h64 = core.feedback_of_rank(rank[H], s, t, w)
    lower = max(lp, dual if dual is not None else -np.inf)
    rec = dict(kind=r["kind"], size=r["size"], seed=r["seed"], m=r["m"], W=r["W"], lb_2cycle=r["lb_2cycle"],
               greedy_23=r["greedy_23"], cycle_lp=lp, n_cycles=len(cyc), n_tight_cycles=n_tight, milp_rounds=mr,
               milp_last_status=status, milp_dual_bound=dual, opt=opt, lower=opt if opt is not None else lower,
               upper=opt if opt is not None else ub_h64, h64_feedback_on_piece=ub_h64,
               times_s=dict(cycle_lp=t_lp, milp=t_milp))
    rec["cycle_lp_share_of_upper"] = lp / rec["upper"]
    out.append(rec)
    print(rec["kind"], rec["size"], rec["seed"], "lb2", r["lb_2cycle"], "g23", r["greedy_23"], "cycLP", round(lp, 3),
          "tight cycles", n_tight, "of", len(cyc), "| opt", opt, "bracket", round(rec["lower"], 3), round(rec["upper"], 3),
          "H64 on piece", ub_h64, "MILP rounds", mr, "status", status, "| t lp", round(t_lp, 1), "t milp", round(t_milp, 1), flush=True)
    CONFIG["out"].write_text(json.dumps(dict(config={a: str(b) for a, b in CONFIG.items()}, instances=out,
                                             runtime_s=time.time() - t00), indent=1))
print("done total s", round(time.time() - t00, 1))
