"""Calibration on the mouse meso graph (148 nodes): all bounds and an independent exact optimum.

Run from the repo root: python -u ceiling/src/m_calib_mouse.py > ceiling/logs/m_calib_mouse.log"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ceiling" / "src"))
from layering.io_utils import load_graph_and_champion_order
from src.mfas.metrics import score_from_order, pct
import m_calib_core as core

CONFIG = dict(dataset="mouse", milp_time=600.0, scout_opt_ff_pct=93.202194,
              out=ROOT / "ceiling" / "results" / "m_calib_mouse.json")

t00 = time.time()
g, order, rank, ff_champ = load_graph_and_champion_order(CONFIG["dataset"])
k, s, t, w = g.n_nodes, g.src.astype(np.int64), g.tgt.astype(np.int64), g.weight.astype(np.float64)
W = float(w.sum())
print("mouse n", k, "m", s.size, "W", round(W, 6))
feedback = core.feedback_of_rank(rank, s, t, w)
res = dict(dataset="mouse", n=k, m=int(s.size), W=W, champion_ff_pct=ff_champ, champion_feedback=feedback)
print("champion H63 ff%", round(ff_champ, 6), "feedback", round(feedback, 6))

tm = {}
t0 = time.time(); res["lb_2cycle"] = core.lb_2cycle(k, s, t, w); tm["lb_2cycle"] = time.time() - t0
t0 = time.time(); res["greedy_23"], res["n_triangles"] = core.greedy_23(k, s, t, w); tm["greedy_23"] = time.time() - t0
t0 = time.time()
res["cycle_lp"], cyc, res["cycle_lp_rounds"], res["cycle_lp_dual_sum"] = core.cycle_lp(k, s, t, w)
res["cycle_lp_ncycles"] = len(cyc)
tm["cycle_lp"] = time.time() - t0
t0 = time.time(); res["triangle_lp"], _, _ = core.ordering_bound(k, s, t, w, integral=False); tm["triangle_lp"] = time.time() - t0
t0 = time.time()
e1, y, ok1, res["lazy_milp_rounds"] = core.lazy_cycle_milp(k, s, t, w, cyc, CONFIG["milp_time"])
tm["exact_lazy_cycle_milp"] = time.time() - t0
t0 = time.time()
e2, ok2, d2 = core.ordering_bound(k, s, t, w, integral=True, time_limit=CONFIG["milp_time"])
tm["exact_ordering_milp"] = time.time() - t0
assert ok1 and ok2, (ok1, ok2)
rk = core.order_from_removed(k, s, t, y)
e_order = core.feedback_of_rank(rk, s, t, w)
ff_oracle = pct(score_from_order(rk, g.src, g.tgt, g.weight), g.total_weight)
res.update(exact_lazy_cycle_milp=e1, exact_ordering_milp=e2, exact_ordering_milp_info=d2,
           exact_order_feedback=e_order, exact_order_ff_pct_oracle=ff_oracle, opt=e1,
           opt_ff_pct=100 * (W - e1) / W, agree_scout=abs(100 * (W - e1) / W - CONFIG["scout_opt_ff_pct"]) < 5e-7,
           times_s=tm)
opt, lb2 = e1, res["lb_2cycle"]
for b in ("lb_2cycle", "greedy_23", "cycle_lp", "triangle_lp"):
    res[b + "_share_opt"] = res[b] / opt
    res[b + "_gap_closed"] = (res[b] - lb2) / (opt - lb2)
    res[b + "_ceiling_pct"] = 100 * (W - res[b]) / W
    print(b, round(res[b], 6), "share of opt", round(res[b] / opt, 6), "gap closed", round(res[b + "_gap_closed"], 4),
          "ceiling ff%", round(res[b + "_ceiling_pct"], 6), "t", round(tm[b], 2))
print("cycle LP rounds", res["cycle_lp_rounds"], "cycles", len(cyc), "dual sum", round(res["cycle_lp_dual_sum"], 6))
print("exact lazy-cycle MILP", round(e1, 6), "rounds", res["lazy_milp_rounds"], "t", round(tm["exact_lazy_cycle_milp"], 2))
print("exact ordering MILP", round(e2, 6), "lazy triangle info", d2, "t", round(tm["exact_ordering_milp"], 2))
print("order from MILP solution: feedback", round(e_order, 6), "oracle ff%", round(ff_oracle, 6))
print("OPT ff%", round(res["opt_ff_pct"], 6), "agrees with scout 93.202194:", res["agree_scout"])
res["runtime_s"] = time.time() - t00
CONFIG["out"].write_text(json.dumps(res, indent=1, default=float))
print("wrote", CONFIG["out"], "total s", round(res["runtime_s"], 1))
