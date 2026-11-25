"""Summary table of the calibration runs (reads ceiling/results/m_calib_*.json).

Usage (repo root): python ceiling/src/m_calib_summary.py > ceiling/logs/m_calib_summary.log"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = dict(files=["m_calib_pieces_wb.json", "m_calib_pieces_rc.json", "m_calib_pieces_big.json", "m_calib_pieces_small.json", "m_calib_pieces_bigbfs.json", "m_calib_pieces_bigct.json"], tol=1e-6,
              bounds=["lb_2cycle", "greedy_23", "cycle_lp", "triangle_lp"])

recs = []
for f in CONFIG["files"]:
    p = ROOT / "ceiling" / "results" / f
    if p.exists():
        recs += json.loads(p.read_text())["instances"]
print("instances", len(recs), "unsolved", sum(r["opt"] is None for r in recs),
      "exact methods disagree",
      sum(r["opt"] is None and r["exact_lazy_cycle_milp"] is not None for r in recs))
print("\nper instance: kind size seed m W largest_scc | lb2 g23 cycLP triLP opt | share of opt | gap closed (lb2->opt)")
for r in recs:
    o = r["opt"]
    sh = [round(r[b] / o, 4) if o and r[b] is not None else None for b in CONFIG["bounds"]]
    gc = [round(r[b + "_gap_closed"], 4) if r.get(b + "_gap_closed") is not None else None for b in CONFIG["bounds"][1:]]
    print(r["kind"], r["size"], r["seed"], r["m"], int(r["W"]), (r["scc_sizes"] or [0])[0], "|",
          round(r["lb_2cycle"], 1), round(r["greedy_23"], 1), round(r["cycle_lp"], 3), r["triangle_lp"] if r["triangle_lp"] is None else round(r["triangle_lp"], 3),
          "(implied)" if r.get("triangle_lp_implied") else "", o,
          "|", sh, "|", gc)

print("\naggregates over instances with opt > lb2 (gap defined): kind, size band, count, mean/min share of opt, mean gap closed")
bands = [("<=20", 0, 20), ("30-80", 21, 80), (">=120", 81, 10 ** 9)]
for kind in ["recip", "celltype", "bfs", "window", "all"]:
    for name, lo, hi in bands:
        sel = [r for r in recs if (kind == "all" or r["kind"] == kind) and lo <= r["size"] <= hi and r["opt"]]
        gap = [r for r in sel if r["opt"] - r["lb_2cycle"] > CONFIG["tol"]]
        if not sel:
            continue
        row = [kind, name, "n", len(sel), "with gap", len(gap)]
        for b in CONFIG["bounds"]:
            s = np.array([r[b] / r["opt"] for r in sel if r[b] is not None])
            row += [b, "share mean", round(float(s.mean()), 4), "min", round(float(s.min()), 4)]
            if b != "lb_2cycle" and gap:
                g = np.array([r[b + "_gap_closed"] for r in gap if r.get(b + "_gap_closed") is not None])
                row += ["closed mean", round(float(g.mean()), 4), "min", round(float(g.min()), 4)]
        print(*row)

print("\ninstances where cycle LP < opt (integrality gap):")
for r in recs:
    if r["opt"] and r["opt"] - r["cycle_lp"] > CONFIG["tol"]:
        print(r["kind"], r["size"], r["seed"], "opt", r["opt"], "cycLP", round(r["cycle_lp"], 4), "triLP",
              r["triangle_lp"], "gap", round(r["opt"] - r["cycle_lp"], 4),
              "rel", round((r["opt"] - r["cycle_lp"]) / r["opt"], 6),
              "triLP closes", None if r["triangle_lp"] is None else round((r["triangle_lp"] - r["cycle_lp"]) / (r["opt"] - r["cycle_lp"]), 4))
print("instances where triangle LP > cycle LP:",
      sum(r["triangle_lp"] is not None and r["triangle_lp"] - r["cycle_lp"] > CONFIG["tol"] for r in recs),
      "of", sum(r["triangle_lp"] is not None and not r.get("triangle_lp_implied") for r in recs), "computed")

small = [r for r in recs if r["size"] <= 20 and r["opt"] is not None]
d = np.array([r["opt"] - r["cycle_lp"] for r in small])
W = np.array([r["W"] for r in small])
print("\nH4 pieces <= 20 nodes:", len(small), "exact - cycleLP: max", round(float(d.max()), 4), "sum", round(float(d.sum()), 4),
      "share of piece W", round(float(d.sum() / W.sum()), 6), "pieces with positive excess", int((d > CONFIG["tol"]).sum()))
mouse = json.loads((ROOT / "ceiling" / "results" / "m_calib_mouse.json").read_text())
print("\nmouse: lb2", round(mouse["lb_2cycle"], 6), "g23", round(mouse["greedy_23"], 6), "cycLP", round(mouse["cycle_lp"], 6),
      "triLP", round(mouse["triangle_lp"], 6), "opt", round(mouse["opt"], 6), "opt - cycLP", mouse["opt"] - mouse["cycle_lp"],
      "cycLP gap closed", round(mouse["cycle_lp_gap_closed"], 6))

for retry in [ROOT / "ceiling" / "results" / f for f in ("m_calib_retry.json", "m_calib_retry_bigct.json")]:
    if not retry.exists():
        continue
    print("\n" + retry.name + ": retry of unsolved big pieces (longer MILP; bracket lower = max(cycle LP, MILP dual bound), upper = H64 on piece):")
    for r in json.loads(retry.read_text())["instances"]:
        print(r["kind"], r["size"], r["seed"], "m", r["m"], "lb2", r["lb_2cycle"], "g23", r["greedy_23"], "cycLP", round(r["cycle_lp"], 3),
              "opt", r["opt"], "bracket", round(r["lower"], 3), round(r["upper"], 3), "cycLP / upper", round(r["cycle_lp_share_of_upper"], 6),
              "t", {a: round(b, 1) for a, b in r["times_s"].items()})
