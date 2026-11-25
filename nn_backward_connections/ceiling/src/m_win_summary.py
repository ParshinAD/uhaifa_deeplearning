"""Summary table for H8 from ceiling/results/m_win_<name>.json (all runs except smoke).
Per group (kind, size): instances, solved to optimality, and sum-based ratios
  (opt - lp) / opt, (F_h64 - opt) / opt, (F_ls - opt) / opt, (F_lpround_ls - opt) / opt
over solved instances; unsolved instances are listed with their bracket. A spec run twice keeps its narrowest bracket.
Usage (repo root): m_win_summary.py [names comma-separated]"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
names = sys.argv[1].split(",") if len(sys.argv) > 1 else None
files = sorted((ROOT / "ceiling" / "results").glob("m_win_*.json"))
recs = []
for f in files:
    name = f.stem[len("m_win_"):]
    if name == "smoke" or (names and name not in names):
        continue
    for r in json.loads(f.read_text())["instances"]:
        r["run"] = name
        recs.append(r)

# a spec re-run in a hard_* run with more time: keep the record with the narrowest bracket
keep = {}
for r in recs:
    if r["spec"] not in keep or r["opt_upper"] - r["opt_lower"] < keep[r["spec"]]["opt_upper"] - keep[r["spec"]]["opt_lower"]:
        keep[r["spec"]] = r
recs = list(keep.values())


def pct(a, b):
    return round(100 * float(a) / float(b), 3) if b else None


groups = defaultdict(list)
for r in recs:
    groups[(r["kind"], r["size"])].append(r)
print("group | n | optimal | sum opt | (opt-lp)/opt % | max per-inst % | (h64-opt)/opt % | (ls-opt)/opt % | (lpls-opt)/opt % | max s")
for key in sorted(groups):
    rs = groups[key]
    ok = [r for r in rs if r["status"] == "optimal"]
    so = sum(r["opt"] for r in ok)
    gaps = [pct(r["opt"] - r["lp"], r["opt"]) for r in ok if r["opt"] > 0]
    print(key[0], key[1], "|", len(rs), "|", len(ok), "|", so, "|", pct(so - sum(r["lp"] for r in ok), so), "|",
          max(gaps) if gaps else None, "|", pct(sum(r["F_h64"] for r in ok) - so, so), "|",
          pct(sum(r["F_ls"] for r in ok) - so, so), "|", pct(sum(r["F_lpround_ls"] for r in ok) - so, so), "|",
          round(max(r["seconds"] for r in rs), 1))
print("unsolved instances (bracket lower = max(lp, MILP dual bound), upper = best order found):")
for r in recs:
    if r["status"] != "optimal":
        print(" ", r["run"], r["spec"], "m_scc", r["m_scc"], "lp", round(r["lp"], 1), "conv", r["lp_converged"],
              "lower", round(r["opt_lower"], 1), "upper", r["opt_upper"], "h64", r["F_h64"], "ls", r["F_ls"],
              "lpls", r["F_lpround_ls"], "(upper-lp)/upper %", pct(r["opt_upper"] - r["lp"], r["opt_upper"]),
              "(upper-lower)/upper %", pct(r["opt_upper"] - r["opt_lower"], r["opt_upper"]), "s", r["seconds"])

# Aligned windows (kind "at"): a contiguous rank block can be re-ordered inside without changing the direction of
# any edge that leaves it, so F_h64 - (best order found) is an exact global gain and disjoint blocks add up.
al = sorted([r for r in recs if r["kind"] == "at"], key=lambda r: r["window_start"])
fh, up, lo, lp = (sum(r[a] for r in al) for a in ("F_h64", "opt_upper", "opt_lower", "lp"))
print("aligned 3000 windows:", len(al), "of 46, optimal", sum(r["status"] == "optimal" for r in al),
      "| sum F_h64", fh, "sum best order", up, "sum lower", round(lo, 1), "sum lp", round(lp, 1),
      "| gain h64 - best order (exact, achievable)", fh - up, "| h64 - lower (max possible gain)", round(fh - lo, 1))
ok = [r for r in al if r["status"] == "optimal"]
so = sum(r["opt"] for r in ok)
print("  solved only:", len(ok), "sum F_h64", sum(r["F_h64"] for r in ok), "sum opt", so, "sum lp",
      round(sum(r["lp"] for r in ok), 1), "| (h64 - opt)/opt %", pct(sum(r["F_h64"] for r in ok) - so, so),
      "(opt - lp)/opt %", pct(so - sum(r["lp"] for r in ok), so))
print("missing window starts:", sorted(set(range(0, 136648, 3000)) - {r["window_start"] for r in al}))
