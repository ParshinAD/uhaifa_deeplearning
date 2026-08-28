"""H82 rung 0d - the DEEP-INTERIOR near-cut profile, i.e. H83's real pre-gate.

Rung 0c showed the D(p) <= 32 positions are piled at the ends (median end-distance
0.0006*n on champion pairs). H83 can only matter if small-D positions exist in the
DEEP INTERIOR of the giant block, where a forced cut would actually split the core.
Restrict to the middle 80% of the line and re-count. Seconds of CPU.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
from mfas.io import load_dataset
from proto_h82_px_census import ranks_from_positions, sym_diff_profile

t0 = time.time()
g = load_dataset("connectome"); n = g.n_nodes
prev = json.load(open("experiments/outputs/proto_H82_rung0.json"))
files = {e["variant"]: e["file"] for e in prev["pool"]}
lo, hi = int(0.10 * n), int(0.90 * n)
rows = []
for a, b in [("H64","H42"),("H64","H79B"),("H64","H79C"),("H64","H73"),("H64","H36"),
             ("H42","H52"),("H79B","H79C")]:
    ra, oa = ranks_from_positions(np.load("results/" + files[a]))
    rb, _  = ranks_from_positions(np.load("results/" + files[b]))
    D = sym_diff_profile(ra, oa, rb)
    Di = D[lo:hi]
    r = dict(a=a, b=b, interior_lo=lo, interior_hi=hi,
             min_D_interior=int(Di.min()), median_D_interior=float(np.median(Di)),
             p01_D_interior=float(np.percentile(Di, 1)))
    for thr in (2, 8, 32, 128, 512):
        r["n_D_le_%d_interior" % thr] = int(np.sum(Di <= thr))
    rows.append(r)
    print("%-5sx%-5s interior[%d,%d): minD=%d  D<=2:%d  D<=8:%d  D<=32:%d  D<=128:%d  "
          "D<=512:%d  medianD=%.0f" % (a, b, lo, hi, r["min_D_interior"],
          r["n_D_le_2_interior"], r["n_D_le_8_interior"], r["n_D_le_32_interior"],
          r["n_D_le_128_interior"], r["n_D_le_512_interior"], r["median_D_interior"]), flush=True)
out = dict(item="H82", rung="0d", generated="2026-08-28", n_nodes=int(n),
           purpose="H83 pre-gate restricted to the deep interior (middle 80% of the line)",
           H83_gate="H83 needs >= 50 positions with D(p) <= 32; rung 0c showed the shallow "
                    "count is dominated by end effects, so this interior count is the one "
                    "that decides it",
           interior_fraction=0.8, rows=rows, wall_clock_s=time.time() - t0)
json.dump(out, open("experiments/outputs/proto_H82_rung0d.json", "w"), indent=2)
print("WROTE experiments/outputs/proto_H82_rung0d.json  best interior D<=32 count = %d"
      % max(r["n_D_le_32_interior"] for r in rows), flush=True)
