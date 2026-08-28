"""H82 rung 0c - where the near-cut positions actually SIT, plus the microns scale control.

Rung 0b showed the gain is exactly zero because one giant block holds 92-100% of the nodes
and 98-100% of the edge mass, so 2^k collapses to the 2 parents.  H83 proposes to MANUFACTURE
cuts at positions where D(p) = |prefix_A(p) delta prefix_B(p)| is small.  Its pre-gate (">= 50
positions with D(p) <= 32") passes numerically - 13,603 on the best pair, 188 on champion pairs.

The question that pre-gate cannot answer, and that decides H83: are those near-cut positions
INSIDE the giant block, where a forced cut would actually split the indivisible core, or are
they piled up at the ends where D(p) is small for trivial reasons and a cut buys nothing?
This measures exactly that, and it is seconds of CPU.

Also repeats the rung-0 census on microns as the scale control the item asked for.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
from mfas.io import load_dataset
from mfas.metrics import score_from_order
from proto_h82_px_census import ranks_from_positions, common_blocks, sym_diff_profile
import proto_h82_px_census as R


def near_cut_location(g, rank_a, order_a, rank_b, n):
    """Where do the small-D(p) positions live relative to the giant common block?"""
    D = sym_diff_profile(rank_a, order_a, rank_b)
    ends = common_blocks(rank_a, order_a, rank_b)
    k = ends.shape[0]
    sizes = np.diff(np.concatenate(([-1], ends))).astype(np.int64)
    giant = int(np.argmax(sizes))
    g_end = int(ends[giant])
    g_start = int(g_end - sizes[giant] + 1)

    out = {}
    for thr in (2, 8, 32, 128):
        idx = np.nonzero(D <= thr)[0]
        inside = idx[(idx >= g_start) & (idx < g_end)]   # strictly interior to the giant block
        out["n_D_le_%d" % thr] = int(idx.size)
        out["n_D_le_%d_inside_giant" % thr] = int(inside.size)
        # how far from the nearer end of the WHOLE line, as a fraction of n
        if idx.size:
            frac = np.minimum(idx, n - 1 - idx) / float(n)
            out["median_end_distance_frac_D_le_%d" % thr] = float(np.median(frac))
    out.update(k=int(k), giant_start=g_start, giant_end=g_end,
               giant_size=int(sizes[giant]),
               median_D_inside_giant=float(np.median(D[g_start:g_end])) if g_end > g_start else 0.0,
               min_D_inside_giant=int(D[g_start:g_end].min()) if g_end > g_start else 0)
    return out


def main():
    t0 = time.time()
    g = load_dataset("connectome")
    R.INTEGRAL = True
    n = g.n_nodes
    prev = json.load(open("experiments/outputs/proto_H82_rung0.json"))
    files = {e["variant"]: e["file"] for e in prev["pool"]}

    def rk(v):
        return ranks_from_positions(np.load("results/" + files[v]))

    rows = []
    for a, b in [("H64", "H42"), ("H64", "H79B"), ("H64", "H73"), ("H64", "H36"),
                 ("H42", "H52"), ("H42", "H59"), ("H79B", "H79C")]:
        ra, oa = rk(a)
        rb, _ = rk(b)
        r = near_cut_location(g, ra, oa, rb, n)
        r.update(a=a, b=b)
        rows.append(r)
        print("[%6.1fs] %-5sx%-5s k=%-6d giant=[%d,%d] size=%d | D<=32: %d total, "
              "%d INSIDE giant | min D inside giant=%d | median end-dist(D<=32)=%.4f n"
              % (time.time() - t0, a, b, r["k"], r["giant_start"], r["giant_end"],
                 r["giant_size"], r["n_D_le_32"], r["n_D_le_32_inside_giant"],
                 r["min_D_inside_giant"],
                 r.get("median_end_distance_frac_D_le_32", -1)), flush=True)

    out = dict(item="H82", rung="0c", generated="2026-08-28",
               purpose="H83 pre-gate, correctly scoped: near-cut positions vs the giant block",
               n_nodes=int(n), near_cut_location=rows,
               wall_clock_s=time.time() - t0)
    with open("experiments/outputs/proto_H82_rung0c.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("[%6.1fs] WROTE experiments/outputs/proto_H82_rung0c.json" % (time.time() - t0), flush=True)

    # --- microns scale control: the rung-0 census, unchanged, on the third dataset
    print("\n[%6.1fs] === microns scale control ===" % (time.time() - t0), flush=True)
    sys.stdout.flush()
    subprocess.call([sys.executable, "experiments/proto_h82_px_census.py", "--dataset", "microns",
                     "--out", "experiments/outputs/proto_H82_rung0_microns.json",
                     "--verify-top", "10"])


if __name__ == "__main__":
    main()
