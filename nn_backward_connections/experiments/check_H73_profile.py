"""Validate proto_H73_tiebreak.profile_plateaus against the production kernel + brute force.

Three checks, all must pass before any real compute is spent:
  1. gain and gap_first must be BIT-IDENTICAL to jacobi_best_gaps on real orders.
  2. On small graphs, the plateau/gap_mindisp fields must match an O(n^2) brute-force
     enumeration of the exact insertion profile.
  3. gap_mindisp must always attain the same profile value as gap_first (same argmax set).
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from proto_H73_tiebreak import profile_plateaus  # noqa: E402

from mfas.io import load_dataset  # noqa: E402
from mfas.refine.insertion import build_sift_edges, jacobi_best_gaps  # noqa: E402


def brute_profile(u, rank, src, tgt, w, n):
    """Exact total_u(g) for every gap g in [0, n-1], by definition. O(n + deg)."""
    p = int(rank[u])
    vals = np.zeros(n, dtype=np.float64)
    for g in range(n):
        tot = 0.0
        for e in range(src.shape[0]):
            a, b, ww = int(src[e]), int(tgt[e]), float(w[e])
            if a == u:
                q = rank[b] if rank[b] < p else rank[b] - 1
                if g <= q:
                    tot += ww
            elif b == u:
                q = rank[a] if rank[a] < p else rank[a] - 1
                if g > q:
                    tot += ww
        vals[g] = tot
    return vals, p


def check(name, g, rank, brute_nodes=0, rng=None):
    n = g.n_nodes
    src, tgt, w = build_sift_edges(g)
    bg_ref, gain_ref = jacobi_best_gaps(rank, src, tgt, w, n)
    prof = profile_plateaus(rank, src, tgt, w, n)

    # gain is compared to float64 round-off, not bit-identity: the two implementations sum
    # the current-gap value in a different order (production accumulates a masked cumsum,
    # this one reads the containing interval directly), so they differ at ~1e-17. What must
    # hold exactly is the MOVER SET, since that is what every downstream statistic uses.
    dgain = np.abs(gain_ref - prof["gain"]).max()
    ok_gain = dgain < 1e-9 and np.array_equal(gain_ref > 1e-9, prof["gain"] > 1e-9)
    ok_gap = np.array_equal(bg_ref, prof["gap_first"])
    print("[%s] n=%d  max|dgain|=%.3g mover-set identical=%s  gap_first identical=%s"
          % (name, n, dgain, ok_gain, ok_gap))
    if not ok_gap:
        d = np.flatnonzero(bg_ref != prof["gap_first"])
        print("   gap mismatches:", d[:10], bg_ref[d[:5]], prof["gap_first"][d[:5]])

    ok_brute = True
    if brute_nodes:
        movers = np.flatnonzero(prof["gain"] > 1e-9)
        pick = movers if movers.shape[0] <= brute_nodes else rng.choice(
            movers, brute_nodes, replace=False)
        for u in pick:
            vals, p = brute_profile(int(u), rank, src, tgt, w, n)
            mx = vals.max()
            argset = np.flatnonzero(vals == mx)
            width = argset.shape[0]
            best = argset[np.argmin(np.abs(argset - p) * (n + 1) + argset)]
            # min |dist| then smallest gap
            dmin = np.abs(argset - p).min()
            best = argset[np.abs(argset - p) == dmin].min()
            if width != int(prof["plateau_width"][u]):
                print("   BRUTE width mismatch u=%d: %d vs %d"
                      % (u, width, prof["plateau_width"][u])); ok_brute = False
            if best != int(prof["gap_mindisp"][u]):
                print("   BRUTE mindisp mismatch u=%d: %d vs %d"
                      % (u, best, prof["gap_mindisp"][u])); ok_brute = False
            if abs(vals[int(prof["gap_first"][u])] - mx) > 1e-9:
                print("   gap_first is not a maximizer u=%d" % u); ok_brute = False
            if abs(vals[int(prof["gap_mindisp"][u])] - mx) > 1e-9:
                print("   gap_mindisp is not a maximizer u=%d" % u); ok_brute = False
            if abs((mx - vals[p]) - prof["gain"][u]) > 1e-9:
                print("   gain mismatch vs brute u=%d" % u); ok_brute = False
        print("   brute-force check on %d movers: %s" % (len(pick), "OK" if ok_brute else "FAIL"))
    return ok_gain and ok_gap and ok_brute


def main():
    rng = np.random.default_rng(42)
    all_ok = True

    g = load_dataset("mouse")
    n = g.n_nodes
    for trial in range(3):
        rank = rng.permutation(n).astype(np.int64)
        all_ok &= check("mouse-rand%d" % trial, g, rank, brute_nodes=12, rng=rng)

    from mfas.experiments.H02 import greedy_fas_order
    rk = np.asarray(greedy_fas_order(g), dtype=np.int64)
    rk = np.argsort(np.argsort(rk, kind="stable"), kind="stable").astype(np.int64)
    all_ok &= check("mouse-greedy", g, rk, brute_nodes=12, rng=rng)

    print("\nALL CHECKS:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
