"""H80 rung 2 - the ruin-and-recreate LNS appended to the CONNECTOME champion.

Rung 1 passed all five sealed kill conditions (``proto_H80_verdict.json``), which under
the pre-registration buys exactly one thing: a connectome measurement.

Why a from-champion prototype is admissible here (M12)
------------------------------------------------------
M12 says a composed prototype started from the champion's own converged order OVERSTATES
an INNER-stage change. This is not an inner-stage change: stages 1-4 are byte-identical
to the champion and the LNS is a pure TERMINAL APPEND. Cycle 14 measured exactly this
case and found a from-champion prototype PREDICTS the screen for a terminal append, the
two agreeing to +0.000062 pp (``experiments/log.md`` 2026-08-26). H66 used the same
instrument for the same reason.

What it tests
-------------
Three arms, all appended to the identical champion order, all at the same wall budget:

* ``sideways``  - accept iff not strictly worse. ZERO temperature. This is the arm that
  WON at the largest scale rung 1 tested (n=4000: +0.167087 pp, against +0.147493 for the
  best annealed cell).
* ``anneal_T4`` / ``anneal_T8`` - Metropolis at the two T0 multipliers that won on S2r and
  S1r respectively. This is H80 proper.

and one control:

* ``m12_control`` - the champion order plus the SAME short sift and no LNS at all. H66
  measured this at EXACTLY 0.000000 pp of headroom on this champion; reproducing that is
  what makes any gain attributable to the destroy-and-recreate rather than to the repair.

THE PREDICTION THIS RUN TESTS, stated before it ran: rung 1 found that the temperature's
worth SHRINKS WITH SCALE - +0.2108 pp at n=400, -0.0196 pp at n=4000 - while the deepest
excursion any schedule achieved fell from 4% of the measured barrier to 0.1% of it. If
that trend is real, then at n=136,648 the annealed arms should be <= the sideways arm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments"))

import torch  # noqa: E402

torch.set_num_threads(2)

import proto_H80_anneal as P  # noqa: E402
from mfas import io  # noqa: E402
from mfas.metrics import pct, score_from_order  # noqa: E402
from mfas.refine.insertion import build_sift_edges, sift  # noqa: E402

# The champion order this stage is appended to (sota.json, 2026-08-27), with its anchor.
CHAMPION = {
    "connectome": {
        "variant": "H64",
        "pct": 84.25817950936937,
        "positions": "results/20260827T024801Z-H64-connectome-s42-confirm-a7490c_positions.npy",
    },
}


def load_champion_rank(g, dataset):
    """Load the champion positions, convert to ranks, and ABORT unless they re-score."""
    spec = CHAMPION[dataset]
    pos = np.load(_ROOT / spec["positions"])
    rank = np.argsort(np.argsort(pos, kind="stable"), kind="stable").astype(np.int64)
    if sorted(rank.tolist()) != list(range(g.n_nodes)):
        raise SystemExit("champion rank vector is not a permutation")
    got = pct(score_from_order(rank, np.asarray(g.src, dtype=np.int64),
                               np.asarray(g.tgt, dtype=np.int64), g.weight),
              g.total_weight)
    if abs(got - spec["pct"]) > 1e-9:
        raise SystemExit(f"ANCHOR FAILED: champion re-scores to {got!r}, "
                         f"expected {spec['pct']!r}")
    return rank, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="connectome")
    ap.add_argument("--budget-s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    g = io.load_dataset(args.dataset)
    rank, anchor = load_champion_rank(g, args.dataset)
    print(f"ANCHOR OK: {args.dataset} champion re-scores to {anchor!r}", flush=True)
    src, tgt, w = build_sift_edges(g)
    n = g.n_nodes
    k = max(4, min(40, int(round(0.05 * n))))

    out = {
        "_doc": "H80 rung 2: the ruin-and-recreate LNS appended to the connectome "
                "champion. From-champion prototype, admissible for a TERMINAL APPEND "
                "under the cycle-14 measurement (predicts the screen to +0.000062 pp).",
        "dataset": args.dataset, "seed": args.seed, "budget_s": args.budget_s,
        "k": k, "champion": CHAMPION[args.dataset], "anchor_pct": anchor,
        "prediction_before_running": "the annealed arms should be <= the sideways arm, "
                                     "because the temperature's worth shrank from "
                                     "+0.2108 pp at n=400 to -0.0196 pp at n=4000",
        "arms": {},
    }

    # M12 control: the champion order plus the SAME short sift, no LNS.
    t0 = time.time()
    c_rank, c_score, c_log = sift(g, rank, max_sweeps=P.INNER_SWEEPS)
    out["arms"]["m12_control_sift_only"] = dict(
        best_pct=pct(c_score, g.total_weight),
        delta_pp=pct(c_score, g.total_weight) - anchor,
        wall_s=round(time.time() - t0, 1), n_sweeps=len(c_log))
    print("m12_control:", out["arms"]["m12_control_sift_only"], flush=True)

    for name, acc, t0m in (("sideways", "sideways", 0.0),
                           ("anneal_T4", "metropolis", 4.0),
                           ("anneal_T8", "metropolis", 8.0)):
        r = P.search(g, rank, destroy="topk", acceptance=acc, k=k,
                     time_budget_s=args.budget_s, seed=args.seed, t0_mult=t0m)
        r["delta_pp"] = r["best_pct"] - anchor
        out["arms"][name] = r
        print(f"{name}: best={r['best_pct']!r} delta={r['delta_pp']:+.6f} "
              f"rounds={r['rounds']} up={r['n_uphill_accepted']} "
              f"side={r['n_sideways_accepted']} exc={r['max_excursion_pp']:.6f} "
              f"dmed={r['delta_med_pp']:.6f} cal={r['cal_source']}", flush=True)

    best = max((v.get("delta_pp", -9), kk) for kk, v in out["arms"].items())
    out["best_arm"] = best[1]
    out["best_delta_pp"] = best[0]
    out["screen_bar_pp"] = 0.012
    out["clears_screen_bar"] = bool(best[0] > 0.012)
    p = _ROOT / "experiments" / "outputs" / f"proto_H80_rung2_{args.dataset}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote " + str(p), flush=True)
    print(json.dumps({kk: {"delta_pp": v.get("delta_pp"), "rounds": v.get("rounds")}
                      for kk, v in out["arms"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
