"""H22 opportunity-sizing — is the Rocket↔best gap reachable by a *bounded-window*
discrete local search (sifting / re-insertion)?

DIAGNOSTIC ONLY. This is the go/no-go gate for the H22 hypothesis (DIRECTION D —
discrete refinement), exactly as H09 sized the tie-recovery pool before being built.
It does NOT run any variant and writes NOTHING to ``results/``; its output goes to
``experiments/outputs/localsearch_sizing.json``. It reads the privileged near-optimal
ordering ONLY through :mod:`mfas.analysis.gap` (the sole sanctioned reader of
``data/best_solution``), the same privilege boundary the Stage-A diagnosis uses. No
optimization path consumes any number produced here.

Background (finding #3). The ~1.69 pp gap is an OPTIMIZATION-GAP irreducible to the
*continuous* class of methods: the smooth sigmoid gradient is a coarse per-node
"majority vote" (≈ weighted in−out imbalance) blind to the distributed reorderings
inside cyclic cores where the gap lives. The corollary is to attack it with a *discrete*
move — but only if the gap's flips are SHORT-range (a bounded-window sift can reach them).
This script measures exactly that.

Two measures, both on H02's converged order (connectome 82.93% / mouse 92.48%):

  (1) window-reachable feedback pool — LEAKAGE-SAFE (ranks + input weights only):
      total feedback weight on edges whose endpoints lie within W ranks. The *ceiling*
      a single-node width-W move could ever flip. Computed on both datasets.

  (2) gap-flip rank-distance — PRIVILEGED (uses best_solution), connectome only: of the
      edges that change orientation between Rocket(H02) and the near-optimal order, their
      rank distance IN ROCKET'S ORDER, split into the recoverable (feedback→feedforward,
      "gain") and broken ("lose") directions. DECISIVE: if most of the net +1.69 pp is
      reachable at small/medium W → short-range → build H22; if it only accumulates at
      large W → long-range/global → bounded local search cannot reach it → self-falsifies
      H22 and *strengthens* finding #3.

Built-in cross-check: gain/lose/net must reproduce the Stage-A diagnosis (≈ +4.60 / −2.91
/ +1.69 pp) and the H02 order must re-score to 82.93% via the frozen oracle. If these fail
the script has a bug — do not draw conclusions.

Usage:
    /opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python experiments/size_localsearch.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mfas import io  # noqa: E402
from mfas.analysis import gap  # noqa: E402
from mfas.metrics import pct, score_from_order, score_from_positions  # noqa: E402

OUT = _ROOT / "experiments" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Rank-window grid the sizing is reported over (a single-node move of width W can only
# flip an edge whose endpoints are within W ranks of each other).
WINDOWS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000]


def _latest(pattern: str) -> str:
    fs = sorted(glob.glob(str(_ROOT / "results" / pattern)))
    if not fs:
        raise FileNotFoundError(f"no results file matches {pattern}")
    return fs[-1]


def _rank_of_node(positions: np.ndarray) -> np.ndarray:
    """rank_of_node[i] = rank of node i (0 = front/source side), from positions.

    Stable argsort matches ``positions_to_order``; with 0 exact position ties (verified
    for H02 in the Stage-A diagnosis) this rank order is faithful to the strict-`>` oracle
    on the raw positions, which the caller asserts via a score cross-check.
    """
    order = np.argsort(np.asarray(positions), kind="stable")   # order[r] = node at rank r
    n = order.shape[0]
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    return rank


def _reachable_pool(rank: np.ndarray, src: np.ndarray, tgt: np.ndarray,
                    w: np.ndarray, total: float) -> dict:
    """Measure 1: feedback weight (pp of total) reachable within each rank window.

    LEAKAGE-SAFE: uses only the current ranks and the input edge weights.
    """
    d = rank[tgt].astype(np.int64) - rank[src].astype(np.int64)   # >0 feedforward
    feedback = d <= 0                                             # strict-`>` oracle: tie = feedback
    absd = np.abs(d)
    fb_w = float(w[feedback].sum())
    per_window = {}
    for W in WINDOWS:
        pool = float(w[feedback & (absd <= W)].sum())
        per_window[str(W)] = dict(pool_pp=100.0 * pool / total,
                                  pool_frac_of_feedback=(pool / fb_w if fb_w else 0.0))
    return dict(feedback_total_pp=100.0 * fb_w / total,
                feedback_total_weight=fb_w,
                reachable_pool=per_window)


def _gap_flip_distance(rock_rank: np.ndarray, best_rank: np.ndarray,
                       src: np.ndarray, tgt: np.ndarray,
                       w: np.ndarray, total: float) -> dict:
    """Measure 2 (PRIVILEGED): rank-distance of the Rocket↔best orientation flips.

    ``gain`` = feedback in Rocket but feedforward in best (the recoverable +).
    ``lose`` = feedforward in Rocket but feedback in best (broken by the reorder).
    Distances are measured IN ROCKET'S ORDER (how far a node must travel to flip the edge).
    """
    rock_d = rock_rank[tgt].astype(np.int64) - rock_rank[src].astype(np.int64)
    best_d = best_rank[tgt].astype(np.int64) - best_rank[src].astype(np.int64)
    rock_ff = rock_d > 0
    best_ff = best_d > 0
    gain = (~rock_ff) & best_ff
    lose = rock_ff & (~best_ff)
    absd = np.abs(rock_d)

    gain_w = float(w[gain].sum())
    lose_w = float(w[lose].sum())

    cumulative = {}
    for W in WINDOWS:
        g_in = float(w[gain & (absd <= W)].sum())
        l_in = float(w[lose & (absd <= W)].sum())
        cumulative[str(W)] = dict(
            gain_pp=100.0 * g_in / total,
            lose_pp=100.0 * l_in / total,
            net_pp=100.0 * (g_in - l_in) / total,
            gain_frac_of_total_gain=(g_in / gain_w if gain_w else 0.0),
        )

    # distance distribution of the recoverable (gain) weight
    gain_dist = absd[gain]
    gain_wt = w[gain]
    order = np.argsort(gain_dist)
    gd, gw = gain_dist[order], gain_wt[order]
    cw = np.cumsum(gw) / max(gw.sum(), 1e-12)
    pctiles = {}
    for q in (0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        idx = int(np.searchsorted(cw, q))
        idx = min(idx, gd.shape[0] - 1)
        pctiles[f"p{int(q * 100)}"] = int(gd[idx])

    return dict(
        gain_total_pp=100.0 * gain_w / total,
        lose_total_pp=100.0 * lose_w / total,
        net_total_pp=100.0 * (gain_w - lose_w) / total,
        n_gain_edges=int(gain.sum()),
        n_lose_edges=int(lose.sum()),
        gain_distance_percentiles=pctiles,   # weighted rank-distance percentiles of gain edges
        cumulative_within_window=cumulative,
    )


def size_dataset(ds: str, with_best: bool) -> dict:
    g = io.load_dataset(ds)
    src = np.asarray(g.src)
    tgt = np.asarray(g.tgt)
    w = np.asarray(g.weight, dtype=np.float64)
    total = float(g.total_weight)

    pos = np.load(_latest(f"*H02-{ds}-s42-*_positions.npy")).astype(np.float64)
    rock_rank = _rank_of_node(pos)

    # cross-check: rank order must reproduce the position score (0 ties => identical)
    pos_pct = pct(score_from_positions(pos, src, tgt, g.weight), total)
    rank_pct = pct(score_from_order(rock_rank, src, tgt, g.weight), total)
    out = dict(
        dataset=ds, n_nodes=int(g.n_nodes), n_edges=int(g.n_edges), total_weight=total,
        h02_pos_pct=pos_pct, h02_rank_pct=rank_pct,
        rank_faithful=bool(abs(pos_pct - rank_pct) < 1e-9),
        measure1_reachable_pool=_reachable_pool(rock_rank, src, tgt, w, total),
    )

    if with_best:
        best_rank, best_pct = gap.load_best_solution(g)
        out["best_pct"] = best_pct
        out["measure2_gap_flip_distance"] = _gap_flip_distance(
            rock_rank, best_rank, src, tgt, w, total)
    return out


def _print_summary(res: dict) -> None:
    print("\n" + "=" * 78)
    print("H22 SIZING — bounded-window discrete local-search opportunity")
    print("=" * 78)
    for ds, r in res.items():
        print(f"\n[{ds}]  n={r['n_nodes']:,}  H02={r['h02_pos_pct']:.4f}%  "
              f"(rank-faithful={r['rank_faithful']})")
        m1 = r["measure1_reachable_pool"]
        print(f"  total feedback weight = {m1['feedback_total_pp']:.4f} pp")
        print("  Measure 1 — feedback weight reachable within rank-window W (ceiling for a width-W move):")
        print("    W       pool(pp)   frac-of-feedback")
        for W in WINDOWS:
            c = m1["reachable_pool"][str(W)]
            print(f"    {W:<6}  {c['pool_pp']:8.4f}   {c['pool_frac_of_feedback']:6.2%}")
        if "measure2_gap_flip_distance" in r:
            m2 = r["measure2_gap_flip_distance"]
            print(f"\n  best = {r['best_pct']:.4f}%   "
                  f"gap (net) = {m2['net_total_pp']:+.4f} pp  "
                  f"[gain {m2['gain_total_pp']:+.4f} / lose {m2['lose_total_pp']:+.4f}]")
            print(f"    cross-check vs diagnosis (expect gain≈+4.60 / lose≈-2.91 / net≈+1.69)")
            pc = m2["gain_distance_percentiles"]
            print(f"    gain-weight rank-distance percentiles: "
                  + ", ".join(f"{k}={v:,}" for k, v in pc.items()))
            print("  Measure 2 — DECISIVE: net gap recoverable within rank-window W:")
            print("    W       net(pp)   gain(pp)   %of-total-gain")
            for W in WINDOWS:
                c = m2["cumulative_within_window"][str(W)]
                print(f"    {W:<6}  {c['net_pp']:+8.4f}  {c['gain_pp']:8.4f}   "
                      f"{c['gain_frac_of_total_gain']:6.2%}")
    print("\n" + "=" * 78)


def main() -> None:
    res = {}
    res["connectome"] = size_dataset("connectome", with_best=True)
    res["mouse"] = size_dataset("mouse", with_best=False)   # no mouse best_solution exists
    _print_summary(res)
    out_path = OUT / "localsearch_sizing.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
