"""H78b - is there an INCREMENTAL path from the champion order to the reference order?

DIVERGENT-MODE DIAGNOSTIC (cycle 21), second half. Privileged: reads the reference
ordering only through :mod:`mfas.analysis.gap`. Writes NOTHING to ``results/``. No
optimization path consumes any number produced here - the reference order may never
enter an algorithm (CAMPAIGN.md rule 4).

THE QUESTION
------------
``h78_residual_gap_structure.py`` measured, from the CHAMPION order (H64, 84.2582%):

    gain +2.3787 pp / lose +2.0222 pp / NET +0.3565 pp

i.e. the champion and the reference disagree about the orientation of edges carrying
4.40 pp of weight, and 85% of that disagreement CANCELS. The two orders are structurally
far apart (53.6% of nodes carry gain stake; median node displacement 6,051 ranks) while
being only 0.357 pp apart in score.

That raises a formulation-level question no move class can answer: **is the reference
reachable from the champion by any monotone process at all?** If walking toward it must
first pay the 2.02 pp `lose` term, no hill-climb - however clever its move class - can
make the trip, and the campaign's remaining 0.357 pp is not a local-search problem.

TWO PROBES, both cheap (one argsort + one exact score each)
----------------------------------------------------------
(A) BLEND / path-relinking curve. key_t(v) = (1-t)*rank_champ(v) + t*rank_best(v), then
    argsort. t=0 is the champion, t=1 is the reference. This is the standard linear-ordering
    path-relinking trajectory. Its SHAPE is the answer:
      - monotone increasing        -> a continuous path exists; some blend already beats
                                      the champion and is directly actionable;
      - dips below the champion    -> a BARRIER separates the two optima; incremental
                                      adoption is impossible and the attack must move to
                                      the decomposition or the formulation.

(B) SUBSET-ADOPTION curve. Take the top-k nodes by residual gain stake (measure 4's
    statistic) and teleport ONLY those to their reference rank, keeping every other node
    at its champion rank. Scored for k on a log grid. Same question, different geometry:
    can a SMALL, well-chosen subset of the disagreement be imported profitably?

Both curves are upper-bounded by the reference itself; neither is an algorithm. Read them
with M11 in hand (capacity is not achievability): a positive point on either curve is a
CEILING that a leakage-free method would still have to discover on its own.

Usage:
    PYTHONPATH=src /c/ProgramData/anaconda3/envs/allen/python.exe
        experiments/diagnostics/h78_path_to_reference.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments"))

from mfas import io  # noqa: E402
from mfas.analysis import gap  # noqa: E402
from mfas.metrics import pct, score_from_order, score_from_positions  # noqa: E402
from size_localsearch import _rank_of_node  # noqa: E402

OUT = _ROOT / "experiments" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CHAMPION_GLOB = "*-H64-connectome-s42-*_positions.npy"
CHAMPION_ID = "H64"

# t grid for probe A: dense near both ends, since a barrier can be narrow.
T_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
          0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95,
          0.97, 0.98, 0.99, 0.995, 1.0]

# k grid for probe B (number of top-stake nodes teleported to their reference rank).
K_GRID = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 73_262, 136_648]


def _latest(pattern: str) -> str:
    fs = sorted(glob.glob(str(_ROOT / "results" / pattern)))
    if not fs:
        raise FileNotFoundError("no results file matches %s" % pattern)
    return fs[-1]


def _keys_to_rank(keys: np.ndarray) -> np.ndarray:
    """Stable argsort of arbitrary keys -> a proper rank vector in [0, n)."""
    order = np.argsort(keys, kind="stable")
    rank = np.empty(order.shape[0], dtype=np.int64)
    rank[order] = np.arange(order.shape[0], dtype=np.int64)
    return rank


def main() -> None:
    ds = "connectome"
    g = io.load_dataset(ds)
    src = np.asarray(g.src)
    tgt = np.asarray(g.tgt)
    w = np.asarray(g.weight, dtype=np.float64)
    total = float(g.total_weight)

    pos_path = _latest(CHAMPION_GLOB)
    pos = np.load(pos_path).astype(np.float64)
    champ_rank = _rank_of_node(pos)
    champ_pct = pct(score_from_order(champ_rank, src, tgt, g.weight), total)

    best_rank, best_pct = gap.load_best_solution(g)
    n = int(g.n_nodes)

    # sanity: the champion base must equal the position score and sota's champion
    sota = json.load(open(_ROOT / "autoresearch" / "sota.json"))
    sota_pct = sota["datasets"][ds]["pct_mean_exact"]
    pos_pct = pct(score_from_positions(pos, src, tgt, g.weight), total)
    assert abs(champ_pct - pos_pct) < 1e-9, (champ_pct, pos_pct)

    cr = champ_rank.astype(np.float64)
    br = best_rank.astype(np.float64)

    # ---------- probe A: blend curve ----------
    curve_a = []
    for t in T_GRID:
        keys = (1.0 - t) * cr + t * br
        r = _keys_to_rank(keys)
        s = pct(score_from_order(r, src, tgt, g.weight), total)
        curve_a.append(dict(t=t, pct=s, delta_vs_champion_pp=s - champ_pct))
        print("  [A] t=%-6.3f  %.10f%%   %+.6f pp" % (t, s, s - champ_pct))

    # ---------- probe B: subset adoption, ranked by residual gain stake ----------
    champ_d = champ_rank[tgt].astype(np.int64) - champ_rank[src].astype(np.int64)
    best_d = best_rank[tgt].astype(np.int64) - best_rank[src].astype(np.int64)
    gain = (champ_d <= 0) & (best_d > 0)
    stake = np.zeros(n, dtype=np.float64)
    np.add.at(stake, src[gain], w[gain])
    np.add.at(stake, tgt[gain], w[gain])
    by_stake = np.argsort(stake)[::-1]          # descending stake

    curve_b = []
    for k in K_GRID:
        if k > n:
            continue
        keys = cr.copy()
        sel = by_stake[:k]
        keys[sel] = br[sel]                      # teleport the chosen nodes only
        r = _keys_to_rank(keys)
        s = pct(score_from_order(r, src, tgt, g.weight), total)
        curve_b.append(dict(k=int(k), pct=s, delta_vs_champion_pp=s - champ_pct,
                            stake_covered_frac=float(stake[sel].sum() / stake.sum())))
        print("  [B] k=%-8d %.10f%%   %+.6f pp   (stake covered %.1f%%)"
              % (k, s, s - champ_pct, 100.0 * stake[sel].sum() / stake.sum()))

    best_a = max(curve_a[1:-1], key=lambda d: d["pct"]) if len(curve_a) > 2 else None
    best_b = max(curve_b[:-1], key=lambda d: d["pct"]) if len(curve_b) > 1 else None
    worst_a = min(curve_a, key=lambda d: d["pct"])

    out = dict(
        _doc=("H78b: is the reference order reachable from the champion incrementally? "
              "Blend (path-relinking) and subset-adoption curves. DIAGNOSTIC ONLY."),
        dataset=ds,
        base_variant=CHAMPION_ID,
        base_positions_file=Path(pos_path).name,
        champion_pct=champ_pct,
        sota_champion_pct=sota_pct,
        base_matches_sota=bool(abs(champ_pct - sota_pct) < 1e-9),
        best_solution_pct=best_pct,
        residual_gap_pp=best_pct - champ_pct,
        probeA_blend_curve=curve_a,
        probeB_subset_adoption_curve=curve_b,
        probeA_best_interior=best_a,
        probeA_worst=worst_a,
        probeB_best_proper_subset=best_b,
        interpretation_rule=(
            "If probeA's interior maximum is <= the champion and probeB's best proper "
            "subset is <= the champion, then EVERY partial adoption of the reference "
            "loses: the two orders are separated by a barrier and the residual 0.357 pp "
            "is not reachable by any monotone/hill-climbing process from this basin. "
            "That is a formulation-level result, not a move-class result."),
    )

    print("\n" + "=" * 78)
    print("H78b - path from the champion to the reference")
    print("=" * 78)
    print("champion        = %.10f%%  (matches sota=%s)" % (champ_pct, out["base_matches_sota"]))
    print("best_solution   = %.10f%%" % best_pct)
    print("residual gap    = %+.6f pp" % out["residual_gap_pp"])
    if best_a:
        print("\n[A] best INTERIOR blend : t=%.3f -> %.10f%%  (%+.6f pp vs champion)"
              % (best_a["t"], best_a["pct"], best_a["delta_vs_champion_pp"]))
    print("[A] worst point         : t=%.3f -> %.10f%%  (%+.6f pp vs champion)"
          % (worst_a["t"], worst_a["pct"], worst_a["delta_vs_champion_pp"]))
    if best_b:
        print("[B] best PROPER subset  : k=%d -> %.10f%%  (%+.6f pp vs champion)"
              % (best_b["k"], best_b["pct"], best_b["delta_vs_champion_pp"]))
    barrier = (best_a is not None and best_a["delta_vs_champion_pp"] <= 0.0
               and best_b is not None and best_b["delta_vs_champion_pp"] <= 0.0)
    out["barrier_confirmed"] = bool(barrier)
    print("\nBARRIER CONFIRMED = %s" % barrier)

    p = OUT / "proto_H78b_path_connectome.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
