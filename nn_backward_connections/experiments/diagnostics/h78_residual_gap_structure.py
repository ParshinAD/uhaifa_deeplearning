"""H78 - where does the RESIDUAL connectome gap live, measured from the CHAMPION?

DIVERGENT-MODE DIAGNOSTIC (cycle 21). Reads the privileged near-optimal ordering ONLY
through :mod:`mfas.analysis.gap`; writes NOTHING to ``results/``. No optimization path
consumes any number produced here.

WHY THIS EXISTS
---------------
Every structural statement this campaign makes about the connectome gap -- M4
("prefer global-range or structurally decomposed move classes"), the H22 sizing kill,
and the p25=8,290 / p50=22,580 / p90=87,497 rank-distance percentiles quoted in
findings.md #3 -- was measured from **H02's converged order at 82.9161%**
(``experiments/outputs/localsearch_sizing.json``, 2026-06-22).

The champion is now **H64 at 84.2582%**. Of the ~1.70 pp gap those numbers describe,
~1.34 pp (~79%) has since been harvested, by exactly the global/structural move classes
M4 recommended. Nobody has ever looked at the ~0.357 pp that is left.

So the campaign's map of the gap describes terrain it has already crossed. This script
redraws it from where the campaign actually stands.

MEASURES
--------
(1) + (2) are the H22 sizing's own two measures, IMPORTED VERBATIM from
    ``experiments/size_localsearch.py`` so the champion-era and H02-era numbers are
    produced by identical code and differ only in the base order:
      (1) ``_reachable_pool``    -- LEAKAGE-SAFE ceiling: feedback weight within W ranks.
      (2) ``_gap_flip_distance`` -- PRIVILEGED: rank-distance of the base<->best
          orientation flips, split gain / lose, in the BASE order.
    Read measure 2 with M4's amendment in hand: it describes how short-range the
    REFERENCE's own rearrangement is, not a bound on a hill-climb.

(3) NEW -- node displacement: |rank_base(v) - rank_best(v)| for every node, unweighted
    and weighted by the node's incident gain weight. Answers "is the residual a few
    badly-placed nodes or a diffuse rearrangement?"

(4) NEW -- concentration: what fraction of the residual gain weight sits on the heaviest
    k gain edges, and how many DISTINCT nodes carry the top decile. A move class can only
    be worth building if the weight it must move is carried by few enough objects.

FALSIFIABLE PREDICTION UNDER TEST (H78)
---------------------------------------
The pipeline harvested the long-range component preferentially (that is what a full-range
sift plus SCC-recursive block refinement DO), so the residual flips should be shifted
substantially SHORTER than the H02-era p50 = 22,580.

KILL CONDITION: if the champion-era gain-distance median is within +/-25% of the H02-era
22,580, the pipeline harvested uniformly across scales, no newly-reachable short-range
family exists, M4's directive stands unamended, and the campaign must attack the
decomposition or the formulation rather than the move class.

Usage:
    PYTHONPATH=src /c/ProgramData/anaconda3/envs/allen/python.exe
        experiments/diagnostics/h78_residual_gap_structure.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments"))

from mfas import io  # noqa: E402
from mfas.analysis import gap  # noqa: E402
from mfas.metrics import pct, score_from_order, score_from_positions  # noqa: E402

# Imported VERBATIM so champion-era and H02-era numbers come from identical code.
from size_localsearch import (  # noqa: E402
    WINDOWS, _rank_of_node, _reachable_pool, _gap_flip_distance,
)

OUT = _ROOT / "experiments" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# The H02-era comparator, quoted from experiments/outputs/localsearch_sizing.json so the
# shift is computed rather than eyeballed.
H02_ERA_JSON = OUT / "localsearch_sizing.json"

# Champion connectome order. H64 is deterministic (never draws from `seed`), so any of its
# seeds is the same vector; s42 implement is the earliest recorded.
CHAMPION_GLOB = "*-H64-connectome-s42-*_positions.npy"
CHAMPION_ID = "H64"


def _latest(pattern: str) -> str:
    fs = sorted(glob.glob(str(_ROOT / "results" / pattern)))
    if not fs:
        raise FileNotFoundError(f"no results file matches {pattern}")
    return fs[-1]


def _weighted_pcts(values: np.ndarray, weights: Optional[np.ndarray]) -> dict:
    """Percentiles of ``values``, optionally weighted by ``weights``."""
    if values.size == 0:
        return {}
    qs = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    if weights is None:
        return {"p%d" % int(q * 100): int(np.percentile(values, q * 100)) for q in qs}
    o = np.argsort(values)
    v, wt = values[o], weights[o]
    cw = np.cumsum(wt) / max(wt.sum(), 1e-12)
    out = {}
    for q in qs:
        idx = min(int(np.searchsorted(cw, q)), v.shape[0] - 1)
        out["p%d" % int(q * 100)] = int(v[idx])
    return out


def _displacement(base_rank: np.ndarray, best_rank: np.ndarray,
                  gain_src: np.ndarray, gain_tgt: np.ndarray,
                  gain_w: np.ndarray, n_nodes: int) -> dict:
    """Measure 3: how far each node must travel from the base order to the reference."""
    disp = np.abs(base_rank.astype(np.int64) - best_rank.astype(np.int64))

    # gain weight incident on each node (a node's "stake" in the residual)
    stake = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(stake, gain_src, gain_w)
    np.add.at(stake, gain_tgt, gain_w)

    involved = stake > 0
    return dict(
        n_nodes=int(n_nodes),
        n_nodes_with_gain_stake=int(involved.sum()),
        frac_nodes_with_gain_stake=float(involved.mean()),
        displacement_unweighted=_weighted_pcts(disp, None),
        displacement_gain_weighted=_weighted_pcts(disp[involved], stake[involved]),
        mean_displacement_unweighted=float(disp.mean()),
        n_nodes_displaced_0=int((disp == 0).sum()),
    )


def _concentration(gain_w: np.ndarray, gain_src: np.ndarray, gain_tgt: np.ndarray,
                   total: float) -> dict:
    """Measure 4: is the residual gain weight carried by few edges / few nodes?"""
    if gain_w.size == 0:
        return {}
    o = np.argsort(gain_w)[::-1]
    w_sorted = gain_w[o]
    cw = np.cumsum(w_sorted)
    tot = float(cw[-1])
    frac_by_topk = {}
    for k in (10, 100, 1_000, 10_000, 100_000):
        if k <= w_sorted.size:
            frac_by_topk[str(k)] = float(cw[k - 1] / tot)
    # how many edges carry 50% / 90% / 99% of the residual gain weight
    n_for = {}
    for q in (0.5, 0.9, 0.99):
        n_for["q%d" % int(q * 100)] = int(np.searchsorted(cw, q * tot) + 1)
    # distinct nodes carrying the top decile of gain weight
    k_dec = max(1, int(np.searchsorted(cw, 0.10 * tot) + 1))
    top_nodes = np.unique(np.concatenate([gain_src[o][:k_dec], gain_tgt[o][:k_dec]]))
    return dict(
        n_gain_edges=int(gain_w.size),
        gain_total_pp=100.0 * tot / total,
        frac_of_gain_in_top_k_edges=frac_by_topk,
        n_edges_carrying=n_for,
        n_edges_in_top_decile=int(k_dec),
        n_distinct_nodes_in_top_decile=int(top_nodes.size),
        max_gain_edge_weight=float(w_sorted[0]),
    )


def main() -> None:
    ds = "connectome"
    g = io.load_dataset(ds)
    src = np.asarray(g.src)
    tgt = np.asarray(g.tgt)
    w = np.asarray(g.weight, dtype=np.float64)
    total = float(g.total_weight)

    pos_path = _latest(CHAMPION_GLOB)
    pos = np.load(pos_path).astype(np.float64)
    base_rank = _rank_of_node(pos)

    # cross-check: the rank order must reproduce the position score, and that score must
    # be the champion recorded in sota.json. If either fails, the base order is wrong.
    pos_pct = pct(score_from_positions(pos, src, tgt, g.weight), total)
    rank_pct = pct(score_from_order(base_rank, src, tgt, g.weight), total)
    sota = json.load(open(_ROOT / "autoresearch" / "sota.json"))
    champ_pct = sota["datasets"][ds]["pct_mean_exact"]

    best_rank, best_pct = gap.load_best_solution(g)

    m2 = _gap_flip_distance(base_rank, best_rank, src, tgt, w, total)

    base_d = base_rank[tgt].astype(np.int64) - base_rank[src].astype(np.int64)
    best_d = best_rank[tgt].astype(np.int64) - best_rank[src].astype(np.int64)
    gain = (base_d <= 0) & (best_d > 0)

    out = dict(
        _doc=("H78 divergent diagnostic: residual connectome gap structure measured from "
              "the CHAMPION order, not H02's. Measures 1 and 2 are imported verbatim from "
              "experiments/size_localsearch.py."),
        dataset=ds,
        base_variant=CHAMPION_ID,
        base_positions_file=Path(pos_path).name,
        n_nodes=int(g.n_nodes), n_edges=int(g.n_edges), total_weight=total,
        base_pos_pct=pos_pct,
        base_rank_pct=rank_pct,
        rank_faithful=bool(abs(pos_pct - rank_pct) < 1e-9),
        sota_champion_pct=champ_pct,
        base_matches_sota=bool(abs(pos_pct - champ_pct) < 1e-9),
        best_pct=best_pct,
        residual_gap_pp=best_pct - pos_pct,
        measure1_reachable_pool=_reachable_pool(base_rank, src, tgt, w, total),
        measure2_gap_flip_distance=m2,
        measure3_node_displacement=_displacement(
            base_rank, best_rank, src[gain], tgt[gain], w[gain], int(g.n_nodes)),
        measure4_concentration=_concentration(w[gain], src[gain], tgt[gain], total),
    )

    # ---- the H02-era comparison, computed not eyeballed ----
    if H02_ERA_JSON.exists():
        old = json.load(open(H02_ERA_JSON))["connectome"]
        o2 = old["measure2_gap_flip_distance"]
        op = o2["gain_distance_percentiles"]
        npc = m2["gain_distance_percentiles"]
        out["h02_era_comparison"] = dict(
            source=str(H02_ERA_JSON.relative_to(_ROOT)),
            h02_base_pct=old["h02_pos_pct"],
            h02_net_gap_pp=o2["net_total_pp"],
            champion_net_gap_pp=m2["net_total_pp"],
            gap_harvested_pp=o2["net_total_pp"] - m2["net_total_pp"],
            frac_of_h02_gap_harvested=(
                (o2["net_total_pp"] - m2["net_total_pp"]) / o2["net_total_pp"]
                if o2["net_total_pp"] else None),
            gain_distance_percentiles_h02=op,
            gain_distance_percentiles_champion=npc,
            p50_shift_ratio=(npc["p50"] / op["p50"] if op.get("p50") else None),
            verdict_rule=("H78 kill condition: |p50_champion/p50_h02 - 1| <= 0.25 means the "
                          "pipeline harvested uniformly across scales -> no short-range "
                          "family revived -> M4 stands and the attack must move to the "
                          "decomposition or the formulation."),
        )

    # ---- summary ----
    print("=" * 78)
    print("H78 - RESIDUAL connectome gap structure, measured from the CHAMPION")
    print("=" * 78)
    print("base            = %s  %s" % (CHAMPION_ID, Path(pos_path).name))
    print("base score      = %.10f%%   (rank-faithful=%s, matches sota=%s)"
          % (pos_pct, out["rank_faithful"], out["base_matches_sota"]))
    print("best_solution   = %.10f%%" % best_pct)
    print("RESIDUAL gap    = %+.6f pp" % out["residual_gap_pp"])
    m2p = m2["gain_distance_percentiles"]
    print("\ngain %+.4f pp / lose %+.4f pp / net %+.4f pp   (%d gain edges, %d lose edges)"
          % (m2["gain_total_pp"], m2["lose_total_pp"], m2["net_total_pp"],
             m2["n_gain_edges"], m2["n_lose_edges"]))
    print("gain-weight rank-distance percentiles (CHAMPION order): "
          + ", ".join("%s=%s" % (k, format(v, ",")) for k, v in m2p.items()))
    if "h02_era_comparison" in out:
        c = out["h02_era_comparison"]
        print("H02-era                                              : "
              + ", ".join("%s=%s" % (k, format(v, ","))
                          for k, v in c["gain_distance_percentiles_h02"].items()))
        print("\nH02-era net gap %+.4f pp -> champion net gap %+.4f pp  (%.1f%% harvested)"
              % (c["h02_net_gap_pp"], c["champion_net_gap_pp"],
                 100.0 * c["frac_of_h02_gap_harvested"]))
        print("p50 shift ratio = %.4f  (KILL if within 0.75-1.25)" % c["p50_shift_ratio"])
    print("\nMeasure 2 - net gap recoverable within rank-window W (M4-amended reading):")
    print("  W        net(pp)    gain(pp)   %of-total-gain")
    for W in WINDOWS:
        c = m2["cumulative_within_window"][str(W)]
        print("  %-7s  %+8.4f  %9.4f   %6.2f%%"
              % (W, c["net_pp"], c["gain_pp"], 100.0 * c["gain_frac_of_total_gain"]))
    m1 = out["measure1_reachable_pool"]
    print("\nMeasure 1 - leakage-safe CEILING; total feedback now %.4f pp"
          % m1["feedback_total_pp"])
    print("  W        pool(pp)   frac-of-feedback")
    for W in WINDOWS:
        c = m1["reachable_pool"][str(W)]
        print("  %-7s  %8.4f   %6.2f%%"
              % (W, c["pool_pp"], 100.0 * c["pool_frac_of_feedback"]))
    m3 = out["measure3_node_displacement"]
    print("\nMeasure 3 - node displacement |rank_champ - rank_best|:")
    print("  nodes with gain stake = %s (%.2f%% of %s)"
          % (format(m3["n_nodes_with_gain_stake"], ","),
             100.0 * m3["frac_nodes_with_gain_stake"], format(m3["n_nodes"], ",")))
    print("  unweighted    : "
          + ", ".join("%s=%s" % (k, format(v, ","))
                      for k, v in m3["displacement_unweighted"].items()))
    print("  gain-weighted : "
          + ", ".join("%s=%s" % (k, format(v, ","))
                      for k, v in m3["displacement_gain_weighted"].items()))
    m4 = out["measure4_concentration"]
    print("\nMeasure 4 - concentration of the %.4f pp residual gain:" % m4["gain_total_pp"])
    print("  edges carrying 50%% / 90%% / 99%% : %s / %s / %s  (of %s)"
          % (format(m4["n_edges_carrying"]["q50"], ","),
             format(m4["n_edges_carrying"]["q90"], ","),
             format(m4["n_edges_carrying"]["q99"], ","),
             format(m4["n_gain_edges"], ",")))
    print("  top decile of weight sits on %s edges spanning %s distinct nodes"
          % (format(m4["n_edges_in_top_decile"], ","),
             format(m4["n_distinct_nodes_in_top_decile"], ",")))

    p = OUT / "proto_H78_connectome.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
