"""H78c - SLOT-PRESERVING partial adoption of the reference order (robustness probe).

DIVERGENT-MODE DIAGNOSTIC (cycle 21), third probe. Privileged: reads the reference order
only through :mod:`mfas.analysis.gap`. Writes NOTHING to ``results/``. No optimization
path consumes any number produced here (CAMPAIGN.md rule 4).

WHY A THIRD PROBE
-----------------
``h78_path_to_reference.py`` found (a) a 0.786 pp valley along the linear rank-blend path
and (b) that teleporting the top-k highest-stake nodes to their REFERENCE RANK loses
score for every proper k, monotonically. Probe (b) has a confound worth removing: a
teleport changes the ranks of the NON-selected nodes too, because inserting k nodes at
foreign ranks shifts everyone they pass. So some of that loss may be collateral rather
than a property of the reference's opinion about the subset.

This probe removes the confound. SLOT-PRESERVING ADOPTION:

    take a subset S; keep exactly the rank-slots S occupies in the CHAMPION order;
    re-fill those slots with the members of S in the order the REFERENCE puts them in.

Every node outside S keeps its champion rank EXACTLY. Only the internal ordering of S
changes, and it changes to the reference's. At S = all nodes this yields the reference
order; at |S| = 1 it is a no-op. It is the cleanest possible question: "is the reference's
opinion about the relative order of these nodes worth importing?"

Three subset selection rules are run, so the answer does not rest on one ranking:
  * ``stake``  - descending residual gain weight incident on the node
  * ``disp``   - descending |rank_champ - rank_best|
  * ``random`` - a seeded control

Read with M11 (capacity is not achievability): a positive point here is a CEILING that a
leakage-free method would still have to discover on its own.

Usage:
    PYTHONPATH=src /c/ProgramData/anaconda3/envs/allen/python.exe
        experiments/diagnostics/h78_slot_adoption.py
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
from mfas.metrics import pct, score_from_order  # noqa: E402
from size_localsearch import _rank_of_node  # noqa: E402

OUT = _ROOT / "experiments" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CHAMPION_GLOB = "*-H64-connectome-s42-*_positions.npy"
CHAMPION_ID = "H64"
SEED = 42
K_GRID = [2, 10, 100, 1_000, 10_000, 30_000, 73_262, 136_648]


def _latest(pattern: str) -> str:
    fs = sorted(glob.glob(str(_ROOT / "results" / pattern)))
    if not fs:
        raise FileNotFoundError("no results file matches %s" % pattern)
    return fs[-1]


def _slot_adopt(champ_rank: np.ndarray, best_rank: np.ndarray,
                subset: np.ndarray) -> np.ndarray:
    """Re-fill the champion rank-slots of ``subset`` in reference-relative order.

    Nodes outside ``subset`` keep their champion rank exactly.
    """
    new_rank = champ_rank.copy()
    slots = np.sort(champ_rank[subset])                    # the slots S occupies
    members_in_best_order = subset[np.argsort(best_rank[subset], kind="stable")]
    new_rank[members_in_best_order] = slots
    return new_rank


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

    sota = json.load(open(_ROOT / "autoresearch" / "sota.json"))
    sota_pct = sota["datasets"][ds]["pct_mean_exact"]

    # residual gain stake and displacement, for the two informed orderings
    champ_d = champ_rank[tgt].astype(np.int64) - champ_rank[src].astype(np.int64)
    best_d = best_rank[tgt].astype(np.int64) - best_rank[src].astype(np.int64)
    gain = (champ_d <= 0) & (best_d > 0)
    stake = np.zeros(n, dtype=np.float64)
    np.add.at(stake, src[gain], w[gain])
    np.add.at(stake, tgt[gain], w[gain])
    disp = np.abs(champ_rank.astype(np.int64) - best_rank.astype(np.int64))

    rng = np.random.default_rng(SEED)
    rules = {
        "stake": np.argsort(stake)[::-1],
        "disp": np.argsort(disp)[::-1],
        "random": rng.permutation(n),
    }

    results = {}
    for name, ordering in rules.items():
        curve = []
        for k in K_GRID:
            if k > n:
                continue
            subset = ordering[:k].astype(np.int64)
            r = _slot_adopt(champ_rank, best_rank, subset)
            s = pct(score_from_order(r, src, tgt, g.weight), total)
            curve.append(dict(k=int(k), pct=s, delta_vs_champion_pp=s - champ_pct))
            print("  [%s] k=%-8d %.10f%%   %+.6f pp" % (name, k, s, s - champ_pct))
        results[name] = curve

    # best PROPER subset across all three rules (k = n is the reference itself)
    proper = [d for name, c in results.items() for d in c if d["k"] < n]
    best_proper = max(proper, key=lambda d: d["pct"])
    best_rule = [name for name, c in results.items()
                 if any(d is best_proper for d in c)][0]

    out = dict(
        _doc=("H78c: slot-preserving partial adoption of the reference order. Nodes outside "
              "the subset keep their champion rank EXACTLY; only the subset's internal order "
              "is replaced by the reference's. DIAGNOSTIC ONLY."),
        dataset=ds,
        base_variant=CHAMPION_ID,
        base_positions_file=Path(pos_path).name,
        champion_pct=champ_pct,
        sota_champion_pct=sota_pct,
        base_matches_sota=bool(abs(champ_pct - sota_pct) < 1e-9),
        best_solution_pct=best_pct,
        residual_gap_pp=best_pct - champ_pct,
        seed=SEED,
        k_grid=K_GRID,
        curves=results,
        best_proper_subset=dict(rule=best_rule, **best_proper),
        no_profitable_partial_adoption=bool(best_proper["delta_vs_champion_pp"] <= 0.0),
        interpretation=(
            "If no_profitable_partial_adoption is true, the reference's advantage is not "
            "decomposable: importing its opinion about ANY tested subset - even with zero "
            "collateral displacement of other nodes - loses score. The 0.357 pp lives in "
            "the whole permutation, not in any part of it."),
    )

    print("\n" + "=" * 78)
    print("H78c - slot-preserving partial adoption")
    print("=" * 78)
    print("champion      = %.10f%%  (matches sota=%s)" % (champ_pct, out["base_matches_sota"]))
    print("best_solution = %.10f%%  (residual %+.6f pp)" % (best_pct, out["residual_gap_pp"]))
    print("best PROPER subset over all rules: %s k=%d -> %.10f%%  (%+.6f pp)"
          % (best_rule, best_proper["k"], best_proper["pct"],
             best_proper["delta_vs_champion_pp"]))
    print("NO PROFITABLE PARTIAL ADOPTION = %s" % out["no_profitable_partial_adoption"])

    p = OUT / "proto_H78c_slot_adoption.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
