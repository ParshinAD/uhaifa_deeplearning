"""H79 rung 1 - are there structurally DIFFERENT starting constructions at all?

DIVERGENT-MODE PRE-GATE (cycle 22). Pure CPU, no GPU, no ``results/`` writes, no
optimization path consumes any number produced here. Leakage-safe: every construction
reads only ``g.src`` / ``g.tgt`` / ``g.weight``. ``data/best_solution`` is never opened.

WHY THIS EXISTS
---------------
M15 (cycle 21) says the reference solution's remaining +0.356498 pp is a property of the
WHOLE permutation and of no part of it, so from the champion's basin no monotone process
reaches it. The champion's basin has never varied: ``H02.greedy_fas_order`` is a
deterministic pure function of the graph, so every variant since 2026-08-09 has refined
the SAME construction. This script asks the cheapest form of the question M15 licenses:
**is there more than one basin to choose from at all?**

It is a MEASUREMENT of geometry, not a warm-start proposal. H48's kill condition
("never again as a drop-in warm start for the current stack") is respected: the
ratio-greedy construction appears here as a REFERENCE POINT with a known post-pipeline
outcome (-0.01382 pp) and is excluded by name from any rung-2 arm.

THE CONTROL, and it is the whole point
--------------------------------------
"Different from the champion's construction" is meaningless without a scale. The scale
used here is the champion's OWN construction ball: greedy-FAS re-run on R randomly
RELABELLED copies of the same graph (an exact isomorphism - only index tie-breaks move),
mapped back to canonical node ids. M10 measured that this relabelling is where ~96% of
the pipeline's nuisance dispersion is created (sigma 0.019124 pp), so the ball is the
campaign's already-established unit of "same construction, different tie-breaks".

A construction is called DISTINCT iff its Kendall tau against canonical greedy-FAS falls
BELOW the minimum pairwise tau inside that ball. Anything inside the ball is the champion's
construction wearing a different hat, and H56/H57/M9/M10 already killed harvesting it.

PRE-REGISTERED DECISION RULE (sealed in experiments/outputs/proto_H79_prereg.json and
committed BEFORE this script is run)
------------------------------------------------------------------------------------
  ball_tau_min = min pairwise Kendall tau over {canonical, relabelled r=1..R}
  distinct(C)  = tau(C, canonical) < ball_tau_min
  RUNG 1 PASSES iff at least TWO admissible constructions (i.e. excluding ratio_greedy,
  which H48 blocks) are distinct. Fewer than two leaves no SPREAD to measure at rung 2,
  only a single alternative, which is a warm-start swap and is what H48 already killed.
  RUNG 1 KILLS H79 otherwise.

WHAT THIS CANNOT SHOW, stated before the numbers
-------------------------------------------------
Distinct INITS need not imply distinct BASINS: M6 ("the init -> plateau curve is flat")
and M2 both describe a stack that erases input differences, and H48 measured the sift
absorbing 99.4% of a +5.70 pp init advantage. Rung 1 can therefore only ever REFUSE - it
cannot establish that the basin is a live degree of freedom. That is rung 2's job, and
rung 2 is the expensive one, which is exactly why this rung runs first.

Run:
    PYTHONPATH=src python experiments/diagnostics/h79_basin_distance.py \
        --dataset connectome --relabellings 5 \
        --out experiments/outputs/proto_H79_rung1.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from mfas.experiments.H02 import greedy_fas_order
from mfas.experiments.H79 import BLOCKED_AS_ARM, CONSTRUCTIONS
from mfas.io import GraphData, load_dataset
from mfas.metrics import pct, score_from_order

# ── the control: the champion's own construction ball ────────────────────────────────
#
# The CONSTRUCTIONS themselves live in src/mfas/experiments/H79.py and are imported, not
# copied, so this measurement and the rung-2 variant arms are guaranteed to be measuring
# the same functions. (Cycle 21's H78 used the same discipline for size_localsearch.py.)


def _relabelled(g: GraphData, seed: int) -> "tuple[GraphData, np.ndarray]":
    """Return an isomorphic copy of ``g`` under a random node relabelling, plus the map.

    ``perm[u]`` is the NEW index of canonical node ``u``. The returned graph is the same
    graph: only the integer names of the nodes change, so any construction's output on it
    can be mapped back and compared rank-for-rank with a canonical-labelling output.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(g.n_nodes).astype(np.int64)
    gg = GraphData(src=perm[np.asarray(g.src, dtype=np.int64)].astype(np.int32),
                   tgt=perm[np.asarray(g.tgt, dtype=np.int64)].astype(np.int32),
                   weight=g.weight, node_ids=g.node_ids, name=g.name)
    return gg, perm


# ── driver ───────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="connectome")
    ap.add_argument("--relabellings", type=int, default=5)
    ap.add_argument("--champion-positions", default="",
                    help="optional .npy of the champion's FINAL positions, for context only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    g = load_dataset(args.dataset)
    total_w = g.total_weight
    print(f"[H79.1] {g!r}", flush=True)

    ranks: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    scores: dict[str, float] = {}

    for name, fn in CONSTRUCTIONS.items():
        t = time.time()
        r = np.asarray(fn(g), dtype=np.int64)
        timings[name] = time.time() - t
        if not np.array_equal(np.sort(r), np.arange(g.n_nodes)):
            raise RuntimeError(f"{name} did not return a permutation")
        ranks[name] = r
        scores[name] = pct(score_from_order(r, g.src, g.tgt, g.weight), total_w)
        print(f"[H79.1] {name:>20s}  {scores[name]:.6f} %  ({timings[name]:.1f} s)", flush=True)

    # The champion's own construction ball: greedy-FAS under R relabellings, mapped back.
    for r_i in range(1, args.relabellings + 1):
        gg, perm = _relabelled(g, seed=1000 + r_i)
        t = time.time()
        rank_relab = np.asarray(greedy_fas_order(gg), dtype=np.int64)
        timings[f"ball_r{r_i}"] = time.time() - t
        rank_back = rank_relab[perm]                    # canonical node u sits at perm[u] there
        ranks[f"ball_r{r_i}"] = rank_back
        scores[f"ball_r{r_i}"] = pct(score_from_order(rank_back, g.src, g.tgt, g.weight), total_w)
        print(f"[H79.1] {'ball_r%d' % r_i:>20s}  {scores[f'ball_r{r_i}']:.6f} %  "
              f"({timings[f'ball_r{r_i}']:.1f} s)", flush=True)

    if args.champion_positions:
        pos = np.load(args.champion_positions)
        seq = np.argsort(np.asarray(pos, dtype=np.float64), kind="stable")
        rk = np.empty(g.n_nodes, dtype=np.int64)
        rk[seq] = np.arange(g.n_nodes, dtype=np.int64)
        ranks["champion_final"] = rk
        scores["champion_final"] = pct(score_from_order(rk, g.src, g.tgt, g.weight), total_w)
        print(f"[H79.1] {'champion_final':>20s}  {scores['champion_final']:.6f} %", flush=True)

    names = list(ranks)
    tau = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            tau[f"{a}|{b}"] = float(kendalltau(ranks[a], ranks[b])[0])
            print(f"[H79.1] tau {a} | {b} = {tau[f'{a}|{b}']:.6f}", flush=True)

    ball = ["greedy_fas"] + [f"ball_r{i}" for i in range(1, args.relabellings + 1)]
    ball_pairs = {k: v for k, v in tau.items()
                  if k.split("|")[0] in ball and k.split("|")[1] in ball}
    ball_tau_min = min(ball_pairs.values())
    ball_tau_max = max(ball_pairs.values())

    verdict = {}
    for name in CONSTRUCTIONS:
        if name == "greedy_fas":
            continue
        key = f"greedy_fas|{name}" if f"greedy_fas|{name}" in tau else f"{name}|greedy_fas"
        t_c = tau[key]
        verdict[name] = {
            "tau_vs_canonical": t_c,
            "distinct": bool(t_c < ball_tau_min),
            "admissible_as_rung2_arm": name not in BLOCKED_AS_ARM,
            "blocked_reason": BLOCKED_AS_ARM.get(name, None),
            "init_pct": scores[name],
        }

    admissible_distinct = [n for n, v in verdict.items()
                           if v["distinct"] and v["admissible_as_rung2_arm"]]
    passed = len(admissible_distinct) >= 2

    out = {
        "id": "H79-rung1",
        "dataset": args.dataset,
        "n_nodes": int(g.n_nodes),
        "total_weight": float(total_w),
        "prereg": "experiments/outputs/proto_H79_prereg.json",
        "relabellings": int(args.relabellings),
        "init_pct": scores,
        "wall_clock_s": timings,
        "kendall_tau": tau,
        "spearman_rho": None,
        "spearman_note": ("NOT RECORDED: scipy.stats.spearmanr aborts this interpreter with "
                          "Windows fatal exception 0xc06d007f inside numpy.corrcoef -> np.dot "
                          "(broken BLAS in the `allen` env), at every n including n=10. Kendall "
                          "tau, the pre-registered statistic, is unaffected. Filed as P26."),
        "ball_tau_min": ball_tau_min,
        "ball_tau_max": ball_tau_max,
        "ball_pairs": ball_pairs,
        "verdict_per_construction": verdict,
        "admissible_distinct": admissible_distinct,
        "rung1_pass": bool(passed),
        "decision_rule": ("distinct(C) iff tau(C, greedy_fas) < min pairwise tau inside the "
                          "champion's relabelling ball; rung 1 passes iff >= 2 ADMISSIBLE "
                          "constructions are distinct (ratio_greedy is blocked by H48)"),
        "total_wall_clock_s": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[H79.1] ball tau in [{ball_tau_min:.6f}, {ball_tau_max:.6f}]", flush=True)
    print(f"[H79.1] admissible distinct: {admissible_distinct}", flush=True)
    print(f"[H79.1] RUNG 1 {'PASS' if passed else 'KILL'} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
