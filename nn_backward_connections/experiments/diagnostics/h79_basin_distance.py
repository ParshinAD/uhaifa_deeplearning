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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import kendalltau

from mfas.baseline.ratio_greedy import ratio_greedy_rank
from mfas.experiments.H02 import greedy_fas_order
from mfas.io import GraphData, load_dataset
from mfas.metrics import pct, score_from_order

# ── constructions ────────────────────────────────────────────────────────────────────


def _relabelled(g: GraphData, seed: int) -> tuple[GraphData, np.ndarray]:
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


def c_greedy_fas(g: GraphData) -> np.ndarray:
    """The CHAMPION's construction (H02, Eades-Lin-Smyth / GreedyAbs peel)."""
    return greedy_fas_order(g)


def c_ratio_greedy(g: GraphData) -> np.ndarray:
    """H48's construction, (out_w+1)/(in_w+1) peel. REFERENCE POINT ONLY - see module doc."""
    return ratio_greedy_rank(g)


def c_reverse_greedy_fas(g: GraphData) -> np.ndarray:
    """Greedy-FAS on the TRANSPOSED graph, rank reversed.

    ELS is not symmetric: it peels sinks to the back, sources to the front, and otherwise
    sends max ``out_w - in_w`` to the FRONT, so the front is built greedily and the back
    only by sink detection. Transposing swaps those roles, which makes this a genuinely
    different rule and not a re-parameterisation of the same one.
    """
    gt = GraphData(src=g.tgt, tgt=g.src, weight=g.weight, node_ids=g.node_ids, name=g.name)
    return (g.n_nodes - 1) - greedy_fas_order(gt)


def c_imbalance_sort(g: GraphData) -> np.ndarray:
    """Static descending sort by ``out_w - in_w`` on the FULL graph - no peeling at all.

    The peeling family's defining feature is that a placement changes its neighbours' keys.
    This construction removes exactly that, keeping only the key. It is the cheapest
    possible structural contrast to greedy-FAS and it is deliberately a WORSE start.
    """
    n = g.n_nodes
    w = np.asarray(g.weight, dtype=np.float64)
    out_w = np.zeros(n, dtype=np.float64)
    in_w = np.zeros(n, dtype=np.float64)
    np.add.at(out_w, np.asarray(g.src, dtype=np.int64), w)
    np.add.at(in_w, np.asarray(g.tgt, dtype=np.int64), w)
    seq = np.lexsort((np.arange(n), -(out_w - in_w)))   # ties -> lowest node id first
    rank = np.empty(n, dtype=np.int64)
    rank[seq] = np.arange(n, dtype=np.int64)
    return rank


def c_scc_topo(g: GraphData) -> np.ndarray:
    """SCC-condensation topological seeding.

    The condensation of a digraph is a DAG, so its topological order is FORCED - every
    edge between distinct SCCs is feedforward in any order that respects it, which is the
    one part of the problem that has an exact answer. Within an SCC nodes are ordered by
    descending ``out_w - in_w`` computed on the INDUCED subgraph only.

    scipy's ``connected_components(connection='strong')`` already returns labels in
    reverse topological order of the condensation, but that is a documented implementation
    detail rather than a guarantee, so the topological order is recomputed here from the
    condensation's own edges (Kahn, lowest label first for determinism).
    """
    n = g.n_nodes
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)

    adj = csr_matrix((np.ones(src.shape[0], dtype=np.int8), (src, tgt)), shape=(n, n))
    n_comp, lab = connected_components(adj, directed=True, connection="strong")
    lab = lab.astype(np.int64)

    # Condensation edges (deduplicated), then Kahn with a lowest-label tie-break.
    ls, lt = lab[src], lab[tgt]
    keep = ls != lt
    ce = np.unique(np.stack([ls[keep], lt[keep]], axis=1), axis=0)
    indeg = np.zeros(n_comp, dtype=np.int64)
    np.add.at(indeg, ce[:, 1], 1)
    order_c = np.argsort(ce[:, 0], kind="stable")
    cstart = np.searchsorted(ce[order_c, 0], np.arange(n_comp + 1))
    cnbr = ce[order_c, 1]

    import heapq
    heap = [int(c) for c in np.nonzero(indeg == 0)[0]]
    heapq.heapify(heap)
    topo = np.empty(n_comp, dtype=np.int64)
    k = 0
    while heap:
        c = heapq.heappop(heap)
        topo[k] = c
        k += 1
        for j in range(cstart[c], cstart[c + 1]):
            d = int(cnbr[j])
            indeg[d] -= 1
            if indeg[d] == 0:
                heapq.heappush(heap, d)
    if k != n_comp:                                    # cannot happen: a condensation is a DAG
        raise RuntimeError(f"condensation is cyclic: {k} of {n_comp} components ordered")
    comp_rank = np.empty(n_comp, dtype=np.int64)
    comp_rank[topo] = np.arange(n_comp, dtype=np.int64)

    # Intra-SCC key: imbalance on the induced subgraph (both endpoints in the same SCC).
    intra = ls == lt
    out_w = np.zeros(n, dtype=np.float64)
    in_w = np.zeros(n, dtype=np.float64)
    np.add.at(out_w, src[intra], w[intra])
    np.add.at(in_w, tgt[intra], w[intra])
    key = -(out_w - in_w)
    seq = np.lexsort((np.arange(n), key, comp_rank[lab]))
    rank = np.empty(n, dtype=np.int64)
    rank[seq] = np.arange(n, dtype=np.int64)
    return rank


CONSTRUCTIONS = {
    "greedy_fas": c_greedy_fas,               # the champion's - the anchor, not an arm
    "ratio_greedy": c_ratio_greedy,           # H48 - REFERENCE POINT ONLY, blocked as an arm
    "reverse_greedy_fas": c_reverse_greedy_fas,
    "imbalance_sort": c_imbalance_sort,
    "scc_topo": c_scc_topo,
}
BLOCKED_AS_ARM = {"ratio_greedy": "H48 revival condition: never again as a drop-in warm start"}


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
