"""Variant family H79 - the connectome champion stack run from a DIFFERENT starting basin.

One change, and it is the first stage: ``order = greedy_fas_order(g)`` becomes
``order = CONSTRUCTIONS[_CONSTRUCTION](g)``. Everything after it - the ASYM surrogate,
the epoch budget, the optimizer, the LR and beta schedules, grad clipping, and stages 3-6
- is the champion's, imported FROM :mod:`mfas.experiments.H64` rather than copied, so the
arms cannot drift from the champion by editing.

Why this exists
---------------
M15 (cycle 21) measured that the reference solution's remaining +0.356498 pp is a property
of the WHOLE permutation and of no part of it: no partial adoption is profitable and the
natural path to it runs through a valley 2.2x the prize. Its operative directive names two
surviving levels, and this is the first of them - a structurally DIFFERENT basin, not a
better one.

**The deliverable is a SPREAD, not a delta.** The question is whether the starting basin is
a degree of freedom the campaign has never used, or whether the stack erases it (which is
M2/M6's shape and would close the axis). Reporting the MAX over arms would be a multi-start
claim, and M9/M10 already govern those.

The novelty argument, in full, because this axis is crowded
-----------------------------------------------------------
* **H48** killed ratio-greedy as a drop-in warm start with a revival condition that reads
  "never again as a drop-in warm start for the current stack". That construction is
  therefore NOT an arm here. It appears in the rung-1 geometry measurement as a reference
  point only, and ``CONSTRUCTIONS`` marks it blocked.
* **H56/H57** killed multi-start over RELABELLINGS of one construction. M10 states the one
  surviving door: "a mechanism that WIDENS the PREFIX's sigma at the champion's per-arm
  cost". A different CONSTRUCTION is exactly a prefix-level mechanism, and H56's revival
  condition (b) names the bar - widen past 0.021269 pp.
* **M6/M7** say better inits do not help and expensive ones lose. Neither is contradicted
  here: two of the three arms are WORSE starts than greedy-FAS and all three cost < 20 s.

Rung 1 (``experiments/diagnostics/h79_basin_distance.py``, which imports ``CONSTRUCTIONS``
from this module) established that these constructions are structurally distinct from the
champion's own relabelling ball: ball Kendall tau in [0.937858, 0.942799] against
tau(C, greedy_fas) of 0.089885 (reverse), 0.370638 (imbalance) and 0.432855 (scc_topo).

Leakage-safety
--------------
Every construction reads only ``g.src`` / ``g.tgt`` / ``g.weight``. The discrete oracle is
consulted exactly where the champion consults it (best-by-oracle tracking), and
``data/best_solution`` is never opened.

Determinism
-----------
No construction draws from ``seed``; ``make_init_positions`` is unreachable, as in the
champion. ``autoresearch/seed_class.py`` should classify every arm ``deterministic``, and
the mouse 3-seed tripwire stands.
"""
from __future__ import annotations

import heapq
import time
from typing import Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..baseline.ratio_greedy import ratio_greedy_rank
from ..baseline.rocket import RocketConfig, RocketResult
from ..io import GraphData
from ..metrics import pct, score_from_order
from ..refine import alternate_scc_sift, sift_underrelaxed
from ..refine.pair_relocate import pair_relocate
from ..refine.reclaim2 import reclaim_arcs_fast
from .H02 import _init_positions_from_order, greedy_fas_order
from .H38 import M, T
from .H64 import (_ALPHA, _ALT_CYCLES, _ALT_K_FULL, _ALT_SIFT_SWEEPS, _CONFLICT_BUDGET,
                  _EPOCHS, _K_FULL, _MAX_SWEEPS, _MIN_BLOCK, _PAIR_MAX_POPS, _PAIR_PASSES,
                  _RECLAIM_BUDGET, _RECLAIM_ROUNDS, _rocket_asym)

ID = "H79"
HYPOTHESIS = (
    "The champion's starting basin is a degree of freedom the campaign has never used. "
    "Running the UNCHANGED champion stack from structurally DIFFERENT (not better) "
    "constructions produces a spread of refined connectome scores materially wider than "
    "the 0.019124 pp relabelling nuisance sigma (M10). A spread <= that sigma says the "
    "stack erases the basin, which is M2/M6's shape, and closes the axis."
)

# ── the constructions ────────────────────────────────────────────────────────────────
# Each returns a RANK vector: rank[u] is the position of node u in [0, n), smaller =
# earlier = source side. Same convention as greedy_fas_order and ratio_greedy_rank; see
# ratio_greedy.py's "RETURN CONVENTION" note for why this is stated everywhere.


def c_greedy_fas(g: GraphData) -> np.ndarray:
    """The CHAMPION's construction (H02, Eades-Lin-Smyth / GreedyAbs peel). Identity anchor."""
    return greedy_fas_order(g)


def c_ratio_greedy(g: GraphData) -> np.ndarray:
    """H48's (out_w+1)/(in_w+1) peel. BLOCKED as an arm by H48's revival condition."""
    return ratio_greedy_rank(g)


def c_reverse_greedy_fas(g: GraphData) -> np.ndarray:
    """Greedy-FAS on the TRANSPOSED graph, rank reversed.

    ELS is not symmetric: it peels sinks to the back, sources to the front, and otherwise
    sends max ``out_w - in_w`` to the FRONT, so the front is built by the greedy rule and
    the back only by sink detection. Transposing swaps those roles, which makes this a
    different rule rather than a re-parameterisation of the same one.
    """
    gt = GraphData(src=g.tgt, tgt=g.src, weight=g.weight, node_ids=g.node_ids, name=g.name)
    return (g.n_nodes - 1) - greedy_fas_order(gt)


def c_imbalance_sort(g: GraphData) -> np.ndarray:
    """Static descending sort by ``out_w - in_w`` on the FULL graph - no peeling at all.

    The peeling family's defining feature is that placing a node changes its neighbours'
    keys. This removes exactly that and keeps only the key: the cheapest possible
    structural contrast to greedy-FAS, at 0.5 s against its 17 s. It is not a quality
    proposal - that it happens to start marginally higher (69.634453 % vs 68.913428 % on
    connectome) is incidental and, per M6/M7/H48, predicts nothing about the outcome.
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

    The condensation of a digraph is a DAG, so its topological order is FORCED - every edge
    between distinct SCCs is feedforward in any order respecting it, which is the one part
    of the problem with an exact answer. Within an SCC, nodes are ordered by descending
    ``out_w - in_w`` computed on the INDUCED subgraph only.

    scipy's ``connected_components(connection='strong')`` happens to return labels in
    reverse topological order, but that is an implementation detail rather than a
    guarantee, so the order is recomputed here from the condensation's own edges (Kahn,
    lowest label first, for determinism).
    """
    n = g.n_nodes
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)

    adj = csr_matrix((np.ones(src.shape[0], dtype=np.int8), (src, tgt)), shape=(n, n))
    n_comp, lab = connected_components(adj, directed=True, connection="strong")
    lab = lab.astype(np.int64)

    ls, lt = lab[src], lab[tgt]
    keep = ls != lt
    ce = np.unique(np.stack([ls[keep], lt[keep]], axis=1), axis=0)
    indeg = np.zeros(n_comp, dtype=np.int64)
    np.add.at(indeg, ce[:, 1], 1)
    order_c = np.argsort(ce[:, 0], kind="stable")
    cstart = np.searchsorted(ce[order_c, 0], np.arange(n_comp + 1))
    cnbr = ce[order_c, 1]

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
    if k != n_comp:                                # cannot happen: a condensation is a DAG
        raise RuntimeError(f"condensation is cyclic: {k} of {n_comp} components ordered")
    comp_rank = np.empty(n_comp, dtype=np.int64)
    comp_rank[topo] = np.arange(n_comp, dtype=np.int64)

    intra = ls == lt
    out_w = np.zeros(n, dtype=np.float64)
    in_w = np.zeros(n, dtype=np.float64)
    np.add.at(out_w, src[intra], w[intra])
    np.add.at(in_w, tgt[intra], w[intra])
    seq = np.lexsort((np.arange(n), -(out_w - in_w), comp_rank[lab]))
    rank = np.empty(n, dtype=np.int64)
    rank[seq] = np.arange(n, dtype=np.int64)
    return rank


CONSTRUCTIONS: dict[str, Callable[[GraphData], np.ndarray]] = {
    "greedy_fas": c_greedy_fas,
    "ratio_greedy": c_ratio_greedy,
    "reverse_greedy_fas": c_reverse_greedy_fas,
    "imbalance_sort": c_imbalance_sort,
    "scc_topo": c_scc_topo,
}
#: Constructions that may NOT be used as a variant arm, with the reason.
BLOCKED_AS_ARM = {
    "ratio_greedy": "H48 revival condition: never again as a drop-in warm start for this stack",
}

#: The construction this module's own ``run`` uses. ``greedy_fas`` makes H79 the IDENTITY
#: ANCHOR: it must reproduce the H64 champion bit-for-bit, which is what validates the arms.
_CONSTRUCTION = "greedy_fas"


def run_with_construction(g: GraphData, seed: int, device, time_limit: Optional[float],
                          construction: str) -> RocketResult:
    """The H64 champion pipeline, with stage 1's construction swapped for ``construction``.

    Line-for-line :func:`mfas.experiments.H64.run`; the ONLY difference is the ``order = ...``
    line below. Every constant and the gradient loop itself are imported from H64, so an
    edit to the champion propagates here rather than silently diverging.
    """
    if construction not in CONSTRUCTIONS:
        raise KeyError(f"unknown construction {construction!r}")
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))

    # -- Stage 1+2: THE ONLY CHANGE vs H64 -> Rocket with the ASYM surrogate (PURE) --
    order = CONSTRUCTIONS[construction](g)
    init_positions = _init_positions_from_order(order, device)
    init_score = score_from_order(np.asarray(order, dtype=np.int64),
                                  np.asarray(g.src, dtype=np.int64),
                                  np.asarray(g.tgt, dtype=np.int64), g.weight)
    rocket = _rocket_asym(g, cfg, seed=seed, device=device,
                          init_positions=init_positions, time_limit=time_limit)

    pure_best_positions = rocket.best_positions
    pure_best_score = rocket.best_score
    total = g.total_weight

    # -- Stage 3: H35's under-relaxed two-phase exact-gain sift -------------------
    rank0 = np.argsort(np.argsort(pure_best_positions, kind="stable"),
                       kind="stable").astype(np.int64)
    sift_budget = None if time_limit is None else max(0.0, time_limit - rocket.wall_clock_s)
    sift_rank, sift_score, sweep_log = sift_underrelaxed(
        g, rank0, k_full=_K_FULL, alpha=_ALPHA,
        max_sweeps=_MAX_SWEEPS.get(g.name, 40), time_budget_s=sift_budget)
    sift_time_s = float(sum(row["wall"] for row in sweep_log))

    # -- Stage 4: alternate block refinement with a SHORT single-node sift --------
    alt_budget = (None if time_limit is None
                  else max(0.0, time_limit - rocket.wall_clock_s - sift_time_s))
    alt_rank, alt_score, alt_log = alternate_scc_sift(
        g, sift_rank,
        n_cycles=_ALT_CYCLES.get(g.name, 32),
        sift_sweeps=_ALT_SIFT_SWEEPS.get(g.name, 2),
        k_full=_ALT_K_FULL, alpha=_ALPHA, min_block=_MIN_BLOCK,
        time_budget_s=alt_budget)
    alt_time_s = float(alt_log[-1]["cum_wall_s"]) if alt_log else 0.0

    # -- Stage 5: pair relocation - mouse only (its champion H63 carries H52's stage) --
    max_pops = _PAIR_MAX_POPS.get(g.name, 0)
    pair_rank, pair_score, pair_log, pair_time_s = alt_rank, alt_score, [], 0.0
    if max_pops:
        pair_budget = (None if time_limit is None else
                       max(0.0, time_limit - rocket.wall_clock_s - sift_time_s - alt_time_s))
        pair_rank, pair_score, pair_log = pair_relocate(
            g, alt_rank, n_passes=_PAIR_PASSES, max_pops=max_pops,
            time_budget_s=pair_budget)
        pair_time_s = float(pair_log[-1]["cum_wall_s"]) if pair_log else 0.0

    # -- Best-by-oracle across stages 2-5 ----------------------------------------
    base_score = pure_best_score
    base_positions = pure_best_positions
    if sift_score > base_score:
        base_score, base_positions = sift_score, sift_rank.astype(np.float32)
    if alt_score > base_score:
        base_score, base_positions = alt_score, alt_rank.astype(np.float32)
    if pair_score > base_score:
        base_score, base_positions = pair_score, pair_rank.astype(np.float32)

    # -- Stage 6: minimal-FAS arc reclamation - mouse only (H63) ------------------
    t_rec0 = time.time()
    rounds = _RECLAIM_ROUNDS.get(g.name, 0)
    rec_log = []
    rec_score, rec_positions = base_score, base_positions
    w_reclaimed, lemma_holds = 0.0, True
    if rounds > 0:
        rec_budget = (None if time_limit is None
                      else max(0.0, time_limit - rocket.wall_clock_s - sift_time_s
                               - alt_time_s - pair_time_s))
        in_rank = np.argsort(np.argsort(base_positions, kind="stable"),
                             kind="stable").astype(np.int64)
        out_rank, rec_log = reclaim_arcs_fast(
            g, in_rank, rounds=rounds, budget=_RECLAIM_BUDGET,
            conflict_budget=_CONFLICT_BUDGET, time_budget_s=rec_budget)
        w_reclaimed = float(sum(e.get("w_accepted", 0.0) for e in rec_log))
        cand_score = score_from_order(out_rank, np.asarray(g.src, dtype=np.int64),
                                      np.asarray(g.tgt, dtype=np.int64), g.weight)
        lemma_holds = bool(float(cand_score) + 1e-9 >= float(base_score) + w_reclaimed)
        if lemma_holds and cand_score > rec_score:
            rec_score, rec_positions = cand_score, out_rank.astype(np.float32)
    rec_time_s = time.time() - t_rec0

    best_score, best_positions = rec_score, rec_positions
    best_pct = pct(best_score, total)

    # -- Provenance: each stage's increment stays separately auditable ------------
    hist = rocket.history
    hist.attrs["construction"] = construction
    hist.attrs["construction_init_score"] = float(init_score)
    hist.attrs["construction_init_pct"] = pct(float(init_score), total)
    hist.attrs["surrogate"] = "asym_one_sided"
    hist.attrs["surrogate_M"] = M
    hist.attrs["surrogate_T"] = T
    hist.attrs["surrogate_exercised"] = bool(rocket.n_epochs_done > 0)
    hist.attrs["pure_best_score"] = pure_best_score
    hist.attrs["pure_best_pct"] = rocket.best_pct
    hist.attrs["epochs_requested"] = _EPOCHS.get(g.name, RocketConfig.epochs)
    hist.attrs["sift_best_score"] = sift_score
    hist.attrs["sift_best_pct"] = pct(sift_score, total)
    hist.attrs["n_sift_sweeps"] = len(sweep_log)
    hist.attrs["sift_sweeps_requested"] = _MAX_SWEEPS.get(g.name, 40)
    hist.attrs["sift_converged"] = bool(sweep_log and sweep_log[-1]["n_movers"] == 0)
    hist.attrs["sift_time_s"] = sift_time_s
    hist.attrs["sift_alpha"] = _ALPHA
    hist.attrs["sift_k_full"] = _K_FULL
    hist.attrs["alt_best_score"] = alt_score
    hist.attrs["alt_best_pct"] = pct(alt_score, total)
    hist.attrs["alt_increment_pp"] = pct(alt_score, total) - pct(sift_score, total)
    hist.attrs["n_alt_cycles"] = len(alt_log)
    hist.attrs["alt_cycles_requested"] = _ALT_CYCLES.get(g.name, 32)
    hist.attrs["alt_time_s"] = alt_time_s
    hist.attrs["alt_min_block"] = _MIN_BLOCK
    hist.attrs["alt_sift_sweeps"] = _ALT_SIFT_SWEEPS.get(g.name, 2)
    hist.attrs["alt_log"] = alt_log
    hist.attrs["sift_log"] = sweep_log
    hist.attrs["pair_max_pops"] = max_pops
    hist.attrs["pair_best_score"] = pair_score
    hist.attrs["pair_best_pct"] = pct(pair_score, total)
    hist.attrs["pair_time_s"] = pair_time_s
    hist.attrs["pair_log"] = pair_log
    hist.attrs["base_best_score"] = base_score
    hist.attrs["base_best_pct"] = pct(base_score, total)
    hist.attrs["reclaim_rounds_requested"] = rounds
    hist.attrs["reclaim_best_score"] = rec_score
    hist.attrs["reclaim_best_pct"] = pct(rec_score, total)
    hist.attrs["reclaim_increment_pp"] = pct(rec_score, total) - pct(base_score, total)
    hist.attrs["reclaim_weight_accepted"] = w_reclaimed
    hist.attrs["reclaim_lemma_holds"] = lemma_holds
    hist.attrs["reclaim_time_s"] = rec_time_s
    hist.attrs["reclaim_log"] = rec_log
    hist.attrs["refined_best_score"] = best_score
    hist.attrs["refined_best_pct"] = best_pct

    return RocketResult(
        best_positions=best_positions,
        best_score=best_score,
        best_pct=best_pct,
        history=hist,
        # UNCHANGED gradient budget vs the champion: stages 3-6 add 0 optimizer steps.
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=(rocket.wall_clock_s + sift_time_s + alt_time_s + pair_time_s
                      + rec_time_s),
    )


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """IDENTITY ANCHOR: the champion stack from the champion's own construction.

    Must reproduce H64 bit-for-bit (connectome 84.25817950936937). If it does not, the arms
    measure the refactor and not the basin, and the whole study is void - the same validity
    check H57's prefix study used.
    """
    return run_with_construction(g, seed, device, time_limit, _CONSTRUCTION)
