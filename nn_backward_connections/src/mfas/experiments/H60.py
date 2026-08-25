"""Variant H60 — condense the NET digraph inside stage 4, not the RAW digraph.

Hypothesis (Phase 7, divergent mode)
------------------------------------
``SccRecursiveRefiner`` builds its condensation matrix over the RAW arc set, every edge
counting the same (``scc_recursive.py``: ``np.ones(eidx.size)``). So a reciprocal pair
``u <-> v`` is an unbreakable 2-cycle — ``u`` and ``v`` land in one SCC and the refiner
preserves each SCC's internal order by construction — even when ``w_uv = 100`` and
``w_vu = 1``. A pair whose orientation is worth 99 units is welded shut by an edge worth 1.

Maximising feedforward weight on ``W`` is identical to maximising it on the NET digraph
``D+ = {(u, v) : w_uv > w_vu}`` with ``d_uv = w_uv - w_vu``, up to the order-independent
constant ``sum over pairs of min(w_uv, w_vu)``. ``D+``'s arcs are a subset of ``G``'s, so
its SCCs STRICTLY REFINE ``G``'s, and the refiner stays exactly monotone on it. The proof
and the leakage argument live in :mod:`mfas.refine.net_condense`.

What changes vs H42
-------------------
Exactly one argument: stage 4 passes ``g_struct=net_structure_graph(g)``. Stages 1-3, all
four stage-4 constants and every cycle count are byte-identical to
:mod:`mfas.experiments.H42`, so H60 - H42 isolates the STRUCTURE GRAPH and nothing else.
No new move class, no extra cycles, 0 extra gradient steps.

Why this is a mechanism claim and not "more compute"
----------------------------------------------------
The cycle count is unchanged, and the net refiner is if anything CHEAPER per cycle than the
raw one — it condenses fewer arcs. The composed prototype ran both arms from each dataset's
own stored champion order at matched cycles and matched constants
(``experiments/outputs/proto_H60_alt_*.json``); connectome::

    arm raw (= the champion's own stage 4, 40 more cycles)  +0.005039 pp in 315.0 s
    arm net (D+ substituted, nothing else changed)          +0.032516 pp in 301.4 s
                                                            ------------------------
    H60 composed delta                                      +0.027477 pp, -13.6 s

The redundancy fraction M8 demands is **negative** (connectome -0.9986, mouse -0.8808):
the single-node sift gains MORE alongside the net refiner, not less, so the two classes are
complementary rather than competing for the same ground. That is the opposite of H41's
failure mode (91.9% redundancy).

Why the standalone refiner number is not the gate
-------------------------------------------------
Run alone from the champion order the net refiner returns only +0.005793 pp on connectome
(``proto_H60_connectome.json``), below the 0.012 pp minimum effect size. H36's own history
says that number is not dispositive: the one-shot top-level condensation measured
**+0.00013 pp** while the same mechanism ALTERNATED with the sift became the campaign's
largest score move, **+0.1837 pp**. The two classes re-open each other's exhausted moves.
The gate here is therefore the composed delta, which is what M8 requires anyway.

Determinism and runtime
-----------------------
``build_net_arcs`` is two ``np.unique`` calls and two ``bincount``s over the edge arrays —
deterministic, no RNG, and it consumes nothing from ``seed``. Stage 4 remains sized by CYCLE
COUNT, so the P05 guard and the bit-reproducibility that ``sota.json`` rests on are
unaffected. P07 is respected: the microns cycle count is H42's, unchanged.

Leakage-safety
--------------
Unchanged from H42, and ``D+`` adds no exposure: it is a function of the input edge arrays
alone — no ordering, no score and no oracle call enters its construction, and
``data/best_solution`` is never read.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct
from ..refine import alternate_scc_sift, sift_underrelaxed
from ..refine.net_condense import net_structure_graph
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H60"
HYPOTHESIS = (
    "The block refiner condenses the RAW digraph, so every reciprocal pair is an "
    "unbreakable 2-cycle regardless of how lopsided its weights are. Condensing the NET "
    "digraph instead strictly refines the components, stays exactly monotone, costs no "
    "extra cycles, and beats the champion on both primaries."
)

# ── Stages 1-3: byte-identical to H36 (which is byte-identical to H35) ───────────────
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12}
_K_FULL = 6
_ALPHA = 0.7

# ── Stage 4: THE ONLY CHANGE. Sized from experiments/outputs/proto_H42{,_mouse}.json ──
# H36 shipped {connectome: 12, microns: 3, mouse: 8} cycles at {8, 4, 8} sweeps.
_ALT_CYCLES = {"connectome": 77, "microns": 5, "mouse": 32}
_ALT_SIFT_SWEEPS = {"connectome": 2, "microns": 2, "mouse": 2}
_ALT_K_FULL = 2          # short warm phase: the incoming order is already refined
_MIN_BLOCK = 32          # below this the scipy call costs more than it returns


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run the H42 pipeline with the NET digraph as stage 4's structure graph.

    Identical to :func:`mfas.experiments.H42.run` except for the ``g_struct`` argument
    to :func:`~mfas.refine.alternate_scc_sift`. Returns the FINAL best (max of pure Rocket, the H35 sift, and the alternation);
    the pure-Rocket and post-sift bests are reported separately via ``history.attrs`` so
    each stage's increment stays separable.
    """
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))

    # ── Stage 1+2: H02 warm start -> unchanged Rocket (PURE) ─────────────────────
    order = greedy_fas_order(g)
    init_positions = _init_positions_from_order(order, device)
    rocket = run_rocket(g, cfg, seed=seed, device=device,
                        init_positions=init_positions, time_limit=time_limit)

    pure_best_positions = rocket.best_positions
    pure_best_score = rocket.best_score
    pure_best_pct = rocket.best_pct
    total = g.total_weight

    # ── Stage 3: H35's under-relaxed two-phase exact-gain sift ───────────────────
    rank0 = np.argsort(np.argsort(pure_best_positions, kind="stable"),
                       kind="stable").astype(np.int64)
    sift_budget = None if time_limit is None else max(0.0, time_limit - rocket.wall_clock_s)
    sift_rank, sift_score, sweep_log = sift_underrelaxed(
        g, rank0, k_full=_K_FULL, alpha=_ALPHA,
        max_sweeps=_MAX_SWEEPS.get(g.name, 40), time_budget_s=sift_budget)
    sift_time_s = float(sum(row["wall"] for row in sweep_log))

    # ── Stage 4: alternate block refinement with a SHORT single-node sift ────────
    alt_budget = (None if time_limit is None
                  else max(0.0, time_limit - rocket.wall_clock_s - sift_time_s))
    # THE ONLY CHANGE vs H42: the refiner condenses D+, the sift and every score still
    # read g. Built once per run, outside the cycle loop.
    t_net = time.time()
    g_struct = net_structure_graph(g)
    net_build_s = time.time() - t_net
    alt_budget = None if alt_budget is None else max(0.0, alt_budget - net_build_s)
    alt_rank, alt_score, alt_log = alternate_scc_sift(
        g, sift_rank,
        n_cycles=_ALT_CYCLES.get(g.name, 32),
        sift_sweeps=_ALT_SIFT_SWEEPS.get(g.name, 2),
        k_full=_ALT_K_FULL, alpha=_ALPHA, min_block=_MIN_BLOCK,
        time_budget_s=alt_budget, g_struct=g_struct)
    alt_time_s = net_build_s + (float(alt_log[-1]["cum_wall_s"]) if alt_log else 0.0)

    # ── Best-by-oracle across every stage ────────────────────────────────────────
    best_score = pure_best_score
    best_positions = pure_best_positions
    if sift_score > best_score:
        best_score, best_positions = sift_score, sift_rank.astype(np.float32)
    if alt_score > best_score:
        best_score, best_positions = alt_score, alt_rank.astype(np.float32)
    best_pct = pct(best_score, total)

    # ── Provenance: each stage's increment stays separately auditable ────────────
    hist = rocket.history
    hist.attrs["pure_best_score"] = pure_best_score
    hist.attrs["pure_best_pct"] = pure_best_pct
    hist.attrs["sift_best_score"] = sift_score
    hist.attrs["sift_best_pct"] = pct(sift_score, total)
    hist.attrs["n_sift_sweeps"] = len(sweep_log)
    # P05: how many were ASKED for, and whether a short log means convergence rather than
    # the wall-clock guard cutting the stage off. Provenance only — nothing below reads it.
    hist.attrs["sift_sweeps_requested"] = _MAX_SWEEPS.get(g.name, 40)
    hist.attrs["sift_converged"] = bool(sweep_log and sweep_log[-1]["n_movers"] == 0)
    hist.attrs["sift_time_s"] = sift_time_s
    hist.attrs["sift_alpha"] = _ALPHA
    hist.attrs["sift_k_full"] = _K_FULL
    hist.attrs["alt_best_score"] = alt_score
    hist.attrs["alt_best_pct"] = pct(alt_score, total)
    hist.attrs["alt_increment_pp"] = pct(alt_score, total) - pct(sift_score, total)
    hist.attrs["n_alt_cycles"] = len(alt_log)
    # P05: alternate_scc_sift's loop breaks on the time budget and nothing else, so
    # n_alt_cycles < alt_cycles_requested is an EXACT signal that the guard truncated it.
    hist.attrs["alt_cycles_requested"] = _ALT_CYCLES.get(g.name, 32)
    hist.attrs["alt_time_s"] = alt_time_s
    hist.attrs["alt_min_block"] = _MIN_BLOCK
    hist.attrs["alt_sift_sweeps"] = _ALT_SIFT_SWEEPS.get(g.name, 2)
    hist.attrs["alt_log"] = alt_log
    # H60 provenance: the structure graph actually condensed.
    hist.attrs["struct_graph"] = "net"
    hist.attrs["net_build_s"] = net_build_s
    hist.attrs["net_n_arcs"] = int(g_struct.src.size)
    hist.attrs["net_const_c"] = float(g_struct.const_c)
    hist.attrs["refined_best_score"] = best_score
    hist.attrs["refined_best_pct"] = best_pct
    hist.attrs["sift_log"] = sweep_log

    return RocketResult(
        best_positions=best_positions,
        best_score=best_score,
        best_pct=best_pct,
        history=hist,
        # UNCHANGED gradient budget: stages 3 and 4 add 0 optimizer steps.
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=rocket.wall_clock_s + sift_time_s + alt_time_s,
    )
