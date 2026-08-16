"""Variant H52 — add SEQUENTIAL exact-gain pair relocation as stage 5.

Hypothesis (Phase 7, queue item H51)
------------------------------------
The champion's two refinement move classes are both blind to the same thing: a pair of nodes
that only improves JOINTLY. Single-node insertion gives each node its own argmax with everything
else held fixed, and SCC block refinement only relocates groups that fall out of a strong
decomposition. Measured on the champion's own connectome order, 3,498 backward-edge pairs have a
strictly positive exact gain and 989 of them (28.3 %) have ZERO solo gain at both endpoints —
invisible by construction. The best single move spans 92,958 positions, 68 % of the line.

What changes vs the champion
----------------------------
Exactly ONE thing: a fifth stage, :func:`mfas.refine.pair_relocate.pair_relocate`, after stage 4.
Every existing constant is byte-identical to :mod:`mfas.experiments.H44` (the current mouse
champion, itself byte-identical to H42 on both primaries), and ``tests/test_experiment_H52.py``
asserts that constant-for-constant rather than trusting it to stay true.

Why SEQUENTIAL is the whole hypothesis
---------------------------------------
The same move class, applied the way every other refiner in this package applies moves — compute
all gains, take a maximal DISJOINT batch — delivers +0.00193 pp on this order (H45). Applied one
move at a time against the updated order it delivers +0.02233 pp, 11.6x more. The reason is
measured, not argued: a pair move's interval spans the distance between the backward edge's
endpoints (mean ~20,536 positions), so disjoint intervals barely fit on a 136,648-position line
and the batched form discards 99.9 % of the improving moves it finds (H50: 4,018,567 candidates
found, 4,193 applied).

Measured on the champions' stored orders (experiments/outputs/proto_H51_*_champion.json)
-----------------------------------------------------------------------------------------

===========  ==========  ==========  ============  ==========
dataset      champion    converged   delta         threshold
===========  ==========  ==========  ============  ==========
connectome   84.154095   84.176420   +0.02233      0.012
microns      83.240853   83.251314   +0.01046      0.002
mouse        93.082880   93.102826   +0.01995      0.01
===========  ==========  ==========  ============  ==========

All three clear their thresholds ON THE STORED ORDER. That is NOT a screen: those numbers come
from post-hoc refinement of a saved position vector, and this module is what turns them into an
end-to-end claim that the gate ladder can adjudicate.

The runtime problem, stated before the screen rather than after it
-------------------------------------------------------------------
This stage is not free and microns has no room. The champion's microns run is 3,258 s against the
runtime guard's 3,450 s deadline, so ~190 s is available; a full pass there costs 984 s. The pop
budget caps microns at 350 k pops (~165 s), which by the measured curve buys roughly
+0.002-0.003 pp — right at its 0.002 pp threshold. A microns failure here is P07 reasserting
itself as a hard budget, NOT a defect of the move class, and it must be reported that way.

connectome has room: 2 passes cost 644 s on a 1,152 s run, i.e. 1,796 s of 3,450 s.

Determinism
-----------
The refiner draws no random numbers and its budget is a POP COUNT, never a wall-clock. A
time-sized budget would make results machine-dependent and break the bit-reproducibility that
sota.json's std = 0 and the campaign's 1-seed screen policy rest on. ``time_budget_s`` is threaded
in only so the run-level guard can ABORT the stage at a pass boundary, exactly as P05 specified
for every other stage.

Leakage-safety
--------------
Unchanged. Stage 5 chooses moves from the input edge weights and the current ranks alone, via a
closed-form gain proved in the refiner's module docstring and pinned against the frozen oracle by
``tests/test_refine_pair_relocate.py`` — including a brute-force check over EVERY split, which is
the assertion that separates "the formula is right" from "the scan is right".
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct
from ..refine import alternate_scc_sift, sift_underrelaxed
from ..refine.pair_relocate import pair_relocate
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H52"
HYPOTHESIS = (
    "The champion's two refinement move classes are both blind to a pair of nodes that only "
    "improves JOINTLY: 989 of 3,498 positive-gain backward-edge pairs on its own connectome "
    "order have ZERO solo gain at both endpoints. Adding sequential exact-gain pair relocation "
    "as stage 5 reaches a strictly better order. The word SEQUENTIAL is the hypothesis: the "
    "identical move class applied as a disjoint batch gives +0.00193 pp, applied one move at a "
    "time it gives +0.02233 pp, because disjoint intervals of ~20,536 positions barely fit on a "
    "136,648-position line and the batched form discards 99.9% of what it finds."
)

# ── Stages 1-3: identical to H42 EXCEPT the mouse epoch count ───────────────────────
# THE ONLY CHANGE IN THIS FILE is mouse 5_000 -> 0, sized by the epoch grid
# (experiments/outputs/proto_P07_mouse.json). connectome and microns are untouched, so
# H44 - H42 isolates the mouse gradient phase and nothing else.
_EPOCHS = {"connectome": 20_000, "mouse": 0, "microns": 80_000}
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12}
_K_FULL = 6
_ALPHA = 0.7

# ── Stage 4: byte-identical to H42 ──────────────────────────────────────────────────
_ALT_CYCLES = {"connectome": 77, "microns": 5, "mouse": 32}
_ALT_SIFT_SWEEPS = {"connectome": 2, "microns": 2, "mouse": 2}
_ALT_K_FULL = 2          # short warm phase: the incoming order is already refined
_MIN_BLOCK = 32          # below this the scipy call costs more than it returns

# -- Stage 5: THE ONLY ADDITION. Sequential exact-gain pair relocation. --------------
# Sized by POP COUNT, never by wall-clock: a time-sized budget would make the result
# machine-dependent and break the bit-reproducibility that sota.json's std = 0 and the
# 1-seed screen policy rest on (the P05 lesson).
#
# connectome: 2 passes drain ~2.4 M pops in 644 s and deliver 93.7% of the converged
#             gain (+0.02093 of +0.02233 pp) - proto_H51_connectome_champion.json.
# microns:    the binding constraint. The champion already runs 3,258 s against a
#             3,450 s guard deadline, so only ~190 s is available; at the measured
#             ~2,100 pops/s that is ~350 k pops. This is P07 as a hard budget.
# mouse:      converges after one move; the budget is never reached.
_PAIR_PASSES = 2
_PAIR_MAX_POPS = {"connectome": 2_400_000, "microns": 350_000, "mouse": 100_000}


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run the H42 pipeline with the mouse gradient phase sized to zero.

    Byte-identical to :func:`mfas.experiments.H42.run`; only ``_EPOCHS["mouse"]`` differs.
    With ``epochs = 0`` ``run_rocket`` returns the warm-start order unchanged, so stage 3
    starts from the greedy-FAS order and ``pure_best_pct`` reports the greedy score.
    Returns the FINAL best (max of pure, the H35 sift, and the alternation).
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
    alt_rank, alt_score, alt_log = alternate_scc_sift(
        g, sift_rank,
        n_cycles=_ALT_CYCLES.get(g.name, 32),
        sift_sweeps=_ALT_SIFT_SWEEPS.get(g.name, 2),
        k_full=_ALT_K_FULL, alpha=_ALPHA, min_block=_MIN_BLOCK,
        time_budget_s=alt_budget)
    alt_time_s = float(alt_log[-1]["cum_wall_s"]) if alt_log else 0.0

    # ── Stage 5: sequential exact-gain pair relocation — THE ONLY ADDITION ───────
    pair_budget = (None if time_limit is None else
                   max(0.0, time_limit - rocket.wall_clock_s - sift_time_s - alt_time_s))
    pair_rank, pair_score, pair_log = pair_relocate(
        g, alt_rank, n_passes=_PAIR_PASSES,
        max_pops=_PAIR_MAX_POPS.get(g.name), time_budget_s=pair_budget)
    pair_time_s = float(pair_log[-1]["cum_wall_s"]) if pair_log else 0.0

    # ── Best-by-oracle across every stage ────────────────────────────────────────
    best_score = pure_best_score
    best_positions = pure_best_positions
    if sift_score > best_score:
        best_score, best_positions = sift_score, sift_rank.astype(np.float32)
    if alt_score > best_score:
        best_score, best_positions = alt_score, alt_rank.astype(np.float32)
    if pair_score > best_score:
        best_score, best_positions = pair_score, pair_rank.astype(np.float32)
    best_pct = pct(best_score, total)

    # ── Provenance: each stage's increment stays separately auditable ────────────
    hist = rocket.history
    hist.attrs["pure_best_score"] = pure_best_score
    hist.attrs["pure_best_pct"] = pure_best_pct
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
    # H44-specific: makes "did the gradient phase run at all?" auditable per dataset.
    hist.attrs["epochs_requested"] = _EPOCHS.get(g.name, RocketConfig.epochs)
    # H52-specific: stage 5 credit stays separately attributable, and the pop budget is
    # recorded so "was this stage truncated by its budget?" is answerable from the record.
    hist.attrs["pair_best_score"] = pair_score
    hist.attrs["pair_best_pct"] = pct(pair_score, total)
    hist.attrs["pair_increment_pp"] = pct(pair_score, total) - pct(alt_score, total)
    hist.attrs["pair_passes_requested"] = _PAIR_PASSES
    hist.attrs["n_pair_passes"] = len(pair_log)
    hist.attrs["n_pair_moves"] = int(sum(r["n_applied"] for r in pair_log))
    hist.attrs["n_pair_pops"] = int(pair_log[-1]["n_popped"]) if pair_log else 0
    hist.attrs["pair_max_pops"] = _PAIR_MAX_POPS.get(g.name)
    hist.attrs["pair_time_s"] = pair_time_s
    hist.attrs["pair_log"] = pair_log
    hist.attrs["refined_best_score"] = best_score
    hist.attrs["refined_best_pct"] = best_pct
    hist.attrs["sift_log"] = sweep_log

    return RocketResult(
        best_positions=best_positions,
        best_score=best_score,
        best_pct=best_pct,
        history=hist,
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=rocket.wall_clock_s + sift_time_s + alt_time_s + pair_time_s,
    )
