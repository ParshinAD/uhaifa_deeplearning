"""Variant H42 — re-allocate stage 4's wall-clock: many cheap cycles, not few expensive ones.

Hypothesis (Phase 7): H36's stage-4 constants were sized when the CYCLE's budget, not the
ALGORITHM's, was binding, and the inner sift was never itself sized. A stage-4 cycle is
(recursive SCC block refine) + (short under-relaxed sift), and the sift is ~88% of the
cost (connectome 23.0 s vs 3.0 s). So at a FIXED stage-4 wall-clock the shipped split buys
few expensive cycles where it could buy many cheap ones. Cutting the inner sift from 8 (4 on
microns) to 2 sweeps and spending the freed seconds on more alternations reaches a strictly
better order **at the same or lower wall-clock on both primaries**.

What changes vs H36
-------------------
Exactly two constants: ``_ALT_SIFT_SWEEPS`` and ``_ALT_CYCLES``. Stages 1-3 and every other
constant are byte-identical to :mod:`mfas.experiments.H36`, so H42 - H36 isolates the
stage-4 ALLOCATION and nothing else. No new mechanism, no new code path, 0 extra gradient
steps — ``alternate_scc_sift`` is the production refiner H36 already ships.

Why this is a mechanism claim and not "more compute"
---------------------------------------------------
The prototype ran the three connectome arms at an IDENTICAL 1200 s budget
(``experiments/outputs/proto_H42.json``), which is the matched-wall-clock control:

    sweeps=8 -> 47 cycles -> 84.133292      (the shipped split)
    sweeps=4 -> 85 cycles -> 84.135287
    sweeps=2 -> 143 cycles -> 84.158803     (+0.0255 pp over the shipped split, same seconds)

Stronger still, at the wall-clock H36 ACTUALLY SPENDS today the re-allocation already wins
on both primaries while costing slightly FEWER seconds — so the gain cannot be attributed to
extra CPU in any accounting:

    connectome  317.7 s shipped -> 84.097174   |  sweeps=2, 37 cyc, 309.7 s -> 84.140028
    microns      92.8 s shipped -> 83.231977   |  sweeps=2,  5 cyc,  87.6 s -> 83.239736

Sizing (Q1), and why the shipped counts are what they are
---------------------------------------------------------
* **connectome** — the curve saturates: increments fall to ~+0.00002 pp/cycle by cycle 140.
  Ships **77 cycles** (~646 s of stage 4, ~1168 s per run). Running to the measured end of
  the curve (143 cycles, ~1722 s per run) would add only **+0.0047 pp** — *less than the
  dataset's own 0.012 pp minimum effect size* — so the extra 555 s/run buys nothing the
  campaign is willing to call a difference. That is the stopping rule, not a budget excuse.
* **microns** — ships **5 cycles** (~87.6 s), which is *less* stage-4 wall-clock than H36's
  3 cycles spend (92.8 s), so the run does not get longer: ~3309 s vs the champion's 3314 s.
  The curve is still rising here (10 cycles -> +0.0146 pp) but 10 cycles puts the run at
  ~3396 s, i.e. flush against the 3400 s warn band. Queue item **P05** was open precisely
  because microns has ~8% margin and stage 4 had no wall-clock guard; spending that margin
  would make the runtime invariant depend on the machine being idle. Deliberately left on
  the table — see log.md.

  **P05 (2026-08-11) found that margin is already gone.** The guard now exists
  (``eval/runtime_guard.py``), but the same cycle measured this machine running ~20% slower
  under ordinary desktop load than it did when H42 was confirmed overnight: connectome
  1231.7 s -> 1483.9 s for a **bit-identical** result. Scaled to microns that is ~4100 s
  against a 3600 s cap. The microns configuration below therefore does not fit the runtime
  invariant on a loaded machine. The guard turns that from a silent cap overrun into a
  flagged, truncated run; it cannot make the configuration fit. See queue item **P07**.
* **mouse** — measured exhaustively over a 9-point (sweeps, cycles) grid
  (``experiments/outputs/proto_H42_mouse.json``): **every** arm from (8, 8) to (1, 128)
  returns exactly 92.917014, the champion value. Mouse is saturated and cannot distinguish
  allocations, so it remains a valid non-inferiority tripwire and is set by the same rule.

Determinism
-----------
Stage 4 is sized by CYCLE COUNT, never by a wall-clock budget. A time-sized stage 4 would
make the cycle count depend on machine load, which would destroy the bit-reproducibility
that ``sota.json`` (std = 0) and the P02 one-seed screen policy both rest on. The P05 guard
keeps that distinction: it is an ABORT at a stage boundary, not a sizing rule, so on a
machine fast enough to finish the configured work it changes nothing (verified bit-identical
on connectome and mouse, 2026-08-11). When it does fire, the run is marked ``degraded`` in
``results/*.json`` and ``audit.py`` FAILs it rather than letting it be pooled into a mean.

Leakage-safety
--------------
Unchanged from H36: every move is decided from the input edge weights and the current ranks
alone, the frozen oracle only accepts/rejects whole candidate vectors, and
``data/best_solution`` is never read.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct
from ..refine import alternate_scc_sift, sift_underrelaxed
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H42"
HYPOTHESIS = (
    "H36's stage-4 inner sift was never sized: it costs ~88% of a cycle, so the shipped "
    "split buys few expensive cycles where it could buy many cheap ones. Cutting the inner "
    "sift to 2 sweeps and spending the freed seconds on more alternations beats the "
    "champion on both primaries at the SAME OR LOWER wall-clock and 0 extra gradient steps."
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
    """Run the H36 pipeline with a re-allocated stage-4 budget.

    Identical to :func:`mfas.experiments.H36.run` except for the two stage-4 constants
    above. Returns the FINAL best (max of pure Rocket, the H35 sift, and the alternation);
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
    alt_rank, alt_score, alt_log = alternate_scc_sift(
        g, sift_rank,
        n_cycles=_ALT_CYCLES.get(g.name, 32),
        sift_sweeps=_ALT_SIFT_SWEEPS.get(g.name, 2),
        k_full=_ALT_K_FULL, alpha=_ALPHA, min_block=_MIN_BLOCK,
        time_budget_s=alt_budget)
    alt_time_s = float(alt_log[-1]["cum_wall_s"]) if alt_log else 0.0

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
