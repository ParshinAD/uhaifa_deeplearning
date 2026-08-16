"""Variant H44 — drop the gradient phase on mouse: ``_EPOCHS["mouse"] = 0``.

Hypothesis (Phase 7, queue item H44)
------------------------------------
On mouse the gradient phase is not neutral, it is **harmful**. Stage 3 is the same
function with the same constants in both arms and both reach a true fixed point, yet:

===========================================  ==================
starting order for stage 3                   stage-3 fixed point
===========================================  ==================
Rocket's plateau order (champion H42)        92.90180243040469
greedy-FAS order directly (no Rocket)        93.08288021668459
===========================================  ==================

The champion's final score on mouse is **92.91701410211007**. So a pipeline that simply
does not run Rocket on this graph lands **+0.16587 pp above the champion**, and does it in
0.76 s instead of 4.78 s.

What changes vs H42
-------------------
Exactly ONE constant: ``_EPOCHS["mouse"]`` goes ``5_000 -> 0``. Every other constant —
``_MAX_SWEEPS``, ``_K_FULL``, ``_ALPHA``, ``_ALT_CYCLES``, ``_ALT_SIFT_SWEEPS``,
``_ALT_K_FULL``, ``_MIN_BLOCK`` — and the whole body of :func:`run` are byte-identical to
:mod:`mfas.experiments.H42`. ``tests/test_experiment_H44.py`` asserts that
constant-for-constant rather than trusting it to stay true.

**Consequence worth stating up front: on connectome and microns this variant is the
champion.** ``_EPOCHS.get(g.name)`` returns 20,000 and 80,000 exactly as before, so H44 and
H42 execute the identical code path with identical constants there. The screen still runs
both primaries — a claim of identity is checked, not asserted.

Why this is a sizing change and not a dataset-specific hack
-----------------------------------------------------------
``_EPOCHS`` is already a per-dataset dict and always has been; H42 itself shipped by
re-sizing two other per-dataset dicts. The value 0 is not hand-picked to fit mouse — it is
what the epoch grid returns as the argmax on this graph, measured the same way the grid
measured connectome's 20,000 (which it confirmed is already optimal) and microns' knee.
The pipeline, the move classes and the scorer are unchanged.

Evidence
--------
* ``experiments/outputs/proto_P07_mouse.json`` — the settling run, whose two arms were
  **pre-registered before it was launched** and both hit exactly:
  ``epochs=0 -> 93.08288021668459`` and ``epochs=5000 -> 92.91701410211007``. The second
  arm is a parity anchor: the harness reproduces the production champion bit-for-bit.
* ``experiments/outputs/proto_H41.json`` (``mouse@10s/_setup.stage3_pct``) — where the
  93.08288021668459 figure was first produced, and then dismissed in one clause.
* ``experiments/outputs/proto_H42_mouse.json`` — nine stage-4 allocations from (8,8) to
  (1,128), all returning exactly 92.91701410211007. The gain is NOT a cycle-budget effect.
* ``dr_tmp/size_global_discrete.json`` — independent 3-seed corroboration eight weeks
  older (93.047 / 93.012 / 93.004 with a different sift).
* ``dr_tmp/investigation_rocket_free_basin.md`` — the full file-by-file trail.

Honest caveats
--------------
* **This does not generalise, and the module does not pretend it does.** The epochs=0 arms
  of the sizing grid go the other way on both primaries: microns 83.11151992749353 vs a
  champion 83.24085291200831, connectome 83.87974978419737 vs 84.15409511053134. The sign
  of the gradient phase's contribution flips with graph size (mouse is n=148).
* The mechanism is ``greedy start x UNDER-RELAXATION`` specifically, not "no Rocket" alone:
  ``experiments/outputs/proto_h32_trophic.json``'s plain-Jacobi ``greedy_sift`` control
  reaches only 92.91630, which does **not** beat the champion.
* mouse is the campaign's **non-inferiority** dataset, not a primary. A +0.16587 pp move
  there does not advance the mission target; it is a real gain on a real dataset and it
  should be reported as exactly that, with the connectome number unchanged.
* Related but separate: the connectome grid showed the refinement stack is NOT monotone in
  the quality of its input (20k -> 40k epochs improves pure by +0.13560 pp and stage 3 by
  +0.03314 pp while the final score falls 0.00666 pp). H44 is the extreme case of the same
  phenomenon. See ``experiments/log.md`` 2026-08-16.

Leakage-safety
--------------
Unchanged from H42. Nothing reads the target metric; the frozen oracle only scores whole
candidate vectors, and ``data/best_solution`` is never read. With ``epochs = 0`` on mouse
the pipeline draws from ``seed`` even less than before, so the determinism the campaign's
seed policy rests on is strengthened, not weakened.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct
from ..refine import alternate_scc_sift, sift_underrelaxed
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H44"
HYPOTHESIS = (
    "On mouse the gradient phase is HARMFUL, not neutral: the under-relaxed sift run from "
    "the greedy-FAS order reaches the fixed point 93.08288021668459, while the same "
    "function with the same constants run from Rocket's order reaches 92.90180243040469. "
    "Setting _EPOCHS['mouse'] = 0 therefore scores +0.16587 pp above the 92.91701410211007 "
    "champion at ~1/6 of its wall clock, with 0 gradient steps and every other constant "
    "unchanged. On connectome and microns the variant IS the champion, byte-for-byte."
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
    hist.attrs["refined_best_score"] = best_score
    hist.attrs["refined_best_pct"] = best_pct
    hist.attrs["sift_log"] = sweep_log

    return RocketResult(
        best_positions=best_positions,
        best_score=best_score,
        best_pct=best_pct,
        history=hist,
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=rocket.wall_clock_s + sift_time_s + alt_time_s,
    )
