"""Variant H36 — champion pipeline + recursive SCC-topological BLOCK refinement.

Hypothesis (Phase 7): the residual gap to the 84.6147% reference lives *inside* the
giant SCC and is a **joint** rearrangement — a set of nodes that only pays off when
moved together. Every refiner in the campaign so far moves ONE node at a time and is
blind to such a move by construction. Adding a structural BLOCK move class —
decompose a contiguous block into strongly connected components, lay them out
topologically, recurse — and alternating it with the champion's single-node sift
reaches a joint fixed point that neither move class reaches alone.

Pipeline
--------
1. Greedy-FAS warm start (H02) -> evenly-spaced positions.
2. UNCHANGED ``run_rocket`` at the baseline epoch budget -> PURE Rocket best,
   tracked and reported separately per CLAUDE.md. Identical to H02/H30/H35 stage 1+2.
3. Under-relaxed two-phase exact-gain sift (H35's stage 3, unchanged).
4. **NEW:** ``n_cycles`` alternations of (recursive SCC block refine -> a short
   under-relaxed sift), best-by-oracle throughout
   (:func:`mfas.refine.alternate_scc_sift`).

Stages 1-3 are exactly H35, so H36 - H35 isolates stage 4. H35 is used as the base on
every dataset — one algorithm, no per-dataset pipeline branch — even though the
microns champion is H30. On microns H35 scores 0.0019 pp *below* H30, so crediting
stage 4 with (H36 - H30 champion) UNDER-states its own increment there. That is the
conservative direction and avoids double-counting (CAMPAIGN.md § How not to fool
yourself).

Why this is not H22 / M4
------------------------
M4 killed *bounded rank-window* neighbourhoods: the recoverable weight is long-range,
so a window recovers <= 0. Its stated revival condition is a **structural**
neighbourhood (SCC / block) instead of a rank window, and names H36. The block here
is defined by strong connectivity, not by rank distance, and the recursion re-enters
the giant SCC that a one-shot condensation cannot touch (E2: one-shot gains
+0.00013 pp).

Compute accounting / leakage-safety
-----------------------------------
Stage 4 adds **0 gradient steps**, so ``n_epochs_done`` stays the baseline gradient
budget and ``budget_basis = total_grad_steps`` comparisons remain matched — the same
accounting under which H30's and H35's sifts were promoted. The added non-gradient
wall-clock is recorded in ``history.attrs`` and folded into ``wall_clock_s``, and the
per-dataset cycle counts are sized so every run stays inside the 3600 s hard budget.
Because that extra wall-clock is real, the honest control was measured before
building this: continuing the champion's OWN move class for a matched 342 s gains
+0.0051 pp, against +0.192 pp for the block move class
(``experiments/outputs/proto_H36.json``).

Every move is decided from the input edge weights and the current ranks alone; the
frozen oracle only accepts/rejects whole candidate vectors, and ``data/best_solution``
is never read. ``best_positions`` is the final best rank vector cast to float32
(exact at ``n < 2**24`` for all three datasets), so the runner's re-score assertion
holds.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct
from ..refine import alternate_scc_sift, sift_underrelaxed
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H36"
HYPOTHESIS = (
    "The residual gap lives inside the giant SCC as a JOINT rearrangement that no "
    "single-node move can reach. Adding a structural block move class (recursive "
    "SCC-topological reordering of contiguous blocks) and alternating it with the "
    "champion's exact-gain sift reaches a joint fixed point neither reaches alone, "
    "beating the champion on both primaries at 0 extra gradient steps."
)

# Stages 1-3: identical to H35 (the connectome champion).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12}
_K_FULL = 6
_ALPHA = 0.7

# Stage 4 budget, per dataset. Sized from the measured cost/benefit curve in
# experiments/outputs/proto_H36.json so that every run stays under the 3600 s hard
# cap AND the 3400 s warn band:
#   connectome  ~26 s/cycle on top of a 591 s champion run  -> 12 cycles ~ 931 s total
#   microns     ~32 s/cycle on top of a 3240 s champion run -> 3 cycles ~ 3335 s total
#               (microns is the binding constraint, hence its cheaper inner sift)
#   mouse       milliseconds; the count is irrelevant to runtime
# The connectome curve is still rising at 31 cycles (+0.214 pp) — 12 is the knee of
# the cost/benefit curve, not a converged optimum. Sizing it up is queue item H42.
_ALT_CYCLES = {"connectome": 12, "microns": 3, "mouse": 8}
_ALT_SIFT_SWEEPS = {"connectome": 8, "microns": 4, "mouse": 8}
_ALT_K_FULL = 2          # short warm phase: the incoming order is already refined
_MIN_BLOCK = 32          # below this the scipy call costs more than it returns


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run the H35 champion pipeline, then the alternating block/node refinement.

    Returns the FINAL best (max of pure Rocket, the H35 sift, and the alternation)
    in ``RocketResult``; the pure-Rocket and post-sift bests are reported separately
    via ``history.attrs`` so each stage's increment stays separable.
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

    # ── Stage 4 (NEW): alternate block refinement with the single-node sift ──────
    alt_budget = (None if time_limit is None
                  else max(0.0, time_limit - rocket.wall_clock_s - sift_time_s))
    alt_rank, alt_score, alt_log = alternate_scc_sift(
        g, sift_rank,
        n_cycles=_ALT_CYCLES.get(g.name, 8),
        sift_sweeps=_ALT_SIFT_SWEEPS.get(g.name, 8),
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
    hist.attrs["sift_time_s"] = sift_time_s
    hist.attrs["sift_alpha"] = _ALPHA
    hist.attrs["sift_k_full"] = _K_FULL
    hist.attrs["alt_best_score"] = alt_score
    hist.attrs["alt_best_pct"] = pct(alt_score, total)
    hist.attrs["alt_increment_pp"] = pct(alt_score, total) - pct(sift_score, total)
    hist.attrs["n_alt_cycles"] = len(alt_log)
    hist.attrs["alt_time_s"] = alt_time_s
    hist.attrs["alt_min_block"] = _MIN_BLOCK
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
