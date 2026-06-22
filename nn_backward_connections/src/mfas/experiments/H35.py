"""Variant H35 — H02-warm-started Rocket + under-relaxed two-phase sift (Phase 6.2).

Hypothesis (Phase 6.2): H30's production sift is a *Jacobi* iteration (every node moves
FULLY to its exact-optimal gap each sweep) that does not converge on large dense
connectomes — it enters a period-2 limit cycle, so best-by-oracle only creeps up via the
oscillation's lucky phase. Replacing the rebuild with an **under-relaxed** step (move each
node only a fraction ``alpha`` of the way to its exact-optimal gap) breaks the cycle, the
iterate converges, and the refined order strictly beats H30's Jacobi sift at the SAME
gradient budget (the sift adds 0 optimizer steps). A two-phase schedule (``alpha=1`` for
``k_full`` warm sweeps, then ``alpha``) plus best-by-oracle makes it non-regressing on
graphs where Jacobi already converges (mouse).

Pipeline
--------
1. Greedy-FAS warm start (H02): ``greedy_fas_order(g)`` -> evenly-spaced positions.
2. UNCHANGED ``run_rocket`` at the baseline epoch budget -> PURE Rocket best (tracked and
   reported separately, per CLAUDE.md; identical pipeline to H02/H30 stage 1+2).
3. Under-relaxed two-phase exact-gain sift (:func:`mfas.refine.sift_underrelaxed`) on the
   Rocket order: same brute-force-verified exact-gain kernel as H30, but the Jacobi rebuild
   is under-relaxed after ``k_full`` warm sweeps; the frozen oracle accepts/rejects whole
   candidate vectors only (best-by-oracle).

Distinct from H30
-----------------
H30 uses ``mfas.refine.sift`` (pure Jacobi, ``key = best_gap``). H35 uses
``mfas.refine.sift_underrelaxed`` (``key = rank + alpha*(best_gap - rank)`` after the warm
phase). At ``alpha=1`` the two are bit-identical (asserted in
``tests/test_refine_underrelax.py``); the win comes entirely from ``alpha < 1`` breaking the
Jacobi limit cycle on the large connectomes. H30's sweep cap was raised to 40 in the same
Phase-6.2 commit, so the comparison H35 vs H30 isolates ``alpha`` (both at 40 sweeps on
connectome/mouse, 12 on microns).

Leakage-safety / compute accounting
-----------------------------------
Identical to H30: the oracle is used ONLY to accept/reject a whole candidate rank vector;
every move is the closed-form exact gain from input weights + current ranks. The sift adds
**0 gradient steps**, so ``n_epochs_done`` stays the baseline gradient budget; the added
non-gradient wall-clock (the sift sweeps) is recorded in ``history.attrs`` and folded into
``wall_clock_s``. ``best_positions`` is the final best rank vector cast to float32 (exact
integer at ``n < 2**24`` for all three datasets), so the runner's ``score == best_score``
re-score assertion holds.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct
from ..refine import sift_underrelaxed
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H35"
HYPOTHESIS = (
    "After H02-warm-started Rocket, an under-relaxed two-phase exact-gain Jacobi sift "
    "(move each node only a fraction alpha toward its exact-optimal gap after k_full warm "
    "sweeps) breaks the limit cycle that plain Jacobi (H30) suffers on large dense "
    "connectomes; refined >= H30's Jacobi sift at equal gradient budget, non-regressing."
)

# Per-dataset total gradient budget (= baseline single-run budget; sift adds 0 grad steps).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}
# Per-dataset max sweeps. Matched to H30 so H35-vs-H30 isolates alpha, not sweep count
# (microns kept at 12 for adequate runtime; mouse converges in ~5 so 40 is a harmless cap).
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12}
_K_FULL = 6      # warm sweeps at alpha=1 before switching to under-relaxation
_ALPHA = 0.7     # under-relaxation factor (sweet spot from the dr_tmp sizing)


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run H02-warm-started Rocket, then the under-relaxed two-phase exact-gain sift.

    Returns the FINAL best (max of pure Rocket and refined) in ``RocketResult``. The
    pure-Rocket best is reported separately via ``history.attrs``.
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

    # ── Stage 3: under-relaxed two-phase exact-gain sift on the Rocket order ──────
    rank0 = np.argsort(np.argsort(pure_best_positions, kind="stable"),
                       kind="stable").astype(np.int64)

    sift_budget = None if time_limit is None else max(0.0, time_limit - rocket.wall_clock_s)
    best_rank, refined_score, sweep_log = sift_underrelaxed(
        g, rank0, k_full=_K_FULL, alpha=_ALPHA,
        max_sweeps=_MAX_SWEEPS.get(g.name, 40), time_budget_s=sift_budget)

    sift_time_s = float(sum(row["wall"] for row in sweep_log))

    # ── Adopt refined ONLY if it strictly beats pure Rocket (best-by-oracle) ─────
    if refined_score > pure_best_score:
        best_positions = best_rank.astype(np.float32)
        best_score = refined_score
    else:
        best_positions = pure_best_positions
        best_score = pure_best_score

    best_pct = pct(best_score, total)

    # ── Provenance (compute accounting + pure vs refined) ────────────────────────
    hist = rocket.history
    hist.attrs["pure_best_score"] = pure_best_score
    hist.attrs["pure_best_pct"] = pure_best_pct
    hist.attrs["n_sift_sweeps"] = len(sweep_log)
    hist.attrs["n_sift_accepted"] = int(sum(1 for r in sweep_log if r["accepted"]))
    hist.attrs["sift_time_s"] = sift_time_s
    hist.attrs["sift_alpha"] = _ALPHA
    hist.attrs["sift_k_full"] = _K_FULL
    hist.attrs["refined_best_score"] = best_score
    hist.attrs["refined_best_pct"] = best_pct
    hist.attrs["sift_log"] = sweep_log

    return RocketResult(
        best_positions=best_positions,
        best_score=best_score,
        best_pct=best_pct,
        history=hist,
        # n_epochs_done UNCHANGED = baseline gradient budget; sift adds 0 grad steps.
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=rocket.wall_clock_s + sift_time_s,
    )
