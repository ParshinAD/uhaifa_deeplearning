"""Variant H31 — H30 (warm-started Rocket + sift) wrapped in an ILS/LNS outer loop.

Hypothesis (backlog H31): wrapping H30's confirmed full-range exact-gain insertion in an
Iterated Local Search / Large-Neighbourhood-Search — perturb the current best order
(ruin-&-recreate: remove the ``k`` nodes carrying the most CURRENT back-edge weight and
re-insert each at its exact-optimal rank), run a short sift sweep, keep-best-by-oracle —
recovers *more* of the gap than single-pass H30 by reaching the coordinated multi-node
reorder the gap requires. Expected direction: positive but smaller than H30; honest
chance of a no-op on the real graphs.

Pipeline
--------
1. Greedy-FAS warm start (H02): ``greedy_fas_order(g)`` -> evenly-spaced positions.
2. UNCHANGED ``run_rocket`` at the baseline epoch budget -> PURE Rocket best (reported
   separately, per CLAUDE.md).
3. Full-range exact-gain Jacobi sift (:func:`mfas.refine.sift`) to a fixed point — the
   H30 refiner; its result is the H30 comparator (also reported separately).
4. ILS/LNS (:func:`mfas.refine.ils_lns`) within the RESIDUAL wall budget: perturb→re-sift
   →keep-best-by-oracle, using the H30 sift result as the seed (so refined >= H30 >= pure).

Budget (the binding constraint)
-------------------------------
The H30 sift alone is already ~1.4× the connectome baseline wall. H31 must keep the WHOLE
refinement (sift + LNS) under a hard 2× ceiling. We measure the Rocket wall, then give the
sift its own fixed-point budget and give the LNS the residual up to a refinement cap of
``_REFINE_WALL_FRAC × rocket_wall`` (target total <= ~1.9×). The realised multiplier
(``(rocket + sift + lns) / rocket``) is recorded in ``history.attrs`` per dataset.

Leakage-safety / compute accounting
-----------------------------------
Every perturbation is target-blind (highest CURRENT back-edge weight, from input weights
+ ranks) and every repair uses the closed-form exact-gain kernel; the frozen oracle is
consulted ONLY to accept/reject a whole candidate rank vector (best-by-oracle), exactly as
the baseline tracks its best. The LNS adds **0 gradient steps**, so ``n_epochs_done`` stays
the baseline gradient budget and the compute-matched comparators are ``H30`` (the key test:
does the wrapper earn its wall?), ``H02`` (isolates all refinement), and
``baseline_passthrough`` (total stacked gain) at the same seeds. ``best_positions`` is the
final best rank vector cast to float32 (an exact integer at ``n < 2**24`` for all three
datasets), so the runner's ``score == best_score`` re-score assertion holds.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct, score_from_order
from ..refine import ils_lns, sift
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H31"
HYPOTHESIS = (
    "Wrapping the confirmed H30 exact-gain sift in an ILS/LNS outer loop "
    "(ruin-&-recreate: re-insert the k highest-back-edge-weight nodes at their exact "
    "gaps, short re-sift, keep-best-by-oracle) recovers more of the gap than single-pass "
    "H30 within a <=2x wall budget; refined >= H30 >= pure Rocket."
)

# Per-dataset total gradient budget (= baseline single-run budget; LNS adds 0 grad steps).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}
# Per-dataset max Jacobi sweeps for the H30 sift fixed point (mouse tiny -> more).
_MAX_SWEEPS = {"connectome": 12, "mouse": 30, "microns": 12}

# LNS configuration per dataset (chosen at the prototype gate; victims k, perturb mode,
# inner re-sift sweeps, and whether per-victim moves are sequential Gauss-Seidel).
#   k         : number of ruin victims per round
#   inner     : short sift sweeps after each perturbation
#   sequential: re-insert victims one-by-one (sees prior moves) vs all-at-once (Jacobi)
_LNS = {
    "connectome": dict(k=64, mode="ruin", inner=1, sequential=False),
    "mouse":      dict(k=8,  mode="ruin", inner=2, sequential=True),
    "microns":    dict(k=64, mode="ruin", inner=1, sequential=False),
}

# Refinement wall ceiling as a fraction of the Rocket wall: total <= (1 + frac) x baseline.
# 0.85 targets <= ~1.85x worst case (sift ~0.4x + LNS residual); hard ceiling is 2x.
_REFINE_WALL_FRAC = 0.85


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run H02-warm-started Rocket, the H30 sift, then the ILS/LNS outer loop.

    Returns the FINAL best (max of pure Rocket, H30 sift, and LNS) in ``RocketResult``.
    The pure-Rocket and H30-sift bests are reported separately via ``history.attrs``.
    """
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))
    total = g.total_weight
    src_o, tgt_o = np.asarray(g.src), np.asarray(g.tgt)

    # ── Stage 1+2: H02 warm start -> unchanged Rocket (PURE) ─────────────────────
    order = greedy_fas_order(g)
    init_positions = _init_positions_from_order(order, device)
    rocket = run_rocket(g, cfg, seed=seed, device=device,
                        init_positions=init_positions, time_limit=time_limit)

    pure_best_positions = rocket.best_positions
    pure_best_score = rocket.best_score
    pure_best_pct = rocket.best_pct
    rocket_wall = rocket.wall_clock_s

    # Refinement wall budget (the binding constraint). If a hard time_limit is given,
    # respect whatever wall remains after Rocket; otherwise cap at a fraction of the
    # Rocket wall so the total stays under the 2x ceiling.
    if time_limit is not None:
        refine_budget = max(0.0, time_limit - rocket_wall)
    else:
        refine_budget = _REFINE_WALL_FRAC * rocket_wall

    # ── Stage 3: H30 full-range exact-gain Jacobi sift to a fixed point ──────────
    rank0 = np.argsort(np.argsort(pure_best_positions, kind="stable"),
                       kind="stable").astype(np.int64)
    t_sift = time.time()
    sift_rank, sift_score, sweep_log = sift(
        g, rank0, max_sweeps=_MAX_SWEEPS.get(g.name, 12), time_budget_s=refine_budget)
    sift_time_s = time.time() - t_sift
    sift_pct = pct(sift_score, total)

    # ── Stage 4: ILS/LNS within the residual refinement budget ──────────────────
    lns_budget = max(0.0, refine_budget - sift_time_s)
    lns_cfg = _LNS.get(g.name, dict(k=8, mode="ruin", inner=1, sequential=True))
    t_lns = time.time()
    lns_rank, lns_score, round_log = ils_lns(
        g, sift_rank, time_budget_s=lns_budget, k=lns_cfg["k"], mode=lns_cfg["mode"],
        inner_sweeps=lns_cfg["inner"], sequential_victims=lns_cfg["sequential"],
        seed=seed)
    lns_time_s = time.time() - t_lns
    lns_pct = pct(lns_score, total)

    # ── Adopt the best of {pure, sift, LNS} by oracle (best-by-oracle) ──────────
    # By construction lns_score >= sift_score >= pure (each stage is seeded best-by-oracle
    # from the previous), but compute the argmax explicitly for safety.
    candidates = [
        (pure_best_score, pure_best_positions),
        (sift_score, sift_rank.astype(np.float32)),
        (lns_score, lns_rank.astype(np.float32)),
    ]
    best_score, best_positions = max(candidates, key=lambda c: c[0])
    best_pct = pct(best_score, total)

    refine_wall = sift_time_s + lns_time_s
    realized_multiplier = (rocket_wall + refine_wall) / max(rocket_wall, 1e-9)

    # ── Provenance (compute accounting + pure vs sift vs LNS) ────────────────────
    hist = rocket.history
    hist.attrs["pure_best_score"] = pure_best_score
    hist.attrs["pure_best_pct"] = pure_best_pct
    hist.attrs["sift_best_score"] = sift_score          # the H30 comparator
    hist.attrs["sift_best_pct"] = sift_pct
    hist.attrs["n_sift_sweeps"] = len(sweep_log)
    hist.attrs["sift_time_s"] = sift_time_s
    hist.attrs["lns_best_score"] = lns_score
    hist.attrs["lns_best_pct"] = lns_pct
    hist.attrs["n_lns_rounds"] = len(round_log)
    hist.attrs["n_lns_accepted"] = int(sum(1 for r in round_log if r["accepted"]))
    hist.attrs["lns_time_s"] = lns_time_s
    hist.attrs["refined_best_score"] = best_score
    hist.attrs["refined_best_pct"] = best_pct
    hist.attrs["lns_gain_vs_sift_pct"] = lns_pct - sift_pct
    hist.attrs["rocket_wall_s"] = rocket_wall
    hist.attrs["realized_multiplier"] = realized_multiplier
    hist.attrs["lns_config"] = lns_cfg
    hist.attrs["lns_round_log_tail"] = round_log[-20:]

    return RocketResult(
        best_positions=best_positions,
        best_score=best_score,
        best_pct=best_pct,
        history=hist,
        # n_epochs_done UNCHANGED = baseline gradient budget; LNS adds 0 grad steps.
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=rocket_wall + refine_wall,
    )
