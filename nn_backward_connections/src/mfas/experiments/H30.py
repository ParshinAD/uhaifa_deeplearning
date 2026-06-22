"""Variant H30 — H02-warm-started Rocket + full-range exact-gain sift post-phase.

Hypothesis (backlog H30): after H02-warm-started Rocket converges, a leakage-safe
full-range exact-gain Jacobi node re-insertion ("sift") post-phase recovers feedforward
weight the continuous optimizer leaves on the table; the refined order is never worse
than pure Rocket (refined >= pure Rocket by oracle accept/reject).

Pipeline
--------
1. Greedy-FAS warm start (H02): ``greedy_fas_order(g)`` -> evenly-spaced positions.
2. UNCHANGED ``run_rocket`` at the baseline epoch budget -> PURE Rocket best (tracked
   and reported separately, per CLAUDE.md).
3. Full-range exact-gain Jacobi sift (:func:`mfas.refine.sift`) on the Rocket order:
   each sweep moves every node toward its EXACT feedforward-maximising rank (whole
   line, not a window), keeping a candidate only if the frozen oracle says it strictly
   beats the current best.

Distinct from killed variants
-----------------------------
* vs H04 (weighted-barycenter, 0 accepts): sift places a node at the *argmax of the
  exact step-function* of incident-edge feedforward weight, not the *mean* of its
  neighbours' ranks; for bimodal cyclic-core neighbourhoods the argmax is far from the
  mean.
* vs H22 (bounded +-W sift, killed long-range): identical exact-gain code, but the
  candidate range is the WHOLE line, so a node can travel ~n ranks (the gap is
  long-range, p50 ~22.6k ranks on the connectome).

Leakage-safety / compute accounting
-----------------------------------
The frozen oracle is used ONLY to accept/reject a whole candidate rank vector
(best-by-oracle), exactly as the baseline already tracks its best; every move is
chosen from input edge weights + current ranks (closed-form gain). The sift adds
**0 gradient steps**, so ``n_epochs_done`` stays the baseline gradient budget and the
compute-matched comparators are ``H02`` (the promotion comparison, isolating the sift)
and ``baseline_passthrough`` (total stacked gain) at the same seeds. The added
non-gradient wall-clock (the sift sweeps) is recorded in ``history.attrs`` and folded
into ``wall_clock_s``. ``best_positions`` is the final best rank vector cast to
float32 (an exact integer at ``n < 2**24`` for all three datasets), so the runner's
``score == best_score`` re-score assertion holds.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct
from ..refine import sift
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H30"
HYPOTHESIS = (
    "After H02-warm-started Rocket, a leakage-safe full-range exact-gain Jacobi node "
    "re-insertion (sift) post-phase recovers feedforward weight the continuous optimizer "
    "leaves on the table; refined >= pure Rocket."
)

# Per-dataset total gradient budget (= baseline single-run budget; sift adds 0 grad steps).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}
# Per-dataset max Jacobi sweeps. Raised to 40 on connectome/mouse in Phase 6.2: the Jacobi
# iterate was still improving at the old cap of 12 (it limit-cycles rather than converges,
# so the extra sweeps recover ~+0.045 pp on connectome via the best-by-oracle phase).
# microns kept at 12 for adequate runtime (its per-sweep cost is ~3x connectome's).
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12}


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run H02-warm-started Rocket, then the full-range exact-gain Jacobi sift.

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

    # ── Stage 3: full-range exact-gain Jacobi sift on the Rocket order ───────────
    # Integer ranks of the Rocket best positions (rank[node] = position on the line).
    rank0 = np.argsort(np.argsort(pure_best_positions, kind="stable"),
                       kind="stable").astype(np.int64)

    sift_budget = None if time_limit is None else max(0.0, time_limit - rocket.wall_clock_s)
    best_rank, refined_score, sweep_log = sift(
        g, rank0, max_sweeps=_MAX_SWEEPS.get(g.name, 12), time_budget_s=sift_budget)

    sift_time_s = float(sum(row["wall"] for row in sweep_log))

    # ── Adopt refined ONLY if it strictly beats pure Rocket (best-by-oracle) ─────
    if refined_score > pure_best_score:
        # float32 rank vector is oracle-exact at n < 2**24 -> runner assert holds.
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
