"""Variant H01 — Multi-start restarts with best-of selection.

Hypothesis: running K independent short Rocket optimizations from DIFFERENT random
inits and keeping the best discrete-scoring result beats one long run at equal total
compute. Seed-stability analysis shows scores are tight (~82.9%) but orderings only
correlate Spearman ~0.96 and sources are unstable (Jaccard@1000 ~0.33-0.40): runs land
in different basins, so best-of-K over a high-variance ordering should harvest the right
tail. The baseline already keeps best-by-oracle, so best-of-K is leakage-safe.

Algorithmic change vs baseline: instead of one run of ``total`` epochs, run K=4 plain
Rocket runs of ``total/K`` epochs each, from distinct deterministic sub-seeds
(``seed + j*7919`` -> different N(0,1) inits), and keep the best discrete-scoring result.

Compute-matched: ``RocketResult.n_epochs_done`` is set to the TOTAL gradient steps
summed across all K restarts (= the baseline's total budget). This is the exact same
total compute as the single baseline run and as ``baseline_multistart`` K=4 (the fair
equal-budget comparator). Only the algorithm (restart strategy) differs; the oracle is
never read inside the loop except via the baseline's existing best-by-oracle tracking.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData

ID = "H01"
HYPOTHESIS = (
    "Multi-start: K=4 independent short Rocket runs from different random inits, "
    "best-of-K by oracle, at equal total compute, beats one long run."
)

# Fixed number of restarts for this variant (compute split into K runs of total/K).
K = 4

# Per-dataset total epoch budget (= baseline's single-run budget; split across K).
_TOTAL_EPOCHS = {"connectome": 20_000, "mouse": 5_000}


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run K=4 plain Rocket restarts (total/K epochs each), keep the best by oracle.

    Each restart uses a distinct deterministic sub-seed (``seed + j*7919``) which seeds
    a different N(0,1) random init. ``n_epochs_done`` is the SUM of gradient steps across
    all restarts (= total compute budget), so this is directly comparable, at EQUAL total
    compute, to the single baseline run and to ``baseline_multistart`` K=4.
    """
    total = _TOTAL_EPOCHS.get(g.name, RocketConfig.epochs)
    per_run = max(1, total // K)
    per_limit = (time_limit / K) if time_limit else None

    best: Optional[RocketResult] = None
    steps_done = 0
    for j in range(K):
        # Distinct, deterministic sub-seed per restart -> different random init.
        # Never peeks at / hardcodes the discrete metric.
        sub_seed = int(seed) + j * 7919
        cfg = RocketConfig(epochs=per_run)
        res = run_rocket(g, cfg, seed=sub_seed, device=device, time_limit=per_limit)
        steps_done += res.n_epochs_done
        # Best-of-K by the exact oracle score (baseline already tracks best-by-oracle).
        if best is None or res.best_score > best.best_score:
            best = res

    assert best is not None
    # Report TOTAL gradient steps across all K restarts as the compute-match basis.
    return RocketResult(
        best_positions=np.asarray(best.best_positions),
        best_score=best.best_score,
        best_pct=best.best_pct,
        history=best.history,
        n_epochs_done=steps_done,
        wall_clock_s=best.wall_clock_s,
    )
