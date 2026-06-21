"""Comparator variant: PLAIN baseline Rocket run as K naive restarts, best-of-K.

This is NOT a research hypothesis — it is the *equal-budget comparator* for
multi-start / restart / subsample variants (e.g. H01, H13). It runs the UNCHANGED
baseline Rocket ``K`` times from different seeds, each for ``total/K`` epochs, and keeps
the best discrete-scoring result (best-of-K by the frozen oracle — exactly what the
baseline already does each run). Its ``n_epochs_done`` equals the total gradient steps
across all restarts, so it can be compared to a restart variant at *equal total compute*.

Why it exists: a restart variant could "win" simply because best-of-K harvests the
tail of a noisy ordering — that is extra sampling, not algorithmic merit. Judging such a
variant against THIS comparator (naive restarts of the plain baseline, same total steps)
isolates whether the variant's *strategy* (e.g. diverse inits) beats mere sampling.

K is read from ``MFAS_MULTISTART_K`` (default 4) so the verifier can match a variant's K.
No algorithmic change to Rocket — only the plain baseline, repeated.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData

ID = "baseline_multistart"
HYPOTHESIS = "Equal-budget comparator: plain baseline Rocket, K naive restarts, best-of-K."

# Total epoch budget per dataset (mirrors baseline_passthrough / configs).
_TOTAL_EPOCHS = {"connectome": 20_000, "mouse": 5_000}


def _k() -> int:
    try:
        k = int(os.environ.get("MFAS_MULTISTART_K", "4"))
    except ValueError:
        k = 4
    return max(1, k)


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run K plain-baseline restarts (total/K epochs each), keep the best by oracle.

    ``n_epochs_done`` is the SUM of epochs across restarts (= total gradient steps),
    so this matches a restart variant's compute budget exactly.
    """
    K = _k()
    total = _TOTAL_EPOCHS.get(g.name, RocketConfig.epochs)
    per_run = max(1, total // K)
    per_limit = (time_limit / K) if time_limit else None

    best: Optional[RocketResult] = None
    steps_done = 0
    for j in range(K):
        # Distinct, deterministic seed per restart (prime offset; never peeks at the metric).
        sub_seed = int(seed) + j * 7919
        cfg = RocketConfig(epochs=per_run)
        res = run_rocket(g, cfg, seed=sub_seed, device=device, time_limit=per_limit)
        steps_done += res.n_epochs_done
        if best is None or res.best_score > best.best_score:
            best = res

    assert best is not None
    # Report TOTAL gradient steps across restarts as the budget basis.
    return RocketResult(
        best_positions=np.asarray(best.best_positions),
        best_score=best.best_score,
        best_pct=best.best_pct,
        history=best.history,
        n_epochs_done=steps_done,
        wall_clock_s=best.wall_clock_s,
    )
