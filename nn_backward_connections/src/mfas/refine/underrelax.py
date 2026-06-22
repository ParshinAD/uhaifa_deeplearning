"""Under-relaxed two-phase exact-gain sift — Phase 6.2 refiner (leakage-safe).

Motivation
----------
The production sift (:func:`mfas.refine.insertion.sift`) is a *Jacobi* iteration: every
node moves FULLY to its exact feedforward-maximising gap each sweep (key = ``best_gap``).
On large dense connectomes this never converges — it enters a period-2 limit cycle
(thousands of nodes leapfrogging each other forever), so best-by-oracle only creeps up
via the lucky phase of the oscillation (diagnosed in ``dr_tmp/FINDINGS_underrelaxation.md``:
connectome ~5,255 movers stuck, candidate alternating 83.807/83.796).

Fix = **under-relaxation** (the textbook cure for Jacobi 2-cycles): move each node only a
fraction ``alpha`` of the way to its exact-optimal gap, ``key = rank + alpha*(target - rank)``.
With ``alpha=0.7`` the iterate converges (movers collapse to a few hundred) and reaches a
strictly higher fixed point on both large connectomes. On tiny graphs (mouse) plain Jacobi
already converges and ``alpha<1`` would find a slightly worse optimum, so a **two-phase
schedule** is used: ``alpha=1`` for the first ``k_full`` warm sweeps (fast progress, captures
the Jacobi optimum on graphs that converge), then ``alpha`` (settle the cycle on graphs that
don't). The best-by-oracle tracker keeps the max over the whole trajectory, so the result is
never worse than plain Jacobi on any dataset.

Relationship to the production sift
-----------------------------------
The exact-gain kernel (:func:`mfas.refine.insertion.jacobi_best_gaps`, brute-force verified)
is reused unchanged; only the *rebuild* changes. At ``alpha=1.0`` the rebuild is bit-identical
to :func:`mfas.refine.insertion.jacobi_rebuild` and this driver reproduces ``sift`` exactly
(asserted in ``tests/test_refine_underrelax.py``).

Leakage-safety
--------------
Every move is chosen purely from input edge weights + current ranks (the closed-form gain
from the kernel). The frozen oracle (:func:`mfas.metrics.score_from_order`) is consulted only
to accept/reject a whole candidate rank vector (best-by-oracle), exactly as ``sift`` does;
it never enters a move choice.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..io import GraphData
from ..metrics import pct, score_from_order
from .insertion import build_sift_edges, jacobi_best_gaps

__all__ = ["underrelaxed_rebuild", "sift_underrelaxed"]


def underrelaxed_rebuild(rank: np.ndarray, best_gap: np.ndarray, gain: np.ndarray,
                         alpha: float, tol: float = 1e-9) -> np.ndarray:
    """Under-relaxed Jacobi rebuild: move each mover a fraction ``alpha`` toward its gap.

    Movers (``gain > tol``) get the fractional key ``rank + alpha*((best_gap-0.5) - rank)``;
    non-movers keep their current ``rank``. The new rank vector is the stable double-argsort
    of the keys. At ``alpha == 1.0`` the mover key collapses to ``best_gap - 0.5`` exactly,
    i.e. this reproduces :func:`mfas.refine.insertion.jacobi_rebuild`.
    """
    rank = np.asarray(rank, dtype=np.int64)
    best_gap = np.asarray(best_gap, dtype=np.int64)
    gain = np.asarray(gain, dtype=np.float64)
    key = rank.astype(np.float64).copy()
    movers = gain > tol
    target = best_gap[movers].astype(np.float64) - 0.5
    key[movers] = rank[movers].astype(np.float64) + alpha * (target - rank[movers])
    return np.argsort(np.argsort(key, kind="stable"), kind="stable").astype(np.int64)


def sift_underrelaxed(g: GraphData, init_rank: np.ndarray, *, k_full: int = 6,
                      alpha: float = 0.7, max_sweeps: int = 40,
                      time_budget_s: Optional[float] = None, tol: float = 1e-9
                      ) -> Tuple[np.ndarray, float, List[Dict]]:
    """Two-phase under-relaxed full-range exact-gain sift, best-by-oracle.

    Schedule: sweeps ``[0, k_full)`` use ``alpha=1`` (full Jacobi step), sweeps
    ``[k_full, max_sweeps)`` use the given ``alpha`` (under-relaxed). The working order
    advances unconditionally each sweep (as in :func:`mfas.refine.insertion.sift`); a
    separate best-by-oracle tracker keeps the strictly-best candidate the frozen oracle has
    ever scored and is what is returned — so the result can never regress below ``init_rank``
    and is >= plain Jacobi on every dataset. Stops early only at a true fixed point (no
    movers) or when ``time_budget_s`` is exceeded.

    Returns ``(best_rank int64[n], best_score float, sweep_log)`` where each ``sweep_log``
    row is ``{sweep, alpha, candidate_pct, accepted, n_movers, wall}``.
    """
    import time

    n = g.n_nodes
    src_o = np.asarray(g.src)
    tgt_o = np.asarray(g.tgt)
    src, tgt, w = build_sift_edges(g)

    work_rank = np.asarray(init_rank, dtype=np.int64).copy()
    best_rank = work_rank.copy()
    best_score = score_from_order(best_rank, src_o, tgt_o, g.weight)
    total = g.total_weight

    sweep_log: List[Dict] = []
    t_start = time.time()
    for s in range(max_sweeps):
        if time_budget_s is not None and (time.time() - t_start) > time_budget_s:
            break
        t0 = time.time()
        a = 1.0 if s < k_full else float(alpha)
        best_gap, gain = jacobi_best_gaps(work_rank, src, tgt, w, n)
        n_movers = int((gain > tol).sum())
        cand_rank = underrelaxed_rebuild(work_rank, best_gap, gain, a, tol=tol)
        cand_score = score_from_order(cand_rank, src_o, tgt_o, g.weight)
        # Working iterate advances unconditionally (relaxation dynamics).
        work_rank = cand_rank
        # Best-by-oracle: keep the strictly-best candidate ever seen.
        accepted = cand_score > best_score
        if accepted:
            best_rank = cand_rank.copy()
            best_score = cand_score
        wall = time.time() - t0
        sweep_log.append(dict(sweep=s, alpha=a, candidate_pct=pct(cand_score, total),
                              accepted=bool(accepted), n_movers=n_movers, wall=wall))
        # Stop only at a true fixed point (no node wants to move).
        if n_movers == 0:
            break
    return best_rank, float(best_score), sweep_log
