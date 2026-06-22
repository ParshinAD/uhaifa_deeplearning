"""Iterated Local Search / Large-Neighbourhood-Search wrapper on the H30 sift move.

This is the H31 refiner. It sits *on top of* the CONFIRMED H30 full-range exact-gain
Jacobi sift (:func:`mfas.refine.insertion.sift`): after the sift reaches a fixed point,
the LNS outer loop repeatedly **perturbs** the current best order and **re-sifts** it,
keeping the strictly-best candidate the frozen oracle has ever scored (best-by-oracle).

The hope (backlog H31) is that the coordinated multi-node reorder the connectome gap
requires is reachable by perturb→re-sift escapes that a single-pass sift cannot make.
Expected direction: positive but smaller than H30; honest chance of a no-op on real
graphs (the prototype found only +0.01–0.03 pp on mouse).

Perturbation (target-blind, leakage-safe)
-----------------------------------------
Two modes, both decided ONLY from input edge weights + current ranks (never the oracle,
never ``data/best_solution``):

* ``ruin`` — pick the ``k`` *victims* carrying the most CURRENT back-edge weight (an edge
  ``(u, v)`` is back iff ``rank[u] > rank[v]``; each endpoint is charged the edge weight).
  These are the nodes most badly placed under the current order. Pull them out and
  re-insert each at its exact-optimal gap given the *others'* current ranks — a
  sequential Gauss–Seidel sweep among only the ``k`` victims (so re-inserting victim
  ``i`` already sees victims ``0..i-1`` in their new spots). This is a coordinated
  multi-node move single-node Jacobi sift cannot make.
* ``kick`` — pick ``k`` victims uniformly at random and re-insert each at its
  exact-optimal gap (a target-blind random restart of the neighbourhood).

Efficiency
----------
The per-round victim gaps are read from a SINGLE vectorized :func:`jacobi_best_gaps`
call over the whole graph (O(m log m)); we then apply only the ``k`` victims' moves
sequentially (re-querying the kernel after each application would be O(k·m) and is the
trap the prototype's O(n)-per-move Gauss–Seidel kernel fell into). A short
:func:`sift` sweep (``inner_sweeps``) cleans up after the perturbation. The whole loop
is time-budgeted so the realised wall multiplier stays under the 2× ceiling.

Monotonicity
------------
The returned order is tracked best-by-oracle and is seeded from the sift result, so
``ils_lns`` best_score >= the sift best_score >= the initial score, by construction.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..io import GraphData
from ..metrics import pct, score_from_order
from .insertion import build_sift_edges, jacobi_best_gaps, sift

__all__ = ["back_edge_weight", "apply_victim_reinsertions", "ils_lns"]


# ──────────────────────────────────────────────────────────────────────────────
# Target-blind ruin selector: per-node current back-edge weight
# ──────────────────────────────────────────────────────────────────────────────
def back_edge_weight(rank: np.ndarray, src: np.ndarray, tgt: np.ndarray,
                     w: np.ndarray, n: int) -> np.ndarray:
    """Per-node CURRENT back-edge weight (input weights + current ranks only).

    An edge ``(u, v)`` is *back* (feedback) under ``rank`` iff ``rank[u] > rank[v]``
    (i.e. NOT feedforward; the oracle counts ``rank[tgt] > rank[src]`` as feedforward).
    Each back edge charges BOTH of its endpoints its weight, so ``back[u]`` is the total
    feedback weight incident to node ``u`` — large values flag the worst-placed nodes.

    Leakage-safe: derived from the input edge weights and the current ranks only; the
    frozen oracle is never consulted. Self-loops never count (``rank[u] > rank[u]`` is
    false).
    """
    rank = np.asarray(rank, dtype=np.int64)
    is_back = rank[src] > rank[tgt]
    back = np.zeros(n, dtype=np.float64)
    if is_back.any():
        np.add.at(back, src[is_back], w[is_back])
        np.add.at(back, tgt[is_back], w[is_back])
    return back


# ──────────────────────────────────────────────────────────────────────────────
# Sequential (Gauss–Seidel) re-insertion of the k victims
# ──────────────────────────────────────────────────────────────────────────────
def apply_victim_reinsertions(rank: np.ndarray, victims: np.ndarray,
                              src: np.ndarray, tgt: np.ndarray, w: np.ndarray,
                              n: int, tol: float = 1e-9) -> np.ndarray:
    """Re-insert each victim at its exact-optimal gap, sequentially (Gauss–Seidel).

    For each victim in turn, the exact best gap is read from a fresh
    :func:`jacobi_best_gaps` restricted-read over the CURRENT (partially updated) ranks,
    so victim ``i`` already sees victims ``0..i-1`` in their new positions. Because only
    ``k`` victims move, this is the coordinated multi-node perturbation; the per-victim
    gap query uses the vectorized kernel (cheap relative to one sift sweep) but we read
    only the victim's slot.

    The move uses a fractional sort key (``best_gap - 0.5``) and a single stable argsort
    rebuild, identical in spirit to :func:`mfas.refine.insertion.jacobi_rebuild`, so the
    rank vector stays a valid permutation. Leakage-safe: every gap comes from the
    closed-form exact-gain kernel (input weights + current ranks); the oracle is never
    consulted here.
    """
    rank = np.asarray(rank, dtype=np.int64).copy()
    for u in victims:
        u = int(u)
        best_gap, gain = jacobi_best_gaps(rank, src, tgt, w, n)
        if gain[u] <= tol or int(best_gap[u]) == int(rank[u]):
            continue
        # Rebuild moving ONLY this victim to its requested gap (fractional key).
        key = rank.astype(np.float64)
        key[u] = float(best_gap[u]) - 0.5
        rank = np.argsort(np.argsort(key, kind="stable"), kind="stable").astype(np.int64)
    return rank


def _apply_victim_reinsertions_jacobi(rank: np.ndarray, victims: np.ndarray,
                                      best_gap: np.ndarray, gain: np.ndarray,
                                      n: int, tol: float = 1e-9) -> np.ndarray:
    """Apply ALL victims' precomputed gaps at once (Jacobi) — one rebuild, O(m log m).

    Cheaper than the sequential version (a single :func:`jacobi_best_gaps` for the whole
    round instead of one per victim), at the cost of the victims not seeing each other's
    moves. Used for large graphs where per-victim kernel calls would dominate the budget.
    The subsequent short sift sweep and the oracle accept/reject absorb any Jacobi
    overshoot, exactly as in :func:`mfas.refine.insertion.sift`.
    """
    rank = np.asarray(rank, dtype=np.int64)
    key = rank.astype(np.float64).copy()
    mask = np.zeros(n, dtype=bool)
    mask[victims] = True
    movers = mask & (gain > tol)
    key[movers] = best_gap[movers].astype(np.float64) - 0.5
    return np.argsort(np.argsort(key, kind="stable"), kind="stable").astype(np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# ILS / LNS driver (perturb → re-sift → keep-best-by-oracle)
# ──────────────────────────────────────────────────────────────────────────────
def ils_lns(g: GraphData, init_rank: np.ndarray, *, time_budget_s: float,
            k: int, mode: str = "ruin", inner_sweeps: int = 2,
            sequential_victims: bool = True, seed: int = 0, tol: float = 1e-9,
            ) -> Tuple[np.ndarray, float, List[Dict]]:
    """Iterated Local Search / LNS wrapping the H30 sift move, best-by-oracle.

    Loop until ``time_budget_s`` is spent (or a max round guard): from the current best
    order, (1) PERTURB by re-inserting ``k`` victims at their exact-optimal gaps — victims
    chosen by highest current back-edge weight (``mode="ruin"``) or uniformly at random
    (``mode="kick"``) — then (2) run a SHORT :func:`sift` sweep (``inner_sweeps``), then
    (3) keep the candidate iff the frozen oracle says it strictly beats the global best.

    Every perturbation is target-blind (input weights + current ranks only); the oracle
    is consulted ONLY to accept/reject whole candidate vectors. The result is seeded from
    ``init_rank`` (already best-by-oracle from the sift) so it never regresses.

    Returns ``(best_rank int64[n], best_score float, round_log)`` where each ``round_log``
    row is ``{round, mode, cand_pct, accepted, n_victims, wall}``.
    """
    n = g.n_nodes
    src_o = np.asarray(g.src)
    tgt_o = np.asarray(g.tgt)
    src, tgt, w = build_sift_edges(g)
    total = g.total_weight
    rng = np.random.RandomState(seed + 7919)

    best_rank = np.asarray(init_rank, dtype=np.int64).copy()
    best_score = score_from_order(best_rank, src_o, tgt_o, g.weight)

    k = max(1, min(int(k), n - 1)) if n > 1 else 0
    round_log: List[Dict] = []
    if k == 0 or time_budget_s <= 0:
        return best_rank, float(best_score), round_log

    t_start = time.time()
    r = 0
    # Hard guard so a degenerate fast loop cannot spin forever within the budget.
    max_rounds = 100_000
    while r < max_rounds:
        if (time.time() - t_start) >= time_budget_s:
            break
        t0 = time.time()
        rank = best_rank.copy()

        # ── 1. Choose victims (target-blind) ────────────────────────────────────
        if mode == "ruin":
            back = back_edge_weight(rank, src, tgt, w, n)
            # Top-k by back-edge weight; break exact ties randomly to vary rounds.
            jitter = rng.random_sample(n) * 1e-12
            victims = np.argpartition(-(back + jitter), kth=k - 1)[:k]
            victims = victims[back[victims] > 0]  # only genuinely back-laden nodes
            if victims.size == 0:
                # No back-edge-laden node left to ruin -> fall back to a random kick.
                victims = rng.choice(n, size=k, replace=False)
        else:  # "kick"
            victims = rng.choice(n, size=k, replace=False)
        rng.shuffle(victims)

        # ── 2. Perturb: re-insert victims at their exact-optimal gaps ────────────
        if sequential_victims:
            rank = apply_victim_reinsertions(rank, victims, src, tgt, w, n, tol=tol)
        else:
            bg, gn_ = jacobi_best_gaps(rank, src, tgt, w, n)
            rank = _apply_victim_reinsertions_jacobi(rank, victims, bg, gn_, n, tol=tol)

        # ── 3. Short sift sweep to clean up the perturbation ────────────────────
        if inner_sweeps > 0:
            rank, _cand_inner, _ = sift(g, rank, max_sweeps=inner_sweeps)
            # sift returns its own best-by-oracle from `rank`; that is the candidate.

        cand_score = score_from_order(rank, src_o, tgt_o, g.weight)
        accepted = cand_score > best_score
        if accepted:
            best_rank = rank.copy()
            best_score = cand_score
        wall = time.time() - t0
        round_log.append(dict(round=r, mode=mode,
                              cand_pct=pct(cand_score, total),
                              accepted=bool(accepted),
                              n_victims=int(np.size(victims)), wall=wall))
        r += 1

    return best_rank, float(best_score), round_log
