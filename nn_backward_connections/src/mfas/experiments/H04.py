"""Variant H04 — In-the-loop discrete refinement (continuous Rocket + periodic local swaps).

Hypothesis (backlog H04): periodically nudging positions toward a locally-improved ordering
(a cheap greedy local search on the CURRENT order, then re-seed positions from the improved
ranks) lets Rocket escape the surrogate plateau and raises the exact feedforward weight.
Crane's whole premise is that local refinement extends quality *beyond* Rocket's plateau
(paper §4.4, 82.87% -> 84.60%); this imports that idea with a CHEAP, in-variant local search
(no Gurobi / no MIP).

Local-search move (leakage-safe)
--------------------------------
The refinement is a **weighted-barycenter rank reposition** -- the classic barycenter
heuristic for one-dimensional vertex arrangement. Given the CURRENT ordering (the stable
argsort of the live positions -> integer ranks in ``[0, n)``), each node is moved toward the
weighted mean rank of its neighbours, using ONLY the input graph's edge weights and the
current ranks:

    target[u] = ( Σ_{(u,v)∈E} w(u,v)·rank[v]  +  Σ_{(t,u)∈E} w(t,u)·rank[t] )
                / ( Σ_{(u,v)} w(u,v) + Σ_{(t,u)} w(t,u) )

i.e. a node is pulled toward the weighted centroid of the ranks of the nodes it connects to.
A feedforward edge ``(u, v)`` wants ``rank[u] < rank[v]``; pulling each node toward its
incident neighbours' centroid (and re-ranking) reduces long "backward" arcs in the
1-D arrangement, which is exactly the local move MFAS heuristics use. The new candidate
ordering is the stable argsort of ``target`` -> new integer ranks. This is decided purely by
LOCAL edge-weight comparisons on the INPUT graph (``g.src``/``g.tgt``/``g.weight``) and the
current ranks; the discrete oracle is NEVER consulted to choose the move.

Leakage-safety / oracle usage
-----------------------------
The frozen discrete oracle (``score_from_positions``) is used EXACTLY as the baseline already
uses it: only to *track* the best-scoring positions seen during the run. The barycenter move
never reads the oracle, never folds the discrete score into the loss, never hardcodes a target
value, and never special-cases a dataset. A refined candidate is only ADOPTED (its ranks
re-seed the live position vector and become the new best) when the oracle says it strictly
beats the current best -- so refinement can never LOWER the tracked best (post >= pre by
construction). If a refinement does not improve, it is discarded and the optimizer continues
from the unmodified iterate.

Anti-collapse guard
-------------------
Re-seeded positions are evenly-spaced standardized values in ``[-1, 1]`` derived from integer
ranks (rank 0 -> -1, rank n-1 -> +1): they are strictly increasing in rank, never tie, never
NaN, and never collapse the vector. The candidate ``target`` is sanitized (NaN/Inf -> current
rank) before the argsort, and we only ever WRITE back positions on an accepted (oracle-better)
candidate.

Compute accounting (fairness)
-----------------------------
H04 runs the FULL single-run gradient budget (connectome 20k, mouse 5k) -- it does NOT split
into restarts -- so the matched-grad-step comparator is ``baseline_passthrough`` at the same
seeds (and the frozen ``baseline_rocket``). ``n_epochs_done`` = the gradient steps actually run
(= baseline budget), so ``total_grad_steps`` matches the baseline.

H04 adds EXTRA NON-GRADIENT compute: the periodic barycenter refinement. It is kept CHEAP and
CAPPED:
  * It runs at most ``MAX_REFINES`` times total (a small fixed number, NOT every step).
  * Each refinement pass is O(m) (two ``np.add.at`` scatter-adds over the edge list) plus one
    O(n log n) argsort plus the standard oracle score (which the baseline already pays at every
    log point). So the extra op cost is ~``MAX_REFINES`` × O(m) on top of the gradient budget.
  * The pre-refinement (PURE Rocket) best is tracked separately and reported (CLAUDE.md rule);
    ``RocketResult`` carries the FINAL (post-refinement) best.
On the 5.6M-edge connectome each refine pass is a couple of vectorised O(m) scatter-adds, so a
small cap keeps the added wall-clock a small fraction of the 20k-step gradient budget. The
exact added-pass count and wall-clock are recorded in ``RocketResult.history`` / reported in
the log so equal-compute fairness is auditable. If the refinement cost is judged non-negligible
vs the gradient budget, the implementer states so and a ``baseline_multistart`` comparison may
also be requested by the verifier.

Implementation note
-------------------
``run_rocket`` exposes no in-loop hook, so this module REPLICATES the ``run_rocket`` main loop
VERBATIM (same N(0,1) default init, Adam, grad-clip=1.0, const->exp LR schedule, cyclic-cosine
β schedule via ``make_beta_schedule``, CPU int64/float64 discrete scoring, best-by-oracle
tracking, history, time-limit handling) and adds ONLY the periodic barycenter refinement.
Everything else is identical to the baseline.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from ..baseline.rocket import RocketConfig, RocketResult, make_beta_schedule
from ..io import GraphData
from ..metrics import pct, score_from_positions

ID = "H04"
HYPOTHESIS = (
    "Periodically refining the current ordering with a cheap leakage-safe weighted-barycenter "
    "local search (decided by input edge weights + current ranks only) and re-seeding positions "
    "from any oracle-improving candidate lets Rocket escape the surrogate plateau and raises the "
    "exact feedforward weight, at the baseline gradient budget (post-refinement >= pure Rocket)."
)

# Per-dataset total gradient budget (= baseline single-run budget; full run, no restarts).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000}

# Refinement schedule (CHEAP, CAPPED). Refinement is attempted at scoring points only and at
# most MAX_REFINES times total, spread across the SECOND HALF of training (after the optimizer
# has left the smooth-β explore phase and the ordering is meaningful). Keeping it to a small
# fixed number of O(m) passes keeps the extra compute a small fraction of the gradient budget.
MAX_REFINES = 8
REFINE_START_FRAC = 0.5   # only start refining after 50% of epochs (ordering has formed)


def _build_csr(g: GraphData):
    """Precompute edge arrays for the O(m) barycenter scatter-adds (leakage-safe inputs only)."""
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    n = g.n_nodes
    # Total incident (in + out) weight per node, for the barycenter denominator.
    incident_w = np.zeros(n, dtype=np.float64)
    np.add.at(incident_w, src, w)
    np.add.at(incident_w, tgt, w)
    return src, tgt, w, incident_w


def _barycenter_candidate_ranks(ranks: np.ndarray, src: np.ndarray, tgt: np.ndarray,
                                w: np.ndarray, incident_w: np.ndarray) -> np.ndarray:
    """One weighted-barycenter pass -> candidate integer ranks (leakage-safe).

    ``ranks[u]`` is the current integer rank of node ``u`` (0 = source side). Each node is
    pulled toward the weighted mean rank of its incident neighbours:
        target[u] = (Σ_out w·rank[v] + Σ_in w·rank[t]) / incident_w[u].
    The candidate ordering is the stable argsort of ``target``. Uses ONLY input edge weights and
    current ranks; never the oracle. Returns a fresh integer-rank vector in ``[0, n)``.
    """
    n = ranks.shape[0]
    rank_f = ranks.astype(np.float64)
    acc = np.zeros(n, dtype=np.float64)
    # out-edges (u, v): node u is pulled toward rank[v]; weight w
    np.add.at(acc, src, w * rank_f[tgt])
    # in-edges (t, u): node u is pulled toward rank[t]; weight w
    np.add.at(acc, tgt, w * rank_f[src])
    with np.errstate(divide="ignore", invalid="ignore"):
        target = acc / incident_w
    # Isolated nodes (incident_w == 0) or any NaN/Inf -> keep current rank (anti-collapse).
    bad = ~np.isfinite(target)
    target[bad] = rank_f[bad]
    # New ordering: stable argsort of the barycenter target -> integer ranks in [0, n).
    order = np.argsort(target, kind="stable")          # order[k] = node at position k
    new_ranks = np.empty(n, dtype=np.int64)
    new_ranks[order] = np.arange(n, dtype=np.int64)
    return new_ranks


def _ranks_to_positions(ranks: np.ndarray) -> np.ndarray:
    """Map integer ranks in [0, n) to evenly-spaced positions in [-1, 1] (strictly increasing)."""
    n = ranks.shape[0]
    return ((ranks.astype(np.float32) / max(n - 1, 1)) * 2.0 - 1.0).astype(np.float32)


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Replicated baseline Rocket loop + periodic leakage-safe barycenter refinement.

    Returns the FINAL (post-refinement) best in ``RocketResult``. The pure-Rocket
    (pre-refinement) best is tracked in ``history`` (column ``pure_best_score`` /
    ``pure_best_pct``) and reported separately in the log per CLAUDE.md.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))
    n = g.n_nodes

    # ── Edge tensors / surrogate inputs (verbatim from run_rocket) ──────────────
    src_t = torch.tensor(np.asarray(g.src), dtype=torch.long, device=device)
    tgt_t = torch.tensor(np.asarray(g.tgt), dtype=torch.long, device=device)
    w_np = np.asarray(g.weight)
    max_w = float(w_np.max())
    nw_t = torch.tensor((w_np / max_w).astype(np.float32), device=device)

    src_np, tgt_np = np.asarray(g.src), np.asarray(g.tgt)
    total_weight = g.total_weight

    def discrete_score(pos_param: torch.Tensor) -> float:
        pos_cpu = pos_param.detach().cpu().numpy()
        return score_from_positions(pos_cpu, src_np, tgt_np, g.weight)

    def discrete_score_arr(arr: np.ndarray) -> float:
        return score_from_positions(arr, src_np, tgt_np, g.weight)

    # CSR-style arrays for the barycenter refinement (built once; input-only -> leakage-safe).
    bsrc, btgt, bw, incident_w = _build_csr(g)

    # ── Init positions (verbatim: N(0,1) default) ───────────────────────────────
    if cfg.init_mode == "random":
        pos_data = torch.randn(n, device=device)
    else:  # pragma: no cover - H04 uses the baseline random init
        from ..baseline.rocket import make_init_positions
        pos_data = make_init_positions(cfg.init_mode, g, seed, device).to(torch.float32)
    positions = torch.nn.Parameter(pos_data)

    # ── Optimizer + LR schedule (verbatim) ──────────────────────────────────────
    optimizer = optim.Adam([positions], lr=cfg.lr)
    milestone = int(cfg.epochs * cfg.lr_decay_start)
    sched_const = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                total_iters=milestone)
    gamma = cfg.lr_end_factor ** (1.0 / max(cfg.epochs - milestone, 1))
    sched_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched_const, sched_exp], milestones=[milestone])

    betas = make_beta_schedule(cfg.epochs, cfg.cycles)

    # ── Best-tracking. ``best_*`` is the FINAL (post-refinement) best returned; ──
    # ``pure_best_*`` tracks the pure-Rocket best (refinement candidates excluded).
    best_score = discrete_score(positions)
    best_positions = positions.detach().clone()
    pure_best_score = best_score
    pure_best_positions = best_positions.clone()

    # Refinement schedule: choose up to MAX_REFINES evenly-spaced scoring points in the
    # second half. We refine when the current scoring iteration index reaches the next target.
    refine_start = int(cfg.epochs * REFINE_START_FRAC)
    log_points = [i for i in range(cfg.epochs)
                  if (i % cfg.log_interval == 0 or i == cfg.epochs - 1) and i >= refine_start]
    if len(log_points) > MAX_REFINES:
        sel = np.linspace(0, len(log_points) - 1, MAX_REFINES).round().astype(int)
        refine_at = set(int(log_points[k]) for k in np.unique(sel))
    else:
        refine_at = set(log_points)

    history = []
    start_time = time.time()
    last_i = 0
    n_refines = 0
    n_refines_accepted = 0
    refine_time = 0.0

    # ── Main loop (verbatim) + periodic refinement ──────────────────────────────
    for i in range(cfg.epochs):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        last_i = i

        beta = float(betas[i])

        optimizer.zero_grad()
        delta = positions[tgt_t] - positions[src_t]
        sig = torch.sigmoid(beta * delta)
        loss = -(sig * nw_t).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([positions], cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if i % cfg.log_interval == 0 or i == cfg.epochs - 1:
            score = discrete_score(positions)
            # Pure-Rocket best (baseline behaviour: raw iterate only).
            if score > pure_best_score:
                pure_best_score = score
                pure_best_positions = positions.detach().clone()
            # Post-refinement best also sees the raw iterate.
            if score > best_score:
                best_score = score
                best_positions = positions.detach().clone()

            # ── Periodic leakage-safe barycenter refinement ──────────────────────
            did_refine = False
            accepted = False
            if i in refine_at:
                t_ref = time.time()
                # Current ordering -> integer ranks (stable argsort of live positions).
                pos_cpu = positions.detach().cpu().numpy()
                cur_ranks = np.empty(n, dtype=np.int64)
                cur_order = np.argsort(pos_cpu, kind="stable")
                cur_ranks[cur_order] = np.arange(n, dtype=np.int64)
                # One weighted-barycenter pass -> candidate ranks (input-only; no oracle).
                cand_ranks = _barycenter_candidate_ranks(
                    cur_ranks, bsrc, btgt, bw, incident_w)
                cand_pos = _ranks_to_positions(cand_ranks)
                cand_score = discrete_score_arr(cand_pos)  # oracle used only to ACCEPT/REJECT
                refine_time += time.time() - t_ref
                n_refines += 1
                did_refine = True
                # Adopt ONLY if the oracle says it strictly beats the current best.
                if cand_score > best_score:
                    best_score = cand_score
                    cand_pos_t = torch.tensor(cand_pos, device=device)
                    best_positions = cand_pos_t.detach().clone()
                    # Re-seed the LIVE optimizer state from the improved ordering so Rocket
                    # continues from the refined point (anti-collapse: strictly increasing).
                    with torch.no_grad():
                        positions.copy_(cand_pos_t)
                    n_refines_accepted += 1
                    accepted = True

            elapsed = time.time() - start_time
            history.append(dict(
                iter=i, score=score, best_score=best_score,
                pct=pct(score, total_weight),
                best_pct=pct(best_score, total_weight),
                pure_best_score=pure_best_score,
                pure_best_pct=pct(pure_best_score, total_weight),
                beta=beta, elapsed=elapsed,
                neg_loss=float(-loss.item()),
                lr=optimizer.param_groups[0]["lr"],
                refined=did_refine, refine_accepted=accepted,
            ))

    wall = time.time() - start_time

    # Provenance for compute accounting (attached to the history DataFrame attrs).
    hist_df = pd.DataFrame(history)
    hist_df.attrs["n_refines"] = n_refines
    hist_df.attrs["n_refines_accepted"] = n_refines_accepted
    hist_df.attrs["refine_time_s"] = refine_time
    hist_df.attrs["pure_best_score"] = pure_best_score
    hist_df.attrs["pure_best_pct"] = pct(pure_best_score, total_weight)

    return RocketResult(
        best_positions=best_positions.detach().cpu().numpy(),
        best_score=best_score,
        best_pct=pct(best_score, total_weight),
        history=hist_df,
        n_epochs_done=last_i + 1,
        wall_clock_s=wall,
    )
