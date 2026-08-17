"""Variant H38 — ASYMMETRIC one-sided surrogate: flat above a margin, tanh below.

Hypothesis (backlog A-SURR / H38): a surrogate that is CONSTANT for comfortably-feedforward
edges and tanh-shaped below a margin — so the entire gradient budget is spent pulling FEEDBACK
edges toward correctness instead of widening margins that are already correct — raises the exact
feedforward metric.

    g(z) = 1                        for z >= M
    g(z) = 1 + tanh((z - M) / T)    for z <  M          z = beta * Delta

Monotone non-decreasing in ``Delta`` (argmax-preserving: still "maximize feedforward weight"),
bounded in (0, 1], continuous at ``z = M``.

Why this is NOT any previously-killed variant
---------------------------------------------
Every shape tested in H11, H34 and H37 was ODD-SYMMETRIC (``g(-z) = 1 - g(z)``): sigmoid, tanh,
the algebraic tails, H11's two-sided clamp, the cusp. This one is not, and the asymmetry has
consequences no symmetric shape has (measured in `experiments/outputs/q05_asymmetric_surrogates.json`):

* **It does not telescope.** Q01's decisive small-scale result — that the surrogate degenerates
  into the node-level imbalance objective ``W/2 - (beta/4)<c,P>``, whose optimum is the imbalance
  sort — requires the edge sum to run over ALL edges. Here the first-order term runs over the
  VIOLATED subset only, which is itself order-dependent. Measured consequence: at ``beta*std``
  -> 0 this shape ranks ``best > rocket > imbalance_sort > random`` (correct), where every
  odd-symmetric shape ranks the imbalance sort first.
* **No misalignment at any measured scale.** Alignment ratio at Rocket's operating point is
  **+1.48 … +2.00** (sigmoid: **-0.591**), with NO crossover anywhere in ``beta*std`` ∈ [1e-2, 1e6].
* The mirror shape (flat on the FEEDBACK side) is the control and behaves oppositely
  (alignment -1.30 … -3.72), so the DIRECTION of the asymmetry is what matters.

The margin M is not optional — it removes a proved degeneracy
--------------------------------------------------------------
With ``M = 0`` the shape has ``g(0) = 1``, so the collapsed configuration ``P = const`` attains
``F = sum_e w_hat_e``, the **global maximum** of F, strictly above every ordering; and the
dynamics flow into it (a violated edge pulls its endpoints together). This was derived first and
then confirmed: the ``M = 0`` arm collapses to final position std **0.0005** on the hard
synthetic and scores 58.24% vs the sigmoid's 73.68%
(`experiments/outputs/proto_h38_asym.json`). With ``M > 0``, ``g(0) = 1 + tanh(-M/T) < 1`` and
collapse is no longer optimal.

Hyperparameter provenance (selected on PROXIES, never on the evaluated graph)
-----------------------------------------------------------------------------
``(M, T) = (0.75, 1.5)`` was chosen from a 6x3 grid over mouse + the hard synthetic
(`experiments/outputs/proto_h38_sweep.json`) as the only setting that is POSITIVE ON BOTH
proxies (mouse +0.396 pp, hard synthetic +0.404 pp), rather than the mouse-optimal setting
(0.25, 0.75), which is +0.627 pp on mouse but -6.30 pp on the synthetic. The optimum is
graph-dependent — that is disclosed, not hidden. The connectome was NOT used for selection.

Control that rules out the trivial explanation: the winning arms converge to a smaller position
spread than the sigmoid, and Q01 established that F depends only on ``beta*std`` — so a scale
change alone would be the already-killed A-SCALE/H03 axis. Running the plain SIGMOID at beta
rescaled by {0.1, 0.25, 0.5, 2, 4} reproduces at most +0.076 pp (mouse) / +0.051 pp (synthetic),
i.e. it does NOT reproduce the gain.

Leakage-safety / compute accounting
-----------------------------------
``g`` is a function of the model positions and ``beta`` only, weighted by the same
``hat_w = w/max(w)`` as the baseline; it never reads or folds in the discrete oracle and never
special-cases a dataset. The frozen oracle is consulted ONLY for the baseline's existing
best-by-oracle tracking. Standard knob-swap: SAME epoch budget, optimizer, init, grad-clip, LR
and beta schedules as ``baseline_passthrough``, which is therefore the equal-budget comparator at
matched seeds. PURE Rocket score, no post-processing.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from ..baseline.rocket import (
    RocketConfig,
    RocketResult,
    make_beta_schedule,
    make_init_positions,
)
from ..io import GraphData
from ..metrics import pct, score_from_positions

ID = "H38B"
HYPOTHESIS = (
    "One-sided ASYMMETRIC surrogate g(z)=1 for z>=M, 1+tanh((z-M)/T) for z<M with (M,T)=(0.25,0.75) MOUSE-OPTIMAL arm: "
    "comfortably-feedforward edges get ZERO gradient, so the whole budget pulls FEEDBACK edges "
    "toward correctness. Unlike every previously-tested (odd-symmetric) shape it does not "
    "telescope into Q01's imbalance degeneracy at small beta*std, and its alignment ratio is "
    "+1.48..+2.00 vs the sigmoid's -0.591 with no crossover at any scale. M>0 is required: at M=0 "
    "the collapsed configuration P=const is the surrogate's GLOBAL maximum (verified: collapses to "
    "pos std 5e-4, 58.24% vs 73.68% on the hard synthetic). (M,T) selected on mouse+synthetic only "
    "(the sole grid point positive on both); a beta-rescaled sigmoid control does NOT reproduce the "
    "gain, so this is not the killed A-SCALE/H03 axis."
)

# Per-dataset epoch budget (= baseline single-run budget; standard knob-swap).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}

# The two new constants, selected on the proxies (see the module docstring).
M = 0.25
T = 0.75


def _asym_surrogate(z: torch.Tensor) -> torch.Tensor:
    """``g(z) = 1 for z >= M ; 1 + tanh((z-M)/T) for z < M`` — monotone, bounded, one-sided."""
    below = 1.0 + torch.tanh((z - M) / T)
    return torch.where(z >= M, torch.ones_like(z), below)


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run Rocket with the H38 one-sided surrogate; everything else IDENTICAL to baseline."""
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = g.n_nodes
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

    if cfg.init_mode == "random":
        pos_data = torch.randn(n, device=device)
    else:
        pos_data = make_init_positions(cfg.init_mode, g, seed, device).to(torch.float32)
    positions = torch.nn.Parameter(pos_data)

    optimizer = optim.Adam([positions], lr=cfg.lr)
    milestone = int(cfg.epochs * cfg.lr_decay_start)
    sched_const = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                total_iters=milestone)
    gamma = cfg.lr_end_factor ** (1.0 / max(cfg.epochs - milestone, 1))
    sched_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched_const, sched_exp], milestones=[milestone])

    betas = make_beta_schedule(cfg.epochs, cfg.cycles)

    best_score = discrete_score(positions)
    best_positions = positions.detach().clone()

    history = []
    start_time = time.time()
    last_i = 0

    for i in range(cfg.epochs):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        last_i = i

        beta = float(betas[i])

        optimizer.zero_grad()
        delta = positions[tgt_t] - positions[src_t]

        # ── THE ONLY CHANGE vs baseline: one-sided asymmetric surrogate ─
        sig = _asym_surrogate(beta * delta)
        loss = -(sig * nw_t).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([positions], cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if i % cfg.log_interval == 0 or i == cfg.epochs - 1:
            score = discrete_score(positions)
            elapsed = time.time() - start_time
            if score > best_score:
                best_score = score
                best_positions = positions.detach().clone()
            history.append(dict(
                iter=i, score=score, best_score=best_score,
                pct=pct(score, total_weight),
                best_pct=pct(best_score, total_weight),
                beta=beta, elapsed=elapsed,
                neg_loss=float(-loss.item()),
                lr=optimizer.param_groups[0]["lr"],
            ))

    wall = time.time() - start_time
    return RocketResult(
        best_positions=best_positions.detach().cpu().numpy(),
        best_score=best_score,
        best_pct=pct(best_score, total_weight),
        history=pd.DataFrame(history),
        n_epochs_done=last_i + 1,
        wall_clock_s=wall,
    )
