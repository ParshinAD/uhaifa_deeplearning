"""Variant H11 — Surrogate swap: margin-shaped (smooth-hinge) reward instead of saturating sigmoid.

Hypothesis (backlog H11): Replacing the saturating sigmoid surrogate sigma_beta(Delta) with a
MARGIN-SHAPED surrogate that KEEPS producing gradient for already-correct-but-small-margin edges
yields higher exact feedforward weight than sigma_beta. Rationale: sigma_beta saturates -- once an
edge is comfortably feedforward (Delta large & positive) its surrogate gradient -> 0, so the
optimizer stops WIDENING the margin that protects the discrete order against later cyclic-beta
reshuffling. A margin/hinge surrogate keeps pushing the margin up to a bounded target, a max-margin
intuition that targets exactly WHY Rocket plateaus (saturated gradients on correct-but-thin edges).

Baseline per-edge term (see ``baseline.rocket.run_rocket``)
----------------------------------------------------------
    loss = -SUM_(u,v)  sigma_beta(Delta) * hat_w ,   Delta = pos[v] - pos[u],  hat_w = w/max(w) in (0,1].
The per-edge contribution to -loss is ``sigma_beta(Delta) * hat_w``: monotone increasing in Delta,
weighted by the SAME max-normalized input weight ``hat_w``. ``beta`` (cyclic in [0.05,1.05]) scales
the margin/temperature: it sets how many position-units count as "one surrogate margin unit".

H11 surrogate (THE ONLY change vs baseline -- the per-edge SHAPE, NOT the weighting)
-----------------------------------------------------------------------------------
Replace ``sigma_beta(Delta)`` with a bounded SMOOTH-HINGE reward ``r_beta(Delta)`` of the
scaled margin ``z = beta * Delta`` (beta keeps EXACTLY its baseline ROLE: it scales Delta into the
surrogate's margin/temperature units). Define on the scaled margin ``z``:

    r(z) = 0.5 + z / (2*M)                     for |z| <= M       (LINEAR ramp: constant gradient)
    r(z) = sigmoid-tail blended to {0,1}        for |z| >  M       (bounded, smooth saturation)

Concretely, using a smooth (C^1) bounded hinge centered at z=0 with margin half-width ``M``:

    r(z) = clamp(0.5 + z/(2*M), 0, 1)           (the piecewise-linear "hard" smooth hinge)

This is the chosen PRIMARY arm. Its decisive property vs sigma_beta: INSIDE the band |z| <= M the
reward is LINEAR, so d r/d z = 1/(2M) is a NON-ZERO CONSTANT across the ENTIRE correct-but-thin band
-- the optimizer keeps widening the margin all the way out to z = M, instead of sigma_beta's gradient
``sigma*(1-sigma)`` which decays to ~0 almost immediately past z=0. OUTSIDE the band (|z| > M) the
reward is flat at 1 (fully feedforward) / 0 (fully feedback), which BOUNDS the reward and removes the
incentive to drive Delta to infinity -- this, plus the baseline grad-clip=1.0, is the anti-divergence
guard (a never-saturating pure hinge could blow up positions; the bounded plateau prevents that).
Margin half-width M is fixed (M = MARGIN below) in the SAME scaled-margin units beta already uses.

UN-RUN ARM (noted, not run this cycle): ``r(z) = (tanh(z)+1)/2``. Rejected as the primary arm
because tanh is just an affine reparametrization of the sigmoid (sigmoid(x) = (tanh(x/2)+1)/2) and
therefore saturates IDENTICALLY -- it would NOT keep gradient on correct-but-thin edges, so it does
not test the hypothesis's mechanism. The smooth-hinge is the surrogate with the stronger a-priori
case because its constant in-band gradient is precisely the "keep widening the margin" behavior the
hypothesis predicts.

Why this is MONOTONE and PRESERVES THE ARGMAX (objective direction)
-------------------------------------------------------------------
``r(z) = clamp(0.5 + z/(2M), 0, 1)`` is NON-DECREASING in z (slope 1/(2M) > 0 on the band, slope 0
on the flats) and ``z = beta*Delta`` with ``beta > 0`` is increasing in Delta, so ``r_beta(Delta)``
is MONOTONICALLY NON-DECREASING in ``Delta = pos[v]-pos[u]``: making an edge MORE feedforward never
decreases its reward and strictly increases it while |z| < M. Hence -loss = SUM r_beta(Delta)*hat_w
is, per edge, a monotone feedforward reward weighted by the same positive ``hat_w`` as baseline. The
per-edge gradient w.r.t. positions points toward ``pos[v] > pos[u]`` (feedforward) for every edge in
its active band and is zero (never inverted) outside it -- no edge is ever pushed toward feedback.
A weighted sum of monotone-in-Delta feedforward rewards still has its optimum at "maximize feedforward
weight"; the discrete oracle (exact total feedforward weight) is what is tracked best-by-oracle and
reported, exactly as baseline. Only the SHAPE of the per-edge reward changes (sigmoid S-curve ->
clamped linear ramp); the per-edge WEIGHT ``hat_w`` is IDENTICAL to baseline, which isolates H11
(surrogate shape) from H06 (per-edge reweighting).

Why this is LEAKAGE-SAFE / TARGET-BLIND
---------------------------------------
``r_beta`` is a function of ONLY (i) the model positions (via Delta) and (ii) ``beta`` (a loss-shape
schedule), weighted by ``hat_w = w/max(w)`` from the INPUT graph. It NEVER reads, hardcodes or folds
in the discrete oracle score; it never special-cases a dataset (same MARGIN and formula for connectome
and mouse). The frozen oracle is consulted ONLY for the baseline's existing best-by-oracle tracking.

Compute-matched
---------------
Standard knob-swap (surrogate SHAPE only): SAME epoch budget as baseline (connectome 20k, mouse 5k),
SAME optimizer-step count, SAME N(0,1) init / Adam / grad-clip=1.0 / constant->exponential LR / cyclic
beta schedule / CPU discrete scoring / best-by-oracle tracking. ``n_epochs_done`` = actual optimizer
steps, so ``baseline_passthrough`` / the frozen baseline at matched seeds is the equal-budget
comparator. Added per-step cost is a clamp+linear op over the edge tensor (negligible). PURE Rocket
score (no post-processing).

Implementation note (required): ``run_rocket`` builds the loss internally and exposes no surrogate
hook, so this module REPLICATES the ``run_rocket`` main loop VERBATIM (same init, Adam, grad-clip,
LR schedule, beta schedule, CPU discrete scoring, best-by-oracle tracking, history, time-limit
handling) and substitutes ONLY the per-edge reward ``sig = sigmoid(beta*delta)`` ->
``r = clamp(0.5 + (beta*delta)/(2*MARGIN), 0, 1)``. The diff vs baseline is confined to that line.
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

ID = "H11"
HYPOTHESIS = (
    "Replacing the saturating sigmoid surrogate sigma_beta(Delta) with a bounded SMOOTH-HINGE reward "
    "r_beta(Delta)=clamp(0.5 + (beta*Delta)/(2*MARGIN), 0, 1) -- LINEAR (constant non-zero gradient) "
    "across the correct-but-thin-margin band |beta*Delta|<=MARGIN, then flat/bounded outside -- keeps "
    "the optimizer WIDENING margins that sigma_beta abandons once saturated, yielding higher exact "
    "feedforward weight. Monotone non-decreasing in Delta (argmax-preserving: still maximizes "
    "feedforward weight); beta keeps its baseline role as the margin/temperature scale; per-edge weight "
    "hat_w UNCHANGED (isolates surrogate SHAPE from H06 reweighting). Leakage-safe: a function of "
    "positions + input weights + beta only, never the discrete oracle; bounded plateau + grad-clip guard "
    "against divergence. Primary arm = smooth-hinge; tanh arm un-run (saturates like sigmoid)."
)

# Per-dataset epoch budget (= baseline single-run budget; standard knob-swap).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000}

# H11 primary-arm hyperparameter (the ONLY new constant vs baseline): the smooth-hinge half-width in
# the SAME scaled-margin units beta produces (z = beta*Delta). MARGIN = 2.0 means the reward ramps
# linearly over z in [-2, +2] before saturating -- a markedly WIDER non-saturated band than sigma_beta
# (whose gradient sigma*(1-sigma) has already decayed to ~0.10 of its peak by z=+2), so correct edges
# keep receiving gradient to widen their margin out to z=2 instead of stalling near z=0. Bounded (the
# reward flats at {0,1} beyond +/-MARGIN) so positions cannot be driven to infinity.
MARGIN = 2.0


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run Rocket with the H11 smooth-hinge surrogate; everything else IDENTICAL to baseline.

    Verbatim copy of ``baseline.rocket.run_rocket`` with the SINGLE change that the per-edge surrogate
    ``sig = sigmoid(beta*delta)`` is replaced by the bounded smooth-hinge
    ``r = clamp(0.5 + (beta*delta)/(2*MARGIN), 0, 1)``. The per-edge weight ``nw_t`` (hat_w), init,
    Adam, grad-clip=1.0, LR schedule, beta schedule, CPU discrete scoring, best-by-oracle tracking,
    history and time-limit handling are unchanged. ``n_epochs_done`` = actual optimizer steps.
    """
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = g.n_nodes
    src_t = torch.tensor(np.asarray(g.src), dtype=torch.long, device=device)
    tgt_t = torch.tensor(np.asarray(g.tgt), dtype=torch.long, device=device)
    w_np = np.asarray(g.weight)
    max_w = float(w_np.max())
    nw_t = torch.tensor((w_np / max_w).astype(np.float32), device=device)  # hat_w in (0,1]

    src_np, tgt_np = np.asarray(g.src), np.asarray(g.tgt)
    total_weight = g.total_weight

    def discrete_score(pos_param: torch.Tensor) -> float:
        pos_cpu = pos_param.detach().cpu().numpy()
        return score_from_positions(pos_cpu, src_np, tgt_np, g.weight)

    # ── Initialise positions (Algorithm 1, line 1) — IDENTICAL to baseline ───
    if cfg.init_mode == "random":
        pos_data = torch.randn(n, device=device)
    else:
        pos_data = make_init_positions(cfg.init_mode, g, seed, device).to(torch.float32)
    positions = torch.nn.Parameter(pos_data)

    # ── Optimizer + LR schedule — IDENTICAL to baseline ──────────────────────
    optimizer = optim.Adam([positions], lr=cfg.lr)
    milestone = int(cfg.epochs * cfg.lr_decay_start)
    sched_const = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                total_iters=milestone)
    gamma = cfg.lr_end_factor ** (1.0 / max(cfg.epochs - milestone, 1))
    sched_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched_const, sched_exp], milestones=[milestone])

    # ── Beta schedule — IDENTICAL to baseline ────────────────────────────────
    betas = make_beta_schedule(cfg.epochs, cfg.cycles)

    best_score = discrete_score(positions)
    best_positions = positions.detach().clone()

    history = []
    start_time = time.time()
    last_i = 0

    inv_2m = 1.0 / (2.0 * MARGIN)
    for i in range(cfg.epochs):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        last_i = i

        beta = float(betas[i])

        optimizer.zero_grad()
        delta = positions[tgt_t] - positions[src_t]      # Delta = PT - PS

        # ── THE ONLY CHANGE vs baseline: smooth-hinge surrogate replaces sigmoid ─
        # r = clamp(0.5 + (beta*delta)/(2*MARGIN), 0, 1): monotone non-decreasing in delta, LINEAR
        # (constant gradient) over the band |beta*delta| <= MARGIN, bounded flat outside. Same hat_w
        # weighting as baseline -> only the surrogate SHAPE changes (see module docstring).
        r = torch.clamp(0.5 + (beta * delta) * inv_2m, 0.0, 1.0)
        loss = -(r * nw_t).sum()                          # -SUM r * hat_w

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
