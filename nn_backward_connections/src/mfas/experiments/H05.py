"""Variant H05 — Optimizer swap: AdamW (decoupled weight decay) instead of Adam.

Hypothesis (backlog H05): Replacing Adam with AdamW (a small DECOUPLED weight decay to keep the
unconstrained positions bounded / scale-regularized) changes the BASIN the optimizer reaches and
improves the final exact feedforward weight. Rationale: positions are unconstrained and can drift to
large magnitudes, where sigma_beta saturates and the surrogate gradient vanishes; a mild decoupled
decay regularizes position SCALE (a scale-invariance argument the paper itself makes for *weight*
normalization, sec 3.1.1). Optimizer choice is target-blind, hence leakage-safe.

FALSIFIER CONTEXT (why this cycle exists)
-----------------------------------------
The campaign's evidence so far -- H02 (warm-start) is the only confirmed win = a BASIN change; while
H01/H03/H04/H13 (multi-start / beta-tail / in-loop refinement / edge subsampling) are all KILLED
late-dynamics / optimization-trajectory interventions that re-converge to or below Rocket's plateau --
suggests the optimizer TRAJECTORY does not set the plateau; the starting BASIN does. H05 tests the
single dynamics knob most able to reach a DIFFERENT basin (the optimizer itself + its decoupled decay).
An honest negative strengthens that finding; a positive would overturn it.

THE ONLY change vs baseline (``baseline.rocket.run_rocket``)
-----------------------------------------------------------
    optimizer = optim.Adam([positions], lr=cfg.lr)            # baseline
    optimizer = optim.AdamW([positions], lr=cfg.lr, weight_decay=WEIGHT_DECAY)  # H05 primary arm

EVERYTHING else is IDENTICAL to baseline and the diff is confined to that single constructor line:
  - init = N(0,1) (cfg.init_mode == "random"), same RNG seeding;
  - betas = first/second moment coefficients = AdamW DEFAULTS = (0.9, 0.999), eps=1e-8 -- BIT-IDENTICAL
    to torch's Adam defaults, so the ONLY behavioural difference vs baseline is the decoupled
    weight-decay term (with WEIGHT_DECAY=0 AdamW reduces EXACTLY to Adam);
  - LR base = cfg.lr = 0.05 (UNCHANGED) and the ConstantLR(50%) -> ExponentialLR(->10%) SequentialLR
    schedule is byte-for-byte the baseline's;
  - grad-clip = clip_grad_norm_(1.0) UNCHANGED (applied before optimizer.step, exactly as baseline);
  - cyclic beta surrogate schedule make_beta_schedule(epochs, 5) UNCHANGED;
  - epoch budget = baseline single-run budget (connectome 20k, mouse 5k);
  - CPU int64/float64 discrete scoring + best-by-oracle tracking + history UNCHANGED.

Note on "beta" overload: in this module ``betas`` is the cyclic SURROGATE sharpness schedule (the
loss-shape parameter the paper calls beta), NOT the Adam (beta1, beta2) momentum coefficients -- those
are left at AdamW's defaults (0.9, 0.999), equal to baseline Adam's, so they are not a variable here.

Decoupled weight decay: AdamW applies ``theta <- theta - lr * wd * theta`` SEPARATELY from the
adaptive gradient step (decoupled from the moment estimates), i.e. an L2 pull of the positions toward
0 that does NOT enter the surrogate gradient. This is purely a scale regularizer on the positions; it
reads only the positions themselves, never the discrete oracle, never the input weights -- target-blind.

Arm choice / un-run arms
------------------------
PRIMARY (this screen): AdamW with WEIGHT_DECAY = 1e-4 -- a small decay sized so the per-step pull
``lr*wd = 0.05*1e-4 = 5e-6`` is a gentle scale prior that nudges positions toward 0 without dominating
the ~O(1) surrogate gradient. UN-RUN this cycle (noted to bound compute):
  - AdamW WEIGHT_DECAY = 1e-2 (a 100x stronger scale prior);
  - Lion (sign-based optimizer): NOT available in this environment -- torch.optim has no Lion and no
    Lion package is installed; per the hard constraint we do NOT add a dependency, and a hand-rolled
    Lion is left as an explicitly-labelled optional follow-up, not run here.
Keeping the screen to ONE primary arm (AdamW 1e-4) bounds compute, per the cycle instructions.

Compute-matched
---------------
Standard knob-swap (optimizer only): SAME epoch budget as baseline (connectome 20k / mouse 5k), SAME
optimizer-step count, SAME init / grad-clip / LR schedule / beta schedule / CPU discrete scoring /
best-by-oracle tracking. ``n_epochs_done`` = actual optimizer steps performed, so
``baseline_passthrough`` / the frozen baseline at matched seeds is the equal-budget comparator. PURE
Rocket score (no post-processing).

Implementation note (required): ``run_rocket`` constructs the optimizer internally and exposes no
optimizer hook, so this module REPLICATES the ``run_rocket`` main loop VERBATIM (same init, grad-clip,
LR schedule, beta schedule, CPU discrete scoring, best-by-oracle tracking, history, time-limit
handling) and substitutes ONLY the optimizer constructor ``optim.Adam(...)`` ->
``optim.AdamW(..., weight_decay=WEIGHT_DECAY)``. The diff vs baseline is confined to that one line.
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

ID = "H05"
HYPOTHESIS = (
    "Replacing Adam with AdamW (decoupled weight decay = 1e-4, a small SCALE prior that keeps the "
    "unconstrained positions bounded so sigma_beta does not saturate at large magnitudes) changes the "
    "basin the optimizer reaches and improves the final exact feedforward weight. The ONLY change vs "
    "baseline is the optimizer constructor optim.Adam -> optim.AdamW(weight_decay=1e-4); init, the "
    "(0.9,0.999) moment coefficients (AdamW defaults == baseline Adam's), LR base=0.05, the "
    "ConstantLR->ExponentialLR schedule, grad-clip=1.0, the cyclic-beta surrogate schedule and the "
    "20k/5k epoch budget are IDENTICAL to baseline. Decoupled decay reads only the positions, never the "
    "oracle or input weights -> target-blind / leakage-safe. FALSIFIER arm: tests whether the optimizer "
    "(a dynamics knob) can reach a different basin than Adam, against the campaign's basin-not-dynamics "
    "inference. Primary arm = AdamW 1e-4; un-run arms = AdamW 1e-2 and Lion (Lion not installed)."
)

# Per-dataset epoch budget (= baseline single-run budget; standard knob-swap).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000}

# H05 primary-arm hyperparameter (the ONLY new constant vs baseline): the AdamW DECOUPLED weight
# decay. 1e-4 is a small scale prior; the per-step decoupled pull toward 0 is lr*wd = 0.05*1e-4 =
# 5e-6 per coordinate, gentle relative to the ~O(1) surrogate gradient under grad-clip=1.0. With
# WEIGHT_DECAY = 0 this AdamW run reduces EXACTLY to the baseline Adam run.
WEIGHT_DECAY = 1e-4


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run Rocket with AdamW (decoupled weight decay); everything else IDENTICAL to baseline.

    Verbatim copy of ``baseline.rocket.run_rocket`` with the SINGLE change that the optimizer
    ``optim.Adam([positions], lr=cfg.lr)`` is replaced by
    ``optim.AdamW([positions], lr=cfg.lr, weight_decay=WEIGHT_DECAY)``. Init, the (0.9,0.999) moment
    coefficients (AdamW defaults, equal to Adam's), grad-clip=1.0, LR schedule, beta schedule, CPU
    discrete scoring, best-by-oracle tracking, history and time-limit handling are unchanged.
    ``n_epochs_done`` = actual optimizer steps.
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

    # ── Optimizer — THE ONLY CHANGE vs baseline: Adam -> AdamW(weight_decay) ──
    # Decoupled weight decay regularizes position SCALE (pull toward 0, separate from the adaptive
    # gradient step). (beta1, beta2)=(0.9,0.999), eps=1e-8 are AdamW defaults == baseline Adam defaults,
    # so weight_decay is the ONLY behavioural difference (weight_decay=0 reduces exactly to baseline).
    optimizer = optim.AdamW([positions], lr=cfg.lr, weight_decay=WEIGHT_DECAY)

    # ── LR schedule — IDENTICAL to baseline ──────────────────────────────────
    milestone = int(cfg.epochs * cfg.lr_decay_start)
    sched_const = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                total_iters=milestone)
    gamma = cfg.lr_end_factor ** (1.0 / max(cfg.epochs - milestone, 1))
    sched_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched_const, sched_exp], milestones=[milestone])

    # ── Beta (surrogate sharpness) schedule — IDENTICAL to baseline ──────────
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
        delta = positions[tgt_t] - positions[src_t]      # Delta = PT - PS
        sig = torch.sigmoid(beta * delta)                # Sig_beta — UNCHANGED vs baseline
        loss = -(sig * nw_t).sum()                       # -SUM Sig * hat_w

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
