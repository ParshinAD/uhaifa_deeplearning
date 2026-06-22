"""Variant H03 — Sharper / extended beta schedule (final monotone-sharpening ramp).

Hypothesis (backlog H03): the sigmoid surrogate sigma_beta only approximates the discrete
feedforward indicator. The baseline cyclic beta schedule (paper footnote 7) sweeps
``beta = (cos(linspace(0, 2*pi*n_cycles, T)) + 1.1) / 2`` over the range ``[0.05, 1.05]``,
so even at its peak a unit position gap maps to sigma(1.05) ~ 0.74 -- many "weakly correct"
edges and near-ties contribute little gradient and the surrogate stays fuzzy. Annealing beta
upward at the END tightens the surrogate->discrete gap (deterministic-annealing practice) and
should retain more high-weight feedforward arcs.

CHOSEN ARM (one, for a bounded screen)
--------------------------------------
PRIMARY ARM: keep the baseline cyclic exploration intact for the first (1 - RAMP_FRAC) of
training, then APPEND a final monotone-sharpening ramp over the last RAMP_FRAC of epochs that
raises beta from its end-of-cyclic value linearly up to BETA_MAX_TERMINAL = 4.0.

Why this arm (vs simply scaling the cyclic amplitude to beta_max all the way through):
  - It preserves the paper's broad cyclic exploration (which the paper credits for dodging
    manual tuning and finding good regions) and only changes the EXPLOITATION tail -- the
    place where a sharper surrogate matters and where over-sharp gradients early would just
    freeze a poor basin.
  - A monotone terminal ramp is the textbook deterministic-/sigmoid-annealing move and is the
    most conservative way to test "raise terminal sharpness" without destabilising exploration.
  - beta_max = 4 gives sigma(4) ~ 0.982 for a unit gap (vs 0.74 at 1.05): a clearly sharper but
    not gradient-killing surrogate. (beta=8 -> sigma ~ 0.9997 risks vanishing gradients.)

UN-RUN SWEEP ARMS (left for a follow-up if this arm is promising): BETA_MAX_TERMINAL in {2, 8},
and varying RAMP_FRAC. Only the single beta_max=4 / ramp_frac=0.25 arm is screened here to keep
screen compute bounded (one variant module, 2 datasets x 3 seeds).

Schedule construction (EXACT)
-----------------------------
1. Build the baseline cyclic schedule over the FIRST ``n_explore = round((1-RAMP_FRAC)*T)``
   epochs using the unchanged ``make_beta_schedule`` building block (same cos formula, same
   ``cycles``), so early training is byte-for-byte the baseline exploration.
2. Over the remaining ``n_ramp = T - n_explore`` epochs, ramp beta LINEARLY from the last
   cyclic value ``betas_explore[-1]`` up to ``BETA_MAX_TERMINAL``.
3. Concatenate -> a length-T beta array. Everything else (loss, Adam, grad-clip, LR schedule,
   init, best-by-oracle tracking, CPU discrete scoring) is IDENTICAL to ``run_rocket``.

Compute-matched & leakage-safe
------------------------------
Standard knob-swap (beta schedule only): SAME epoch budget as baseline (connectome 20k,
mouse 5k), SAME number of optimizer steps. ``n_epochs_done`` = actual steps performed, so the
frozen baseline / ``baseline_passthrough`` at matched seeds is the equal-budget comparator.
beta is purely a loss-shape parameter: the discrete oracle is read ONLY for the baseline's
existing best-by-oracle tracking; it is never folded into the loss, hardcoded, or used to
special-case a dataset. No post-processing / local search -> PURE Rocket score.

Implementation note (required by the task): ``run_rocket`` builds its beta schedule
internally and exposes no hook to override it, so to change ONLY the beta schedule this module
REPLICATES the ``run_rocket`` main loop verbatim (same optimizer, LR schedule, clipping, loss,
scoring, best-tracking, history, time-limit handling) and substitutes the H03 beta array. The
diff vs baseline is confined to ``_make_h03_beta_schedule`` and the line that selects ``betas``.
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

ID = "H03"
HYPOTHESIS = (
    "Appending a final monotone-sharpening beta ramp (beta_max=4 over the last 25% of "
    "epochs, baseline cyclic exploration kept for the first 75%) tightens the "
    "surrogate->discrete gap and yields higher exact feedforward weight than the "
    "baseline [0.05,1.05] cyclic schedule."
)

# Per-dataset epoch budget (= baseline single-run budget; standard knob-swap).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}

# H03 primary-arm hyperparameters (the ONLY change vs baseline).
BETA_MAX_TERMINAL = 4.0   # terminal sharpness at end of ramp (baseline peak ~1.05)
RAMP_FRAC = 0.25          # fraction of epochs given to the final monotone ramp


def _make_h03_beta_schedule(num_epochs: int, n_cycles: int) -> np.ndarray:
    """Baseline cyclic schedule for the first (1-RAMP_FRAC), then a linear ramp to BETA_MAX.

    The exploration segment reuses the UNCHANGED ``make_beta_schedule`` (identical cos formula
    and cycle count) so early training matches the baseline. The final segment ramps beta
    linearly from the last explore value up to ``BETA_MAX_TERMINAL``. This is the only
    algorithmic difference from baseline.
    """
    n_explore = int(round((1.0 - RAMP_FRAC) * num_epochs))
    n_explore = max(1, min(n_explore, num_epochs))
    n_ramp = num_epochs - n_explore

    betas_explore = make_beta_schedule(n_explore, n_cycles)
    if n_ramp <= 0:
        return betas_explore

    start = float(betas_explore[-1])
    # Linear ramp start -> BETA_MAX_TERMINAL over the final segment (monotone, sharpening).
    betas_ramp = np.linspace(start, BETA_MAX_TERMINAL, n_ramp, dtype=betas_explore.dtype)
    return np.concatenate([betas_explore, betas_ramp])


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run Rocket with the H03 beta schedule; everything else IDENTICAL to ``run_rocket``.

    This is a verbatim copy of ``baseline.rocket.run_rocket`` with the SINGLE change that the
    beta schedule comes from ``_make_h03_beta_schedule`` instead of ``make_beta_schedule``.
    The loss, Adam optimizer, gradient clipping, constant->exponential LR schedule, init,
    CPU discrete scoring, best-by-oracle tracking, history logging and time-limit handling are
    unchanged. ``n_epochs_done`` = actual optimizer steps (matched-budget basis).
    """
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

    # ── THE ONLY CHANGE vs baseline: H03 beta schedule ───────────────────────
    betas = _make_h03_beta_schedule(cfg.epochs, cfg.cycles)

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
        sig = torch.sigmoid(beta * delta)
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
