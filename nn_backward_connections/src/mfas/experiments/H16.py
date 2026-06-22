"""Variant H16 — Monotone (non-cyclic) beta-continuation from the H02 greedy warm-start.

Hypothesis (backlog H16, DIRECTION O — optimization). The Phase-4 drift probe showed that
Rocket's CYCLIC beta schedule -- which repeatedly re-melts ``beta`` back to 0.05 -- destroys a
good ordering: started AT the 84.6% solution, baseline Rocket collapses to ~82.9% (and dips to
~78% during each low-beta phase). The decisive surrogate experiment showed the gap is an
OPTIMIZATION-GAP (best out-surrogates Rocket at every beta), not surrogate-misalignment. So the
schedule's re-melting, not the surrogate, is a plausible cause of basin loss. H16 replaces the
cyclic schedule with a single MONOTONE beta rise (same [0.05, 1.05] endpoints as the baseline
range, PATH only changed: never decreasing) and starts from H02's greedy-FAS warm-start, so the
optimizer commits to and sharpens a good basin instead of repeatedly re-melting it.

Distinct from killed H03 (do not conflate):
  * H03: RANDOM init, KEPT the baseline cyclic exploration for the first 75%, only APPENDED a
    terminal ramp raising beta_max to 4.0. It changed the *endpoint sharpness*, not the cycling.
  * H16: GREEDY warm-start (H02) + a fully MONOTONE schedule over ALL epochs (no cycling), SAME
    [0.05, 1.05] endpoints. It changes the *path* (removes re-melting), which the diagnosis
    directly implicates, and combines it with the only confirmed lever (warm-start).

Compute-matched & leakage-safe. Standard knob-swap (init + beta-schedule path only): SAME epoch
budget (connectome 20k, mouse 5k), SAME optimizer-step count, so ``n_epochs_done`` equals the
baseline budget and the equal-budget comparators are H02 (current best, for the stacking delta)
and ``baseline_passthrough`` (for the absolute-vs-baseline check). The greedy ordering uses ONLY
``g.src``/``g.tgt``/``g.weight`` (never ``data/best_solution`` or the discrete oracle in the
loss/init); the oracle is read only for the baseline's existing best-by-oracle tracking. No
post-processing / local search -> PURE Rocket score.

Implementation note: ``run_rocket`` builds its beta schedule internally and exposes no override
hook, so (following the H03 convention) this module REPLICATES the ``run_rocket`` main loop
verbatim and substitutes (a) the warm-start init and (b) the monotone beta array. Loss, Adam,
grad-clip, constant->exponential LR schedule, CPU discrete scoring, best-by-oracle tracking,
history and time-limit handling are byte-for-byte the baseline's.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from ..baseline.rocket import RocketConfig, RocketResult
from ..io import GraphData
from ..metrics import pct, score_from_positions
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H16"
HYPOTHESIS = (
    "Replacing Rocket's cyclic beta schedule (which re-melts beta->0.05 and destroys good "
    "orders) with a single MONOTONE beta rise over [0.05,1.05], started from the H02 greedy "
    "warm-start, lets the optimizer commit to and sharpen a better basin and beats H02 on the "
    "exact feedforward metric at equal compute."
)

_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}
BETA_MIN = 0.05
BETA_MAX = 1.05


def _monotone_beta_schedule(num_epochs: int) -> np.ndarray:
    """Monotone non-decreasing beta from BETA_MIN to BETA_MAX (no cycling).

    Same endpoints as the baseline cyclic range ``[0.05, 1.05]``; the ONLY change is the path
    (strictly non-decreasing instead of cosine-cycling), so re-melting never occurs. Linear is
    the most conservative monotone continuation; graduated-optimization theory only requires a
    slow, monotone sharpening.
    """
    return np.linspace(BETA_MIN, BETA_MAX, num_epochs, dtype=np.float64)


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run Rocket from the greedy warm-start with a monotone beta schedule.

    Verbatim copy of ``baseline.rocket.run_rocket`` except (1) ``init_positions`` is H02's
    greedy-FAS warm-start and (2) ``betas`` is the monotone schedule. Everything else identical.
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

    # ── (1) THE warm-start init (H02; graph-structure only, leakage-safe) ─────
    order = greedy_fas_order(g)
    init_positions = _init_positions_from_order(order, device)
    pos_data = init_positions.clone().detach().to(torch.float32)
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

    # ── (2) THE monotone beta schedule (no cycling) ──────────────────────────
    betas = _monotone_beta_schedule(cfg.epochs)

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
