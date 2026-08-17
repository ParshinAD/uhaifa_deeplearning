"""Variant H37B — polynomial-tail surrogate at its NATIVE (narrow) width: the arm the
static theory predicted would win on the connectome.

This is the second arm of the A-SURR / H37 cycle. It differs from :mod:`H37` in exactly one
constant: ``SCALE = 1.0`` instead of 4.4361, i.e. the surrogate keeps its own narrow core
(transition width 0.4954 in ``z = beta*Delta`` units) instead of being widened to match the
sigmoid's 2.1973.

Why this arm exists
-------------------
The Q04 theory gate (`experiments/outputs/q04_surrogate_tails.json`) measured, on the fly
connectome at Rocket's operating point ``beta*std = 148`` and with even spacing for every
order, the ALIGNMENT RATIO

    A(g) = (F_g(best_order) - F_g(rocket_order)) / (ceiling(best) - ceiling(rocket))

i.e. what fraction of the near-optimal order's TRUE +296.02 advantage the surrogate actually
sees. ``A > 0`` means the surrogate ranks the better order higher — the precondition for any
gradient step to point toward it. The result:

    sigmoid            (width 2.1973)   A = -0.591     16.1% of nodes have grad == 0
    sigmoid_sharp      (width 0.4954)   A = +0.132     42.5% of nodes have grad == 0
    THIS ARM  poly_q4  (width 0.4954)   A = +0.173      0.006% of nodes have grad == 0

So of every shape tested, **this arm is the only one that both ranks the near-optimal order
above Rocket's own order at Rocket's operating scale AND keeps essentially every node
mobile.** The exponentially-tailed shape can buy the same alignment only by freezing 42.5% of
the nodes (and that purchase is exactly the killed H03 / A-SCALE axis, since narrowing the
core is what raising ``beta`` does).

That makes this arm the direct empirical test of the static-alignment argument: if surrogate
misalignment is what caps Rocket (finding #3), the shape that fixes the misalignment should
score higher. H37 (matched width) instead isolates the tail with the core held fixed.

Honest note recorded BEFORE the large-graph run: this arm LOST at the cheap prototype gate
(`experiments/outputs/proto_h37_tails.json`: mouse -0.220 pp, hard synthetic -2.834 pp vs the
sigmoid, 3 seeds each). It is nevertheless run on the connectome because the alignment
measurement above is connectome-specific — mouse and the synthetic have no ``best_solution``,
so the property this arm is built to exploit was never measured on them.

Leakage-safety / compute accounting: identical to :mod:`H37` — the surrogate reads only the
model positions, ``beta`` and the input ``hat_w``; the frozen oracle is used only for the
baseline's existing best-by-oracle tracking; same epoch budget/optimizer/init/schedules as
``baseline_passthrough``, which is therefore the equal-budget comparator at matched seeds.
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

ID = "H37B"
HYPOTHESIS = (
    "Same algebraic-tail surrogate as H37 but at its NATIVE narrow core width "
    "(g(z)=0.5+0.5*sign(z)*(1-(1+|z|)^-4), z=beta*Delta, SCALE=1). This is the ONLY shape "
    "measured that ranks the near-optimal connectome order ABOVE Rocket's own order at "
    "Rocket's operating scale (alignment ratio +0.173 vs the sigmoid's -0.591) while leaving "
    "essentially every node mobile (0.006% zero gradients vs 42.5% for a width-matched "
    "sigmoid). Tests directly whether fixing the surrogate's static MISALIGNMENT (finding #3) "
    "raises the exact metric. Pre-registered caveat: this arm already lost the cheap prototype "
    "gate (mouse -0.220 pp, hard synthetic -2.834 pp); it is run on the connectome because the "
    "alignment property it exploits is connectome-specific and unmeasurable on those proxies."
)

# Per-dataset epoch budget (= baseline single-run budget; standard knob-swap).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}

# Tail exponent, and NO width matching (this is the difference vs H37).
Q = 4.0
SCALE = 1.0


def _poly_surrogate(z: torch.Tensor) -> torch.Tensor:
    """``g(z) = 1/2 + 1/2 sign(z) (1 - (1+|z|)^-Q)`` — monotone, bounded, algebraic tail.

    Autograd gives ``dg/dz = (Q/2)(1+|z|)^-(Q+1)``; verified against central differences in
    ``experiments/proto_h37_tails.py::check_grads`` (max err 1.9e-11).
    """
    a = z.abs()
    return 0.5 + 0.5 * torch.sign(z) * (1.0 - (1.0 + a) ** (-Q))


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run Rocket with the H37B narrow-core polynomial surrogate; else IDENTICAL to baseline."""
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

        # ── THE ONLY CHANGE vs baseline: algebraic-tail surrogate at NATIVE width ─
        sig = _poly_surrogate(beta * delta / SCALE)
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
