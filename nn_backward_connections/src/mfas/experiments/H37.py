"""Variant H37 — polynomial-tail surrogate at MATCHED CORE WIDTH (isolates the tail axis).

Hypothesis (backlog A-SURR -> H37): replacing Rocket's sigmoid with a surrogate whose core
has the SAME transition width but whose tail decays ALGEBRAICALLY instead of exponentially
raises the exact feedforward metric, because the long-range pairs that the sigmoid's
exponential tail sends to (numerically) zero force keep exerting a correctly-signed pull.

The surrogate
-------------
    g_q(z) = 1/2 + 1/2 * sign(z) * (1 - (1+|z|)^-q),      z = beta * Delta / SCALE

with ``q = 4`` and ``SCALE = 4.4361``. Monotone non-decreasing in ``Delta`` (argmax-preserving:
still "maximize feedforward weight"), bounded in (0,1), ``g(0) = 1/2``, and C^1 at 0.

``SCALE`` is the ONLY tuned constant and it is not free: it is fixed so this shape's
transition width (the ``z`` at which ``g`` reaches 0.9) equals the sigmoid's **exactly**
(2.1973 / 0.4954 = 4.4361, measured in ``experiments/outputs/q04_surrogate_tails.json``).
That matching is what makes this variant a test of the TAIL and not of the core sharpness —
core sharpness is reachable by simply raising ``beta``, i.e. the already-KILLED H03 /
A-SCALE axis (Q01: ``F`` depends only on the product ``beta*std``).

Why the naive readings of "use a slower-decaying tanh" are NOT this variant
--------------------------------------------------------------------------
* ``(tanh(z/2)+1)/2 == sigmoid(z)`` exactly (verified to 2.2e-16 over 2e5 points in
  `q04_surrogate_tails.json`), so a tanh surrogate is the sigmoid, full stop.
* ``tanh(z/a)`` is therefore ``sigmoid(2z/a)`` — a pure ``beta`` rescaling, which by Q01's
  ``beta*std`` law moves along an axis H03 already killed (and measurably makes the
  surrogate's ranking WORSE: alignment ratio -1.90 at ``a=10`` vs -0.59 for the sigmoid).
* A HARD flat top/bottom (clamping g to exactly {0,1}) is the H11 form, already KILLED;
  Q04 shows why — it freezes 65.8% of the nodes' gradients at the operating point.

Only the tail EXPONENT is a genuinely new degree of freedom, and this variant isolates it.

Leakage-safety / compute accounting
-----------------------------------
``g`` is a function of the model positions and ``beta`` only, weighted by the same
``hat_w = w/max(w)`` as the baseline; it never reads, hardcodes or folds in the discrete
oracle, and never special-cases a dataset. The frozen oracle is consulted ONLY for the
baseline's existing best-by-oracle tracking. Standard knob-swap: SAME epoch budget, optimizer,
init, grad-clip, LR and beta schedules as ``baseline_passthrough``, so that is the
equal-budget comparator at matched seeds. Added per-step cost is one abs/pow over the edge
tensor. PURE Rocket score, no post-processing.
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

ID = "H37"
HYPOTHESIS = (
    "Replacing the sigmoid with an ALGEBRAICALLY-tailed surrogate "
    "g(z)=0.5+0.5*sign(z)*(1-(1+|z|)^-4) evaluated at z=beta*Delta/4.4361 -- whose core "
    "transition width is matched EXACTLY to the sigmoid's (2.1973), so only the TAIL differs "
    "(polynomial z^-4 instead of exponential e^-|z|) -- raises the exact feedforward metric, "
    "because long-range pairs keep a correctly-signed non-vanishing force instead of being "
    "sent to numerically zero gradient. Monotone (argmax-preserving), bounded, leakage-safe, "
    "same epoch budget as baseline_passthrough. Isolates the tail axis from core sharpness "
    "(= beta = the killed H03/A-SCALE axis)."
)

# Per-dataset epoch budget (= baseline single-run budget; standard knob-swap).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}

# Tail exponent and the width-matching scale (the ONLY constants vs baseline).
# SCALE = width(sigmoid)/width(poly_q4) = 2.1973/0.4954, measured in q04_surrogate_tails.json.
Q = 4.0
SCALE = 4.4361


def _poly_surrogate(z: torch.Tensor) -> torch.Tensor:
    """``g(z) = 1/2 + 1/2 sign(z) (1 - (1+|z|)^-Q)`` — monotone, bounded, algebraic tail.

    Autograd gives ``dg/dz = (Q/2)(1+|z|)^-(Q+1)`` because ``d|z|/dz = sign(z)`` and
    ``sign(z)^2 = 1``; verified against central differences in
    ``experiments/proto_h37_tails.py::check_grads`` (max err 1.9e-11).
    """
    a = z.abs()
    return 0.5 + 0.5 * torch.sign(z) * (1.0 - (1.0 + a) ** (-Q))


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run Rocket with the H37 polynomial-tail surrogate; everything else IDENTICAL to baseline.

    Verbatim copy of ``baseline.rocket.run_rocket`` with the SINGLE change that the per-edge
    surrogate ``sig = sigmoid(beta*delta)`` becomes ``_poly_surrogate(beta*delta/SCALE)``.
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

    # ── Initialise positions — IDENTICAL to baseline ─────────────────────────
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

        # ── THE ONLY CHANGE vs baseline: algebraic-tail surrogate at matched core width ─
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
