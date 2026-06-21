"""Variant H19 — Soft-rank (rank-space) surrogate Rocket — PROTOTYPE (mouse + synthetic).

Hypothesis (backlog H19 / DIRECTION R): the Rocket↔best gap is an OPTIMIZATION-GAP whose
named mechanism is the position-scale BLOW-UP — Rocket's converged ``pos_std ≈ 142``
saturates the sigmoid, so a unit reorder barely moves the surrogate and the optimizer
drifts off good orders (diagnosis Step-1 + drift probe). Optimizing in RANK space —
bounded, scale-free — should keep gradients alive on borderline edges and let the
optimizer hold a better basin.

Mechanism
---------
Differentiable soft ranks of the learnable positions ``p``::

    r_i = Σ_j σ(α · (p_j − p_i))          # O(n²) all-pairs soft rank (Blondel-style)
    r̂  = r / (n − 1)                       # normalize to [0, 1] (scale-free, bounded)

then the SAME feedforward surrogate, but on rank gaps::

    loss = −Σ_(u,v)∈E  ŵ(u,v) · σ(β · (r̂_v − r̂_u))            ŵ = w / max(w)

Because soft-rank is bounded in [0, n−1] and monotone in ``p``, the scale cannot blow up
and σ cannot globally saturate; adjacent swaps always carry O(1) rank-gap gradient.

Equal compute / leakage-safety
------------------------------
Same Adam / LR schedule / epoch budget and the H02 greedy-FAS warm-start init as the
baseline comparison; only the surrogate's *parametrisation* (rank space) changes. The
loss is a function of positions + input weights ONLY — no oracle, no ``best_solution``.
Best-by-oracle tracking is used exactly as the baseline already uses it.

PROTOTYPE SCOPE (HARD GUARD)
----------------------------
The all-pairs soft rank is O(n²) per step → infeasible at the 136k-node connectome
(≈1.9e10 entries/step). ``run`` REFUSES any graph with ``n_nodes`` above
``_MAX_PROTOTYPE_NODES``. This module is a prototype gate for mouse + a hard synthetic;
scaling to connectome would require an O(n log n) soft-rank (torchsort / fast-soft-sort,
NOT installed in env ``allen``).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from ..baseline.rocket import (RocketConfig, RocketResult, make_beta_schedule)
from ..io import GraphData
from ..metrics import pct, score_from_positions

ID = "H19"
HYPOTHESIS = (
    "Optimizing the Rocket feedforward surrogate in differentiable soft-RANK space "
    "(bounded, scale-free r_i = Σ_j σ(α·(p_j−p_i)), then loss on rank gaps) instead of "
    "raw-position space prevents the scale blow-up that saturates the sigmoid, keeps "
    "gradients alive on borderline edges, and reaches/holds a better basin than "
    "raw-position Rocket. Prototype-scale (mouse + hard synthetic); O(n^2) soft-rank."
)

# Per-dataset / per-graph epoch budget (mouse = baseline 5k; synthetic budget set by caller).
_EPOCHS = {"mouse": 5_000, "hard_synthetic": 4_000, "synthetic": 4_000}
_MAX_PROTOTYPE_NODES = 5_000   # O(n^2) soft-rank guard; connectome (136k) is refused.

# Soft-rank sharpness α: larger = closer to a hard rank but stiffer gradients. The diagnosis
# motivates a scale-free rank, so α is set on the standardized positions (unit-ish spread).
_ALPHA = 4.0


def _greedy_fas_order(g: GraphData) -> np.ndarray:
    """Leakage-safe greedy-FAS warm-start order (delegates to H02's implementation)."""
    from .H02 import greedy_fas_order
    return greedy_fas_order(g)


def _init_positions(g: GraphData, device) -> torch.Tensor:
    """H02 init: greedy-FAS rank → evenly-spaced standardized positions in [-1, 1]."""
    order = _greedy_fas_order(g)
    n = order.shape[0]
    pos = ((order.astype(np.float32) / max(n - 1, 1)) * 2.0 - 1.0).astype(np.float32)
    pos = (pos - pos.mean()) / (pos.std() + 1e-12)        # standardize (scale-free start)
    return torch.tensor(pos, device=device)


def _soft_rank(p: torch.Tensor, alpha: float) -> torch.Tensor:
    """Differentiable all-pairs soft rank: r_i = Σ_j σ(α·(p_j − p_i)). O(n²)."""
    # diff[i, j] = p[j] - p[i]; rank of i = how many j sit "after" i (larger p).
    diff = p.unsqueeze(0) - p.unsqueeze(1)                # [n, n], diff[i,j]=p[j]-p[i]
    # number of j with p_j > p_i  ≈ Σ_j σ(α·(p_j − p_i)); subtract the self term σ(0)=0.5.
    r = torch.sigmoid(alpha * diff).sum(dim=1) - 0.5
    return r                                              # ∈ [0, n−1], higher = later (sink side)


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run rank-space (soft-rank) Rocket on ``g`` (PROTOTYPE: mouse / hard synthetic only).

    Refuses graphs above ``_MAX_PROTOTYPE_NODES`` (O(n²) soft-rank). Equal compute with
    baseline: same Adam/LR/β schedule + epochs, H02 warm-start; only the surrogate's
    parametrisation (soft-rank instead of raw positions) differs.
    """
    if g.n_nodes > _MAX_PROTOTYPE_NODES:
        raise ValueError(
            f"H19 is a PROTOTYPE: O(n^2) soft-rank refuses n_nodes={g.n_nodes:,} "
            f"(> {_MAX_PROTOTYPE_NODES:,}). Connectome needs an O(n log n) soft-rank "
            "(torchsort/fast-soft-sort, not installed).")

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))
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

    positions = torch.nn.Parameter(_init_positions(g, device).to(torch.float32))

    optimizer = optim.Adam([positions], lr=cfg.lr)
    milestone = int(cfg.epochs * cfg.lr_decay_start)
    sched_const = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                total_iters=milestone)
    gamma = cfg.lr_end_factor ** (1.0 / max(cfg.epochs - milestone, 1))
    sched_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched_const, sched_exp], milestones=[milestone])

    betas = make_beta_schedule(cfg.epochs, cfg.cycles)
    inv_n1 = 1.0 / max(n - 1, 1)

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
        r = _soft_rank(positions, _ALPHA) * inv_n1        # normalized soft rank ∈ [0, 1]
        delta = r[tgt_t] - r[src_t]                       # rank gap (feedforward wants > 0)
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
