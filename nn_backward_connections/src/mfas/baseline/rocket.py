"""Rocket sub-algorithm — behavior-preserving port of the notebook's ``run_rocket``.

Ported verbatim (logic-for-logic) from ``create_notebook.py`` / Algorithm 1 of
Bader et al. (2025). The ONLY restructuring is packaging into ``RocketConfig`` /
``RocketResult`` dataclasses and routing the discrete score through the exact
:mod:`mfas.metrics` scorer. NO algorithmic / numerical changes.

Rocket optimizes continuous node positions to maximize a sigmoid surrogate of the
feedforward weight:

    loss = -Σ_(u,v)∈E  σ(β · (pos[v] - pos[u])) · ŵ(u,v)        ŵ = w / max(w)

with Adam, gradient clipping, a cyclic-cosine β schedule, and a constant→exponential
LR schedule. The discrete feedforward weight (the real objective) is tracked every
``log_interval`` epochs on CPU in int64/float64 (MPS index ops on float32 can return
garbage — see CLAUDE.md), and the best-scoring positions are retained.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from ..io import GraphData
from ..metrics import pct, score_from_positions

__all__ = [
    "RocketConfig",
    "RocketResult",
    "make_beta_schedule",
    "make_init_positions",
    "run_rocket",
]


# ──────────────────────────────────────────────────────────────────────────────
# Config / result
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RocketConfig:
    """Rocket hyperparameters (defaults = paper / notebook CONFIG)."""

    epochs: int = 20_000
    cycles: int = 5
    lr: float = 0.05
    grad_clip: float = 1.0
    lr_decay_start: float = 0.5     # start LR decay at 50% of training
    lr_end_factor: float = 0.1      # final LR = 10% of initial
    log_interval: int = 100
    init_mode: str = "random"       # random|uniform|degree_diff|degree_abs


@dataclass
class RocketResult:
    """Output of a Rocket run."""

    best_positions: np.ndarray      # float32, on CPU
    best_score: float               # exact (int64/float64) feedforward weight
    best_pct: float
    history: pd.DataFrame
    n_epochs_done: int
    wall_clock_s: float


# ──────────────────────────────────────────────────────────────────────────────
# Schedules & initialization
# ──────────────────────────────────────────────────────────────────────────────
def make_beta_schedule(num_epochs: int, n_cycles: int) -> np.ndarray:
    """Cyclic sigmoid-sharpness schedule (paper footnote 7).

    ``β = (cos(linspace(0, 2π·n_cycles, T)) + 1.1) / 2`` — range ``[0.05, 1.05]``.
    """
    return (np.cos(np.linspace(0, 2 * n_cycles * np.pi, num_epochs)) + 1.1) / 2


def make_init_positions(mode: str, g: GraphData, seed: int, device) -> torch.Tensor:
    """Initial position vector. Modes ported verbatim from the notebook.

    ``random`` (N(0,1), paper default) | ``uniform`` (U[-1,1]) |
    ``degree_diff`` (scaled out-in weight + noise) | ``degree_abs`` (rank by |diff|).
    """
    n = g.n_nodes
    src, tgt, weights = g.src, g.tgt, np.asarray(g.weight, dtype=np.float64)
    rng = np.random.RandomState(seed)

    if mode == "random":
        pos = rng.randn(n).astype(np.float32)
    elif mode == "uniform":
        pos = rng.uniform(-1, 1, n).astype(np.float32)
    elif mode == "degree_diff":
        out_w = np.zeros(n, dtype=np.float64)
        in_w = np.zeros(n, dtype=np.float64)
        np.add.at(out_w, src, weights)
        np.add.at(in_w, tgt, weights)
        diff = (out_w - in_w).astype(np.float32)
        m = np.abs(diff).max()
        pos = diff / (m + 1e-8)
        pos += rng.randn(n).astype(np.float32) * 0.01
    elif mode == "degree_abs":
        out_w = np.zeros(n, dtype=np.float64)
        in_w = np.zeros(n, dtype=np.float64)
        np.add.at(out_w, src, weights)
        np.add.at(in_w, tgt, weights)
        diff = (out_w - in_w).astype(np.float32)
        abs_rank = np.argsort(np.argsort(-np.abs(diff))).astype(np.float32)
        pos = abs_rank / n * 2 - 1
        pos += rng.randn(n).astype(np.float32) * 0.01
    else:
        raise ValueError(f"Unknown init mode: {mode}")
    return torch.tensor(pos, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# Main algorithm
# ──────────────────────────────────────────────────────────────────────────────
def run_rocket(g: GraphData, cfg: RocketConfig, seed: int, device,
               init_positions: Optional[torch.Tensor] = None,
               time_limit: Optional[float] = None,
               logger=None) -> RocketResult:
    """Run Rocket (Algorithm 1, Bader et al. 2025) on ``g``.

    Parameters
    ----------
    g : GraphData
        The graph (contiguous indices, native-dtype weights).
    cfg : RocketConfig
        Hyperparameters.
    seed : int
        Seeds torch + numpy at the start of the run.
    device : torch.device
        Where the optimization runs (discrete scoring is always on CPU).
    init_positions : torch.Tensor, optional
        Override the initial positions; otherwise built from ``cfg.init_mode``.
    time_limit : float, optional
        Stop after this many seconds regardless of ``cfg.epochs``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = g.n_nodes
    # Edge tensors on device (long indices; float32 weights for the surrogate loss).
    src_t = torch.tensor(np.asarray(g.src), dtype=torch.long, device=device)
    tgt_t = torch.tensor(np.asarray(g.tgt), dtype=torch.long, device=device)
    w_np = np.asarray(g.weight)
    max_w = float(w_np.max())
    nw_t = torch.tensor((w_np / max_w).astype(np.float32), device=device)

    # Numpy edge arrays for the exact CPU scorer.
    src_np, tgt_np = np.asarray(g.src), np.asarray(g.tgt)
    total_weight = g.total_weight

    def discrete_score(pos_param: torch.Tensor) -> float:
        # Always score on CPU in int64/float64 (MPS float32 index ops can corrupt).
        pos_cpu = pos_param.detach().cpu().numpy()
        return score_from_positions(pos_cpu, src_np, tgt_np, g.weight)

    # ── Initialise positions (Algorithm 1, line 1) ──────────────────────────
    if init_positions is not None:
        pos_data = init_positions.clone().detach().to(torch.float32)
    elif cfg.init_mode == "random":
        pos_data = torch.randn(n, device=device)
    else:
        pos_data = make_init_positions(cfg.init_mode, g, seed, device).to(torch.float32)
    positions = torch.nn.Parameter(pos_data)

    # ── Optimizer + LR schedule ─────────────────────────────────────────────
    optimizer = optim.Adam([positions], lr=cfg.lr)
    milestone = int(cfg.epochs * cfg.lr_decay_start)
    sched_const = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                total_iters=milestone)
    gamma = cfg.lr_end_factor ** (1.0 / max(cfg.epochs - milestone, 1))
    sched_exp = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched_const, sched_exp], milestones=[milestone])

    betas = make_beta_schedule(cfg.epochs, cfg.cycles)

    # ── Initial score (Algorithm 1, line 2) ──────────────────────────────────
    best_score = discrete_score(positions)
    best_positions = positions.detach().clone()

    history = []
    start_time = time.time()
    last_i = 0

    # ── Main loop (Algorithm 1, lines 4-16) ──────────────────────────────────
    for i in range(cfg.epochs):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        last_i = i

        beta = float(betas[i])

        optimizer.zero_grad()
        delta = positions[tgt_t] - positions[src_t]      # Δ = PT - PS
        sig = torch.sigmoid(beta * delta)                # Sig_β
        loss = -(sig * nw_t).sum()                       # -Σ Sig·ŵ

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
            if logger is not None:
                logger.info(f"  iter={i} best_pct={pct(best_score, total_weight):.4f} "
                            f"beta={beta:.3f} elapsed={elapsed:.1f}s")

    wall = time.time() - start_time
    return RocketResult(
        best_positions=best_positions.detach().cpu().numpy(),
        best_score=best_score,
        best_pct=pct(best_score, total_weight),
        history=pd.DataFrame(history),
        n_epochs_done=last_i + 1,
        wall_clock_s=wall,
    )
