"""Variant H06 — Weight-aware loss reweighting (focus gradient on heavy & borderline edges).

Hypothesis (backlog H06): Reweighting the surrogate so that high-weight AND currently-borderline
edges receive proportionally MORE gradient (vs the flat max-normalized weight hat-w) increases
retained high-weight feedforward arcs. Both connectomes have extremely skewed edge-weight
distributions (paper Fig. 2: connectome max 2,405 vs mostly tiny). Max-normalization keeps the
objective faithful but means the bulk of (tiny) edges contribute near-equal small gradients while
the few decisive heavy edges are diluted late in training, and saturated edges (sigma -> 0 or 1)
keep getting gradient they no longer need. Emphasizing heavy/uncertain edges aligns gradient effort
with the metric's own weighting.

Baseline per-edge loss term (see ``baseline.rocket.run_rocket``)
---------------------------------------------------------------
    loss = -SUM_(u,v)  sigma_beta(Delta) * hat_w        with hat_w = w / max(w) in (0, 1],
                                                              Delta  = pos[v] - pos[u].
The per-edge contribution to -loss is ``sigma_beta(Delta) * hat_w``: monotone increasing in
``Delta`` (so reducing loss == making the edge more feedforward), weighted by ``hat_w``.

H06 reweighting (THE ONLY change vs baseline)
---------------------------------------------
Multiply each per-edge term by a STRICTLY POSITIVE, BOUNDED, DETACHED emphasis factor m_e:

    loss = -SUM_(u,v)  m_e * sigma_beta(Delta) * hat_w

    m_e = 1 + ALPHA * hat_w * b_e ,    b_e = 4 * sigma_beta(Delta) * (1 - sigma_beta(Delta))

where:
  * ``hat_w in (0,1]`` is the SAME max-normalized input weight the baseline already uses
    (heavy-edge emphasis term),
  * ``b_e = 4*sigma*(1-sigma) in [0,1]`` is the normalized sigmoid sensitivity: it is 1 exactly
    when the edge is BORDERLINE (sigma = 0.5, i.e. ``|sigma - 0.5| = 0``) and decays to 0 as the
    edge saturates feedforward (sigma -> 1) OR saturates feedback (sigma -> 0). This is the
    target-blind "currently-borderline" emphasis, expressed via the model's own surrogate state
    (NOT the discrete oracle).
  * ``b_e`` is computed under ``torch.no_grad()`` and DETACHED, so it acts as a fixed per-edge
    gradient-magnitude scale at each step, not a differentiable term (see direction argument).
  * ``ALPHA >= 0`` bounds the extra emphasis. With ``hat_w <= 1`` and ``b_e <= 1`` the factor is
    bounded ``1 <= m_e <= 1 + ALPHA`` for EVERY edge.

This is the chosen PRIMARY arm: a clean COMBINATION of the backlog's two ablations
((a) heavy-edge ``hat_w`` emphasis and (b) borderline ``|sigma-0.5|`` emphasis), because the
hypothesis is specifically about edges that are heavy AND uncertain -- a heavy edge that is already
firmly feedforward needs no extra push, and a borderline tiny edge barely moves the metric. The
product ``hat_w * b_e`` routes the EXTRA gradient to exactly the heavy-and-undecided edges.

UN-RUN ARMS (left for a follow-up if this screens positive): (a) heavy-only ``m_e = (hat_w)^(gamma-1)``
power emphasis, (b) borderline-only ``m_e = 1 + ALPHA * b_e`` (no ``hat_w``), and an ALPHA sweep.
Only the single combined arm (ALPHA below) is screened here to keep screen compute bounded
(one module, 2 datasets x 3 seeds).

Why this PRESERVES THE OBJECTIVE'S DIRECTION (argmax-preservation argument)
--------------------------------------------------------------------------
The hard constraint is that the loss must still monotonically reward making edges feedforward and
still target the TRUE metric (total feedforward weight) -- the reweighting must not move the
optimum away from "maximize feedforward weight".

1. PER-EDGE SIGN IS NEVER INVERTED. Because ``b_e`` is DETACHED (a no-grad scalar at each step),
   the gradient of the H06 loss w.r.t. positions is, per edge,
        -m_e * hat_w * d/dpos[ sigma_beta(Delta) ]
   i.e. EXACTLY the baseline per-edge gradient scaled by the strictly positive factor ``m_e > 0``.
   Every edge therefore still pushes its endpoints in the SAME direction as the baseline (toward
   ``pos[v] > pos[u]``, i.e. toward feedforward). No edge's contribution sign is flipped; the
   reweighting only changes the RELATIVE magnitude of gradient effort across edges. A per-edge
   positive rescaling of a gradient field cannot create a new ascent direction that opposes the
   feedforward objective on any edge.

2. THE FLOOR ``m_e >= 1`` GUARANTEES NO EDGE IS EVER SILENCED. Since ``ALPHA >= 0``,
   ``hat_w >= 0`` and ``b_e >= 0``, we have ``m_e >= 1`` for every edge at every step: H06 NEVER
   reduces an edge's pull below its baseline ``hat_w`` -- it only ADDS extra emphasis on top.
   So the baseline's faithful weighting is retained as a lower bound and the heavy/borderline
   emphasis is a bounded additive boost.

3. IT IS A REWEIGHTED-BUT-ALIGNED SURROGATE, NOT A DIFFERENT METRIC. The emphasis multiplier is a
   positive, edge-wise weight on the same monotone feedforward surrogate term; like ``hat_w``
   itself it changes how conflicts between edges are traded off, but every term still rewards
   feedforward orientation. The discrete oracle (total feedforward weight) is what is reported and
   tracked best-by-oracle, exactly as the baseline does.

Why this is LEAKAGE-SAFE / TARGET-BLIND
---------------------------------------
The multiplier ``m_e`` uses ONLY (i) input edge weights via ``hat_w = w/max(w)`` -- part of the
input graph -- and (ii) the model's OWN current surrogate state ``sigma_beta(Delta)`` (a borderline
edge is one whose ``sigma_beta`` is near 0.5). It NEVER reads, hardcodes, or folds in the discrete
oracle score; it never special-cases a dataset (same formula for connectome and mouse). The oracle
is consulted ONLY for the baseline's existing best-by-oracle tracking. ``b_e`` being detached also
makes this a pure gradient-magnitude reweighting, not an objective that secretly optimizes a
known target value.

Compute-matched
---------------
Standard knob-swap (loss reweighting only): SAME epoch budget as baseline (connectome 20k,
mouse 5k), SAME number of optimizer steps, SAME init / Adam / grad-clip / LR / beta schedule.
``n_epochs_done`` = actual optimizer steps, so ``baseline_passthrough`` / the frozen baseline at
matched seeds is the equal-budget comparator. Added per-step cost is one elementwise multiply over
the edge tensor (negligible). PURE Rocket score (no post-processing).

Implementation note (required): ``run_rocket`` builds the loss internally and exposes no hook to
reweight the per-edge term, so to change ONLY the per-edge loss weighting this module REPLICATES
the ``run_rocket`` main loop VERBATIM (same N(0,1)/init, Adam optimizer, grad-clip=1.0,
constant->exponential LR schedule, cyclic beta schedule, CPU discrete scoring, best-by-oracle
tracking, history, time-limit handling) and substitutes ONLY the per-edge emphasis factor in the
loss. The diff vs baseline is confined to the two lines that compute ``b_e`` and multiply it in.
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

ID = "H06"
HYPOTHESIS = (
    "Reweighting the surrogate per-edge term by a strictly-positive, bounded, detached emphasis "
    "factor m_e = 1 + ALPHA * hat_w * (4*sigma*(1-sigma)) -- which boosts gradient on edges that "
    "are both HEAVY (hat_w) and BORDERLINE (sigmoid near 0.5) while never reducing any edge below "
    "its baseline hat_w -- retains more high-weight feedforward arcs than the flat hat_w baseline. "
    "Direction-preserving (m_e > 0 detached => per-edge gradient sign unchanged) and leakage-safe "
    "(uses only input weights + the model's own surrogate state, never the discrete oracle)."
)

# Per-dataset epoch budget (= baseline single-run budget; standard knob-swap).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000}

# H06 primary-arm hyperparameter (the ONLY new constant vs baseline). Bounds the extra emphasis:
# m_e in [1, 1 + ALPHA]. ALPHA = 4 lets a heavy, fully-borderline edge get up to ~5x its baseline
# gradient magnitude while leaving saturated/tiny edges essentially at baseline -- a clear but
# bounded reshaping that stays close to the faithful Eq.-7 objective (lower bound m_e >= 1).
ALPHA = 4.0


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run Rocket with the H06 per-edge loss reweighting; everything else IDENTICAL to baseline.

    Verbatim copy of ``baseline.rocket.run_rocket`` with the SINGLE change that the per-edge loss
    term ``sigma * nw_t`` is multiplied by the detached emphasis factor
    ``m = 1 + ALPHA * nw_t * (4*sigma*(1-sigma))``. Init, Adam, grad-clip, LR schedule, beta
    schedule, CPU discrete scoring, best-by-oracle tracking, history and time-limit handling are
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

    for i in range(cfg.epochs):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        last_i = i

        beta = float(betas[i])

        optimizer.zero_grad()
        delta = positions[tgt_t] - positions[src_t]      # Delta = PT - PS
        sig = torch.sigmoid(beta * delta)                # Sig_beta

        # ── THE ONLY CHANGE vs baseline: detached heavy-&-borderline emphasis ─
        # b_e = 4*sigma*(1-sigma) in [0,1]: 1 at the borderline (sigma=0.5), -> 0 when saturated.
        # m_e = 1 + ALPHA * hat_w * b_e in [1, 1+ALPHA], strictly positive, DETACHED so it only
        # rescales each edge's gradient magnitude (direction-preserving; see module docstring).
        with torch.no_grad():
            borderline = 4.0 * sig * (1.0 - sig)
            emphasis = 1.0 + ALPHA * nw_t * borderline
        loss = -(emphasis * sig * nw_t).sum()            # -SUM m * Sig * hat_w

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
