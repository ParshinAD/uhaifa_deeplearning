"""Variant H13 — Mini-batch / stochastic edge subsampling per step (SGD-style Rocket).

Hypothesis (backlog H13): computing the surrogate loss on a RANDOM SUBSET of edges each
step (SGD-style) injects useful gradient noise that escapes the surrogate plateau, matching
or beating full-batch Rocket. Baseline Rocket is FULL-BATCH (the surrogate loss sums over ALL
edges every step -> deterministic descent into the nearest basin, consistent with the observed
~82.9% / ~92.1% plateau). Stochastic edge sampling is the textbook way to add exploration noise
to the descent direction.

Baseline per-step loss (see ``baseline.rocket.run_rocket``)
----------------------------------------------------------
    loss = -SUM_(u,v) in E  sigma_beta(Delta) * hat_w ,  Delta = pos[v]-pos[u], hat_w = w/max(w).
The per-edge term ``sigma_beta(Delta)*hat_w`` is summed over the ENTIRE edge set E every step.

H13 change (THE ONLY change vs baseline: the edge SET used each step)
--------------------------------------------------------------------
Each optimizer step, draw a UNIFORM random subset S_i ⊆ E of size ``m_batch = round(FRAC*|E|)``
(without replacement, ``FRAC`` = 0.5 primary arm) and compute the loss only over S_i:

    loss_i = -(|E| / |S_i|) * SUM_(u,v) in S_i  sigma_beta(Delta) * hat_w .

Everything else -- N(0,1) init, Adam, grad-clip=1.0, constant->exponential LR schedule, cyclic
beta schedule, CPU int64/float64 discrete scoring, best-by-oracle tracking, history, time-limit
-- is IDENTICAL to baseline. The per-edge weighting ``hat_w`` and the surrogate ``sigma_beta`` are
UNCHANGED; ONLY which edges enter the sum changes.

Unbiasedness (why the subset gradient is an unbiased descent direction)
-----------------------------------------------------------------------
The full-batch loss is a SUM over E. For a uniform-without-replacement subset S of fixed size m,
the Horvitz-Thompson / scaled-sum estimator ``(|E|/m) * SUM_{e in S} f(e)`` is an UNBIASED
estimator of ``SUM_{e in E} f(e)`` for any per-edge term ``f(e)`` (each edge is included with
probability m/|E|, so E[(|E|/m)*SUM_S f] = (|E|/m) * SUM_E (m/|E|) f(e) = SUM_E f(e)). This holds
termwise for the gradient (a finite linear combination of per-edge gradients), so the expected
subset gradient EQUALS the full-batch gradient: it is an unbiased stochastic-descent direction,
just with added zero-mean noise. The ``|E|/m`` scale (rather than the mean ``1/m``) keeps the
gradient MAGNITUDE on the same scale as full-batch, so grad-clip=1.0, the LR schedule and beta all
retain their baseline meaning and the only injected effect is the SGD noise the hypothesis is about.

Target-blindness / leakage-safety
---------------------------------
The subset is drawn UNIFORMLY over the INPUT edge index set ``[0, |E|)`` using a dedicated
``numpy.random.RandomState`` seeded ONLY from the run seed (``seed + 104729``, a fixed offset so
the batch stream is independent of the init RNG yet fully reproducible from the run seed). The
sampling NEVER consults the discrete oracle, NEVER uses edge orientation / current positions /
the target metric, and is NOT dataset-special-cased (same FRAC and same scheme for connectome and
mouse). The frozen oracle is consulted ONLY for the baseline's existing best-by-oracle tracking.
(Weight-proportional sampling would ALSO be target-blind -- it uses only input ``w`` -- but it
changes the estimator's variance/weighting and is left as an un-run arm; the primary arm is plain
uniform, the cleanest unbiased estimator of the baseline full-batch sum.)

Compute-matched (PROTOCOL basis = total_grad_steps)
---------------------------------------------------
The comparison basis is ``total_grad_steps`` = number of ``optimizer.step()`` calls. H13 runs the
SAME number of gradient steps as baseline (connectome 20k, mouse 5k), each on a random edge subset,
so ``n_epochs_done`` = 20000 / 5000 EXACTLY matches the baseline single run, and the equal-budget
comparator is ``baseline_passthrough`` / the frozen baseline at matched seeds (connectome
82.8958 ± 0.0189; mouse 92.0696 ± 0.2624). Wall-clock per step is LOWER (half the edges), but per
PROTOCOL the comparison is gradient steps, NOT wall-clock -- so NO extra "spend the saved
wall-clock" steps are added (that would break the grad-step match). An EXTRA-STEPS arm (use the
saved wall-clock to take more steps) is noted here as an UN-RUN arm, not run in this screen.

Arms
----
PRIMARY (this screen): FRAC = 0.5 (a moderate, well-justified mini-batch fraction).
UN-RUN arms (noted, not run): FRAC = 0.25 (stronger noise, higher variance); weight-proportional
sampling; an extra-steps arm spending the per-step wall-clock saving.

Implementation note (required): ``run_rocket`` builds the loss internally and exposes no edge-set
hook, so this module REPLICATES the ``run_rocket`` main loop VERBATIM (same init, Adam, grad-clip,
LR schedule, beta schedule, CPU discrete scoring, best-by-oracle tracking, history, time-limit
handling) and adds ONLY (i) a dedicated batch RNG and (ii) per-step uniform edge subsampling with
the ``|E|/m`` unbiased rescale. The diff vs baseline is confined to those additions.
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

ID = "H13"
HYPOTHESIS = (
    "Computing the surrogate loss on a UNIFORM random subset of edges each step (mini-batch "
    "SGD, FRAC=0.5, scaled by |E|/m so the subset gradient is an UNBIASED estimator of the "
    "full-batch gradient) injects exploration noise that escapes the surrogate plateau, matching "
    "or beating full-batch Rocket at EQUAL gradient steps (connectome 20k, mouse 5k). Only the "
    "edge SET per step changes; init/Adam/grad-clip=1.0/LR/beta/hat_w/sigma_beta are baseline. "
    "Target-blind (uniform over input edge indices via a run-seed-derived RNG; never reads the "
    "oracle, never dataset-special-cased). Compute-matched on total_grad_steps per PROTOCOL; "
    "wall-clock is lower per step but NOT the comparison basis. Primary arm FRAC=0.5; FRAC=0.25, "
    "weight-proportional and extra-steps arms un-run."
)

# Per-dataset epoch budget (= baseline single-run gradient-step budget; standard match).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000}

# H13 primary-arm hyperparameter (the ONLY new numeric constant): mini-batch fraction.
FRAC = 0.5

# Fixed offset for the dedicated batch RNG, so the batch stream is reproducible from the run
# seed yet independent of torch's init RNG. (104729 is just a fixed prime; not tuned.)
_BATCH_SEED_OFFSET = 104729


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run Rocket with per-step uniform edge subsampling (FRAC=0.5); else IDENTICAL to baseline.

    Verbatim copy of ``baseline.rocket.run_rocket`` with the SINGLE algorithmic addition that each
    step samples a uniform random subset of edge indices and scales the loss by ``|E|/m`` (unbiased
    estimator of the full-batch sum). Init, Adam, grad-clip=1.0, LR/beta schedules, CPU discrete
    scoring, best-by-oracle tracking, history and time-limit handling are unchanged.
    ``n_epochs_done`` = actual optimizer steps (= baseline budget).
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
    n_edges = int(src_t.shape[0])
    m_batch = max(1, int(round(FRAC * n_edges)))      # subset size per step
    scale = float(n_edges) / float(m_batch)           # |E|/m unbiased rescale

    # Dedicated, reproducible-from-seed, target-blind batch RNG (independent of init RNG).
    batch_rng = np.random.RandomState(int(seed) + _BATCH_SEED_OFFSET)

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

        # ── THE ONLY ADDITION vs baseline: sample a uniform edge subset this step ─
        # Target-blind uniform-without-replacement subset of edge indices; the loss over the
        # subset is scaled by |E|/m so its gradient is an UNBIASED estimator of the full-batch
        # gradient (see module docstring). The surrogate sigma_beta and weight hat_w are unchanged.
        batch_idx_np = batch_rng.choice(n_edges, size=m_batch, replace=False)
        batch_idx = torch.from_numpy(batch_idx_np).to(device=device, dtype=torch.long)
        bsrc = src_t[batch_idx]
        btgt = tgt_t[batch_idx]
        bw = nw_t[batch_idx]

        optimizer.zero_grad()
        delta = positions[btgt] - positions[bsrc]        # Delta = PT - PS on the subset
        sig = torch.sigmoid(beta * delta)                # Sig_beta (UNCHANGED surrogate)
        loss = -scale * (sig * bw).sum()                 # -(|E|/m) SUM_S Sig*hat_w (unbiased)

        loss.backward()
        torch.nn.utils.clip_grad_norm_([positions], cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if i % cfg.log_interval == 0 or i == cfg.epochs - 1:
            # Discrete score is the EXACT full-graph oracle (never the subset) — baseline tracking.
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
