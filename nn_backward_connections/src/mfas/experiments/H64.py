"""Variant H64 - H38's ASYMMETRIC one-sided surrogate, COMPOSED with the champion stack.

The change is one line of arithmetic: stage 2's ``torch.sigmoid(beta * delta)`` becomes
H38's :func:`mfas.experiments.H38._asym_surrogate`. Everything else - the greedy-FAS warm
start, the epoch budget, the optimizer, the LR and beta schedules, grad clipping, and every
discrete stage after the gradient phase - is byte-identical to the champion of the dataset
being run. So ``H64 - champion`` isolates the SURROGATE SHAPE and nothing else.

Why this exists
---------------
H38 is the only continuous mechanism that ever moved this metric: **+0.36675 pp** pure Rocket
on connectome, 3/3 seeds positive, with controls H38C (the mirror shape, -2.88905) and H38D
(plain sigmoid at beta x 4, -0.58043) ruling out "any one-sided shape" and the beta/scale axis
respectively. ``killed.json`` meta-rule **M1**, as amended 2026-08-26, states the open question
in its own words: *"Whether a shape gain survives composition with the refinement stack is OPEN
and is what H64 measures."*

The original measurement has three defects, each verified in the artifacts:

* **comparator** - H38 was scored against ``baseline_passthrough`` (82.89580), not the champion;
* **basin** - ``H38.py:113,132-136`` builds ``RocketConfig`` with ``init_mode="random"`` and
  ``run()`` takes no ``init_positions``, so the whole gain lives in the RANDOM-INIT basin, while
  the champion's stage 2 starts from greedy-FAS;
* **device** - all 18 H38* records are Apple MPS with ``env.cuda`` null, so none of that
  evidence is re-scorable here.

This module removes all three at once: champion comparator, greedy-FAS warm start, CUDA.

Per-dataset champion composition (P14) - and the consequence for mouse
----------------------------------------------------------------------
``sota.json`` holds a DIFFERENT champion per dataset, and P14 (cycle 15) established that a
variant must compose on the champion of the dataset it is running, or say why not. Here:

    connectome   H42  (stages 1-4, 20,000 epochs)
    microns      H42  (stages 1-4, 80,000 epochs)
    mouse        H63  (``_EPOCHS = 0``; stages 3, 4, 5 (H52 pair relocation), 6 (reclamation))

**On mouse the mouse champion runs ZERO gradient steps.** H44 measured that and it is why
``_EPOCHS["mouse"] = 0``: dropping the gradient phase entirely IMPROVED the final mouse score
by +0.16588 pp. A stage-2 surrogate swap is therefore a **structural no-op on mouse** - the
surrogate is never called, the run is bit-identical to H63, and the mouse delta is exactly 0.

That is the honest state of affairs and it is stated here rather than discovered later: the
mouse leg of this variant is **vacuous as a tripwire**. It cannot falsify a deterministic
classification and it cannot detect a small-graph artefact, because it does not exercise the
changed code at all. The connectome leg is the whole test. (Running mouse at H42's 5,000-epoch
configuration instead WOULD exercise the surrogate, but it would be measuring a variant of a
pipeline that is NOT the mouse champion, against a champion 0.26 pp above it - precisely the
moving-comparator fault P14 exists to prevent. The diagnostic value of that measurement is real
and it was taken, at the PROTOTYPE rung, where it is labelled a proxy:
``experiments/outputs/proto_H64_rung2.json``.)

What the cheap rungs already established (``experiments/outputs/proto_H64_rung01.json``)
----------------------------------------------------------------------------------------
* ``cheap_gate(_asym_surrogate)`` = **WARN**, and the only reason is the documented derivative
  kink at ``z = M``. It is NOT the sigmoid at another beta and ``g(0) < sup g``, so M1's burden
  (a) is discharged on this machine.
* **No crossover** in ``beta*std`` over ``[1e-2, 1e6]`` on the hard synthetic: the alignment
  ratio stays in ``[+0.0035, +1.380]`` and never changes sign. M1's burden (b) is discharged.
* **The reach premise is false, and in the unhelpful direction.** On the H42 champion order,
  ASYM's live gradient support covers **13.10%** of the backward weight - *less* than the plain
  sigmoid's **15.66%** - because ``tanh`` dies at ``(z-M)/T < -4``, giving a reach of ~1,799
  ranks against the sigmoid's ~2,132, while 88.4% of the backward weight sits beyond 1,000
  ranks and 48.4% beyond 20,000. So if ASYM helps, it is NOT by reaching further; it can only
  be by not spending gradient on already-satisfied edges.

Leakage-safety and compute accounting
-------------------------------------
``_asym_surrogate`` is a function of the model positions and ``beta`` only, weighted by the same
``hat_w = w/max(w)`` as the baseline. It never reads the discrete oracle, never special-cases a
dataset, and ``data/best_solution`` is never touched. The gradient budget is UNCHANGED - same
epochs, same optimizer, same schedules - so this is a pure knob swap and not extra compute.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from ..baseline.rocket import RocketConfig, RocketResult, make_beta_schedule
from ..io import GraphData
from ..metrics import pct, score_from_order, score_from_positions
from ..refine import alternate_scc_sift, sift_underrelaxed
from ..refine.pair_relocate import pair_relocate
from ..refine.reclaim2 import reclaim_arcs_fast
from .H02 import _init_positions_from_order, greedy_fas_order
from .H38 import M, T, _asym_surrogate

ID = "H64"
HYPOTHESIS = (
    "Replacing torch.sigmoid with H38's one-sided _asym_surrogate in stage 2 of the champion "
    "pipeline -- stages 1, 3, 4 (and 5, 6 where the dataset's champion has them) byte-identical "
    "-- changes the FINAL connectome score by more than +0.012 pp. H38's +0.36675 pp was PURE "
    "Rocket, from a RANDOM init, on MPS, against baseline_passthrough; this measures the same "
    "shape composed with the refinement stack, warm-started from greedy-FAS, on CUDA, against "
    "the champion. M1 names this measurement as the open clause it cannot close."
)

# -- The champion of EACH dataset (P14). Sigmoid -> ASYM is the ONLY change. -----------
# connectome/microns: H42's constants. mouse: H63's (which carries H44's _EPOCHS = 0).
_EPOCHS = {"connectome": 20_000, "mouse": 0, "microns": 80_000}
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12}
_K_FULL = 6
_ALPHA = 0.7
_ALT_CYCLES = {"connectome": 77, "microns": 5, "mouse": 32}
_ALT_SIFT_SWEEPS = {"connectome": 2, "microns": 2, "mouse": 2}
_ALT_K_FULL = 2
_MIN_BLOCK = 32

# Stage 5 (H52 pair relocation) and stage 6 (H63 arc reclamation) exist in the MOUSE champion
# only. H42 holds both primaries and contains neither, so a 0 budget keeps those legs
# byte-identical to H42 - the same convention H63 uses and the critic verified numerically.
_PAIR_PASSES = 2
_PAIR_MAX_POPS = {"connectome": 0, "microns": 0, "mouse": 100_000}
_RECLAIM_ROUNDS = {"connectome": 0, "microns": 0, "mouse": 1}
_RECLAIM_BUDGET = 1_000_000
_CONFLICT_BUDGET = 200_000


def _rocket_asym(g: GraphData, cfg: RocketConfig, seed: int, device,
                 init_positions: Optional[torch.Tensor] = None,
                 time_limit: Optional[float] = None) -> RocketResult:
    """``run_rocket`` verbatim, with ONE line changed: the surrogate.

    This is a copy rather than a new parameter on ``run_rocket`` on purpose -
    ``src/mfas/baseline/rocket.py`` is the shared baseline every other variant is measured
    against, and a variant must not reach into it. The loop below is line-for-line
    ``run_rocket``; the single difference is marked in the loop body.
    """
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
        # Always score on CPU in int64/float64 (CLAUDE.md oracle fact).
        return score_from_positions(pos_param.detach().cpu().numpy(),
                                    src_np, tgt_np, g.weight)

    if init_positions is not None:
        pos_data = init_positions.clone().detach().to(torch.float32)
    elif cfg.init_mode == "random":
        pos_data = torch.randn(n, device=device)
    else:
        from ..baseline.rocket import make_init_positions
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
    n_epochs_done = 0

    for i in range(cfg.epochs):
        if time_limit and (time.time() - start_time) > time_limit:
            break
        n_epochs_done += 1

        beta = float(betas[i])

        optimizer.zero_grad()
        delta = positions[tgt_t] - positions[src_t]
        # -- THE ONLY CHANGE vs run_rocket: one-sided asymmetric surrogate ------------
        sig = _asym_surrogate(beta * delta)
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
        n_epochs_done=n_epochs_done,
        wall_clock_s=wall,
    )


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run this dataset's champion pipeline with the ASYMMETRIC surrogate in stage 2."""
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))

    # -- Stage 1+2: H02 warm start -> Rocket with the ASYM surrogate (PURE) -------
    order = greedy_fas_order(g)
    init_positions = _init_positions_from_order(order, device)
    rocket = _rocket_asym(g, cfg, seed=seed, device=device,
                          init_positions=init_positions, time_limit=time_limit)

    pure_best_positions = rocket.best_positions
    pure_best_score = rocket.best_score
    total = g.total_weight

    # -- Stage 3: H35's under-relaxed two-phase exact-gain sift -------------------
    rank0 = np.argsort(np.argsort(pure_best_positions, kind="stable"),
                       kind="stable").astype(np.int64)
    sift_budget = None if time_limit is None else max(0.0, time_limit - rocket.wall_clock_s)
    sift_rank, sift_score, sweep_log = sift_underrelaxed(
        g, rank0, k_full=_K_FULL, alpha=_ALPHA,
        max_sweeps=_MAX_SWEEPS.get(g.name, 40), time_budget_s=sift_budget)
    sift_time_s = float(sum(row["wall"] for row in sweep_log))

    # -- Stage 4: alternate block refinement with a SHORT single-node sift --------
    alt_budget = (None if time_limit is None
                  else max(0.0, time_limit - rocket.wall_clock_s - sift_time_s))
    alt_rank, alt_score, alt_log = alternate_scc_sift(
        g, sift_rank,
        n_cycles=_ALT_CYCLES.get(g.name, 32),
        sift_sweeps=_ALT_SIFT_SWEEPS.get(g.name, 2),
        k_full=_ALT_K_FULL, alpha=_ALPHA, min_block=_MIN_BLOCK,
        time_budget_s=alt_budget)
    alt_time_s = float(alt_log[-1]["cum_wall_s"]) if alt_log else 0.0

    # -- Stage 5: pair relocation - mouse only (its champion H63 carries H52's stage) --
    max_pops = _PAIR_MAX_POPS.get(g.name, 0)
    pair_rank, pair_score, pair_log, pair_time_s = alt_rank, alt_score, [], 0.0
    if max_pops:
        pair_budget = (None if time_limit is None else
                       max(0.0, time_limit - rocket.wall_clock_s - sift_time_s - alt_time_s))
        pair_rank, pair_score, pair_log = pair_relocate(
            g, alt_rank, n_passes=_PAIR_PASSES, max_pops=max_pops,
            time_budget_s=pair_budget)
        pair_time_s = float(pair_log[-1]["cum_wall_s"]) if pair_log else 0.0

    # -- Best-by-oracle across stages 2-5 ----------------------------------------
    base_score = pure_best_score
    base_positions = pure_best_positions
    if sift_score > base_score:
        base_score, base_positions = sift_score, sift_rank.astype(np.float32)
    if alt_score > base_score:
        base_score, base_positions = alt_score, alt_rank.astype(np.float32)
    if pair_score > base_score:
        base_score, base_positions = pair_score, pair_rank.astype(np.float32)

    # -- Stage 6: minimal-FAS arc reclamation - mouse only (H63) ------------------
    t_rec0 = time.time()
    rounds = _RECLAIM_ROUNDS.get(g.name, 0)
    rec_log = []
    rec_score, rec_positions = base_score, base_positions
    w_reclaimed, lemma_holds = 0.0, True
    if rounds > 0:
        rec_budget = (None if time_limit is None
                      else max(0.0, time_limit - rocket.wall_clock_s - sift_time_s
                               - alt_time_s - pair_time_s))
        in_rank = np.argsort(np.argsort(base_positions, kind="stable"),
                             kind="stable").astype(np.int64)
        out_rank, rec_log = reclaim_arcs_fast(
            g, in_rank, rounds=rounds, budget=_RECLAIM_BUDGET,
            conflict_budget=_CONFLICT_BUDGET, time_budget_s=rec_budget)
        w_reclaimed = float(sum(e.get("w_accepted", 0.0) for e in rec_log))
        cand_score = score_from_order(out_rank, np.asarray(g.src, dtype=np.int64),
                                      np.asarray(g.tgt, dtype=np.int64), g.weight)
        lemma_holds = bool(float(cand_score) + 1e-9 >= float(base_score) + w_reclaimed)
        if lemma_holds and cand_score > rec_score:
            rec_score, rec_positions = cand_score, out_rank.astype(np.float32)
    rec_time_s = time.time() - t_rec0

    best_score, best_positions = rec_score, rec_positions
    best_pct = pct(best_score, total)

    # -- Provenance: each stage's increment stays separately auditable ------------
    hist = rocket.history
    hist.attrs["surrogate"] = "asym_one_sided"
    hist.attrs["surrogate_M"] = M
    hist.attrs["surrogate_T"] = T
    # True iff the surrogate was actually EXERCISED. False on mouse (0 epochs), where this
    # run is bit-identical to H63 by construction and the delta is structurally 0.
    hist.attrs["surrogate_exercised"] = bool(rocket.n_epochs_done > 0)
    hist.attrs["pure_best_score"] = pure_best_score
    hist.attrs["pure_best_pct"] = rocket.best_pct
    hist.attrs["epochs_requested"] = _EPOCHS.get(g.name, RocketConfig.epochs)
    hist.attrs["sift_best_score"] = sift_score
    hist.attrs["sift_best_pct"] = pct(sift_score, total)
    hist.attrs["n_sift_sweeps"] = len(sweep_log)
    hist.attrs["sift_sweeps_requested"] = _MAX_SWEEPS.get(g.name, 40)
    hist.attrs["sift_converged"] = bool(sweep_log and sweep_log[-1]["n_movers"] == 0)
    hist.attrs["sift_time_s"] = sift_time_s
    hist.attrs["sift_alpha"] = _ALPHA
    hist.attrs["sift_k_full"] = _K_FULL
    hist.attrs["alt_best_score"] = alt_score
    hist.attrs["alt_best_pct"] = pct(alt_score, total)
    hist.attrs["alt_increment_pp"] = pct(alt_score, total) - pct(sift_score, total)
    hist.attrs["n_alt_cycles"] = len(alt_log)
    hist.attrs["alt_cycles_requested"] = _ALT_CYCLES.get(g.name, 32)
    hist.attrs["alt_time_s"] = alt_time_s
    hist.attrs["alt_min_block"] = _MIN_BLOCK
    hist.attrs["alt_sift_sweeps"] = _ALT_SIFT_SWEEPS.get(g.name, 2)
    hist.attrs["alt_log"] = alt_log
    hist.attrs["sift_log"] = sweep_log
    hist.attrs["pair_max_pops"] = max_pops
    hist.attrs["pair_best_score"] = pair_score
    hist.attrs["pair_best_pct"] = pct(pair_score, total)
    hist.attrs["pair_time_s"] = pair_time_s
    hist.attrs["pair_log"] = pair_log
    hist.attrs["base_best_score"] = base_score
    hist.attrs["base_best_pct"] = pct(base_score, total)
    hist.attrs["reclaim_rounds_requested"] = rounds
    hist.attrs["reclaim_best_score"] = rec_score
    hist.attrs["reclaim_best_pct"] = pct(rec_score, total)
    hist.attrs["reclaim_increment_pp"] = pct(rec_score, total) - pct(base_score, total)
    hist.attrs["reclaim_weight_accepted"] = w_reclaimed
    hist.attrs["reclaim_lemma_holds"] = lemma_holds
    hist.attrs["reclaim_time_s"] = rec_time_s
    hist.attrs["reclaim_log"] = rec_log
    hist.attrs["refined_best_score"] = best_score
    hist.attrs["refined_best_pct"] = best_pct

    return RocketResult(
        best_positions=best_positions,
        best_score=best_score,
        best_pct=best_pct,
        history=hist,
        # UNCHANGED gradient budget vs the champion: stages 3-6 add 0 optimizer steps.
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=(rocket.wall_clock_s + sift_time_s + alt_time_s + pair_time_s
                      + rec_time_s),
    )
