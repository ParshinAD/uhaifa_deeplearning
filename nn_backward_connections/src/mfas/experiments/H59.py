"""Variant H59 — the champion pipeline plus minimal-FAS arc reclamation (stage 6).

Hypothesis (Phase 7, divergent mode)
------------------------------------
The champion's feedback arc set is **not minimal**: there exist backward edges whose two
endpoints are not connected by any path in the forward DAG, so the arc can be put back and
the graph re-sorted with no loss anywhere. The claim is that reclaiming them is worth more
than the campaign's minimum effect size on both primaries.

Measured at the prototype rung on the champion's own stored orders
(``experiments/outputs/proto_H59_{connectome,microns,mouse}.json``,
pre-registration ``experiments/prereg/H59_arc_reclamation.md``):

    connectome  84.15409511 -> 84.17169144   = +0.017596 pp   (bar 0.012)
    microns     83.24085291 -> 83.25961197   = +0.018759 pp   (bar 0.002)
    mouse       93.10282596 -> 93.17538326   = +0.072557 pp   (bar 0.010)

and on all three a SECOND round then found exactly zero reclaimable arcs, i.e. the stage
runs to a fixed point and the resulting arc set is certifiably minimal.

What changes vs H42
-------------------
One terminal stage is appended. Stages 1-4 are byte-identical to :mod:`mfas.experiments.H42`
— same constants, same calls, same order — so H59 - H42 isolates the new move class and
nothing else. 0 extra gradient steps.

Why the prototype number should transfer, where H60's did not
--------------------------------------------------------------
Meta-rule **M12** says a prototype composed from the champion's converged order overstates
the pipeline delta, because the control arm starts at its own fixed point and the variant arm
does not. That mechanism does not operate here, and the difference is structural rather than
a matter of degree: H60 altered an INNER stage, so its variant arm reached stage 4 on a
different trajectory than the control. H59 appends a TERMINAL stage to an unchanged pipeline,
so the variant arm's input is exactly the control arm's output — and H42 is deterministic on
this device (sigma = 0), so it is the same vector bit for bit. The prototype is therefore not
a proxy for the pipeline here; it is the pipeline's last stage run on the pipeline's real
output. The screen is what decides whether that reasoning is right.

Runtime — and why microns is not configured to run this stage
--------------------------------------------------------------
Measured stage cost at the prototype rung (one round, connectome and microns running
concurrently on CPU, so these are upper bounds rather than clean timings):

    connectome   443.3 s  (scan 63.0 s + conflict resolution 375.8 s)
    microns     1075.3 s  (scan 222.0 s + conflict resolution 843.2 s)
    mouse        < 0.1 s

The champion's microns run already takes 3398-3418 s against a 3600 s cap and a 3450 s guard
deadline (``sota.json``, and queue item **P07**). 3400 + 1075 = ~4475 s, so the stage cannot
be afforded there at this cost: the P05 guard would abort it at the stage boundary and the run
would be a degraded copy of H42. Setting ``_RECLAIM_ROUNDS['microns'] = 0`` states that
honestly in the code rather than shipping a stage that is silently truncated. **microns is
blocked on P07, not on the mechanism** — the mechanism is worth +0.018759 pp there, which is
9.4x that dataset's bar and the largest single piece of evidence P07 has been handed.

Determinism
-----------
The round count is a fixed constant and the candidate order is a stable sort on
(-weight, u, v), so the stage is deterministic. ``time_budget_s`` is an ABORT, never a sizing
rule (the P05 distinction): on a machine fast enough to finish the configured work it changes
nothing, and when it fires the round is dropped whole rather than half-applied.

Monotonicity
------------
Guaranteed by the reclamation lemma (see :mod:`mfas.refine.reclaim`) and additionally checked
here at runtime: the stage's result is accepted only if the frozen-oracle score is at least
the incoming score plus the reclaimed weight. If that ever fails the stage is discarded and
the pipeline returns the H42 order, so H59 cannot score below H42 by construction.

Leakage-safety
--------------
Unchanged from H42: every decision comes from the input edge weights and the current ranks,
the oracle only scores whole candidate vectors, and ``data/best_solution`` is never read.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct, score_from_order
from ..refine import alternate_scc_sift, sift_underrelaxed
from ..refine.reclaim import reclaim_arcs
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H59"
HYPOTHESIS = (
    "The champion's feedback arc set is not minimal: some backward edges have no forward "
    "path between their endpoints, so they can be re-added and the graph re-sorted with no "
    "loss. Reclaiming them beats the champion by more than the minimum effect size on both "
    "primaries."
)

# ── Stages 1-4: byte-identical to H42 ────────────────────────────────────────────────
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12}
_K_FULL = 6
_ALPHA = 0.7
_ALT_CYCLES = {"connectome": 77, "microns": 5, "mouse": 32}
_ALT_SIFT_SWEEPS = {"connectome": 2, "microns": 2, "mouse": 2}
_ALT_K_FULL = 2
_MIN_BLOCK = 32

# ── Stage 6: THE ONLY CHANGE ─────────────────────────────────────────────────────────
# One round suffices everywhere: the prototype's second round found exactly 0 reclaimable
# arcs on all three datasets, so a second round is pure cost. microns is 0 on runtime
# grounds alone (P07) - see the module docstring.
_RECLAIM_ROUNDS = {"connectome": 1, "microns": 0, "mouse": 1}
_RECLAIM_BUDGET = 1_000_000        # per-query edge expansions in the reachability BFS
_CONFLICT_BUDGET = 200_000         # per-arc DFS expansions during conflict resolution


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run the H42 pipeline, then reclaim every backward arc that closes no cycle."""
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))

    # ── Stage 1+2: H02 warm start -> unchanged Rocket (PURE) ─────────────────────
    order = greedy_fas_order(g)
    init_positions = _init_positions_from_order(order, device)
    rocket = run_rocket(g, cfg, seed=seed, device=device,
                        init_positions=init_positions, time_limit=time_limit)

    pure_best_positions = rocket.best_positions
    pure_best_score = rocket.best_score
    total = g.total_weight

    # ── Stage 3: H35's under-relaxed two-phase exact-gain sift ───────────────────
    rank0 = np.argsort(np.argsort(pure_best_positions, kind="stable"),
                       kind="stable").astype(np.int64)
    sift_budget = None if time_limit is None else max(0.0, time_limit - rocket.wall_clock_s)
    sift_rank, sift_score, sweep_log = sift_underrelaxed(
        g, rank0, k_full=_K_FULL, alpha=_ALPHA,
        max_sweeps=_MAX_SWEEPS.get(g.name, 40), time_budget_s=sift_budget)
    sift_time_s = float(sum(row["wall"] for row in sweep_log))

    # ── Stage 4: alternate block refinement with a SHORT single-node sift ────────
    alt_budget = (None if time_limit is None
                  else max(0.0, time_limit - rocket.wall_clock_s - sift_time_s))
    alt_rank, alt_score, alt_log = alternate_scc_sift(
        g, sift_rank,
        n_cycles=_ALT_CYCLES.get(g.name, 32),
        sift_sweeps=_ALT_SIFT_SWEEPS.get(g.name, 2),
        k_full=_ALT_K_FULL, alpha=_ALPHA, min_block=_MIN_BLOCK,
        time_budget_s=alt_budget)
    alt_time_s = float(alt_log[-1]["cum_wall_s"]) if alt_log else 0.0

    # ── Best-by-oracle across stages 2-4: the H42 result, unchanged ──────────────
    h42_score = pure_best_score
    h42_positions = pure_best_positions
    if sift_score > h42_score:
        h42_score, h42_positions = sift_score, sift_rank.astype(np.float32)
    if alt_score > h42_score:
        h42_score, h42_positions = alt_score, alt_rank.astype(np.float32)

    # ── Stage 6: minimal-FAS arc reclamation ─────────────────────────────────────
    t_rec0 = time.time()
    rounds = _RECLAIM_ROUNDS.get(g.name, 1)
    rec_log = []
    rec_score = h42_score
    rec_positions = h42_positions
    if rounds > 0:
        rec_budget = (None if time_limit is None
                      else max(0.0, time_limit - rocket.wall_clock_s - sift_time_s
                               - alt_time_s))
        in_rank = np.argsort(np.argsort(h42_positions, kind="stable"),
                             kind="stable").astype(np.int64)
        out_rank, rec_log = reclaim_arcs(
            g, in_rank, rounds=rounds, budget=_RECLAIM_BUDGET,
            conflict_budget=_CONFLICT_BUDGET, time_budget_s=rec_budget)
        w_reclaimed = float(sum(e.get("w_accepted", 0.0) for e in rec_log))
        cand_score = score_from_order(out_rank, np.asarray(g.src, dtype=np.int64),
                                      np.asarray(g.tgt, dtype=np.int64), g.weight)
        # The lemma says cand_score >= h42_score + w_reclaimed. Assert it rather than trust
        # it: if it ever fails the stage is discarded whole and H59 falls back to H42, so a
        # bug here can cost time but can never cost score.
        lemma_holds = bool(float(cand_score) + 1e-9 >= float(h42_score) + w_reclaimed)
        if lemma_holds and cand_score > rec_score:
            rec_score, rec_positions = cand_score, out_rank.astype(np.float32)
    else:
        w_reclaimed, lemma_holds = 0.0, True
    rec_time_s = time.time() - t_rec0

    best_score, best_positions = rec_score, rec_positions
    best_pct = pct(best_score, total)

    # ── Provenance: each stage's increment stays separately auditable ────────────
    hist = rocket.history
    hist.attrs["pure_best_score"] = pure_best_score
    hist.attrs["pure_best_pct"] = rocket.best_pct
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
    # Stage 6 - the isolated increment of the new move class.
    hist.attrs["h42_best_score"] = h42_score
    hist.attrs["h42_best_pct"] = pct(h42_score, total)
    hist.attrs["reclaim_rounds_requested"] = rounds
    hist.attrs["reclaim_best_score"] = rec_score
    hist.attrs["reclaim_best_pct"] = pct(rec_score, total)
    hist.attrs["reclaim_increment_pp"] = pct(rec_score, total) - pct(h42_score, total)
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
        # UNCHANGED gradient budget: stages 3, 4 and 6 add 0 optimizer steps.
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=rocket.wall_clock_s + sift_time_s + alt_time_s + rec_time_s,
    )
