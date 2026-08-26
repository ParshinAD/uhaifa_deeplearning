"""Variant H63 — H59's arc reclamation, composed on the PER-DATASET champion and made cheap.

The hypothesis is H59's, unchanged
----------------------------------
The champion's feedback arc set is **not minimal**: there are backward edges whose endpoints
are not connected by any path in the forward DAG ``F``, so the arc can be put back and the
graph re-sorted with no loss anywhere. Reclaiming them is worth more than the campaign's
minimum effect size on both primaries.

H59 established the mechanism at the prototype rung on all three datasets and PASSED the
connectome screen (84.17175348 = +0.017658 pp against a 0.012 bar, with its internal control
``h42_best_pct`` bit-identical to the champion, so the whole increment was the new stage). It
could not be promoted for two reasons, neither of which was about the mechanism. H63 removes
both.

**(1) The mouse leg was built on the wrong base — queue item P14.** ``sota.json`` holds a
champion PER DATASET: H42 / H42 / **H52**. H59 was written on H42 throughout, so on mouse it
lacked H44's ``_EPOCHS["mouse"] = 0`` and H52's stage 5 and scored 92.94798739 — a number
measured against a base 0.155 pp below the mouse champion. H63 composes on the champion of
each dataset instead:

    connectome   H42  (stage 5 off: ``_PAIR_MAX_POPS = 0`` -> stages 1-4 only)
    microns      H42  (same)
    mouse        H52  (``_EPOCHS = 0`` and stage 5 at 100 k pops)

With the pop budget at 0 the pair-relocation stage returns its input untouched, so it is
skipped outright on the primaries and those legs stay bit-identical to H42.

**(2) The stage did not fit microns' runtime — queue items P13 and P07.** The reference
implementation costs 1125.7 s on microns against ~30-250 s of slack under the P05 guard
deadline. :mod:`mfas.refine.reclaim2` computes the same answer by a cheaper route and brings
that to **159.2 s** (measured, ``experiments/outputs/proto_H63_microns.json``):

    ============  ==========  ==========  =========
    stage cost    reference   reclaim2    speed-up
    ============  ==========  ==========  =========
    connectome     478.7 s     160.5 s     x2.9
    microns       1125.7 s     159.2 s     x7.2
    mouse           <0.1 s      <0.1 s      --
    ============  ==========  ==========  =========

**How closely it agrees, stated exactly rather than rounded to "identical".** The accepted
arc list is identical element for element on connectome (2,231 arcs) and on mouse (5). On
microns it differs by ONE arc of 2,833, and the difference is understood and is in the
faster implementation's favour: both scans carry the same fail-safe expansion budget, but the
reference charges a query the WHOLE adjacency row it touches while ``reclaim2`` charges only
the part inside the rank window, so the reference abandoned exactly one query
(``n_unknown = 1``) and recorded it as "reachable" — the safe direction, which forgoes gain.
``reclaim2`` settles every query on all three datasets (``n_unknown = 0``) and accepts no arc
on a budget default (``n_budget_rejects = 0``), so its answer is not an approximation of the
reference's: it IS the exact reclaimable set, of which the reference's is a subset short by
one arc on one dataset. Evidence: ``experiments/outputs/proto_H63_<ds>.json``, and
``tests/test_reclaim2.py`` for identity against the reference on randomised digraphs where
neither budget binds.

Why microns still has to buy seconds, and what it pays
------------------------------------------------------
159 s is not free. The champion's microns run has been logged between 3196.6 s and 3418.0 s
of variant-internal time against a 3450 s guard deadline, so on its worse days there is no
room for a 159 s stage. H63 therefore also takes queue item **H62**: it buys the seconds from
the Rocket epoch budget, at the smallest dose that does the job.

    ``_EPOCHS["microns"]``:  80,000 -> 70,000

Sized from measurement, not from preference. H43 measured microns Rocket at
**753.98 s / 20,000 epochs = 0.0377 s per epoch** (``proto_H43_microns.json``), so 10,000
epochs is ~377 s — more than twice the stage cost, which is deliberate: it leaves microns
with ~250-470 s of margin instead of the ~30-250 s it has now, so this change *improves* the
runtime-safety problem P07 records rather than spending the last of it.

What the cut costs in score is bounded above by an existing measured arm rather than guessed:
H43 cut microns Rocket by **75%** (80,000 -> 20,000) and, after stages 3 and 4 absorbed it,
the fully refined result was 83.23842442 against the champion's 83.24085291 — a cost of only
**0.00243 pp** for four times this cut. Against that, reclamation is worth +0.018928 pp on
this dataset, measured on the champion's own order -- 9.5x the 0.002 bar. The screen decides it.

    HONESTLY STATED: this makes the microns leg a COMPOUND change, and its pre-reclaim base
    is expected to land slightly BELOW the champion rather than on it. The two increments
    stay separately auditable — ``base_best_pct`` is the score before reclamation and
    ``reclaim_increment_pp`` is the stage's own contribution — so the screen reports the
    epoch cost and the reclamation gain as separate numbers, not as one net figure. On
    connectome and mouse nothing is bought and nothing is compounded: those legs are the
    champion plus one terminal stage, exactly as H59's connectome leg was.

Determinism
-----------
Every budget here is a fixed constant (epochs, pop count, round count, expansion budgets), so
the result does not depend on how fast the machine is. ``time_budget_s`` remains an ABORT and
never a sizing rule (the P05 distinction). The candidate arc order is a stable sort on
``(-weight, u, v)``. No stage draws from ``seed``.

Monotonicity
------------
Stage 6 is monotone by the reclamation lemma (:mod:`mfas.refine.reclaim`), and that is
asserted at runtime against the frozen oracle rather than trusted: the stage's result is kept
only if it scores at least the incoming score plus the reclaimed weight. If the assert ever
fails the stage is discarded whole and the pipeline returns its stage-5 result.

Leakage-safety
--------------
Unchanged: every decision comes from the input edge weights and the current ranks, the oracle
only scores whole candidate vectors, and ``data/best_solution`` is never read.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct, score_from_order
from ..refine import alternate_scc_sift, sift_underrelaxed
from ..refine.pair_relocate import pair_relocate
from ..refine.reclaim2 import reclaim_arcs_fast
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H63"
HYPOTHESIS = (
    "The champion's feedback arc set is not minimal: some backward edges have no forward "
    "path between their endpoints, so they can be re-added and the graph re-sorted with no "
    "loss. Composed on the champion of EACH dataset and made cheap enough to run on microns, "
    "reclaiming them beats the champion by more than the minimum effect size on both "
    "primaries."
)

# ── Stages 1-4 ───────────────────────────────────────────────────────────────────────
# connectome/microns: H42's constants. mouse: H52's (which is H44's _EPOCHS = 0).
# microns' 70,000 is the H62 purchase — see the module docstring.
_EPOCHS = {"connectome": 20_000, "mouse": 0, "microns": 70_000}
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12}
_K_FULL = 6
_ALPHA = 0.7
_ALT_CYCLES = {"connectome": 77, "microns": 5, "mouse": 32}
_ALT_SIFT_SWEEPS = {"connectome": 2, "microns": 2, "mouse": 2}
_ALT_K_FULL = 2
_MIN_BLOCK = 32

# ── Stage 5: H52's pair relocation, ON for mouse ONLY (P14) ──────────────────────────
# 0 means the champion of that dataset does not contain this stage, so H63 must not either:
# H42 is the champion on both primaries and has no stage 5. A 0 budget makes pair_relocate
# return its input unchanged, so skipping the call is equivalent and also saves its CSR build.
_PAIR_PASSES = 2
_PAIR_MAX_POPS = {"connectome": 0, "microns": 0, "mouse": 100_000}

# ── Stage 6: minimal-FAS arc reclamation, ON EVERYWHERE ──────────────────────────────
# One round suffices: the cycle-14 prototype's second round found exactly 0 reclaimable arcs
# on all three datasets, i.e. the stage reaches a fixed point in one pass and certifies the
# resulting arc set minimal. A second round would be a pure repeat of the scan.
_RECLAIM_ROUNDS = {"connectome": 1, "microns": 1, "mouse": 1}
_RECLAIM_BUDGET = 1_000_000        # per-query edge expansions in the reachability BFS
_CONFLICT_BUDGET = 200_000         # per-arc DFS expansions during conflict resolution


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run each dataset's champion pipeline, then reclaim every arc that closes no cycle."""
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

    # ── Stage 5: pair relocation — mouse only, because only mouse's champion has it ──
    max_pops = _PAIR_MAX_POPS.get(g.name, 0)
    pair_rank, pair_score, pair_log, pair_time_s = alt_rank, alt_score, [], 0.0
    if max_pops:
        pair_budget = (None if time_limit is None else
                       max(0.0, time_limit - rocket.wall_clock_s - sift_time_s - alt_time_s))
        pair_rank, pair_score, pair_log = pair_relocate(
            g, alt_rank, n_passes=_PAIR_PASSES, max_pops=max_pops,
            time_budget_s=pair_budget)
        pair_time_s = float(pair_log[-1]["cum_wall_s"]) if pair_log else 0.0

    # ── Best-by-oracle across stages 2-5: the CHAMPION's result for this dataset ──
    base_score = pure_best_score
    base_positions = pure_best_positions
    if sift_score > base_score:
        base_score, base_positions = sift_score, sift_rank.astype(np.float32)
    if alt_score > base_score:
        base_score, base_positions = alt_score, alt_rank.astype(np.float32)
    if pair_score > base_score:
        base_score, base_positions = pair_score, pair_rank.astype(np.float32)

    # ── Stage 6: minimal-FAS arc reclamation ─────────────────────────────────────
    t_rec0 = time.time()
    rounds = _RECLAIM_ROUNDS.get(g.name, 1)
    rec_log = []
    rec_score = base_score
    rec_positions = base_positions
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
        # The lemma says cand_score >= base_score + w_reclaimed. Assert it rather than trust
        # it: if it ever fails the stage is discarded whole and H63 falls back to stage 5, so
        # a bug here can cost time but can never cost score.
        lemma_holds = bool(float(cand_score) + 1e-9 >= float(base_score) + w_reclaimed)
        if lemma_holds and cand_score > rec_score:
            rec_score, rec_positions = cand_score, out_rank.astype(np.float32)
    rec_time_s = time.time() - t_rec0

    best_score, best_positions = rec_score, rec_positions
    best_pct = pct(best_score, total)

    # ── Provenance: each stage's increment stays separately auditable ────────────
    hist = rocket.history
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
    # Stage 5 - present on mouse only, because only mouse's champion (H52) contains it.
    hist.attrs["pair_max_pops"] = max_pops
    hist.attrs["pair_best_score"] = pair_score
    hist.attrs["pair_best_pct"] = pct(pair_score, total)
    hist.attrs["pair_time_s"] = pair_time_s
    hist.attrs["pair_log"] = pair_log
    # The per-dataset CHAMPION's score, i.e. everything before the new move class. On
    # connectome and mouse this must equal the champion exactly; on microns it is expected to
    # sit slightly BELOW it, by the price of the 10,000-epoch Rocket cut (H62).
    hist.attrs["base_best_score"] = base_score
    hist.attrs["base_best_pct"] = pct(base_score, total)
    # Stage 6 - the isolated increment of the new move class.
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
        # Gradient budget: stages 3-6 add 0 optimizer steps. microns runs 10,000 FEWER
        # epochs than the champion, which is the whole point of the H62 purchase.
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=(rocket.wall_clock_s + sift_time_s + alt_time_s + pair_time_s
                      + rec_time_s),
    )
