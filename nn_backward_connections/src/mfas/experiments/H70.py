"""Variant H70 - RE-SIZE the microns compute allocation so the run fits 3600 s with margin.

What this changes, and what it deliberately does not
-----------------------------------------------------
Exactly two constants move, and both only on **microns**: the stage-2 epoch count and the
stage-4 alternation cycle count. Every other constant, every stage, and both other datasets are
byte-identical to the champion of the dataset being run. So ``H70 - champion`` on microns
isolates the COMPUTE ALLOCATION and nothing else.

Why it exists: P19, the campaign's blocking item
-------------------------------------------------
The shipped microns configuration (H42: 80,000 epochs, 5 alternation cycles) does not satisfy
the campaign's own 3600 s runtime invariant with any usable margin. Its clean confirmed walls
are 3397.8-3418.0 s against a 3450 s guard deadline - **32-52 s of slack, about 1%** - so
ordinary machine noise truncates it. The measured tally on this box is not ambiguous:

* 6 of 8 H64 microns runs truncated, across two independent sessions;
* the **unmodified champion H42 itself** truncated on its own control run
  (``results/20260827T092101Z-H42-microns-s42-verify-f91b66.json``, 3495.3 s, ``degraded``,
  stage 3 cut to 4/12 sweeps and stage 4 to 1/5 cycles).

So the attribution is settled and it is not any variant: the configuration is over-budget. The
consequence is blocking - microns cannot produce a clean confirm pool, so no variant can be
promoted on it whatever its score.

The OPERATOR decided this on 2026-08-27 (``queue.json`` P19, commit a95e2f6): do **not** raise
``max_wall_clock_s_per_run``. Make microns fit the existing invariant with real margin, and
treat speed as a goal in its own right - *"let it squeeze into what it has, that is already too
much in my opinion and ideally it should be faster"*. The epoch count is to be **sized on the
curve, never picked**, and it takes the normal ladder because the cut is not free.

Which knob, and why only this one
----------------------------------
Attribution is measured, on H64's own microns confirm run: of 3419.6 s total, stage 2 (Rocket)
is 3252.8 s = **95.1%**, stage 3 is 82.2 s and stage 4 is 84.5 s. The gradient phase is the only
lever with enough seconds in it. Cutting it frees thousands of seconds, and some of the score
lost can be bought back by spending a fraction of them on stage 4 instead - which is why both
constants move together, and why the pair had to be sized jointly rather than one at a time.

How the two constants below were chosen
----------------------------------------
From the iso-budget study ``experiments/outputs/proto_P19_microns.json``
(``experiments/proto_P19_sizing.py``), which ran the FULL pipeline from scratch at each of
E = 30,000 / 40,000 / 50,000 and drove stage 4 one cycle at a time, logging score and cumulative
wall after every cycle, to a 2600 s study cap. Two prior artifacts pin the ends of that range:
``proto_P07.json`` (the epoch grid at the champion's 5 cycles) and ``proto_H43_microns.json``
(the stage-4 recovery curve at E = 20,000, out to 120 cycles / 3401.9 s).

That last artifact is what ruled out the obvious cut. At E = 20,000 stage 4 **asymptotes short
of the champion**: -0.019188 pp at 5 cycles, -0.006149 at 20, -0.003636 at 50, -0.002708 at 100,
-0.002428 at 120 - and -0.002428 pp still exceeds the 0.002 pp microns bar. So E = 20,000 is too
deep a cut at any affordable number of cycles, and the operating point had to come from higher
in the range.

(An aside that P19 asked to have re-derived, because it read as self-contradictory: that
artifact's ``recovered_cut: false`` and the separately quoted "87.3% recovery" are BOTH correct.
The flag means "final delta > 0", which is false at -0.002428 pp. The 87.3% is the fraction *of
the cut* recovered, (0.019188 - 0.002428) / 0.019188. They answer different questions.)

The selected point is recorded in ``_SIZING`` below with the curve row it was read off.

Per-dataset composition (P14): each leg runs ITS OWN champion
--------------------------------------------------------------
``sota.json`` holds a different champion per dataset, and P14 requires a variant to compose on
the champion of the dataset it is running. Since 2026-08-27 those champions no longer share a
stage-2 surrogate:

    connectome   H64  (ASYM one-sided surrogate, 20,000 epochs)
    microns      H42  (plain sigmoid, 80,000 epochs)      <- the leg this variant re-sizes
    mouse        H63  (``_EPOCHS = 0``; stages 3, 4, 5, 6)

Hence ``_SURROGATE`` below. It is not a new mechanism and it is not a knob to tune: it names,
per dataset, the surrogate that dataset's champion already uses, so that the connectome and
mouse legs come out **bit-identical to H64 and H63 respectively** and their deltas are exactly
0 by construction. The whole measured change is on microns.

Stated plainly, because it bounds what this variant can show
--------------------------------------------------------------
* **connectome and mouse are structural no-ops.** They are controls, not evidence. This variant
  cannot improve the mission dataset and does not claim to.
* **The microns change is not free.** Any epoch cut trades score for seconds; the sizing above
  picks the operating point, it does not abolish the trade. The verdict must report the delta
  and the wall clock together, and a delta below the 0.002 pp bar is a regression to be named as
  such, not rounding.
* **This variant deliberately does not also swap microns to the ASYM surrogate**, even though
  H64's microns leg is cheaper per epoch (stage 2 at 3252.8 s vs the sigmoid's 3451 s) and its
  2 untruncated runs sit +0.005344 pp over the champion. Doing both at once would change two
  things against the H42 microns champion and make the attribution impossible. Re-running H64's
  microns leg on top of a configuration that FITS is the next step and is already recorded as
  such in ``sota.json``'s microns caveat - it is not this variant's job.

Leakage-safety and compute accounting
--------------------------------------
No stage reads the discrete oracle to choose a move; the oracle scores whole candidate rank
vectors only, exactly as in the champion. ``data/best_solution`` is never touched. The gradient
budget CHANGES on microns by design - that is the entire point - and both the requested and the
executed epoch counts are recorded in ``history.attrs`` so the accounting stays auditable.
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
from .H38 import M, T
from .H64 import _rocket_asym

ID = "H70"
HYPOTHESIS = (
    "Re-allocating microns' compute from the gradient phase to the alternating refiner -- "
    "stage-2 epochs cut from 80,000 and stage-4 cycles raised from 5, every other constant and "
    "both other datasets byte-identical to their champions -- produces a microns run that "
    "finishes with REAL MARGIN under the 3450 s guard deadline while staying non-inferior to "
    "the champion's 83.24085291200831 (delta >= -0.002 pp). P19: the shipped configuration has "
    "~1% margin and truncates, including on the unmodified champion itself, so no variant can "
    "be promoted on microns until this is fixed."
)

# -- Per-dataset SURROGATE = the surrogate that dataset's champion already uses (P14). --
# Not a tunable: it exists so connectome comes out bit-identical to H64 and mouse to H63.
_SURROGATE = {"connectome": "asym", "microns": "sigmoid", "mouse": "sigmoid"}

# -- THE ONLY CHANGE. Both entries are microns-only and were read off the curve in --------
#    experiments/outputs/proto_P19_microns.json; see _SIZING for the row and its numbers.
#    connectome/mouse values are their champions' and are untouched.
_EPOCHS = {"connectome": 20_000, "mouse": 0, "microns": 50_000}      # champion microns: 80_000
_ALT_CYCLES = {"connectome": 77, "microns": 20, "mouse": 32}         # champion microns: 5

# Provenance of the two numbers above, filled in from the sizing study. Kept in the module so
# the operating point is auditable from the code, not only from the log.
_SIZING = {
    "selected": {"epochs": 50_000, "alt_cycles": 20},
    "champion": {"epochs": 80_000, "alt_cycles": 5},
    "read_off": "experiments/outputs/proto_P19_microns.json, arm epochs=50000, cycle index 19",
    "rule": "experiments/outputs/proto_P19_sizing_rule.json (sealed at 72fd700, BEFORE the "
            "E=50000 arm finished - the arms CROSS, so a rule chosen afterwards is fitted)",
    # measured on that arm, seed 42
    "proto_best_pct": 83.25311870,
    "proto_delta_vs_champion_pp": +0.01226579,
    "proto_run_wall_s": 2338.3,
    "champion_pct": 83.24085291200831,
    "champion_run_wall_s_approx": 3418.0,
    "speedup_x": 1.46,
    "x_over_microns_bar": 6.1,
    "slowdown_tolerated_pct": 48,
    # what the rule rejected, and why - kept so the choice stays auditable
    "runner_up_faster": {"epochs": 30_000, "alt_cycles": 40, "delta_pp": +0.008467,
                         "run_wall_s": 1920.0,
                         "why_not": "1.78x faster, but 0.0058 pp below the front's best - MORE "
                                    "than one microns bar, so the rule's speed tie-break does "
                                    "not reach it"},
    "not_measured": "The epoch axis was sampled at 30k/40k/50k only. Score at a fixed budget was "
                    "MONOTONE INCREASING in E across all three, so 60k may be better still at "
                    "~2600 s; 70k+ cannot fit (rocket alone would be ~2640 s). This is the best "
                    "MEASURED allocation, not a proven optimum.",
}

# -- Everything below is the champion's, unchanged on every dataset. ---------------------
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12}
_K_FULL = 6
_ALPHA = 0.7
_ALT_SIFT_SWEEPS = {"connectome": 2, "microns": 2, "mouse": 2}
_ALT_K_FULL = 2
_MIN_BLOCK = 32

# Stages 5 and 6 exist in the MOUSE champion only; a 0 budget keeps the primary legs
# byte-identical to their champions, the same convention H63/H64 use.
_PAIR_PASSES = 2
_PAIR_MAX_POPS = {"connectome": 0, "microns": 0, "mouse": 100_000}
_RECLAIM_ROUNDS = {"connectome": 0, "microns": 0, "mouse": 1}
_RECLAIM_BUDGET = 1_000_000
_CONFLICT_BUDGET = 200_000


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run this dataset's champion pipeline, re-sized on microns only.

    Identical in structure to :func:`mfas.experiments.H64.run`. The two differences are that
    stage 2 dispatches on :data:`_SURROGATE` (so each dataset gets its own champion's
    surrogate) and that microns' entries in :data:`_EPOCHS` / :data:`_ALT_CYCLES` are the
    re-sized ones.
    """
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))
    surrogate = _SURROGATE.get(g.name, "sigmoid")

    # -- Stage 1+2: H02 warm start -> Rocket with THIS dataset's champion surrogate ----
    order = greedy_fas_order(g)
    init_positions = _init_positions_from_order(order, device)
    _rocket_fn = _rocket_asym if surrogate == "asym" else run_rocket
    rocket = _rocket_fn(g, cfg, seed=seed, device=device,
                        init_positions=init_positions, time_limit=time_limit)

    pure_best_positions = rocket.best_positions
    pure_best_score = rocket.best_score
    total = g.total_weight

    # -- Stage 3: H35's under-relaxed two-phase exact-gain sift -----------------------
    rank0 = np.argsort(np.argsort(pure_best_positions, kind="stable"),
                       kind="stable").astype(np.int64)
    sift_budget = None if time_limit is None else max(0.0, time_limit - rocket.wall_clock_s)
    sift_rank, sift_score, sweep_log = sift_underrelaxed(
        g, rank0, k_full=_K_FULL, alpha=_ALPHA,
        max_sweeps=_MAX_SWEEPS.get(g.name, 40), time_budget_s=sift_budget)
    sift_time_s = float(sum(row["wall"] for row in sweep_log))

    # -- Stage 4: alternate block refinement with a SHORT single-node sift ------------
    alt_budget = (None if time_limit is None
                  else max(0.0, time_limit - rocket.wall_clock_s - sift_time_s))
    alt_rank, alt_score, alt_log = alternate_scc_sift(
        g, sift_rank,
        n_cycles=_ALT_CYCLES.get(g.name, 32),
        sift_sweeps=_ALT_SIFT_SWEEPS.get(g.name, 2),
        k_full=_ALT_K_FULL, alpha=_ALPHA, min_block=_MIN_BLOCK,
        time_budget_s=alt_budget)
    alt_time_s = float(alt_log[-1]["cum_wall_s"]) if alt_log else 0.0

    # -- Stage 5: pair relocation - mouse only ----------------------------------------
    max_pops = _PAIR_MAX_POPS.get(g.name, 0)
    pair_rank, pair_score, pair_log, pair_time_s = alt_rank, alt_score, [], 0.0
    if max_pops:
        pair_budget = (None if time_limit is None else
                       max(0.0, time_limit - rocket.wall_clock_s - sift_time_s - alt_time_s))
        pair_rank, pair_score, pair_log = pair_relocate(
            g, alt_rank, n_passes=_PAIR_PASSES, max_pops=max_pops,
            time_budget_s=pair_budget)
        pair_time_s = float(pair_log[-1]["cum_wall_s"]) if pair_log else 0.0

    # -- Best-by-oracle across stages 2-5 ---------------------------------------------
    base_score = pure_best_score
    base_positions = pure_best_positions
    if sift_score > base_score:
        base_score, base_positions = sift_score, sift_rank.astype(np.float32)
    if alt_score > base_score:
        base_score, base_positions = alt_score, alt_rank.astype(np.float32)
    if pair_score > base_score:
        base_score, base_positions = pair_score, pair_rank.astype(np.float32)

    # -- Stage 6: minimal-FAS arc reclamation - mouse only (H63) ----------------------
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

    # -- Provenance: each stage's increment stays separately auditable ----------------
    hist = rocket.history
    hist.attrs["surrogate"] = "asym_one_sided" if surrogate == "asym" else "sigmoid"
    if surrogate == "asym":
        hist.attrs["surrogate_M"] = M
        hist.attrs["surrogate_T"] = T
    hist.attrs["surrogate_exercised"] = bool(rocket.n_epochs_done > 0)
    # The re-sizing itself, recorded per run so a pooled mean can never hide which
    # allocation produced it.
    hist.attrs["resized_dataset"] = "microns"
    hist.attrs["epochs_champion"] = {"connectome": 20_000, "microns": 80_000, "mouse": 0}.get(
        g.name)
    hist.attrs["alt_cycles_champion"] = {"connectome": 77, "microns": 5, "mouse": 32}.get(
        g.name)
    hist.attrs["is_structural_noop_vs_champion"] = g.name in ("connectome", "mouse")
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
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=(rocket.wall_clock_s + sift_time_s + alt_time_s + pair_time_s
                      + rec_time_s),
    )
