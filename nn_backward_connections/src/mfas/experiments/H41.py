"""Variant H41 — add a THIRD move class to stage 4: exact-gain bounded-span segment moves.

Hypothesis (Phase 7, queue item H41)
------------------------------------
The champion :mod:`mfas.experiments.H42` alternates exactly two move classes, and both are
blind to the same failure:

* stage 3 / the inner sift (:func:`mfas.refine.underrelax.sift_underrelaxed`) relocates ONE
  node at a time, so it cannot make a move that is only profitable **jointly** — a run of
  nodes that belongs 800 positions earlier is invisible if no single member profits alone;
* the block refiner (:func:`mfas.refine.scc_recursive.scc_recursive_refine`) relocates a
  group only when that group falls out of a strong decomposition, and never moves a node
  relative to its own SCC.

``experiments/diagnosis.md`` measures Kendall-tau 0.610 and a median rank-distance of
~22,580 against the 84.6147% reference, with the disagreement spread uniformly over weight
buckets and sitting on median-degree (not hub) endpoints. That is the signature of whole
REGIONS sitting in the wrong place — exactly the hole between the two existing classes.
H41 adds :func:`mfas.refine.segment.segment_refine` as a third stage inside the stage-4
alternation: take a contiguous run of positions and move it, as a rigid unit, to a nearby
target position.

The exact-gain lemma (one paragraph)
------------------------------------
Moving the segment ``S = [i, i+L)`` right by ``d`` positions is exactly the swap of the two
ADJACENT contiguous blocks ``S`` and ``J = [i+L, i+L+d)``. Under such a swap every edge
inside ``S``, inside ``J``, or with at most one endpoint in ``S ∪ J`` keeps its orientation
(the same contiguous-block lemma stage 4 already relies on); **only ``S``–``J`` cross edges
flip**. So the exact delta is ``W(J→S) − W(S→J)``, computable in closed form without
rescoring the graph. Only strictly-positive exact gains are applied and accepted moves have
pairwise disjoint windows (disjoint windows compose exactly), so a sweep is monotone
non-decreasing by construction and alternates safely with the other two stages.

What changes vs H42
-------------------
Exactly ONE thing: stage 4 calls
:func:`mfas.refine.segment.alternate_scc_sift_segment` instead of
:func:`mfas.refine.scc_recursive.alternate_scc_sift`. Every constant — epochs, stage-3
sweeps, ``k_full``, ``alpha``, ``min_block``, the stage-4 cycle counts and the inner sift
sweeps — is byte-identical to H42, so ``H41 − H42`` isolates the MOVE CLASS and nothing
else. ``tests/test_experiment_H41.py`` asserts that constant-for-constant equality rather
than trusting it to stay true.

Prototype evidence (cheap proxies only; ``experiments/outputs/proto_H41.json``)
------------------------------------------------------------------------------
Matched wall-clock, same starting order, champion stage 4 given a cycle count far above what
the budget allows so the CLOCK — not the configuration — stops both arms::

    fixture     budget    A champion      B +segment      B − A
    mouse       10 s      93.082880       93.141754      +0.058873 pp
    hard400     12 s      75.394752       75.406625      +0.011872 pp
    hard6000    20 s      74.613343       74.650595      +0.037252 pp
    hard6000    60 s      74.613343       74.650595      +0.037252 pp   <- both CONVERGED

The 60 s row is the load-bearing one: at 3x the budget both arms return identical scores
(435 vs 139 cycles for A, 326 vs 100 for B), so ``B − A`` is a difference between two FIXED
POINTS and cannot be an extra-compute artefact. On mouse arm A is dead flat over 7,432
cycles while B gains — the champion's two classes are genuinely exhausted there.

Honest caveats (read these before adjudicating the screen)
----------------------------------------------------------
* **The bounded-span regime is untested.** The default ladder's widest window is 2048
  positions. On the proxies that is 34% (hard6000) to 64% (hard400) of the whole line; on
  connectome it is **1.5%** (2048 / 136,648). No cheap proxy reaches that ratio, so nothing
  measured so far predicts how much a 1.5%-of-the-line window can recover.
* **The motivating diagnosis argues for a longer ladder than this ships.** The median
  rank-distance to the reference is ~22,580 — 11x beyond the widest default rung. A longer
  ladder is affordable (an extended ``2^0…2^16`` grid measured 7.5–8.7 s/sweep at n=136,648
  / m=2.6 M) but interacts badly with the greedy disjoint packing (17 applied moves per
  sweep vs 97), so it needs its own tuning cycle. **Deliberately NOT folded into this
  variant** — one variable at a time.
* **Effect sizes are small.** ``B − A`` on the proxies is +0.012 to +0.059 pp against a
  connectome minimum effect size of 0.012 pp. The proxies are at or barely above it.
* **The move class alone is weak.** ``segment_refine`` on its own reaches a fixed point in
  2–3 sweeps for +0.005 to +0.030 pp. The value is in the alternation, not the class.
* **Segment credit >= net advantage.** On hard6000 the segment stage is credited +0.0639 pp
  but the arm ends only +0.0373 pp above the champion — the other two classes reach some of
  the same orders. ``seg_increment_pp`` below is a credit figure, NOT the variant's gain.
* **microns wall-clock is the risk.** H42's microns run already sits ~68 s (2.0%) inside the
  runtime guard's 3450 s deadline. H41 adds one segment sweep per stage-4 cycle (5 cycles on
  microns). That was never measured at microns scale here (a GPU measurement was in flight),
  so the added seconds are an ESTIMATE. If it does not fit, the guard aborts at a stage
  boundary and ``eval/runtime_guard.summarise`` marks the run ``degraded`` via
  ``n_alt_cycles < alt_cycles_requested`` — a recorded truncation, not a silent one.

Runtime + determinism
---------------------
``time_limit`` is honoured exactly as in H42: the remaining budget is threaded into stage 3
and then into stage 4, and ``alternate_scc_sift_segment`` checks it at each cycle boundary
and passes the remainder down to each of its three inner stages. Stage 4 is still sized by
CYCLE COUNT, never by wall-clock, so the deadline is an abort and not a sizing rule.

The segment refiner draws no random numbers at all — not even for a tie-break. Ties in the
per-start ``(L, d)`` argmax go to the smallest segment then the shortest travel (fixed
iteration order); ties in the greedy move selection go to the lowest start index. The whole
pipeline therefore still never draws from ``seed``, which is what the campaign's 1-seed
screen policy and ``sota.json``'s std = 0 rest on (``autoresearch/seed_class.py`` verifies
this mechanically from the call graph).

Leakage-safety
--------------
Unchanged from H42: every move is decided from the input edge weights and the current ranks
alone (a closed-form gain), the frozen oracle only accepts/rejects whole candidate vectors,
and ``data/best_solution`` is never read.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct
from ..refine import DEFAULT_OFFSETS, DEFAULT_SEG_LENGTHS, sift_underrelaxed
from ..refine.segment import alternate_scc_sift_segment
from .H02 import _init_positions_from_order, greedy_fas_order

ID = "H41"
HYPOTHESIS = (
    "The champion's two move classes (single-node insertion, SCC block decomposition) are "
    "both blind to a contiguous REGION sitting in the wrong place: insertion cannot make a "
    "jointly-profitable move and SCC decomposition can only relocate an SCC-separable group. "
    "Adding exact-gain bounded-span SEGMENT moves - relocate a contiguous run of positions "
    "as a rigid unit, with a closed-form delta - reaches a strictly better joint fixed point "
    "than the champion's stage 4 does, at the same cycle count and 0 extra gradient steps."
)

# ── Stages 1-3: byte-identical to H42 (hence to H36 / H35) ───────────────────────────
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12}
_K_FULL = 6
_ALPHA = 0.7

# ── Stage 4: cycle counts and inner-sift sweeps byte-identical to H42 ────────────────
# THE ONLY CHANGE is which alternation driver runs below. Holding these fixed is what
# makes H41 - H42 an isolation of the move class; tests/test_experiment_H41.py asserts it.
_ALT_CYCLES = {"connectome": 77, "microns": 5, "mouse": 32}
_ALT_SIFT_SWEEPS = {"connectome": 2, "microns": 2, "mouse": 2}
_ALT_K_FULL = 2          # short warm phase: the incoming order is already refined
_MIN_BLOCK = 32          # below this the scipy call costs more than it returns

# ── The new stage's only constants ───────────────────────────────────────────────────
# One segment sweep per alternation cycle: the sweep is monotone and reaches its own fixed
# point in 2-3 sweeps, so extra sweeps inside a cycle mostly re-scan; the alternation is what
# re-opens moves. The ladders are the module defaults (2^0..2^10 for BOTH segment length and
# travel distance). Ladder tuning is deliberately a SEPARATE follow-up - see the docstring.
_SEG_SWEEPS = 1
_SEG_LENGTHS = DEFAULT_SEG_LENGTHS
_SEG_OFFSETS = DEFAULT_OFFSETS


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run the H42 pipeline with a THIRD move class inside stage 4.

    Identical to :func:`mfas.experiments.H42.run` except that stage 4 calls
    :func:`mfas.refine.segment.alternate_scc_sift_segment` instead of
    :func:`mfas.refine.scc_recursive.alternate_scc_sift`. Returns the FINAL best (max of
    pure Rocket, the H35 sift, and the alternation); the pure-Rocket and post-sift bests are
    reported separately via ``history.attrs`` so each stage's increment stays separable.
    """
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))

    # ── Stage 1+2: H02 warm start -> unchanged Rocket (PURE) ─────────────────────
    order = greedy_fas_order(g)
    init_positions = _init_positions_from_order(order, device)
    rocket = run_rocket(g, cfg, seed=seed, device=device,
                        init_positions=init_positions, time_limit=time_limit)

    pure_best_positions = rocket.best_positions
    pure_best_score = rocket.best_score
    pure_best_pct = rocket.best_pct
    total = g.total_weight

    # ── Stage 3: H35's under-relaxed two-phase exact-gain sift ───────────────────
    rank0 = np.argsort(np.argsort(pure_best_positions, kind="stable"),
                       kind="stable").astype(np.int64)
    sift_budget = None if time_limit is None else max(0.0, time_limit - rocket.wall_clock_s)
    sift_rank, sift_score, sweep_log = sift_underrelaxed(
        g, rank0, k_full=_K_FULL, alpha=_ALPHA,
        max_sweeps=_MAX_SWEEPS.get(g.name, 40), time_budget_s=sift_budget)
    sift_time_s = float(sum(row["wall"] for row in sweep_log))

    # ── Stage 4: block refine + SHORT single-node sift + SEGMENT sweep ───────────
    alt_budget = (None if time_limit is None
                  else max(0.0, time_limit - rocket.wall_clock_s - sift_time_s))
    alt_rank, alt_score, alt_log = alternate_scc_sift_segment(
        g, sift_rank,
        n_cycles=_ALT_CYCLES.get(g.name, 32),
        sift_sweeps=_ALT_SIFT_SWEEPS.get(g.name, 2),
        k_full=_ALT_K_FULL, alpha=_ALPHA, min_block=_MIN_BLOCK,
        seg_sweeps=_SEG_SWEEPS, seg_lengths=_SEG_LENGTHS, offsets=_SEG_OFFSETS,
        time_budget_s=alt_budget)
    alt_time_s = float(alt_log[-1]["cum_wall_s"]) if alt_log else 0.0

    # ── Best-by-oracle across every stage ────────────────────────────────────────
    best_score = pure_best_score
    best_positions = pure_best_positions
    if sift_score > best_score:
        best_score, best_positions = sift_score, sift_rank.astype(np.float32)
    if alt_score > best_score:
        best_score, best_positions = alt_score, alt_rank.astype(np.float32)
    best_pct = pct(best_score, total)

    # ── Provenance: each stage's increment stays separately auditable ────────────
    hist = rocket.history
    hist.attrs["pure_best_score"] = pure_best_score
    hist.attrs["pure_best_pct"] = pure_best_pct
    hist.attrs["sift_best_score"] = sift_score
    hist.attrs["sift_best_pct"] = pct(sift_score, total)
    hist.attrs["n_sift_sweeps"] = len(sweep_log)
    # P05: how many were ASKED for, and whether a short log means convergence rather than
    # the wall-clock guard cutting the stage off. Read by eval/runtime_guard.summarise.
    hist.attrs["sift_sweeps_requested"] = _MAX_SWEEPS.get(g.name, 40)
    hist.attrs["sift_converged"] = bool(sweep_log and sweep_log[-1]["n_movers"] == 0)
    hist.attrs["sift_time_s"] = sift_time_s
    hist.attrs["sift_alpha"] = _ALPHA
    hist.attrs["sift_k_full"] = _K_FULL
    hist.attrs["alt_best_score"] = alt_score
    hist.attrs["alt_best_pct"] = pct(alt_score, total)
    hist.attrs["alt_increment_pp"] = pct(alt_score, total) - pct(sift_score, total)
    hist.attrs["n_alt_cycles"] = len(alt_log)
    # P05: alternate_scc_sift_segment's loop breaks on the time budget and nothing else, so
    # n_alt_cycles < alt_cycles_requested is an EXACT signal that the guard truncated it.
    # eval/runtime_guard.summarise reads exactly this pair.
    hist.attrs["alt_cycles_requested"] = _ALT_CYCLES.get(g.name, 32)
    hist.attrs["alt_time_s"] = alt_time_s
    hist.attrs["alt_min_block"] = _MIN_BLOCK
    hist.attrs["alt_sift_sweeps"] = _ALT_SIFT_SWEEPS.get(g.name, 2)
    hist.attrs["alt_log"] = alt_log

    # ── H41-specific: the three move classes' credit, kept SEPARATE ──────────────
    # These are per-cycle increments summed over the run, so they add up to alt_increment_pp
    # with no double counting. seg_increment_pp is a CREDIT figure — the variant's gain is
    # H41 minus H42 at the same constants, not this number (see the module docstring).
    hist.attrs["alt_seg_sweeps"] = _SEG_SWEEPS
    hist.attrs["alt_seg_lengths"] = list(_SEG_LENGTHS)
    hist.attrs["alt_seg_offsets"] = list(_SEG_OFFSETS)
    hist.attrs["scc_increment_pp"] = float(sum(r["d_scc_pp"] for r in alt_log))
    hist.attrs["alt_sift_increment_pp"] = float(sum(r["d_sift_pp"] for r in alt_log))
    hist.attrs["seg_increment_pp"] = float(sum(r["d_seg_pp"] for r in alt_log))
    hist.attrs["n_segment_moves"] = int(sum(r["seg_moves"] for r in alt_log))
    hist.attrs["alt_scc_time_s"] = float(sum(r["t_scc_s"] for r in alt_log))
    hist.attrs["alt_sift_time_s"] = float(sum(r["t_sift_s"] for r in alt_log))
    hist.attrs["alt_seg_time_s"] = float(sum(r["t_seg_s"] for r in alt_log))

    hist.attrs["refined_best_score"] = best_score
    hist.attrs["refined_best_pct"] = best_pct
    hist.attrs["sift_log"] = sweep_log

    return RocketResult(
        best_positions=best_positions,
        best_score=best_score,
        best_pct=best_pct,
        history=hist,
        # UNCHANGED gradient budget: stages 3 and 4 add 0 optimizer steps.
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=rocket.wall_clock_s + sift_time_s + alt_time_s,
    )
