"""H73 - the exact-gain sift's argmax TIE-BREAK, chosen instead of inherited.

The change
----------
``jacobi_best_gaps`` returns each node's exact-optimal insertion gap. That optimum is
almost never a single gap: the insertion profile is piecewise-constant with at most
``deg(u)`` breakpoints over ``n`` gaps, so its maximum is attained on an INTERVAL. The
production rule returns the smallest maximizing breakpoint, with gap 0 winning ties
(insertion.py, "ties -> smallest b, fine"). Both are LEFTWARD, which is
direction-asymmetric: a node whose plateau lies to its right lands on the NEAR edge, one
whose plateau lies to its left is transported to the FAR edge -- for the same exact gain.

H73 sets ``tie_break="mindisp"``: among the SAME maximizing gaps, take the one nearest the
node's current rank. Deterministic, RNG-free, one extra segmented min, runtime-neutral
(which matters under P19). Under-relaxation is what makes the choice load-bearing rather
than cosmetic -- ``underrelaxed_rebuild`` interpolates toward the selected gap, so the
choice sets the transport DISTANCE, not just the destination.

Everything else is H64, the connectome champion. Stages 1-2 (greedy-FAS warm start ->
Rocket with the ASYMMETRIC surrogate) are IMPORTED from ``H64`` rather than copied, so they
are identical by construction; stages 3-6 are H64's ``run`` with ``tie_break`` threaded
into the two calls that consume the sift. The gradient budget is unchanged.

Measured premise (rung 1, ``experiments/outputs/proto_H73_connectome.json``)
---------------------------------------------------------------------------
On a real stage-3 trajectory: 95.0-99.7% of movers sit on a plateau of width > 1 (the
item's free pre-gate was 1%), median plateau width 574-1646 ranks, and the rule changes
the destination for 44-75% of movers, cutting mean transport by 800-4,961 ranks per mover.
On the H64 champion's own final order: 67 movers, 28.4% on plateaus, 16.4% redirected.

Leakage-safety
--------------
The tie-break reads only the current rank vector and the exact profile built from input
edge weights. The frozen oracle still decides acceptance of whole candidate vectors and
nothing else; ``data/best_solution`` is never touched.

Reproduce
---------
    PYTHONPATH=src python eval/harness.py --exp H73 --dataset connectome --seed 42         --role implement
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult
from ..io import GraphData
from ..metrics import pct, score_from_order
from ..refine import alternate_scc_sift, sift_underrelaxed
from ..refine.pair_relocate import pair_relocate
from ..refine.reclaim2 import reclaim_arcs_fast
from .H02 import _init_positions_from_order, greedy_fas_order
from .H38 import M, T
from .H64 import (_ALPHA, _ALT_CYCLES, _ALT_K_FULL, _ALT_SIFT_SWEEPS, _CONFLICT_BUDGET,
                  _EPOCHS, _K_FULL, _MAX_SWEEPS, _MIN_BLOCK, _PAIR_MAX_POPS, _PAIR_PASSES,
                  _RECLAIM_BUDGET, _RECLAIM_ROUNDS, _rocket_asym)

ID = "H73"
HYPOTHESIS = (
    "Replacing the exact-gain sift's argmax tie-break -- first maximizing breakpoint, gap 0 "
    "wins ties, i.e. always the LEFTMOST gap of the optimal plateau -- with the maximizing "
    "gap NEAREST the node's current rank changes the final connectome score by more than "
    "0.012 pp. Two-sided on purpose: a significant negative would show the leftward bias is "
    "load-bearing, which is equally a finding. Same exact gain, same mover set, same "
    "complexity, no RNG; only the destination within the optimal plateau changes."
)

# The one knob. "first" reproduces H64 bit-identically (tests/test_H73_tiebreak.py).
_TIE_BREAK = "mindisp"


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """H64's pipeline with the sift tie-break set to minimum displacement (H73)."""
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
        max_sweeps=_MAX_SWEEPS.get(g.name, 40), time_budget_s=sift_budget,
        tie_break=_TIE_BREAK)
    sift_time_s = float(sum(row["wall"] for row in sweep_log))

    # -- Stage 4: alternate block refinement with a SHORT single-node sift --------
    alt_budget = (None if time_limit is None
                  else max(0.0, time_limit - rocket.wall_clock_s - sift_time_s))
    alt_rank, alt_score, alt_log = alternate_scc_sift(
        g, sift_rank,
        n_cycles=_ALT_CYCLES.get(g.name, 32),
        sift_sweeps=_ALT_SIFT_SWEEPS.get(g.name, 2),
        k_full=_ALT_K_FULL, alpha=_ALPHA, min_block=_MIN_BLOCK,
        time_budget_s=alt_budget, tie_break=_TIE_BREAK)
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
    hist.attrs["tie_break"] = _TIE_BREAK
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
