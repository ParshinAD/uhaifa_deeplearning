"""Variant H49 — replace the greedy-FAS warm start with a RATIO greedy.

Hypothesis (Phase 7, queue item H48)
------------------------------------
The champion warm-starts from :func:`mfas.experiments.H02.greedy_fas_order` (Eades-Lin-Smyth /
GreedyAbs, maximising ``out_w - in_w``). The reference family starts from a peel maximising the
RATIO ``(out_w + 1) / (in_w + 1)`` instead. On connectome that is a **+5.70387 pp** better
starting order at essentially identical cost — 74.61730 % against 68.91343 %, 20.0 s against
17.4 s (``experiments/outputs/proto_H48_connectome.json``, both scored by the frozen oracle).

The claim under test is NOT that the better start is better — that is already measured. It is
that some of the advantage **survives the refinement stack** and lands on the composed score.

What changes vs the champion
----------------------------
Exactly ONE thing: ``order = ratio_greedy_rank(g)`` instead of ``greedy_fas_order(g)``.
Every constant — ``_EPOCHS``, ``_MAX_SWEEPS``, ``_K_FULL``, ``_ALPHA``, ``_ALT_CYCLES``,
``_ALT_SIFT_SWEEPS``, ``_ALT_K_FULL``, ``_MIN_BLOCK`` — and the whole body of :func:`run` are
byte-identical to :mod:`mfas.experiments.H44`, which is the current mouse champion and is itself
byte-identical to H42 on both primaries. So ``H49 - champion`` isolates the initial order and
nothing else, and ``tests/test_experiment_H49.py`` asserts that constant-for-constant.

What the prototype already establishes, and what it does NOT
-------------------------------------------------------------
Measured through the champion's stage 3 alone (no Rocket, no stage 4):

===========  ==================  ==================  ================
dataset      champion init       ratio init          composed delta
===========  ==================  ==================  ================
connectome   83.44721 %          83.48176 %          **+0.03456 pp**
mouse        93.08288 %          92.72105 %          **-0.36183 pp**
===========  ==================  ==================  ================

Three things to hold in mind while adjudicating this variant:

* **+0.03456 pp is ~2.9x the connectome minimum effect size (0.012 pp)** — but it was measured
  with the gradient phase ABSENT. In the shipped connectome pipeline Rocket runs between the
  init and the sift, and ``killed.json`` M6-init-is-flat says the plateau washes the init out
  (a <=0.06 pp ceiling for init-alone, measured through Rocket). This variant is the direct test
  of whether that washout is total.
* **The sift already absorbs 99.4 % of the advantage.** +5.70387 pp at the init becomes
  +0.03456 pp after stage 3. Nothing here contradicts M6; it narrows it.
* **mouse goes the other way**, -0.36183 pp after stage 3, which is worse than the -0.26 pp
  non-inferiority margin. Stage 4 may recover it; it may not. The init is deliberately NOT
  dataset-keyed here — keying the ALGORITHM per dataset (as opposed to a compute budget, which
  ``_EPOCHS`` already does) is a much stronger claim and would need its own justification. If
  this variant wins connectome and loses mouse, that is a per-dataset promotion question and it
  should be reported as one, not engineered away in advance.

Runtime and determinism
-----------------------
``ratio_greedy_rank`` is ``O((n + m) log n)`` with no RNG, ties broken by node id. The pipeline
therefore still never draws from ``seed``, which is what the campaign's 1-seed screen policy and
``sota.json``'s std = 0 rest on. Cost is +2.6 s against the incumbent init on connectome, i.e.
0.2 % of a 1,152 s run — the equal-compute question does not arise.

Leakage-safety
--------------
Unchanged. The init reads only edge weights and remaining-subgraph degree sums; the frozen
oracle scores whole candidate vectors afterwards; ``data/best_solution`` is never read.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct
from ..refine import alternate_scc_sift, sift_underrelaxed
from ..baseline.ratio_greedy import ratio_greedy_rank
from .H02 import _init_positions_from_order

ID = "H49"
HYPOTHESIS = (
    "The champion warm-starts from greedy-FAS (maximising out_w - in_w), which reaches "
    "68.91343% on connectome. A ratio greedy maximising (out_w+1)/(in_w+1) reaches 74.61730% "
    "at the same cost - +5.70387 pp of starting quality. The claim is that part of that "
    "advantage SURVIVES the refinement stack: through stage 3 alone it is still +0.03456 pp, "
    "about 2.9x the connectome minimum effect size, and this variant tests whether Rocket and "
    "stage 4 wash it out."
)

# ── Stages 1-3: identical to H42 EXCEPT the mouse epoch count ───────────────────────
# THE ONLY CHANGE IN THIS FILE is mouse 5_000 -> 0, sized by the epoch grid
# (experiments/outputs/proto_P07_mouse.json). connectome and microns are untouched, so
# H44 - H42 isolates the mouse gradient phase and nothing else.
_EPOCHS = {"connectome": 20_000, "mouse": 0, "microns": 80_000}
_MAX_SWEEPS = {"connectome": 40, "mouse": 40, "microns": 12}
_K_FULL = 6
_ALPHA = 0.7

# ── Stage 4: byte-identical to H42 ──────────────────────────────────────────────────
_ALT_CYCLES = {"connectome": 77, "microns": 5, "mouse": 32}
_ALT_SIFT_SWEEPS = {"connectome": 2, "microns": 2, "mouse": 2}
_ALT_K_FULL = 2          # short warm phase: the incoming order is already refined
_MIN_BLOCK = 32          # below this the scipy call costs more than it returns


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run the H42 pipeline with the mouse gradient phase sized to zero.

    Byte-identical to :func:`mfas.experiments.H42.run`; only ``_EPOCHS["mouse"]`` differs.
    With ``epochs = 0`` ``run_rocket`` returns the warm-start order unchanged, so stage 3
    starts from the greedy-FAS order and ``pure_best_pct`` reports the greedy score.
    Returns the FINAL best (max of pure, the H35 sift, and the alternation).
    """
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))

    # ── Stage 1+2: H02 warm start -> unchanged Rocket (PURE) ─────────────────────
    order = ratio_greedy_rank(g)
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
    # H44-specific: makes "did the gradient phase run at all?" auditable per dataset.
    hist.attrs["epochs_requested"] = _EPOCHS.get(g.name, RocketConfig.epochs)
    hist.attrs["refined_best_score"] = best_score
    hist.attrs["refined_best_pct"] = best_pct
    hist.attrs["sift_log"] = sweep_log

    return RocketResult(
        best_positions=best_positions,
        best_score=best_score,
        best_pct=best_pct,
        history=hist,
        n_epochs_done=rocket.n_epochs_done,
        wall_clock_s=rocket.wall_clock_s + sift_time_s + alt_time_s,
    )
