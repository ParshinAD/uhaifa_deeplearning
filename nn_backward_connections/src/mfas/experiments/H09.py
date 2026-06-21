"""Variant H09 — Anti-tie / near-equal-position handling (recover strict-`>` lost edges).

Hypothesis (backlog H09): the exact feedforward oracle counts ``pos[tgt] > pos[src]``
*strictly* (see ``mfas.metrics._ff_weight``: ``ties are counted as NOT feedforward``). So any
edge whose two endpoints land at exactly-equal (or numerically indistinguishable) positions
contributes ZERO discrete feedforward weight even when the intended order is correct.
Encouraging position separation — here a target-BLIND symmetric tie-break applied to the
positions before the (unchanged) oracle scores them — should recover that silently-dropped
feedforward weight "for free", without changing the optimization basin or dynamics.

Opportunity sizing (measured on the logged baseline plateau positions, both datasets)
------------------------------------------------------------------------------------
On the logged baseline final positions there are **zero exact ties** (``d == 0``) on either
dataset. The continuous Adam optimizer spreads positions apart, so near-ties are vanishingly
rare: connectome has only ~6 edges with ``|d| < 1e-3`` (recoverable currently-feedback weight
~20 / 41,912,141 ≈ 0.00005 pp) and the mouse graph has **zero** edges with ``|d| < 1e-2``.
The recoverable set is therefore ~3 orders of magnitude below the screen thresholds
(connectome 0.04 pp, mouse 0.52 pp). H09 is expected to be a no-op / self-falsify; this module
implements it faithfully so the screen can CONFIRM that empirically.

Chosen arm: post-hoc, target-BLIND, symmetric tie-break jitter
--------------------------------------------------------------
The arm is a *scoring-time* position transform applied to the positions the baseline Rocket
run returns (``run_rocket`` is used UNCHANGED — same N(0,1) init, loss, Adam, grad-clip, LR and
β schedules, same epoch budget). After the run we add to each node ``u`` a tiny deterministic
jitter:

    jitter[u] = EPS_JITTER * h(u, seed)

where ``h(u, seed)`` is a fixed permutation-free hash of the node *index* and the run seed,
mapped to the symmetric interval ``[-1, +1]`` (mean ~0). The jittered positions are then scored
by the FROZEN oracle, and we keep whichever of {raw positions, jittered positions} the oracle
scores higher (best-by-oracle tracking — exactly the candidate-tracking the baseline already
does; never folded into a differentiable loss, never hardcoded, never special-cased per
dataset).

Why this tie-break is TARGET-BLIND (no leakage)
-----------------------------------------------
The jitter ``h(u, seed)`` depends ONLY on the node index and the run seed. It does NOT look at:
  * which direction (``src -> tgt`` or ``tgt -> src``) a tie would score higher,
  * the edge list, edge weights, the surrogate, or the discrete oracle score.
Because the perturbation of an endpoint is the same regardless of whether that endpoint is a
source or a target of any given edge, the *sign* of ``jitter[tgt] - jitter[src]`` for a tied
edge is a deterministic function of two node hashes that were fixed before any scoring — it is
symmetric in expectation and cannot be steered toward the favourable orientation. The oracle is
consulted ONLY to pick the better of the two whole-vector candidates (raw vs jittered), which is
the same best-by-oracle candidate selection the baseline performs on its own iterates; it never
chooses a *per-edge* orientation. EPS_JITTER is far smaller than the spacing between distinct
positions, so for the (overwhelming majority of) edges that are NOT ties the jitter cannot flip
their orientation — it only resolves exact / sub-EPS ties.

Compute-matched
---------------
Standard knob-swap (position post-processing only). The gradient budget is the baseline's
(connectome 20k, mouse 5k); ``run_rocket`` is invoked unchanged, so ``n_epochs_done`` =
baseline optimizer steps and ``total_grad_steps`` matches ``baseline_passthrough`` at the same
seeds. The added cost is O(n) jitter + one extra O(m) oracle score per run (negligible vs the
gradient budget). The comparator is ``baseline_passthrough`` / the frozen baseline at matched
seeds.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData
from ..metrics import pct, score_from_positions

ID = "H09"
HYPOTHESIS = (
    "A target-blind, deterministic, symmetric per-node tie-break jitter applied to the "
    "baseline Rocket positions before the strict-`>` oracle scores them recovers feedforward "
    "weight from exact/near-tie edges 'for free', without changing the optimization basin."
)

# Per-dataset epoch budget (= baseline single-run budget; standard knob-swap).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000}

# Jitter magnitude: small enough that it can only resolve exact / sub-EPS ties and cannot flip
# any edge whose endpoints are at genuinely distinct positions. The baseline plateau positions
# span O(1) magnitude with smallest non-tie gaps >> 1e-9, so 1e-9 is a safe tie-only scale.
EPS_JITTER = 1e-9


def _symmetric_jitter(n: int, seed: int) -> np.ndarray:
    """Deterministic, target-blind, ~zero-mean jitter in [-EPS_JITTER, +EPS_JITTER].

    Depends ONLY on node index and the run seed (NOT on edges, weights, the surrogate, or the
    oracle). A SeedSequence-seeded PRNG over node indices gives a reproducible permutation-free
    hash; centring to [-1, 1] keeps it symmetric so it cannot be biased toward the scoring
    direction of any edge.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x9E09]))
    u = rng.random(n, dtype=np.float64)          # in [0, 1)
    return (EPS_JITTER * (2.0 * u - 1.0)).astype(np.float64)  # in [-EPS, +EPS], mean ~0


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run UNCHANGED baseline Rocket, then apply target-blind anti-tie jitter at scoring time.

    The baseline Rocket result (raw positions) is computed first. We then build a jittered copy
    and let the FROZEN oracle score both; the returned ``RocketResult`` carries whichever scores
    higher (best-by-oracle candidate selection — never lowers the baseline best by construction).
    """
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))
    base = run_rocket(g, cfg, seed=seed, device=device, time_limit=time_limit)

    src_np, tgt_np = np.asarray(g.src), np.asarray(g.tgt)
    total_weight = g.total_weight

    raw_pos = np.asarray(base.best_positions, dtype=np.float32)
    raw_score = float(base.best_score)

    # Target-blind symmetric jitter (float64 add so sub-float32 ties can be resolved exactly).
    jitter = _symmetric_jitter(g.n_nodes, seed)
    jit_pos = (raw_pos.astype(np.float64) + jitter)
    jit_score = score_from_positions(jit_pos, src_np, tgt_np, g.weight)

    if jit_score > raw_score:
        best_positions = jit_pos.astype(np.float32)
        best_score = jit_score
    else:
        best_positions = raw_pos
        best_score = raw_score

    return RocketResult(
        best_positions=best_positions,
        best_score=best_score,
        best_pct=pct(best_score, total_weight),
        history=base.history,
        n_epochs_done=base.n_epochs_done,
        wall_clock_s=base.wall_clock_s,
    )
