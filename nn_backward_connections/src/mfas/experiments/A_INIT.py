"""Variant A_INIT — very tight position initialization (roadmap A-INIT, TODO 5).

Hypothesis (backlog A-INIT): the *scale* of Rocket's initial positions is not a neutral
knob. Shrinking the paper's ``N(0,1)`` init by a large factor (``std = 1e-4``) — changing
NOTHING else — beats the baseline on the exact feedforward metric and removes its
seed-to-seed variance.

Mechanism (derived, then measured — ``experiments/proto_ainit_scale.py``)
------------------------------------------------------------------------
The surrogate is ``F(P) = sum_e w_hat_e * sigma(beta * (p_tgt - p_src))``. Expanding
``sigma(x) = 1/2 + x/4 - x^3/48 + O(x^5)`` and collecting per node, with the weight
imbalance ``c_k = out_w_hat(k) - in_w_hat(k)``:

    F(P) = W_hat/2 - (beta/4) <c, P> - (beta^3/48) sum_e w_hat_e D_e^3 + O((beta*D)^5)

so ``grad_k F = -(beta/4) c_k + O((beta*std)^2)``. Every pairwise ("who precedes whom")
term is suppressed as **(beta*std)^2** — measured slope 2.006 (mouse) / 1.996 (hard
synthetic), ``cos(grad F, -c) = 1.000000`` at ``beta*std = 1e-6``.

Consequence: from a tight init the first phase of optimization is driven by the constant
vector ``-c``, i.e. Rocket **warm-starts itself from the weight-imbalance signal** — the
same quantity GreedyAbs ranks on — instead of spending gradient steps undoing an
uninformative random draw. Measured on the proxies: the sign(c) split explains **96.5%**
(mouse) / **93.2%** (synthetic) of position variance after 20 steps, and the final order
keeps essentially **no memory of the init** (Spearman(final, init) = +0.03 / -0.02 at
std <= 1e-4, vs +0.14 / +0.12 at the baseline std = 1). Because the init is forgotten,
the run becomes **seed-independent** (std = 0.0000 pp over 3 seeds on both proxies).

The same mechanism predicts the flip side, also measured: a tight init erases an
*informative* init too. Compressing H02's greedy-FAS warm start to std = 1e-4 lands on
exactly the tight-random score (mouse 92.4359 both) — so A_INIT is an **alternative** to
H02's warm start, never an addition to it.

Isolation
---------
This is a pure one-constant change against ``baseline_passthrough``: identical epoch
budget, loss, optimizer, LR / beta schedules — and even the *identical random draw*.
``torch.manual_seed(seed)`` then ``torch.randn(n)`` reproduces exactly the vector
``run_rocket`` would have drawn for the baseline (``run_rocket`` re-seeds with the same
seed on entry and consumes no RNG before its own init draw), so the variant differs from
the baseline by the scalar factor ``INIT_STD`` and nothing else.

Leakage-safety: the init is a scaled Gaussian — it reads neither the graph nor the
oracle. The oracle is used only as the baseline uses it (best-by-oracle tracking inside
``run_rocket``). Compute-matched: same ``epochs``, so ``n_epochs_done`` equals the
baseline's gradient budget; the change costs 0 extra steps and 0 extra wall-clock.

Prototype evidence: ``experiments/outputs/proto_ainit_scale.json`` (+ the two PNGs).
"""
from __future__ import annotations

from typing import Optional

import torch

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData

ID = "A_INIT"
HYPOTHESIS = (
    "Shrinking Rocket's N(0,1) init to std=1e-4 (nothing else changed) beats the "
    "baseline on the exact feedforward metric and removes its seed variance, because "
    "at small beta*std the gradient reduces to the weight-imbalance vector -(beta/4)c, "
    "so the optimizer self-warm-starts from the imbalance signal instead of undoing an "
    "uninformative random draw."
)

# Per-dataset epoch budget — identical to baseline_passthrough (compute-matched).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000, "microns": 80_000}

# The one changed constant. The proto sweep is flat over std in [1e-6, 1e-2]
# (mouse 92.4359/92.4359/92.4285, synthetic 73.8395/73.8494/73.7900); 1e-4 sits in the
# middle of that plateau, far above float32 resolution and far below the baseline's 1.0.
INIT_STD = 1e-4


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run the unchanged baseline Rocket from a tightly-scaled random init."""
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))

    # Same draw as the baseline's internal init, scaled. (run_rocket re-seeds with the
    # same seed and draws nothing before its init, so baseline_init == this / INIT_STD.)
    torch.manual_seed(seed)
    init_positions = torch.randn(g.n_nodes, device=device) * INIT_STD

    return run_rocket(g, cfg, seed=seed, device=device,
                      init_positions=init_positions, time_limit=time_limit)
