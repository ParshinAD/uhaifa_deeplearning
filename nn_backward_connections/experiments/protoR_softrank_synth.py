"""Phase-4 Stage-B DIRECTION R — soft-rank (H19) vs baseline Rocket on the HARD synthetic.

PROTOTYPE comparison script (writes nothing to results/; the connectome harness only
knows baseline_rocket and run_variant only loads the two real datasets, so the synthetic
comparison is run here directly with the FROZEN scorer
``mfas.metrics.score_from_positions``). Equal compute: both soft-rank and baseline use the
same ``SYNTH_EPOCHS`` budget on the SAME generated graph per seed.

Reports per seed and mean±std: reference% (diagnostic best-known), baseline-Rocket%,
soft-rank%, and the deltas. Run via the `allen` env:

    python experiments/protoR_softrank_synth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas.analysis.gap import make_hard_synthetic_graph
from mfas.baseline.rocket import RocketConfig, run_rocket
from mfas.metrics import pct, score_from_positions
from mfas.experiments import H19

# Locked hard-synthetic config (cfg0 from the H21 tuning: stable +0.80 pp gap).
SYNTH_PARAMS = dict(n=400, avg_out=10, feedback_frac=0.65, n_clusters=8,
                    intra_cycle_frac=0.55, weight_alpha=2.0)
SYNTH_EPOCHS = 4000        # equal-compute budget for BOTH baseline and soft-rank
SEEDS = (42, 123, 999)


def baseline_pct(g, seed, device):
    res = run_rocket(g, RocketConfig(epochs=SYNTH_EPOCHS), seed=seed, device=device)
    return res.best_pct


def softrank_pct(g, seed, device):
    # H19.run keys its epoch budget off g.name; the generator names it 'hard_synthetic'
    # which maps to 4000 in H19._EPOCHS (== SYNTH_EPOCHS).
    res = H19.run(g, seed=seed, device=device)
    # authoritative re-score with the frozen oracle (mirrors run_variant)
    sc = score_from_positions(res.best_positions, np.asarray(g.src),
                              np.asarray(g.tgt), g.weight)
    assert sc == res.best_score
    return pct(sc, g.total_weight)


def main():
    assert H19._EPOCHS["hard_synthetic"] == SYNTH_EPOCHS, "epoch budget mismatch"
    device = torch.device("cpu")
    ref, base, soft = [], [], []
    print(f"hard synthetic: {SYNTH_PARAMS} | epochs={SYNTH_EPOCHS}")
    for s in SEEDS:
        g, ref_order, ref_pct = make_hard_synthetic_graph(seed=s, **SYNTH_PARAMS)
        b = baseline_pct(g, s, device)
        r = softrank_pct(g, s, device)
        ref.append(ref_pct); base.append(b); soft.append(r)
        print(f"  seed {s}: reference={ref_pct:.4f}  baseline={b:.4f}  "
              f"softrank={r:.4f}  (soft-base={r-b:+.4f}, ref-base={ref_pct-b:+.4f})")
    ref = np.array(ref); base = np.array(base); soft = np.array(soft)
    print()
    print(f"reference%  : {ref.mean():.4f} ± {ref.std():.4f}")
    print(f"baseline%   : {base.mean():.4f} ± {base.std():.4f}")
    print(f"softrank%   : {soft.mean():.4f} ± {soft.std():.4f}")
    print(f"Δ soft-base : {(soft-base).mean():+.4f} ± {(soft-base).std():.4f} pp")
    print(f"Δ ref-base  : {(ref-base).mean():+.4f} ± {(ref-base).std():.4f} pp (the gap)")


if __name__ == "__main__":
    main()
