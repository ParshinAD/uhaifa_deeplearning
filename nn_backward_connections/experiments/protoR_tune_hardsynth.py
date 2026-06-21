"""Phase-4 Stage-B DIRECTION R — tune the HARD synthetic so baseline Rocket plateaus
≥~0.5 pp BELOW the greedy-FAS reference order (the H21 gate's gate).

DIAGNOSTIC / PROTOTYPE script (NOT a reportable Rocket run; writes nothing to results/).
Sweeps a few generator settings, runs UNCHANGED baseline Rocket (random init) on each at
a fixed small epoch budget, and prints (reference%, baseline-Rocket plateau%, gap) over a
few seeds. Run via the `allen` env:

    python experiments/protoR_tune_hardsynth.py
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

SYNTH_EPOCHS = 4000        # prototype budget (used for BOTH baseline and soft-rank)
SEEDS = (42, 123, 999)


def baseline_plateau(g, epochs, seed, device):
    cfg = RocketConfig(epochs=epochs)
    res = run_rocket(g, cfg, seed=seed, device=device)
    return res.best_pct


def main():
    device = torch.device("cpu")   # n<=600: CPU is fine & deterministic for scoring
    configs = [
        dict(n=400, avg_out=10, feedback_frac=0.65, n_clusters=8,
             intra_cycle_frac=0.55, weight_alpha=2.0),
        dict(n=400, avg_out=10, feedback_frac=0.80, n_clusters=8,
             intra_cycle_frac=0.70, weight_alpha=2.0),
        dict(n=400, avg_out=12, feedback_frac=0.90, n_clusters=10,
             intra_cycle_frac=0.80, weight_alpha=2.5),
        dict(n=500, avg_out=12, feedback_frac=0.85, n_clusters=10,
             intra_cycle_frac=0.75, weight_alpha=2.5),
        dict(n=300, avg_out=10, feedback_frac=0.75, n_clusters=6,
             intra_cycle_frac=0.65, weight_alpha=2.0),
    ]
    for ci, params in enumerate(configs):
        ref_pcts, base_pcts, gaps = [], [], []
        n_edges = None
        for s in SEEDS:
            g, ref_order, ref_pct = make_hard_synthetic_graph(seed=s, **params)
            n_edges = g.n_edges
            base = baseline_plateau(g, SYNTH_EPOCHS, seed=s, device=device)
            ref_pcts.append(ref_pct)
            base_pcts.append(base)
            gaps.append(ref_pct - base)
        ref_pcts = np.array(ref_pcts); base_pcts = np.array(base_pcts); gaps = np.array(gaps)
        print(f"[cfg{ci}] {params} | n_edges≈{n_edges}")
        print(f"   reference%   = {ref_pcts.mean():.4f} ± {ref_pcts.std():.4f}")
        print(f"   baseline%    = {base_pcts.mean():.4f} ± {base_pcts.std():.4f}")
        print(f"   GAP (ref-base) = {gaps.mean():+.4f} ± {gaps.std():.4f} pp  "
              f"{'<-- PASS (>=0.5pp)' if gaps.mean() >= 0.5 else ''}")
        print()


if __name__ == "__main__":
    main()
