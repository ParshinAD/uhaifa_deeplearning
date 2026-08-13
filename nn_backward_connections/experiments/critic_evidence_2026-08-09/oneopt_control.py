"""CONTROL the probe is missing: is the H35 order really a 1-opt fixed point?

If a pure SINGLE-NODE exact-gain move class still finds gain on the H35 order, then
the S1/S2 gain cannot be attributed to *collectivity*.
"""
import sys, numpy as np
from pathlib import Path
ROOT = Path("/Users/abed359/IdeaProjects/university/deeplearning_thesis/nn_backward_connections")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"experiments"))
from mfas import io
from mfas.metrics import score_from_positions, pct
from mfas.refine import build_sift_edges, jacobi_best_gaps
import size_collective_moves as S

for ds in ("connectome", "microns", "mouse"):
    g = io.load_dataset(ds)
    src, tgt, w = build_sift_edges(g)
    rank = S.rank_of(np.load(S.latest_h35_positions(ds, 42)))
    base = score_from_positions(rank, src, tgt, w)
    best_gap, gain = jacobi_best_gaps(rank, src, tgt, w, g.n_nodes)
    mv = gain > 1e-9
    print(f"{ds:11s} H35@s42 = {pct(base, g.total_weight):.5f}% | "
          f"SINGLE-NODE movers = {int(mv.sum()):,}/{g.n_nodes:,} "
          f"({100*mv.mean():.2f}%) | independent 1-opt gain sum = "
          f"{gain.sum():.6g} weight = {100*gain.sum()/g.total_weight:+.5f} pp | "
          f"max single gain = {gain.max():.6g}")
