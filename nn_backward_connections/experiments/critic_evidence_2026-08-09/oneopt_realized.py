"""Realizable single-node control: restart H35's OWN refiner (and plain Jacobi sift)
from the H35 order. Any gain here is NOT attributable to the collective move class."""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path("/Users/abed359/IdeaProjects/university/deeplearning_thesis/nn_backward_connections")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"experiments"))
from mfas import io
from mfas.metrics import score_from_positions, pct
from mfas.refine import build_sift_edges, sift, sift_underrelaxed
import size_collective_moves as S

for ds, sw in (("connectome", 20), ("microns", 12)):
    g = io.load_dataset(ds)
    src, tgt, w = build_sift_edges(g)
    rank = S.rank_of(np.load(S.latest_h35_positions(ds, 42)))
    base = score_from_positions(rank, src, tgt, w)
    for name, fn in (("plain Jacobi sift", lambda: sift(g, rank, max_sweeps=sw)),
                     ("H35 under-relaxed sift", lambda: sift_underrelaxed(
                         g, rank, k_full=6, alpha=0.7, max_sweeps=sw))):
        t0 = time.time(); br, bs, log = fn(); el = time.time() - t0
        print(f"{ds:11s} restart {name:24s} x{sw:>2} sweeps: "
              f"{pct(base,g.total_weight):.5f}% -> {pct(bs,g.total_weight):.5f}%  "
              f"= {100*(bs-base)/g.total_weight:+.5f} pp  ({el:.0f}s, "
              f"movers sweep0={log[0]['n_movers'] if log else '-'})", flush=True)
