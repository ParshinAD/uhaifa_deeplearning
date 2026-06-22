"""In-repo prototype for the full-range exact-gain sift (H30), using the PRODUCTION kernel.

Standalone CPU script (NOT a runner dataset). For seeds 42/123/999 on the hard
synthetic graph (``mfas.analysis.gap.make_hard_synthetic_graph``, n=400, has a verified
Rocket<->reference gap) and the mouse connectome (``io.load_dataset("mouse")``), reports
greedy / H02-Rocket / GS-reference-sift / Jacobi-sift feedforward % and the Δ-vs-H02.

This reproduces the ``dr_tmp`` falsifier's positive sift gains with the in-repo
production kernel (``mfas.refine``), so the H30 numbers are reproducible without scratch
code. Writes ONLY to ``experiments/outputs/proto_sift_fullrange.json``.

Leakage-safety: every move is chosen from input edge weights + current ranks only; the
frozen oracle is used solely to score / best-by-oracle accept whole candidate vectors.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402
torch.set_num_threads(2)

from mfas import io  # noqa: E402
from mfas.analysis import gap as gapmod  # noqa: E402
from mfas.baseline.rocket import RocketConfig, run_rocket  # noqa: E402
from mfas.experiments.H02 import _init_positions_from_order, greedy_fas_order  # noqa: E402
from mfas.metrics import pct, score_from_order  # noqa: E402
from mfas.refine import sift  # noqa: E402
from mfas.refine.insertion import build_adj, sift_gauss_seidel_ref  # noqa: E402

CPU = torch.device("cpu")


def _rocket_rank(g, seed, init_positions, epochs):
    cfg = RocketConfig(epochs=epochs)
    res = run_rocket(g, cfg, seed=seed, device=CPU, init_positions=init_positions)
    rank = np.argsort(np.argsort(res.best_positions, kind="stable"),
                      kind="stable").astype(np.int64)
    return rank, res.best_pct


def _run_one(g, seed, epochs, gs_sweeps, jacobi_sweeps, ref_pct=None):
    t0 = time.time()
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    adj = build_adj(g)

    greedy = greedy_fas_order(g)
    greedy_pct = pct(score_from_order(greedy, src, tgt, g.weight), g.total_weight)

    h02_init = _init_positions_from_order(greedy, CPU)
    h02_rank, h02_pct = _rocket_rank(g, seed, h02_init, epochs)

    # Gauss-Seidel reference sift (one node at a time) on the H02 order.
    _, gs_pct = sift_gauss_seidel_ref(g, h02_rank, adj, max_sweeps=gs_sweeps, seed=seed)
    # Production Jacobi sift on the H02 order.
    _, jac_score, jac_log = sift(g, h02_rank, max_sweeps=jacobi_sweeps)
    jac_pct = pct(jac_score, g.total_weight)

    row = dict(
        seed=seed, n=int(g.n_nodes), m=int(g.n_edges),
        greedy_pct=greedy_pct, h02_pct=h02_pct,
        gs_sift_pct=gs_pct, jacobi_sift_pct=jac_pct,
        gs_recov_vs_h02=gs_pct - h02_pct,
        jacobi_recov_vs_h02=jac_pct - h02_pct,
        n_jacobi_sweeps=len(jac_log),
        n_jacobi_accepted=int(sum(1 for r in jac_log if r["accepted"])),
        wall_s=round(time.time() - t0, 1),
    )
    if ref_pct is not None:
        row["reference_pct"] = ref_pct
        row["gap_h02_to_ref"] = ref_pct - h02_pct
    return row


def main():
    out = {"synthetic": [], "mouse": []}

    for seed in (42, 123, 999):
        g, _ref_order, ref_pct = gapmod.make_hard_synthetic_graph(seed=seed)
        row = _run_one(g, seed, epochs=4000, gs_sweeps=20, jacobi_sweeps=20,
                       ref_pct=ref_pct)
        out["synthetic"].append(row)
        print(f"[synthetic s{seed}] ref={row['reference_pct']:.3f} "
              f"greedy={row['greedy_pct']:.3f} h02={row['h02_pct']:.3f} "
              f"GS-sift={row['gs_sift_pct']:.3f} Jacobi-sift={row['jacobi_sift_pct']:.3f} "
              f"(Δ vs h02: GS={row['gs_recov_vs_h02']:+.3f} "
              f"Jacobi={row['jacobi_recov_vs_h02']:+.3f}) "
              f"[acc={row['n_jacobi_accepted']}/{row['n_jacobi_sweeps']}, {row['wall_s']}s]")

    g_mouse = io.load_dataset("mouse")
    for seed in (42, 123, 999):
        row = _run_one(g_mouse, seed, epochs=5000, gs_sweeps=30, jacobi_sweeps=30)
        out["mouse"].append(row)
        print(f"[mouse s{seed}] greedy={row['greedy_pct']:.3f} h02={row['h02_pct']:.3f} "
              f"GS-sift={row['gs_sift_pct']:.3f} Jacobi-sift={row['jacobi_sift_pct']:.3f} "
              f"(Δ vs h02: GS={row['gs_recov_vs_h02']:+.3f} "
              f"Jacobi={row['jacobi_recov_vs_h02']:+.3f}) "
              f"[acc={row['n_jacobi_accepted']}/{row['n_jacobi_sweeps']}, {row['wall_s']}s]")

    out_dir = _ROOT / "experiments" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proto_sift_fullrange.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
