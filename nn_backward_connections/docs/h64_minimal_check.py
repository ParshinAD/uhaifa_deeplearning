"""Check that docs/h64_minimal.py reproduces the production functions bit-for-bit.

Runs on the mouse graph (148 nodes) in a few seconds:

    $PY docs/h64_minimal_check.py

Asserts, stage by stage:
  1. greedy_fas_order         == mfas.experiments.H02.greedy_fas_order
  2. rocket_asym (200 epochs) == mfas.experiments.H64._rocket_asym   (same positions, CPU)
  3. per-node best gaps/gains == mfas.refine.insertion.jacobi_best_gaps
     sift_underrelaxed        == mfas.refine.underrelax.sift_underrelaxed
  4. scc_refine               == mfas.refine.scc_recursive.scc_recursive_refine (5 split fracs)
     alternate_scc_sift       == mfas.refine.scc_recursive.alternate_scc_sift
  0. score                    == mfas.metrics.score_from_order
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE))

import h64_minimal as mini                                   # noqa: E402
from mfas.baseline.rocket import RocketConfig                # noqa: E402
from mfas.experiments.H02 import greedy_fas_order, _init_positions_from_order  # noqa: E402
from mfas.experiments.H64 import _rocket_asym                # noqa: E402
from mfas.io import load_dataset                             # noqa: E402
from mfas.metrics import score_from_order                    # noqa: E402
from mfas.refine.insertion import build_sift_edges, jacobi_best_gaps  # noqa: E402
from mfas.refine.scc_recursive import alternate_scc_sift, scc_recursive_refine  # noqa: E402
from mfas.refine.underrelax import sift_underrelaxed         # noqa: E402


def main() -> int:
    g = load_dataset("mouse")
    n = g.n_nodes
    src, tgt, w = np.asarray(g.src), np.asarray(g.tgt), np.asarray(g.weight)
    rng = np.random.RandomState(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    print("0. oracle")
    for _ in range(5):
        r = rng.permutation(n).astype(np.int64)
        check("score == score_from_order", mini.score(r, src, tgt, w) == score_from_order(r, src, tgt, w))

    print("1. greedy-FAS warm start")
    r_mini = mini.greedy_fas_order(n, src, tgt, w)
    r_prod = greedy_fas_order(g)
    check("identical rank vector", np.array_equal(r_mini, r_prod))
    check("positions_from_rank == _init_positions_from_order",
          np.array_equal(mini.positions_from_rank(r_prod), _init_positions_from_order(r_prod, "cpu").numpy()))

    print("2. Rocket + ASYM (200 epochs, CPU, seed 42)")
    cfg = dict(mini.CONFIG, epochs=200)
    init = mini.positions_from_rank(r_prod)
    pos_mini, s_mini = mini.rocket_asym(n, src, tgt, w, init, cfg, seed=42, device="cpu", verbose=False)
    res = _rocket_asym(g, RocketConfig(epochs=200), seed=42, device=torch.device("cpu"),
                       init_positions=torch.tensor(init))
    check("identical best positions", np.array_equal(pos_mini, res.best_positions))
    check("identical best score", s_mini == res.best_score)

    print("3. exact-gain sift")
    r0 = np.argsort(np.argsort(pos_mini, kind="stable"), kind="stable").astype(np.int64)
    adj = mini.build_adj(n, src, tgt, w)
    s_e, t_e, w_e = build_sift_edges(g)
    for r in [r0] + [rng.permutation(n).astype(np.int64) for _ in range(3)]:
        gaps = np.empty(n, dtype=np.int64); gain = np.empty(n)
        for u in range(n):
            gaps[u], gain[u] = mini.best_gap_for_node(u, r, adj)
        bg, gn = jacobi_best_gaps(r, s_e, t_e, w_e, n)
        check("best gaps == jacobi_best_gaps", np.array_equal(gaps, bg))
        check("gains == jacobi_best_gaps (1e-6)", np.allclose(gain, gn, atol=1e-6))
    r3_mini, s3_mini = mini.sift_underrelaxed(n, src, tgt, w, r0, max_sweeps=40, k_full=6, alpha=0.7, verbose=False)
    r3_prod, s3_prod, _ = sift_underrelaxed(g, r0, k_full=6, alpha=0.7, max_sweeps=40)
    check("sift: identical rank vector", np.array_equal(r3_mini, r3_prod))
    check("sift: identical score", s3_mini == s3_prod)

    print("4. SCC block refinement and alternation")
    for frac in mini.CONFIG["split_fracs"]:
        a = mini.scc_refine(n, src, tgt, r3_prod, min_block=4, split_frac=frac)
        b = scc_recursive_refine(g, r3_prod, min_block=4, split_frac=frac)
        check(f"scc_refine split={frac}: identical", np.array_equal(a, b))
    cfg4 = dict(mini.CONFIG, alt_cycles=6, min_block=4)
    r4_mini, s4_mini = mini.alternate_scc_sift(n, src, tgt, w, r3_prod, cfg4, verbose=False)
    r4_prod, s4_prod, _ = alternate_scc_sift(g, r3_prod, n_cycles=6, sift_sweeps=2, k_full=2,
                                             alpha=0.7, min_block=4)
    check("alternation: identical rank vector", np.array_equal(r4_mini, r4_prod))
    check("alternation: identical score", s4_mini == s4_prod)
    check("alternation never regresses", s4_mini >= s3_prod)

    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
