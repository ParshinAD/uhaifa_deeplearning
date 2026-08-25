"""Residual 1-opt diagnostic: how much single-node insertion gain is left unclaimed?

Given an ordering, this reports how far it is from being a fixed point of the
full-range exact-gain insertion move ("sift") that stage 3 of the champion pipeline
owns. It is the instrument that separates two very different worlds:

* a large residual means the pipeline has NOT converged the move class it already
  has, and no new move class should be designed until that debt is paid;
* a small residual means the incumbent is dry on 1-opt, and whatever closes the
  remaining gap is NOT more single-node insertion.

Measured on the two orders we can score exactly (connectome, this machine):

    parity anchor  82.916074%  ->  17,949 movers,  naive_sum 0.5461 pp
    reference      84.614678%  ->     131 movers,  naive_sum 0.0010 pp

So the reference order is essentially 1-opt dry, which rules out "more 1-opt" as
the route to the remaining 0.4606 pp.

READ ``naive_sum_pp`` CORRECTLY -- this is the rule that invalidated an earlier
sizing attempt. Each per-node gain is computed against the SAME frozen order, so
the gains overlap: applying one move changes the gain of every other. The sum is an
UPPER BOUND with unbounded slack and is NOT an estimate of recoverable weight.
Multiplying a mover count by a mean gain is invalid for the same reason. To size
what a move class can actually recover you must SELECT a disjoint subset and
RE-SCORE it through the oracle.

Read-only. Nothing here feeds an algorithm; the discrete score is produced only by
the frozen oracle after an ordering already exists.

Usage
-----
    $PY experiments/diagnostics/residual_1opt.py --dataset connectome --order anchor
    $PY experiments/diagnostics/residual_1opt.py --dataset connectome --order reference
    $PY experiments/diagnostics/residual_1opt.py --order-npy results/champions/H42_connectome_s42.npy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mfas.io import load_dataset                                  # noqa: E402
from mfas.metrics import pct, score_from_order                    # noqa: E402
from mfas.refine.insertion import build_sift_edges, jacobi_best_gaps  # noqa: E402

TOL = 1e-9


@dataclass(frozen=True)
class Residual:
    """One measurement. ``naive_sum_pp`` is an upper bound -- see the module docstring."""
    dataset: str
    label: str
    n_nodes: int
    pct: float
    movers: int
    mover_frac: float
    naive_sum_gain: float
    naive_sum_pp: float
    best_single_gain: float
    wall_s: float


def residual_1opt(g, rank: np.ndarray, *, dataset: str, label: str) -> Residual:
    """Measure unclaimed single-node insertion gain at ``rank``.

    ``rank[node] = position``. Returns counts and an UPPER BOUND on the gain; it
    deliberately does not attempt to estimate recoverable weight.
    """
    rank = np.asarray(rank, dtype=np.int64)
    n = int(g.n_nodes)
    if rank.shape != (n,) or not np.array_equal(np.sort(rank), np.arange(n)):
        raise ValueError(f"{label}: rank must be a permutation of [0, {n})")

    t0 = time.perf_counter()
    src, tgt, w = build_sift_edges(g)
    _best_gap, gain = jacobi_best_gaps(rank, src, tgt, w, n)
    movers = np.flatnonzero(gain > TOL)
    total = float(gain[movers].sum()) if movers.size else 0.0
    s = score_from_order(rank, g.src, g.tgt, g.weight)
    return Residual(
        dataset=dataset,
        label=label,
        n_nodes=n,
        pct=float(pct(s, g.total_weight)),
        movers=int(movers.size),
        mover_frac=float(movers.size) / n,
        naive_sum_gain=total,
        naive_sum_pp=100.0 * total / float(g.total_weight),
        best_single_gain=float(gain[movers].max()) if movers.size else 0.0,
        wall_s=time.perf_counter() - t0,
    )


def order_from_positions(positions: np.ndarray) -> np.ndarray:
    """Continuous positions -> integer rank, via the project's canonical double argsort."""
    return np.argsort(np.argsort(positions)).astype(np.int64)


def load_order(g, which: str, npy: Optional[Path]) -> tuple[np.ndarray, str]:
    if npy is not None:
        arr = np.load(npy)
        rank = arr.astype(np.int64) if arr.dtype.kind in "iu" else order_from_positions(arr)
        return rank, npy.name
    if which == "anchor":
        pos = np.load(REPO_ROOT / "results" / "rocket_best_positions.npy")
        return order_from_positions(pos), "parity anchor"
    if which == "reference":
        # The leakage firewall: every read of data/best_solution goes through this module.
        from mfas.analysis.gap import load_best_solution
        ref, _reported = load_best_solution(g)
        return np.asarray(ref, dtype=np.int64), "reference"
    raise ValueError(f"unknown order {which!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="connectome")
    ap.add_argument("--order", default="anchor", choices=("anchor", "reference"))
    ap.add_argument("--order-npy", type=Path, default=None,
                    help="score an arbitrary saved order/positions instead")
    ap.add_argument("--json", action="store_true", help="emit one JSON object")
    args = ap.parse_args(argv)

    g = load_dataset(args.dataset)
    rank, label = load_order(g, args.order, args.order_npy)
    r = residual_1opt(g, rank, dataset=args.dataset, label=label)

    if args.json:
        print(json.dumps(asdict(r), indent=1))
    else:
        print(f"{r.label} [{r.dataset}]: pct={r.pct:.6f}  movers={r.movers:,} "
              f"({100 * r.mover_frac:.2f}% of nodes)  naive_sum={r.naive_sum_pp:.4f} pp "
              f"(UPPER BOUND, overlapping)  best_single={r.best_single_gain:.0f}  "
              f"({r.wall_s:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
