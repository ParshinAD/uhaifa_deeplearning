"""Is H35's returned order a single-node (1-opt) fixed point? — the attribution control.

Written by the critic while red-teaming `experiments/size_collective_moves.py` and promoted
here because the 2026-08-09 log entry cites its numbers (see `.gitignore`: keepers must be
promoted out of `dr_tmp/`, not left there, so they stay reproducible-by-checkout).

Why it matters
--------------
The H36 sizing gate credits its gain to a **collective** (multi-node) move class. That
attribution is only valid if the *single-node* class is already exhausted on the order it
starts from. The original draft asserted "H35 converges to a 1-opt fixed point" — this
script is what falsified that: H35's sweep cap can bite before convergence, so some of the
raw sizing gain is really just more of H35's own sift.

Two measurements, both on the logged H35 orders (seed 42 by default):

  (1) **Potential** — how many nodes still want to move (`jacobi_best_gaps` gain > 0) and
      the sum of their *independent* 1-opt gains. This is a diagnostic upper bound, not
      realizable (the moves interact).
  (2) **Realized** — restart H35's own refiner (and plain Jacobi `sift`) from the H35 order
      and record what the single-node class actually recovers. THIS is the number that must
      be subtracted before any "collective" claim.

Leakage: none — reads the graph and our own logged H35 positions only, never
`data/best_solution`; scores exclusively with the frozen `mfas.metrics`. Writes nothing to
`results/`.

Input dependency: `results/*_positions.npy` is gitignored, so on a clean checkout the H35
orders must be regenerated first (see the header of `experiments/size_collective_moves.py`).

Usage:
    /opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python \
        experiments/diagnostics/control_1opt_movers.py
    -> experiments/outputs/control_1opt_movers.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments"))

from mfas import io                                                   # noqa: E402
from mfas.metrics import pct, score_from_positions                    # noqa: E402
from mfas.refine import build_sift_edges, jacobi_best_gaps, sift, sift_underrelaxed  # noqa: E402
import size_collective_moves as S                                     # noqa: E402

# Restart sweep budgets = each dataset's production H35 cap, so the control asks
# "what would one more full run of the existing refiner give?", not "what would an
# unbounded refiner give" (the latter is the --sift-first control inside the probe).
CONFIG = {
    "seed": 42,
    "restart_sweeps": {"connectome": 20, "microns": 12, "mouse": 40},
    "tol": 1e-9,
}


def main() -> None:
    out = {"probe": "1-opt attribution control for the H36 sizing gate",
           "config": CONFIG, "datasets": {}}

    for ds in ("connectome", "microns", "mouse"):
        g = io.load_dataset(ds)
        src, tgt, w = build_sift_edges(g)
        pos_file = S.latest_h35_positions(ds, CONFIG["seed"])
        rank = S.rank_of(np.load(pos_file))
        base = score_from_positions(rank, src, tgt, w)

        # (1) potential
        _, gain = jacobi_best_gaps(rank, src, tgt, w, g.n_nodes)
        movers = int((gain > CONFIG["tol"]).sum())
        rec = {
            "h35_positions_file": pos_file.name,
            "h35_pct": pct(base, g.total_weight),
            "single_node_movers": movers,
            "single_node_movers_frac": movers / g.n_nodes,
            "independent_1opt_gain_pp": 100.0 * float(gain.sum()) / g.total_weight,
            "max_single_node_gain_weight": float(gain.max()),
            "restarts": {},
        }
        print(f"{ds:11s} H35@s{CONFIG['seed']} = {rec['h35_pct']:.5f}% | "
              f"single-node movers = {movers:,}/{g.n_nodes:,} "
              f"({100.0 * movers / g.n_nodes:.2f}%)", flush=True)

        # (2) realized
        sw = CONFIG["restart_sweeps"][ds]
        for name, fn in (
            ("plain_jacobi_sift", lambda: sift(g, rank, max_sweeps=sw)),
            ("h35_underrelaxed_sift",
             lambda: sift_underrelaxed(g, rank, k_full=6, alpha=0.7, max_sweeps=sw)),
        ):
            t0 = time.time()
            _, best_score, log = fn()
            rec["restarts"][name] = {
                "sweeps": sw,
                "pct_after": pct(best_score, g.total_weight),
                "recovery_pp": 100.0 * (float(best_score) - base) / g.total_weight,
                "movers_at_sweep0": int(log[0]["n_movers"]) if log else None,
                "seconds": round(time.time() - t0, 1),
            }
            r = rec["restarts"][name]
            print(f"  restart {name:24s} x{sw:>2} sweeps: {rec['h35_pct']:.5f}% -> "
                  f"{r['pct_after']:.5f}% = {r['recovery_pp']:+.5f} pp "
                  f"({r['seconds']}s)", flush=True)

        out["datasets"][ds] = rec

    dest = _ROOT / "experiments" / "outputs" / "control_1opt_movers.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {dest.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
