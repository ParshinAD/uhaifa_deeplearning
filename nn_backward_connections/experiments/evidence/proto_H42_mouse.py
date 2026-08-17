"""H42 prototype addendum — the mouse leg the main prototype did not cover.

proto_H42.py sized stage 4 on the two PRIMARIES only, because they are the datasets the
screen gate is decided on. But H42 changes ``_ALT_SIFT_SWEEPS`` for EVERY dataset, and
mouse is the standing non-inferiority tripwire, so shipping a mouse config that was never
measured would be exactly the kind of unmeasured constant H42 exists to remove.

Mouse costs milliseconds per stage-4 cycle, so this sweeps the (sift_sweeps, n_cycles)
grid exhaustively instead of sizing against a wall-clock budget. Stages 1-3 are run once
(identical to H36) and the resulting stage-3 order is reused for every arm, so the arms
differ only in stage 4.

Run:
    /c/ProgramData/anaconda3/envs/allen/python.exe dr_tmp/proto_H42_mouse.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfas.baseline.rocket import RocketConfig, run_rocket        # noqa: E402
from mfas.experiments.H02 import (                               # noqa: E402
    _init_positions_from_order, greedy_fas_order)
from mfas.io import load_dataset                                 # noqa: E402
from mfas.metrics import pct                                     # noqa: E402
from mfas.refine import alternate_scc_sift, sift_underrelaxed    # noqa: E402

# Stages 1-3 constants, copied from src/mfas/experiments/H36.py (unchanged by H42).
_EPOCHS, _MAX_SWEEPS, _K_FULL, _ALPHA, _MIN_BLOCK = 5_000, 40, 6, 0.7, 32
_ALT_K_FULL = 2

# H36 ships (sweeps=8, cycles=8) on mouse. Sweep around it.
GRID = [(8, 8), (8, 32), (4, 8), (4, 32), (2, 8), (2, 32), (2, 64), (2, 128), (1, 128)]


def main() -> None:
    g = load_dataset("mouse")
    device = torch.device("cpu")

    t0 = time.time()
    order = greedy_fas_order(g)
    rocket = run_rocket(g, RocketConfig(epochs=_EPOCHS), seed=42, device=device,
                        init_positions=_init_positions_from_order(order, device))
    rank0 = np.argsort(np.argsort(rocket.best_positions, kind="stable"),
                       kind="stable").astype(np.int64)
    sift_rank, sift_score, _ = sift_underrelaxed(
        g, rank0, k_full=_K_FULL, alpha=_ALPHA, max_sweeps=_MAX_SWEEPS)
    total = g.total_weight
    print(f"stages 1-3 done in {time.time()-t0:.1f}s; "
          f"stage-3 order = {pct(sift_score, total):.6f}%", flush=True)

    out = {
        "_doc": "H42 mouse leg — stage-4 (sift_sweeps, n_cycles) grid on the mouse "
                "tripwire dataset. Stages 1-3 run once and shared by every arm, so "
                "arms differ only in stage 4. CPU only.",
        "variant": "H42",
        "gate": "prototype",
        "date": "2026-08-10",
        "dataset": "mouse",
        "seed": 42,
        "champion_pct": 92.9170,
        "ship_config_H36": {"sift_sweeps": 8, "n_cycles": 8},
        "stage3_pct": pct(sift_score, total),
        "reproduce": ["/c/ProgramData/anaconda3/envs/allen/python.exe "
                      "dr_tmp/proto_H42_mouse.py"],
        "arms": [],
    }

    for sweeps, cycles in GRID:
        t1 = time.time()
        _, score, log = alternate_scc_sift(
            g, sift_rank, n_cycles=cycles, sift_sweeps=sweeps, k_full=_ALT_K_FULL,
            alpha=_ALPHA, min_block=_MIN_BLOCK)
        rec = {
            "sift_sweeps": sweeps,
            "n_cycles": cycles,
            "n_cycles_done": len(log),
            "wall_s": time.time() - t1,
            "final_pct": pct(score, total),
            "delta_vs_champion_pp": pct(score, total) - 92.9170,
        }
        out["arms"].append(rec)
        print(f"  sweeps={sweeps:2d} cycles={cycles:4d}: {rec['final_pct']:.6f}% "
              f"({rec['wall_s']:.1f}s, delta vs champ "
              f"{rec['delta_vs_champion_pp']:+.6f} pp)", flush=True)

    dst = ROOT / "experiments/outputs/proto_H42_mouse.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"WROTE {dst.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
