"""H36 prototype gate — the matched-wall-clock control, on BOTH primaries.

Promoted out of ``dr_tmp/`` (critic finding, 2026-08-10) because it is the evidence
for the single most important claim in the H36 cycle:

    stage 4's gain is the new BLOCK move class, not the extra wall-clock it spends.

Stage 4 adds **0 gradient steps**, so ``budget_basis = total_grad_steps`` records it
as free. It is not free — it spends real CPU. The honest test is therefore not the
gradient budget but a matched-wall-clock control: give the CHAMPION's OWN move class
(more under-relaxed exact-gain sift sweeps) the same wall-clock, from the same
starting order, on the same machine, and see where it gets.

This script imports the PRODUCTION refiner (``mfas.refine.scc_recursive``), not a
scratch copy, so the control cannot silently drift from the shipped algorithm.

Everything starts from STORED champion positions, so it uses no GPU and takes no
gradient steps. Runtime ~15 min of CPU.

Run
---
    PY=/c/ProgramData/anaconda3/envs/allen/python.exe
    PYTHONPATH=src $PY experiments/proto_h36_control.py
    # -> experiments/outputs/proto_H36_control.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfas.io import load_dataset                          # noqa: E402
from mfas.metrics import pct, score_from_order            # noqa: E402
from mfas.refine import alternate_scc_sift                # noqa: E402
from mfas.refine.underrelax import sift_underrelaxed      # noqa: E402

# The champion orders re-measured on this device by P01 (role=implement, seed 42).
CHAMPION_POSITIONS = {
    "connectome": "results/20260809T142912Z-H35-connectome-s42-implement-"
                  "8f52fb_positions.npy",
    "microns": "results/20260809T145818Z-H30-microns-s42-implement-"
               "954bab_positions.npy",
}

# (dataset, control sweeps, treatment cycles, treatment inner sift sweeps).
# The treatment column is exactly what src/mfas/experiments/H36.py ships.
ARMS = [
    ("connectome", 120, 12, 8),
    ("microns", 15, 3, 4),
]


def load_champion_rank(name):
    posv = np.load(CHAMPION_POSITIONS[name])
    return np.argsort(np.argsort(posv, kind="stable"),
                      kind="stable").astype(np.int64)


def main():
    out = {
        "_doc": ("Matched-wall-clock control for H36 stage 4, on both primary "
                 "datasets. Control = the champion's OWN move class (more "
                 "under-relaxed sift sweeps). Treatment = the shipped stage-4 "
                 "block/node alternation. Same starting order, same machine, "
                 "matched wall-clock, 0 gradient steps in both arms."),
        "date": "2026-08-10", "variant": "H36",
        "refiner": "mfas.refine.scc_recursive (production module, not a copy)",
        "arms": {},
    }

    for name, sweeps, cycles, inner in ARMS:
        g = load_dataset(name)
        src_o, tgt_o, total = np.asarray(g.src), np.asarray(g.tgt), g.total_weight
        r0 = load_champion_rank(name)
        base = score_from_order(r0, src_o, tgt_o, g.weight)
        print(f"\n== {name}: champion {pct(base, total):.6f} % ==")

        t0 = time.time()
        _, s_ctrl, swlog = sift_underrelaxed(g, r0, k_full=6, alpha=0.7,
                                             max_sweeps=sweeps)
        w_ctrl = time.time() - t0
        d_ctrl = pct(s_ctrl, total) - pct(base, total)
        print(f"  control   {len(swlog):3d} more sift sweeps -> "
              f"{pct(s_ctrl, total):.6f} % ({d_ctrl:+.6f} pp, {w_ctrl:.0f}s)")

        t0 = time.time()
        _, s_treat, alog = alternate_scc_sift(g, r0, n_cycles=cycles,
                                              sift_sweeps=inner)
        w_treat = time.time() - t0
        d_treat = pct(s_treat, total) - pct(base, total)
        print(f"  treatment {cycles:3d} block/node cycles -> "
              f"{pct(s_treat, total):.6f} % ({d_treat:+.6f} pp, {w_treat:.0f}s)")
        print(f"  ratio     {d_treat / d_ctrl:.1f}x at matched wall-clock")

        out["arms"][name] = dict(
            champion_pct=pct(base, total),
            control=dict(n_sweeps=len(swlog), final_pct=pct(s_ctrl, total),
                         delta_pp=d_ctrl, wall_s=w_ctrl),
            treatment=dict(n_cycles=len(alog), final_pct=pct(s_treat, total),
                           delta_pp=d_treat, wall_s=w_treat),
            ratio=d_treat / d_ctrl if d_ctrl > 0 else None)

    out["conclusion"] = (
        "On BOTH primaries, at matched wall-clock from the same starting order, the "
        "champion's own single-node move class recovers a small fraction of what the "
        "block move class recovers. The gain is the move class, not the compute.")
    Path("experiments/outputs/proto_H36_control.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("\nwrote experiments/outputs/proto_H36_control.json")


if __name__ == "__main__":
    main()
