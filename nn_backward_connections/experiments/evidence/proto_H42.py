"""H42 prototype (rung 2) — how should stage 4's wall-clock be spent?

H36 ships a hand-set alternation CYCLE COUNT (connectome 12, microns 3) chosen at a
moment when the cycle's own budget, not the algorithm's, was binding. Two questions
were left open and they are the same question:

  Q1  SIZING   — the connectome curve was still rising at 31 cycles (+0.214 pp).
                 Where does it actually flatten, and at what wall-clock?
  Q2  ALLOCATION — a stage-4 cycle is (block refine) + (short under-relaxed sift).
                 The sift is 88% of the cost (connectome 23.0 s vs 3.0 s; microns
                 26.5 s vs 5.0 s). At a FIXED wall-clock, are 8 sift sweeps per
                 cycle the right split, or do more/cheaper cycles buy more?

Q2 is what makes a microns answer possible at all: microns has ~290 s of headroom
under the 3600 s hard cap and ~86 s under the 3400 s warn band, so it cannot buy
more cycles — it can only spend the same seconds differently.

Every arm starts from the STORED champion order and runs the PRODUCTION refiner
(``mfas.refine.alternate_scc_sift``), CPU only, no GPU, no gradient steps. Arms are
run STRICTLY SEQUENTIALLY: they are compared at matched wall-clock, so CPU contention
between them would invalidate the comparison.

Run:
    PY=/c/ProgramData/anaconda3/envs/allen/python.exe
    $PY dr_tmp/proto_H42.py            # all arms, sequential (~75 min CPU)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfas.io import load_dataset                          # noqa: E402
from mfas.metrics import pct, score_from_order            # noqa: E402
from mfas.refine import alternate_scc_sift                # noqa: E402

# Stage-3 output orders (the exact input H36's stage 4 receives), from P01's
# re-baseline runs on this machine. Same files proto_H36c used.
CHAMP = {
    "connectome": "results/20260809T142912Z-H35-connectome-s42-implement-"
                  "8f52fb_positions.npy",
    "microns": "results/20260809T145818Z-H30-microns-s42-implement-"
               "954bab_positions.npy",
}

# (dataset, sift_sweeps, wall budget s, cycle cap). The first arm of each dataset is
# the SHIP config; the others re-allocate the same seconds toward more, cheaper cycles.
ARMS = [
    ("connectome", 8, 1200.0, 400),   # ship split, sized far past the 12 we ship
    ("connectome", 4, 1200.0, 800),
    ("connectome", 2, 1200.0, 1600),
    ("microns", 4, 400.0, 200),       # ship split
    ("microns", 2, 400.0, 400),
]

_ALPHA = 0.7        # unchanged from H36
_K_FULL = 2         # unchanged from H36
_MIN_BLOCK = 32     # unchanged from H36


def run_arm(g, rank0, base_score, sweeps, budget_s, cap):
    """One arm: alternate at a fixed sift-sweep count until the wall budget runs out."""
    t0 = time.time()
    best_rank, best_score, log = alternate_scc_sift(
        g, rank0, n_cycles=cap, sift_sweeps=sweeps, k_full=_K_FULL,
        alpha=_ALPHA, min_block=_MIN_BLOCK, time_budget_s=budget_s)
    wall = time.time() - t0
    total = g.total_weight
    return {
        "sift_sweeps": sweeps,
        "wall_budget_s": budget_s,
        "wall_s": wall,
        "n_cycles_done": len(log),
        "base_pct": pct(base_score, total),
        "final_pct": pct(best_score, total),
        "delta_pp": pct(best_score, total) - pct(base_score, total),
        "log": log,
    }


def main() -> None:
    out = {
        "_doc": "H42 prototype (rung 2) — stage-4 sizing (Q1) and wall-clock "
                "allocation (Q2). CPU only, from stored champion orders, "
                "production refiner mfas.refine.alternate_scc_sift. Arms run "
                "strictly sequentially so the matched-wall-clock comparison holds.",
        "variant": "H42",
        "gate": "prototype",
        "date": "2026-08-10",
        "machine": "Windows 10 / RTX 4060 Laptop GPU (CUDA); CPU-only for this gate",
        "reproduce": ["/c/ProgramData/anaconda3/envs/allen/python.exe "
                      "dr_tmp/proto_H42.py"],
        "fixed": {"alpha": _ALPHA, "k_full": _K_FULL, "min_block": _MIN_BLOCK},
        "sources": CHAMP,
        "arms": [],
    }

    cache = {}
    for ds, sweeps, budget_s, cap in ARMS:
        if ds not in cache:
            g = load_dataset(ds)
            pos = np.load(ROOT / CHAMP[ds])
            rank0 = np.argsort(np.argsort(pos, kind="stable"),
                               kind="stable").astype(np.int64)
            base = score_from_order(rank0, np.asarray(g.src), np.asarray(g.tgt),
                                    g.weight)
            cache[ds] = (g, rank0, base)
            print(f"[{ds}] loaded; stage-3 order = {pct(base, g.total_weight):.6f}%",
                  flush=True)
        g, rank0, base = cache[ds]
        print(f"[{ds}] arm sweeps={sweeps} budget={budget_s:.0f}s ...", flush=True)
        rec = run_arm(g, rank0, base, sweeps, budget_s, cap)
        rec["dataset"] = ds
        out["arms"].append(rec)
        print(f"[{ds}] sweeps={sweeps}: {rec['n_cycles_done']} cycles, "
              f"{rec['wall_s']:.0f}s, delta={rec['delta_pp']:+.6f} pp", flush=True)
        dst = ROOT / "experiments/outputs/proto_H42.json"
        dst.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("WROTE experiments/outputs/proto_H42.json", flush=True)


if __name__ == "__main__":
    main()
