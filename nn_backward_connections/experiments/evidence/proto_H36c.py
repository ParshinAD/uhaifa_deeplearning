"""H36 prototype, part 3 — sizing the alternation on BOTH primaries.

Part 2 showed alternating (recursive SCC block refine <-> under-relaxed sift)
gains +0.128 pp on the connectome in 104 s CPU and is still climbing. Two things
must be sized before building the variant:

  S1  connectome: where does the alternation saturate, and at what wall-clock?
  S2  microns: what does ONE alternation cycle cost? The microns champion already
      burns 3240 s of the 3600 s hard budget, so the refinement phase there has
      only ~350 s of headroom (warn band 3400 s). If a cycle costs more than that,
      microns needs a time-budgeted refinement, not a cycle count.

Both start from the STORED champion positions, so no GPU is used.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfas.io import load_dataset                      # noqa: E402
from mfas.metrics import pct, score_from_order        # noqa: E402
from mfas.refine.underrelax import sift_underrelaxed  # noqa: E402

from proto_H36 import SccRecursiveRefiner             # noqa: E402

FRACS = (0.5, 0.382, 0.618, 0.25, 0.75)
CHAMP = {
    "connectome": "results/20260809T142912Z-H35-connectome-s42-implement-"
                  "8f52fb_positions.npy",
    "microns": "results/20260809T145818Z-H30-microns-s42-implement-"
               "954bab_positions.npy",
}


def alternate(g, rank0, max_cycles, time_budget_s, min_block=32,
              k_full=2, alpha=0.7, sift_sweeps=8, tag=""):
    src_o, tgt_o, total = np.asarray(g.src), np.asarray(g.tgt), g.total_weight
    base = score_from_order(rank0, src_o, tgt_o, g.weight)
    rank, best, log = rank0.copy(), base, []
    t0 = time.time()
    print(f"[{tag}] champion start {pct(base, total):.6f} %")
    for c in range(max_cycles):
        if time.time() - t0 > time_budget_s:
            print(f"[{tag}] time budget reached after {c} cycles")
            break
        ta = time.time()
        ref = SccRecursiveRefiner(src_o, tgt_o, g.n_nodes, min_block=min_block,
                                  split_frac=FRACS[c % len(FRACS)])
        rank = ref.run(rank)
        s_scc = score_from_order(rank, src_o, tgt_o, g.weight)
        t_scc = time.time() - ta
        tb = time.time()
        rank, s_sift, _ = sift_underrelaxed(g, rank, k_full=k_full, alpha=alpha,
                                            max_sweeps=sift_sweeps)
        t_sift = time.time() - tb
        best = max(best, s_scc, s_sift)
        log.append(dict(cycle=c, after_scc_pct=pct(s_scc, total),
                        after_sift_pct=pct(s_sift, total),
                        t_scc_s=t_scc, t_sift_s=t_sift,
                        cum_wall_s=time.time() - t0,
                        cum_delta_pp=pct(best, total) - pct(base, total)))
        print(f"[{tag}] cycle {c:2d}  scc {pct(s_scc, total):.6f}  "
              f"sift {pct(s_sift, total):.6f}  "
              f"cum {pct(best, total)-pct(base, total):+.6f} pp   "
              f"(scc {t_scc:.1f}s + sift {t_sift:.1f}s, cum {time.time()-t0:.0f}s)")
    return dict(champion_pct=pct(base, total), final_pct=pct(best, total),
                delta_pp=pct(best, total) - pct(base, total),
                wall_s=time.time() - t0, n_cycles=len(log), log=log)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    out = {}
    if which in ("both", "connectome"):
        g = load_dataset("connectome")
        r0 = np.argsort(np.argsort(np.load(CHAMP["connectome"]), kind="stable"),
                        kind="stable").astype(np.int64)
        out["connectome"] = alternate(g, r0, max_cycles=40, time_budget_s=780,
                                      tag="connectome")
    if which in ("both", "microns"):
        g = load_dataset("microns")
        r0 = np.argsort(np.argsort(np.load(CHAMP["microns"]), kind="stable"),
                        kind="stable").astype(np.int64)
        out["microns"] = alternate(g, r0, max_cycles=6, time_budget_s=900,
                                   tag="microns")
    Path("dr_tmp/proto_H36c_out.json").write_text(
        json.dumps(out, indent=2, default=float))
    for k, v in out.items():
        print(f"{k}: {v['champion_pct']:.6f} -> {v['final_pct']:.6f} "
              f"({v['delta_pp']:+.6f} pp) in {v['wall_s']:.0f}s "
              f"over {v['n_cycles']} cycles")


if __name__ == "__main__":
    main()
