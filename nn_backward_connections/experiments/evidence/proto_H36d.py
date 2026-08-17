"""H36 prototype, part 4 — the EXTRA-COMPUTE control, and the microns ship config.

C1 (the control the critic will demand). The alternation adds 0 gradient steps but
    ~340 s of non-gradient wall-clock. So: does giving the CHAMPION the same extra
    wall-clock IN ITS OWN MOVE CLASS get to the same place? We continue the champion's
    own under-relaxed sift (k_full=6, alpha=0.7 — H35's exact configuration) from the
    champion's OUTPUT order for a matched ~340 s. If that gains ~0 while the
    alternation gains +0.19 pp, the win is the new BLOCK move class, not the compute.

C2 (ship config for microns). The microns champion already burns 3240 s of a 3600 s
    hard budget, so the refinement phase there gets a cheap configuration
    (sift_sweeps=4). Measure what it actually costs and buys.
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
from proto_H36c import CHAMP, alternate               # noqa: E402


def main():
    out = {}

    # ── C1: matched-wall-clock control, champion's own move class ────────────
    print("== C1: champion order + 120 MORE under-relaxed sift sweeps ==")
    g = load_dataset("connectome")
    src_o, tgt_o, total = np.asarray(g.src), np.asarray(g.tgt), g.total_weight
    r0 = np.argsort(np.argsort(np.load(CHAMP["connectome"]), kind="stable"),
                    kind="stable").astype(np.int64)
    base = score_from_order(r0, src_o, tgt_o, g.weight)
    print(f"  champion start        {pct(base, total):.6f} %")
    t0 = time.time()
    _, s_more, swlog = sift_underrelaxed(g, r0, k_full=6, alpha=0.7,
                                         max_sweeps=120)
    wall = time.time() - t0
    print(f"  +120 sift sweeps      {pct(s_more, total):.6f} %  "
          f"({pct(s_more, total) - pct(base, total):+.6f} pp, {wall:.0f}s, "
          f"{len(swlog)} sweeps)")
    out["C1_sift_only_control"] = dict(
        champion_pct=pct(base, total), final_pct=pct(s_more, total),
        delta_pp=pct(s_more, total) - pct(base, total), wall_s=wall,
        n_sweeps=len(swlog),
        note="matched-wall-clock control vs the alternation's +0.192 pp at 340 s")

    # ── C2: microns ship configuration ───────────────────────────────────────
    print("\n== C2: microns, ship config (3 cycles, sift_sweeps=4) ==")
    gm = load_dataset("microns")
    rm = np.argsort(np.argsort(np.load(CHAMP["microns"]), kind="stable"),
                    kind="stable").astype(np.int64)
    out["C2_microns_ship"] = alternate(gm, rm, max_cycles=3, time_budget_s=600,
                                       sift_sweeps=4, tag="microns-ship")

    Path("dr_tmp/proto_H36d_out.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("\n" + json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "log"}
         for k, v in out.items()}, indent=2, default=float))


if __name__ == "__main__":
    main()
