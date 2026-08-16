"""H43 prototype — the microns stage-4 recovery curve, measured rather than extrapolated.

The trade H43 proposes
----------------------
The microns epoch grid (``experiments/outputs/proto_P07.json``, 2026-08-16) showed the last
60,000 of the champion's 80,000 Rocket epochs are **71.7% of the run's wall clock** and buy
**+0.01919 pp**. Cutting to 20,000 epochs frees **2,337.7 s**. At the measured microns stage-4
rate of 17.2 s/cycle (86.00 s / 5 cycles) that is ~136 extra alternation cycles, taking stage 4
from 5 to ~140 — against the ~143 where connectome's stage-4 curve saturates.

Why this must be MEASURED and not extrapolated
-----------------------------------------------
The per-stage marginal-value table says stage 4 returns 4.33e-4 pp/s on microns against the
Rocket tail's 0.08e-4 pp/s, a 54:1 ratio. That would make the trade look like arithmetic. It is
not, and three separate results from the same 24 hours say so: the refinement stack is
**non-monotone in the quality of its input** (connectome 20k→40k epochs improves stage 3 by
+0.03314 pp while the composed score FALLS 0.00666 pp; H44 on mouse; H49's inverted init). A
20,000-epoch start is a *different* order, not a *worse* one, and how stage 4 behaves from it is
an empirical question.

So this script does not assume the recovery. It runs the champion pipeline at ``epochs=20_000``
on microns and then drives stage 4 for ``N_CYCLES`` cycles, logging the score after **every**
cycle. The output is the whole curve, which answers three things at once:

* does the 20,000-epoch start recover the 0.01919 pp the cut costs, and after how many cycles?
* where does the microns stage-4 curve saturate (the champion ships 5 cycles and has never been
  measured past that)?
* what does the curve cost in wall clock, so the result can be stated against the 3,450 s guard
  deadline rather than in the abstract?

Kill condition, stated before the run (queue.json H43): if the curve flattens before recovering
+0.01919 pp, the trade is dead as a SCORE hypothesis and H43 degrades to a pure runtime fix for
P07 — still worth taking, but then it must be reported as a −0.019 pp regression bought for
2,338 s, never as a win.

Leakage-safety: the frozen oracle scores whole candidate rank vectors after each cycle, for the
log. It never enters a move choice. ``data/best_solution`` is never read.

Run:  PYTHONPATH=src python experiments/proto_H43_alt_curve.py [n_cycles]
Out:  experiments/outputs/proto_H43_microns.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io                                            # noqa: E402
from mfas.baseline.rocket import RocketConfig, run_rocket      # noqa: E402
from mfas.metrics import pct, score_from_order                 # noqa: E402
from mfas.refine.scc_recursive import (                        # noqa: E402
    DEFAULT_SPLIT_FRACS, scc_recursive_refine)
from mfas.refine.underrelax import sift_underrelaxed           # noqa: E402
from mfas.experiments.H02 import (                             # noqa: E402
    _init_positions_from_order, greedy_fas_order)
from mfas.experiments import H42                               # noqa: E402
from mfas.utils.seeding import select_device                   # noqa: E402

SEED = 42
DATASET = "microns"
EPOCHS = 20_000                    # the knee measured in proto_P07.json
CHAMPION_PCT = 83.24085291200831   # H42/H44 microns, results/20260815T195540Z-...json
CUT_COST_PP = 0.01919              # what 80,000 -> 20,000 costs, from the epoch grid


def main() -> int:
    n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    device, dev_name = select_device("auto")
    g = io.load_dataset(DATASET)
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight

    out = dict(item="H43", dataset=DATASET, seed=SEED, device=dev_name,
               epochs=EPOCHS, n_cycles_requested=n_cycles,
               champion_pct=CHAMPION_PCT, cut_cost_pp=CUT_COST_PP,
               champion_alt_cycles=H42._ALT_CYCLES[DATASET])

    t0 = time.time()
    order = greedy_fas_order(g)
    t_greedy = time.time() - t0

    # ── stages 1-2: Rocket at the REDUCED epoch count ───────────────────────────
    t0 = time.time()
    rocket = run_rocket(g, RocketConfig(epochs=EPOCHS), seed=SEED, device=device,
                        init_positions=_init_positions_from_order(order, device),
                        time_limit=None)
    t_rocket = time.time() - t0
    print(f"rocket({EPOCHS}): {rocket.best_pct:.5f}%  {t_rocket:.1f}s", flush=True)

    # ── stage 3: unchanged ──────────────────────────────────────────────────────
    rank = np.argsort(np.argsort(rocket.best_positions, kind="stable"),
                      kind="stable").astype(np.int64)
    t0 = time.time()
    rank, sift_score, sweep_log = sift_underrelaxed(
        g, rank, k_full=H42._K_FULL, alpha=H42._ALPHA,
        max_sweeps=H42._MAX_SWEEPS[DATASET], time_budget_s=None)
    t_sift = time.time() - t0
    print(f"stage3: {pct(sift_score, total):.5f}%  {t_sift:.1f}s  sweeps={len(sweep_log)}",
          flush=True)

    # ── stage 4, one cycle at a time, logged ────────────────────────────────────
    # Same three-stage body as mfas.refine.scc_recursive.alternate_scc_sift, unrolled so the
    # score after every cycle is recorded. Constants are H42's.
    best_score = sift_score
    best_rank = rank.copy()
    rows = []
    t_alt0 = time.time()
    for c in range(n_cycles):
        tA = time.time()
        rank = scc_recursive_refine(g, rank, min_block=H42._MIN_BLOCK,
                                    split_frac=DEFAULT_SPLIT_FRACS[c % len(DEFAULT_SPLIT_FRACS)])
        s_scc = score_from_order(rank, src, tgt, g.weight)
        rank, s_sift, _ = sift_underrelaxed(
            g, rank, k_full=H42._ALT_K_FULL, alpha=H42._ALPHA,
            max_sweeps=H42._ALT_SIFT_SWEEPS[DATASET], time_budget_s=None)
        if s_sift > best_score:
            best_score, best_rank = s_sift, rank.copy()
        elif s_scc > best_score:
            best_score = s_scc
        cum = time.time() - t_alt0
        rows.append(dict(cycle=c, after_scc_pct=pct(s_scc, total),
                         after_sift_pct=pct(s_sift, total),
                         best_pct=pct(best_score, total),
                         cycle_wall_s=time.time() - tA, cum_alt_wall_s=cum,
                         run_wall_s=t_rocket + t_sift + cum,
                         delta_vs_champion_pp=pct(best_score, total) - CHAMPION_PCT))
        if c < 10 or c % 10 == 0 or c == n_cycles - 1:
            print(f"  cycle {c:>3}: best={pct(best_score, total):.5f}%  "
                  f"vs champ {pct(best_score, total) - CHAMPION_PCT:+.5f} pp  "
                  f"run_wall={t_rocket + t_sift + cum:.0f}s", flush=True)
        # Stop if we are about to breach the runtime guard's deadline; the point of H43 is a
        # configuration that FITS, so a curve past 3,450 s is not actionable.
        if t_rocket + t_sift + cum > 3400.0:
            print(f"  stopping at cycle {c}: run wall would breach the 3450 s guard", flush=True)
            break

    out.update(
        t_greedy_fas_s=t_greedy, t_rocket_s=t_rocket, t_sift_s=t_sift,
        rocket_pct=rocket.best_pct, sift_pct=pct(sift_score, total),
        n_sift_sweeps=len(sweep_log),
        cycles=rows,
        final_best_pct=rows[-1]["best_pct"] if rows else pct(sift_score, total),
        final_delta_vs_champion_pp=(rows[-1]["delta_vs_champion_pp"] if rows
                                    else pct(sift_score, total) - CHAMPION_PCT),
        recovered_cut=(bool(rows and rows[-1]["delta_vs_champion_pp"] > 0.0)),
    )
    op = _ROOT / "experiments" / "outputs" / "proto_H43_microns.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"FINAL best={out['final_best_pct']:.5f}%  "
          f"vs champion {out['final_delta_vs_champion_pp']:+.5f} pp", flush=True)
    print(f"wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
