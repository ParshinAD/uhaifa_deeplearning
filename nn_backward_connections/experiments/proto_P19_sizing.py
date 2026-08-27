"""P19 prototype - ISO-BUDGET compute allocation on microns: how should the run's seconds
be split between the gradient phase (stage 2) and the alternating refiner (stage 4)?

The problem this exists to solve
--------------------------------
P19 is the campaign's blocking item. The shipped microns configuration (H42: 80,000 Rocket
epochs, 5 alternation cycles) does not fit the 3600 s runtime invariant with any margin. Its
own clean runs are 3397.8-3418.0 s against a 3450 s guard deadline - 32-52 s of slack, i.e.
~1% - so ordinary machine noise truncates it. Measured tally on this box: 6 of 8 H64 microns
runs truncated, AND the unmodified champion H42 truncated on its own control run
(results/20260827T092101Z-H42-microns-s42-verify-f91b66.json, 3495.3 s, degraded).

The OPERATOR decided on 2026-08-27 (queue.json P19, commit a95e2f6): do NOT raise
``max_wall_clock_s_per_run``. Make microns FIT the existing invariant with real margin, and
treat speed as a goal in its own right - *"let it squeeze into what it has, that is already too
much in my opinion and ideally it should be faster"*. The epoch count must be **sized on the
curve, never picked**.

Which knob, and why only this one
----------------------------------
Attribution is already measured, on H64's own microns confirm run: of 3419.6 s total, stage 2
(Rocket) is 3252.8 s = **95.1%**; stage 3 (sift) is 82.2 s and stage 4 is 84.5 s. So the
gradient phase is the only lever with enough seconds in it to matter.

What is NOT yet known - and what makes this a study rather than an arithmetic problem - is how
the two stages TRADE. Cutting epochs frees thousands of seconds; some of the score lost can be
bought back by spending those seconds on stage 4 instead. Both halves of that trade are on disk
for exactly ONE epoch count:

* ``experiments/outputs/proto_P07.json`` - the epoch grid at the champion's 5 alternation
  cycles: E=0 -> 83.11152, 2500 -> 83.12094, 5000 -> 83.13558, 10000 -> 83.15791,
  20000 -> 83.22167, against the champion's 80,000 -> 83.24085.
* ``experiments/outputs/proto_H43_microns.json`` - the stage-4 recovery curve AT E=20,000:
  -0.019188 pp at 5 cycles, -0.011116 at 10, -0.006149 at 20, -0.003636 at 50, -0.002708 at
  100, -0.002428 at 120, where it ran out of the 3400 s budget. It is ASYMPTOTING short of the
  champion, so at E=20,000 stage 4 cannot recover the cut at any affordable number of cycles.

Two notes on that H43 artifact, because P19 flagged it as self-contradictory and it is not.
``recovered_cut: false`` means ``final delta > 0`` is false - correct, the delta is -0.002428.
The separately quoted "87.3% recovery" is the fraction OF THE CUT recovered,
(0.019188 - 0.002428) / 0.019188 = 87.3% - also correct. They answer different questions and
both stand. What the artifact really shows is the load-bearing fact: **E=20,000 is too deep a
cut**, and its endpoint -0.002428 pp exceeds the 0.002 pp microns bar.

The design
----------
So the open region is 20,000 < E < 80,000, and the question is two-dimensional: (E, C) -> score
at a given wall clock. This script measures that surface directly. For each arm E it runs the
FULL champion pipeline from scratch - greedy-FAS, Rocket at E epochs, stage 3, then stage 4 one
cycle at a time - logging score and cumulative wall after every cycle, until the run's wall
clock reaches ``WALL_CAP_S``.

Because every arm is driven to the SAME wall cap, the arms are directly comparable at equal
compute, and the per-cycle log lets any smaller budget be read off after the fact: "at a
1,800 s budget, which E wins?" is answered by the artifact rather than by another run. That is
what "sized on the curve" requires.

``WALL_CAP_S`` is 2600 s, deliberately below the 3450 s guard deadline: the deliverable is a
configuration with REAL MARGIN, so measuring up to the deadline would only characterise
configurations we have already decided not to ship.

Why this prototype is not subject to M12
-----------------------------------------
Meta-rule M12 says a composed prototype started from the champion's own converged order
overstates, because the control arm begins at its own fixed point. That bias cannot arise here:
every arm runs the whole pipeline from greedy-FAS, so this prototype IS the pipeline, measured
at different compute allocations. The comparator is the champion's recorded confirm mean, which
is a from-scratch number produced by the same code path.

What the M8 marginal-rate clause asks of this trade
----------------------------------------------------
M8 requires a stage to clear the marginal rate of the stage it competes with - microns stage 4
was measured at 4.33e-4 pp/s against the Rocket tail's 0.08e-4 pp/s, a 54:1 ratio. If that
ratio held uniformly the trade would be arithmetic. It does not: the H43 curve shows stage 4's
rate collapsing as it saturates. This script measures both rates on the same axis so the
comparison is made at the operating point rather than at the average.

Leakage-safety: the frozen oracle scores whole candidate rank vectors after each cycle, for the
log only. It never enters a move choice. ``data/best_solution`` is never read. The stage
constants are H42's (the microns champion); only the epoch count and the cycle count vary.

Run:  PYTHONPATH=src python experiments/proto_P19_sizing.py [wall_cap_s]
Out:  experiments/outputs/proto_P19_microns.json
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

# The open region between the two epoch counts already measured end-to-end (20,000 in
# proto_H43_microns.json, 80,000 = the champion). Three arms, because one point is a guess and
# two cannot show curvature.
ARMS = [30_000, 40_000, 50_000]

# Below the 3450 s guard deadline BY DESIGN - see the module docstring. The deliverable is a
# configuration with margin, so the study characterises the region we would actually ship.
WALL_CAP_S = 2600.0

# Hard ceiling on cycles per arm, so a pathologically cheap cycle cannot spin forever.
MAX_CYCLES = 400

# H42's confirmed microns mean, results/*-H42-microns-*-confirm-*.json (sota.json champion row).
CHAMPION_PCT = 83.24085291200831
CHAMPION_WALL_S = 3418.0      # the upper end of its clean confirmed spread (3397.8-3418.0 s)
MICRONS_BAR_PP = 0.002        # campaign.yaml datasets.microns.min_promotion_delta_pp


def run_arm(g, src, tgt, total, device, epochs: int, order, t_greedy: float) -> dict:
    """Run the champion pipeline at ``epochs`` Rocket epochs, then drive stage 4 to the cap.

    Returns the arm record: per-stage timings, the score after each stage, and the full
    per-cycle stage-4 curve with cumulative run wall clock.
    """
    print(f"\n=== ARM epochs={epochs} ===", flush=True)

    # -- stage 2: Rocket at the reduced epoch count -----------------------------------
    t0 = time.time()
    rocket = run_rocket(g, RocketConfig(epochs=epochs), seed=SEED, device=device,
                        init_positions=_init_positions_from_order(order, device),
                        time_limit=None)
    t_rocket = time.time() - t0
    print(f"  stage2 rocket({epochs}): {rocket.best_pct:.6f}%  {t_rocket:.1f}s", flush=True)

    # -- stage 3: unchanged, H42's constants -------------------------------------------
    rank = np.argsort(np.argsort(rocket.best_positions, kind="stable"),
                      kind="stable").astype(np.int64)
    t0 = time.time()
    rank, sift_score, sweep_log = sift_underrelaxed(
        g, rank, k_full=H42._K_FULL, alpha=H42._ALPHA,
        max_sweeps=H42._MAX_SWEEPS[DATASET], time_budget_s=None)
    t_sift = time.time() - t0
    print(f"  stage3 sift: {pct(sift_score, total):.6f}%  {t_sift:.1f}s  "
          f"sweeps={len(sweep_log)}", flush=True)

    # -- stage 4, one cycle at a time, to the wall cap ---------------------------------
    # Body identical to mfas.refine.scc_recursive.alternate_scc_sift, unrolled so the score
    # after every cycle is recorded. Constants are H42's.
    base_wall = t_greedy + t_rocket + t_sift
    best_score = sift_score
    rows = []
    t_alt0 = time.time()
    for c in range(MAX_CYCLES):
        tA = time.time()
        rank = scc_recursive_refine(
            g, rank, min_block=H42._MIN_BLOCK,
            split_frac=DEFAULT_SPLIT_FRACS[c % len(DEFAULT_SPLIT_FRACS)])
        s_scc = score_from_order(rank, src, tgt, g.weight)
        rank, s_sift, _ = sift_underrelaxed(
            g, rank, k_full=H42._ALT_K_FULL, alpha=H42._ALPHA,
            max_sweeps=H42._ALT_SIFT_SWEEPS[DATASET], time_budget_s=None)
        if s_sift > best_score:
            best_score = s_sift
        elif s_scc > best_score:
            best_score = s_scc
        cum = time.time() - t_alt0
        run_wall = base_wall + cum
        rows.append(dict(cycle=c,
                         after_scc_pct=pct(s_scc, total),
                         after_sift_pct=pct(s_sift, total),
                         best_pct=pct(best_score, total),
                         cycle_wall_s=time.time() - tA,
                         cum_alt_wall_s=cum,
                         run_wall_s=run_wall,
                         delta_vs_champion_pp=pct(best_score, total) - CHAMPION_PCT))
        if c < 5 or c % 10 == 0:
            print(f"    cycle {c:>3}: best={pct(best_score, total):.6f}%  "
                  f"vs champ {pct(best_score, total) - CHAMPION_PCT:+.6f} pp  "
                  f"run_wall={run_wall:.0f}s", flush=True)
        if run_wall > WALL_CAP_S:
            print(f"    stopping at cycle {c}: run_wall {run_wall:.0f}s past the "
                  f"{WALL_CAP_S:.0f}s study cap", flush=True)
            break

    # Cheapest point on this arm that is NON-INFERIOR to the champion (delta >= -bar), and the
    # best point reachable at all. Both are read off the curve, not chosen.
    ni = [r for r in rows if r["delta_vs_champion_pp"] >= -MICRONS_BAR_PP]
    best_row = max(rows, key=lambda r: r["best_pct"]) if rows else None

    return dict(
        epochs=epochs,
        t_rocket_s=t_rocket, t_sift_s=t_sift, t_greedy_fas_s=t_greedy,
        s_per_epoch=t_rocket / epochs if epochs else None,
        rocket_pct=rocket.best_pct,
        sift_pct=pct(sift_score, total),
        n_sift_sweeps=len(sweep_log),
        n_cycles_run=len(rows),
        cycles=rows,
        # the champion's own stage-4 budget, for a like-for-like read of the epoch axis alone
        at_champion_cycles=(rows[H42._ALT_CYCLES[DATASET] - 1]
                            if len(rows) >= H42._ALT_CYCLES[DATASET] else None),
        first_noninferior_cycle=(ni[0] if ni else None),
        best_row=best_row,
    )


def main() -> int:
    global WALL_CAP_S
    if len(sys.argv) > 1:
        WALL_CAP_S = float(sys.argv[1])

    device, dev_name = select_device("auto")
    g = io.load_dataset(DATASET)
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight

    print(f"P19 sizing: {DATASET} n={g.n_nodes} edges={len(src)} device={dev_name} "
          f"cap={WALL_CAP_S:.0f}s arms={ARMS}", flush=True)

    # greedy-FAS is stage 1 and is identical for every arm - run it ONCE and charge its wall
    # clock to each arm, rather than paying for it three times.
    t0 = time.time()
    order = greedy_fas_order(g)
    t_greedy = time.time() - t0
    print(f"stage1 greedy-FAS: {t_greedy:.1f}s (shared across arms, charged to each)",
          flush=True)

    out = dict(
        item="P19",
        what=("iso-budget compute allocation on microns: gradient epochs vs stage-4 "
              "alternation cycles, full pipeline from scratch per arm"),
        dataset=DATASET, seed=SEED, device=dev_name,
        n_nodes=int(g.n_nodes), n_edges=int(len(src)),
        wall_cap_s=WALL_CAP_S, max_cycles=MAX_CYCLES,
        champion_pipeline="H42 (sota.json microns champion)",
        champion_pct=CHAMPION_PCT, champion_epochs=H42._EPOCHS[DATASET],
        champion_alt_cycles=H42._ALT_CYCLES[DATASET],
        champion_wall_s_approx=CHAMPION_WALL_S,
        microns_bar_pp=MICRONS_BAR_PP,
        guard_deadline_s=3450.0,
        t_greedy_fas_s=t_greedy,
        prior_points_on_disk=dict(
            proto_P07="epoch grid at 5 alternation cycles, E=0/2500/5000/10000/20000",
            proto_H43_microns="stage-4 recovery curve at E=20000, to 120 cycles / 3401.9 s",
        ),
        arms=[],
    )
    op = _ROOT / "experiments" / "outputs" / "proto_P19_microns.json"
    op.parent.mkdir(parents=True, exist_ok=True)

    for epochs in ARMS:
        arm = run_arm(g, src, tgt, total, device, epochs, order, t_greedy)
        out["arms"].append(arm)
        # write after every arm, so a killed run still leaves the arms it finished
        op.write_text(json.dumps(out, indent=2))
        print(f"  wrote {op} ({len(out['arms'])}/{len(ARMS)} arms)", flush=True)

    print("\n=== SUMMARY (best point per arm, and cheapest non-inferior point) ===",
          flush=True)
    for a in out["arms"]:
        b, ni = a["best_row"], a["first_noninferior_cycle"]
        print(f"  E={a['epochs']:>6}  rocket={a['t_rocket_s']:.0f}s  "
              f"best={b['best_pct']:.6f} ({b['delta_vs_champion_pp']:+.6f} pp) "
              f"@ cycle {b['cycle']} / {b['run_wall_s']:.0f}s", flush=True)
        if ni:
            print(f"           first NON-INFERIOR (>= -{MICRONS_BAR_PP} pp): cycle "
                  f"{ni['cycle']} at {ni['run_wall_s']:.0f}s "
                  f"({ni['delta_vs_champion_pp']:+.6f} pp), "
                  f"{CHAMPION_WALL_S / ni['run_wall_s']:.2f}x faster than the champion",
                  flush=True)
        else:
            print(f"           NO non-inferior point within {WALL_CAP_S:.0f}s", flush=True)
    print(f"wrote {op}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
