"""P07/S01 continuation — finish the microns epoch grid and run the same grid on connectome.

Why this exists
---------------
``dr_tmp/proto_P07.py`` was killed on 2026-08-11 at 01:53 mid-arm (``epochs=5000``), leaving
2 of 5 microns arms on disk. Its own resume note says: do NOT relaunch arms already present.
This script honours that — it reads ``experiments/outputs/proto_<tag>.json``, skips every arm
already recorded, and merges new arms into the same file.

It also extends the probe to CONNECTOME, which the original did not cover and which is the
mission dataset. The reasons that is worth the GPU time:

* connectome ships **20,000** Rocket epochs and uses ~1232 s of the 3600 s cap, i.e. it has
  ~2368 s of unused budget. If the gradient phase is largely ceremonial there too, that budget
  is free to spend on refinement instead — and if it is NOT ceremonial, that kills S01's
  premise on the dataset that actually carries the mission target, which is equally worth
  knowing before any Phase-2 work is planned.
* The one instrumented connectome run (P06a) splits as stage 3 = 134.4 s, stage 4 = 697.3 s of
  1483.9 s total, i.e. **56% of the run is the CPU refiner**. So the epoch axis and the
  refinement axis are separately sized, and only the epoch axis has never been measured.

Method is unchanged from ``proto_P07.py`` and deliberately so: every constant except ``epochs``
comes from :mod:`mfas.experiments.H42`, greedy-FAS is computed once and shared across arms, and
each arm is a properly SCALED shorter schedule (``make_beta_schedule`` spans ``cfg.cycles``
cosine cycles over ``cfg.epochs``), not a truncated one. That is the S01 question — "how much
Rocket does the pipeline need" — rather than "stop it early".

Scores are deterministic and load-free; TIMINGS are taken under whatever load the machine has,
and the recorded load note says which. Nothing here reads the target metric into an algorithm:
the champion's score is quoted for reporting only.

Run:  PYTHONPATH=src python experiments/proto_P07_sizing.py <dataset>
Out:  experiments/outputs/proto_P07.json     (microns, merged)
      experiments/outputs/proto_S01_connectome.json   (connectome)
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

from mfas import io                                        # noqa: E402
from mfas.baseline.rocket import RocketConfig, run_rocket   # noqa: E402
from mfas.metrics import pct                                # noqa: E402
from mfas.refine import alternate_scc_sift, sift_underrelaxed   # noqa: E402
from mfas.experiments.H02 import (                          # noqa: E402
    _init_positions_from_order, greedy_fas_order)
from mfas.experiments import H42                            # noqa: E402
from mfas.utils.seeding import select_device                # noqa: E402

SEED = 42

# Per dataset: (output file, arm list). The microns arms are the ORIGINAL grid — already-present
# arms are skipped on merge, so this re-states the full grid rather than a hand-picked remainder,
# which keeps the file self-describing if it is ever re-run from scratch.
PLAN = {
    "microns": ("proto_P07.json", [0, 2500, 5000, 10000, 20000]),
    # connectome ships 20,000; the grid brackets it on both sides so saturation is visible
    # rather than assumed. 40,000 is included because "the curve was still rising" is exactly
    # the mistake H36 made at stage 4, and it costs ~500 s to rule out here.
    "connectome": ("proto_S01_connectome.json", [0, 2500, 5000, 10000, 20000, 40000]),
}


def one_arm(g, epochs: int, order, device) -> dict:
    """Run the full H42 pipeline on ``g`` with ``epochs`` Rocket epochs; time each stage.

    Byte-for-byte the same body as ``proto_P07.py::one_arm`` so the merged arms in
    ``proto_P07.json`` remain comparable with the two produced on 2026-08-11.
    """
    total = g.total_weight

    t0 = time.time()
    init_positions = _init_positions_from_order(order, device)
    t_init = time.time() - t0

    cfg = RocketConfig(epochs=epochs)
    t0 = time.time()
    rocket = run_rocket(g, cfg, seed=SEED, device=device,
                        init_positions=init_positions, time_limit=None)
    t_rocket = time.time() - t0

    rank0 = np.argsort(np.argsort(rocket.best_positions, kind="stable"),
                       kind="stable").astype(np.int64)
    t0 = time.time()
    sift_rank, sift_score, sweep_log = sift_underrelaxed(
        g, rank0, k_full=H42._K_FULL, alpha=H42._ALPHA,
        max_sweeps=H42._MAX_SWEEPS.get(g.name, 40), time_budget_s=None)
    t_sift = time.time() - t0

    t0 = time.time()
    alt_rank, alt_score, alt_log = alternate_scc_sift(
        g, sift_rank,
        n_cycles=H42._ALT_CYCLES.get(g.name, 32),
        sift_sweeps=H42._ALT_SIFT_SWEEPS.get(g.name, 2),
        k_full=H42._ALT_K_FULL, alpha=H42._ALPHA, min_block=H42._MIN_BLOCK,
        time_budget_s=None)
    t_alt = time.time() - t0

    best = max(rocket.best_score, sift_score, alt_score)
    return dict(
        epochs=epochs,
        pure_pct=rocket.best_pct,
        sift_pct=pct(sift_score, total),
        alt_pct=pct(alt_score, total),
        final_pct=pct(best, total),
        final_score=float(best),
        n_epochs_done=int(rocket.n_epochs_done) if epochs else 0,
        n_sift_sweeps=len(sweep_log),
        n_alt_cycles=len(alt_log),
        t_init_positions_s=t_init,
        t_rocket_s=t_rocket,
        t_sift_s=t_sift,
        t_alt_s=t_alt,
        s_per_epoch=(t_rocket / epochs) if epochs else None,
        est_run_wall_s=None,
    )


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "microns"
    if dataset not in PLAN:
        print(f"unknown dataset {dataset!r}; expected one of {sorted(PLAN)}")
        return 2
    fname, arms = PLAN[dataset]
    out_path = _ROOT / "experiments" / "outputs" / fname

    device, dev_name = select_device("auto")

    t0 = time.time()
    g = io.load_dataset(dataset)
    t_load = time.time() - t0

    t0 = time.time()
    order = greedy_fas_order(g)
    t_greedy = time.time() - t0

    # ── merge, never clobber ────────────────────────────────────────────────────────
    if out_path.exists():
        out = json.loads(out_path.read_text())
        done = {a.get("epochs") for a in out.get("arms", []) if "error" not in a}
        print(f"resuming {fname}: arms already on disk = {sorted(x for x in done if x is not None)}",
              flush=True)
    else:
        out = dict(
            item="P07/S01",
            what=("Rocket epoch sizing on the shipped H42 pipeline: does the gradient phase "
                  "buy score, or only wall-clock?"),
            dataset=dataset,
            seed=SEED,
            device=dev_name,
            champion_pipeline="H42 (sota.json champion)",
            champion_epochs=H42._EPOCHS[dataset],
            arms=[],
        )
        done = set()

    out["n_nodes"] = int(g.n_nodes)
    out["n_edges"] = int(np.asarray(g.src).size)
    out["t_load_s"] = t_load
    out["t_greedy_fas_s"] = t_greedy
    out.setdefault("load_notes", []).append(
        "2026-08-15 continuation: machine IDLE (operator asleep, no game, GPU ~13% at launch). "
        "The 2026-08-11 arms in this file were taken under desktop load and their TIMINGS are "
        "not comparable with these; their SCORES are deterministic and are.")

    for epochs in arms:
        if epochs in done:
            print(f"=== arm epochs={epochs} — already on disk, skipping ===", flush=True)
            continue
        print(f"=== arm epochs={epochs} ===", flush=True)
        try:
            row = one_arm(g, epochs, order, device)
        except Exception as exc:                      # keep the completed arms
            row = dict(epochs=epochs, error=f"{type(exc).__name__}: {exc}")
            print(f"    FAILED: {row['error']}", flush=True)
        else:
            row["est_run_wall_s"] = (t_greedy + row["t_init_positions_s"]
                                     + row["t_rocket_s"] + row["t_sift_s"]
                                     + row["t_alt_s"])
            print(f"    final={row['final_pct']:.4f}%  "
                  f"pure={row['pure_pct']:.4f}  sift={row['sift_pct']:.4f}  "
                  f"alt={row['alt_pct']:.4f}  est_wall={row['est_run_wall_s']:.1f}s",
                  flush=True)
        out["arms"].append(row)
        out["arms"].sort(key=lambda a: (a.get("epochs") is None, a.get("epochs", 0)))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))   # incremental: survive a kill

    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
