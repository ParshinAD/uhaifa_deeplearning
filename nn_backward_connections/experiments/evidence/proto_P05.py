"""P05 prototype — size the run-level wall-clock guard.

The guard question is not "does stage 4 stop when told" (it does — ``alternate_scc_sift``
already honours ``time_budget_s``). It is: **what deadline can the runner hand a variant
such that the guard never binds on an idle machine, yet the TOTAL run stays under the
3600 s hard cap?**

The two clocks are not the same clock, and that is the whole sizing problem:

* ``results/*.json`` ``wall_clock_s`` is measured by ``eval/run_variant.py`` around the
  WHOLE ``variant.run(...)`` call. That is the number sota.json quotes (microns 3418 s)
  and the number the 3600 s cap governs.
* ``time_limit`` as consumed by H42 is measured from the start of ``run_rocket`` — i.e.
  AFTER ``greedy_fas_order`` (stage 0) has already run — and the stage-3/4 budgets are
  derived from it by subtraction.

So the un-guardable prologue (dataset already loaded; greedy-FAS + init tensor) and the
epilogue (the runner's frozen re-score + ``np.save``) both sit OUTSIDE the deadline the
variant enforces. This script measures exactly those, per dataset, so ``reserve_s`` is a
measurement rather than a guess.

Cheap by construction: no Rocket, no refinement, no GPU training. Just stage 0 plus a
scoring call. CPU only.

Run:  PYTHONPATH=src python dr_tmp/proto_P05.py
Out:  experiments/outputs/proto_P05.json
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
from mfas.metrics import pct, score_from_positions         # noqa: E402
from mfas.experiments.H02 import (                         # noqa: E402
    _init_positions_from_order, greedy_fas_order)
from mfas.utils.seeding import select_device               # noqa: E402


def measure(ds: str, device) -> dict:
    """Time the prologue (stage 0) and the epilogue (runner re-score) for ``ds``."""
    t = time.time()
    g = io.load_dataset(ds)
    t_load = time.time() - t

    # ── Prologue: everything the variant does BEFORE run_rocket starts its clock ──
    t = time.time()
    order = greedy_fas_order(g)
    t_greedy = time.time() - t

    t = time.time()
    pos = _init_positions_from_order(order, device)
    t_init = time.time() - t

    # ── Epilogue: what run_variant does AFTER variant.run returns ────────────────
    # A float32 position vector of the right shape is all the scorer needs; the
    # cost is a gather/argsort over the edge list and does not depend on quality.
    pos_np = np.asarray(pos.detach().cpu().numpy(), dtype=np.float32)
    t = time.time()
    score = score_from_positions(pos_np, np.asarray(g.src), np.asarray(g.tgt), g.weight)
    t_score = time.time() - t

    t = time.time()
    tmp = _ROOT / "dr_tmp" / f"_proto_P05_{ds}.npy"
    np.save(tmp, pos_np)
    t_save = time.time() - t
    tmp.unlink(missing_ok=True)

    return dict(
        dataset=ds, n_nodes=int(g.n_nodes), n_edges=int(np.asarray(g.src).size),
        load_s=t_load,
        prologue_s=t_greedy + t_init,
        greedy_fas_s=t_greedy, init_positions_s=t_init,
        epilogue_s=t_score + t_save,
        rescore_s=t_score, save_s=t_save,
        # The total un-guarded overhead: wall_clock_s(record) - time_limit(variant).
        unguarded_overhead_s=t_greedy + t_init + t_score + t_save,
        greedy_pct=pct(score, g.total_weight),
    )


def main() -> int:
    device, dev_name = select_device("auto")
    rows = [measure(ds, device) for ds in ("mouse", "microns", "connectome")]
    out = dict(
        item="P05",
        what=("prologue/epilogue timings that sit OUTSIDE the variant's own time_limit "
              "clock; used to size the runner's reserve_s"),
        device=dev_name,
        rows=rows,
    )
    dst = _ROOT / "experiments" / "outputs" / "proto_P05.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
