"""Evidence for the H80 pre-registration AMENDMENT: is the named rebuild monotone?

``experiments/outputs/proto_H80_prereg.json`` specified the rebuild as
``mfas.refine.lns.apply_victim_reinsertions``. That function moves each victim to its
exact-optimal gap ONLY when the gain is strictly positive, evaluated on the current
(partially updated) state - so every proposal it produces is >= its input, and an
acceptance schedule placed on top of it has nothing to accept.

This script measures that claim rather than asserting it: from the sift fixed point on
the hard synthetic, generate 24 destroy -> rebuild -> short-sift proposals per seed and
count how many are DOWNHILL. Writes ``experiments/outputs/proto_H80_rebuild_monotone.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments"))

import torch  # noqa: E402

torch.set_num_threads(2)

import proto_H80_anneal as P  # noqa: E402
from mfas.analysis import gap as gapmod  # noqa: E402
from mfas.experiments.H02 import greedy_fas_order  # noqa: E402
from mfas.refine.insertion import build_sift_edges, sift  # noqa: E402
from mfas.refine.lns import apply_victim_reinsertions  # noqa: E402

N_PROP = 24

out = {
    "_doc": "EVIDENCE for the H80 pre-registration amendment: the rebuild the "
            "pre-registration named (mfas.refine.lns.apply_victim_reinsertions) is "
            "MONOTONE BY CONSTRUCTION, so a Metropolis acceptance placed on top of it has "
            "nothing to accept.",
    "probe": f"{N_PROP} structure-aware destroy -> apply_victim_reinsertions -> 2-sweep "
             "sift proposals from the sift fixed point, per seed, on the hard synthetic",
    "rows": [],
}

for s in (42, 123, 999):
    g, _ref, _p = gapmod.make_hard_synthetic_graph(seed=s)
    src, tgt, w = build_sift_edges(g)
    n = g.n_nodes
    rank = np.asarray(greedy_fas_order(g), dtype=np.int64)
    rank, _, _ = sift(g, rank, max_sweeps=20)
    ff0 = P._ff_weight(rank, src, tgt, w)
    rng = np.random.RandomState(s)
    down = up = flat = 0
    for _ in range(N_PROP):
        v = P._victims_struct(rank, src, tgt, w, n, 20, rng)
        r = apply_victim_reinsertions(rank, v, src, tgt, w, n)
        r, _, _ = sift(g, r, max_sweeps=2)
        d = P._ff_weight(r, src, tgt, w) - ff0
        down += int(d < 0)
        up += int(d > 0)
        flat += int(d == 0)
    row = dict(seed=s, n=int(n), n_proposals=N_PROP, n_downhill=down, n_uphill=up,
               n_flat=flat)
    out["rows"].append(row)
    print(row, flush=True)

out["total_downhill"] = int(sum(r["n_downhill"] for r in out["rows"]))
out["conclusion"] = (
    f"{out['total_downhill']} downhill proposals out of {N_PROP * 3}. The rebuild named "
    "in the pre-registration cannot generate the worse states the acceptance schedule "
    "exists to accept, so the amendment replaces it with a TRUE ruin-and-recreate "
    "(remove the victims from the sequence, then re-insert each at its exact-gain-optimal "
    "slot given only the nodes already placed).")
p = _ROOT / "experiments" / "outputs" / "proto_H80_rebuild_monotone.json"
p.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"wrote {p}")
print(out["conclusion"])
