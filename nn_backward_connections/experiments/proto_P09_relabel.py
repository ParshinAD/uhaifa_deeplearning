"""P09 - measure the nuisance dispersion of a DETERMINISTIC pipeline by node relabelling.

Why this script exists
----------------------
``autoresearch/campaign.yaml`` carries two thresholds a connectome promotion must clear, and on
2026-08-17 they disagreed for the first time (queue item P09): H52 beat the champion by
+0.02093 pp, which clears ``min_promotion_delta_pp`` (0.012) and fails the PROTOCOL CI
(1.96 * baseline_sigma_pp * sqrt(2/n) = 0.02343 pp at n = 5).

The PROTOCOL CI floors sigma at ``baseline_sigma_pp = 0.0189``, a noise floor inherited from the
MPS era when the pipeline was measured as stochastic. On CUDA these pipelines never draw from
``seed``: the confirm pools are five runs with ONE distinct value and std exactly 0. So the test
asks whether a deterministic delta exceeds a historical STOCHASTIC noise floor from a different
device era, and the answer moves with ``n`` - a quantity that carries no information when every
run returns the same bits. That is not a criterion; it is a fossil.

What replaces it has to be a MEASUREMENT of the thing the fossil stands in for: *how much would
this result move for reasons that have nothing to do with the mechanism?* For a deterministic
algorithm on a fixed instance there is no sampling error, but there is nuisance sensitivity, and
there is an exact symmetry that exposes it.

The symmetry
------------
Relabelling the nodes - replacing contiguous index ``i`` by ``perm[i]`` throughout - produces an
ISOMORPHIC graph. The edge multiset, the weights and the total weight are unchanged, and the set
of achievable feedforward scores is identical: any order on the original has an image on the
relabelled graph with exactly the same score. Nothing an algorithm *ought* to do changes.

Everything a tie-break does change. greedy-FAS breaks ties by node index, the sift sweeps in
index order, and the SCC recursion enumerates components in index order. So the score spread
across relabellings is exactly the quantity ``baseline_sigma_pp`` is guessing at, measured on
THIS device, for THIS pipeline.

(An earlier draft also listed stage 5's candidate scan here. It does NOT move: its heap key is
``(-weight, edge_index)`` and edge ROW order is deliberately left untouched below, so stage 5 is
invariant under this perturbation and changes only through its input order. Cycle-9 critic, D7.)

The design is PAIRED: for each relabelling both arms run on the same relabelled graph, so
``delta_r = pct_variant(r) - pct_comparator(r)`` is a matched difference and the nuisance common
to both arms cancels. r = 0 is the identity permutation and must reproduce the recorded confirm
numbers exactly - that is the validity check on this script's harness bypass.

Run
---
    PY=/c/ProgramData/anaconda3/envs/allen/python.exe
    PYTHONPATH=src $PY experiments/proto_P09_relabel.py --dataset mouse --relabels 20
    PYTHONPATH=src $PY experiments/proto_P09_relabel.py --dataset connectome --relabels 4

Scores come from the FROZEN oracle (``mfas.metrics.score_from_positions``); the frozen manifest is
verified before anything runs. This writes to ``experiments/outputs/``, never to ``results/`` -
these are not harness runs and must never be pooled into a champion's evidence.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io                                              # noqa: E402
from mfas.metrics import pct, score_from_positions               # noqa: E402
from mfas.utils.seeding import seed_everything, select_device    # noqa: E402
from eval.frozen_guard import verify_frozen_manifest             # noqa: E402
from eval import runtime_guard                                   # noqa: E402

# Permutation seeds are FIXED constants, not derived from anything, so the study is
# reproducible by checkout. r = 0 is reserved for the identity.
_PERM_SEED_BASE = 909000


def relabel(g, r):
    """Return an isomorphic copy of ``g`` under relabelling ``r`` (r = 0 -> identity).

    ``perm[i]`` is the NEW contiguous index of old index ``i``. Node ids follow the nodes:
    ``node_ids_new[perm[i]] = node_ids_old[i]``, so the new object still satisfies GraphData's
    contract that ``node_ids[j]`` is the on-disk id of contiguous index ``j``. Edge ROW order is
    left untouched: the perturbation under study is the labelling, not the file.
    """
    n = g.n_nodes
    if r == 0:
        return g, np.arange(n, dtype=np.int64)
    perm = np.random.default_rng(_PERM_SEED_BASE + r).permutation(n).astype(np.int64)
    node_ids_new = np.empty_like(g.node_ids)
    node_ids_new[perm] = g.node_ids
    g2 = replace(g,
                 src=perm[np.asarray(g.src)].astype(g.src.dtype, copy=False),
                 tgt=perm[np.asarray(g.tgt)].astype(g.tgt.dtype, copy=False),
                 node_ids=node_ids_new)
    return g2, perm


def run_arm(exp_id, g, dataset, seed, device_str, positions_path=None):
    """Run one variant on one (possibly relabelled) graph and score it with the frozen oracle.

    ``positions_path`` persists the produced order (D6, cycle-9 critic). Without it these runs
    are the only numbers in the campaign that cannot be re-scored against the frozen oracle after
    the fact — and since P09 they are numbers a promotion turns on. ~546 KB per connectome run.
    """
    variant = importlib.import_module("mfas.experiments." + exp_id)
    seed_everything(seed)
    torch_device, dev_name = select_device(device_str)
    plan = runtime_guard.resolve(dataset, None)          # same deadline production runs get
    t0 = time.time()
    result = variant.run(g, seed=seed, device=torch_device, time_limit=plan["time_limit_s"])
    wall = time.time() - t0
    score = score_from_positions(result.best_positions, np.asarray(g.src),
                                 np.asarray(g.tgt), g.weight)
    assert score == result.best_score, "proto_P09/variant score mismatch"
    out = dict(exp=exp_id, score=float(score), pct=pct(score, g.total_weight),
               wall_clock_s=wall, device=dev_name,
               n_epochs_done=int(result.n_epochs_done))
    if positions_path is not None:
        Path(positions_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(positions_path, result.best_positions)
        out["positions_path"] = str(Path(positions_path).name)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="P09 relabelling-dispersion study")
    p.add_argument("--dataset", required=True, choices=list(io.DATASETS))
    p.add_argument("--relabels", type=int, default=4,
                   help="number of NON-identity relabellings; r=0 (identity) is always run")
    p.add_argument("--comparator", default="H42")
    p.add_argument("--variant", default="H52")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--no-positions", action="store_true", dest="no_positions",
                   help="do not persist the produced orders (they are what makes the study "
                        "re-scorable; only use this for a throwaway smoke test)")
    args = p.parse_args(argv)

    verify_frozen_manifest()
    out = Path(args.out) if args.out else (
        _ROOT / "experiments" / "outputs" / ("proto_P09_" + args.dataset + ".json"))
    out.parent.mkdir(parents=True, exist_ok=True)

    g0 = io.load_dataset(args.dataset)
    print("[P09] " + repr(g0), flush=True)

    rows = []
    rec = dict(item="P09", dataset=args.dataset, comparator=args.comparator,
               variant=args.variant, seed=args.seed, n_nodes=g0.n_nodes,
               total_weight=g0.total_weight, perm_seed_base=_PERM_SEED_BASE, rows=rows)

    for r in range(0, args.relabels + 1):
        g, perm = relabel(g0, r)
        assert float(g.weight.sum()) == g0.total_weight, "relabelling changed total weight"
        row = dict(r=r, identity=(r == 0))
        for role, exp_id in (("comparator", args.comparator), ("variant", args.variant)):
            ppath = (out.parent / "relabel_positions" /
                     ("%s_%s_%s_r%d_positions.npy" % (args.dataset, exp_id, role, r)))
            res = run_arm(exp_id, g, args.dataset, args.seed, args.device,
                          positions_path=(None if args.no_positions else ppath))
            row[role] = res
            print("[P09] r=%d %s=%s pct=%.8f wall=%.1fs"
                  % (r, role, exp_id, res["pct"], res["wall_clock_s"]), flush=True)
        row["delta_pp"] = row["variant"]["pct"] - row["comparator"]["pct"]
        rows.append(row)
        print("[P09] r=%d delta=%+.6f pp" % (r, row["delta_pp"]), flush=True)
        # Write after EVERY relabelling: a cycle that dies keeps every point it paid for.
        with open(out, "w") as f:
            json.dump(rec, f, indent=2)

    d = np.array([x["delta_pp"] for x in rows], dtype=np.float64)
    c = np.array([x["comparator"]["pct"] for x in rows], dtype=np.float64)
    v = np.array([x["variant"]["pct"] for x in rows], dtype=np.float64)
    many = d.size > 1
    rec["summary"] = dict(
        n_points=int(d.size),
        comparator_pct_mean=float(c.mean()),
        comparator_pct_std=float(c.std(ddof=1)) if many else 0.0,
        variant_pct_mean=float(v.mean()),
        variant_pct_std=float(v.std(ddof=1)) if many else 0.0,
        delta_mean_pp=float(d.mean()),
        delta_std_pp=float(d.std(ddof=1)) if many else 0.0,
        delta_min_pp=float(d.min()), delta_max_pp=float(d.max()),
        n_positive=int((d > 0).sum()),
    )
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print("[P09] wrote " + str(out))
    print(json.dumps(rec["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
