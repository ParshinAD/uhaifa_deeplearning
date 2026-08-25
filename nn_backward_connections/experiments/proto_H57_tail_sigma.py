"""H57 prototype - how much of the champion's relabelling dispersion lives in the TAIL?

Question this settles
---------------------
H56 died on arithmetic (meta-rule M9): best-of-R over R full-cost pipeline runs gains
``sigma * a_R`` and ``R = floor(guard_deadline / per_run_wall)``, which on connectome is 2,
so the required sigma is 0.021269 pp against a measured 0.019124 pp.

H57 changes the DENOMINATOR rather than the framing: pay the expensive prefix
(greedy-FAS -> Rocket -> first under-relaxed sift) ONCE, and spend the remaining seconds on
R randomised refinement TAILS (stage 4, ``alternate_scc_sift``). That makes a larger R
affordable, which lowers the required sigma.

It only works if the dispersion is actually IN the tail. H56 measured the WHOLE-pipeline
relabelling sigma (0.019124 pp, 5 points, ``proto_P09_connectome.json``); nothing has ever
measured how that splits between the prefix and the tail. If greedy-FAS tie-breaks create
it, it lives in the shared prefix and this design cannot reach it - H57 is then dead on
arrival for the same reason H56 was, and this prototype is what says so.

Design
------
The perturbation is a node RELABELLING, exactly as in P09/H56: replacing contiguous index
``i`` by ``perm[i]`` yields an ISOMORPHIC graph, so nothing an algorithm ought to do changes
and only index-order tie-breaks move. Reusing ``proto_P09_relabel.relabel`` and its fixed
``_PERM_SEED_BASE = 909000`` means arms r = 1..5 here are the SAME five labellings whose
whole-pipeline scores P09 measured - the two sigmas are therefore paired, not merely
comparable.

``--arm prefix``  runs H42 stages 1-3 once on the CANONICAL labelling (seed 42) and persists
                  the post-sift rank plus per-stage wall clocks.
``--arm tails``   maps that one rank into labelling r and runs the champion's stage 4 on it,
                  for r = 0..R. Every arm therefore starts from the SAME order up to
                  isomorphism, so any score spread is tail tie-breaks and nothing else.

r = 0 is the identity and is a VALIDITY GATE, not a data point: prefix + identity tail is
the champion pipeline, so it must reproduce 84.15409511053134 exactly. If it does not, the
prefix was not the champion's and every other arm is meaningless.

Everything after the prefix is CPU: ``alternate_scc_sift`` is numpy/scipy and draws from no
RNG, so the only input that changes between arms is the labelling.

Reproduce
---------
    PYTHONPATH=src $PY experiments/proto_H57_tail_sigma.py --arm prefix --dataset connectome
    PYTHONPATH=src $PY experiments/proto_H57_tail_sigma.py --arm tails  --dataset connectome --r-max 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io                                              # noqa: E402
from mfas.metrics import pct, score_from_order                   # noqa: E402
from mfas.refine import alternate_scc_sift, sift_underrelaxed    # noqa: E402
from mfas.baseline.rocket import RocketConfig, run_rocket        # noqa: E402
from mfas.experiments.H02 import (_init_positions_from_order,    # noqa: E402
                                  greedy_fas_order)
from mfas.utils.seeding import seed_everything, select_device    # noqa: E402
from eval.frozen_guard import verify_frozen_manifest             # noqa: E402
from proto_P09_relabel import relabel, _PERM_SEED_BASE           # noqa: E402

# The champion's constants are IMPORTED rather than copied, so a drift in H42 cannot
# silently make this study describe a pipeline the campaign no longer runs.
from mfas.experiments import H42                                 # noqa: E402

_OUT = _ROOT / "experiments" / "outputs"
_SCRATCH = _ROOT / "dr_tmp"
_SEED = 42

# a_R = E[max of R independent standard normals]. Same table as proto_H56 / meta-rule M9.
_A_R = {1: 0.0, 2: 0.5641895835477563, 3: 0.8462843753216345,
        4: 1.0293753730039641, 5: 1.1629644736405101, 6: 1.2672063606114712,
        7: 1.3521783756070915, 8: 1.4236003060469847}

_GUARD_DEADLINE_S = 3450.0
_MIN_PROMOTION_DELTA_PP = 0.012          # campaign.yaml datasets.connectome
_WHOLE_PIPELINE_SIGMA_PP = 0.019124286772961963   # proto_P09_connectome.json


def _prefix_path(ds: str) -> Path:
    return _SCRATCH / ("proto_H57_prefix_rank_" + ds + ".npy")


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def run_prefix(ds: str, device_str: str) -> dict:
    """Run H42 stages 1-3 once on the canonical labelling and persist the post-sift rank."""
    g = io.load_dataset(ds)
    seed_everything(_SEED)
    torch_device, dev_name = select_device(device_str)

    t0 = time.time()
    order = greedy_fas_order(g)
    t_greedy = time.time() - t0

    init_positions = _init_positions_from_order(order, torch_device)
    t1 = time.time()
    rocket = run_rocket(g, RocketConfig(epochs=H42._EPOCHS[ds]), seed=_SEED,
                        device=torch_device, init_positions=init_positions,
                        time_limit=None)
    t_rocket = time.time() - t1

    rank0 = np.argsort(np.argsort(rocket.best_positions, kind="stable"),
                       kind="stable").astype(np.int64)
    t2 = time.time()
    sift_rank, sift_score, sweep_log = sift_underrelaxed(
        g, rank0, k_full=H42._K_FULL, alpha=H42._ALPHA,
        max_sweeps=H42._MAX_SWEEPS[ds], time_budget_s=None)
    t_sift = time.time() - t2

    _SCRATCH.mkdir(parents=True, exist_ok=True)
    np.save(_prefix_path(ds), sift_rank.astype(np.int64))

    return dict(
        dataset=ds, seed=_SEED, device=dev_name,
        pure_best_pct=float(rocket.best_pct),
        sift_best_score=float(sift_score),
        sift_best_pct=float(pct(sift_score, g.total_weight)),
        n_sift_sweeps=len(sweep_log),
        t_greedy_s=t_greedy, t_rocket_s=t_rocket, t_sift_s=t_sift,
        prefix_wall_s=t_greedy + t_rocket + t_sift,
        rank_sha16=_sha(sift_rank.astype(np.int64)),
        rank_path=str(_prefix_path(ds).name),
    )


def run_tails(ds: str, r_max: int, out_path: Path, prefix_meta: dict) -> dict:
    """Run the champion's stage 4 from ONE prefix rank under r = 0..r_max labellings."""
    g0 = io.load_dataset(ds)
    rank0 = np.load(_prefix_path(ds)).astype(np.int64)
    assert rank0.shape == (g0.n_nodes,), "prefix rank does not match the graph"

    rows = []
    for r in range(r_max + 1):
        g, perm = relabel(g0, r)
        # The rank follows the nodes: the node at old index i sits at new index perm[i],
        # so the ORDER is identical up to isomorphism in every arm.
        rank_r = np.empty_like(rank0)
        rank_r[perm] = rank0

        t0 = time.time()
        best_rank, best_score, log = alternate_scc_sift(
            g, rank_r,
            n_cycles=H42._ALT_CYCLES[ds],
            sift_sweeps=H42._ALT_SIFT_SWEEPS[ds],
            k_full=H42._ALT_K_FULL, alpha=H42._ALPHA, min_block=H42._MIN_BLOCK,
            time_budget_s=None)
        wall = time.time() - t0

        # Re-score with the frozen oracle rather than trusting the refiner's own number.
        oracle = score_from_order(best_rank, np.asarray(g.src), np.asarray(g.tgt), g.weight)
        assert oracle == best_score, "H57 arm r=%d: refiner/oracle score mismatch" % r

        rows.append(dict(r=r, identity=(r == 0), score=float(oracle),
                         pct=pct(oracle, g.total_weight), tail_wall_s=wall,
                         n_alt_cycles=len(log)))
        print("[H57] r=%d pct=%.8f tail=%.1fs" % (r, rows[-1]["pct"], wall), flush=True)
        _write(out_path, ds, prefix_meta, rows, r_max)

    return _write(out_path, ds, prefix_meta, rows, r_max)


def _write(out_path: Path, ds: str, prefix_meta: dict, rows, r_max: int) -> dict:
    """Serialise after every arm, so a session that dies still leaves usable evidence."""
    rnd = [x for x in rows if not x["identity"]]
    ident = next((x for x in rows if x["identity"]), None)
    doc = dict(
        item="H57", gate="prototype", dataset=ds, date="2026-08-25",
        champion="H42", prereg="experiments/outputs/proto_H57_prereg.json",
        perm_seed_base=_PERM_SEED_BASE, r_max=r_max,
        prefix=prefix_meta, rows=rows,
    )
    if rnd:
        pcts = np.array([x["pct"] for x in rnd], dtype=float)
        tail_walls = np.array([x["tail_wall_s"] for x in rows], dtype=float)
        prefix_wall = float(prefix_meta.get("prefix_wall_s", float("nan")))
        tail_wall = float(np.median(tail_walls))
        r_feasible = int(max(1, np.floor((_GUARD_DEADLINE_S - prefix_wall) / tail_wall)))
        a_r = _A_R.get(min(r_feasible, 8), _A_R[8])
        req = (_MIN_PROMOTION_DELTA_PP / a_r) if a_r > 0 else float("inf")
        s = float(pcts.std(ddof=1)) if len(pcts) > 1 else 0.0
        doc["summary"] = dict(
            n_random=int(len(pcts)), tail_mean_pct=float(pcts.mean()),
            tail_sigma_pp=s, tail_min_pct=float(pcts.min()),
            tail_max_pct=float(pcts.max()),
            tail_span_pp=float(pcts.max() - pcts.min()),
            n_distinct=int(len(np.unique(pcts))),
            identity_pct=(ident["pct"] if ident else None),
            canonical_z=(((ident["pct"] - pcts.mean()) / s)
                         if (ident and s > 0) else None),
            prefix_wall_s=prefix_wall, tail_wall_s_median=tail_wall,
            guard_deadline_s=_GUARD_DEADLINE_S, r_feasible=r_feasible, a_r=a_r,
            required_sigma_pp=req, clears_m9=bool(s >= req),
            whole_pipeline_sigma_pp=_WHOLE_PIPELINE_SIGMA_PP,
            tail_share_of_variance=float(s ** 2 / _WHOLE_PIPELINE_SIGMA_PP ** 2),
        )
    out_path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(description="H57 tail-only relabelling dispersion")
    p.add_argument("--arm", choices=["prefix", "tails"], required=True)
    p.add_argument("--dataset", default="connectome")
    p.add_argument("--r-max", type=int, default=8)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    verify_frozen_manifest()
    _OUT.mkdir(parents=True, exist_ok=True)
    out_path = _OUT / ("proto_H57_" + a.dataset + ".json")
    meta_path = _SCRATCH / ("proto_H57_prefix_" + a.dataset + ".json")

    if a.arm == "prefix":
        meta = run_prefix(a.dataset, a.device)
        _SCRATCH.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(json.dumps(meta, indent=1))
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        doc = run_tails(a.dataset, a.r_max, out_path, meta)
        print(json.dumps(doc.get("summary", {}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
