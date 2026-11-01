"""Null-ladder v2 runner (study: biology/studies/null_ladder_v2/CHARTER.md).

Applies the IDENTICAL G01 protocol (``nullmodels_run.run_protocol`` — imported,
not copied: greedy-FAS warm start -> Rocket-ASYM at the fixed epoch budget ->
under-relaxed sift, frozen oracle on CPU) to the v2 rungs:

    W0   w_topo_shuffle   real topology, weights permuted
    L2R  dyad_fixedmut    real reciprocal backbone + exact degrees,
                          unidirectional wiring randomized

Real-graph comparators are REUSED from G01's logged runs (same protocol, same
budget) — this runner never re-runs the real graph. Each run JSON additionally
records the null's 2-cycle lower bound (estimator-free), since W0 changes it
and L2R preserves it exactly.

Usage (allen interpreter $PY):

    $PY biology/analysis/nullmodels2_run.py --dataset connectome --variant w_topo_shuffle
    $PY biology/analysis/nullmodels2_run.py --dataset connectome --variant dyad_fixedmut

Existing result files are skipped, so the same command resumes a sweep.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "biology" / "analysis"))

from mfas.io import GraphData, load_dataset                        # noqa: E402

from nullmodels_run import run_protocol, _finish_run               # noqa: E402
from nullmodels2_generators import (V2_VARIANTS,                    # noqa: E402
                                    dyad_fixedmut_null,
                                    two_cycle_bound,
                                    w_topo_shuffle_null)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — every knob lives here; no magic numbers below
# ──────────────────────────────────────────────────────────────────────────────
CONFIG2 = dict(
    datasets=dict(
        connectome=dict(
            epochs=4_000,            # IDENTICAL to the G01 protocol budget
            device="mps",
            n_nulls=dict(w_topo_shuffle=12, dyad_fixedmut=12),
            null_pipeline_seed=42,   # one pipeline seed per null (G01 convention:
                                     # real across-seed sigma 0.0033 pp << draw sigma)
            save_orders=True,
        ),
        # stage E (MICrONS) is added here only after its sizing run is logged
        # and the budget is frozen in the charter — not before.
    ),
    gen_seed_base=dict(w_topo_shuffle=6_000, dyad_fixedmut=7_000),
    results_dir=_REPO / "biology" / "results" / "nullmodels_v2",
)


def _draw_null(variant: str, src, tgt, w, n: int, gen_seed: int):
    if variant == "w_topo_shuffle":
        return w_topo_shuffle_null(src, tgt, w, seed=gen_seed)
    if variant == "dyad_fixedmut":
        return dyad_fixedmut_null(src, tgt, w, n=n, seed=gen_seed)
    raise ValueError(f"unknown v2 variant {variant!r}")


def run_nulls(dataset: str, variant: str, n_nulls_override: int | None,
              start_index: int) -> None:
    ds = CONFIG2["datasets"][dataset]
    epochs = ds["epochs"]
    device = ds["device"]
    n_nulls = n_nulls_override or ds["n_nulls"][variant]
    pipe_seed = ds["null_pipeline_seed"]
    base = CONFIG2["gen_seed_base"][variant]
    runs_dir = CONFIG2["results_dir"] / "runs"
    orders_dir = (CONFIG2["results_dir"] / "orders") if ds["save_orders"] else None
    g = load_dataset(dataset)
    print(g, flush=True)
    src = np.asarray(g.src); tgt = np.asarray(g.tgt); w = np.asarray(g.weight)
    bound_real = two_cycle_bound(src, tgt, w, g.n_nodes)
    for i in range(start_index, n_nulls):
        gen_seed = base + i
        stem = f"{dataset}__{variant}__gs{gen_seed}__ps{pipe_seed}__e{epochs}"
        out = runs_dir / f"{stem}.json"
        if out.exists():
            print(f"[skip] {out.name} exists", flush=True)
            continue
        t0 = time.time()
        s2, t2, w2, gen_stats = _draw_null(variant, src, tgt, w, g.n_nodes,
                                           gen_seed)
        t_gen = time.time() - t0
        bound_null = two_cycle_bound(s2, t2, w2, g.n_nodes)
        g_null = GraphData(src=s2.astype(np.int32), tgt=t2.astype(np.int32),
                           weight=w2, node_ids=np.arange(g.n_nodes),
                           name=f"null_{variant}")
        res = run_protocol(g_null, epochs=epochs, pipeline_seed=pipe_seed,
                           device=device)
        res["t_generate_s"] = t_gen
        _finish_run(res, out, orders_dir, stem,
                    dict(dataset=dataset, kind=variant, gen_seed=gen_seed,
                         gen_stats=gen_stats,
                         bound_2cycle=bound_null,
                         bound_2cycle_pct=100.0 * bound_null / float(w.sum()),
                         bound_2cycle_real=bound_real,
                         bound_2cycle_real_pct=100.0 * bound_real / float(w.sum())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=list(CONFIG2["datasets"]))
    ap.add_argument("--variant", required=True, choices=list(V2_VARIANTS))
    ap.add_argument("--n-nulls", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()
    run_nulls(args.dataset, args.variant, args.n_nulls, args.start_index)


if __name__ == "__main__":
    main()
