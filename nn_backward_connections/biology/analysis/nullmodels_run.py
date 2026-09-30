"""G01 runner — identical ordering protocol on real connectomes and their nulls.

Estimator contract (experiments/randomgraph.md): best_feedforward must be
estimated the SAME way for the real graph and every null, with a FIXED
epoch/sweep budget (never wall-clock — another agent may share the GPU, and
epoch-budgeting makes contention harmless to fairness), and every ordering is
scored by the frozen oracle ``mfas.metrics.score_from_order`` on CPU.

Protocol (applied identically to real graph and every null):
  stage 1  greedy-FAS warm start   mfas.experiments.H02.greedy_fas_order
           (also reported alone: the cheap lower-bound control of the
           estimator contract — trends must hold for BOTH bounds)
  stage 2  Rocket + ASYM surrogate mfas.experiments.H64._rocket_asym
           at a REDUCED fixed epoch budget (CONFIG, per dataset)
  stage 3  under-relaxed exact-gain sift  mfas.refine.underrelax.sift_underrelaxed
           at reduced sweeps (CONFIG)
  oracle   frozen scorer on CPU, int64/float64

Each run writes its JSON to ``biology/results/nullmodels/runs/`` IMMEDIATELY on
completion (partial progress is never lost); connectome runs also persist the
final rank vector to ``biology/results/nullmodels/orders/`` (int32 .npy) for
the per-lane feedback decomposition.

Usage (always via the ``allen`` interpreter $PY):

    $PY biology/analysis/nullmodels_run.py --dataset mouse      --mode real
    $PY biology/analysis/nullmodels_run.py --dataset mouse      --mode nulls --variant w_shuffled
    $PY biology/analysis/nullmodels_run.py --dataset connectome --mode real
    $PY biology/analysis/nullmodels_run.py --dataset connectome --mode nulls --variant w_source

Existing result files are skipped, so the same command resumes an interrupted
sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "biology" / "analysis"))

import torch  # noqa: E402

from mfas.baseline.rocket import RocketConfig                     # noqa: E402
from mfas.experiments.H02 import (_init_positions_from_order,     # noqa: E402
                                  greedy_fas_order)
from mfas.experiments.H64 import _rocket_asym                     # noqa: E402
from mfas.io import GraphData, load_dataset                       # noqa: E402
from mfas.metrics import pct, score_from_order                    # noqa: E402
from mfas.refine.underrelax import sift_underrelaxed              # noqa: E402

from nullmodels_generators import (VARIANTS,                       # noqa: E402
                                   block_configuration_null,
                                   directed_configuration_null,
                                   fly_lane_labels, gnm_null, sbm_null)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — every knob lives here; no magic numbers below
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = dict(
    datasets=dict(
        mouse=dict(
            epochs=2_000,            # stage-2 Rocket epoch budget (fixed, per graph)
            device="cpu",            # 148 nodes / 583 edges: CPU beats MPS overhead
            # ladder rungs for the mouse: L1 + L2 only. The mouse's nodes ARE
            # brain regions, so L3 (region-blocked config model) degenerates to
            # the real graph and is not run.
            n_nulls=dict(w_shuffled=100, w_source=100, L1_gnm=100),
            real_pipeline_seeds=(42, 123, 999),   # estimator-noise probe on the real graph
            null_pipeline_seed=42,   # one pipeline seed per null (graph seed varies)
            save_orders=False,
        ),
        connectome=dict(
            epochs=4_000,            # calibrated: one full run ~100 s on MPS
            device="mps",
            n_nulls=dict(w_shuffled=20, w_source=20,
                         L1_gnm=12, L3a_sbm=12, L3_dcsbm=12),
            real_pipeline_seeds=(42, 123, 999),
            null_pipeline_seed=42,
            save_orders=True,        # int32 rank vectors for the per-lane decomposition
        ),
    ),
    # stage-3 sift budget (identical for every graph of a dataset)
    sift_max_sweeps=8,
    sift_k_full=2,
    sift_alpha=0.7,
    # graph-generation seeds: null i of a variant uses base + i
    gen_seed_base=dict(w_shuffled=1_000, w_source=2_000, L1_gnm=3_000,
                       L3a_sbm=4_000, L3_dcsbm=5_000),
    results_dir=_REPO / "biology" / "results" / "nullmodels",
)

ALL_VARIANTS = ("w_shuffled", "w_source", "L1_gnm", "L3a_sbm", "L3_dcsbm")


def _draw_null(variant: str, src, tgt, w, n: int, gen_seed: int, labels):
    """Dispatch one null draw for the requested ladder rung."""
    if variant in VARIANTS:                       # L2 (config model, 2 weight modes)
        return directed_configuration_null(src, tgt, w, variant=variant,
                                           seed=gen_seed)
    if variant == "L1_gnm":
        return gnm_null(src, tgt, w, n=n, seed=gen_seed)
    if variant == "L3a_sbm":
        return sbm_null(src, tgt, w, labels, seed=gen_seed)
    if variant == "L3_dcsbm":
        return block_configuration_null(src, tgt, w, labels, seed=gen_seed)
    raise ValueError(f"unknown variant {variant!r}")


# ──────────────────────────────────────────────────────────────────────────────
# The fixed ordering protocol (the estimator)
# ──────────────────────────────────────────────────────────────────────────────
def run_protocol(g: GraphData, *, epochs: int, pipeline_seed: int, device: str
                 ) -> dict:
    """Apply the fixed 3-stage protocol to one graph; return all logged numbers.

    Returns a dict with per-stage scores/pcts/runtimes and the final rank
    vector under key ``"final_rank"`` (int64, popped by the caller before the
    JSON dump). Every discrete score comes from the frozen oracle on CPU.
    """
    src = np.asarray(g.src)
    tgt = np.asarray(g.tgt)
    w = np.asarray(g.weight)
    total = g.total_weight

    # stage 1 — greedy-FAS warm start (also the lower-bound control)
    t0 = time.time()
    order1 = greedy_fas_order(g)
    t_greedy = time.time() - t0
    score_greedy = score_from_order(order1, src, tgt, w)

    # stage 2 — Rocket with the ASYM surrogate, fixed epoch budget
    init_pos = _init_positions_from_order(order1, device)
    cfg = RocketConfig(epochs=epochs)
    rocket = _rocket_asym(g, cfg, seed=pipeline_seed, device=device,
                          init_positions=init_pos)
    rank0 = np.argsort(np.argsort(rocket.best_positions, kind="stable"),
                       kind="stable").astype(np.int64)
    score_rank0 = score_from_order(rank0, src, tgt, w)

    # stage 3 — under-relaxed exact-gain sift, fixed sweep budget
    t0 = time.time()
    sift_rank, sift_score, sweep_log = sift_underrelaxed(
        g, rank0, k_full=CONFIG["sift_k_full"], alpha=CONFIG["sift_alpha"],
        max_sweeps=CONFIG["sift_max_sweeps"])
    t_sift = time.time() - t0

    # final = best-by-oracle (sift tracker already includes rank0)
    final_score = float(max(sift_score, score_rank0))
    final_rank = sift_rank if sift_score >= score_rank0 else rank0

    return dict(
        n_nodes=g.n_nodes, n_edges=g.n_edges, total_weight=total,
        pipeline_seed=pipeline_seed, epochs=epochs, device=device,
        sift_max_sweeps=CONFIG["sift_max_sweeps"],
        sift_k_full=CONFIG["sift_k_full"], sift_alpha=CONFIG["sift_alpha"],
        score_greedy=float(score_greedy), pct_greedy=pct(score_greedy, total),
        score_rocket=float(rocket.best_score),
        pct_rocket=pct(rocket.best_score, total),
        score_rank0=float(score_rank0), pct_rank0=pct(score_rank0, total),
        score_final=final_score, pct_final=pct(final_score, total),
        unavoidable_fb_pct_final=100.0 - pct(final_score, total),
        unavoidable_fb_pct_greedy=100.0 - pct(score_greedy, total),
        n_sift_sweeps=len(sweep_log),
        n_epochs_done=int(rocket.n_epochs_done),
        t_greedy_s=t_greedy, t_rocket_s=float(rocket.wall_clock_s),
        t_sift_s=t_sift,
        t_total_s=t_greedy + float(rocket.wall_clock_s) + t_sift,
        final_rank=final_rank,
    )


def _finish_run(res: dict, out_json: Path, orders_dir: Path | None,
                stem: str, extra: dict) -> None:
    """Pop the rank vector, optionally persist it, write the JSON record."""
    final_rank = res.pop("final_rank")
    if orders_dir is not None:
        orders_dir.mkdir(parents=True, exist_ok=True)
        np.save(orders_dir / f"{stem}.npy", final_rank.astype(np.int32))
        res["order_npy"] = str((orders_dir / f"{stem}.npy").relative_to(_REPO))
    res.update(extra)
    res["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    res["torch_version"] = torch.__version__
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.rename(out_json)
    print(f"[done] {out_json.name}  final={res['pct_final']:.4f}%  "
          f"greedy={res['pct_greedy']:.4f}%  ({res['t_total_s']:.1f}s)", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Drivers
# ──────────────────────────────────────────────────────────────────────────────
def run_real(dataset: str, epochs_override: int | None, device_override: str | None,
             seeds_override: str | None) -> None:
    """Run the protocol on the REAL graph at each real_pipeline_seed."""
    ds = CONFIG["datasets"][dataset]
    epochs = epochs_override or ds["epochs"]
    device = device_override or ds["device"]
    seeds = ([int(s) for s in seeds_override.split(",")] if seeds_override
             else list(ds["real_pipeline_seeds"]))
    runs_dir = CONFIG["results_dir"] / "runs"
    orders_dir = (CONFIG["results_dir"] / "orders") if ds["save_orders"] else None
    g = load_dataset(dataset)
    print(g, flush=True)
    for seed in seeds:
        stem = f"{dataset}__real__ps{seed}__e{epochs}"
        out = runs_dir / f"{stem}.json"
        if out.exists():
            print(f"[skip] {out.name} exists", flush=True)
            continue
        res = run_protocol(g, epochs=epochs, pipeline_seed=seed, device=device)
        _finish_run(res, out, orders_dir, stem,
                    dict(dataset=dataset, kind="real", gen_seed=None))


def run_nulls(dataset: str, variant: str, n_nulls_override: int | None,
              epochs_override: int | None, device_override: str | None,
              start_index: int) -> None:
    """Draw nulls i = start_index .. n_nulls-1 and run the protocol on each."""
    ds = CONFIG["datasets"][dataset]
    epochs = epochs_override or ds["epochs"]
    device = device_override or ds["device"]
    if variant not in ds["n_nulls"]:
        raise SystemExit(f"variant {variant!r} is not part of the {dataset} ladder "
                         f"(available: {list(ds['n_nulls'])})")
    n_nulls = n_nulls_override or ds["n_nulls"][variant]
    pipe_seed = ds["null_pipeline_seed"]
    base = CONFIG["gen_seed_base"][variant]
    runs_dir = CONFIG["results_dir"] / "runs"
    orders_dir = (CONFIG["results_dir"] / "orders") if ds["save_orders"] else None
    g = load_dataset(dataset)
    print(g, flush=True)
    src = np.asarray(g.src); tgt = np.asarray(g.tgt); w = np.asarray(g.weight)
    labels = None
    if variant in ("L3a_sbm", "L3_dcsbm"):
        labels, lane_list = fly_lane_labels()
        print(f"lanes: K={len(lane_list)}", flush=True)
    for i in range(start_index, n_nulls):
        gen_seed = base + i
        stem = f"{dataset}__{variant}__gs{gen_seed}__ps{pipe_seed}__e{epochs}"
        out = runs_dir / f"{stem}.json"
        if out.exists():
            print(f"[skip] {out.name} exists", flush=True)
            continue
        t0 = time.time()
        s2, t2, w2, gen_stats = _draw_null(variant, src, tgt, w, g.n_nodes,
                                           gen_seed, labels)
        t_gen = time.time() - t0
        g_null = GraphData(src=s2.astype(np.int32), tgt=t2.astype(np.int32),
                           weight=w2, node_ids=np.arange(g.n_nodes),
                           name=f"null_{variant}")
        res = run_protocol(g_null, epochs=epochs, pipeline_seed=pipe_seed,
                           device=device)
        res["t_generate_s"] = t_gen
        _finish_run(res, out, orders_dir, stem,
                    dict(dataset=dataset, kind=variant, gen_seed=gen_seed,
                         gen_stats=gen_stats))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=list(CONFIG["datasets"]))
    ap.add_argument("--mode", required=True, choices=["real", "nulls"])
    ap.add_argument("--variant", choices=list(ALL_VARIANTS),
                    help="ladder rung / null variant (required for --mode nulls)")
    ap.add_argument("--n-nulls", type=int, default=None,
                    help="override the CONFIG null count")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the CONFIG epoch budget (calibration only)")
    ap.add_argument("--device", default=None, help="override the CONFIG device")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated pipeline seeds for --mode real")
    ap.add_argument("--start-index", type=int, default=0,
                    help="first null index to run (resume aid)")
    ap.add_argument("--sift-sweeps", type=int, default=None,
                    help="override CONFIG sift_max_sweeps (robustness arm)")
    args = ap.parse_args()

    if args.sift_sweeps is not None:
        CONFIG["sift_max_sweeps"] = args.sift_sweeps

    if args.mode == "real":
        run_real(args.dataset, args.epochs, args.device, args.seeds)
    else:
        if not args.variant:
            ap.error("--variant is required for --mode nulls")
        run_nulls(args.dataset, args.variant, args.n_nulls, args.epochs,
                  args.device, args.start_index)


if __name__ == "__main__":
    main()
