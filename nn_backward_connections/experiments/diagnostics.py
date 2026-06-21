"""Phase-4 Stage-A diagnostics driver — Rocket↔best-solution gap.

Runs the five diagnostic steps and writes a machine-readable summary to
``experiments/outputs/diagnosis.json`` plus figures. Imports the PRIVILEGED analysis
module :mod:`mfas.analysis.gap` (the only reader of ``data/best_solution``); this driver
itself never feeds the answer into any optimization path. Diagnostic outputs go to
``experiments/outputs/`` — NEVER ``results/`` (which is for real, leakage-clean variant
scores only).

Usage
-----
    python experiments/diagnostics.py [--quick] [--steps 0,1,2,3,4]

``--quick`` skips the slow connectome drift probe and trims the init→plateau grid.
Each connectome Rocket run is ~75 s on MPS; the full driver runs ~15–25 min.

Reproducibility: numbers cited in ``experiments/diagnosis.md`` come from this script's
JSON output. Re-run with the same code/seed to reproduce.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

from mfas import io  # noqa: E402
from mfas.analysis import gap  # noqa: E402
from mfas.baseline.rocket import (RocketConfig, make_init_positions,  # noqa: E402
                                  run_rocket)
from mfas.metrics import pct, score_from_positions  # noqa: E402

OUT = _ROOT / "experiments" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
BETAS = [0.05, 0.5, 1.05]                 # schedule range (min, mid, max/convergence)
EPOCHS = {"connectome": 20_000, "mouse": 5_000}


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _score_pct(pos, g):
    return pct(score_from_positions(pos, np.asarray(g.src), np.asarray(g.tgt), g.weight),
               g.total_weight)


def _latest(pattern):
    fs = sorted(glob.glob(str(_ROOT / "results" / pattern)))
    return fs[-1] if fs else None


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — decisive surrogate (connectome): scale-fair static + drift probe
# ──────────────────────────────────────────────────────────────────────────────
def step1_decisive(g, order, dev, quick: bool) -> dict:
    out = {"betas": BETAS}
    # converged Rocket positions (reuse H02 + baseline saved artifacts; no re-run)
    h02_f = _latest("*H02-connectome-s42-confirm*_positions.npy")
    base_f = _latest("*baseline_passthrough-connectome-s42-implement*_positions.npy")
    h02 = np.load(h02_f).astype(np.float64)
    base = np.load(base_f).astype(np.float64)
    out["rocket_converged_pct"] = {"H02": _score_pct(h02, g), "baseline": _score_pct(base, g)}
    out["rocket_converged_pos_std"] = {"H02": float(h02.std()), "baseline": float(base.std())}

    # scale-fair: best order's MAX achievable surrogate vs Rocket's converged surrogate
    static = []
    for beta in BETAS:
        _, best_max = gap.optimize_spacing(order, g, beta=beta, iters=250, lr=0.1)
        static.append(dict(beta=beta,
                           best_order_max_surrogate=best_max,
                           H02_converged_surrogate=gap.surrogate_objective(h02, g, beta),
                           base_converged_surrogate=gap.surrogate_objective(base, g, beta)))
    out["static_surrogate"] = static
    # verdict primitive: at every beta, does best's order out-surrogate Rocket's converged?
    out["best_outsurrogates_rocket_all_beta"] = all(
        s["best_order_max_surrogate"] > s["H02_converged_surrogate"] for s in static)

    # drift probe: run Rocket FROM best; cyclic vs constant-beta (does it hold?)
    drift = []
    probes = [("cyclic_lr.05", RocketConfig(epochs=EPOCHS["connectome"], cycles=5, lr=0.05), "even")]
    if not quick:
        probes += [
            ("constB_lr.05", RocketConfig(epochs=EPOCHS["connectome"], cycles=0, lr=0.05), "even"),
            ("constB_lr.005", RocketConfig(epochs=EPOCHS["connectome"], cycles=0, lr=0.005), "even"),
        ]
    for name, cfg, scale in probes:
        t = time.time()
        r = gap.drift_probe(g, order, seed=42, device=dev, init_scale=scale, cfg=cfg)
        drift.append(dict(probe=name, init_pct=r.init_pct, final_pct=r.final_pct,
                         best_seen_pct=r.best_pct, min_pct=float(r.history["pct"].min()),
                         dropped=r.dropped, wall_s=round(time.time() - t, 1)))
    out["drift_probe"] = drift
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — gap decomposition (connectome)
# ──────────────────────────────────────────────────────────────────────────────
def step2_decomp(g, order) -> dict:
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    w = np.asarray(g.weight, dtype=np.int64)
    total = int(w.sum())
    h02 = np.load(_latest("*H02-connectome-s42-confirm*_positions.npy")).astype(np.float64)
    rock_rank = np.argsort(np.argsort(h02, kind="stable"), kind="stable")

    ff_best = order[tgt] > order[src]
    ff_rock = rock_rank[tgt] > rock_rank[src]
    disagree = ff_best != ff_rock
    best_gain = disagree & ff_best & ~ff_rock
    rock_gain = disagree & ff_rock & ~ff_best

    by_w = []
    for lo, hi in [(1, 1), (2, 2), (3, 5), (6, 20), (21, 100), (101, 10**9)]:
        m = (w >= lo) & (w <= hi)
        if m.sum() == 0:
            continue
        by_w.append(dict(lo=lo, hi=hi, n_edges=int(m.sum()),
                        disagree_rate=float((m & disagree).sum() / m.sum())))
    deg = (np.bincount(src, minlength=g.n_nodes) + np.bincount(tgt, minlength=g.n_nodes))
    emax = np.maximum(deg[src], deg[tgt])
    qs = np.quantile(emax, [0.5, 0.9, 0.99])
    by_deg = {name: float(w[(mask) & disagree].sum() / total)
              for name, mask in [("le_median", emax <= qs[0]),
                                 ("gt_p90", emax > qs[1]), ("gt_p99", emax > qs[2])]}
    return dict(
        disagree_edges=int(disagree.sum()),
        disagree_edge_frac=float(disagree.sum() / len(w)),
        disagree_weight_frac=float(w[disagree].sum() / total),
        best_gain_weight_frac=float(w[best_gain].sum() / total),
        rock_gain_weight_frac=float(w[rock_gain].sum() / total),
        net_gap_pct=float(100 * (w[best_gain].sum() - w[rock_gain].sum()) / total),
        disagree_rate_by_weight=by_w,
        disagree_weight_frac_by_degree=by_deg,
        spearman=float(spearmanr(rock_rank, order).correlation),
        kendall_tau=float(kendalltau(rock_rank, order).correlation),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — self-diagnostic: loss-descent + near-ties (connectome)
# ──────────────────────────────────────────────────────────────────────────────
def step3_self(g, dev, quick: bool) -> dict:
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    w = np.asarray(g.weight, dtype=np.int64)
    total = int(w.sum())
    h02 = np.load(_latest("*H02-connectome-s42-confirm*_positions.npy")).astype(np.float64)
    delta = h02[tgt] - h02[src]
    fb = delta <= 0
    absd = np.abs(delta)
    near = {f"eps_{eps}": int(w[fb & (absd < eps)].sum())
            for eps in [1e-6, 1e-4, 1e-2, 1e-1, 0.5, 1.0]}
    out = dict(exact_ties=int((delta == 0).sum()),
               feedback_weight_frac=float(w[fb].sum() / total),
               near_tie_recoverable_weight=near,
               near_tie_recoverable_frac_eps1=float(w[fb & (absd < 1.0)].sum() / total))
    # loss-descent at stop: short fresh baseline run, inspect neg_loss tail slope
    if not quick:
        cfg = RocketConfig(epochs=EPOCHS["connectome"])
        r = run_rocket(g, cfg, seed=42, device=dev)
        h = r.history
        tail = h.iloc[-len(h) // 10:]   # last 10% of logged points
        out["neg_loss_final"] = float(h["neg_loss"].iloc[-1])
        out["neg_loss_tail_slope_per_logstep"] = float(
            np.polyfit(np.arange(len(tail)), tail["neg_loss"].values, 1)[0])
        out["best_pct_tail_slope"] = float(
            np.polyfit(np.arange(len(tail)), tail["best_pct"].values, 1)[0])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — init→plateau curve (both datasets, leakage-safe inits)
# ──────────────────────────────────────────────────────────────────────────────
def step4_init_plateau(quick: bool) -> dict:
    dev = _device()
    res = {}
    inits = ["random", "uniform", "degree_diff", "degree_abs", "greedy"]
    if quick:
        inits = ["random", "degree_abs", "greedy"]
    for ds in ["mouse", "connectome"]:
        g = io.load_dataset(ds)
        cfg = RocketConfig(epochs=EPOCHS[ds])
        rows = []
        for mode in inits:
            # Build inits on CPU as float32 then move (avoids an MPS float64-construction
            # quirk in make_init_positions for connectome 'degree_abs'); run_rocket recasts.
            if mode == "greedy":
                from mfas.experiments.H02 import _init_positions_from_order, greedy_fas_order
                init = _init_positions_from_order(greedy_fas_order(g), "cpu")
            else:
                init = make_init_positions(mode, g, seed=42, device="cpu")
            init = init.to(torch.float32).to(dev)
            init_pct = _score_pct(init.detach().cpu().numpy(), g)
            t = time.time()
            r = run_rocket(g, cfg, seed=42, device=dev, init_positions=init)
            rows.append(dict(init=mode, init_pct=init_pct, plateau_pct=r.best_pct,
                            wall_s=round(time.time() - t, 1)))
        res[ds] = rows
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", default="0,1,2,3,4")
    args = ap.parse_args()
    steps = set(args.steps.split(","))
    dev = _device()
    p = OUT / "diagnosis.json"
    g = io.load_connectome()
    order, best_pct = gap.load_best_solution(g)
    summary = {"device": str(dev), "best_solution_pct": best_pct,
               "rocket_only_best_pct": 82.9273, "crane_reference_pct": 84.60}

    def _flush():
        p.write_text(json.dumps(summary, indent=2))

    print(f"[step0] best_solution = {best_pct:.4f}%  (anchor)", flush=True)
    _flush()
    if "1" in steps:
        summary["step1_decisive"] = step1_decisive(g, order, dev, args.quick)
        print("[step1] decisive surrogate done", flush=True); _flush()
    if "2" in steps:
        summary["step2_decomp"] = step2_decomp(g, order)
        print("[step2] gap decomposition done", flush=True); _flush()
    if "3" in steps:
        summary["step3_self"] = step3_self(g, dev, args.quick)
        print("[step3] self-diagnostic done", flush=True); _flush()
    if "4" in steps:
        summary["step4_init_plateau"] = step4_init_plateau(args.quick)
        print("[step4] init->plateau done", flush=True); _flush()
    # synthetic generalization check of the decisive comparison
    gs, planted, planted_pct = gap.make_synthetic_graph(n=400, seed=0)
    syn = {"planted_pct": planted_pct, "betas": BETAS, "static": []}
    for beta in BETAS:
        _, bm = gap.optimize_spacing(planted, gs, beta=beta, iters=300)
        rr = run_rocket(gs, RocketConfig(epochs=4000), seed=42, device=_device())
        syn["static"].append(dict(beta=beta, planted_max_surrogate=bm,
                                  rocket_converged_surrogate=gap.surrogate_objective(
                                      rr.best_positions.astype(np.float64), gs, beta),
                                  rocket_pct=rr.best_pct))
    summary["synthetic_check"] = syn
    _flush()
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
