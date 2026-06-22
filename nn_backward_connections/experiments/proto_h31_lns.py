"""In-repo prototype gate for H31 (ILS/LNS on the H30 sift), using PRODUCTION kernels.

Cheapest-first DECISION GATE (run BEFORE any connectome/microns compute): at MATCHED
wall-clock, compare H31's ``ils_lns`` against single-pass H30 ``sift`` on the gap-bearing
hard synthetic (``mfas.analysis.gap.make_hard_synthetic_graph``, 3 seeds) and mouse
(3 seeds), CPU, ``torch.set_num_threads(2)``.

Matched wall-clock framing
--------------------------
For each instance we first sift the H02-Rocket order to its fixed point and record that
time ``t_sift`` and score (the H30 result — single-pass sift cannot improve past its fixed
point, so any *extra* wall spent "on sift" yields the same score). H31 then spends a
refinement budget ``T = matched_frac * t_rocket`` (>= t_sift) as sift + LNS-for-remainder.
The comparison is therefore: at the SAME refinement wall, does ``sift + LNS`` beat
``sift alone``? Reported as ``lns_pct - sift_pct`` per seed, plus the synthetic
reference-order gap for context.

GATE: if ils_lns does NOT beat single-pass sift beyond noise on the SYNTHETIC at matched
wall-clock, STOP (ship H30 alone). Writes ONLY experiments/outputs/proto_h31_lns.json.

Leakage-safety: every move is chosen from input edge weights + current ranks only; the
frozen oracle is used solely to score / best-by-oracle accept whole candidate vectors.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402
torch.set_num_threads(2)

from mfas import io  # noqa: E402
from mfas.analysis import gap as gapmod  # noqa: E402
from mfas.baseline.rocket import RocketConfig, run_rocket  # noqa: E402
from mfas.experiments.H02 import _init_positions_from_order, greedy_fas_order  # noqa: E402
from mfas.metrics import pct, score_from_order  # noqa: E402
from mfas.refine import ils_lns, sift  # noqa: E402

CPU = torch.device("cpu")


def _rocket_rank(g, seed, init_positions, epochs):
    t0 = time.time()
    cfg = RocketConfig(epochs=epochs)
    res = run_rocket(g, cfg, seed=seed, device=CPU, init_positions=init_positions)
    rank = np.argsort(np.argsort(res.best_positions, kind="stable"),
                      kind="stable").astype(np.int64)
    return rank, res.best_pct, time.time() - t0


def _run_one(g, seed, epochs, sift_sweeps, lns_cfg, matched_frac, ref_pct=None):
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)

    greedy = greedy_fas_order(g)
    greedy_pct = pct(score_from_order(greedy, src, tgt, g.weight), g.total_weight)

    h02_init = _init_positions_from_order(greedy, CPU)
    h02_rank, h02_pct, t_rocket = _rocket_rank(g, seed, h02_init, epochs)

    # ── H30: sift to fixed point ────────────────────────────────────────────────
    t0 = time.time()
    sift_rank, sift_score, slog = sift(g, h02_rank, max_sweeps=sift_sweeps)
    t_sift = time.time() - t0
    sift_pct = pct(sift_score, g.total_weight)

    # ── H31: LNS for the remainder of a matched refinement budget ───────────────
    refine_budget = matched_frac * t_rocket
    lns_budget = max(0.2, refine_budget - t_sift)  # ensure at least a little LNS time
    t0 = time.time()
    lns_rank, lns_score, rlog = ils_lns(
        g, sift_rank, time_budget_s=lns_budget, k=lns_cfg["k"], mode=lns_cfg["mode"],
        inner_sweeps=lns_cfg["inner"], sequential_victims=lns_cfg["sequential"],
        seed=seed)
    t_lns = time.time() - t0
    lns_pct = pct(lns_score, g.total_weight)

    row = dict(
        seed=seed, n=int(g.n_nodes), m=int(g.n_edges),
        greedy_pct=greedy_pct, h02_pct=h02_pct,
        sift_pct=sift_pct, lns_pct=lns_pct,
        lns_minus_sift=lns_pct - sift_pct,
        t_rocket_s=round(t_rocket, 2), t_sift_s=round(t_sift, 2), t_lns_s=round(t_lns, 2),
        refine_budget_s=round(refine_budget, 2),
        n_sift_sweeps=len(slog), n_lns_rounds=len(rlog),
        n_lns_accepted=int(sum(1 for r in rlog if r["accepted"])),
        lns_cfg=lns_cfg,
    )
    if ref_pct is not None:
        row["reference_pct"] = ref_pct
        row["gap_h02_to_ref"] = ref_pct - h02_pct
        row["gap_sift_to_ref"] = ref_pct - sift_pct
    return row


def main():
    out = {"synthetic": [], "mouse": []}

    # Synthetic: small (n=400), so allow more victims / sequential moves and a longer
    # matched refinement budget relative to the (very fast) Rocket run.
    syn_cfg = dict(k=20, mode="ruin", inner=2, sequential=True)
    for seed in (42, 123, 999):
        g, _ref, ref_pct = gapmod.make_hard_synthetic_graph(seed=seed)
        row = _run_one(g, seed, epochs=4000, sift_sweeps=20, lns_cfg=syn_cfg,
                       matched_frac=2.0, ref_pct=ref_pct)
        out["synthetic"].append(row)
        print(f"[syn s{seed}] ref={row['reference_pct']:.3f} h02={row['h02_pct']:.3f} "
              f"sift={row['sift_pct']:.3f} LNS={row['lns_pct']:.3f} "
              f"(LNS-sift={row['lns_minus_sift']:+.4f}) "
              f"[rounds={row['n_lns_rounds']} acc={row['n_lns_accepted']} "
              f"t_sift={row['t_sift_s']}s t_lns={row['t_lns_s']}s]")

    mouse_cfg = dict(k=8, mode="ruin", inner=2, sequential=True)
    g_mouse = io.load_dataset("mouse")
    for seed in (42, 123, 999):
        row = _run_one(g_mouse, seed, epochs=5000, sift_sweeps=30, lns_cfg=mouse_cfg,
                       matched_frac=1.5)
        out["mouse"].append(row)
        print(f"[mouse s{seed}] h02={row['h02_pct']:.3f} sift={row['sift_pct']:.3f} "
              f"LNS={row['lns_pct']:.3f} (LNS-sift={row['lns_minus_sift']:+.4f}) "
              f"[rounds={row['n_lns_rounds']} acc={row['n_lns_accepted']} "
              f"t_sift={row['t_sift_s']}s t_lns={row['t_lns_s']}s]")

    # Summary deltas (mean LNS-minus-sift per dataset).
    for ds in ("synthetic", "mouse"):
        d = np.array([r["lns_minus_sift"] for r in out[ds]])
        out[f"{ds}_summary"] = dict(
            mean_lns_minus_sift=float(d.mean()),
            std_lns_minus_sift=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            min=float(d.min()), max=float(d.max()))
        print(f"== {ds}: mean LNS-sift = {d.mean():+.4f} pp "
              f"(min {d.min():+.4f}, max {d.max():+.4f})")

    out_dir = _ROOT / "experiments" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proto_h31_lns.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
