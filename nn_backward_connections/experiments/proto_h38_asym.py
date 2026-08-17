"""H38 prototype gate — ASYMMETRIC (one-sided) surrogate: flat above, tanh below.

The idea under test (structurally different from everything in Q04/H37, which were all
odd-symmetric): give the surrogate a CONSTANT top branch and a tanh bottom branch, so that
already-feedforward edges receive NO gradient and the entire gradient budget is spent pulling
FEEDBACK edges toward correctness.

    naive:   g(z) = 1                      for z >= 0
             g(z) = 1 + tanh(z/T)          for z <  0

Why it looked extremely promising (Q05 theory gate, `q05_asymmetric_surrogates.json`)
--------------------------------------------------------------------------------------
* It is the FIRST shape measured that escapes Q01's small-scale degeneracy: at beta*std -> 0
  it ranks `best > rocket > imbalance_sort > random` (correct), where every odd-symmetric
  shape ranks the imbalance sort first. The reason is structural: the first-order term
  `sum_{e: Delta_e<0} w_e Delta_e` runs over the VIOLATED subset, which is order-dependent, so
  it does not telescope into the node-level imbalance objective `<c,P>`.
* Alignment ratio at Rocket's operating point is **+1.48 … +2.00** (vs the sigmoid's -0.591),
  and there is NO crossover anywhere in beta*std in [1e-2, 1e6] — it never prefers the worse
  order at any scale.
* The mirror arm (flat on the FEEDBACK side) is the clean control: alignment -1.30 … -3.72.

THE FATAL FLAW, derived before running anything
-----------------------------------------------
`g(0) = 1`, so the CONSTANT configuration `P = const` (every Delta = 0) gives

    F = sum_e w_hat_e * g(0) = sum_e w_hat_e = W_hat

which is the **global maximum** of F, strictly above every ordering (any real order has some
feedback edge with Delta < 0, hence g < 1). And the dynamics flow into it: a violated edge pulls
its two endpoints TOWARD each other, which shrinks |Delta|. So the naive shape has a trivial
degenerate optimum at total collapse, whose discrete score is 0% (ties are not feedforward under
the strict `>` oracle). ARM `asym_naive` is included precisely to demonstrate this, not because
it is expected to work.

The minimal fix: a MARGIN
-------------------------
    g(z) = 1                        for z >= m
    g(z) = 1 + tanh((z-m)/T)        for z <  m           (m > 0)

Now `g(0) = 1 + tanh(-m/T) < 1`, so collapse is no longer optimal, while the two properties the
idea was about are preserved: comfortably-correct edges (z >= m) still get exactly zero
gradient, and the bottom branch never flattens, so deeply-violated edges keep being pulled.
This is a one-sided (hinge-like) loss with a smooth tanh ramp instead of a linear one.

PRE-REGISTERED predictions (recorded BEFORE the run)
----------------------------------------------------
  Q1  asym_naive COLLAPSES: final position std << the sigmoid's, and discrete score far below
      the sigmoid (this is the derivation above, tested as a falsifiable claim, not assumed).
  Q2  the margin arms do NOT collapse (final position std of the same order as the sigmoid's).
  Q3  at least one margin arm beats the sigmoid on the hard synthetic  <-- THE hypothesis.
  Q4  asym_flat_neg (mirror control) < sigmoid.

Note the standing lesson from the H37 cycle: static alignment ANTI-correlated with achieved
score there, so the excellent Q05 alignment numbers are NOT evidence of a win. This gate is.

Leakage: no oracle read inside any surrogate; the frozen scorer is used exactly as the baseline
uses it (best-by-oracle tracking). Writes to `experiments/outputs/` only.

Run (env `allen`, from the repo root):
    PYTHONPATH=src python experiments/proto_h38_asym.py
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from mfas.analysis.gap import make_hard_synthetic_graph
from mfas.baseline.rocket import RocketConfig, make_beta_schedule
from mfas.io import load_dataset
from mfas.metrics import pct, score_from_positions

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "experiments" / "outputs"

CONFIG = {
    "seeds": [42, 123, 999],
    "epochs": {"mouse": 5_000, "hard_synthetic": 8_000},
    "synthetic": {"n": 400, "avg_out": 10, "feedback_frac": 0.65, "n_clusters": 8,
                  "intra_cycle_frac": 0.55, "weight_alpha": 2.0, "seed": 0},
    "predictions": {
        "Q1": "asym_naive collapses (final std << sigmoid, score far below)",
        "Q2": "margin arms do not collapse",
        "Q3": "at least one margin arm > sigmoid on hard_synthetic (THE hypothesis)",
        "Q4": "asym_flat_neg (mirror control) < sigmoid",
    },
}


def _asym(z: torch.Tensor, m: float, T: float) -> torch.Tensor:
    """g(z) = 1 for z >= m ; 1 + tanh((z-m)/T) for z < m. Monotone, bounded, C^0 at m."""
    below = 1.0 + torch.tanh((z - m) / T)
    return torch.where(z >= m, torch.ones_like(z), below)


def _asym_neg(z: torch.Tensor, T: float) -> torch.Tensor:
    """MIRROR control: g(z) = tanh(z/T) for z > 0 ; 0 for z <= 0."""
    return torch.where(z <= 0, torch.zeros_like(z), torch.tanh(z.clamp(min=0.0) / T))


def arms():
    return {
        "sigmoid": lambda z: torch.sigmoid(z),
        "asym_naive_T1.5": lambda z: _asym(z, 0.0, 1.5),
        "asym_m0.5_T1.5": lambda z: _asym(z, 0.5, 1.5),
        "asym_m2_T1.5": lambda z: _asym(z, 2.0, 1.5),
        "asym_m5_T1.5": lambda z: _asym(z, 5.0, 1.5),
        "asym_m2_T3": lambda z: _asym(z, 2.0, 3.0),
        "asym_m5_T3": lambda z: _asym(z, 5.0, 3.0),
        "asym_flat_neg_T1.5": lambda z: _asym_neg(z, 1.5),
    }


def run_arm(g, cfg: RocketConfig, seed: int, shape_fn) -> dict:
    """``baseline.rocket.run_rocket`` with the single change sigmoid -> shape_fn."""
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = g.n_nodes
    src_t = torch.tensor(np.asarray(g.src), dtype=torch.long, device=device)
    tgt_t = torch.tensor(np.asarray(g.tgt), dtype=torch.long, device=device)
    w_np = np.asarray(g.weight)
    nw_t = torch.tensor((w_np / float(w_np.max())).astype(np.float32), device=device)
    src_np, tgt_np = np.asarray(g.src), np.asarray(g.tgt)

    def score(p):
        return score_from_positions(p.detach().cpu().numpy(), src_np, tgt_np, g.weight)

    positions = torch.nn.Parameter(torch.randn(n, device=device))
    optimizer = optim.Adam([positions], lr=cfg.lr)
    milestone = int(cfg.epochs * cfg.lr_decay_start)
    sched = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                  total_iters=milestone),
                    optim.lr_scheduler.ExponentialLR(
                        optimizer,
                        gamma=cfg.lr_end_factor ** (1.0 / max(cfg.epochs - milestone, 1)))],
        milestones=[milestone])
    betas = make_beta_schedule(cfg.epochs, cfg.cycles)

    best = score(positions)
    t0 = time.time()
    for i in range(cfg.epochs):
        beta = float(betas[i])
        optimizer.zero_grad()
        delta = positions[tgt_t] - positions[src_t]
        loss = -(shape_fn(beta * delta) * nw_t).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([positions], cfg.grad_clip)
        optimizer.step()
        sched.step()
        if i % cfg.log_interval == 0 or i == cfg.epochs - 1:
            s = score(positions)
            if s > best:
                best = s
    return {"best_pct": pct(best, g.total_weight),
            "final_std": float(positions.detach().std()),
            "final_pct": pct(score(positions), g.total_weight),
            "wall_s": time.time() - t0}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_ROOT, text=True).strip()
    except Exception:                                     # pragma: no cover
        return "unknown"


def main() -> None:
    report = {"config": CONFIG, "git_commit": git_commit()}

    # ── the collapse degeneracy, checked as an explicit numeric claim ─────────
    gm = load_dataset("mouse")
    w = np.asarray(gm.weight, dtype=np.float64)
    nw = w / w.max()
    zc = torch.zeros(gm.n_edges, dtype=torch.float64)
    coll = {}
    for name, fn in arms().items():
        F_collapse = float((fn(zc).numpy() * nw).sum())
        coll[name] = {"F_at_total_collapse": F_collapse, "W_hat": float(nw.sum()),
                      "collapse_is_global_max": bool(
                          abs(F_collapse - float(nw.sum())) < 1e-9)}
    report["collapse_check_mouse"] = coll
    print("COLLAPSE CHECK on mouse — F at P=const vs the surrogate's upper bound W_hat:")
    for k, v in coll.items():
        flag = "  <-- GLOBAL MAX (degenerate)" if v["collapse_is_global_max"] else ""
        print(f"  {k:>20}: F(collapse)={v['F_at_total_collapse']:10.3f}  "
              f"W_hat={v['W_hat']:10.3f}{flag}")

    graphs = {"mouse": (gm, None)}
    sc = CONFIG["synthetic"]
    gh, _ref_order, ref_pct = make_hard_synthetic_graph(
        n=sc["n"], avg_out=sc["avg_out"], feedback_frac=sc["feedback_frac"],
        n_clusters=sc["n_clusters"], intra_cycle_frac=sc["intra_cycle_frac"],
        weight_alpha=sc["weight_alpha"], seed=sc["seed"])
    graphs["hard_synthetic"] = (gh, ref_pct)

    results = {}
    for gname, (g, ref) in graphs.items():
        epochs = CONFIG["epochs"][gname]
        results[gname] = {"reference_pct_diagnostic_only": ref, "arms": {}}
        print(f"\n=== {gname} (n={g.n_nodes}, {epochs} epochs, 3 seeds) ===")
        print(f"{'arm':>20} {'mean best':>10} {'std':>8} {'final pos std':>14} "
              f"{'delta vs sigmoid':>17}")
        base = None
        for aname, fn in arms().items():
            pcts, stds = [], []
            for seed in CONFIG["seeds"]:
                r = run_arm(g, RocketConfig(epochs=epochs), seed, fn)
                pcts.append(r["best_pct"])
                stds.append(r["final_std"])
            mean = float(np.mean(pcts))
            if base is None:
                base = mean
            results[gname]["arms"][aname] = {
                "per_seed_pct": pcts, "mean_pct": mean,
                "std_pct": float(np.std(pcts, ddof=1)),
                "mean_final_position_std": float(np.mean(stds)),
                "delta_vs_sigmoid": mean - base}
            print(f"{aname:>20} {mean:10.4f} {np.std(pcts, ddof=1):8.4f} "
                  f"{np.mean(stds):14.4f} {mean - base:+17.4f}")

    report["results"] = results

    # ── pre-registered checks ────────────────────────────────────────────────
    checks = {}
    for gname in graphs:
        a = results[gname]["arms"]
        sig_std = a["sigmoid"]["mean_final_position_std"]
        margin_arms = [k for k in a if k.startswith("asym_m")]
        checks[gname] = {
            "Q1_naive_collapses": bool(
                a["asym_naive_T1.5"]["mean_final_position_std"] < 0.1 * sig_std
                and a["asym_naive_T1.5"]["mean_pct"] < a["sigmoid"]["mean_pct"]),
            "Q2_margin_arms_do_not_collapse": bool(all(
                a[k]["mean_final_position_std"] > 0.1 * sig_std for k in margin_arms)),
            "Q3_a_margin_arm_beats_sigmoid": bool(any(
                a[k]["mean_pct"] > a["sigmoid"]["mean_pct"] for k in margin_arms)),
            "Q4_mirror_control_loses": bool(
                a["asym_flat_neg_T1.5"]["mean_pct"] < a["sigmoid"]["mean_pct"]),
            "best_margin_arm": max(margin_arms, key=lambda k: a[k]["mean_pct"]),
        }
    report["prediction_check"] = checks
    print("\nPRE-REGISTERED PREDICTION CHECK:")
    for gname, v in checks.items():
        print(f"  {gname}: best margin arm = {v['best_margin_arm']}")
        for k in ("Q1_naive_collapses", "Q2_margin_arms_do_not_collapse",
                  "Q3_a_margin_arm_beats_sigmoid", "Q4_mirror_control_loses"):
            print(f"     {k:>34}: {'HOLDS' if v[k] else 'FAILS'}")

    _OUT.mkdir(parents=True, exist_ok=True)
    dest = _OUT / "proto_h38_asym.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
