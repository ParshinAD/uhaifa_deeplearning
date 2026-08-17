"""H37 prototype gate — does a NARROW-CORE + POLYNOMIAL-TAIL surrogate beat the sigmoid?

Cheap CPU gate on the two standard proxies (mouse + the hard synthetic that carries a
verified optimization gap), BEFORE spending any large-graph compute. Same pattern as the
H32/H33/H34 prototype gates.

What the theory gate (Q04, `experiments/outputs/q04_surrogate_tails.json`) established
---------------------------------------------------------------------------------------
Writing ``F_g(P) = sum_e w_hat_e g(beta*Delta_e)`` and measuring on the fly connectome:

  * ``(tanh(z/2)+1)/2 == sigmoid(z)`` to 2.2e-16, and a rescaled tanh reproduces the
    sigmoid's crossover EXACTLY in beta*std units -> the literal "use tanh" reading is a
    no-op on an already-closed axis (A-SCALE == H03, killed).
  * Every shape aligns (ranks the better order above Rocket's) only above
    ``beta*std ~ 200 * width(g)``, where ``width(g)`` is the z at which g reaches 0.9. So
    ALIGNMENT is bought by narrowing the core — that is the beta axis, nothing new.
  * The NEW degree of freedom is what the narrow core COSTS. At matched width 0.495:

        shape                 alignment ratio @ beta*std=148   frac nodes with grad==0
        sigmoid (width 2.20)            -0.591                        16.1%
        sigmoid_sharp (width 0.495)     +0.132                        42.5%     <- control
        poly_q4       (width 0.495)     +0.173                         0.006%   <- arm

    i.e. the exponential tail can only buy alignment by freezing 42.5% of the nodes,
    while the polynomial tail buys the SAME alignment with essentially every node still
    mobile. That is the one claim no rescaling of beta can reproduce.
  * Hard flat tails (the "constant top and bottom" reading) are the HARMFUL half:
    H11's clamp freezes 65.8% of nodes and has the worst alignment (-1.063) — a
    mechanistic explanation of the already-logged H11 kill.

PRE-REGISTERED predictions (falsifiers — recorded BEFORE the run, checked after)
--------------------------------------------------------------------------------
  P1  poly_q4      >  sigmoid          (the hypothesis: aligned AND mobile)
  P2  poly_q1      <  sigmoid          (heavy tail WITHOUT a narrow core is misaligned,
                                        alignment ratio -0.79 -> must NOT help)
  P3  sigmoid_sharp ~<= sigmoid        (aligned but 42.5% frozen: the H03/A-SCALE control;
                                        if THIS wins, the effect is sharpening, not tails)
  P4  h11_clip_M5  <  sigmoid          (reproduces the logged H11 kill as a sanity anchor)

If P1 fails -> KILL at the prototype gate, no large-graph compute.
If P1 and P3 BOTH win -> the effect is confounded with sharpening; report as such.

Leakage: no oracle score is ever read inside a surrogate; the frozen `mfas.metrics` scorer
is used exactly as the baseline uses it (best-by-oracle tracking of whole position vectors).
The hard synthetic's reference order is a DIAGNOSTIC comparator only (built by
`mfas.analysis.gap`), never read by any arm. Writes to `experiments/outputs/` only.

Run (env `allen`, from the repo root):
    PYTHONPATH=src python experiments/proto_h37_tails.py
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
    # poly_q4's transition width relative to the sigmoid's (2.1973 / 0.4954).
    "sharp_factor": 4.4361,
    "h11_margin": 5.0,
    "synthetic": {"n": 400, "avg_out": 10, "feedback_frac": 0.65, "n_clusters": 8,
                  "intra_cycle_frac": 0.55, "weight_alpha": 2.0, "seed": 0},
    "predictions": {
        "P1": "poly_q4 > sigmoid",
        "P2": "poly_q1 < sigmoid",
        "P3": "sigmoid_sharp <= sigmoid (H03/A-SCALE control)",
        "P4": "h11_clip_M5 < sigmoid (H11 anchor)",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# The surrogate shapes (torch, differentiable)
# ──────────────────────────────────────────────────────────────────────────────
def _poly(z: torch.Tensor, q: float) -> torch.Tensor:
    """g_q(z) = 1/2 + 1/2 sign(z) (1 - (1+|z|)^-q).

    Monotone, bounded in (0,1), g(0)=1/2, C^1 at 0 with slope q/2, tail 1-g ~ z^-q.
    Autograd note: d/dz = 1/2 * sign(z) * q(1+|z|)^-(q+1) * sign(z) = q/2 (1+|z|)^-(q+1),
    because d|z|/dz = sign(z) and sign(z)^2 = 1 — verified numerically in `check_grads`.
    """
    a = z.abs()
    return 0.5 + 0.5 * torch.sign(z) * (1.0 - (1.0 + a) ** (-q))


def shape_fns(cfg):
    sf = cfg["sharp_factor"]
    m = cfg["h11_margin"]
    return {
        "sigmoid": lambda z: torch.sigmoid(z),
        "sigmoid_sharp": lambda z: torch.sigmoid(sf * z),
        "poly_q4": lambda z: _poly(z, 4.0),
        "poly_q2": lambda z: _poly(z, 2.0),
        "poly_q1": lambda z: _poly(z, 1.0),
        # WIDTH-MATCHED polynomial arms: z rescaled so the CORE width equals the
        # sigmoid's (2.1973). These isolate the TAIL axis alone — same core sharpness as
        # baseline, polynomial instead of exponential decay. Without them, poly_q4's
        # 4.4x narrower core confounds "shape" with "effective beta" (the H03 axis).
        "poly_q4_wmatch": lambda z: _poly(z / sf, 4.0),
        "poly_q1_wmatch": lambda z: _poly(z * (4.0 / 2.1973), 1.0),
        "h11_clip_M5": lambda z: torch.clamp(0.5 + z / (2.0 * m), 0.0, 1.0),
    }


def check_grads(cfg) -> dict:
    """Numerically verify every shape's autograd derivative against a central difference."""
    out = {}
    zs = torch.tensor([-50.0, -7.3, -1.0, -0.2, 0.35, 1.0, 4.0, 33.0], dtype=torch.float64)
    for name, fn in shape_fns(cfg).items():
        z = zs.clone().requires_grad_(True)
        fn(z).sum().backward()
        ana = z.grad.detach().numpy().copy()
        h = 1e-6
        num = ((fn(zs + h) - fn(zs - h)) / (2 * h)).detach().numpy()
        # the H11 clamp is non-differentiable at |z|=M; compare where both are defined
        err = float(np.max(np.abs(ana - num)))
        out[name] = {"max_abs_grad_err_vs_finite_diff": err,
                     "analytic": ana.tolist(), "numeric": num.tolist()}
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Rocket with a swappable per-edge shape (everything else verbatim from baseline)
# ──────────────────────────────────────────────────────────────────────────────
def run_rocket_shape(g, cfg: RocketConfig, seed: int, shape_fn) -> dict:
    """``baseline.rocket.run_rocket`` with the SINGLE change ``sigmoid(z) -> shape_fn(z)``."""
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = g.n_nodes
    src_t = torch.tensor(np.asarray(g.src), dtype=torch.long, device=device)
    tgt_t = torch.tensor(np.asarray(g.tgt), dtype=torch.long, device=device)
    w_np = np.asarray(g.weight)
    nw_t = torch.tensor((w_np / float(w_np.max())).astype(np.float32), device=device)
    src_np, tgt_np = np.asarray(g.src), np.asarray(g.tgt)

    def discrete_score(p):
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

    best_score = discrete_score(positions)
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
            s = discrete_score(positions)
            if s > best_score:
                best_score = s
    return {"best_pct": pct(best_score, g.total_weight),
            "final_std": float(positions.detach().std()),
            "wall_s": time.time() - t0, "epochs": cfg.epochs}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_ROOT, text=True).strip()
    except Exception:                                     # pragma: no cover
        return "unknown"


def main() -> None:
    report = {"config": CONFIG, "git_commit": git_commit()}

    report["grad_check"] = check_grads(CONFIG)
    print("AUTOGRAD CHECK (max |analytic - central difference|):")
    for k, v in report["grad_check"].items():
        print(f"  {k:>16}: {v['max_abs_grad_err_vs_finite_diff']:.3e}")

    # ── the two proxies ──────────────────────────────────────────────────────
    graphs = {}
    graphs["mouse"] = (load_dataset("mouse"), None)
    sc = CONFIG["synthetic"]
    gh, ref_order, ref_pct = make_hard_synthetic_graph(
        n=sc["n"], avg_out=sc["avg_out"], feedback_frac=sc["feedback_frac"],
        n_clusters=sc["n_clusters"], intra_cycle_frac=sc["intra_cycle_frac"],
        weight_alpha=sc["weight_alpha"], seed=sc["seed"])
    graphs["hard_synthetic"] = (gh, ref_pct)
    print(f"\nhard synthetic: n={gh.n_nodes} edges={gh.n_edges} "
          f"reference(diagnostic only)={ref_pct:.4f}%")

    fns = shape_fns(CONFIG)
    results = {}
    for gname, (g, ref) in graphs.items():
        epochs = CONFIG["epochs"][gname]
        results[gname] = {"reference_pct_diagnostic_only": ref, "arms": {}}
        print(f"\n=== {gname} (n={g.n_nodes}, {epochs} epochs, "
              f"{len(CONFIG['seeds'])} seeds) ===")
        print(f"{'arm':>16} {'mean':>10} {'std':>8} {'per-seed':>34} {'final std':>10}")
        for aname, fn in fns.items():
            pcts, stds = [], []
            for seed in CONFIG["seeds"]:
                cfg = RocketConfig(epochs=epochs)
                r = run_rocket_shape(g, cfg, seed, fn)
                pcts.append(r["best_pct"])
                stds.append(r["final_std"])
            results[gname]["arms"][aname] = {
                "per_seed_pct": pcts, "mean_pct": float(np.mean(pcts)),
                "std_pct": float(np.std(pcts, ddof=1)) if len(pcts) > 1 else 0.0,
                "mean_final_position_std": float(np.mean(stds))}
            ps = " ".join(f"{p:10.4f}" for p in pcts)
            print(f"{aname:>16} {np.mean(pcts):10.4f} "
                  f"{(np.std(pcts, ddof=1) if len(pcts) > 1 else 0.0):8.4f} {ps} "
                  f"{np.mean(stds):10.2f}")

    # ── pre-registered predictions ───────────────────────────────────────────
    verdict = {}
    for gname in graphs:
        base = results[gname]["arms"]["sigmoid"]["mean_pct"]
        verdict[gname] = {
            "sigmoid_mean": base,
            "P1_poly_q4_gt_sigmoid": bool(
                results[gname]["arms"]["poly_q4"]["mean_pct"] > base),
            "P2_poly_q1_lt_sigmoid": bool(
                results[gname]["arms"]["poly_q1"]["mean_pct"] < base),
            "P3_sigmoid_sharp_le_sigmoid": bool(
                results[gname]["arms"]["sigmoid_sharp"]["mean_pct"] <= base),
            "P4_h11_lt_sigmoid": bool(
                results[gname]["arms"]["h11_clip_M5"]["mean_pct"] < base),
            "deltas_vs_sigmoid": {a: results[gname]["arms"][a]["mean_pct"] - base
                                  for a in results[gname]["arms"]},
        }
    report["results"] = results
    report["prediction_check"] = verdict

    print("\nPRE-REGISTERED PREDICTION CHECK (delta vs sigmoid, pp):")
    for gname, v in verdict.items():
        print(f"  {gname}:")
        for a, d in v["deltas_vs_sigmoid"].items():
            print(f"     {a:>16}: {d:+8.4f}")
        for k in ("P1_poly_q4_gt_sigmoid", "P2_poly_q1_lt_sigmoid",
                  "P3_sigmoid_sharp_le_sigmoid", "P4_h11_lt_sigmoid"):
            print(f"     {k:>32}: {'HOLDS' if v[k] else 'FAILS'}")

    _OUT.mkdir(parents=True, exist_ok=True)
    dest = _OUT / "proto_h37_tails.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
