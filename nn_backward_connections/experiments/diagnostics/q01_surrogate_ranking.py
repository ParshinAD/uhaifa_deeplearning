"""Q01 — WHY the surrogate ranks a worse order above a better one, as a function of scale.

Answers two questions with numbers instead of intuition:

1. **At small scale (beta*std -> 0), why is a monotone surrogate not monotone in the true
   score?**  Because the sum TELESCOPES. With ``sigma(x) = 1/2 + x/4 + O(x^3)``,

       F(P) = sum_e w_hat_e * sigma(beta*D_e)
            = W_hat/2 + (beta/4) * sum_e w_hat_e * D_e + O((beta*std)^3)
            = W_hat/2 - (beta/4) * <c, P>              , c_k = out_w_hat(k) - in_w_hat(k)

   ``sum_e w_hat_e * D_e`` collapses to a NODE-level quantity: it no longer knows which node
   precedes which, only where each node sits, weighted by its own imbalance. So in the small-
   scale limit the surrogate stops being a relaxation of the feedback-arc-set objective and
   becomes a *different problem* — "sort the nodes by imbalance" — whose optimum (by the
   rearrangement inequality: largest ``c`` to the smallest position) is the imbalance sort, NOT
   the feedforward optimum. Per-edge monotonicity is intact throughout; what fails is that
   ranking two whole ORDERS is not the same as improving one edge at a time.

2. **At Rocket's operating scale, why does the better order still lose?**  Decompose

       F(order) = ceiling(order) - smoothing_loss(order),
       ceiling(order) = sum_e w_hat_e * 1[D_e > 0] = discrete_score / max_w

   The ceiling is the true objective in w_hat units; ``smoothing_loss`` is what the surrogate
   discounts for edges that are only NARROWLY correct (sigma ~ 0.5 instead of 1). A better
   order can be beaten if it pays more smoothing loss than the true advantage it holds.

This script measures both on the fly connectome for four orders: the near-optimal reference,
Rocket's converged order, the imbalance sort, and a random control.

Leakage: ``data/best_solution`` is read ONLY through ``mfas.analysis.gap.load_best_solution``.
Diagnostic only — nothing here feeds any variant's init, loss or move choice. Writes to
``experiments/outputs/`` only, never ``results/``.

Run (env `allen`, from the repo root):
    PYTHONPATH=src python experiments/diagnostics/q01_surrogate_ranking.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from mfas.analysis.gap import load_best_solution
from mfas.io import load_dataset
from mfas.metrics import pct, score_from_order

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "experiments" / "outputs"

CONFIG = {
    "dataset": "connectome",
    "rocket_positions": "results/rocket_best_positions.npy",
    "seed": 42,
    # Rocket's terminal sharpness; the cyclic schedule spans [0.05, 1.05].
    "beta_main": 1.05,
    "betas": [0.05, 0.30, 1.05],
    # Scale grid: fine enough to bracket the crossover without grid-snapping the answer
    # (the previous "std=459" number was a grid node of a 40-point logspace).
    "std_grid": list(np.round(np.logspace(np.log10(0.2), np.log10(1e5), 61), 6)),
    # Rocket's own operating scale, measured from its converged positions.
    "operating_std": 141.0,
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_ROOT, text=True).strip()
    except Exception:                                     # pragma: no cover
        return "unknown"


def imbalance(g) -> np.ndarray:
    """``c_k = out_w_hat(k) - in_w_hat(k)`` on max-normalized weights."""
    w = np.asarray(g.weight, dtype=np.float64) / float(np.asarray(g.weight).max())
    out_w = np.zeros(g.n_nodes)
    in_w = np.zeros(g.n_nodes)
    np.add.at(out_w, np.asarray(g.src), w)
    np.add.at(in_w, np.asarray(g.tgt), w)
    return out_w - in_w


def even_positions(order: np.ndarray, std: float) -> np.ndarray:
    """Embed a rank vector at even spacing, rescaled to the requested std."""
    n = order.shape[0]
    p = (order.astype(np.float64) / max(n - 1, 1)) * 2.0 - 1.0
    return p * (std / p.std())


def surrogate_F(g, pos: np.ndarray, beta: float) -> float:
    """``F(P) = sum_e w_hat_e * sigma(beta * (p_tgt - p_src))`` in float64."""
    w = np.asarray(g.weight, dtype=np.float64)
    nw = w / w.max()
    d = beta * (pos[np.asarray(g.tgt)] - pos[np.asarray(g.src)])
    # Numerically stable logistic.
    s = np.where(d >= 0, 1.0 / (1.0 + np.exp(-np.clip(d, -700, 700))),
                 np.exp(np.clip(d, -700, 700)) / (1.0 + np.exp(np.clip(d, -700, 700))))
    return float((s * nw).sum())


def main() -> None:
    g = load_dataset(CONFIG["dataset"])
    n = g.n_nodes
    w_max = float(np.asarray(g.weight).max())
    total = g.total_weight
    c = imbalance(g)

    # ── the four orders (order[i] = rank of node i) ──────────────────────────────
    best_order = load_best_solution(g)
    if isinstance(best_order, tuple):
        best_order = best_order[0]
    best_order = np.asarray(best_order).astype(np.int64)

    rock_pos = np.load(_ROOT / CONFIG["rocket_positions"])
    rock_order = np.argsort(np.argsort(rock_pos, kind="stable"),
                            kind="stable").astype(np.int64)

    # Imbalance sort: largest c (most source-like) -> rank 0 (front).
    imb_order = np.argsort(np.argsort(-c, kind="stable"), kind="stable").astype(np.int64)

    rng = np.random.RandomState(CONFIG["seed"])
    rand_order = np.argsort(np.argsort(rng.randn(n), kind="stable"),
                            kind="stable").astype(np.int64)

    orders = {"best": best_order, "rocket": rock_order,
              "imbalance_sort": imb_order, "random": rand_order}

    report = {"config": CONFIG, "git_commit": git_commit(),
              "graph": {"n_nodes": n, "n_edges": g.n_edges, "total_weight": total,
                        "w_max": w_max},
              "orders": {}}

    print(f"{'order':>15} {'discrete %':>11} {'ceiling':>12} {'<c,P> @std=1':>14}")
    for name, o in orders.items():
        sc = score_from_order(o, np.asarray(g.src), np.asarray(g.tgt), g.weight)
        ceiling = sc / w_max
        p1 = even_positions(o, 1.0)
        cp = float(c @ p1)
        report["orders"][name] = {"discrete_pct": pct(sc, total), "discrete_score": sc,
                                  "ceiling": ceiling, "c_dot_P_at_std1": cp}
        print(f"{name:>15} {pct(sc, total):11.4f} {ceiling:12.2f} {cp:14.2f}")

    # ── Q1: the small-scale limit is the imbalance objective ────────────────────
    # Prediction: as beta*std -> 0, F ranks orders by -<c,P>, so imbalance_sort wins and
    # `best` does NOT, even though `best` has by far the highest discrete score.
    small = {}
    W_hat = float((np.asarray(g.weight, dtype=np.float64) / w_max).sum())
    for std in (0.001, 0.01, 0.1, 0.58):
        row = {}
        for name, o in orders.items():
            p = even_positions(o, std)
            F = surrogate_F(g, p, CONFIG["beta_main"])
            predicted = W_hat / 2.0 - (CONFIG["beta_main"] / 4.0) * float(c @ p)
            row[name] = {"F": F, "F_linear_prediction": predicted,
                         "rel_err": abs(F - predicted) / abs(predicted)}
        small[str(std)] = row
    report["small_scale"] = small
    print(f"\nSMALL SCALE (beta={CONFIG['beta_main']}): F, and the linear "
          f"(imbalance-only) prediction   [W_hat/2 = {W_hat / 2:.2f}]")
    for std, row in small.items():
        cells = "  ".join(f"{k}={v['F']:.3f}" for k, v in row.items())
        err = max(v["rel_err"] for v in row.values())
        print(f"  std={std:>6}: {cells}   max rel err vs linear model = {err:.2e}")

    # ── Q2: ceiling vs smoothing loss across the scale grid ─────────────────────
    curves = {b: {name: [] for name in orders} for b in CONFIG["betas"]}
    for beta in CONFIG["betas"]:
        for std in CONFIG["std_grid"]:
            for name, o in orders.items():
                curves[beta][name].append(surrogate_F(g, even_positions(o, std), beta))
    report["F_curves"] = {str(b): {k: v for k, v in d.items()} for b, d in curves.items()}

    # Interpolated crossover (log-linear in std), NOT a grid node.
    cross = {}
    for beta in CONFIG["betas"]:
        fb = np.array(curves[beta]["best"])
        fr = np.array(curves[beta]["rocket"])
        diff = fb - fr
        idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
        if len(idx):
            i = int(idx[0])
            s0, s1 = CONFIG["std_grid"][i], CONFIG["std_grid"][i + 1]
            t = -diff[i] / (diff[i + 1] - diff[i])
            std_c = float(10 ** (np.log10(s0) + t * (np.log10(s1) - np.log10(s0))))
            cross[str(beta)] = {"crossover_std": std_c, "crossover_beta_std": beta * std_c,
                                "bracket": [s0, s1]}
        else:
            cross[str(beta)] = None
    report["crossover"] = cross
    print("\nCROSSOVER (best overtakes rocket), log-interpolated, even spacing:")
    for b, v in cross.items():
        print(f"  beta={b}: std={v['crossover_std']:.1f}  -> beta*std={v['crossover_beta_std']:.1f}"
              if v else f"  beta={b}: none in grid")

    # ── the decomposition at Rocket's operating scale ───────────────────────────
    op = {}
    for name, o in orders.items():
        p = even_positions(o, CONFIG["operating_std"])
        F = surrogate_F(g, p, CONFIG["beta_main"])
        ceil = report["orders"][name]["ceiling"]
        op[name] = {"F": F, "ceiling": ceil, "smoothing_loss": ceil - F,
                    "frac_of_ceiling": F / ceil}
    report["at_operating_scale"] = op
    print(f"\nAT ROCKET'S OPERATING SCALE (std={CONFIG['operating_std']}, "
          f"beta={CONFIG['beta_main']}), even spacing:")
    print(f"{'order':>15} {'ceiling':>11} {'F':>11} {'smoothing loss':>15} {'% of ceiling':>13}")
    for name, v in op.items():
        print(f"{name:>15} {v['ceiling']:11.2f} {v['F']:11.2f} "
              f"{v['smoothing_loss']:15.2f} {100 * v['frac_of_ceiling']:12.3f}%")
    gap_true = op["best"]["ceiling"] - op["rocket"]["ceiling"]
    gap_smooth = op["best"]["smoothing_loss"] - op["rocket"]["smoothing_loss"]
    report["operating_decomposition"] = {
        "true_advantage_of_best": gap_true,
        "extra_smoothing_loss_of_best": gap_smooth,
        "net_F_advantage": gap_true - gap_smooth}
    print(f"\n  best's true advantage        : {gap_true:+10.2f}")
    print(f"  best's EXTRA smoothing loss  : {-gap_smooth:+10.2f}")
    print(f"  net surrogate advantage      : {gap_true - gap_smooth:+10.2f}"
          f"   <-- negative means the surrogate prefers the WORSE order")

    _OUT.mkdir(parents=True, exist_ok=True)
    dest = _OUT / "q01_surrogate_ranking.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
