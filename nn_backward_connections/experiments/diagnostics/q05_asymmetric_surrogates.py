"""Q05 — Does a ONE-SIDED (asymmetric) surrogate escape the trade-off that Q01/Q04 found?

The question
------------
Every shape tested in Q04 was ODD-SYMMETRIC: ``g(-z) = 1 - g(z)`` (sigmoid, tanh, the algebraic
tails, H11's clamp, the cusp). This script tests a structurally different class:

    ASYM_FLAT_POS  (the proposal):   g(z) = 1                for z >= 0
                                     g(z) = 1 + tanh(z/T)    for z <  0
    ASYM_FLAT_NEG  (the mirror):     g(z) = tanh(z/T)        for z >  0
                                     g(z) = 0                for z <= 0

Both are monotone non-decreasing in ``Delta`` (so still argmax-preserving: "maximize
feedforward weight"), bounded in [0,1], and continuous. They are NOT reparametrizations of the
sigmoid — unlike the tanh identity ``(tanh(z/2)+1)/2 == sigmoid(z)`` that made the naive reading
of TODO 7 a no-op.

Why this class could behave differently — three structural claims to test
-------------------------------------------------------------------------
1. **All gradient goes to the violated edges.** With a flat positive branch, ``g' = 0`` on every
   edge that is already feedforward. The entire gradient budget is spent pulling FEEDBACK edges
   toward correctness, instead of widening margins that are already correct. (The mirror arm
   does the exact opposite and is included as the control.)

2. **It does not TELESCOPE, so it should escape Q01's small-scale degeneracy.** Q01's decisive
   small-scale result was that ``sum_e w_e sigma(beta*Delta_e)`` collapses to
   ``W/2 - (beta/4)<c,P>`` — a NODE-level imbalance objective that no longer knows which node
   precedes which, whose optimum is the imbalance sort, not the feedforward optimum. That
   derivation needs the sum to run over ALL edges. Here the first-order term is

       F ~ W_hat + (beta/T) * sum_{e: Delta_e < 0} w_hat_e * Delta_e

   restricted to the VIOLATED subset — which is itself order-dependent, so it does not
   telescope into ``<c,P>``. Prediction: at small ``beta*std`` the asymmetric shapes should NOT
   rank the imbalance sort first, unlike every shape in Q04.

3. **No margin-widening pressure ⇒ no scale inflation.** Q01 noted ``F`` has no finite-scale
   maximum: it grows monotonically with scale toward the ceiling, so positions inflate. With a
   flat positive branch, a correct edge exerts NO outward force, so the only pressure is
   whatever is needed to fix violations.

What is measured (same conventions as q01/q04: connectome, float64, EVEN spacing per order)
--------------------------------------------------------------------------------------------
* the small-scale ranking (test of claim 2);
* the crossover ``beta*std`` at which ``F_g(best)`` overtakes ``F_g(rocket)``;
* the scale-free alignment ratio ``A = (F_g(best)-F_g(rocket)) / (ceiling(best)-ceiling(rocket))``
  at Rocket's operating point;
* the split of gradient mass between feedforward and feedback edges (test of claim 1);
* the zero-gradient NODE fraction and the gradient direction vs the sigmoid, at Rocket's real
  converged positions.

Leakage: ``data/best_solution`` is read ONLY via ``mfas.analysis.gap.load_best_solution``.
Diagnostic only; writes to ``experiments/outputs/`` only, never ``results/``.

Run (env `allen`, from the repo root):
    PYTHONPATH=src python experiments/diagnostics/q05_asymmetric_surrogates.py
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
    "beta_main": 1.05,
    "operating_std": 141.0,
    "std_grid": list(np.round(np.logspace(-2.0, 6.0, 41), 6)),
    # T values for the tanh branch. 1.4925 makes the one-sided branch reach 0.1 at the same
    # |z| as the sigmoid does (2.1973), i.e. width-matched to the baseline.
    "T_values": [0.5, 1.0, 1.4925, 3.0],
    "audit_stds": [141.0, 447.9],
}


def _sigmoid(z):
    zc = np.clip(z, -700, 700)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-zc)), np.exp(zc) / (1.0 + np.exp(zc)))


def _sigmoid_prime(z):
    s = _sigmoid(z)
    return s * (1.0 - s)


def _asym_flat_pos(z, T):
    """g = 1 for z>=0 ; 1 + tanh(z/T) for z<0. Gradient ONLY on feedback edges."""
    return np.where(z >= 0.0, 1.0, 1.0 + np.tanh(np.clip(z, -700, 0.0) / T))


def _asym_flat_pos_prime(z, T):
    t = np.tanh(np.clip(z, -700, 0.0) / T)
    return np.where(z >= 0.0, 0.0, (1.0 - t * t) / T)


def _asym_flat_neg(z, T):
    """Mirror control: g = tanh(z/T) for z>0 ; 0 for z<=0. Gradient ONLY on feedforward edges."""
    return np.where(z <= 0.0, 0.0, np.tanh(np.clip(z, 0.0, 700) / T))


def _asym_flat_neg_prime(z, T):
    t = np.tanh(np.clip(z, 0.0, 700) / T)
    return np.where(z <= 0.0, 0.0, (1.0 - t * t) / T)


def shapes():
    out = [("sigmoid", _sigmoid, _sigmoid_prime, "symmetric baseline")]
    for T in CONFIG["T_values"]:
        out.append((f"asym_flat_pos_T{T}",
                    (lambda z, T=T: _asym_flat_pos(z, T)),
                    (lambda z, T=T: _asym_flat_pos_prime(z, T)),
                    f"flat on FEEDFORWARD, tanh(z/{T}) on FEEDBACK"))
    for T in CONFIG["T_values"]:
        out.append((f"asym_flat_neg_T{T}",
                    (lambda z, T=T: _asym_flat_neg(z, T)),
                    (lambda z, T=T: _asym_flat_neg_prime(z, T)),
                    f"MIRROR CONTROL: tanh(z/{T}) on FEEDFORWARD, flat on FEEDBACK"))
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_ROOT, text=True).strip()
    except Exception:                                     # pragma: no cover
        return "unknown"


def imbalance(g):
    w = np.asarray(g.weight, dtype=np.float64) / float(np.asarray(g.weight).max())
    out_w = np.zeros(g.n_nodes)
    in_w = np.zeros(g.n_nodes)
    np.add.at(out_w, np.asarray(g.src), w)
    np.add.at(in_w, np.asarray(g.tgt), w)
    return out_w - in_w


def even_positions(order, std):
    n = order.shape[0]
    p = (order.astype(np.float64) / max(n - 1, 1)) * 2.0 - 1.0
    return p * (std / p.std())


def main() -> None:
    g = load_dataset(CONFIG["dataset"])
    n = g.n_nodes
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    w = np.asarray(g.weight, dtype=np.float64)
    w_max = float(w.max())
    nw = w / w_max
    total = g.total_weight
    beta = CONFIG["beta_main"]
    c = imbalance(g)

    best_order = load_best_solution(g)
    if isinstance(best_order, tuple):
        best_order = best_order[0]
    best_order = np.asarray(best_order).astype(np.int64)
    rock_pos = np.load(_ROOT / CONFIG["rocket_positions"])
    rock_order = np.argsort(np.argsort(rock_pos, kind="stable"),
                            kind="stable").astype(np.int64)
    imb_order = np.argsort(np.argsort(-c, kind="stable"), kind="stable").astype(np.int64)
    rng = np.random.RandomState(CONFIG["seed"])
    rand_order = np.argsort(np.argsort(rng.randn(n), kind="stable"),
                            kind="stable").astype(np.int64)
    orders = {"best": best_order, "rocket": rock_order,
              "imbalance_sort": imb_order, "random": rand_order}

    report = {"config": CONFIG, "git_commit": git_commit(),
              "graph": {"n_nodes": n, "n_edges": g.n_edges, "total_weight": total},
              "orders": {}}
    for name, o in orders.items():
        sc = score_from_order(o, src, tgt, g.weight)
        report["orders"][name] = {"discrete_pct": pct(sc, total), "ceiling": sc / w_max}

    cache = {k: {} for k in orders}

    def positions(k, std):
        if std not in cache[k]:
            cache[k][std] = even_positions(orders[k], std)
        return cache[k][std]

    def F(k, std, gf):
        p = positions(k, std)
        return float((gf(beta * (p[tgt] - p[src])) * nw).sum())

    # ── claim 2: small-scale ranking (does it avoid the imbalance degeneracy?) ──
    print("SMALL-SCALE RANKING (claim 2: asymmetric shapes should NOT telescope to imbalance)")
    small = {}
    for sname, gf, _gp, desc in shapes():
        row = {}
        for std in (0.01, 0.1):
            vals = {k: F(k, std, gf) for k in orders}
            row[str(std)] = {"F": vals,
                             "ranking": [k for k, _ in sorted(vals.items(),
                                                              key=lambda kv: -kv[1])]}
        small[sname] = row
        print(f"  {sname:>22} @std=0.01: "
              f"{' > '.join(row['0.01']['ranking'])}")
    report["small_scale"] = small

    # ── crossover + alignment ratio ─────────────────────────────────────────────
    true_adv = report["orders"]["best"]["ceiling"] - report["orders"]["rocket"]["ceiling"]
    print(f"\nCROSSOVER + ALIGNMENT (true advantage of best = {true_adv:.2f})")
    print(f"{'shape':>22} {'crossover beta*std':>19} {'A @ operating':>14} "
          f"{'ranks best>rocket':>18}")
    res = {}
    for sname, gf, _gp, desc in shapes():
        fb = np.array([F("best", s, gf) for s in CONFIG["std_grid"]])
        fr = np.array([F("rocket", s, gf) for s in CONFIG["std_grid"]])
        d = fb - fr
        idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        cross = None
        if len(idx):
            i = int(idx[0])
            s0, s1 = CONFIG["std_grid"][i], CONFIG["std_grid"][i + 1]
            t = -d[i] / (d[i + 1] - d[i])
            std_c = float(10 ** (np.log10(s0) + t * (np.log10(s1) - np.log10(s0))))
            cross = {"crossover_std": std_c, "crossover_beta_std": beta * std_c}
        op_b = F("best", CONFIG["operating_std"], gf)
        op_r = F("rocket", CONFIG["operating_std"], gf)
        A = (op_b - op_r) / true_adv
        res[sname] = {"description": desc, "crossover": cross,
                      "F_best_at_operating": op_b, "F_rocket_at_operating": op_r,
                      "alignment_ratio": A,
                      "F_best_curve": fb.tolist(), "F_rocket_curve": fr.tolist()}
        cs = f"{cross['crossover_beta_std']:19.4g}" if cross else f"{'none in grid':>19}"
        print(f"{sname:>22} {cs} {A:14.4f} {str(A > 0):>18}")
    report["shapes"] = res

    # ── claim 1 + 3: gradient mass split and node mobility at REAL positions ────
    print("\nGRADIENT AT ROCKET'S CONVERGED POSITIONS "
          f"(std={rock_pos.std():.2f}, beta={beta}):")
    print(f"{'shape':>22} {'% mass on FEEDBACK':>19} {'frac nodes grad==0':>19} "
          f"{'cos vs sigmoid':>15}")
    pos_real = rock_pos.astype(np.float64)
    d_real = beta * (pos_real[tgt] - pos_real[src])
    is_fb = d_real < 0
    grads = {}
    for sname, _gf, gpf, _desc in shapes():
        gp = gpf(d_real.astype(np.float32)).astype(np.float32).astype(np.float64)
        contrib = gp * nw * beta
        mass = np.abs(contrib)
        frac_fb = float(mass[is_fb].sum() / mass.sum()) if mass.sum() > 0 else float("nan")
        gvec = np.zeros(n, dtype=np.float64)
        np.add.at(gvec, tgt, contrib)
        np.add.at(gvec, src, -contrib)
        grads[sname] = gvec
        ref = grads["sigmoid"]
        den = np.linalg.norm(gvec) * np.linalg.norm(ref)
        cos = float(gvec @ ref / den) if den > 0 else float("nan")
        fz = float((gvec == 0.0).mean())
        res[sname]["gradient_at_rocket_positions"] = {
            "frac_mass_on_feedback_edges": frac_fb,
            "frac_nodes_zero_grad_float32": fz, "cos_vs_sigmoid": cos,
            "grad_l2": float(np.linalg.norm(gvec))}
        print(f"{sname:>22} {100 * frac_fb:18.3f}% {fz:19.6f} {cos:15.6f}")
    report["feedback_weight_frac_at_rocket"] = float(nw[is_fb].sum() / nw.sum())
    print(f"\n(for reference: feedback edges carry "
          f"{100 * report['feedback_weight_frac_at_rocket']:.3f}% of hat-w at this order)")

    _OUT.mkdir(parents=True, exist_ok=True)
    dest = _OUT / "q05_asymmetric_surrogates.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
