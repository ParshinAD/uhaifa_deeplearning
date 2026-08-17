"""Q04 — Is the continuous family's failure a property of *any* g(Delta), or only of
EXPONENTIALLY-TAILED g in finite precision?

Motivation
----------
`diagnosis.md` Q01 established two things about ``F_g(P) = sum_e w_hat_e * g(beta*Delta_e)``:

  (1) F depends only on the product ``s = beta*std`` (beta and position-scale are one knob);
  (2) with the sigmoid, the near-optimal order is ranked BELOW Rocket's own order at the
      operating point ``s ~ 148``, and only overtakes it at ``s* ~ 470`` — which the schedule
      never reaches.

Q01 then generalised this to "it is not the SHAPE of the sigmoid", citing four shapes. Two
gaps in that argument are what this script closes:

  * that four-shape table has **no committed artifact** (`q01_surrogate_ranking.json` contains
    no shape comparison) — it is re-measured here and written out;
  * its "slower-decaying tanh" arm was ``tanh(x/10)``, which is *not* a different shape:
    ``(tanh(z/2)+1)/2 == sigmoid(z)`` exactly, so ``tanh(x/10) `` is the sigmoid at ``beta/5``.
    By (1) that arm only moves along the ALREADY-SWEPT ``beta*std`` axis. The axis it was
    meant to test — **how fast the tail decays** — was therefore never varied.

The theory this script tests
----------------------------
**Fact A (alignment is universal, not the obstacle).** For ANY monotone bounded g with
``g(-inf)=0, g(+inf)=1``, ``F_g(order) -> ceiling(order)`` as ``s -> inf``. Since
``ceiling(best) > ceiling(rocket)``, a crossover ``s*(g)`` always exists. So "the surrogate
prefers the worse order" is a statement about the *reachable scale range*, never about g.

**Fact B (the obstacle is gradient SURVIVAL at s*, which is shape-dependent).**

    dF/dp_k = sum_{e: tgt=k} w_hat_e * beta * g'(beta*Delta_e)
            - sum_{e: src=k} w_hat_e * beta * g'(beta*Delta_e)

For the sigmoid ``g'(z) = sigma(z)(1-sigma(z)) ~ exp(-|z|)``, which **underflows to exactly 0
in float32** past |z| ~ 103. At ``s* ~ 470`` a typical pair sits at |z| ~ s*sqrt(2) ~ 665, so
in the aligned regime the long-range signal is not merely small — it is *identically zero* in
the precision the optimizer runs in. The aligned regime is invisible, not just distant.

For an **algebraically** (heavy) tailed g, ``1-g(z) ~ C z^-q``, we get ``g'(z) ~ qC z^-(q+1)``:
at z = 665, q = 1 gives ~2e-6 — a perfectly ordinary float32 number. No underflow at ANY
reachable scale.

**Fact C (with Adam, smallness is free; only exact zero costs).** Adam's per-coordinate step
``m_hat/(sqrt(v_hat)+eps)`` is invariant to a uniform rescaling of the gradient (both moments
scale), so a uniformly tiny but non-zero gradient still yields full-size steps. This repo has
already measured that twice: the A-INIT analysis (Adam keeps only ``sign(c)`` at small scale)
and Q01 insight #3 (Adam normalises the beta prefactor). Hence the decisive quantity is the
*fraction of the signal that survives float32*, not its magnitude.

So the falsifiable prediction is: **shapes differ not in whether they align (Fact A: all do)
but in whether the aligned regime is representable.** This script measures exactly that.

What is computed
----------------
For each candidate shape g, on the fly connectome, with EVEN spacing for every order (the
like-for-like convention of `q01_surrogate_ranking.py`):

  1. ``crossover s*(g)`` — log-interpolated scale at which F_g(best) overtakes F_g(rocket).
  2. **float32 gradient survival** at s*(g) and at Rocket's operating point: the fraction of
     edges whose ``g'`` is a non-zero float32, and the fraction of the total ``sum w_hat|g'|``
     they carry.
  3. The ``ceiling / smoothing-loss`` decomposition (Q01's) at the operating point.
  4. An exactness check of the identity ``(tanh(z/2)+1)/2 == sigmoid(z)`` — i.e. proof that
     the naive "use tanh" reading is a no-op, plus the same crossover for two tanh rescalings
     (they must land on the SAME beta*std as the sigmoid, verifying Q01's law).

Leakage: ``data/best_solution`` is read ONLY via ``mfas.analysis.gap.load_best_solution``.
Diagnostic only — nothing here feeds any variant's init, loss or move choice. Writes to
``experiments/outputs/`` only, never ``results/``.

Run (env `allen`, from the repo root):
    PYTHONPATH=src python experiments/diagnostics/q04_surrogate_tails.py
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
    # Rocket's terminal sharpness (the cyclic schedule spans [0.05, 1.05]).
    "beta_main": 1.05,
    # Rocket's own operating scale, measured from its converged positions (Q01).
    "operating_std": 141.0,
    # Wide log grid: sigmoid crosses over at beta*std ~ 470; heavy tails saturate far more
    # slowly, so the grid must reach much higher to bracket their crossover too.
    "std_grid": list(np.round(np.logspace(0.0, 8.0, 41), 6)),
    # Scales at which the float32 gradient-survival audit is run (in beta*std units the
    # operating point is 148; the sigmoid's crossover is 470).
    "audit_stds": [141.0, 447.9, 5000.0],
}


# ──────────────────────────────────────────────────────────────────────────────
# Shape family
# ──────────────────────────────────────────────────────────────────────────────
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic (identical form to q01_surrogate_ranking.py)."""
    zc = np.clip(z, -700, 700)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-zc)), np.exp(zc) / (1.0 + np.exp(zc)))


def _sigmoid_prime(z: np.ndarray) -> np.ndarray:
    s = _sigmoid(z)
    return s * (1.0 - s)


def _poly(z: np.ndarray, q: float) -> np.ndarray:
    """Heavy (algebraically) tailed monotone CDF-like shape.

        g_q(z) = 1/2 + 1/2 * sign(z) * (1 - (1+|z|)^-q)

    Bounded in (0,1), strictly increasing, g(0)=1/2 — the same normalisation as the sigmoid.
    Tail ``1 - g(z) = 1/2 (1+z)^-q`` decays POLYNOMIALLY, so ``g'(z) = q/2 (1+|z|)^-(q+1)``
    never underflows at any scale the optimizer can reach. q -> large recovers a fast
    (but still polynomial) saturation; q = 1 is the heaviest arm tested.
    """
    a = np.abs(z)
    return 0.5 + 0.5 * np.sign(z) * (1.0 - (1.0 + a) ** (-q))


def _poly_prime(z: np.ndarray, q: float) -> np.ndarray:
    return 0.5 * q * (1.0 + np.abs(z)) ** (-(q + 1.0))


def _cauchy(z: np.ndarray) -> np.ndarray:
    """Cauchy CDF: g(z) = 1/2 + arctan(z)/pi. Tail ~ 1/(pi z); g'(z) = 1/(pi(1+z^2))."""
    return 0.5 + np.arctan(z) / np.pi


def _cauchy_prime(z: np.ndarray) -> np.ndarray:
    return 1.0 / (np.pi * (1.0 + z * z))


def _clip_ramp(z: np.ndarray, m: float) -> np.ndarray:
    """H11's exact surrogate: clamp(0.5 + z/(2M), 0, 1) — linear core, HARD flat tails."""
    return np.clip(0.5 + z / (2.0 * m), 0.0, 1.0)


def _clip_ramp_prime(z: np.ndarray, m: float) -> np.ndarray:
    return np.where(np.abs(z) <= m, 1.0 / (2.0 * m), 0.0)


def _clipped(fn, fnp, cut: float):
    """The user's 'constant at top and bottom': force g to exactly {0,1} beyond |z| > cut.

    Rescaled so the core still spans the full (0,1) range continuously at +-cut, i.e.
    g(z) = clip( 1/2 + (core(z)-1/2)/(2*(core(cut)-1/2)) * ... ) — implemented as a linear
    renormalisation of the core so that g(+-cut) = {0,1} exactly and g is C^0.
    """
    def g(z):
        half = fn(np.asarray([float(cut)]))[0] - 0.5     # core half-range at the cut
        out = 0.5 + (fn(z) - 0.5) * (0.5 / half)
        return np.clip(out, 0.0, 1.0)

    def gp(z):
        half = fn(np.asarray([float(cut)]))[0] - 0.5
        return np.where(np.abs(z) <= cut, fnp(z) * (0.5 / half), 0.0)

    return g, gp


def _cusp(z: np.ndarray) -> np.ndarray:
    """Deliberately non-differentiable-at-0 UNBOUNDED shape sign(z)|z|^0.5 (Q01's 4th row).

    Included only to give the `diagnosis.md` shape table a committed artifact. Being
    unbounded it has no ceiling/smoothing decomposition; only its RANKING is meaningful.
    """
    return np.sign(z) * np.sqrt(np.abs(z))


def _cusp_prime(z: np.ndarray) -> np.ndarray:
    a = np.maximum(np.abs(z), 1e-300)
    return 0.5 / np.sqrt(a)


def shapes():
    """The candidate per-edge shapes: (name, g, g', tail description, bounded?)."""
    poly1_g, poly1_gp = (lambda z: _poly(z, 1.0)), (lambda z: _poly_prime(z, 1.0))
    poly2_g, poly2_gp = (lambda z: _poly(z, 2.0)), (lambda z: _poly_prime(z, 2.0))
    poly4_g, poly4_gp = (lambda z: _poly(z, 4.0)), (lambda z: _poly_prime(z, 4.0))
    clip1e3_g, clip1e3_gp = _clipped(poly1_g, poly1_gp, 1000.0)
    clip1e2_g, clip1e2_gp = _clipped(poly1_g, poly1_gp, 100.0)
    return [
        # name,           g,           g',              tail,               bounded
        ("sigmoid", _sigmoid, _sigmoid_prime, "exp(-|z|)", True),
        # THE CONTROL that separates "sharper core" from "heavier tail": the sigmoid
        # rescaled to poly_q4's transition width (2.1973/0.4954 = 4.4361). It has the same
        # core sharpness but keeps the EXPONENTIAL tail. Any gain that this control also
        # shows is the already-killed A-SCALE/H03 axis (== raising beta), NOT a tail effect.
        ("sigmoid_sharp_w0p5", lambda z: _sigmoid(4.4361 * z),
         lambda z: 4.4361 * _sigmoid_prime(4.4361 * z),
         "exp(-4.44|z|) = sigmoid at poly_q4's WIDTH (H03/A-SCALE control)", True),
        # tanh arms — MUST reproduce the sigmoid's crossover in beta*std units (no-op proof).
        ("tanh_half", lambda z: 0.5 * (np.tanh(z / 2.0) + 1.0),
         lambda z: 0.25 / np.cosh(z / 2.0) ** 2, "exp(-|z|) (== sigmoid exactly)", True),
        ("tanh_slow_x10", lambda z: 0.5 * (np.tanh(z / 10.0) + 1.0),
         lambda z: 0.05 / np.cosh(z / 10.0) ** 2, "exp(-|z|/5) (== sigmoid at beta/5)", True),
        # the genuinely untested axis: algebraic (heavy) tails
        ("poly_q1", poly1_g, poly1_gp, "z^-1  (g' ~ z^-2)", True),
        ("poly_q2", poly2_g, poly2_gp, "z^-2  (g' ~ z^-3)", True),
        ("poly_q4", poly4_g, poly4_gp, "z^-4  (g' ~ z^-5)", True),
        ("cauchy_arctan", _cauchy, _cauchy_prime, "z^-1  (g' ~ z^-2)", True),
        # the user's "constant at top and bottom" applied to the heavy core
        ("poly_q1_clip1000", clip1e3_g, clip1e3_gp, "z^-1 then HARD 0 beyond |z|>1000", True),
        ("poly_q1_clip100", clip1e2_g, clip1e2_gp, "z^-1 then HARD 0 beyond |z|>100", True),
        # provenance rows for the diagnosis.md four-shape table
        ("h11_clip_M5", lambda z: _clip_ramp(z, 5.0), lambda z: _clip_ramp_prime(z, 5.0),
         "HARD 0 beyond |z|>5 (H11 form)", True),
        ("cusp_sqrt", _cusp, _cusp_prime, "unbounded |z|^0.5", False),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (identical conventions to q01_surrogate_ranking.py)
# ──────────────────────────────────────────────────────────────────────────────
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_ROOT, text=True).strip()
    except Exception:                                     # pragma: no cover
        return "unknown"


def imbalance(g) -> np.ndarray:
    w = np.asarray(g.weight, dtype=np.float64) / float(np.asarray(g.weight).max())
    out_w = np.zeros(g.n_nodes)
    in_w = np.zeros(g.n_nodes)
    np.add.at(out_w, np.asarray(g.src), w)
    np.add.at(in_w, np.asarray(g.tgt), w)
    return out_w - in_w


def even_positions(order: np.ndarray, std: float) -> np.ndarray:
    """Embed a rank vector at even spacing, rescaled to the requested std (Q01 convention)."""
    n = order.shape[0]
    p = (order.astype(np.float64) / max(n - 1, 1)) * 2.0 - 1.0
    return p * (std / p.std())


def surrogate_F(nw, src, tgt, pos: np.ndarray, beta: float, g_fn) -> float:
    d = beta * (pos[tgt] - pos[src])
    return float((g_fn(d) * nw).sum())


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

    # ── the four orders (order[i] = rank of node i) ─────────────────────────────
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
              "graph": {"n_nodes": n, "n_edges": g.n_edges, "total_weight": total,
                        "w_max": w_max},
              "orders": {}}
    for name, o in orders.items():
        sc = score_from_order(o, src, tgt, g.weight)
        report["orders"][name] = {"discrete_pct": pct(sc, total),
                                  "discrete_score": sc, "ceiling": sc / w_max}

    # ── 0. the tanh == sigmoid identity (proves the naive reading is a no-op) ────
    zz = np.linspace(-40, 40, 200001)
    lhs = 0.5 * (np.tanh(zz / 2.0) + 1.0)
    rhs = _sigmoid(zz)
    report["tanh_identity"] = {
        "claim": "(tanh(z/2)+1)/2 == sigmoid(z) exactly; hence tanh(a*z) == sigmoid(2a*z)",
        "max_abs_err_float64": float(np.max(np.abs(lhs - rhs))),
        "max_rel_err_float64": float(np.max(np.abs(lhs - rhs) / np.maximum(rhs, 1e-300))),
        "n_points": int(zz.size), "range": [-40, 40],
    }
    print(f"tanh identity: max |(tanh(z/2)+1)/2 - sigmoid(z)| = "
          f"{report['tanh_identity']['max_abs_err_float64']:.3e}  (float64, 2e5 points)")

    # ── 1. F curves + crossover per shape ───────────────────────────────────────
    pos_cache = {name: {} for name in orders}

    def positions(name, std):
        if std not in pos_cache[name]:
            pos_cache[name][std] = even_positions(orders[name], std)
        return pos_cache[name][std]

    grid = CONFIG["std_grid"]
    report["shapes"] = {}
    print(f"\n{'shape':>18} {'tail':>34} {'crossover std':>14} {'beta*std':>11}")
    for sname, gf, gpf, tail, bounded in shapes():
        fb, fr = [], []
        for std in grid:
            fb.append(surrogate_F(nw, src, tgt, positions("best", std), beta, gf))
            fr.append(surrogate_F(nw, src, tgt, positions("rocket", std), beta, gf))
        fb, fr = np.array(fb), np.array(fr)
        diff = fb - fr
        idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
        cross = None
        if len(idx):
            i = int(idx[0])
            s0, s1 = grid[i], grid[i + 1]
            t = -diff[i] / (diff[i + 1] - diff[i])
            std_c = float(10 ** (np.log10(s0) + t * (np.log10(s1) - np.log10(s0))))
            cross = {"crossover_std": std_c, "crossover_beta_std": beta * std_c,
                     "bracket": [s0, s1]}
        report["shapes"][sname] = {
            "tail": tail, "bounded": bounded, "crossover": cross,
            "F_best": fb.tolist(), "F_rocket": fr.tolist(),
            "ranks_best_above_rocket_at_operating_std": bool(
                surrogate_F(nw, src, tgt, positions("best", CONFIG["operating_std"]), beta, gf)
                > surrogate_F(nw, src, tgt, positions("rocket", CONFIG["operating_std"]),
                              beta, gf)),
        }
        cs = (f"{cross['crossover_std']:14.4g} {cross['crossover_beta_std']:11.4g}"
              if cross else f"{'none in grid':>14} {'-':>11}")
        print(f"{sname:>18} {tail:>34} {cs}")

    # ── 2. ranking of all four orders at two scales (the diagnosis.md table) ────
    rank_tbl = {}
    for sname, gf, _gp, _t, _b in shapes():
        rank_tbl[sname] = {}
        for std in (0.01, CONFIG["operating_std"]):
            vals = {k: surrogate_F(nw, src, tgt, positions(k, std), beta, gf)
                    for k in orders}
            rank_tbl[sname][str(std)] = {
                "F": vals,
                "ranking": [k for k, _ in sorted(vals.items(), key=lambda kv: -kv[1])]}
    report["shape_ranking_table"] = rank_tbl
    print(f"\nRANKING BY SHAPE (beta={beta}):")
    print(f"{'shape':>18} | {'at std=0.01':>44} | {'at std=141':>44}")
    for sname in rank_tbl:
        a = " > ".join(rank_tbl[sname]["0.01"]["ranking"])
        b = " > ".join(rank_tbl[sname][str(CONFIG['operating_std'])]["ranking"])
        print(f"{sname:>18} | {a:>44} | {b:>44}")

    # ── 3. float32 gradient survival — the decisive quantity (Fact B) ───────────
    # At each audit scale, on ROCKET's order (the configuration the optimizer is actually
    # in), compute g' in float32 and measure how much of the signal underflows to zero.
    print(f"\nFLOAT32 GRADIENT SURVIVAL on Rocket's order "
          f"(min normal float32 = {np.finfo(np.float32).tiny:.3e}):")
    print(f"{'shape':>18} {'beta*std':>10} {'frac edges gp!=0':>16} "
          f"{'frac gp mass kept':>20}")
    surv = {}
    for sname, _gf, gpf, _t, _b in shapes():
        surv[sname] = {}
        for std in CONFIG["audit_stds"]:
            p = positions("rocket", std)
            d64 = beta * (p[tgt] - p[src])
            gp64 = gpf(d64)                                   # reference, float64
            gp32 = gpf(d64.astype(np.float32)).astype(np.float32)
            mass64 = float((np.abs(gp64) * nw).sum())
            nz = gp32 != 0.0
            mass32 = float((np.abs(gp32[nz]).astype(np.float64) * nw[nz]).sum())
            surv[sname][f"beta_std_{beta * std:.4g}"] = {
                "std": std, "beta_std": beta * std,
                "frac_edges_nonzero_float32": float(nz.mean()),
                "frac_gradient_mass_surviving": (mass32 / mass64) if mass64 > 0 else 0.0,
                "float64_mass": mass64,
            }
            print(f"{sname:>18} {beta * std:10.4g} {nz.mean():16.6f} "
                  f"{(mass32 / mass64 if mass64 > 0 else 0.0):20.6f}")
    report["float32_gradient_survival"] = surv

    # ── 3b. is the crossover just a RESCALING? (transition-width normalisation) ──
    # If every shape's crossover is a fixed multiple of its own transition width, then
    # "change the shape" == "change beta", i.e. the already-killed A-SCALE/H03 axis and
    # Q01's law. Width := the z at which g reaches 0.9 (half-range 0.5 -> 0.9).
    zgrid = np.logspace(-4, 4, 400001)
    widths = {}
    for sname, gf, _gp, _t, bounded in shapes():
        if not bounded:
            continue
        vals = gf(zgrid)
        idx = int(np.argmax(vals >= 0.9))
        widths[sname] = float(zgrid[idx]) if vals[idx] >= 0.9 else None
    report["transition_width_z_at_g0p9"] = widths
    print("\nIS THE CROSSOVER JUST A RESCALING? (crossover / transition width)")
    print(f"{'shape':>18} {'width(z@g=0.9)':>15} {'crossover beta*std':>19} {'ratio':>8}")
    for sname in widths:
        cr = report["shapes"][sname]["crossover"]
        if cr and widths[sname]:
            ratio = cr["crossover_beta_std"] / widths[sname]
            report["shapes"][sname]["crossover_over_width"] = ratio
            print(f"{sname:>18} {widths[sname]:15.4f} {cr['crossover_beta_std']:19.4f} "
                  f"{ratio:8.1f}")

    # ── 3c. ALIGNMENT RATIO at the operating point (scale-free, comparable) ─────
    # A(g) = (F_g(best) - F_g(rocket)) / (ceiling(best) - ceiling(rocket)).
    # A > 0  <=> the surrogate ranks the better order higher AT ROCKET'S OWN SCALE.
    # A = 1  <=> it sees the full true advantage; A < 0 <=> it prefers the worse order.
    true_adv = (report["orders"]["best"]["ceiling"]
                - report["orders"]["rocket"]["ceiling"])
    align = {}
    for sname, gf, _gp, _t, bounded in shapes():
        if not bounded:
            continue
        fb = surrogate_F(nw, src, tgt, positions("best", CONFIG["operating_std"]), beta, gf)
        fr = surrogate_F(nw, src, tgt, positions("rocket", CONFIG["operating_std"]),
                         beta, gf)
        align[sname] = {"F_best": fb, "F_rocket": fr, "margin": fb - fr,
                        "alignment_ratio": (fb - fr) / true_adv}
    report["alignment_at_operating_point"] = {"true_advantage": true_adv, "by_shape": align}
    print(f"\nALIGNMENT RATIO at Rocket's operating point (beta*std=148); "
          f"true advantage = {true_adv:.2f}")
    print(f"{'shape':>18} {'F(best)-F(rocket)':>18} {'alignment ratio':>16}")
    for sname, v in align.items():
        print(f"{sname:>18} {v['margin']:18.2f} {v['alignment_ratio']:16.4f}")

    # ── 3d. node-level gradient at ROCKET'S REAL positions (not even spacing) ───
    # The decisive dynamical quantity: a node whose float32 gradient is EXACTLY zero cannot
    # move at all, however Adam rescales. Also compare each shape's gradient DIRECTION to
    # the sigmoid's: cos ~ 1 would mean the shape swap is dynamically a no-op.
    print("\nNODE-LEVEL GRADIENT at Rocket's CONVERGED positions "
          f"(std={rock_pos.std():.2f}, beta={beta}):")
    print(f"{'shape':>18} {'frac nodes grad==0':>19} {'cos vs sigmoid':>15}")
    pos_real = rock_pos.astype(np.float64)
    d_real = beta * (pos_real[tgt] - pos_real[src])
    node_grads = {}
    for sname, _gf, gpf, _t, _b in shapes():
        gp = gpf(d_real.astype(np.float32)).astype(np.float32).astype(np.float64)
        contrib = gp * nw * beta
        gvec = np.zeros(n, dtype=np.float64)
        np.add.at(gvec, tgt, contrib)
        np.add.at(gvec, src, -contrib)
        node_grads[sname] = gvec
    ref = node_grads["sigmoid"]
    grad_report = {}
    for sname, gvec in node_grads.items():
        denom = np.linalg.norm(gvec) * np.linalg.norm(ref)
        cos = float(gvec @ ref / denom) if denom > 0 else float("nan")
        frac_zero = float((gvec == 0.0).mean())
        grad_report[sname] = {"frac_nodes_zero_grad_float32": frac_zero,
                              "cos_vs_sigmoid": cos,
                              "grad_l2": float(np.linalg.norm(gvec))}
        print(f"{sname:>18} {frac_zero:19.6f} {cos:15.6f}")
    report["node_gradient_at_rocket_positions"] = {
        "positions_std": float(rock_pos.std()), "by_shape": grad_report}

    # ── 4. ceiling / smoothing decomposition at the operating point (Q01's form) ─
    dec = {}
    for sname, gf, _gp, _t, bounded in shapes():
        if not bounded:
            continue
        row = {}
        for k in orders:
            F = surrogate_F(nw, src, tgt, positions(k, CONFIG["operating_std"]), beta, gf)
            ceil = report["orders"][k]["ceiling"]
            row[k] = {"F": F, "ceiling": ceil, "smoothing_loss": ceil - F}
        row["net_F_advantage_of_best"] = (
            (row["best"]["ceiling"] - row["rocket"]["ceiling"])
            - (row["best"]["smoothing_loss"] - row["rocket"]["smoothing_loss"]))
        dec[sname] = row
    report["operating_decomposition_by_shape"] = dec

    _OUT.mkdir(parents=True, exist_ok=True)
    dest = _OUT / "q04_surrogate_tails.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
