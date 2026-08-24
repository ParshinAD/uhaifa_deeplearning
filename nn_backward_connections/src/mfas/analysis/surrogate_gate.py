"""Surrogate gate — the reusable battery every NEW per-edge surrogate must pass.

Distilled from the Phase-6.5/6.6 cycles (H37 tails KILL, H38 asymmetry GRAPH-DEPENDENT).
Those two cycles cost several hours of large-graph compute and produced three lessons that
are cheap to re-check and expensive to re-learn:

1. **Most "new" surrogates are the sigmoid in disguise.** ``(tanh(z/2)+1)/2 == sigmoid(z)``
   exactly, so "use tanh" / "use a slower-decaying tanh" is a pure ``beta`` rescaling — the
   axis H03 / A-SCALE already killed. :func:`sigmoid_reparametrization_residual` detects this
   in milliseconds.
2. **A surrogate can have a trivial degenerate optimum.** If ``g(0) == sup g`` then the
   COLLAPSED configuration ``P = const`` attains ``F = sum_e w_hat_e``, the global maximum of
   ``F``, strictly above every ordering — and the dynamics flow into it (a violated edge pulls
   its endpoints together). H38's first form had exactly this defect: it collapsed to position
   std 5e-4 and scored 58.24% where the sigmoid scored 73.68%. :func:`collapse_check` is an
   EXACT, dataset-free test for it.
3. **Static alignment does not predict achieved score**, so alignment must never be used as a
   go/no-go on its own — but it does explain *why* a shape behaves as it does, and it is the
   one place the asymmetric family visibly differs from the symmetric one.

The gate is deliberately split into CHEAP checks (no dataset, milliseconds — run these in
pytest and before writing any variant) and GRAPH checks (need a graph, and the alignment
probes additionally need a reference order, so they are privileged/diagnostic).

Leakage note: this module lives in the PRIVILEGED :mod:`mfas.analysis` package because
:func:`alignment_report` compares against a reference order. Nothing here may be imported by
a variant in ``mfas.experiments`` — it is analysis/diagnostic machinery only.

Typical use for a new idea, cheapest first::

    from mfas.analysis.surrogate_gate import cheap_gate
    report = cheap_gate(lambda z: my_shape(z))
    assert report["verdict"] != "REJECT", report["reasons"]

then the graph-level probes, then the prototype gate on proxies, and only then a variant run.
See ``experiments/surrogate_gate.py`` for the CLI driver and REQUIRED_CONTROLS below for the
controls a positive result must survive.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "REQUIRED_CONTROLS",
    "sigmoid_reparametrization_residual",
    "monotone_bounded_check",
    "autograd_check",
    "collapse_check",
    "transition_width",
    "cheap_gate",
    "alignment_report",
]

Surrogate = Callable[[torch.Tensor], torch.Tensor]


# ──────────────────────────────────────────────────────────────────────────────
# The controls a positive result MUST survive (learned the hard way in H38)
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_CONTROLS = {
    "mirror": (
        "Reflect the shape's asymmetry (swap which side is flat / steep). If the mirror gains "
        "too, the effect is 'any one-sided shape' or the position scale it induces, not the "
        "mechanism claimed. H38 measured mirror = -2.8890 pp vs H38 = +0.3668 pp on connectome."
    ),
    "beta_rescale": (
        "Run the PLAIN sigmoid with beta multiplied by a constant for the WHOLE run. Q01 showed "
        "F depends only on beta*std, so this sweeps exactly the axis a scale change would (the "
        "killed A-SCALE == H03). If it reproduces the gain, the shape is not the mechanism. "
        "H38 measured sigmoid at beta x4 = -0.5804 pp. NOTE this is stronger than H03 itself, "
        "which only ramped beta over the last 25% of epochs."
    ),
    "final_vs_best": (
        "Compare the FINAL-epoch discrete score, not only the best-by-oracle tracked maximum. "
        "If the gain exists only in the tracked max, it is a sampling artifact of a noisier "
        "trajectory. H38: +0.392 final vs +0.396 best (mouse) — genuine."
    ),
    "three_datasets": (
        "connectome AND microns are both PRIMARY (PROTOCOL.md Phase-5 table); mouse is "
        "supporting/non-inferiority only. H38 passed connectome by 9x the gate and REGRESSED on "
        "microns (-0.6689 pp) => GRAPH-DEPENDENT, not a general win. A proxy-only or "
        "connectome-only result must never be reported as general."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# CHEAP checks — no dataset, milliseconds
# ──────────────────────────────────────────────────────────────────────────────
def sigmoid_reparametrization_residual(g: Surrogate, z_max: float = 20.0,
                                       n: int = 20001) -> Dict[str, float]:
    """Is ``g`` just ``sigmoid(a*z)`` for some ``a``? Returns the best-fit residual.

    ``(tanh(z/2)+1)/2 == sigmoid(z)``, so every "tanh surrogate" and every argument rescaling
    is the sigmoid at a different ``beta``. Because ``F`` depends only on the product
    ``beta*std`` (Q01), such a shape moves along an axis that H03 / A-SCALE already killed and
    is a **no-op as a hypothesis**.

    Scans ``a`` on a coarse log grid and then REFINES locally — the refinement is not optional:
    with a coarse grid alone, ``tanh(z/10)`` (a true reparametrization, ``a = 0.2``) lands on a
    neighbouring grid point and reports a residual of 5e-4, which is far above any sane
    threshold and would let exactly the case this function exists for slip through.

    ``residual < 1e-6`` ⇒ the shape IS the sigmoid reparametrized.
    """
    z = torch.linspace(-z_max, z_max, n, dtype=torch.float64)
    gv = g(z).to(torch.float64)

    def resid(a: float) -> float:
        return float((gv - torch.sigmoid(float(a) * z)).abs().max())

    grid = np.logspace(-2, 2, 401)
    best_a = float(min(grid, key=resid))
    # Local refinement in log-space around the coarse winner (ternary search on a unimodal
    # residual), 60 iterations => the bracket shrinks by ~(2/3)^60, well past float precision.
    step = np.log10(grid[1]) - np.log10(grid[0])
    lo, hi = np.log10(best_a) - step, np.log10(best_a) + step
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if resid(10 ** m1) < resid(10 ** m2):
            hi = m2
        else:
            lo = m1
    a = 10 ** ((lo + hi) / 2.0)
    return {"residual": resid(a), "a": float(a), "coarse_a": best_a}


def monotone_bounded_check(g: Surrogate, z_max: float = 500.0,
                           n: int = 200001) -> Dict[str, object]:
    """Monotone non-decreasing (⇒ argmax-preserving) and bounded.

    A surrogate must be non-decreasing in ``Delta`` so that "more feedforward" is never
    penalised — this is what keeps ``-loss`` a relaxation of "maximize feedforward weight".
    Unbounded shapes are flagged: they let the optimizer buy objective by inflating positions.
    """
    z = torch.linspace(-z_max, z_max, n, dtype=torch.float64)
    v = g(z).to(torch.float64)
    d = torch.diff(v)
    return {
        "monotone_non_decreasing": bool((d >= -1e-15).all()),
        "min_increment": float(d.min()),
        "min_value": float(v.min()),
        "max_value": float(v.max()),
        "bounded": bool(torch.isfinite(v).all() and v.abs().max() < 1e6),
    }


def autograd_check(g: Surrogate, probes: Optional[Sequence[float]] = None,
                   h: float = 1e-7, tol: float = 1e-4) -> Dict[str, object]:
    """Analytic (autograd) vs central-difference derivative, INCLUDING z = 0.

    Two real defects were found this way and both were missed by probe sets that skipped the
    interesting points: ``torch.sign(0) == 0`` silently zeroes the H37 gradient at ``z = 0``,
    and ``torch.where`` zeroes the H38 gradient exactly at the branch point ``z = M``.
    Mismatches at isolated points are measure-zero kinks (usually harmless), but they must be
    DOCUMENTED, never claimed away — so this returns them rather than raising.
    """
    if probes is None:
        probes = [-100.0, -50.0, -7.3, -3.0, -1.0, -0.5, -0.2, 0.0,
                  0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 4.0, 33.0, 100.0]
    rows, mismatches = [], []
    for zz in probes:
        x = torch.tensor([zz], dtype=torch.float64, requires_grad=True)
        g(x).sum().backward()
        ana = float(x.grad.item())
        num = float(((g(torch.tensor([zz + h], dtype=torch.float64))
                      - g(torch.tensor([zz - h], dtype=torch.float64))) / (2 * h)).item())
        ok = abs(ana - num) <= tol
        rows.append({"z": zz, "autograd": ana, "central_diff": num, "ok": ok})
        if not ok:
            mismatches.append(zz)
    return {"rows": rows, "mismatch_points": mismatches,
            "all_match": len(mismatches) == 0}


def collapse_check(g: Surrogate, z_max: float = 500.0, n: int = 200001) -> Dict[str, object]:
    """EXACT, dataset-free test for the degenerate collapsed optimum.

    ``F(P) = sum_e w_hat_e * g(beta*Delta_e)``. The collapsed configuration ``P = const`` has
    every ``Delta = 0``, so ``F(collapse) = (sum_e w_hat_e) * g(0)``, while
    ``sup_P F <= (sum_e w_hat_e) * sup_z g``. Hence

        collapse is the GLOBAL MAXIMUM  <=>  g(0) == sup_z g

    and when that holds the dynamics also flow into it (a violated edge pulls its endpoints
    together), so it is not a theoretical curiosity: H38's ``M = 0`` form collapsed to position
    std 5e-4 and scored 58.24% vs the sigmoid's 73.68% on the hard synthetic.

    ``headroom`` is ``sup g - g(0)`` in units of the shape's own range; the fix that rescued
    H38 was introducing a margin ``M > 0`` so that ``g(0) < sup g``.
    """
    z = torch.linspace(-z_max, z_max, n, dtype=torch.float64)
    v = g(z).to(torch.float64)
    g0 = float(g(torch.zeros(1, dtype=torch.float64)).item())
    sup, inf = float(v.max()), float(v.min())
    rng = max(sup - inf, 1e-300)
    headroom = (sup - g0) / rng
    return {"g_at_0": g0, "sup_g": sup, "inf_g": inf,
            "headroom_frac_of_range": headroom,
            "collapse_is_global_max": bool(headroom < 1e-12)}


def transition_width(g: Surrogate, level: float = 0.9, z_max: float = 1e4,
                     n: int = 400001) -> Optional[float]:
    """The ``z`` at which ``g`` first reaches ``level`` of its range — the shape's core width.

    Q04 found the crossover scale is ≈ 180–230 × this width across eleven symmetric shapes,
    i.e. within that family "change the shape" is mostly "change beta". ⚠ The constant is
    THRESHOLD-DEPENDENT (the ratio spread is 1.18x at level 0.875 but 12.5x at 0.99), so use it
    as a comparator between shapes at a FIXED level, never as a universal law.
    """
    z = torch.logspace(-4, np.log10(z_max), n, dtype=torch.float64)
    v = g(z).to(torch.float64)
    lo, hi = float(g(torch.tensor([-z_max], dtype=torch.float64)).item()), float(v.max())
    target = lo + level * (hi - lo)
    idx = torch.nonzero(v >= target)
    return float(z[int(idx[0])]) if len(idx) else None


def cheap_gate(g: Surrogate, name: str = "candidate") -> Dict[str, object]:
    """Run every dataset-free check and return a verdict with reasons.

    ``REJECT`` = the idea is a no-op or degenerate; do not spend compute.
    ``WARN``   = usable but with a documented defect (e.g. a derivative kink).
    ``PASS``   = proceed to the graph-level probes, then the prototype gate on the proxies.
    """
    rep: Dict[str, object] = {"name": name}
    rep["sigmoid_reparametrization"] = sigmoid_reparametrization_residual(g)
    rep["monotone_bounded"] = monotone_bounded_check(g)
    rep["autograd"] = autograd_check(g)
    rep["collapse"] = collapse_check(g)
    rep["transition_width_at_0.9"] = transition_width(g)

    reasons = []
    if rep["sigmoid_reparametrization"]["residual"] < 1e-6:
        a = rep["sigmoid_reparametrization"]["a"]
        res = rep["sigmoid_reparametrization"]["residual"]
        rep["is_baseline_sigmoid"] = bool(abs(a - 1.0) < 1e-3)
        reasons.append(
            f"REJECT: this IS the baseline sigmoid (residual {res:.2e})."
            if rep["is_baseline_sigmoid"] else
            f"REJECT: this is sigmoid({a:.4g}*z) to {res:.2e} — a pure beta rescaling "
            f"(A-SCALE == H03, already killed), not a new hypothesis. F depends only on "
            f"beta*std (Q01), so this moves along an axis that is already closed.")
    if rep["collapse"]["collapse_is_global_max"]:
        reasons.append(
            "REJECT: g(0) == sup g, so the collapsed configuration P=const is the GLOBAL "
            "maximum of F and the dynamics flow into it. Add a margin so g(0) < sup g.")
    if not rep["monotone_bounded"]["monotone_non_decreasing"]:
        reasons.append(
            "REJECT: not monotone non-decreasing in Delta — the surrogate would penalise "
            "making an edge more feedforward, so it is not a relaxation of the objective.")
    if not rep["monotone_bounded"]["bounded"]:
        reasons.append(
            "WARN: unbounded — the optimizer can buy objective by inflating positions.")
    if not rep["autograd"]["all_match"]:
        reasons.append(
            f"WARN: autograd != central difference at z in "
            f"{rep['autograd']['mismatch_points']} (kink / branch point). Harmless if "
            f"measure-zero, but it must be documented, not claimed away.")

    rep["reasons"] = reasons
    rep["verdict"] = ("REJECT" if any(r.startswith("REJECT") for r in reasons)
                      else "WARN" if reasons else "PASS")
    return rep


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH-level probes (need a graph; alignment additionally needs a reference order)
# ──────────────────────────────────────────────────────────────────────────────
def alignment_report(g_shape: Surrogate, graph, orders: Dict[str, np.ndarray],
                     beta: float = 1.05, operating_std: float = 141.0,
                     small_std: float = 0.01) -> Dict[str, object]:
    """Small-scale ranking + alignment ratio at the operating point.

    ``orders`` maps a label to a rank vector; it must contain ``"best"`` and ``"rocket"``, and
    ideally ``"imbalance_sort"`` (the shape the small-scale limit degenerates to) and
    ``"random"``. Every order is embedded at EVEN spacing rescaled to the requested std, the
    like-for-like convention of ``q01_surrogate_ranking.py``.

    Two things are measured:

    * **small-scale ranking** — with a symmetric shape the edge sum telescopes into the
      node-level imbalance objective and the imbalance sort wins; a shape that ranks ``best``
      first at ``small_std`` has escaped that degeneracy (only the asymmetric family does).
    * **alignment ratio** ``A = (F(best) - F(rocket)) / (ceiling(best) - ceiling(rocket))`` —
      the fraction of the better order's TRUE advantage the surrogate sees at the operating
      point. ``A < 0`` means it prefers the worse order.

    ⚠ ``A`` is diagnostic ONLY. Q04 showed it anti-correlates with achieved score inside the
    symmetric family and correlates in the asymmetric one — never gate on it alone.
    """
    src, tgt = np.asarray(graph.src), np.asarray(graph.tgt)
    w = np.asarray(graph.weight, dtype=np.float64)
    w_max = float(w.max())
    nw = torch.as_tensor(w / w_max, dtype=torch.float64)
    src_t = torch.as_tensor(src, dtype=torch.long)
    tgt_t = torch.as_tensor(tgt, dtype=torch.long)

    def embed(order: np.ndarray, std: float) -> torch.Tensor:
        n = order.shape[0]
        p = (order.astype(np.float64) / max(n - 1, 1)) * 2.0 - 1.0
        return torch.as_tensor(p * (std / p.std()), dtype=torch.float64)

    def F(order: np.ndarray, std: float) -> float:
        p = embed(order, std)
        return float((g_shape(beta * (p[tgt_t] - p[src_t])) * nw).sum())

    from ..metrics import score_from_order
    ceilings = {k: score_from_order(np.asarray(o), src, tgt, graph.weight) / w_max
                for k, o in orders.items()}

    small = {k: F(o, small_std) for k, o in orders.items()}
    op = {k: F(o, operating_std) for k, o in orders.items()}
    true_adv = ceilings["best"] - ceilings["rocket"]
    A = (op["best"] - op["rocket"]) / true_adv if true_adv != 0 else float("nan")
    return {
        "beta": beta, "operating_std": operating_std, "small_std": small_std,
        "ceilings": ceilings,
        "F_small_scale": small,
        "small_scale_ranking": [k for k, _ in sorted(small.items(), key=lambda kv: -kv[1])],
        "escapes_small_scale_telescoping": bool(
            max(small, key=lambda k: small[k]) == "best"),
        "F_at_operating": op,
        "operating_ranking": [k for k, _ in sorted(op.items(), key=lambda kv: -kv[1])],
        "true_advantage_of_best": true_adv,
        "alignment_ratio": A,
    }
