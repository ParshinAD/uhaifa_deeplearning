"""Proto-H34 — perturbed / blackbox differentiable SORT surrogate (CPU PROTOTYPE GATE).

Backlog H34 is the ONLY untried CONTINUOUS mechanism and carries the lowest EV: a
perturb-and-MAP gradient through a SORT (Berthet 2020, "Learning with Differentiable
Perturbed Optimizers"). It is the clean falsifier for finding #3 ("continuous gradient
flow collapses even a PERFECT order back to the ~82.9% basin") and the H19 KILL
("soft-rank stalls at init because O(1/n) rank gaps give vanishing gradients").

Why this might differ from H19
------------------------------
H19 differentiated THROUGH the rank gaps (soft-rank r_i = Σ_j σ(α·(p_j−p_i))); the
feedforward surrogate's gradient w.r.t. p then scales with the rank gaps, which are
O(1/n) → too flat → the optimizer stalls at the init. The perturbed-sort estimator does
NOT differentiate through the argsort at all. It uses the score-function / Stein form of
Berthet's perturbed optimizer:

    F(p) = E_{Z~N(0,I)}[ ff_weight( argsort(p + ε·Z) ) ]
    ∇_p F(p) = (1/ε) · E_{Z}[ ff_weight(argsort(p + ε·Z)) · Z ]              (Stein)

where ``ff_weight(order)`` = Σ_(u,v) w(u,v)·1[rank(u) < rank(v)] is the EXACT (discrete,
linear-in-order-indicator) feedforward weight on a perturbed argsort. The inner solver is
a SORT (O(n log n)) — NEVER a per-step greedy-FAS pass. Because the gradient is an
average of (objective × noise) over a few perturbed sorts, its magnitude is governed by ε
and the spread of ff_weight across perturbations, NOT by O(1/n) rank gaps. So if a
continuous lever can EVER move this metric, it should move here. That makes a stall the
decisive falsification.

The gate
--------
On the gap-bearing hard synthetic (and optionally mouse), for seeds 42/123/999, compare
the EXACT feedforward pct of:
  * sigmoid Rocket (the baseline continuous surrogate)  — via run_rocket
  * perturbed-sort surrogate, swept over a small ε / schedule grid (best ε reported)
Track whether the order MOVES from its init or STALLS (like H19). GO iff perturbed-sort
beats sigmoid Rocket beyond noise on the gap-bearing synthetic; otherwise NO-GO /
FALSIFIED (continuous methods exhausted on this problem).

This is a SELF-CONTAINED prototype: nothing is written to src/ and nothing is written to
results/. It writes only experiments/outputs/proto_h34_perturbsort.json. CPU ONLY.

Run:  PYTHONPATH=src python experiments/proto_h34_perturbsort.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas.analysis.gap import make_hard_synthetic_graph                 # noqa: E402
from mfas.baseline.rocket import (RocketConfig, make_beta_schedule,     # noqa: E402
                                  run_rocket)
from mfas.io import GraphData, load_mouse                               # noqa: E402
from mfas.metrics import pct, score_from_positions                     # noqa: E402

# ── CPU-only guard (per task spec) ─────────────────────────────────────────────
torch.set_num_threads(2)

# Locked hard-synthetic config (cfg0 from the H21 tuning: stable ~+0.80 pp gap).
SYNTH_PARAMS = dict(n=400, avg_out=10, feedback_frac=0.65, n_clusters=8,
                    intra_cycle_frac=0.55, weight_alpha=2.0)
SYNTH_EPOCHS = 4000        # equal-compute budget for the sigmoid Rocket baseline
SEEDS = (42, 123, 999)

# Perturbed-sort optimizer budget. We give it the SAME number of optimizer steps as the
# sigmoid baseline; each step draws ``N_PERTURB`` Monte-Carlo sorts (O(n log n) each).
PS_EPOCHS = 4000
N_PERTURB = 16             # MC samples per gradient estimate (averaged over ±antithetic)
PS_LR = 0.05               # Adam LR (matches baseline)
PS_GRAD_CLIP = 1.0
# ε grid (graduated schedule explored): noise std added to STANDARDIZED positions before
# the argsort. Each entry is (eps_start, eps_end); ε decays linearly start→end across the
# run (graduated annealing, as Berthet's temperature is annealed).
EPS_GRID = [
    (1.0, 1.0),    # constant moderate
    (0.5, 0.5),    # constant small
    (2.0, 0.5),    # graduated large→small
    (1.0, 0.2),    # graduated moderate→small
    (0.3, 0.3),    # constant tiny (closest to a near-hard sort)
]
LOG_INTERVAL = 50


# ──────────────────────────────────────────────────────────────────────────────
# Perturbed-sort surrogate optimizer (Berthet 2020 / Stein score-function gradient)
# ──────────────────────────────────────────────────────────────────────────────
def _ff_weight_of_perm(rank: np.ndarray, src: np.ndarray, tgt: np.ndarray,
                       w: np.ndarray) -> float:
    """Exact feedforward weight of a rank vector (rank[node] = position on the line).

    ``rank`` here means the *position* of each node (smaller = earlier = source side);
    an edge (u,v) is feedforward iff rank[u] < rank[v]. This is the same linear-in-order
    indicator the frozen oracle uses; we compute it locally (float64) for the MC estimate.
    """
    ff = rank[tgt] > rank[src]
    return float(w[ff].sum())


def run_perturbed_sort(g: GraphData, seed: int, init_positions: np.ndarray,
                       eps_start: float, eps_end: float, epochs: int = PS_EPOCHS,
                       n_perturb: int = N_PERTURB, lr: float = PS_LR,
                       time_limit: Optional[float] = None):
    """Optimize continuous positions with a perturb-and-MAP SORT (Berthet) gradient.

    Mechanism (per optimizer step):
      1. draw ``n_perturb`` antithetic Gaussian noises Z, build p̃ = p + ε·Z
      2. argsort each p̃ (the SORT inner solver, O(n log n)) → a permutation
      3. score ff_weight on each permutation with the EXACT linear indicator
      4. Stein gradient estimate: ∇_p F ≈ (1/ε) · mean_k[ (F_k − baseline) · Z_k ]
         (the baseline = mean F_k is a variance-reduction control variate; it does not
          bias the estimate since E[Z]=0). We MAXIMIZE F, so Adam steps on −∇.
    Antithetic sampling (±Z paired) further reduces variance and is standard for these
    perturbed-optimizer estimators.

    Best-by-oracle tracking mirrors the baseline: every ``LOG_INTERVAL`` steps the EXACT
    feedforward score of the *current* (un-perturbed) argsort is computed and the best is
    retained. The loss/perturbation use input weights + positions ONLY — no oracle peeking.

    Returns a dict with best_pct, best_positions, movement diagnostics, history.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    src = np.asarray(g.src)
    tgt = np.asarray(g.tgt)
    # float64 weights for the MC ff-weight; the final score is via the FROZEN oracle.
    w = np.asarray(g.weight, dtype=np.float64)
    n = g.n_nodes
    total = g.total_weight

    p = np.asarray(init_positions, dtype=np.float64).copy()
    # standardize the init (scale-free start, like H19/diagnosis convention)
    p = (p - p.mean()) / (p.std() + 1e-12)
    init_p = p.copy()

    # Adam state (manual, float64 — keeps everything off MPS and exact).
    m = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    b1, b2, adam_eps = 0.9, 0.999, 1e-8

    def exact_pct(pos: np.ndarray) -> float:
        sc = score_from_positions(pos, src, tgt, g.weight)
        return pct(sc, total)

    # init order (the argsort of the current positions) for the movement probe
    init_order = np.argsort(np.argsort(p, kind="stable"), kind="stable")

    best_score = score_from_positions(p, src, tgt, g.weight)
    best_pos = p.copy()

    history = []
    start = time.time()
    last_i = 0
    for i in range(epochs):
        if time_limit and (time.time() - start) > time_limit:
            break
        last_i = i
        frac = i / max(epochs - 1, 1)
        eps = eps_start + (eps_end - eps_start) * frac     # linear ε schedule

        # ── perturbed sorts + Stein gradient (antithetic) ──────────────────────
        grad = np.zeros(n, dtype=np.float64)
        Fs = []
        half = n_perturb // 2
        Zs = rng.randn(half, n)
        F_pos = np.empty(half, dtype=np.float64)
        F_neg = np.empty(half, dtype=np.float64)
        for k in range(half):
            Z = Zs[k]
            # +Z
            pp = p + eps * Z
            order_p = np.argsort(np.argsort(pp, kind="stable"), kind="stable")
            F_pos[k] = _ff_weight_of_perm(order_p, src, tgt, w)
            # −Z (antithetic)
            pn = p - eps * Z
            order_n = np.argsort(np.argsort(pn, kind="stable"), kind="stable")
            F_neg[k] = _ff_weight_of_perm(order_n, src, tgt, w)
        Fs = np.concatenate([F_pos, F_neg])
        baseline = Fs.mean()                               # control variate
        # Stein: ∇_p E[F(p+εZ)] = (1/ε) E[F·Z]; antithetic pairs share Z and −Z.
        for k in range(half):
            grad += (F_pos[k] - baseline) * Zs[k]
            grad += (F_neg[k] - baseline) * (-Zs[k])
        grad /= (eps * n_perturb)
        # normalize gradient scale by total weight so LR is comparable across graphs
        grad = grad / max(total, 1.0)

        # we MAXIMIZE F → ascend ∇; Adam on −grad (descend the negative).
        g_step = -grad
        # gradient clipping (global norm), matches baseline
        gn = np.linalg.norm(g_step)
        if gn > PS_GRAD_CLIP:
            g_step = g_step * (PS_GRAD_CLIP / (gn + 1e-12))

        m = b1 * m + (1 - b1) * g_step
        v = b2 * v + (1 - b2) * (g_step * g_step)
        mhat = m / (1 - b1 ** (i + 1))
        vhat = v / (1 - b2 ** (i + 1))
        p = p - lr * mhat / (np.sqrt(vhat) + adam_eps)

        if i % LOG_INTERVAL == 0 or i == epochs - 1:
            sc = score_from_positions(p, src, tgt, g.weight)
            if sc > best_score:
                best_score = sc
                best_pos = p.copy()
            cur_order = np.argsort(np.argsort(p, kind="stable"), kind="stable")
            moved = int((cur_order != init_order).sum())
            history.append(dict(
                iter=i, pct=pct(sc, total), best_pct=pct(best_score, total),
                eps=eps, mean_F_pct=pct(baseline, total),
                grad_norm=float(np.linalg.norm(grad)),
                moved_nodes=moved,
            ))

    wall = time.time() - start
    final_order = np.argsort(np.argsort(p, kind="stable"), kind="stable")
    moved_nodes = int((final_order != init_order).sum())
    return dict(
        best_pct=pct(best_score, total),
        best_positions=best_pos,
        init_pct=exact_pct(init_p),
        final_pct=exact_pct(p),
        moved_nodes=moved_nodes,        # how many nodes left their init rank
        frac_moved=moved_nodes / n,
        n_epochs=last_i + 1,
        wall_s=wall,
        history=history,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Baseline (sigmoid Rocket) on the same graph + same init
# ──────────────────────────────────────────────────────────────────────────────
def sigmoid_rocket_pct(g: GraphData, seed: int, device, init_positions=None,
                       epochs: int = SYNTH_EPOCHS):
    init_t = None
    if init_positions is not None:
        init_t = torch.tensor(np.asarray(init_positions, dtype=np.float32),
                              device=device)
    res = run_rocket(g, RocketConfig(epochs=epochs), seed=seed, device=device,
                     init_positions=init_t)
    sc = score_from_positions(res.best_positions, np.asarray(g.src),
                              np.asarray(g.tgt), g.weight)
    assert sc == res.best_score
    return res.best_pct


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────
def run_dataset(name: str, make_graph_fn, epochs_base: int, ps_epochs: int):
    """Run the full sigmoid-vs-perturbed-sort comparison for one dataset."""
    device = torch.device("cpu")
    per_seed = []
    print(f"\n=== {name} : sigmoid Rocket vs perturbed-sort ===")
    for s in SEEDS:
        g, _ref_order, ref_pct = make_graph_fn(s)
        n = g.n_nodes
        rng = np.random.RandomState(s)
        # Common cold-start init (N(0,1)) handed to BOTH methods => fair head-to-head.
        init_pos = rng.randn(n).astype(np.float64)

        t0 = time.time()
        base_pct = sigmoid_rocket_pct(g, s, device, init_positions=init_pos,
                                      epochs=epochs_base)
        t_base = time.time() - t0

        # sweep ε grid; keep the best-by-exact-metric perturbed-sort run
        best_eps = None
        best_ps = None
        ps_by_eps = {}
        for (e0, e1) in EPS_GRID:
            r = run_perturbed_sort(g, s, init_positions=init_pos,
                                   eps_start=e0, eps_end=e1, epochs=ps_epochs)
            ps_by_eps[f"{e0}->{e1}"] = dict(best_pct=r["best_pct"],
                                            frac_moved=r["frac_moved"],
                                            final_pct=r["final_pct"],
                                            init_pct=r["init_pct"])
            if best_ps is None or r["best_pct"] > best_ps["best_pct"]:
                best_ps = r
                best_eps = (e0, e1)

        delta = best_ps["best_pct"] - base_pct
        rec = dict(
            seed=s, n_nodes=n, reference_pct=ref_pct,
            sigmoid_pct=base_pct,
            perturbsort_best_pct=best_ps["best_pct"],
            best_eps=f"{best_eps[0]}->{best_eps[1]}",
            delta_ps_minus_sigmoid=delta,
            ps_init_pct=best_ps["init_pct"],
            ps_final_pct=best_ps["final_pct"],
            ps_frac_moved=best_ps["frac_moved"],
            ps_moved_nodes=best_ps["moved_nodes"],
            ps_eps_sweep=ps_by_eps,
            t_base_s=t_base, t_ps_best_s=best_ps["wall_s"],
        )
        per_seed.append(rec)
        print(f"  seed {s}: ref={ref_pct:.4f}  sigmoid={base_pct:.4f}  "
              f"perturbsort={best_ps['best_pct']:.4f} (eps {best_eps[0]}->{best_eps[1]})  "
              f"Δ={delta:+.4f}  moved {best_ps['moved_nodes']}/{n} "
              f"({100*best_ps['frac_moved']:.1f}%)  init->final "
              f"{best_ps['init_pct']:.3f}->{best_ps['final_pct']:.3f}")

    base = np.array([r["sigmoid_pct"] for r in per_seed])
    ps = np.array([r["perturbsort_best_pct"] for r in per_seed])
    ref = np.array([r["reference_pct"] for r in per_seed])
    d = ps - base
    summary = dict(
        sigmoid_mean=float(base.mean()), sigmoid_std=float(base.std()),
        perturbsort_mean=float(ps.mean()), perturbsort_std=float(ps.std()),
        reference_mean=float(ref.mean()), reference_std=float(ref.std()),
        delta_mean=float(d.mean()), delta_std=float(d.std()),
        gap_ref_minus_sigmoid_mean=float((ref - base).mean()),
    )
    print(f"  -- sigmoid     : {summary['sigmoid_mean']:.4f} ± {summary['sigmoid_std']:.4f}")
    print(f"  -- perturbsort : {summary['perturbsort_mean']:.4f} ± {summary['perturbsort_std']:.4f}")
    print(f"  -- reference   : {summary['reference_mean']:.4f} ± {summary['reference_std']:.4f}")
    print(f"  -- Δ ps-sigmoid: {summary['delta_mean']:+.4f} ± {summary['delta_std']:.4f} pp")
    print(f"  -- gap (ref-sig): {summary['gap_ref_minus_sigmoid_mean']:+.4f} pp")
    return dict(per_seed=per_seed, summary=summary)


def verdict(synth_summary, mouse_summary=None):
    """GO iff perturbed-sort beats sigmoid Rocket BEYOND NOISE on the gap-bearing synth.

    'Beyond noise' = Δmean strictly positive AND |Δmean| > combined std (a gain whose
    overlapping std would be within-noise is NOT a gain, per CLAUDE.md invariant #4). We
    also require the optimizer to have MOVED (otherwise it stalled like H19 → NO-GO).
    """
    s = synth_summary["summary"]
    dmean, dstd = s["delta_mean"], s["delta_std"]
    moved = [r["ps_frac_moved"] for r in synth_summary["per_seed"]]
    any_move = max(moved) > 0.01      # >1% of nodes left their init rank in best run
    beyond_noise = dmean > 0 and dmean > dstd and dmean > 0.04   # 0.04pp = synth noise floor
    go = bool(beyond_noise and any_move)
    return dict(
        GO=go,
        delta_mean=dmean, delta_std=dstd,
        moved_any=bool(any_move),
        max_frac_moved=float(max(moved)),
        reason=(
            "GO: perturbed-sort exceeds sigmoid Rocket beyond noise on the gap-bearing "
            "synthetic AND the optimizer moved the order."
            if go else
            ("NO-GO / FALSIFIED: perturbed-sort did NOT beat sigmoid Rocket beyond noise"
             + (" (and the order STALLED at init, like H19)." if not any_move
                else " — the order moved but the continuous lever still could not close "
                     "the gap, confirming finding #3 (continuous gradient flow cannot "
                     "exceed the ~82.9% basin regardless of gradient source).")
             + " Continuous methods exhausted on this problem.")
        ),
    )


def main():
    out = {}
    out["config"] = dict(
        synth_params=SYNTH_PARAMS, synth_epochs=SYNTH_EPOCHS, ps_epochs=PS_EPOCHS,
        n_perturb=N_PERTURB, ps_lr=PS_LR, eps_grid=[f"{a}->{b}" for a, b in EPS_GRID],
        seeds=list(SEEDS), n_threads=2, device="cpu",
        mechanism="Berthet-2020 perturb-and-MAP SORT, Stein score-function gradient, antithetic",
    )

    out["synthetic"] = run_dataset(
        "hard_synthetic",
        lambda s: make_hard_synthetic_graph(seed=s, **SYNTH_PARAMS),
        epochs_base=SYNTH_EPOCHS, ps_epochs=PS_EPOCHS,
    )

    # Optional mouse run (148 nodes, trivial CPU cost). Mouse baseline budget = 5000.
    try:
        gm = load_mouse()
        print(f"\nmouse loaded: n={gm.n_nodes}, m={gm.n_edges}")
        out["mouse"] = run_dataset(
            "mouse",
            lambda s, _g=gm: (_g, None, np.nan),   # no diagnostic reference for mouse
            epochs_base=5000, ps_epochs=5000,
        )
    except Exception as e:   # pragma: no cover - mouse is optional
        print(f"mouse skipped: {e}")
        out["mouse"] = None

    out["verdict"] = verdict(out["synthetic"],
                             out.get("mouse"))
    print("\n=== VERDICT ===")
    print(json.dumps(out["verdict"], indent=2))

    out_path = _ROOT / "experiments" / "outputs" / "proto_h34_perturbsort.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
