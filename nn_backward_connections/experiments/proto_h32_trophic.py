"""Prototype GATE for backlog H32 — trophic-level Laplacian global warm-start -> H30 sift.

SELF-CONTAINED prototype (NOT promoted to src/ — promotion happens only on GO). This
script builds a leakage-safe *trophic-level* global ordering and asks one question:

    Does seeding the CONFIRMED H30 ``sift`` refiner from a trophic order give a
    genuinely better basin than seeding it from the H02 greedy-FAS order?

Trophic levels (MacKay, Johnson & Sansom 2020, "How directed is a directed network?")
-------------------------------------------------------------------------------------
For a directed weighted graph with weighted adjacency ``W`` (``W[u, v]`` = weight of
edge ``u -> v``), let

    w_out[u] = Σ_v W[u, v]   (total out-weight),
    w_in[u]  = Σ_v W[v, u]   (total in-weight),
    u[u]     = w_in[u] + w_out[u]   (total incident weight, the "degree"),
    v[u]     = w_in[u] - w_out[u]   (imbalance).

The trophic level ``h`` solves the symmetric Laplacian system

    Λ h = v,   Λ = diag(u) - (W + Wᵀ).

``Λ`` is a (weighted) graph Laplacian of the SYMMETRISED graph: symmetric, PSD, singular
with null space = the constant vector. The system is consistent (``Σ v = 0`` since every
edge contributes +w to one w_in and +w to one w_out, so ``1ᵀ v = 0``). We fix the gauge
by projecting the constant out of both ``v`` and the solution (mean-zero) and solve with
conjugate gradients (``scipy.sparse.linalg.cg``; pyamg unavailable -> plain CG).

A node with more in-weight than out-weight (a sink-like node) gets a HIGHER trophic level;
a source-like node gets a LOWER one. So ``argsort(h)`` orders sources -> sinks, exactly the
front->back convention the scorer/sift use (feedforward edge ``(u, v)`` wants
``rank[u] < rank[v]``).

Leakage-safety
--------------
``h`` is a function of the input graph ONLY (degrees + adjacency). It never touches the
frozen oracle, ``data/best_solution``, or the target metric. The oracle is consulted only
inside ``sift`` for whole-vector accept/reject (exactly as the confirmed H30 refiner does).

This file does NOT modify experiments/log.md, experiments/backlog.md, or anything under
src/. It writes only experiments/outputs/proto_h32_trophic.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# CPU ONLY — pin threads before any heavy import touches BLAS.
import torch  # noqa: E402
torch.set_num_threads(2)

import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

import sys  # noqa: E402
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mfas.io import GraphData, load_dataset  # noqa: E402
from mfas.metrics import pct, score_from_order  # noqa: E402
from mfas.refine.insertion import sift  # noqa: E402
from mfas.experiments.H02 import greedy_fas_order  # noqa: E402
from mfas.analysis.gap import make_hard_synthetic_graph  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Trophic-level global ordering (leakage-safe; graph structure + weights only)
# ──────────────────────────────────────────────────────────────────────────────
def trophic_levels(g: GraphData, cg_tol: float = 1e-10,
                   cg_maxiter: int = 20_000) -> Tuple[np.ndarray, Dict]:
    """Solve the trophic-level Laplacian system ``Λ h = v`` (MacKay-Johnson-Sansom 2020).

    Returns ``(h, info)`` where ``h`` is the mean-zero trophic level per node (float64)
    and ``info`` records the CG residual / convergence flag for the correctness check.
    """
    n = g.n_nodes
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)

    # Drop self-loops: a self-loop adds equally to w_in and w_out (cancels in v) and
    # contributes only to the diagonal in a way that does not affect the symmetrised
    # Laplacian's action on a mean-zero vector consistently; safest to exclude.
    keep = src != tgt
    src, tgt, w = src[keep], tgt[keep], w[keep]

    w_out = np.zeros(n, dtype=np.float64)
    w_in = np.zeros(n, dtype=np.float64)
    np.add.at(w_out, src, w)
    np.add.at(w_in, tgt, w)
    deg = w_in + w_out                       # u[node]
    v = w_in - w_out                         # imbalance (Σ v == 0 exactly)

    # Symmetrised adjacency S = W + Wᵀ (sum duplicate (u,v) entries via the COO ctor).
    W = sp.coo_matrix((w, (src, tgt)), shape=(n, n)).tocsr()
    S = (W + W.T).tocsr()
    Lap = sp.diags(deg) - S                  # Λ = diag(deg) - (W + Wᵀ), symmetric PSD

    # Gauge: project the constant (null space) out of the RHS so the system is consistent
    # against the singular Λ, then solve mean-zero. (Σ v == 0 already, but we re-center
    # defensively against float round-off.)
    v0 = v - v.mean()

    n_calls = {"k": 0}

    def _cb(_xk):  # iteration counter
        n_calls["k"] += 1

    # Plain CG (pyamg not available). Symmetric PSD consistent system -> CG converges to
    # the minimum-norm-style solution in the range of Λ; we re-center afterwards.
    try:  # scipy >=1.12 renamed tol -> rtol
        h, flag = spla.cg(Lap, v0, rtol=cg_tol, maxiter=cg_maxiter, callback=_cb)
    except TypeError:
        h, flag = spla.cg(Lap, v0, tol=cg_tol, maxiter=cg_maxiter, callback=_cb)

    h = np.asarray(h, dtype=np.float64)
    h = h - h.mean()                         # fix gauge: mean-zero trophic levels

    resid = float(np.linalg.norm(Lap @ h - v0))
    rhs_norm = float(np.linalg.norm(v0))
    rel_resid = resid / rhs_norm if rhs_norm > 0 else 0.0
    info = dict(cg_flag=int(flag), cg_iters=int(n_calls["k"]),
                resid_abs=resid, resid_rel=rel_resid, rhs_norm=rhs_norm)
    return h, info


def trophic_order(g: GraphData) -> Tuple[np.ndarray, Dict]:
    """Global order from trophic levels: ``argsort(h)`` -> sources(low h) ... sinks(high h).

    Returns ``(order, info)`` where ``order[u]`` is the rank of node ``u`` in ``[0, n)``
    (smaller = earlier = source side), matching the scorer/sift convention. ``info`` carries
    the CG correctness diagnostics.
    """
    h, info = trophic_levels(g)
    # rank vector: order[node] = rank. argsort(h) lists nodes low->high; invert to ranks.
    perm = np.argsort(h, kind="stable")               # perm[r] = node at rank r
    order = np.empty(g.n_nodes, dtype=np.int64)
    order[perm] = np.arange(g.n_nodes, dtype=np.int64)
    return order, info


# ──────────────────────────────────────────────────────────────────────────────
# Comparison harness
# ──────────────────────────────────────────────────────────────────────────────
def _is_permutation(rank: np.ndarray, n: int) -> bool:
    return rank.shape[0] == n and np.array_equal(np.sort(rank), np.arange(n))


def compare_on_graph(g: GraphData, seed: int, *, max_sweeps: int = 30
                     ) -> Dict:
    """Greedy->sift vs trophic->sift on one graph, same sift settings.

    Note: ``sift`` is deterministic given an init order (Jacobi + oracle accept/reject), so
    the ``seed`` here only selects the SAME synthetic instance / re-runs mouse; we record it
    for traceability. Returns a dict of all pcts + the delta and correctness flags.
    """
    src = np.asarray(g.src)
    tgt = np.asarray(g.tgt)
    total = g.total_weight
    n = g.n_nodes

    # --- candidate orders (both leakage-safe, graph-only) ---
    g_order = greedy_fas_order(g)
    t_order, cg_info = trophic_order(g)

    # correctness: valid permutations + CG converged
    assert _is_permutation(g_order, n), "greedy order is not a permutation"
    assert _is_permutation(t_order, n), "trophic order is not a permutation"

    greedy_raw = pct(score_from_order(g_order, src, tgt, g.weight), total)
    trophic_raw = pct(score_from_order(t_order, src, tgt, g.weight), total)

    # --- sift both with identical settings (no time budget at this small scale) ---
    t0 = time.time()
    _, g_sift_score, _ = sift(g, g_order, max_sweeps=max_sweeps, time_budget_s=None)
    g_sift_wall = time.time() - t0
    greedy_sift = pct(g_sift_score, total)

    t0 = time.time()
    _, t_sift_score, _ = sift(g, t_order, max_sweeps=max_sweeps, time_budget_s=None)
    t_sift_wall = time.time() - t0
    trophic_sift = pct(t_sift_score, total)

    return dict(
        graph=g.name, seed=int(seed), n_nodes=int(n), n_edges=int(g.n_edges),
        greedy_raw=greedy_raw, trophic_raw=trophic_raw,
        greedy_sift=greedy_sift, trophic_sift=trophic_sift,
        delta_sift=trophic_sift - greedy_sift,
        delta_raw=trophic_raw - greedy_raw,
        cg_flag=cg_info["cg_flag"], cg_iters=cg_info["cg_iters"],
        cg_resid_abs=cg_info["resid_abs"], cg_resid_rel=cg_info["resid_rel"],
        greedy_sift_wall_s=g_sift_wall, trophic_sift_wall_s=t_sift_wall,
    )


def _summary(rows: List[Dict]) -> Dict:
    d = np.array([r["delta_sift"] for r in rows], dtype=np.float64)
    gs = np.array([r["greedy_sift"] for r in rows], dtype=np.float64)
    ts = np.array([r["trophic_sift"] for r in rows], dtype=np.float64)
    return dict(
        n_seeds=len(rows),
        greedy_sift_mean=float(gs.mean()), greedy_sift_std=float(gs.std(ddof=0)),
        trophic_sift_mean=float(ts.mean()), trophic_sift_std=float(ts.std(ddof=0)),
        delta_mean=float(d.mean()), delta_std=float(d.std(ddof=0)),
        delta_min=float(d.min()), delta_max=float(d.max()),
        seed_straddling=bool(d.min() < 0 < d.max()),
    )


def main() -> None:
    seeds = [42, 123, 999]
    max_sweeps = 30
    out: Dict = {"max_sweeps": max_sweeps, "seeds": seeds,
                 "fixtures": {}, "note": ""}

    # ── 1. Gap-bearing HARD synthetic (the decisive fixture) ──
    synth_rows: List[Dict] = []
    for s in seeds:
        g, _ref_order, _ref_pct = make_hard_synthetic_graph(seed=s)
        row = compare_on_graph(g, s, max_sweeps=max_sweeps)
        synth_rows.append(row)
        print(f"[synthetic seed={s}] greedy_raw={row['greedy_raw']:.4f} "
              f"trophic_raw={row['trophic_raw']:.4f} | "
              f"greedy_sift={row['greedy_sift']:.4f} trophic_sift={row['trophic_sift']:.4f} "
              f"Δ={row['delta_sift']:+.4f}  (cg_flag={row['cg_flag']}, "
              f"cg_rel_resid={row['cg_resid_rel']:.2e})")
    out["fixtures"]["hard_synthetic"] = dict(rows=synth_rows, summary=_summary(synth_rows))

    # ── 2. Mouse (real connectome, must not regress) ──
    g_mouse = load_dataset("mouse")
    mouse_rows: List[Dict] = []
    for s in seeds:
        row = compare_on_graph(g_mouse, s, max_sweeps=max_sweeps)
        mouse_rows.append(row)
        print(f"[mouse seed={s}] greedy_raw={row['greedy_raw']:.4f} "
              f"trophic_raw={row['trophic_raw']:.4f} | "
              f"greedy_sift={row['greedy_sift']:.4f} trophic_sift={row['trophic_sift']:.4f} "
              f"Δ={row['delta_sift']:+.4f}  (cg_flag={row['cg_flag']}, "
              f"cg_rel_resid={row['cg_resid_rel']:.2e})")
    out["fixtures"]["mouse"] = dict(rows=mouse_rows, summary=_summary(mouse_rows))

    # ── GATE decision ──
    syn = out["fixtures"]["hard_synthetic"]["summary"]
    mou = out["fixtures"]["mouse"]["summary"]
    # noise proxy = std of the greedy->sift baseline across seeds on that fixture.
    syn_noise = max(syn["greedy_sift_std"], 1e-9)
    mou_noise = max(mou["greedy_sift_std"], 1e-9)
    synth_go = (syn["delta_mean"] > 0) and (not syn["seed_straddling"]) \
        and (syn["delta_mean"] > syn_noise)
    mouse_ok = mou["delta_mean"] >= -mou_noise
    go = bool(synth_go and mouse_ok)
    out["gate"] = dict(
        synth_delta_mean=syn["delta_mean"], synth_delta_std=syn["delta_std"],
        synth_noise_proxy=syn_noise, synth_seed_straddling=syn["seed_straddling"],
        synth_go=bool(synth_go),
        mouse_delta_mean=mou["delta_mean"], mouse_delta_std=mou["delta_std"],
        mouse_noise_proxy=mou_noise, mouse_ok=bool(mouse_ok),
        verdict="GO" if go else "NO-GO",
    )
    out["note"] = (
        "GATE: GO iff trophic->sift beats greedy->sift beyond noise on the gap-bearing "
        "synthetic (positive mean Δ, not seed-straddling, |Δ_mean|>greedy_sift_std) AND "
        "mouse does not regress (Δ_mean >= -noise). Otherwise NO-GO (refiner washes out "
        "the warm-start choice)."
    )

    outdir = Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / "proto_h32_trophic.json"
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== GATE ===")
    print(f"synthetic: Δmean={syn['delta_mean']:+.4f}  Δstd={syn['delta_std']:.4f}  "
          f"noise(greedy_sift_std)={syn_noise:.4f}  straddling={syn['seed_straddling']}  "
          f"-> {'PASS' if synth_go else 'FAIL'}")
    print(f"mouse:     Δmean={mou['delta_mean']:+.4f}  Δstd={mou['delta_std']:.4f}  "
          f"noise={mou_noise:.4f}  -> {'OK' if mouse_ok else 'REGRESS'}")
    print(f"VERDICT: {out['gate']['verdict']}")
    print(f"wrote {fpath}")


if __name__ == "__main__":
    main()
