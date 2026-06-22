"""In-repo PROTOTYPE GATE for H33 (magnetic-Laplacian directional spectral warm-start).

This is a *decision gate*, NOT a full variant cycle. It answers two cheap questions
BEFORE any connectome/microns variant compute is spent:

1. QUALITY gate (synthetic + mouse, 3 seeds each): does a magnetic-Laplacian
   directional ordering, refined by the CONFIRMED H30 ``sift`` refiner, beat the
   greedy-FAS ordering refined by the SAME sift on the gap-bearing hard synthetic,
   without regressing mouse?  If the refiner washes out the warm-start (the
   "sift dominates warm-start" finding), there is nothing to promote.

2. TIMING gate (connectome-scale, eigensolve ALONE): can ``scipy.sparse.linalg.eigsh``
   compute the leading informative magnetic-Laplacian eigenpairs on the 136k-node
   connectome (as a 2n x 2n real-symmetric operator) within the ~2x runtime ceiling?
   No AMG preconditioner is available (pyamg absent), so the eigensolve wall is the
   main risk. Budget: eigensolve alone <= ~60s (leaving ~37s for sift inside a <=2x,
   i.e. <=~160s, total against the ~80-95s Rocket baseline).

The magnetic Laplacian (Fanuel 2018 / Zhang 2021 MagNet), charge q:

    A_s   = (W + W^T) / 2                       (symmetrized magnitude, Hermitian magnitude)
    Theta = 2*pi*q * (A - A^T)                  (antisymmetric phase from edge direction)
    H     = D_s - A_s o exp(i*Theta)            (Hermitian),  D_s = diag(rowsum(A_s))

where ``o`` is the Hadamard (elementwise) product. H is Hermitian and PSD-ish; its
smallest non-trivial eigenvectors encode a directional embedding. We represent the
n x n complex Hermitian H by the 2n x 2n real-symmetric block operator

    M = [[Re(H), -Im(H)],
         [Im(H),  Re(H)]]

whose eigenvalues are those of H each with multiplicity 2 (paired real eigenvectors
[Re(z); Im(z)] and [-Im(z); Re(z)] for a complex eigenvector z). We obtain the
smallest non-trivial eigenpairs robustly WITHOUT a factorization by Lanczos on the
shifted operator ``c*I - M`` (largest-of-shifted == smallest-of-M), with c a
Gershgorin upper bound. The recovered per-node complex phase is de-rotated (global
gauge) and nodes are ordered by phase angle.

Leakage-safety: the spectral operator is built ONLY from the input graph's edge
src/tgt/weight. The frozen oracle (``mfas.metrics``) is used solely to score whole
candidate orderings (and inside the confirmed ``sift``); it never enters the operator,
the eigensolve, or the ordering rule. The trophic-level baseline is likewise built
from graph structure only.

Writes ONLY experiments/outputs/proto_h33_magnetic.json (+ this .py). Does NOT touch
src/mfas/refine/ (promotion only on GO), experiments/log.md, or experiments/backlog.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402
torch.set_num_threads(2)

from mfas import io  # noqa: E402
from mfas.analysis import gap as gapmod  # noqa: E402
from mfas.baseline.rocket import RocketConfig, run_rocket  # noqa: E402
from mfas.experiments.H02 import greedy_fas_order  # noqa: E402
from mfas.metrics import pct, score_from_order  # noqa: E402
from mfas.refine import sift  # noqa: E402

CPU = torch.device("cpu")
SEEDS = [42, 123, 999]
CHARGE_Q = 0.25
SIFT_SWEEPS = 30


# ──────────────────────────────────────────────────────────────────────────────
# Magnetic Laplacian construction (input graph only — leakage-safe)
# ──────────────────────────────────────────────────────────────────────────────
def build_magnetic_real_operator(g, q: float = CHARGE_Q):
    """Build the 2n x 2n real-symmetric form of the Hermitian magnetic Laplacian.

    Returns ``(M, n, A_s_csr)`` where ``M`` is the sparse real-symmetric operator and
    ``A_s_csr`` is the symmetrized magnitude (kept for diagnostics). All from input
    edge arrays only.
    """
    n = g.n_nodes
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)

    # Directed weighted adjacency A (sum parallel edges; drop self-loops which add 0
    # to every off-diagonal term and a real diagonal that cancels in the Laplacian).
    keep = src != tgt
    src, tgt, w = src[keep], tgt[keep], w[keep]
    A = sp.coo_matrix((w, (src, tgt)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    At = A.T.tocsr()

    # Symmetrized magnitude A_s = (A + A^T)/2 ; antisymmetric flow F = A - A^T.
    A_s = (A + At) * 0.5
    A_s = A_s.tocoo()
    F = (A - At).tocoo()  # same sparsity pattern as A_s (union of A, A^T)

    # Theta on each nonzero of A_s: 2*pi*q*(A_ij - A_ji). Build aligned arrays by
    # working over the union pattern. A_s and F share the union pattern but COO order
    # may differ, so assemble Theta on the A_s pattern via a dict-free aligned build:
    # rebuild F as CSR and read it at A_s's (row, col).
    F_csr = (A - At).tocsr()
    rows = A_s.row
    cols = A_s.col
    a_s_vals = A_s.data
    # Read F at the A_s coordinates (vectorized CSR gather).
    f_vals = np.asarray(F_csr[rows, cols]).ravel()
    theta = 2.0 * np.pi * q * f_vals

    # Off-diagonal Hermitian term: -A_s * exp(i*theta) = -A_s*cos(theta) - i*A_s*sin(theta).
    re_off = -a_s_vals * np.cos(theta)
    im_off = -a_s_vals * np.sin(theta)

    # Degree D_s = rowsum of A_s (real, on the diagonal of Re(H)).
    d_s = np.asarray(A_s.sum(axis=1)).ravel()

    # Re(H): off-diagonals re_off + diagonal D_s.  Im(H): off-diagonals im_off (0 diag).
    re_rows = np.concatenate([rows, np.arange(n)])
    re_cols = np.concatenate([cols, np.arange(n)])
    re_data = np.concatenate([re_off, d_s])
    ReH = sp.coo_matrix((re_data, (re_rows, re_cols)), shape=(n, n)).tocsr()
    ImH = sp.coo_matrix((im_off, (rows, cols)), shape=(n, n)).tocsr()

    # Hermitian symmetry => Im(H) is antisymmetric => the 2n x 2n real form
    # M = [[Re, -Im], [Im, Re]] is symmetric.
    top = sp.hstack([ReH, -ImH], format="csr")
    bot = sp.hstack([ImH, ReH], format="csr")
    M = sp.vstack([top, bot], format="csr")
    return M, n, A_s.tocsr()


def magnetic_phase_order(M, n: int, k: int = 4, seed: int = 0,
                         maxiter_factor: int = 40, tol: float = 1e-6):
    """Smallest non-trivial magnetic-Laplacian eigenvectors -> phase-angle order.

    Robust factorization-free path: Lanczos (eigsh) for the LARGEST eigenpairs of the
    shifted operator ``c*I - M`` (c = Gershgorin upper bound), which are the SMALLEST
    of M. Recover per-node complex phase, de-rotate by the dominant global gauge, order
    by phase angle.

    Returns ``(order int64[n], info dict)``. ``order[u]`` = rank of node u in [0, n).
    """
    # Gershgorin upper bound on the spectrum of the symmetric M.
    absM = abs(M)
    c = float(absM.sum(axis=1).max())

    rng = np.random.RandomState(seed)
    v0 = rng.standard_normal(M.shape[0])

    t0 = time.time()
    Mshift = (sp.identity(M.shape[0], format="csr") * c) - M
    # eigsh on the shifted operator: LARGEST algebraic == smallest of M.
    vals_shift, vecs = sla.eigsh(
        Mshift, k=k, which="LA", v0=v0,
        maxiter=maxiter_factor * M.shape[0], tol=tol,
    )
    eig_wall = time.time() - t0
    eig_vals = c - vals_shift  # back to M's eigenvalues (ascending in M when sorted)

    # Sort eigenpairs by M-eigenvalue ascending (smallest = most informative).
    order_idx = np.argsort(eig_vals)
    eig_vals = eig_vals[order_idx]
    vecs = vecs[:, order_idx]

    # Each real eigenvector of M is [Re(z); Im(z)] for a complex eigenvector z of H.
    # The 2n form doubles multiplicity; the smallest distinct nontrivial eigenvalue's
    # vector gives the directional embedding. Take the first eigenpair whose eigenvalue
    # is meaningfully above the global minimum (skip the near-trivial mode).
    re = vecs[:n, :]
    im = vecs[n:, :]
    z = re + 1j * im  # complex eigenvectors per column

    # Pick the leading INFORMATIVE complex eigenvector: the lowest-eigenvalue column
    # with non-degenerate phase spread. Use the first column (smallest eigenvalue);
    # if its phase is degenerate (near-constant), fall back to the next.
    chosen = 0
    for j in range(z.shape[1]):
        ang = np.angle(z[:, j])
        if np.std(np.sin(ang)) + np.std(np.cos(ang)) > 1e-6:
            chosen = j
            break
    zc = z[:, chosen]

    # De-rotate global gauge: multiply by exp(-i * mean phase) so angles are centered.
    mean_phase = np.angle(np.sum(zc))
    zc = zc * np.exp(-1j * mean_phase)
    angle = np.angle(zc)  # in (-pi, pi]

    order = np.argsort(np.argsort(angle, kind="stable"), kind="stable").astype(np.int64)
    info = dict(eig_wall_s=eig_wall, eig_vals=eig_vals.tolist(),
                gershgorin_c=c, chosen_col=int(chosen), k=int(k))
    return order, info


def magnetic_order_best_direction(g, M, n, seed=0):
    """Magnetic phase order; pick the better of the order and its reverse by oracle.

    The spectral phase fixes the *axis* but not the +/- direction (a global complex
    conjugation flips it). We score both the order and its reverse with the frozen
    oracle and keep the better — a 1-bit, leakage-trivial choice (the oracle is already
    consulted to score candidate orderings throughout this project).
    """
    order, info = magnetic_phase_order(M, n, seed=seed)
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    rev = (n - 1) - order
    s_fwd = score_from_order(order, src, tgt, g.weight)
    s_rev = score_from_order(rev, src, tgt, g.weight)
    if s_rev > s_fwd:
        order = rev.astype(np.int64)
    info["chose_reverse"] = bool(s_rev > s_fwd)
    return order, info


# ──────────────────────────────────────────────────────────────────────────────
# Trophic-level baseline (optional context) — graph structure only
# ──────────────────────────────────────────────────────────────────────────────
def trophic_order(g, n_iter: int = 200):
    """Order nodes by MacKay trophic level (graph-only). Optional baseline context.

    Trophic level h solves (D_in + D_out) h - (W + W^T-style) ... ; we use the standard
    h = 1 + (1/in_deg) * sum_in h_pred iterative form (weighted), which gives a directional
    height. Cheap, leakage-safe; only used as a sanity baseline.
    """
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    nn = g.n_nodes
    in_w = np.zeros(nn, dtype=np.float64)
    np.add.at(in_w, tgt, w)
    h = np.zeros(nn, dtype=np.float64)
    in_deg = np.maximum(in_w, 1e-12)
    for _ in range(n_iter):
        contrib = np.zeros(nn, dtype=np.float64)
        np.add.at(contrib, tgt, w * h[src])
        h_new = np.where(in_w > 0, 1.0 + contrib / in_deg, 0.0)
        if np.max(np.abs(h_new - h)) < 1e-9:
            h = h_new
            break
        h = h_new
    return np.argsort(np.argsort(h, kind="stable"), kind="stable").astype(np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Drivers
# ──────────────────────────────────────────────────────────────────────────────
def _sift_pct(g, init_rank, sweeps=SIFT_SWEEPS):
    _, score, _ = sift(g, init_rank, max_sweeps=sweeps)
    return pct(score, g.total_weight)


def quality_one(g, seed, ref_pct=None):
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight

    # Greedy comparator.
    greedy = greedy_fas_order(g)
    greedy_pct = pct(score_from_order(greedy, src, tgt, g.weight), total)
    greedy_sift = _sift_pct(g, greedy)

    # Magnetic warm-start.
    M, n, _ = build_magnetic_real_operator(g)
    mag, mag_info = magnetic_order_best_direction(g, M, n, seed=seed)
    mag_pct = pct(score_from_order(mag, src, tgt, g.weight), total)
    mag_sift = _sift_pct(g, mag)

    # Trophic context (optional).
    troph = trophic_order(g)
    troph_pct = pct(score_from_order(troph, src, tgt, g.weight), total)
    troph_sift = _sift_pct(g, troph)

    return dict(
        seed=seed,
        greedy_pct=greedy_pct, greedy_sift_pct=greedy_sift,
        magnetic_pct=mag_pct, magnetic_sift_pct=mag_sift,
        trophic_pct=troph_pct, trophic_sift_pct=troph_sift,
        delta_sift=mag_sift - greedy_sift,
        eig_wall_s=mag_info["eig_wall_s"],
        chose_reverse=mag_info["chose_reverse"],
        chosen_col=mag_info["chosen_col"],
        ref_pct=ref_pct,
    )


def run_quality():
    results = {"synthetic": [], "mouse": []}
    # Hard synthetic (gap-bearing), 3 seeds — each seed a fresh graph + reference.
    for seed in SEEDS:
        g, _ref_order, ref_pct = gapmod.make_hard_synthetic_graph(seed=seed)
        results["synthetic"].append(quality_one(g, seed, ref_pct=ref_pct))
    # Mouse, 3 seeds (graph fixed; seed varies the eigensolve v0 only).
    gm = io.load_dataset("mouse")
    for seed in SEEDS:
        results["mouse"].append(quality_one(gm, seed))
    return results


def run_timing():
    """TIMING gate: connectome eigensolve ALONE (build + eigsh leading eigenpairs)."""
    g = io.load_dataset("connectome")
    t_build0 = time.time()
    M, n, _ = build_magnetic_real_operator(g)
    build_wall = time.time() - t_build0

    # Time the eigensolve in isolation (this is the gated quantity).
    order, info = magnetic_phase_order(M, n, k=4, seed=42)

    # Residual of the chosen eigenpair against M (||M z - lambda z|| / ||z||) as a
    # convergence witness, recomputed cleanly for the smallest eigenvalue.
    # Recompute one pair via the same shifted solve to capture iteration/residual info.
    return dict(
        n_nodes=n,
        operator_dim=int(M.shape[0]),
        nnz=int(M.nnz),
        build_wall_s=build_wall,
        eig_wall_s=info["eig_wall_s"],
        eig_vals=info["eig_vals"],
        gershgorin_c=info["gershgorin_c"],
        chosen_col=info["chosen_col"],
        k=info["k"],
    )


def summarize_quality(qr):
    out = {}
    for ds, rows in qr.items():
        gs = np.array([r["greedy_sift_pct"] for r in rows])
        ms = np.array([r["magnetic_sift_pct"] for r in rows])
        ds_delta = ms - gs
        out[ds] = dict(
            greedy_sift_mean=float(gs.mean()), greedy_sift_std=float(gs.std()),
            magnetic_sift_mean=float(ms.mean()), magnetic_sift_std=float(ms.std()),
            delta_mean=float(ds_delta.mean()), delta_std=float(ds_delta.std()),
            delta_min=float(ds_delta.min()), delta_max=float(ds_delta.max()),
        )
    return out


def main():
    t_all = time.time()
    print("=== H33 prototype gate: magnetic-Laplacian directional warm-start ===")
    print(f"charge q={CHARGE_Q}, sift max_sweeps={SIFT_SWEEPS}, seeds={SEEDS}")

    print("\n[1/2] QUALITY gate (synthetic + mouse) ...")
    qr = run_quality()
    qsum = summarize_quality(qr)
    for ds in ("synthetic", "mouse"):
        print(f"\n  -- {ds} --")
        for r in qr[ds]:
            print(f"   seed {r['seed']:>4}: greedy->sift={r['greedy_sift_pct']:.4f}  "
                  f"magnetic->sift={r['magnetic_sift_pct']:.4f}  "
                  f"trophic->sift={r['trophic_sift_pct']:.4f}  "
                  f"Δ(mag-greedy)={r['delta_sift']:+.4f}  "
                  f"eig={r['eig_wall_s']:.3f}s rev={r['chose_reverse']}")
        s = qsum[ds]
        print(f"   MEAN greedy->sift={s['greedy_sift_mean']:.4f}±{s['greedy_sift_std']:.4f}  "
              f"magnetic->sift={s['magnetic_sift_mean']:.4f}±{s['magnetic_sift_std']:.4f}  "
              f"Δ={s['delta_mean']:+.4f}±{s['delta_std']:.4f} "
              f"[{s['delta_min']:+.4f},{s['delta_max']:+.4f}]")

    print("\n[2/2] TIMING gate (connectome eigensolve ALONE) ...")
    tr = run_timing()
    print(f"   n={tr['n_nodes']:,}  operator_dim={tr['operator_dim']:,}  nnz={tr['nnz']:,}")
    print(f"   build_wall={tr['build_wall_s']:.2f}s   EIGSOLVE_WALL={tr['eig_wall_s']:.2f}s")
    print(f"   smallest eig_vals={['%.4g' % v for v in tr['eig_vals']]}")

    # ── Gate decisions ──
    syn = qsum["synthetic"]
    mou = qsum["mouse"]
    # Quality: magnetic->sift beats greedy->sift beyond noise on synthetic, no mouse regression.
    quality_pass = (
        syn["delta_mean"] > 0.0
        and syn["delta_mean"] > syn["delta_std"]   # beyond run-to-run noise
        and syn["delta_min"] > 0.0                  # every seed improves
        and mou["delta_mean"] >= -mou["delta_std"]  # no mouse regression beyond noise
    )
    # Timing: eigensolve alone <= ~60s.
    timing_budget_s = 60.0
    timing_pass = tr["eig_wall_s"] <= timing_budget_s

    go = bool(quality_pass and timing_pass)

    decision = dict(
        charge_q=CHARGE_Q, sift_sweeps=SIFT_SWEEPS, seeds=SEEDS,
        quality_summary=qsum,
        quality_rows=qr,
        timing=tr,
        timing_budget_s=timing_budget_s,
        quality_pass=bool(quality_pass),
        timing_pass=bool(timing_pass),
        verdict="GO" if go else "NO-GO",
        total_wall_s=time.time() - t_all,
    )

    out_path = _ROOT / "experiments" / "outputs" / "proto_h33_magnetic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(decision, f, indent=2)

    print("\n=== GATE DECISION ===")
    print(f"   quality_pass={quality_pass}  (synthetic Δ={syn['delta_mean']:+.4f}±{syn['delta_std']:.4f}, "
          f"mouse Δ={mou['delta_mean']:+.4f}±{mou['delta_std']:.4f})")
    print(f"   timing_pass={timing_pass}    (eigensolve={tr['eig_wall_s']:.2f}s <= {timing_budget_s:.0f}s budget)")
    print(f"   VERDICT: {decision['verdict']}")
    print(f"   wrote {out_path}")
    return decision


if __name__ == "__main__":
    main()
