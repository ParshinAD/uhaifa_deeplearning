"""Prototype GATE for roadmap A-INIT (TODO 5) — VERY TIGHT position initialization.

SELF-CONTAINED prototype (NOT promoted to ``src/``). It answers one question in two
halves, as requested:

    (I)  standalone: what does a very tight init (std -> 0) DO to Rocket, mechanically,
         and does it help or hurt the exact feedforward metric?
    (II) in combination with our confirmed findings: does it interact with the H02
         greedy-FAS warm start (#1) and with the H30/H35 discrete sift (#4/#5)?

================================================================================
THEORY — why the scale of the init is not a free knob
================================================================================

Rocket maximizes the surrogate (``mfas.baseline.rocket``; w-hat = w / max w)

    F(P) = sum_e  w_hat_e * sigma(beta * D_e),      D_e = p_tgt(e) - p_src(e).

Expand sigma around 0 (radius of convergence |x| < pi):

    sigma(x) = 1/2 + x/4 - x^3/48 + O(x^5).

Substituting and collecting terms per node, with the **weight imbalance**

    c_k := out_w_hat(k) - in_w_hat(k)        (source-like: c_k > 0),

and using  sum_e w_hat_e * D_e = sum_k p_k * (in_w_hat(k) - out_w_hat(k)) = -<c, P>:

    F(P) = W_hat/2  -  (beta/4) <c, P>  -  (beta^3/48) sum_e w_hat_e D_e^3  +  O((beta*D)^5)
           ^^^^^^^     ^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           constant    NODE-LOCAL term      the first term that couples node PAIRS

    grad_k F = -(beta/4) c_k  +  O((beta*std)^2 * c-scale).

Three consequences, each of which this script measures:

**(T1) Pairwise information is suppressed as (beta*std)^2.**  All the ordering
information — which node must precede which — lives in the cubic and higher terms. At
std = 1e-4 and beta ~ 1 the pairwise part of the gradient is ~1e-8 of the node-local
part. The gradient is then a *constant vector* -(beta/4)c that does not depend on the
positions at all. This is the same scale-blindness Q01 found from the other end
(`diagnosis.md` Q01: F is a function of beta*std only, and at std -> 0, F -> W_hat/2).

**(T2) Adam discards |c| and keeps only sign(c).**  For a (near-)constant gradient the
Adam update is  -lr * m_hat / (sqrt(v_hat) + eps) -> -lr * sign(g_k)  after warm-up. So
every node translates at the SAME speed +-lr regardless of how imbalanced it is: the
configuration splits into exactly two rigid blocks, {c_k > 0} moving front and
{c_k < 0} moving back, with the within-block order frozen at whatever the init gave.
The magnitude of the imbalance — the actual GreedyAbs signal — is thrown away.

**(T3) The dynamics become a recursive bisection, not an optimization.**  Once the two
blocks are separated by more than ~1/beta, the edges BETWEEN them saturate
(sigma' -> 0) and drop out of the gradient. What remains inside each block is the same
linear problem on the induced subgraph, again at a tiny internal scale — so the block
splits again by its own internal imbalance sign, and so on. Tight-init Rocket is
therefore an approximate **recursive weight-imbalance bisection**, whose quality is
bounded by that of a degree-difference ordering — a family already known here to be
weak (diagnosis Step 4: init quality 36-69%, plateau flat within 0.06 pp).

Predictions this script tests (each can falsify the story):

    P1  cos(grad F, -c) -> 1 as std -> 0; the residual scales as (beta*std)^2.
    P2  from a tight init the position cloud splits into ~2 blocks aligned with sign(c).
    P3  a tight init does NOT beat the N(0,1) baseline on the exact metric  [the kill]
    P4  compressing the H02 greedy warm start to a tiny scale DESTROYS part of its
        advantage (the sign(c) split reorders every pair that disagrees with sign(c)).
    P5  after the H35 sift, the differences between inits collapse (finding-#4/#5
        cross-cutting insight: the discrete refiner does the work).

Leakage-safety: everything here is a function of the input graph only (weights,
degrees, positions). ``data/best_solution`` is never read; the hard-synthetic reference
order is used ONLY as a printed diagnostic ceiling, never inside any run. The frozen
oracle (``mfas.metrics``) is used exactly as the baseline uses it — to score a finished
position vector.

Writes: ``experiments/outputs/proto_ainit_scale.json`` (+ two PNGs). Nothing under
``results/`` (that namespace belongs to the frozen runner).

Run (env `allen`, from the repo root):
    PYTHONPATH=src python experiments/proto_ainit_scale.py
    PYTHONPATH=src python experiments/proto_ainit_scale.py --stage theory   # single stage
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from mfas.analysis.gap import make_hard_synthetic_graph
from mfas.baseline.rocket import RocketConfig, run_rocket
from mfas.experiments.H02 import _init_positions_from_order, greedy_fas_order
from mfas.io import load_dataset
from mfas.metrics import pct, score_from_positions
from mfas.refine import sift_underrelaxed

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "experiments" / "outputs"

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — every knob lives here (CLAUDE.md: no magic numbers)
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "seed": 42,
    "seeds": [42, 123, 999],            # >= 3 seeds for every scored comparison
    "device": "cpu",                    # proxies are small; CPU keeps this deterministic

    # Proxy graphs. mouse = a real connectome that runs in seconds; hard_synthetic =
    # the purpose-built fixture that carries a verified optimization gap (H21).
    "datasets": ["mouse", "hard_synthetic"],
    "synth": {"n": 400, "avg_out": 10, "feedback_frac": 0.65, "n_clusters": 8, "seed": 0},

    # Init scales to sweep. 1.0 is the paper/baseline default (N(0,1)); 0.577 is the
    # std of H02's evenly-spaced [-1,1] warm start.
    "init_stds": [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0],
    "baseline_std": 1.0,

    # Gradient budget per run: the baseline single-run budget for mouse.
    "epochs": {"mouse": 5_000, "hard_synthetic": 5_000},

    # Stage 1 (theory): betas span the cyclic schedule's range [0.05, 1.05].
    "theory_betas": [0.05, 1.05],
    "theory_stds": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],

    # Stage 2 (mechanism probe): constant-beta Adam, mirrors Rocket's update but with a
    # fixed beta so the split is not confounded by the cyclic schedule.
    "cascade": {"beta": 1.05, "lr": 0.05, "steps": 400, "snapshot_every": 20,
                "init_std": 1e-4, "gap_factor": 10.0},

    # Stage 4 (combined): scales applied to the H02 greedy warm-start positions.
    "h02_stds": [1e-4, 1e-2, 0.577, 5.0],

    # H35 refiner settings for the proxies (same constants as src/mfas/experiments/H35).
    "sift": {"k_full": 6, "alpha": 0.7, "max_sweeps": 40},
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def git_commit() -> str:
    """Current HEAD, for provenance in the JSON artifact."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_ROOT, text=True).strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


def load_graph(name: str):
    """Load a proxy graph. Returns ``(g, reference_pct or None)``."""
    if name == "hard_synthetic":
        s = CONFIG["synth"]
        g, _ref_order, ref_pct = make_hard_synthetic_graph(
            n=s["n"], avg_out=s["avg_out"], feedback_frac=s["feedback_frac"],
            n_clusters=s["n_clusters"], seed=s["seed"])
        return g, ref_pct
    return load_dataset(name), None


def imbalance(g) -> np.ndarray:
    """Node weight imbalance ``c_k = out_w_hat(k) - in_w_hat(k)`` (normalized weights).

    This is the vector the theory predicts the small-scale gradient is proportional to,
    and it is exactly the quantity GreedyAbs ranks on (``out_w - in_w`` -> front).
    """
    w = np.asarray(g.weight, dtype=np.float64) / float(np.asarray(g.weight).max())
    out_w = np.zeros(g.n_nodes, dtype=np.float64)
    in_w = np.zeros(g.n_nodes, dtype=np.float64)
    np.add.at(out_w, np.asarray(g.src), w)
    np.add.at(in_w, np.asarray(g.tgt), w)
    return out_w - in_w


def surrogate_grad(g, pos: np.ndarray, beta: float) -> np.ndarray:
    """Exact autograd gradient of the surrogate F at ``pos`` (float64, CPU)."""
    w = np.asarray(g.weight, dtype=np.float64)
    nw = torch.tensor(w / w.max(), dtype=torch.float64)
    src = torch.tensor(np.asarray(g.src), dtype=torch.long)
    tgt = torch.tensor(np.asarray(g.tgt), dtype=torch.long)
    p = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    f = (torch.sigmoid(beta * (p[tgt] - p[src])) * nw).sum()
    f.backward()
    return p.grad.detach().numpy().copy()


def score_pct(g, pos: np.ndarray) -> float:
    """Exact feedforward percentage of a position vector (frozen oracle)."""
    s = score_from_positions(np.asarray(pos, dtype=np.float32),
                             np.asarray(g.src), np.asarray(g.tgt), g.weight)
    return pct(s, g.total_weight)


def block_structure(pos: np.ndarray, c: np.ndarray, gap_factor: float) -> dict:
    """Describe the position cloud: how many blocks, and are they the sign(c) split?

    ``n_blocks`` counts clusters separated by a gap > ``gap_factor`` x the median gap;
    ``sign_var_frac`` is the fraction of position variance explained by grouping nodes
    by ``sign(c)`` (1.0 = the cloud IS the two-block split predicted by T2).
    """
    s = np.sort(pos)
    gaps = np.diff(s)
    med = float(np.median(gaps)) if gaps.size else 0.0
    n_blocks = int(1 + np.sum(gaps > gap_factor * med)) if med > 0 else 1

    total_var = float(np.var(pos))
    if total_var <= 0:
        return {"n_blocks": n_blocks, "sign_var_frac": 0.0}
    grand = float(pos.mean())
    between = 0.0
    for mask in (c > 0, c <= 0):
        if mask.sum():
            between += mask.sum() * (float(pos[mask].mean()) - grand) ** 2
    return {"n_blocks": n_blocks,
            "sign_var_frac": float(between / (len(pos) * total_var))}


def rocket_from_init(g, init_pos: np.ndarray, seed: int, epochs: int):
    """Run the UNCHANGED ``run_rocket`` from an explicit initial position vector."""
    cfg = RocketConfig(epochs=epochs)
    device = torch.device(CONFIG["device"])
    res = run_rocket(g, cfg, seed=seed, device=device,
                     init_positions=torch.tensor(init_pos.astype(np.float32),
                                                 device=device))
    return res


def sift_of(g, positions: np.ndarray) -> float:
    """Apply the confirmed H35 under-relaxed sift to a position vector -> exact pct."""
    rank0 = np.argsort(np.argsort(positions, kind="stable"),
                       kind="stable").astype(np.int64)
    s = CONFIG["sift"]
    _best_rank, score, _log = sift_underrelaxed(
        g, rank0, k_full=s["k_full"], alpha=s["alpha"], max_sweeps=s["max_sweeps"])
    return pct(score, g.total_weight)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — THEORY: is the small-scale gradient really the imbalance vector?
# ──────────────────────────────────────────────────────────────────────────────
def stage_theory(graphs: dict) -> dict:
    """Test P1: cos(grad F, -c) -> 1 and residual ~ (beta*std)^2 as std -> 0."""
    out = {}
    for name, (g, _ref) in graphs.items():
        c = imbalance(g)
        rows = []
        for beta in CONFIG["theory_betas"]:
            analytic = -(beta / 4.0) * c
            for std in CONFIG["theory_stds"]:
                rng = np.random.RandomState(CONFIG["seed"])
                pos = rng.randn(g.n_nodes) * std
                grad = surrogate_grad(g, pos, beta)
                cos = float(grad @ analytic /
                            (np.linalg.norm(grad) * np.linalg.norm(analytic)))
                resid = float(np.linalg.norm(grad - analytic) /
                              np.linalg.norm(analytic))
                rows.append({"beta": beta, "std": std, "beta_std": beta * std,
                             "cos_grad_vs_minus_c": cos,
                             "rel_residual": resid})
        # Slope of log(residual) vs log(beta*std) over the small-scale rows: expect ~2.
        small = [r for r in rows if r["beta_std"] < 1e-2 and r["rel_residual"] > 0]
        slope = None
        if len(small) >= 2:
            x = np.log10([r["beta_std"] for r in small])
            y = np.log10([r["rel_residual"] for r in small])
            slope = float(np.polyfit(x, y, 1)[0])
        out[name] = {"rows": rows, "residual_loglog_slope": slope,
                     "predicted_slope": 2.0}
        print(f"[theory:{name}] residual ~ (beta*std)^{slope:.3f} "
              f"(theory: ^2); cos at beta*std=1e-6: "
              f"{rows[0]['cos_grad_vs_minus_c']:.6f}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — MECHANISM: does a tight init produce the predicted block cascade?
# ──────────────────────────────────────────────────────────────────────────────
def stage_cascade(graphs: dict) -> dict:
    """Test T2/T3 (P2): constant-beta Adam from a tight init -> block structure.

    This is a MECHANISM PROBE, not a Rocket run: beta is held constant (Rocket cycles
    it) so that the observed splitting cannot be an artifact of the beta schedule. The
    update rule (Adam, lr, grad-norm clip) mirrors ``run_rocket``.
    """
    cc = CONFIG["cascade"]
    out = {}
    for name, (g, _ref) in graphs.items():
        c = imbalance(g)
        rng = np.random.RandomState(CONFIG["seed"])
        pos0 = (rng.randn(g.n_nodes) * cc["init_std"]).astype(np.float32)

        p = torch.nn.Parameter(torch.tensor(pos0))
        opt = torch.optim.Adam([p], lr=cc["lr"])
        w = np.asarray(g.weight, dtype=np.float64)
        nw = torch.tensor((w / w.max()).astype(np.float32))
        src = torch.tensor(np.asarray(g.src), dtype=torch.long)
        tgt = torch.tensor(np.asarray(g.tgt), dtype=torch.long)

        snaps = []
        for step in range(cc["steps"] + 1):
            if step % cc["snapshot_every"] == 0:
                pos = p.detach().numpy().copy()
                bs = block_structure(pos, c, cc["gap_factor"])
                snaps.append({"step": step, "std": float(pos.std()),
                              "pct": score_pct(g, pos), **bs})
            opt.zero_grad()
            loss = -(torch.sigmoid(cc["beta"] * (p[tgt] - p[src])) * nw).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p], 1.0)
            opt.step()

        # How much of the FINAL order is explained by the imbalance sign / rank?
        final = p.detach().numpy()
        rho_c = float(spearmanr(final, -c)[0])
        rho_init = float(spearmanr(final, pos0)[0])
        out[name] = {"snapshots": snaps, "spearman_final_vs_minus_c": rho_c,
                     "spearman_final_vs_init": rho_init}
        print(f"[cascade:{name}] blocks {snaps[0]['n_blocks']} -> "
              f"{snaps[-1]['n_blocks']}, sign(c) variance share "
              f"{snaps[-1]['sign_var_frac']:.3f}, spearman(final, -c)={rho_c:.3f}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — STANDALONE: does a tight init beat the N(0,1) baseline?
# ──────────────────────────────────────────────────────────────────────────────
def stage_standalone(graphs: dict) -> dict:
    """Test P3: full Rocket runs across init scales x seeds -> exact metric."""
    out = {}
    for name, (g, ref_pct) in graphs.items():
        c = imbalance(g)
        epochs = CONFIG["epochs"][name]
        rows = []
        for std in CONFIG["init_stds"]:
            for seed in CONFIG["seeds"]:
                rng = np.random.RandomState(seed)
                init = (rng.randn(g.n_nodes) * std).astype(np.float32)
                t0 = time.time()
                res = rocket_from_init(g, init, seed, epochs)
                fp = res.best_positions
                rows.append({
                    "std": std, "seed": seed, "pct": res.best_pct,
                    "init_pct": score_pct(g, init),
                    "final_std": float(fp.std()),
                    "spearman_final_vs_minus_c": float(spearmanr(fp, -c)[0]),
                    "spearman_final_vs_init": float(spearmanr(fp, init)[0]),
                    "wall_s": time.time() - t0,
                })
        agg = {}
        for std in CONFIG["init_stds"]:
            v = [r["pct"] for r in rows if r["std"] == std]
            agg[str(std)] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                             "n": len(v)}
        base = agg[str(CONFIG["baseline_std"])]
        for std in CONFIG["init_stds"]:
            agg[str(std)]["delta_vs_baseline_pp"] = agg[str(std)]["mean"] - base["mean"]
        out[name] = {"rows": rows, "aggregate": agg, "reference_pct": ref_pct}
        print(f"[standalone:{name}] " + "  ".join(
            f"std={s}:{agg[str(s)]['mean']:.4f}" for s in CONFIG["init_stds"]))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — COMBINED with our findings (#1 H02 warm start, #4/#5 the sift)
# ──────────────────────────────────────────────────────────────────────────────
def stage_combined(graphs: dict) -> dict:
    """Test P4 (compressing the H02 warm start) and P5 (does the sift erase it?)."""
    out = {}
    for name, (g, ref_pct) in graphs.items():
        epochs = CONFIG["epochs"][name]
        device = torch.device(CONFIG["device"])

        order = greedy_fas_order(g)
        h02_pos = _init_positions_from_order(order, device).cpu().numpy()
        h02_std = float(h02_pos.std())

        rows = []
        # (a) H02's ordering, re-scaled: does tightening destroy the warm start?
        for target_std in CONFIG["h02_stds"]:
            init = (h02_pos * (target_std / h02_std)).astype(np.float32)
            for seed in CONFIG["seeds"]:
                res = rocket_from_init(g, init, seed, epochs)
                rows.append({"init": "H02_greedy", "std": target_std, "seed": seed,
                             "init_pct": score_pct(g, init), "rocket_pct": res.best_pct,
                             "sifted_pct": sift_of(g, res.best_positions)})
        # (b) random init at the extreme scales, same treatment (control arm)
        for target_std in [min(CONFIG["init_stds"]), CONFIG["baseline_std"]]:
            for seed in CONFIG["seeds"]:
                rng = np.random.RandomState(seed)
                init = (rng.randn(g.n_nodes) * target_std).astype(np.float32)
                res = rocket_from_init(g, init, seed, epochs)
                rows.append({"init": "random", "std": target_std, "seed": seed,
                             "init_pct": score_pct(g, init), "rocket_pct": res.best_pct,
                             "sifted_pct": sift_of(g, res.best_positions)})

        agg = {}
        for r in rows:
            k = f"{r['init']}@{r['std']}"
            agg.setdefault(k, {"rocket": [], "sifted": []})
            agg[k]["rocket"].append(r["rocket_pct"])
            agg[k]["sifted"].append(r["sifted_pct"])
        for k, v in agg.items():
            for field in ("rocket", "sifted"):
                arr = v.pop(field)
                v[f"{field}_mean"] = float(np.mean(arr))
                v[f"{field}_std"] = float(np.std(arr, ddof=1))
        out[name] = {"rows": rows, "aggregate": agg, "h02_native_std": h02_std,
                     "greedy_order_pct": score_pct(g, np.asarray(order,
                                                                dtype=np.float32)),
                     "reference_pct": ref_pct}
        print(f"[combined:{name}] " + "  ".join(
            f"{k}: rocket {v['rocket_mean']:.4f} -> sift {v['sifted_mean']:.4f}"
            for k, v in agg.items()))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def make_plots(report: dict) -> list:
    """Two figures: the theory check and the score-vs-init-scale curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    theory, standalone = report.get("theory"), report.get("standalone")

    if theory:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for name, d in theory.items():
            for beta in CONFIG["theory_betas"]:
                rows = [r for r in d["rows"] if r["beta"] == beta]
                x = [r["beta_std"] for r in rows]
                axes[0].plot(x, [r["cos_grad_vs_minus_c"] for r in rows], "o-",
                             label=f"{name}, beta={beta}")
                axes[1].loglog(x, [r["rel_residual"] for r in rows], "o-",
                               label=f"{name}, beta={beta}")
        ref = np.array([1e-7, 1e1])
        axes[1].loglog(ref, ref ** 2, "k--", label="theory: slope 2")
        axes[0].set(xscale="log", xlabel="beta * std(positions)",
                    ylabel="cos(grad F, -c)",
                    title="Small-scale gradient IS the imbalance vector")
        axes[1].set(xlabel="beta * std(positions)",
                    ylabel="||grad F + (beta/4)c|| / ||(beta/4)c||",
                    title="Pairwise information is suppressed as (beta*std)^2")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle("A-INIT theory check: what a tight init does to the gradient")
        fig.tight_layout()
        p = _OUT / "proto_ainit_theory.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p.relative_to(_ROOT)))

    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for name, d in standalone.items():
            stds = CONFIG["init_stds"]
            m = [d["aggregate"][str(s)]["mean"] for s in stds]
            e = [d["aggregate"][str(s)]["std"] for s in stds]
            ax.errorbar(stds, m, yerr=e, fmt="o-", capsize=3, label=name)
        ax.axvline(CONFIG["baseline_std"], color="k", ls="--", lw=1,
                   label="baseline init std = 1 (N(0,1))")
        ax.set(xscale="log", xlabel="init position std",
               ylabel="exact feedforward %",
               title="A-INIT standalone: final score vs initialization scale\n"
                     "(Rocket, 3 seeds, mean +- std)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = _OUT / "proto_ainit_scale.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p.relative_to(_ROOT)))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["all", "theory", "cascade", "standalone", "combined"])
    ap.add_argument("--datasets", nargs="+", default=CONFIG["datasets"])
    ap.add_argument("--out", default=str(_OUT / "proto_ainit_scale.json"))
    args = ap.parse_args()

    t0 = time.time()
    graphs = {}
    for name in args.datasets:
        g, ref = load_graph(name)
        graphs[name] = (g, ref)
        print(f"loaded {name}: n={g.n_nodes:,} m={g.n_edges:,} "
              f"total_w={g.total_weight:,.0f}"
              + (f"  reference={ref:.4f}%" if ref else ""))

    report = {"config": CONFIG, "git_commit": git_commit(),
              "datasets": {k: {"n_nodes": v[0].n_nodes, "n_edges": v[0].n_edges}
                           for k, v in graphs.items()}}

    if args.stage in ("all", "theory"):
        report["theory"] = stage_theory(graphs)
    if args.stage in ("all", "cascade"):
        report["cascade"] = stage_cascade(graphs)
    if args.stage in ("all", "standalone"):
        report["standalone"] = stage_standalone(graphs)
    if args.stage in ("all", "combined"):
        report["combined"] = stage_combined(graphs)

    report["plots"] = make_plots(report)
    report["wall_clock_s"] = time.time() - t0

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest}  ({report['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
