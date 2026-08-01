"""Q01 — Why does starting Rocket from the best solution drift the score DOWN?

TRACK-B DIAGNOSTIC (reads data/best_solution via mfas.analysis.gap — the privileged reader).
Never writes to results/; artifacts go to experiments/outputs/.

Answers the objection: "with a very small step from the optimum we should NOT drift — the loss
is lower there." The claim is CORRECT, but only at the right SCALE. The loss depends on position
*gaps*, not on the order alone, so the best ORDER embedded at an arbitrary (even) spacing is a
high-loss point that gradient descent correctly flees — which also reshuffles the order and drops
the discrete score. At the best order's surrogate-OPTIMAL spacing P* (the true low-loss point,
std ~141), P* is a local max of the surrogate with F(P*) > F(Rocket), and small-lr GD HOLDS it.

This consolidates the two dr_tmp probes (drift_from_optimal_spacing.py, drift_scale_sweep.py) and
fills the scale grid. It corrects findings.md #3's over-strong "unholdable under EVERY scale".

Run (env `allen`, ~3-6 min on Apple MPS):
    PYTHONPATH=src /opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python \
        experiments/diagnostics/q01_drift_from_optimum.py

Artifacts:
    experiments/outputs/q01_drift.json          (all numbers)
    experiments/outputs/q01_hold_from_optimum.png
    experiments/outputs/q01_scale_sweep.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from mfas.io import load_dataset
from mfas.metrics import score_from_positions, pct
from mfas.analysis import gap

# ── CONFIG (no magic numbers) ──────────────────────────────────────────────────
CONFIG = dict(
    DATASET       = "connectome",          # best_solution exists only for connectome
    BETA          = 1.05,                  # Rocket's convergence beta
    SEED          = 42,
    SPACING_ITERS = 1500,                  # optimize_spacing budget (float64, CPU)
    SPACING_LR    = 0.1,
    HOLD_LRS      = [5e-4, 5e-3, 5e-2],    # "very small step" ... up to Rocket's default 5e-2
    HOLD_STEPS    = 2000,
    HOLD_LOG_EVERY= 100,
    # rescale the optimal-spacing SHAPE to these std values; 0.58 = the logged even-spacing
    # scale (the artefact), ~141 = Rocket's converged operating scale.
    SCALE_GRID    = [0.58, 2.0, 10.0, 50.0, 141.0, 500.0, 5000.0, 53000.0],
    SCALE_STEPS   = 1000,
    SCALE_LR      = 5e-3,
    ROCKET_REF    = "results/rocket_best_positions.npy",  # committed parity anchor (82.9161%)
    OUT_DIR       = Path("experiments/outputs"),
)
OUT = CONFIG["OUT_DIR"]; OUT.mkdir(parents=True, exist_ok=True)
BETA = CONFIG["BETA"]

dev = torch.device("mps" if torch.backends.mps.is_available() else
                   ("cuda" if torch.cuda.is_available() else "cpu"))
g = load_dataset(CONFIG["DATASET"])
src = np.asarray(g.src); tgt = np.asarray(g.tgt); w = np.asarray(g.weight)
tot = int(w.sum())
print(f"device={dev}  dataset={CONFIG['DATASET']}  total_weight={tot:,}")

# torch surrogate F(pos) = Σ σ(β·Δ)·ŵ  (drives the optimizer; discrete score stays on CPU)
src_t = torch.as_tensor(src, dtype=torch.long, device=dev)
tgt_t = torch.as_tensor(tgt, dtype=torch.long, device=dev)
nw_t  = torch.as_tensor((w / w.max()).astype(np.float32), device=dev)
def F(pos):
    d = pos[tgt_t] - pos[src_t]
    return (torch.sigmoid(BETA * d) * nw_t).sum()

def disc(pos_np):
    return pct(score_from_positions(np.asarray(pos_np, dtype=np.float64), src, tgt, w), tot)

# ── Anchors: best order, its optimal spacing P*, and the Rocket reference ────────
order, best_pct_reported = gap.load_best_solution(g)          # (rank_of_node, pct)
disc_best = disc(gap.order_to_positions(order))
print(f"\nbest_solution: discrete={disc_best:.4f}%  (reported {best_pct_reported:.4f}%)")

Pstar, Fstar = gap.optimize_spacing(order, g, beta=BETA, iters=CONFIG["SPACING_ITERS"],
                                    lr=CONFIG["SPACING_LR"], device="cpu", free_scale=True)
disc_Pstar = disc(Pstar)
print(f"P* (optimal spacing): F={Fstar:.1f}  std={Pstar.std():.2f}  discrete={disc_Pstar:.4f}%")

rock = np.load(CONFIG["ROCKET_REF"])
F_rock = gap.surrogate_objective(rock, g, BETA)
disc_rock = disc(rock)
print(f"Rocket reference    : F={F_rock:.1f}  std={rock.std():.2f}  discrete={disc_rock:.4f}%")
print(f"==> F(P*) - F(Rocket) = {Fstar - F_rock:+.1f}   "
      f"(GD ascends F, so from P* it cannot flow down into Rocket's lower-F basin)")

# ‖∇F(P*)‖ in float64 on CPU (is P* a critical point / local max of the surrogate?)
p64 = torch.tensor(Pstar, dtype=torch.float64, requires_grad=True)
s64 = torch.as_tensor(src, dtype=torch.long); t64 = torch.as_tensor(tgt, dtype=torch.long)
nw64 = torch.as_tensor((w / w.max()).astype(np.float64))
F64 = (torch.sigmoid(BETA * (p64[t64] - p64[s64])) * nw64).sum()
F64.backward()
gnorm_Pstar = float(p64.grad.norm()); gmax_Pstar = float(p64.grad.abs().max())
print(f"‖∇F(P*)‖={gnorm_Pstar:.4g}  max|∇F(P*)|={gmax_Pstar:.4g}  (≈0 => critical point)\n")

# ── HOLD test: from P*, small-lr Adam ascent, constant β — does the score hold? ──
def ascent(pos0_np, lr, steps, log_every):
    pos = torch.tensor(np.asarray(pos0_np, dtype=np.float32), device=dev, requires_grad=True)
    opt = torch.optim.Adam([pos], lr=lr)
    traj = []
    for s in range(steps + 1):
        if s % log_every == 0:
            traj.append((s, disc(pos.detach().cpu().numpy())))
        if s == steps:
            break
        opt.zero_grad()
        (-F(pos)).backward()
        torch.nn.utils.clip_grad_norm_([pos], 1.0)   # Rocket uses grad_clip=1.0
        opt.step()
    return traj

hold = {}
for lr in CONFIG["HOLD_LRS"]:
    traj = ascent(Pstar, lr, CONFIG["HOLD_STEPS"], CONFIG["HOLD_LOG_EVERY"])
    hold[f"{lr:g}"] = traj
    print(f"HOLD from P* lr={lr:g}: {traj[0][1]:.4f}% -> {traj[-1][1]:.4f}%  "
          f"(min {min(d for _, d in traj):.4f}%)")

# ── SCALE sweep: same optimal SHAPE, different std — where does it stop holding? ─
shape = (Pstar - Pstar.mean()) / Pstar.std()          # unit-variance optimal shape
disc_shape = disc(shape)
print(f"\noptimal shape (unit variance): discrete={disc_shape:.4f}%  (scale-invariant)")
scale_rows = []
for std in CONFIG["SCALE_GRID"]:
    p0 = (shape * std).astype(np.float32)
    pg = torch.tensor(p0, device=dev, requires_grad=True)
    F(pg).backward(); gnorm = float(pg.grad.norm())
    traj = ascent(p0, CONFIG["SCALE_LR"], CONFIG["SCALE_STEPS"], CONFIG["SCALE_STEPS"])
    d0, dN = traj[0][1], traj[-1][1]
    scale_rows.append(dict(std=std, gradnorm=gnorm, disc0=d0, discN=dN, drop=d0 - dN))
    print(f"  std={std:8.1f}  ‖∇F‖={gnorm:10.4g}  disc {d0:.4f}% -> {dN:.4f}%  drop={d0 - dN:.4f}")

# ── Persist numbers (NOT to results/) ───────────────────────────────────────────
out = dict(
    config={k: (str(v) if isinstance(v, Path) else v) for k, v in CONFIG.items()},
    device=str(dev), total_weight=tot,
    best_pct=disc_best, Pstar=dict(F=Fstar, std=float(Pstar.std()), disc=disc_Pstar),
    rocket=dict(F=F_rock, std=float(rock.std()), disc=disc_rock),
    dF_Pstar_minus_rocket=Fstar - F_rock,
    gradnorm_Pstar=gnorm_Pstar, gradmax_Pstar=gmax_Pstar,
    hold=hold, scale_sweep=scale_rows,
)
(OUT / "q01_drift.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved {OUT / 'q01_drift.json'}")

# ── Plot A: hold trajectories ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
for lr, traj in hold.items():
    xs = [s for s, _ in traj]; ys = [d for _, d in traj]
    ax.plot(xs, ys, "-o", ms=3, label=f"Adam lr={lr}")
ax.axhline(disc_best, color="tab:green", ls="--", label=f"best_solution = {disc_best:.4f}%")
ax.axhline(disc_rock, color="tab:blue",  ls="--", label=f"Rocket reference = {disc_rock:.4f}%")
ax.set_xlabel("ascent step (constant β=1.05)"); ax.set_ylabel("discrete feedforward (%)")
ax.set_title("Q01 — from the optimal spacing P*, small-lr GD HOLDS the best score")
ax.legend(loc="center right", fontsize=8); plt.tight_layout()
plt.savefig(OUT / "q01_hold_from_optimum.png", dpi=120, bbox_inches="tight")
print(f"saved {OUT / 'q01_hold_from_optimum.png'}")

# ── Plot B: scale sweep (drop vs std) ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
stds = [r["std"] for r in scale_rows]; drops = [r["drop"] for r in scale_rows]
ax.plot(stds, drops, "-o", color="tab:red")
ax.set_xscale("log")
ax.axvline(0.58, color="gray", ls=":", label="logged even-spacing scale (std≈0.58)")
ax.axvline(141.0, color="tab:blue", ls=":", label="Rocket operating scale (std≈141)")
ax.set_xlabel("position scale  std(pos)  [log]")
ax.set_ylabel(f"discrete drop after {CONFIG['SCALE_STEPS']} steps (pp)")
ax.set_title("Q01 — the 'collapse' is a SCALE artefact: big drop only at tiny scale")
ax.legend(loc="upper right", fontsize=8); plt.tight_layout()
plt.savefig(OUT / "q01_scale_sweep.png", dpi=120, bbox_inches="tight")
print(f"saved {OUT / 'q01_scale_sweep.png'}")
