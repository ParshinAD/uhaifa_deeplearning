"""Prototype-gate smoke test: A-ALT / gradient-kick after the H35 sift fixed point.

Throwaway (dr_tmp, gitignored). Does NOT write to results/. Discrete score via the
frozen CPU scorer only. See task brief for the kill/surprise threshold.

Pipeline (connectome, seed 42):
  greedy_fas_order -> run_rocket(20k) -> rank0 -> sift_underrelaxed  => S (fixed point)
  for SCALE in {150,500,1500}:
     embed S even[-1,1], rescale std=SCALE, Adam-ascend surrogate F (beta=1, K=150),
     S_kick = argsort^2(pos); re-sift -> S''; report pct(S_kick), dS = pct_S'' - pct_S.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from mfas.io import load_connectome
from mfas.baseline.rocket import RocketConfig, run_rocket
from mfas.experiments.H02 import greedy_fas_order, _init_positions_from_order
from mfas.refine import sift_underrelaxed
from mfas.metrics import score_from_order, pct
from mfas.utils.seeding import select_device

SEED = 42
EPOCHS = 20_000
K_FULL = 6
ALPHA = 0.7
MAX_SWEEPS = 40
RESIFT_SWEEPS = 40      # warm re-sift converges fast (breaks at fixed point)
SCALES = [150.0, 500.0, 1500.0]
K_KICK = 150
BETA = 1.0
LR = 0.05
GRAD_CLIP = 1.0


def log(*a):
    print(*a, flush=True)


def double_argsort(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.int64)


def surrogate_kick(rank_S, src_t, tgt_t, nw_t, device, scale):
    """Embed rank_S even in [-1,1], rescale to std=scale, Adam-ascend F for K_KICK steps.

    Returns the kicked positions (numpy, float32).
    """
    n = rank_S.shape[0]
    pos0 = (rank_S.astype(np.float32) / max(n - 1, 1)) * 2.0 - 1.0
    pos0 = (pos0 - pos0.mean()) / pos0.std() * float(scale)
    positions = torch.nn.Parameter(torch.tensor(pos0, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([positions], lr=LR)
    for _ in range(K_KICK):
        opt.zero_grad()
        delta = positions[tgt_t] - positions[src_t]
        sig = torch.sigmoid(BETA * delta)
        loss = -(sig * nw_t).sum()          # ascend F = Σ σ(β·Δ)·ŵ
        loss.backward()
        torch.nn.utils.clip_grad_norm_([positions], GRAD_CLIP)
        opt.step()
    return positions.detach().cpu().numpy()


def main():
    t_all = time.time()
    device, dev_name = select_device("auto")
    log(f"[env] device={dev_name} torch={torch.__version__}")

    g = load_connectome()
    total = g.total_weight
    src_o, tgt_o = np.asarray(g.src), np.asarray(g.tgt)
    log(f"[data] {g!r}")

    # ── Stage 1+2: greedy warm start -> Rocket (PURE) ────────────────────────────
    t0 = time.time()
    order = greedy_fas_order(g)
    init_pos = _init_positions_from_order(order, device)
    rocket = run_rocket(g, RocketConfig(epochs=EPOCHS), seed=SEED, device=device,
                        init_positions=init_pos)
    log(f"[rocket] pure best_pct={rocket.best_pct:.6f} "
        f"({time.time()-t0:.0f}s, {rocket.n_epochs_done} epochs)")

    # ── Stage 3: under-relaxed two-phase sift -> S ───────────────────────────────
    rank0 = double_argsort(rocket.best_positions)
    t0 = time.time()
    S, score_S, log_S = sift_underrelaxed(g, rank0, k_full=K_FULL, alpha=ALPHA,
                                          max_sweeps=MAX_SWEEPS)
    pct_S = pct(score_S, total)
    log(f"[sift] pct_S={pct_S:.6f} sweeps={len(log_S)} ({time.time()-t0:.0f}s)")

    # Prep surrogate tensors (device, float32) once.
    src_t = torch.tensor(src_o, dtype=torch.long, device=device)
    tgt_t = torch.tensor(tgt_o, dtype=torch.long, device=device)
    w_np = np.asarray(g.weight)
    nw_t = torch.tensor((w_np / float(w_np.max())).astype(np.float32), device=device)

    results = {"pct_S": pct_S, "scales": {}}
    for scale in SCALES:
        t0 = time.time()
        pos_kick = surrogate_kick(S, src_t, tgt_t, nw_t, device, scale)
        S_kick = double_argsort(pos_kick)
        score_kick = score_from_order(S_kick, src_o, tgt_o, g.weight)
        pct_kick = pct(score_kick, total)

        Spp, score_Spp, log_Spp = sift_underrelaxed(
            g, S_kick, k_full=K_FULL, alpha=ALPHA, max_sweeps=RESIFT_SWEEPS)
        pct_Spp = pct(score_Spp, total)
        dS = pct_Spp - pct_S
        best_oracle = max(pct_S, pct_kick, pct_Spp)
        results["scales"][scale] = dict(
            pct_kick=pct_kick, pct_Spp=pct_Spp, dS=dS, best_oracle=best_oracle,
            resift_sweeps=len(log_Spp), Spp=Spp.tolist())
        log(f"[SCALE={scale:>6.0f}] pct(S_kick)={pct_kick:.6f}  "
            f"pct_S''={pct_Spp:.6f}  dS={dS:+.6f}pp  best_oracle={best_oracle:.6f}  "
            f"resift_sweeps={len(log_Spp)}  ({time.time()-t0:.0f}s)")

    # ── Long-range check for best SCALE (by dS) ──────────────────────────────────
    best_scale = max(results["scales"], key=lambda s: results["scales"][s]["dS"])
    Spp = np.asarray(results["scales"][best_scale]["Spp"], dtype=np.int64)
    dist = np.abs(S.astype(np.int64) - Spp)
    moved = dist[dist > 0]
    if moved.size:
        lr_stats = dict(n_moved=int(moved.size),
                        median=float(np.median(moved)),
                        p90=float(np.percentile(moved, 90)),
                        max=int(moved.max()))
    else:
        lr_stats = dict(n_moved=0, median=0.0, p90=0.0, max=0)
    results["best_scale"] = best_scale
    results["longrange"] = lr_stats
    log(f"[longrange] best_scale={best_scale:.0f}  n={g.n_nodes}  moved={lr_stats['n_moved']}  "
        f"median={lr_stats['median']:.0f}  p90={lr_stats['p90']:.0f}  max={lr_stats['max']}")

    # ── Verdict ──────────────────────────────────────────────────────────────────
    max_dS = max(results["scales"][s]["dS"] for s in results["scales"])
    arg_dS = max(results["scales"], key=lambda s: results["scales"][s]["dS"])
    verdict = "SURPRISE" if max_dS > 0.012 else "KILL"
    log(f"[VERDICT] max dS={max_dS:+.6f}pp at SCALE={arg_dS:.0f}  "
        f"(threshold 2σ=+0.012pp) -> {verdict}")

    # Strip large Spp arrays before dumping a compact summary.
    for s in results["scales"]:
        results["scales"][s].pop("Spp", None)
    results["max_dS"] = max_dS
    results["arg_max_dS"] = arg_dS
    results["verdict"] = verdict
    log("[SUMMARY_JSON] " + json.dumps(results))
    log(f"[done] total {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
