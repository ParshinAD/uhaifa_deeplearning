"""Prototype GATE for A-MBAND — a MULTI-BAND surrogate (one blur width cannot serve all pairs).

SELF-CONTAINED prototype (NOT promoted to ``src/``).

The problem it attacks (measured in `diagnosis.md` § Q01)
--------------------------------------------------------
The surrogate scores an edge by ``sigma(beta * Delta)``, so its ability to tell "u before v" apart
from "v before u" has a *resolution*: two nodes must be a certain number of RANKS apart before the
credit stops being ~0.5. With even spacing that resolution is

    resolution_ranks(0.75 credit) = ln(3) / (beta * gap),   gap = span/n ~= sqrt(12)*std/n
                                  ~= 0.317 * n / (beta*std)

Measured at each graph's converged Rocket positions (seed 42, beta=1.05):

    mouse       n=    148  std=  21.5   ->    2.1 ranks   (1.40% of the graph)
    microns     n= 67,534  std= 908.4   ->   22.5 ranks   (0.033%)
    connectome  n=136,648  std= 141.0   ->  292.6 ranks   (0.214%)

So on the fly connectome the optimizer cannot resolve the order below ~293 ranks, and 46.2% of the
weight the near-optimal order wins over Rocket sits *inside* 1000 ranks (in the good order's own
ranking). The surrogate is structurally blind to exactly the wins that are missing.

The hypothesis
--------------
A single ``beta`` must serve both a pair 1 rank apart and a pair 100,000 ranks apart, and it cannot:
wide blur gives gradient but no resolution, narrow blur gives resolution but no gradient (at
beta=1.05 already 99% of edge gradient mass sits on 2.83% of edges). Add a SECOND, much sharper
band on top of the existing one:

    F = sum_e w_hat_e * sigma(beta_coarse * D_e)  +  lam * sum_e w_hat_e * sigma(beta_fine * D_e)

with ``beta_fine`` pinned to the *rank* scale, which makes it scale-free:

    beta_fine = rho * n / (sqrt(12) * std(P).detach())      # half-width = 1/rho ranks

The coarse band keeps doing the long-range transport (its gradient never dies); the fine band
supplies the reward for narrow wins that the coarse band discounts to ~0.5.

Prior evidence that this is worth building (ad-hoc probe, connectome, std=141, even spacing):
the single-band surrogate ranks Rocket's 82.92% order ABOVE the 84.61% order by 174.88; adding a
fine band at 1-rank half-width (beta_fine=280) with lam=1 FLIPS the ranking to +102.32 in favour of
the better order. Bands coarser than ~20 ranks do not flip it. Stage B below re-derives this as a
committed artifact.

Honest expectations
-------------------
Ranking correctly is necessary, not sufficient. The fine band is nearly silent at Rocket's own
configuration (only 0.004% of edge weight lies inside its live zone there, vs 1.25% at the good
order), so it supplies reward, not tow. Stage C is the real gate: does it move the exact metric?

Leakage-safety: the fine band is a function of positions and input weights only. `data/best_solution`
is read ONLY via `mfas.analysis.gap` and ONLY in Stage B (ranking measurement); it never touches the
training loop of Stage C. Writes to `experiments/outputs/` only, never `results/`.

Run (env `allen`, from the repo root):
    PYTHONPATH=src python experiments/proto_amband.py --stage all
    PYTHONPATH=src python experiments/proto_amband.py --stage sweep --datasets connectome
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from mfas.baseline.rocket import RocketConfig, make_beta_schedule
from mfas.io import load_dataset
from mfas.metrics import pct, score_from_positions

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "experiments" / "outputs"

CONFIG = {
    "seeds": [42, 123, 999],
    "sweep_seed": 42,                    # single seed for the cheap region scan
    "epochs": {"connectome": 20_000, "mouse": 5_000, "microns": 80_000},
    "beta_main": 1.05,
    "operating_scale_probe_std": 141.0,  # connectome's converged std, for Stage B

    # Fine-band knobs. rho = 1 puts the sigmoid's half-width at ONE rank; lam is its weight.
    # The ad-hoc probe found the ranking flips for half-widths <= ~1 rank and not at ~20 ranks.
    "sweep_rho": [0.3, 1.0, 3.0],
    "sweep_lam": [0.3, 1.0, 3.0],
    # lam ramps linearly from 0 over the first `lam_warmup_frac` of training when enabled, so the
    # coarse band does the transport before the fine band starts rewarding narrow wins.
    "lam_warmup_frac": 0.0,
    "log_interval": 100,
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_ROOT, text=True).strip()
    except Exception:                                     # pragma: no cover
        return "unknown"


def resolution_ranks(n: int, beta: float, std: float, target: float = 0.75) -> float:
    """Ranks two nodes must be apart before the surrogate gives them ``target`` credit."""
    gap = np.sqrt(12) * std / n
    return float(np.log(target / (1 - target)) / (beta * gap))


# ──────────────────────────────────────────────────────────────────────────────
# The multi-band Rocket loop — mirrors mfas.baseline.rocket.run_rocket exactly
# except for the added fine band. At lam = 0 it must reproduce the baseline
# (asserted in Stage C's control arm).
# ──────────────────────────────────────────────────────────────────────────────
def run_rocket_mband(g, cfg: RocketConfig, seed: int, device, lam: float, rho: float,
                     warmup_frac: float = 0.0, init_positions=None,
                     beta_fine_fixed: float = 0.0):
    """Rocket with an extra rank-scale sigmoid band. ``lam = 0`` ⇒ the unchanged baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = g.n_nodes
    src_t = torch.tensor(np.asarray(g.src), dtype=torch.long, device=device)
    tgt_t = torch.tensor(np.asarray(g.tgt), dtype=torch.long, device=device)
    w_np = np.asarray(g.weight)
    nw_t = torch.tensor((w_np / float(w_np.max())).astype(np.float32), device=device)
    src_np, tgt_np = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight

    pos_data = (init_positions.clone().detach().to(torch.float32)
                if init_positions is not None else torch.randn(n, device=device))
    positions = torch.nn.Parameter(pos_data)

    optimizer = torch.optim.Adam([positions], lr=cfg.lr)
    milestone = int(cfg.epochs * cfg.lr_decay_start)
    sched = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                        total_iters=milestone),
                    torch.optim.lr_scheduler.ExponentialLR(
                        optimizer,
                        gamma=cfg.lr_end_factor ** (1.0 / max(cfg.epochs - milestone, 1)))],
        milestones=[milestone])
    betas = make_beta_schedule(cfg.epochs, cfg.cycles)

    def discrete(p):
        return score_from_positions(p.detach().cpu().numpy(), src_np, tgt_np, g.weight)

    best_score = discrete(positions)
    best_positions = positions.detach().clone()
    hist, t0 = [], time.time()

    for i in range(cfg.epochs):
        beta = float(betas[i])
        lam_i = lam if warmup_frac <= 0 else lam * min(1.0, i / max(warmup_frac * cfg.epochs, 1))

        optimizer.zero_grad()
        delta = positions[tgt_t] - positions[src_t]
        obj = (torch.sigmoid(beta * delta) * nw_t).sum()
        if lam_i > 0:
            # Rank-scale band: half-width = 1/rho ranks, at ANY position scale (detached std).
            if beta_fine_fixed > 0:
                beta_fine = beta_fine_fixed       # non-adaptive control: cannot self-amplify
            else:
                std = positions.detach().std().clamp_min(1e-12)
                beta_fine = rho * n / (float(np.sqrt(12)) * std)
            obj = obj + lam_i * (torch.sigmoid(beta_fine * delta) * nw_t).sum()
        loss = -obj
        loss.backward()
        torch.nn.utils.clip_grad_norm_([positions], cfg.grad_clip)
        optimizer.step()
        sched.step()

        if i % CONFIG["log_interval"] == 0 or i == cfg.epochs - 1:
            s = discrete(positions)
            if s > best_score:
                best_score, best_positions = s, positions.detach().clone()
            hist.append({"iter": i, "pct": pct(s, total), "best_pct": pct(best_score, total),
                         "std": float(positions.detach().std())})

    return {"best_pct": pct(best_score, total), "best_score": best_score,
            "best_positions": best_positions.detach().cpu().numpy(),
            "final_std": float(positions.detach().std()),
            "wall_s": time.time() - t0, "history": hist}


# ──────────────────────────────────────────────────────────────────────────────
# Stage A — where does the resolution pathology exist?
# ──────────────────────────────────────────────────────────────────────────────
def stage_resolution() -> dict:
    """Audit resolution-in-ranks at each graph's converged Rocket positions."""
    import glob
    out = {}
    print(f"{'graph':>12} {'n':>9} {'std':>10} {'beta*std':>10} {'resolution(ranks)':>19} {'%graph':>9}")
    for ds in ("mouse", "microns", "connectome"):
        f = sorted(glob.glob(str(_ROOT / f"results/*-baseline_passthrough-{ds}-s42-*_positions.npy")))
        if not f:
            continue
        pos = np.load(f[-1]); n = len(pos); std = float(pos.std())
        if abs(std - n / np.sqrt(12)) / (n / np.sqrt(12)) < 0.02:
            print(f"{ds:>12}  saved vector is a RANK vector, not positions — skipped")
            continue
        r = resolution_ranks(n, CONFIG["beta_main"], std)
        out[ds] = {"n": n, "std": std, "beta_std": CONFIG["beta_main"] * std,
                   "resolution_ranks": r, "frac_of_graph": r / n, "source": Path(f[-1]).name}
        print(f"{ds:>12} {n:>9,} {std:10.2f} {CONFIG['beta_main']*std:10.1f} {r:19.1f} "
              f"{100*r/n:8.3f}%")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stage B — does the two-band surrogate RANK the better order higher?
# ──────────────────────────────────────────────────────────────────────────────
def stage_ranking() -> dict:
    """Connectome only (it is the graph with a reference order and the severe pathology)."""
    from mfas.analysis.gap import load_best_solution
    g = load_dataset("connectome"); n = g.n_nodes
    w = np.asarray(g.weight, dtype=np.float64); nw = w / w.max()
    src, tgt = np.asarray(g.src), np.asarray(g.tgt)

    best = np.asarray(load_best_solution(g)[0]).astype(np.int64)
    rock = np.argsort(np.argsort(np.load(_ROOT / "results/rocket_best_positions.npy"),
                                 kind="stable"), kind="stable").astype(np.int64)
    std = CONFIG["operating_scale_probe_std"]

    def epos(o):
        p = (o.astype(np.float64) / (n - 1)) * 2 - 1
        return p * (std / p.std())

    gap = float(np.diff(epos(best)[np.argsort(best)]).mean())
    sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))
    D = {"best": epos(best)[tgt] - epos(best)[src],
         "rocket": epos(rock)[tgt] - epos(rock)[src]}

    rows = []
    for rho in [None] + CONFIG["sweep_rho"] + [0.05, 0.1]:
        for lam in ([0.0] if rho is None else CONFIG["sweep_lam"]):
            vals = {}
            for k in ("best", "rocket"):
                v = float((sig(CONFIG["beta_main"] * D[k]) * nw).sum())
                if rho:
                    v += lam * float((sig((rho / gap) * D[k]) * nw).sum())
                vals[k] = v
            rows.append({"rho": rho, "half_width_ranks": None if not rho else 1.0 / rho,
                         "lam": lam, "beta_fine": None if not rho else rho / gap,
                         "F_best": vals["best"], "F_rocket": vals["rocket"],
                         "diff": vals["best"] - vals["rocket"]})
    print(f"\n{'rho':>6} {'half-width(ranks)':>18} {'lam':>6} {'F(best)-F(rocket)':>19}  ranks correctly?")
    for r in rows:
        hw = "-" if r["half_width_ranks"] is None else f"{r['half_width_ranks']:.2f}"
        print(f"{str(r['rho']):>6} {hw:>18} {r['lam']:>6} {r['diff']:+19.2f}  "
              f"{'YES' if r['diff'] > 0 else 'no'}")
    return {"neighbour_gap": gap, "rows": rows}


# ──────────────────────────────────────────────────────────────────────────────
# Stage C — does TRAINING with the extra band move the exact metric?
# ──────────────────────────────────────────────────────────────────────────────
def stage_train(datasets, seeds, sweep: bool) -> dict:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = {}
    for ds in datasets:
        g = load_dataset(ds)
        cfg = RocketConfig(epochs=CONFIG["epochs"][ds])
        rows = []
        arms = [("control", 0.0, 0.0)]
        if sweep:
            arms += [(f"rho{r}_lam{l}", l, r)
                     for r in CONFIG["sweep_rho"] for l in CONFIG["sweep_lam"]]
        for name, lam, rho in arms:
            for seed in seeds:
                res = run_rocket_mband(g, cfg, seed, device, lam=lam, rho=rho,
                                       warmup_frac=CONFIG["lam_warmup_frac"])
                rows.append({"arm": name, "lam": lam, "rho": rho, "seed": seed,
                             "pct": res["best_pct"], "final_std": res["final_std"],
                             "wall_s": res["wall_s"],
                             "resolution_ranks": resolution_ranks(
                                 g.n_nodes, CONFIG["beta_main"], res["final_std"])})
                print(f"  [{ds}] {name:>14} seed={seed:<5} pct={res['best_pct']:.4f}  "
                      f"std={res['final_std']:8.1f}  res={rows[-1]['resolution_ranks']:7.1f} ranks"
                      f"  {res['wall_s']:.0f}s")
        base = float(np.mean([r["pct"] for r in rows if r["arm"] == "control"]))
        agg = {}
        for name, _, _ in arms:
            v = [r["pct"] for r in rows if r["arm"] == name]
            agg[name] = {"mean": float(np.mean(v)), "n": len(v),
                         "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                         "delta_vs_control": float(np.mean(v)) - base}
        out[ds] = {"rows": rows, "aggregate": agg, "control_mean": base}
        print(f"  [{ds}] best arm: " + max(agg.items(), key=lambda kv: kv[1]["mean"])[0]
              + f"  (control {base:.4f})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["all", "resolution", "ranking", "sweep", "confirm"])
    ap.add_argument("--datasets", nargs="+", default=["connectome"])
    ap.add_argument("--out", default=str(_OUT / "proto_amband.json"))
    args = ap.parse_args()

    t0 = time.time()
    report = {"config": CONFIG, "git_commit": git_commit()}
    if args.stage in ("all", "resolution"):
        report["resolution"] = stage_resolution()
    if args.stage in ("all", "ranking"):
        report["ranking"] = stage_ranking()
    if args.stage in ("all", "sweep"):
        report["sweep"] = stage_train(args.datasets, [CONFIG["sweep_seed"]], sweep=True)
    if args.stage == "confirm":
        report["confirm"] = stage_train(args.datasets, CONFIG["seeds"], sweep=True)

    report["wall_clock_s"] = time.time() - t0
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(dest.read_text()) if dest.exists() else {}
    prev.update(report)
    dest.write_text(json.dumps(prev, indent=2))
    print(f"\nwrote {dest}  ({report['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
