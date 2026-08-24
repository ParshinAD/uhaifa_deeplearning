"""H38 follow-up — is the asymmetric margin arm's optimum a real plateau or a grid artifact?

The coarse gate (`proto_h38_asym.py`) found the one-sided surrogate

    g(z) = 1                      for z >= m
    g(z) = 1 + tanh((z-m)/T)      for z <  m

wins on mouse at (m=0.5, T=1.5) by +0.56 pp but LOSES badly at m=2 (-2.9) and m=5 (-5.7). A
narrow optimum sitting right next to a known degeneracy (m -> 0 is total collapse, verified:
final position std 0.0005 on the hard synthetic) is exactly the shape of an artifact, so before
spending connectome compute this script asks three things:

  1. **Is there a PLATEAU in (m, T) or a knife edge?** Fine grid, both proxies.
  2. **Does the mouse gain survive more seeds?** Mouse runs cost ~1 s, so the winner is re-run
     at 10 seeds instead of 3.
  3. **Is the win just "operate at a smaller scale"?** The winner's final position std is 9.7 vs
     the sigmoid's 25.1, and Q01 established that F depends only on the product beta*std — so a
     scale change alone is the killed A-SCALE/H03 axis. The control is the SIGMOID run with beta
     rescaled to land at a comparable beta*std; if that control reproduces the gain, the
     asymmetry is not the mechanism.

Leakage: no oracle read inside any surrogate. Writes to `experiments/outputs/` only.

Run (env `allen`, from the repo root):
    PYTHONPATH=src python experiments/proto_h38_sweep.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from mfas.analysis.gap import make_hard_synthetic_graph
from mfas.baseline.rocket import RocketConfig
from mfas.io import load_dataset

from proto_h38_asym import _asym, run_arm            # reuse the verified runner

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "experiments" / "outputs"

CONFIG = {
    "grid_m": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5],
    "grid_T": [0.75, 1.5, 3.0],
    "seeds_grid": [42, 123, 999],
    "seeds_mouse_deep": [42, 123, 999, 7, 31415, 2718, 1618, 1414, 1732, 2236],
    "epochs": {"mouse": 5_000, "hard_synthetic": 8_000},
    # scale control: sigmoid with beta multiplied by these factors (beta*std is the only
    # thing F depends on, so this sweeps the SAME axis a scale change would).
    "sigmoid_beta_factors": [0.1, 0.25, 0.5, 2.0, 4.0],
    "synthetic": {"n": 400, "avg_out": 10, "feedback_frac": 0.65, "n_clusters": 8,
                  "intra_cycle_frac": 0.55, "weight_alpha": 2.0, "seed": 0},
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_ROOT, text=True).strip()
    except Exception:                                     # pragma: no cover
        return "unknown"


def bench(g, fn, seeds, epochs):
    pcts, stds = [], []
    for s in seeds:
        r = run_arm(g, RocketConfig(epochs=epochs), s, fn)
        pcts.append(r["best_pct"])
        stds.append(r["final_std"])
    return {"per_seed_pct": pcts, "mean_pct": float(np.mean(pcts)),
            "std_pct": float(np.std(pcts, ddof=1)),
            "mean_final_position_std": float(np.mean(stds))}


def main() -> None:
    report = {"config": CONFIG, "git_commit": git_commit()}
    gm = load_dataset("mouse")
    sc = CONFIG["synthetic"]
    gh, _o, ref_pct = make_hard_synthetic_graph(
        n=sc["n"], avg_out=sc["avg_out"], feedback_frac=sc["feedback_frac"],
        n_clusters=sc["n_clusters"], intra_cycle_frac=sc["intra_cycle_frac"],
        weight_alpha=sc["weight_alpha"], seed=sc["seed"])
    graphs = {"mouse": gm, "hard_synthetic": gh}

    base = {}
    for gname, g in graphs.items():
        base[gname] = bench(g, torch.sigmoid, CONFIG["seeds_grid"],
                            CONFIG["epochs"][gname])
        print(f"sigmoid baseline {gname}: {base[gname]['mean_pct']:.4f} "
              f"+/- {base[gname]['std_pct']:.4f} (pos std "
              f"{base[gname]['mean_final_position_std']:.2f})")
    report["sigmoid_baseline"] = base

    # ── 1. the (m, T) grid ───────────────────────────────────────────────────
    grid = {}
    for gname, g in graphs.items():
        grid[gname] = {}
        print(f"\n=== (m,T) GRID on {gname} — delta vs sigmoid, pp "
              f"(3 seeds; pos std in parens) ===")
        header = "   m\\T  " + "".join(f"{T:>22}" for T in CONFIG["grid_T"])
        print(header)
        for m in CONFIG["grid_m"]:
            row = f"{m:>7} "
            for T in CONFIG["grid_T"]:
                r = bench(g, (lambda z, m=m, T=T: _asym(z, m, T)),
                          CONFIG["seeds_grid"], CONFIG["epochs"][gname])
                grid[gname][f"m{m}_T{T}"] = r
                d = r["mean_pct"] - base[gname]["mean_pct"]
                row += f"{d:+13.4f}({r['mean_final_position_std']:6.2f})"
            print(row)
    report["grid"] = grid

    # ── 2. deep mouse re-run of the grid winner ──────────────────────────────
    winner = max(grid["mouse"], key=lambda k: grid["mouse"][k]["mean_pct"])
    m_w = float(winner.split("_")[0][1:])
    T_w = float(winner.split("_")[1][1:])
    deep = bench(gm, (lambda z: _asym(z, m_w, T_w)), CONFIG["seeds_mouse_deep"],
                 CONFIG["epochs"]["mouse"])
    deep_base = bench(gm, torch.sigmoid, CONFIG["seeds_mouse_deep"],
                      CONFIG["epochs"]["mouse"])
    report["mouse_deep"] = {"winner": winner, "m": m_w, "T": T_w,
                            "arm": deep, "sigmoid": deep_base,
                            "delta": deep["mean_pct"] - deep_base["mean_pct"]}
    print(f"\nDEEP MOUSE ({len(CONFIG['seeds_mouse_deep'])} seeds), winner={winner}:")
    print(f"  arm     {deep['mean_pct']:.4f} +/- {deep['std_pct']:.4f}")
    print(f"  sigmoid {deep_base['mean_pct']:.4f} +/- {deep_base['std_pct']:.4f}")
    print(f"  delta   {deep['mean_pct'] - deep_base['mean_pct']:+.4f} pp")

    # ── 3. the SCALE control: is this just a smaller beta*std? ───────────────
    print("\nSCALE CONTROL — plain sigmoid at rescaled beta (the A-SCALE/H03 axis):")
    scale = {}
    for gname, g in graphs.items():
        scale[gname] = {}
        for f in CONFIG["sigmoid_beta_factors"]:
            r = bench(g, (lambda z, f=f: torch.sigmoid(f * z)), CONFIG["seeds_grid"],
                      CONFIG["epochs"][gname])
            scale[gname][f"beta_x{f}"] = r
            print(f"  {gname:>15} beta x{f:<5}: {r['mean_pct']:8.4f} "
                  f"(delta {r['mean_pct'] - base[gname]['mean_pct']:+.4f}, "
                  f"pos std {r['mean_final_position_std']:.2f})")
    report["scale_control"] = scale

    _OUT.mkdir(parents=True, exist_ok=True)
    dest = _OUT / "proto_h38_sweep.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
