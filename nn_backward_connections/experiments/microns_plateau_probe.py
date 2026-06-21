"""Find the MICrONS plateau budget: run baseline Rocket at several epoch budgets,
plus capture the within-run convergence trajectory of the longest run."""
import sys; sys.path.insert(0, ".")
import numpy as np
from src.mfas import io
from src.mfas.baseline.rocket import RocketConfig, run_rocket
from src.mfas.utils.seeding import seed_everything, select_device

g = io.load_dataset("microns")
dev, name = select_device("auto")
print("device:", name, flush=True)

# Full-run final scores at increasing budgets (each its own complete schedule).
for ep in (80000, 120000):
    seed_everything(42)
    res = run_rocket(g, RocketConfig(epochs=ep), seed=42, device=dev, time_limit=None)
    print(f"BUDGET {ep}: final best_pct={res.best_pct:.4f}% (score={res.best_score:,})", flush=True)
    if ep == 120000:
        h = res.history
        # print trajectory milestones to see where it flattens
        for frac in (0.25, 0.5, 0.625, 0.75, 0.875, 1.0):
            it = int(frac * ep)
            row = h.iloc[(h["iter"] - it).abs().argmin()]
            print(f"   traj iter~{it}: best_pct={row['best_pct']:.4f}%", flush=True)
