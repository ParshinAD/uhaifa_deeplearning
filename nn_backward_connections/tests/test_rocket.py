"""Algorithm-parity tests for the ported Rocket baseline.

Exact trajectory parity is impossible on MPS (nondeterministic kernels). So we
assert:
  (1) CPU-determinism: two identical CPU runs give the identical best_score
      (guards that our port is internally reproducible);
  (2) a convergence floor on a short connectome run (guards against gross
      optimizer-port regressions).

The authoritative exact reproduction lives in ``test_metrics.py`` (scorer parity).
These are marked ``slow`` and skipped if the dataset is unavailable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mfas import io
from mfas.baseline.rocket import RocketConfig, make_beta_schedule, run_rocket
from mfas.utils.seeding import seed_everything, select_device

ROOT = Path(__file__).resolve().parent.parent
_CONNECTOME = ROOT / "data"


def test_beta_schedule_range():
    betas = make_beta_schedule(20_000, 5)
    assert abs(betas.min() - 0.05) < 1e-6
    assert abs(betas.max() - 1.05) < 1e-6
    assert betas.shape == (20_000,)


def _connectome_available() -> bool:
    return (_CONNECTOME / "connectome_graph.csv.gz").exists() or \
           (_CONNECTOME / "raw" / "connectome_graph.csv.gz").exists()


@pytest.mark.slow
@pytest.mark.skipif(not _connectome_available(), reason="connectome data missing")
def test_cpu_determinism_small():
    """Two identical CPU runs (200 epochs) must give the identical best_score."""
    import torch

    g = io.load_dataset("connectome")
    cfg = RocketConfig(epochs=200, log_interval=50)
    dev = torch.device("cpu")

    seed_everything(42)
    r1 = run_rocket(g, cfg, seed=42, device=dev)
    seed_everything(42)
    r2 = run_rocket(g, cfg, seed=42, device=dev)
    assert r1.best_score == r2.best_score


@pytest.mark.slow
@pytest.mark.skipif(not _connectome_available(), reason="connectome data missing")
def test_convergence_floor():
    """A short run (2000 epochs) clears a conservative feedforward floor."""
    g = io.load_dataset("connectome")
    cfg = RocketConfig(epochs=2000, log_interval=100)
    dev, _ = select_device("auto")
    seed_everything(42)
    r = run_rocket(g, cfg, seed=42, device=dev, time_limit=180)
    assert r.best_pct >= 70.0, f"only reached {r.best_pct:.2f}%"
    # best_pct is non-decreasing by construction
    bp = r.history["best_pct"].to_numpy()
    assert np.all(np.diff(bp) >= -1e-9)
