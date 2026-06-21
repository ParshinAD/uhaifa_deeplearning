"""Variant: baseline passthrough — the UNCHANGED Rocket baseline, wrapped as a variant.

This introduces NO algorithmic change. Its sole purpose is to prove the research-loop
wiring (run_variant -> verifier -> critic -> log) end to end: running it through the new
pipeline must reproduce the frozen baseline numbers within the noise floor on both
datasets. It is the dry-run / smoke-test variant, not a research hypothesis.
"""
from __future__ import annotations

from typing import Optional

from ..baseline.rocket import RocketConfig, RocketResult, run_rocket
from ..io import GraphData

ID = "baseline_passthrough"
HYPOTHESIS = "Unchanged baseline Rocket, wrapped as a variant (wiring smoke test)."

# Per-dataset epoch budgets mirror configs/baseline_rocket.yaml so the passthrough
# reproduces the logged baseline (connectome 20k, mouse 5k).
_EPOCHS = {"connectome": 20_000, "mouse": 5_000}


def run(g: GraphData, seed: int, device, time_limit: Optional[float] = None
        ) -> RocketResult:
    """Run the unchanged baseline Rocket with default hyperparameters."""
    cfg = RocketConfig(epochs=_EPOCHS.get(g.name, RocketConfig.epochs))
    return run_rocket(g, cfg, seed=seed, device=device, time_limit=time_limit)
