"""Seeding and device selection.

Reproducibility note
--------------------
We seed Python's ``random``, NumPy and PyTorch (and CUDA if present). On Apple
**MPS**, kernels (notably reductions and large-magnitude float32 index ops) are
**not guaranteed deterministic** regardless of the seed — so run-to-run scores can
jitter by a fraction of a percentage point. The authoritative reproduction claim
therefore rests on the deterministic scorer-parity test, not on bit-identical
training trajectories. See CLAUDE.md / RESEARCH_BRIEF.md.
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np

__all__ = ["seed_everything", "select_device", "mps_determinism_note"]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch always present in `allen`
        pass


def select_device(prefer: str = "auto") -> Tuple["object", str]:
    """Select a torch device. Priority MPS > CUDA > CPU (matches the notebook).

    Parameters
    ----------
    prefer : {"auto", "mps", "cuda", "cpu"}
        ``"auto"`` picks the best available; otherwise forces the named device
        (falling back to CPU if unavailable).

    Returns
    -------
    (torch.device, human_name)
    """
    import torch

    if prefer == "cpu":
        return torch.device("cpu"), "CPU"
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple MPS"

    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple MPS"
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    return torch.device("cpu"), "CPU"


def mps_determinism_note() -> str:
    """Return the standard caveat string for logging into result records."""
    return ("MPS kernels are not guaranteed deterministic; discrete scores are "
            "computed on CPU in int64/float64 and the exact reproduction claim "
            "rests on scorer-parity, not training-trajectory parity.")
