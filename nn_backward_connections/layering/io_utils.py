"""Load a dataset together with its champion MFAS order, parity-gated.

The champion ordering comes from the TRACKED artifact ``results/champions/*.npz``
(pinned by ``tools/pin_champion.py``, which re-scores it through the frozen
oracle before writing) - NOT from the gitignored per-run ``*_positions.npy`` -
so the layering study is reproducible by checkout on both machines.

Before returning, the order's feedforward percentage is re-computed with the
frozen oracle and compared against ``autoresearch/sota.json``'s
``pct_mean_exact`` for the dataset's sitting champion: a stale or wrong
artifact fails loudly rather than silently layering a non-champion order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Pinned champion orderings (int32 ranks; see tools/pin_champion.py).
CHAMPION_ORDER_NPZ = {
    "mouse": "results/champions/H63_mouse_s42_cuda.npz",
    "connectome": "results/champions/H64_connectome_s42_cuda.npz",
}

__all__ = ["REPO_ROOT", "CHAMPION_ORDER_NPZ", "load_graph_and_champion_order"]


def load_graph_and_champion_order(dataset: str = "mouse"
                                  ) -> Tuple["object", np.ndarray, np.ndarray, float]:
    """Return ``(g, order, rank, ff_pct)`` for the dataset's champion order.

    g       : GraphData with src/tgt/weight (see src.mfas.io)
    order   : order[k] = node at slot k of the pi line
    rank    : rank[v] = slot of node v (what the frozen oracle consumes)
    ff_pct  : feedforward percentage of this order, verified == sota.json's
              pct_mean_exact for the sitting champion (abs tol 1e-9)
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.mfas.io import load_dataset                       # read-only use
    from src.mfas.metrics import score_from_order, pct         # frozen oracle

    if dataset not in CHAMPION_ORDER_NPZ:
        raise KeyError(f"No pinned champion ordering registered for {dataset!r}; "
                       f"pin one with tools/pin_champion.py and add it to "
                       f"CHAMPION_ORDER_NPZ.")
    g = load_dataset(dataset)
    npz_path = REPO_ROOT / CHAMPION_ORDER_NPZ[dataset]
    with np.load(npz_path) as npz:
        rank = npz["rank"].astype(np.int64)
    if rank.shape[0] != g.n_nodes or set(np.unique(rank).tolist()) != set(range(g.n_nodes)):
        raise AssertionError(f"{npz_path} does not hold a permutation of "
                             f"[0, {g.n_nodes})")
    order = np.argsort(rank, kind="stable")

    sota = json.loads((REPO_ROOT / "autoresearch" / "sota.json").read_text(encoding="utf-8"))
    row = sota["datasets"][dataset]
    expected = float(row.get("pct_mean_exact", row["pct_mean"]))
    got = pct(score_from_order(rank, g.src, g.tgt, g.weight), g.total_weight)
    if abs(got - expected) > 1e-9:
        raise AssertionError(
            f"Champion-order parity FAILED for {dataset}: artifact scores "
            f"{got!r} but sota.json ({row['champion']}) says {expected!r}. "
            f"The pinned npz is stale or wrong - re-pin before layering.")
    return g, order, rank, got
