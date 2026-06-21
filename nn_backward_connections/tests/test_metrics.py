"""Tests for the oracle: known-answer scorer checks + deterministic parity gate.

GOVERNANCE: these tests pin down the ground-truth metric. Once they pass, the
scorer (``src/mfas/metrics.py``) must not change. See CLAUDE.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mfas import io
from mfas.metrics import (
    feedback_weight,
    pct,
    positions_to_order,
    score_from_order,
    score_from_positions,
)

ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────────────
# Known-answer test on a tiny hand-built graph
# ──────────────────────────────────────────────────────────────────────────────
# 4 nodes (0,1,2,3). Edges (src -> tgt, weight):
#   0->1 : 5      0->2 : 3      1->2 : 2
#   2->0 : 4      3->1 : 1      2->3 : 6
# total = 5+3+2+4+1+6 = 21
# With identity positions [0,1,2,3], an edge is feedforward iff pos[tgt] > pos[src]:
#   0->1 (1>0) +5 | 0->2 (2>0) +3 | 1->2 (2>1) +2 | 2->0 (0>2) x
#   3->1 (1>3) x  | 2->3 (3>2) +6
# feedforward = 5+3+2+6 = 16 ; pct = 16/21 = 76.190476%
_SRC = np.array([0, 0, 1, 2, 3, 2])
_TGT = np.array([1, 2, 2, 0, 1, 3])
_W = np.array([5, 3, 2, 4, 1, 6], dtype=np.int64)
_TOTAL = 21
_FORWARD = 16


def test_known_answer_positions():
    pos = np.array([0.0, 1.0, 2.0, 3.0])
    assert score_from_positions(pos, _SRC, _TGT, _W) == 16.0
    assert abs(pct(16.0, _TOTAL) - 76.19047619047619) < 1e-9


def test_known_answer_order():
    order = np.array([0, 1, 2, 3])
    assert score_from_order(order, _SRC, _TGT, _W) == 16.0


def test_reverse_is_complement():
    """Reversing positions turns feedforward into feedback: 21 - 16 = 5.

    forward + reverse == total only when there are no ties — guards the strict
    ``>`` direction and that no edge is double-counted.
    """
    pos = np.array([0.0, 1.0, 2.0, 3.0])
    rev = pos[::-1].copy()
    assert score_from_positions(rev, _SRC, _TGT, _W) == 5.0
    assert score_from_positions(pos, _SRC, _TGT, _W) + \
        score_from_positions(rev, _SRC, _TGT, _W) == _TOTAL
    assert feedback_weight(16.0, _TOTAL) == 5.0


def test_ties_are_not_feedforward():
    """Equal positions => edge is NOT feedforward (strict ``>``)."""
    # one edge 0->1 with both nodes at the same position
    src = np.array([0])
    tgt = np.array([1])
    w = np.array([10], dtype=np.int64)
    pos_tie = np.array([1.0, 1.0])
    assert score_from_positions(pos_tie, src, tgt, w) == 0.0
    pos_ff = np.array([0.0, 1.0])
    assert score_from_positions(pos_ff, src, tgt, w) == 10.0


def test_float32_weights_summed_exactly():
    """float32 weights must be upcast so the sum is exact, not lossy."""
    # 10 million unit edges all feedforward; float32 cannot represent the exact
    # sum but our scorer upcasts to int64/float64.
    n = 10_000_001
    src = np.zeros(n, dtype=np.int64)
    tgt = np.ones(n, dtype=np.int64)
    w = np.ones(n, dtype=np.float32)
    pos = np.array([0.0, 1.0])
    assert score_from_positions(pos, src, tgt, w) == float(n)


def test_positions_to_order():
    pos = np.array([3.0, 1.0, 2.0, 0.0])
    # smallest position (node 3) -> rank 0
    np.testing.assert_array_equal(positions_to_order(pos), np.array([3, 1, 2, 0]))


# ──────────────────────────────────────────────────────────────────────────────
# Remap correctness: searchsorted == dict-comprehension
# ──────────────────────────────────────────────────────────────────────────────
def test_remap_matches_dict_comprehension():
    rng = np.random.RandomState(0)
    src_raw = rng.randint(0, 10_000, size=500).astype(np.int64) * 7  # sparse-ish ids
    tgt_raw = rng.randint(0, 10_000, size=500).astype(np.int64) * 7
    src_idx, tgt_idx, node_ids = io.remap_node_ids(src_raw, tgt_raw)

    nodes = np.unique(np.concatenate([src_raw, tgt_raw]))
    node2idx = {n: i for i, n in enumerate(nodes)}
    exp_src = np.array([node2idx[n] for n in src_raw])
    exp_tgt = np.array([node2idx[n] for n in tgt_raw])

    np.testing.assert_array_equal(src_idx, exp_src)
    np.testing.assert_array_equal(tgt_idx, exp_tgt)
    np.testing.assert_array_equal(node_ids, nodes)


# ──────────────────────────────────────────────────────────────────────────────
# SCORER PARITY — the hard, deterministic reproduction gate
# ──────────────────────────────────────────────────────────────────────────────
# Scoring the saved best-position artifact with the exact (int64) scorer MUST yield
# 34,751,902 on total weight 41,912,141 (= 82.9161%). The notebook's historical
# 34,751,904 / 41,912,104 were float32 rounding noise (verified during planning).
_ARTIFACT = ROOT / "results" / "rocket_best_positions.npy"
_PARITY_SCORE = 34_751_902
_PARITY_TOTAL = 41_912_141


@pytest.mark.skipif(not _ARTIFACT.exists(),
                    reason="rocket_best_positions.npy artifact not present")
def test_scorer_parity_connectome():
    g = io.load_dataset("connectome")
    pos = np.load(_ARTIFACT)
    assert pos.shape == (g.n_nodes,), \
        f"artifact length {pos.shape} != n_nodes {g.n_nodes}"
    assert g.total_weight == _PARITY_TOTAL
    score = score_from_positions(pos, g.src, g.tgt, g.weight)
    assert int(score) == _PARITY_SCORE, \
        f"expected {_PARITY_SCORE:,}, got {score:,.0f}"
    assert abs(pct(score, g.total_weight) - 82.91607436613654) < 1e-6
