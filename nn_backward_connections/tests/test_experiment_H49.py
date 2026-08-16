"""Shippability tests for variant H49 (ratio-greedy warm start).

H49's claim is "the champion pipeline with ONE thing changed — the initial order". What has to
hold before a screen can adjudicate it:

1. **Isolation** — every constant H49 shares with H44 (the current mouse champion, itself
   byte-identical to H42 on both primaries) is equal, so ``H49 - champion`` isolates the init.
2. **The init is a genuine rank vector** — a permutation of ``[0, n)``. The convention matters:
   :func:`mfas.experiments.H02.greedy_fas_order` returns a RANK despite its name, and feeding a
   true ORDER into a rank-expecting path is silent, not loud (it scored our mouse greedy at
   43.797% instead of 90.126% during development). This is pinned so the confusion cannot
   recur.
3. **Determinism** — no RNG anywhere, so repeated calls are bit-identical and the campaign's
   1-seed screen policy and ``sota.json``'s std = 0 keep holding.
4. **The measured init figures reproduce** — the numbers H49's docstring and queue entry quote
   are regression-pinned on mouse, so a future edit to the peel cannot quietly change them.

Everything here runs on mouse (148 nodes) on the CPU, in seconds. Connectome and MICrONS are
NEVER touched by a unit test.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mfas.baseline.ratio_greedy import ratio_greedy_rank
from mfas.experiments import H02, H42, H44, H49
from mfas.io import load_dataset
from mfas.metrics import score_from_order, score_from_positions, pct

# Measured 2026-08-16, read verbatim out of experiments/outputs/proto_H48_mouse.json
# (`ratio_pct` / `ours_pct`), frozen scorer. Copied from the artifact, never transcribed from a
# rounded console line — an earlier version of this file carried 88.38255751047947, which was
# the printed 88.38256 with digits invented after it, and this test is what caught that.
MOUSE_RATIO_INIT_PCT = 88.38256063787692
MOUSE_GREEDY_INIT_PCT = 90.12629794323948


@pytest.fixture(scope="module")
def mouse():
    return load_dataset("mouse")


# ── 1. Isolation ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["_EPOCHS", "_MAX_SWEEPS", "_ALT_CYCLES", "_ALT_SIFT_SWEEPS"])
def test_shared_dicts_match_the_champion(name):
    assert getattr(H49, name) == getattr(H44, name)


@pytest.mark.parametrize("name", ["_K_FULL", "_ALPHA", "_ALT_K_FULL", "_MIN_BLOCK"])
def test_shared_scalars_match_the_champion(name):
    assert getattr(H49, name) == getattr(H44, name)


def test_primaries_share_the_h42_constants():
    """H44 is H42 on both primaries, so H49 must be too — only the init may differ."""
    for ds in ("connectome", "microns"):
        assert H49._EPOCHS[ds] == H42._EPOCHS[ds]
        assert H49._ALT_CYCLES[ds] == H42._ALT_CYCLES[ds]


# ── 2. The init is a rank vector, not an order ───────────────────────────────────────
def test_ratio_greedy_returns_a_permutation(mouse):
    rank = ratio_greedy_rank(mouse)
    assert rank.dtype == np.int64
    assert rank.shape == (mouse.n_nodes,)
    assert np.array_equal(np.sort(rank), np.arange(mouse.n_nodes))


def test_rank_convention_matches_greedy_fas(mouse):
    """Both inits must be interpretable by the SAME scoring call, or one of them is scrambled."""
    a = np.asarray(H02.greedy_fas_order(mouse), dtype=np.int64)
    b = ratio_greedy_rank(mouse)
    src, tgt = np.asarray(mouse.src), np.asarray(mouse.tgt)
    pa = pct(score_from_order(a, src, tgt, mouse.weight), mouse.total_weight)
    pb = pct(score_from_order(b, src, tgt, mouse.weight), mouse.total_weight)
    assert pa == pytest.approx(MOUSE_GREEDY_INIT_PCT, abs=1e-9)
    assert pb == pytest.approx(MOUSE_RATIO_INIT_PCT, abs=1e-9)


# ── 3. Determinism ───────────────────────────────────────────────────────────────────
def test_init_is_deterministic(mouse):
    assert np.array_equal(ratio_greedy_rank(mouse), ratio_greedy_rank(mouse))


@pytest.mark.parametrize("seed", [42, 123])
def test_pipeline_is_seed_independent(mouse, seed):
    a = H49.run(mouse, seed=seed, device=torch.device("cpu"))
    b = H49.run(mouse, seed=seed + 1, device=torch.device("cpu"))
    assert a.best_score == b.best_score


# ── 4. Oracle agreement ──────────────────────────────────────────────────────────────
def test_reported_score_matches_the_frozen_scorer(mouse):
    res = H49.run(mouse, seed=42, device=torch.device("cpu"))
    exact = score_from_positions(res.best_positions, np.asarray(mouse.src),
                                 np.asarray(mouse.tgt), mouse.weight)
    assert exact == res.best_score
