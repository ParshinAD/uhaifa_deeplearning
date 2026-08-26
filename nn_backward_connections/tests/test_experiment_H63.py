"""Shippability tests for variant H63 (arc reclamation on the PER-DATASET champion).

H63 makes three claims that a screen cannot check for itself, so they are checked here:

1. **Per-dataset champion composition (P14).** ``sota.json`` holds H42 / H42 / H52. So on
   the primaries H63's stages 1-5 must be H42's — same constants, stage 5 absent — and on
   mouse they must be H52's. Anything else and the screen's comparator is moving.
2. **The microns purchase is the only constant that differs from H42 on a primary (H62).**
   ``_EPOCHS["microns"]`` is 70,000 against the champion's 80,000, deliberately, and every
   other shared constant is equal.
3. **Stage 6 is monotone and reaches a fixed point**, on the real pipeline, end to end.

Everything here runs on mouse (148 nodes) on the CPU, in seconds. Connectome and MICrONS are
NEVER touched by a unit test. The claim that ``reclaim2`` computes the same arcs as the
reference lives in ``tests/test_reclaim2.py``; the claim that it does so on the primaries is
evidence, not a test, and lives in ``experiments/outputs/proto_H63_<ds>.json``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mfas.experiments import H42, H52, H63
from mfas.io import load_dataset
from mfas.metrics import score_from_positions


@pytest.fixture(scope="module")
def mouse():
    return load_dataset("mouse")


@pytest.fixture(scope="module")
def h63_mouse(mouse):
    return H63.run(mouse, seed=42, device=torch.device("cpu"), time_limit=None)


@pytest.fixture(scope="module")
def h52_mouse(mouse):
    return H52.run(mouse, seed=42, device=torch.device("cpu"), time_limit=None)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Composition: stages 1-5 are the champion's, per dataset
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("const", [
    "_MAX_SWEEPS", "_K_FULL", "_ALPHA", "_ALT_CYCLES", "_ALT_SIFT_SWEEPS",
    "_ALT_K_FULL", "_MIN_BLOCK", "_PAIR_PASSES",
])
def test_shared_constants_match_the_champion(const):
    assert getattr(H63, const) == getattr(H52, const), const


def test_primary_epochs_match_h42_except_the_microns_purchase():
    assert H63._EPOCHS["connectome"] == H42._EPOCHS["connectome"] == 20_000
    # THE deliberate difference, and the only one on a primary: H62 buys the seconds for
    # stage 6 out of the Rocket epoch budget. If this ever equals H42's, the microns leg no
    # longer fits its guard deadline and the test should fail loudly rather than silently.
    assert H42._EPOCHS["microns"] == 80_000
    assert H63._EPOCHS["microns"] == 70_000


def test_mouse_epochs_match_the_mouse_champion():
    """Mouse's champion is H52, not H42 — this is exactly the P14 defect H59 tripped on."""
    assert H63._EPOCHS["mouse"] == H52._EPOCHS["mouse"] == 0
    assert H63._EPOCHS["mouse"] != H42._EPOCHS["mouse"]


def test_stage5_is_on_for_mouse_only_and_matches_h52_there():
    assert H63._PAIR_MAX_POPS["mouse"] == H52._PAIR_MAX_POPS["mouse"] == 100_000
    assert H63._PAIR_MAX_POPS["connectome"] == 0        # H42 has no stage 5
    assert H63._PAIR_MAX_POPS["microns"] == 0


def test_reclamation_is_on_for_every_dataset():
    """H59 shipped microns at 0 rounds for want of seconds; H63's whole point is that it no
    longer has to."""
    assert set(H63._RECLAIM_ROUNDS) == {"connectome", "microns", "mouse"}
    assert all(v == 1 for v in H63._RECLAIM_ROUNDS.values())


# ──────────────────────────────────────────────────────────────────────────────
# 2. End-to-end on mouse
# ──────────────────────────────────────────────────────────────────────────────
def test_runs_and_reports_the_stage_split(h63_mouse):
    a = h63_mouse.history.attrs
    for k in ("base_best_pct", "reclaim_best_pct", "reclaim_increment_pp",
              "reclaim_weight_accepted", "reclaim_lemma_holds", "reclaim_time_s"):
        assert k in a, k
    assert a["reclaim_lemma_holds"] is True
    assert a["reclaim_best_pct"] == pytest.approx(h63_mouse.best_pct)


def test_stage6_never_loses_to_its_own_input(h63_mouse):
    """Monotone BY CONSTRUCTION — stage 6 is kept only if it scores at least its input."""
    a = h63_mouse.history.attrs
    assert a["reclaim_best_score"] >= a["base_best_score"]
    assert a["reclaim_increment_pp"] >= 0.0


def test_mouse_base_is_the_h52_champion_and_stage6_improves_on_it(h63_mouse, h52_mouse):
    """The P14 fix, checked on the real pipeline: H63's pre-reclaim mouse score IS H52's."""
    assert h63_mouse.history.attrs["base_best_score"] == h52_mouse.best_score
    assert h63_mouse.best_score >= h52_mouse.best_score


def test_reclaim_gain_equals_the_score_it_actually_realises(h63_mouse, mouse):
    """No number in the attrs is allowed to be an estimate: re-score with the frozen oracle."""
    got = score_from_positions(h63_mouse.best_positions, mouse.src, mouse.tgt, mouse.weight)
    assert got == h63_mouse.best_score


def test_stage6_reaches_a_fixed_point_on_mouse(h63_mouse, mouse):
    """A second round must find nothing — that is the certificate that the arc set is
    minimal, and it is why _RECLAIM_ROUNDS is 1 rather than a tuned number."""
    from mfas.refine.reclaim2 import reclaim_arcs_fast

    rank = np.argsort(np.argsort(h63_mouse.best_positions, kind="stable"),
                      kind="stable").astype(np.int64)
    out, log = reclaim_arcs_fast(mouse, rank, rounds=1)
    assert log[0]["n_accepted"] == 0
    assert np.array_equal(out, rank)


def test_deterministic_across_seeds_on_mouse(mouse):
    """No stage draws from `seed`. The classifier still calls H63 `rng` (pair_relocate reads
    the wall clock, which it cannot resolve statically) and the screen obeys the classifier —
    but the underlying pipeline is seed-invariant and that is worth pinning."""
    dev = torch.device("cpu")
    a = H63.run(mouse, seed=42, device=dev, time_limit=None)
    b = H63.run(mouse, seed=999, device=dev, time_limit=None)
    assert np.array_equal(a.best_positions, b.best_positions)
    assert a.best_score == b.best_score
