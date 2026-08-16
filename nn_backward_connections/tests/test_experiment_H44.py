"""Shippability tests for variant H44 (mouse gradient phase sized to zero).

H44's whole claim is "H42 with ONE constant changed". Four things have to hold before a
screen can adjudicate it, and each is checked here rather than trusted:

1. **Isolation** — every constant H44 shares with H42 is equal, and ``_EPOCHS`` differs in
   the mouse entry ONLY. A variant that changes two things at once cannot be adjudicated.
2. **Identity on the primaries** — because ``_EPOCHS`` is untouched for connectome and
   microns, H44 must BE H42 there. This is asserted on the constants (a full primary run
   is a screen's job, not a unit test's) so that a later edit which quietly perturbs a
   primary cannot pass unnoticed.
3. **The mouse claim reproduces exactly** — 93.08288021668459, the pre-registered number
   from ``experiments/outputs/proto_P07_mouse.json``, and it beats the champion's
   92.91701410211007.
4. **Determinism** — the campaign's 1-seed screen policy and ``sota.json``'s std = 0 both
   rest on the pipeline never drawing from ``seed``. With ``epochs = 0`` on mouse there is
   no gradient phase to draw at all, so this is strictly stronger than for H42; it is
   checked anyway, because "obviously true" is how silent breakage survives.

Everything here runs on mouse (148 nodes) on the CPU, in seconds. Connectome and MICrONS
are NEVER touched by a unit test.

On the strength of claim 3
--------------------------
That H44 > H42 on mouse is an EMPIRICAL fact about this graph, not a theorem: the
refinement stack is provably non-monotone in the quality of its input (see
``experiments/log.md`` 2026-08-16 — on connectome, 20k -> 40k epochs improves stage 3 by
+0.03314 pp while the composed score falls 0.00666 pp). H44 exploits that non-monotonicity;
it does not repeal it. The exact-value assertion below is a regression pin on a measured
number, and is documented as such so nobody reads it as a proof that dropping Rocket helps.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mfas.experiments import H42, H44
from mfas.io import load_dataset
from mfas.metrics import score_from_positions

# Pre-registered in autoresearch/queue.json (H44 kill_condition) BEFORE the settling run,
# and reproduced exactly by experiments/outputs/proto_P07_mouse.json.
MOUSE_H44_PCT = 93.08288021668459
MOUSE_CHAMPION_PCT = 92.91701410211007


@pytest.fixture(scope="module")
def mouse():
    return load_dataset("mouse")


# ── 1. Isolation: exactly one constant differs, and it is the mouse epoch count ──────
def test_only_the_mouse_epoch_count_differs():
    assert H44._EPOCHS["mouse"] == 0, "H44's whole mechanism is the mouse epoch count"
    assert H42._EPOCHS["mouse"] == 5_000, "H42's mouse epochs changed — re-derive H44"
    assert set(H44._EPOCHS) == set(H42._EPOCHS)


@pytest.mark.parametrize("name", ["_MAX_SWEEPS", "_ALT_CYCLES", "_ALT_SIFT_SWEEPS"])
def test_shared_dicts_are_identical(name):
    assert getattr(H44, name) == getattr(H42, name)


@pytest.mark.parametrize("name", ["_K_FULL", "_ALPHA", "_ALT_K_FULL", "_MIN_BLOCK"])
def test_shared_scalars_are_identical(name):
    assert getattr(H44, name) == getattr(H42, name)


# ── 2. Identity on the primaries — H44 IS the champion there ─────────────────────────
@pytest.mark.parametrize("ds", ["connectome", "microns"])
def test_primaries_are_untouched(ds):
    """H44 must be H42 on both primaries: same epochs, hence the same code path.

    Asserted on constants rather than by running: a primary run costs 21-53 minutes and is
    the screen's job. What a unit test can guarantee is that no future edit silently moves
    a primary while claiming to be a mouse-only change.
    """
    assert H44._EPOCHS[ds] == H42._EPOCHS[ds]
    assert H44._MAX_SWEEPS[ds] == H42._MAX_SWEEPS[ds]
    assert H44._ALT_CYCLES[ds] == H42._ALT_CYCLES[ds]
    assert H44._ALT_SIFT_SWEEPS[ds] == H42._ALT_SIFT_SWEEPS[ds]


# ── 3. The mouse claim, reproduced exactly ───────────────────────────────────────────
def test_mouse_reproduces_the_preregistered_score(mouse):
    res = H44.run(mouse, seed=42, device=torch.device("cpu"))
    got = 100.0 * res.best_score / mouse.total_weight
    assert got == pytest.approx(MOUSE_H44_PCT, abs=1e-9), (
        f"H44 on mouse gave {got!r}, expected the pre-registered {MOUSE_H44_PCT!r}")
    assert got > MOUSE_CHAMPION_PCT


def test_no_gradient_steps_are_taken_on_mouse(mouse):
    """epochs = 0 must mean literally zero optimizer steps, not 'a few'."""
    res = H44.run(mouse, seed=42, device=torch.device("cpu"))
    assert res.n_epochs_done == 0
    assert res.history.attrs["epochs_requested"] == 0


def test_reported_score_matches_the_frozen_scorer(mouse):
    """The oracle, not the variant, defines the number that gets reported."""
    res = H44.run(mouse, seed=42, device=torch.device("cpu"))
    exact = score_from_positions(res.best_positions, np.asarray(mouse.src),
                                 np.asarray(mouse.tgt), mouse.weight)
    assert exact == res.best_score


# ── 4. Determinism ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", [42, 123, 999])
def test_seed_does_not_change_the_result(mouse, seed):
    res = H44.run(mouse, seed=seed, device=torch.device("cpu"))
    assert res.best_score == pytest.approx(
        MOUSE_H44_PCT * mouse.total_weight / 100.0, abs=1e-6)
