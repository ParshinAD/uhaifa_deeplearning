"""H64 - structural pins.

H64 is "the champion pipeline with ONE line changed". The whole verdict rests on that
claim, and nothing mechanical checked it before this file existed. These tests pin the two
things a reader has to take on trust otherwise:

1. **Every constant equals the champion of that dataset** (P14). connectome/microns take
   H42's; mouse takes H63's, which carries H44's ``_EPOCHS = 0`` and H52's stage 5.
2. **The surrogate really is the only difference**, and on mouse it is not even reached -
   so H64's mouse leg is bit-identical to H63 and its mouse delta is structurally 0.

These are cheap: only the mouse run touches a graph, and mouse runs 0 gradient steps.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mfas.experiments import H42, H63, H64
from mfas.experiments.H38 import _asym_surrogate
from mfas.io import load_mouse
from mfas.utils.seeding import select_device


def test_primaries_take_H42s_constants():
    """connectome and microns: H42 is the champion, so every constant must be H42's."""
    for ds in ("connectome", "microns"):
        assert H64._EPOCHS[ds] == H42._EPOCHS[ds]
        assert H64._MAX_SWEEPS[ds] == H42._MAX_SWEEPS[ds]
        assert H64._ALT_CYCLES[ds] == H42._ALT_CYCLES[ds]
        assert H64._ALT_SIFT_SWEEPS[ds] == H42._ALT_SIFT_SWEEPS[ds]
    assert H64._K_FULL == H42._K_FULL
    assert H64._ALPHA == H42._ALPHA
    assert H64._ALT_K_FULL == H42._ALT_K_FULL
    assert H64._MIN_BLOCK == H42._MIN_BLOCK


def test_primaries_do_not_get_the_mouse_champions_extra_stages():
    """H42 holds both primaries and contains neither stage 5 nor stage 6."""
    for ds in ("connectome", "microns"):
        assert H64._PAIR_MAX_POPS[ds] == 0, "stage 5 is not in the primaries' champion"
        assert H64._RECLAIM_ROUNDS[ds] == 0, "stage 6 is not in the primaries' champion"


def test_mouse_takes_H63s_constants():
    """mouse: H63 is the champion, so mouse must be H63's configuration exactly."""
    assert H64._EPOCHS["mouse"] == H63._EPOCHS["mouse"] == 0
    assert H64._MAX_SWEEPS["mouse"] == H63._MAX_SWEEPS["mouse"]
    assert H64._ALT_CYCLES["mouse"] == H63._ALT_CYCLES["mouse"]
    assert H64._PAIR_MAX_POPS["mouse"] == H63._PAIR_MAX_POPS["mouse"]
    assert H64._PAIR_PASSES == H63._PAIR_PASSES
    assert H64._RECLAIM_ROUNDS["mouse"] == H63._RECLAIM_ROUNDS["mouse"]
    assert H64._RECLAIM_BUDGET == H63._RECLAIM_BUDGET
    assert H64._CONFLICT_BUDGET == H63._CONFLICT_BUDGET


def test_microns_keeps_the_champions_80k_epochs_not_H63s_purchase():
    """H63 cut microns to 70,000 epochs (the H62 purchase), but H63 is NOT the microns
    champion - H42 is, at 80,000. H64 must compose on the champion, not on H63."""
    assert H64._EPOCHS["microns"] == 80_000
    assert H63._EPOCHS["microns"] == 70_000


def test_the_gradient_phase_uses_the_asymmetric_surrogate():
    """The one changed line: stage 2's surrogate is H38's, not torch.sigmoid.

    Checked behaviourally rather than by reading source: the ASYM shape is flat at 1 above
    the margin M and strictly below 1 beneath it, which the sigmoid is not.
    """
    src = H64._rocket_asym.__code__.co_names + H64._rocket_asym.__code__.co_freevars
    assert "_asym_surrogate" in src
    assert "sigmoid" not in src

    z = torch.tensor([H64.M + 1.0, H64.M + 10.0])
    assert torch.allclose(_asym_surrogate(z), torch.ones_like(z)), "flat above the margin"
    z0 = torch.tensor([0.0])
    assert float(_asym_surrogate(z0)) < 1.0, "g(0) < sup g, so collapse is not optimal"
    assert not torch.allclose(_asym_surrogate(z), torch.sigmoid(z))


def test_mouse_leg_is_bit_identical_to_the_mouse_champion():
    """0 epochs => the surrogate is never called => H64 IS H63 on mouse.

    This is the honest reading of H64's mouse leg and it is asserted, not assumed: the
    mouse tripwire is vacuous for this variant because it does not exercise the change.
    """
    device, _ = select_device("auto")
    g = load_mouse()
    r64 = H64.run(g, seed=42, device=device)
    r63 = H63.run(g, seed=42, device=device)

    assert r64.n_epochs_done == 0
    assert r64.history.attrs["surrogate_exercised"] is False
    assert r64.best_score == r63.best_score
    assert np.array_equal(r64.best_positions, r63.best_positions)


def test_gradient_budget_is_unchanged_versus_the_champion():
    """A knob swap, not extra compute: same epoch budget as the champion everywhere."""
    assert H64._EPOCHS["connectome"] == 20_000
    assert H64._EPOCHS["microns"] == 80_000
    assert H64._EPOCHS["mouse"] == 0


@pytest.mark.parametrize("attr", ["surrogate", "surrogate_M", "surrogate_T",
                                  "surrogate_exercised", "pure_best_pct",
                                  "base_best_pct", "refined_best_pct"])
def test_provenance_attrs_are_present(attr):
    """The stage attribution the audit reads must exist on every run."""
    device, _ = select_device("auto")
    g = load_mouse()
    r = H64.run(g, seed=42, device=device)
    assert attr in r.history.attrs
