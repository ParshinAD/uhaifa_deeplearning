"""Shippability tests for variant H52 (sequential pair relocation as stage 5).

H52's claim is "the champion pipeline with ONE stage added". What must hold before a screen can
adjudicate it:

1. **Isolation** — every constant H52 shares with H44 (the current mouse champion, itself
   byte-identical to H42 on both primaries) is equal, so ``H52 - champion`` isolates stage 5.
2. **Stage 5 is monotone** — it is best-by-oracle over strictly-positive exact gains, so the
   variant can never score below the champion's own stage-4 order. That is a theorem and is
   asserted as one; it is what makes a screen failure impossible to confuse with a regression.
3. **The budget is a POP COUNT, not a clock** — so two runs are bit-identical. The campaign's
   1-seed screen policy and ``sota.json``'s std = 0 depend on it, and P05 established that a
   time-sized stage would break exactly that.
4. **Provenance of the reported number** — the frozen oracle, not the variant, defines it.

Everything runs on mouse (148 nodes) on the CPU, in seconds. Connectome and MICrONS are NEVER
touched by a unit test.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mfas.experiments import H42, H44, H52
from mfas.io import load_dataset
from mfas.metrics import score_from_positions

# Read VERBATIM out of experiments/outputs/proto_H51_mouse_champion.json (`final_pct`), and
# reproduced end-to-end by H52.run. Never transcribed from a rounded console line: an earlier
# version of this file carried 93.10282628481847, which was the printed 93.102826 with digits
# invented after it. This test is what caught that, for the second time in one session.
MOUSE_H52_PCT = 93.10282596057698
MOUSE_CHAMPION_PCT = 93.08288021668459


@pytest.fixture(scope="module")
def mouse():
    return load_dataset("mouse")


# ── 1. Isolation ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["_EPOCHS", "_MAX_SWEEPS", "_ALT_CYCLES", "_ALT_SIFT_SWEEPS"])
def test_shared_dicts_match_the_champion(name):
    assert getattr(H52, name) == getattr(H44, name)


@pytest.mark.parametrize("name", ["_K_FULL", "_ALPHA", "_ALT_K_FULL", "_MIN_BLOCK"])
def test_shared_scalars_match_the_champion(name):
    assert getattr(H52, name) == getattr(H44, name)


def test_primaries_keep_the_h42_constants():
    for ds in ("connectome", "microns"):
        assert H52._EPOCHS[ds] == H42._EPOCHS[ds]
        assert H52._ALT_CYCLES[ds] == H42._ALT_CYCLES[ds]


def test_every_dataset_has_a_pop_budget():
    """A missing budget would silently mean 'drain', which is how microns overruns its guard."""
    for ds in ("connectome", "microns", "mouse"):
        assert isinstance(H52._PAIR_MAX_POPS.get(ds), int)
        assert H52._PAIR_MAX_POPS[ds] > 0


# ── 2. Stage 5 is monotone, and it fires ─────────────────────────────────────────────
def test_stage5_never_lowers_the_stage4_score(mouse):
    res = H52.run(mouse, seed=42, device=torch.device("cpu"))
    a = res.history.attrs
    assert a["pair_best_pct"] >= a["alt_best_pct"]
    assert a["pair_increment_pp"] >= 0.0


def test_mouse_reproduces_the_measured_value(mouse):
    res = H52.run(mouse, seed=42, device=torch.device("cpu"))
    got = 100.0 * res.best_score / mouse.total_weight
    assert got == pytest.approx(MOUSE_H52_PCT, abs=1e-9)
    assert got > MOUSE_CHAMPION_PCT


def test_stage5_actually_fires_on_mouse(mouse):
    """A stage that never fires would pass every other test here while doing nothing."""
    res = H52.run(mouse, seed=42, device=torch.device("cpu"))
    assert res.history.attrs["n_pair_moves"] >= 1


# ── 3. Determinism ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", [42, 123, 999])
def test_seed_does_not_change_the_result(mouse, seed):
    res = H52.run(mouse, seed=seed, device=torch.device("cpu"))
    got = 100.0 * res.best_score / mouse.total_weight
    assert got == pytest.approx(MOUSE_H52_PCT, abs=1e-9)


def test_repeat_run_is_bit_identical(mouse):
    a = H52.run(mouse, seed=42, device=torch.device("cpu"))
    b = H52.run(mouse, seed=42, device=torch.device("cpu"))
    assert a.best_score == b.best_score
    assert np.array_equal(a.best_positions, b.best_positions)


# ── 4. The oracle defines the number ─────────────────────────────────────────────────
def test_reported_score_matches_the_frozen_scorer(mouse):
    res = H52.run(mouse, seed=42, device=torch.device("cpu"))
    exact = score_from_positions(res.best_positions, np.asarray(mouse.src),
                                 np.asarray(mouse.tgt), mouse.weight)
    assert exact == res.best_score


def test_pair_log_is_recorded_for_audit(mouse):
    """P05/P06: 'was this stage truncated by its budget?' must be answerable from the record."""
    a = H52.run(mouse, seed=42, device=torch.device("cpu")).history.attrs
    for k in ("pair_increment_pp", "n_pair_passes", "pair_passes_requested",
              "n_pair_moves", "n_pair_pops", "pair_max_pops", "pair_time_s", "pair_log"):
        assert k in a
