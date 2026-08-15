"""Shippability tests for variant H41 (segment moves inside stage 4).

H41's whole claim is "H42 with ONE thing changed". Three things therefore have to hold
before a screen can adjudicate it, and each is checked here rather than trusted:

1. **Isolation** — every constant H41 shares with H42 is equal, so ``H41 - H42`` isolates
   the move class. A variant that changes two things at once cannot be adjudicated, and a
   constant drifting apart later is exactly the kind of silent breakage that would not
   otherwise surface until a screen result was already being argued about.
2. **It runs end-to-end and is deterministic** — the campaign's 1-seed screen policy and
   ``sota.json``'s std = 0 both rest on the pipeline never drawing from ``seed``.
3. **It does not lose to H42 on the same input order** — checked empirically on mouse.

Everything here runs on mouse (148 nodes) on the CPU, in seconds. Connectome and MICrONS
are NEVER touched by a unit test.

On the strength of claim 3
--------------------------
``segment_refine`` is monotone by construction, and stage 4 is best-by-oracle, so H41 can
never score below its OWN stage-3 order — that is a theorem and is asserted as one. H41 >=
H42 is NOT a theorem: the segment moves change the trajectory, and a later block/sift stage
could in principle do less well from the different order it now starts at. The mouse
assertion below is therefore an EMPIRICAL non-inferiority tripwire on one small graph, and
is documented as such so nobody reads it as a proof.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mfas.experiments import H41, H42
from mfas.io import load_dataset
from mfas.metrics import score_from_positions


@pytest.fixture(scope="module")
def mouse():
    return load_dataset("mouse")


@pytest.fixture(scope="module")
def h41_mouse(mouse):
    """One H41 run on mouse, CPU, seed 42 — shared by several tests (it is the slow bit)."""
    return H41.run(mouse, seed=42, device=torch.device("cpu"), time_limit=None)


@pytest.fixture(scope="module")
def h42_mouse(mouse):
    return H42.run(mouse, seed=42, device=torch.device("cpu"), time_limit=None)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Isolation: H41 differs from H42 in the stage-4 driver and NOTHING else
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("const", [
    "_EPOCHS", "_MAX_SWEEPS", "_K_FULL", "_ALPHA",
    "_ALT_CYCLES", "_ALT_SIFT_SWEEPS", "_ALT_K_FULL", "_MIN_BLOCK",
])
def test_shared_constants_are_identical_to_H42(const):
    assert getattr(H41, const) == getattr(H42, const), (
        f"H41.{const} drifted from H42.{const}; H41 - H42 no longer isolates the move class")


def test_H41_uses_the_segment_alternation_and_H42_does_not():
    from mfas.refine.scc_recursive import alternate_scc_sift
    from mfas.refine.segment import alternate_scc_sift_segment
    assert H41.alternate_scc_sift_segment is alternate_scc_sift_segment
    assert H42.alternate_scc_sift is alternate_scc_sift
    assert not hasattr(H41, "alternate_scc_sift"), \
        "H41 should not also import the champion's driver"


def test_variant_metadata_present():
    assert H41.ID == "H41"
    assert isinstance(H41.HYPOTHESIS, str) and len(H41.HYPOTHESIS) > 80


def test_segment_entry_points_are_exported_from_refine():
    """The variant imports the ladders through ``mfas.refine``; keep that surface stable."""
    import mfas.refine as refine
    for name in ("alternate_scc_sift_segment", "segment_refine", "segment_sweep",
                 "DEFAULT_SEG_LENGTHS", "DEFAULT_OFFSETS"):
        assert hasattr(refine, name), f"mfas.refine does not export {name}"
        assert name in refine.__all__, f"{name} missing from mfas.refine.__all__"


# ──────────────────────────────────────────────────────────────────────────────
# 2. End-to-end on mouse: valid output, correct score, full provenance
# ──────────────────────────────────────────────────────────────────────────────
def test_runs_end_to_end_on_mouse(mouse, h41_mouse):
    res = h41_mouse
    assert res.best_positions.shape == (mouse.n_nodes,)
    # the runner re-scores with the frozen oracle and asserts equality; do the same here
    rescored = score_from_positions(res.best_positions, np.asarray(mouse.src),
                                    np.asarray(mouse.tgt), mouse.weight)
    assert rescored == res.best_score, "variant score disagrees with the frozen oracle"
    assert 0.0 < res.best_pct <= 100.0
    assert res.n_epochs_done > 0


def test_stage_provenance_is_reported(h41_mouse):
    a = h41_mouse.history.attrs
    # everything H42 reports, so nothing downstream regresses...
    for k in ("pure_best_score", "pure_best_pct", "sift_best_score", "sift_best_pct",
              "n_sift_sweeps", "sift_sweeps_requested", "sift_converged", "sift_time_s",
              "alt_best_score", "alt_best_pct", "alt_increment_pp", "n_alt_cycles",
              "alt_cycles_requested", "alt_time_s", "alt_min_block", "alt_sift_sweeps",
              "refined_best_score", "refined_best_pct"):
        assert k in a, f"missing H42-parity attr {k!r}"
    # ...plus the per-move-class split that makes the segment credit separable
    for k in ("scc_increment_pp", "alt_sift_increment_pp", "seg_increment_pp",
              "n_segment_moves", "alt_seg_time_s", "alt_seg_sweeps", "alt_seg_lengths"):
        assert k in a, f"missing H41 attr {k!r}"
    # the three per-class increments partition the stage-4 increment exactly
    parts = a["scc_increment_pp"] + a["alt_sift_increment_pp"] + a["seg_increment_pp"]
    assert parts == pytest.approx(a["alt_increment_pp"], rel=0, abs=1e-9), \
        "per-move-class increments do not sum to the stage-4 increment"


def test_runtime_guard_can_read_the_truncation_signal(h41_mouse):
    """``eval/runtime_guard.summarise`` needs these exact keys to flag a degraded run."""
    from eval import runtime_guard
    a = dict(h41_mouse.history.attrs)
    plan = runtime_guard.resolve("mouse", explicit=3450.0)
    summary = runtime_guard.summarise(plan, wall_clock_s=1.0,
                                      variant_wall_s=h41_mouse.wall_clock_s, attrs=a)
    assert summary["armed"] is True
    # an untruncated mouse run must not look degraded
    assert a["n_alt_cycles"] == a["alt_cycles_requested"]
    assert summary["stages_truncated"] == {}
    assert summary["degraded"] is False


def test_time_limit_is_honoured(mouse):
    """A tiny deadline must truncate cleanly at a stage boundary, not run to completion."""
    res = H41.run(mouse, seed=42, device=torch.device("cpu"), time_limit=0.0)
    a = res.history.attrs
    assert a["n_alt_cycles"] < a["alt_cycles_requested"], \
        "stage 4 ignored time_limit=0 — the run-level wall-clock guard would be broken"
    # even a fully truncated run returns a valid, non-regressing order
    assert res.best_score >= a["pure_best_score"]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Determinism and non-inferiority
# ──────────────────────────────────────────────────────────────────────────────
def test_two_runs_are_bit_identical(mouse, h41_mouse):
    again = H41.run(mouse, seed=42, device=torch.device("cpu"), time_limit=None)
    assert np.array_equal(again.best_positions, h41_mouse.best_positions), \
        "H41 is not deterministic"
    assert again.best_score == h41_mouse.best_score


def test_seed_is_never_drawn_from(mouse, h41_mouse):
    """Different seeds must give a bit-identical result (no RNG anywhere in the pipeline)."""
    other = H41.run(mouse, seed=999, device=torch.device("cpu"), time_limit=None)
    assert np.array_equal(other.best_positions, h41_mouse.best_positions), \
        "H41 draws from its seed; the 1-seed screen policy would not apply"
    assert other.best_score == h41_mouse.best_score


def test_never_regresses_below_its_own_stage3(h41_mouse):
    """Guaranteed by construction (monotone sweeps + best-by-oracle), so assert it as such."""
    a = h41_mouse.history.attrs
    assert h41_mouse.best_score >= a["sift_best_score"]
    assert a["alt_best_score"] >= a["sift_best_score"]
    assert a["seg_increment_pp"] >= 0.0, "a segment sweep lowered the score"


def test_not_worse_than_H42_on_mouse(h41_mouse, h42_mouse):
    """EMPIRICAL non-inferiority tripwire on one small graph — not a theorem (see module doc).

    Both variants start from the same greedy-FAS warm start and the same deterministic
    Rocket run at seed 42 on the CPU, so this is a like-for-like comparison from an
    identical input order.
    """
    assert h41_mouse.history.attrs["sift_best_score"] == \
        h42_mouse.history.attrs["sift_best_score"], \
        "stages 1-3 diverged; the two variants are no longer comparable"
    assert h41_mouse.best_score >= h42_mouse.best_score, (
        f"H41 {h41_mouse.best_pct:.6f}% < H42 {h42_mouse.best_pct:.6f}% on mouse")
