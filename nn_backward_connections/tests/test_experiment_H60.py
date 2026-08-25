"""Shippability tests for variant H60 (net-digraph condensation inside stage 4).

H60's whole claim is "H42 with ONE argument changed". Four things therefore have to hold
before a screen can adjudicate it, and each is checked here rather than trusted:

1. **Isolation** — every constant H60 shares with H42 is equal, so ``H60 - H42`` isolates
   the STRUCTURE GRAPH and nothing else.
2. **The champions are untouched** — ``alternate_scc_sift`` grew a ``g_struct`` keyword to
   make H60 possible, and that function backs the sitting H36/H42/H52 champions on all
   three datasets. Its default path must be bit-identical to the one that produced
   ``sota.json``, so the default is asserted to equal an explicit ``g_struct=g``.
3. **It runs end-to-end and is deterministic** — the campaign's 1-seed screen policy and
   ``sota.json``'s std = 0 both rest on the pipeline never drawing from ``seed``.
4. **It does not lose to H42 on the same input order** — checked empirically on mouse.

Everything here runs on mouse (148 nodes) on the CPU, in seconds. Connectome and MICrONS
are NEVER touched by a unit test.

On the strength of claim 4
--------------------------
The block refiner is monotone in the RAW score for BOTH structure graphs (proved in
:mod:`mfas.refine.net_condense`, tested in ``tests/test_net_condense.py``), and stage 4 is
best-by-oracle, so H60 can never score below its OWN stage-3 order — that is a theorem and
is asserted as one. H60 >= H42 is NOT a theorem: the two arms follow different trajectories
through stage 4, and a later cycle could in principle do less well from the different order
it now starts at. The mouse assertion below is therefore an EMPIRICAL non-inferiority
tripwire on one small graph, and is documented as such so nobody reads it as a proof.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mfas.experiments import H42, H60
from mfas.io import load_dataset
from mfas.metrics import score_from_positions


@pytest.fixture(scope="module")
def mouse():
    return load_dataset("mouse")


@pytest.fixture(scope="module")
def h60_mouse(mouse):
    """One H60 run on mouse, CPU, seed 42 — shared by several tests (it is the slow bit)."""
    return H60.run(mouse, seed=42, device=torch.device("cpu"), time_limit=None)


@pytest.fixture(scope="module")
def h42_mouse(mouse):
    return H42.run(mouse, seed=42, device=torch.device("cpu"), time_limit=None)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Isolation: H60 differs from H42 in the structure graph and NOTHING else
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("const", [
    "_EPOCHS", "_MAX_SWEEPS", "_K_FULL", "_ALPHA",
    "_ALT_CYCLES", "_ALT_SIFT_SWEEPS", "_ALT_K_FULL", "_MIN_BLOCK",
])
def test_shared_constants_are_identical_to_H42(const):
    assert getattr(H60, const) == getattr(H42, const), (
        f"H60.{const} drifted from H42.{const}; H60 - H42 no longer isolates the "
        f"structure graph")


def test_H60_uses_the_net_structure_graph_and_H42_does_not():
    from mfas.refine.net_condense import net_structure_graph
    assert H60.net_structure_graph is net_structure_graph
    assert not hasattr(H42, "net_structure_graph"), \
        "H42 should not import the net constructor; the two variants would stop differing"


def test_variant_metadata_present():
    assert H60.ID == "H60"
    assert isinstance(H60.HYPOTHESIS, str) and len(H60.HYPOTHESIS) > 80


# ──────────────────────────────────────────────────────────────────────────────
# 2. The new keyword does not disturb the sitting champions
# ──────────────────────────────────────────────────────────────────────────────
def test_default_g_struct_is_the_graph_itself(mouse):
    """``g_struct=None`` must reproduce the pre-H60 driver exactly — H36/H42/H52 depend on it."""
    from mfas.refine import alternate_scc_sift
    rng = np.random.default_rng(11)
    rank0 = rng.permutation(mouse.n_nodes).astype(np.int64)
    a_rank, a_score, a_log = alternate_scc_sift(mouse, rank0, n_cycles=6, sift_sweeps=2,
                                                k_full=2, min_block=8)
    b_rank, b_score, b_log = alternate_scc_sift(mouse, rank0, n_cycles=6, sift_sweeps=2,
                                                k_full=2, min_block=8, g_struct=mouse)
    assert np.array_equal(a_rank, b_rank)
    assert a_score == b_score
    assert [r["after_scc_pct"] for r in a_log] == [r["after_scc_pct"] for r in b_log]


def test_H42_is_unchanged_by_the_new_keyword(mouse, h42_mouse):
    """H42 on mouse must still be the sitting-champion pipeline it was before H60 landed."""
    again = H42.run(mouse, seed=42, device=torch.device("cpu"), time_limit=None)
    assert np.array_equal(again.best_positions, h42_mouse.best_positions)
    assert "struct_graph" not in h42_mouse.history.attrs, \
        "H42 picked up H60's provenance attr; the variants are no longer separable"


# ──────────────────────────────────────────────────────────────────────────────
# 3. End-to-end on mouse: valid output, correct score, full provenance
# ──────────────────────────────────────────────────────────────────────────────
def test_runs_end_to_end_on_mouse(mouse, h60_mouse):
    res = h60_mouse
    assert res.best_positions.shape == (mouse.n_nodes,)
    # the runner re-scores with the frozen oracle and asserts equality; do the same here
    rescored = score_from_positions(res.best_positions, np.asarray(mouse.src),
                                    np.asarray(mouse.tgt), mouse.weight)
    assert rescored == res.best_score, "variant score disagrees with the frozen oracle"
    assert 0.0 < res.best_pct <= 100.0
    assert res.n_epochs_done > 0


def test_stage_provenance_is_reported(h60_mouse):
    a = h60_mouse.history.attrs
    # everything H42 reports, so nothing downstream regresses...
    for k in ("pure_best_score", "pure_best_pct", "sift_best_score", "sift_best_pct",
              "n_sift_sweeps", "sift_sweeps_requested", "sift_converged", "sift_time_s",
              "alt_best_score", "alt_best_pct", "alt_increment_pp", "n_alt_cycles",
              "alt_cycles_requested", "alt_time_s", "alt_min_block", "alt_sift_sweeps",
              "refined_best_score", "refined_best_pct"):
        assert k in a, f"missing H42-parity attr {k!r}"
    # ...plus the structure-graph provenance
    assert a["struct_graph"] == "net"
    for k in ("net_build_s", "net_n_arcs", "net_const_c"):
        assert k in a, f"missing H60 attr {k!r}"
    assert a["net_n_arcs"] > 0 and a["net_const_c"] > 0


def test_runtime_guard_can_read_the_truncation_signal(h60_mouse):
    """``eval/runtime_guard.summarise`` needs these exact keys to flag a degraded run."""
    from eval import runtime_guard
    a = dict(h60_mouse.history.attrs)
    plan = runtime_guard.resolve("mouse", explicit=3450.0)
    summary = runtime_guard.summarise(plan, wall_clock_s=1.0,
                                      variant_wall_s=h60_mouse.wall_clock_s, attrs=a)
    assert summary["armed"] is True
    # an untruncated mouse run must not look degraded
    assert a["n_alt_cycles"] == a["alt_cycles_requested"]
    assert summary["stages_truncated"] == {}
    assert summary["degraded"] is False


def test_time_limit_is_honoured(mouse):
    """A tiny deadline must truncate cleanly at a stage boundary, not run to completion."""
    res = H60.run(mouse, seed=42, device=torch.device("cpu"), time_limit=0.0)
    a = res.history.attrs
    assert a["n_alt_cycles"] < a["alt_cycles_requested"], \
        "stage 4 ignored time_limit=0 — the run-level wall-clock guard would be broken"
    # even a fully truncated run returns a valid, non-regressing order
    assert res.best_score >= a["pure_best_score"]


# ──────────────────────────────────────────────────────────────────────────────
# 4. Determinism and non-inferiority
# ──────────────────────────────────────────────────────────────────────────────
def test_two_runs_are_bit_identical(mouse, h60_mouse):
    again = H60.run(mouse, seed=42, device=torch.device("cpu"), time_limit=None)
    assert np.array_equal(again.best_positions, h60_mouse.best_positions), \
        "H60 is not deterministic"
    assert again.best_score == h60_mouse.best_score


def test_seed_is_never_drawn_from(mouse, h60_mouse):
    """Different seeds must give a bit-identical result (no RNG anywhere in the pipeline)."""
    other = H60.run(mouse, seed=999, device=torch.device("cpu"), time_limit=None)
    assert np.array_equal(other.best_positions, h60_mouse.best_positions), \
        "H60 draws from its seed; the 1-seed screen policy would not apply"
    assert other.best_score == h60_mouse.best_score


def test_never_regresses_below_its_own_stage3(h60_mouse):
    """Guaranteed by construction (monotone refiner + best-by-oracle), so assert it as such."""
    a = h60_mouse.history.attrs
    assert h60_mouse.best_score >= a["sift_best_score"]
    assert a["alt_best_score"] >= a["sift_best_score"]
    assert a["alt_increment_pp"] >= 0.0, "stage 4 lowered the score"


def test_not_worse_than_H42_on_mouse(h60_mouse, h42_mouse):
    """EMPIRICAL non-inferiority tripwire on one small graph — not a theorem (see module doc).

    Both variants start from the same greedy-FAS warm start and the same deterministic
    Rocket run at seed 42 on the CPU, so this is a like-for-like comparison from an
    identical input order.
    """
    assert h60_mouse.history.attrs["sift_best_score"] == \
        h42_mouse.history.attrs["sift_best_score"], \
        "stages 1-3 diverged; the two variants are no longer comparable"
    assert h60_mouse.best_score >= h42_mouse.best_score, (
        f"H60 {h60_mouse.best_pct:.6f}% < H42 {h42_mouse.best_pct:.6f}% on mouse")
