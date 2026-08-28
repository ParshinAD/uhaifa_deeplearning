"""Pin variant H70's two re-sized constants and its structural-no-op property.

Added on the cycle-18 critic's condition 9: H63 and H64 both carry a test file pinning what
makes them what they are, and H70 did not. H70's whole content is TWO NUMBERS chosen off a
measured curve (``experiments/outputs/proto_P19_microns.json``) under a rule sealed before the
curve was complete (``experiments/outputs/proto_P19_sizing_rule.json``, commit 72fd700). A silent
edit to either number would invalidate every H70 result on disk while leaving the module
importable and the pipeline runnable, which is exactly the failure a pin exists to prevent.

The second thing pinned here is the claim the cycle's controls rest on: on **connectome and
mouse** H70 is supposed to be byte-identical to the H64 and H63 champions, so that the only
measured change is on microns. That is a property of the constants, and it is checked here
against those modules directly rather than restated as a comment.

These are unit tests over module constants. They run no pipeline and touch no GPU.
"""
from __future__ import annotations

import pytest

from mfas.experiments import H63, H64, H70


# ── the two constants that ARE the variant ──────────────────────────────────────────

def test_microns_epochs_is_the_sized_value():
    """50,000, read off proto_P19_microns.json arm E=50000. The champion ships 80,000."""
    assert H70._EPOCHS["microns"] == 50_000


def test_microns_alt_cycles_is_the_sized_value():
    """20, i.e. cycle index 19 on that arm. The champion ships 5."""
    assert H70._ALT_CYCLES["microns"] == 20


def test_sizing_provenance_matches_the_constants():
    """``_SIZING`` documents the operating point; it must not drift from the code it describes."""
    assert H70._SIZING["selected"] == {"epochs": 50_000, "alt_cycles": 20}
    assert H70._SIZING["selected"]["epochs"] == H70._EPOCHS["microns"]
    assert H70._SIZING["selected"]["alt_cycles"] == H70._ALT_CYCLES["microns"]
    assert H70._SIZING["champion"] == {"epochs": 80_000, "alt_cycles": 5}


# ── the structural-no-op property the connectome / mouse controls rest on ───────────

@pytest.mark.parametrize("dataset,champion", [("connectome", H64), ("mouse", H63)])
def test_non_microns_legs_match_their_champion(dataset, champion):
    """Every constant reaching stage 2-6 on this dataset equals its champion's.

    If this fails, the H70 run on ``dataset`` is no longer a control and the cycle-18 claim that
    its delta is 0 by construction is void.
    """
    assert H70._EPOCHS[dataset] == champion._EPOCHS[dataset]
    assert H70._ALT_CYCLES[dataset] == champion._ALT_CYCLES[dataset]
    assert H70._MAX_SWEEPS[dataset] == champion._MAX_SWEEPS[dataset]
    assert H70._ALT_SIFT_SWEEPS[dataset] == champion._ALT_SIFT_SWEEPS[dataset]
    assert H70._PAIR_MAX_POPS[dataset] == champion._PAIR_MAX_POPS[dataset]
    assert H70._RECLAIM_ROUNDS[dataset] == champion._RECLAIM_ROUNDS[dataset]


def test_shared_scalars_match_the_champions():
    """The scalar refinement constants are shared by all three legs and are H42's throughout."""
    assert (H70._K_FULL, H70._ALPHA, H70._ALT_K_FULL, H70._MIN_BLOCK) == \
           (H64._K_FULL, H64._ALPHA, H64._ALT_K_FULL, H64._MIN_BLOCK)


def test_surrogate_map_names_each_dataset_own_champion():
    """``_SURROGATE`` keys the ALGORITHM on the dataset name, which is only legitimate because
    sota.json is per-dataset (P14). Pinned so that fact stays deliberate: connectome's champion
    H64 uses the asymmetric surrogate, the microns champion H42 uses the plain sigmoid, and mouse
    runs 0 epochs so its entry is never exercised."""
    assert H70._SURROGATE == {"connectome": "asym", "microns": "sigmoid", "mouse": "sigmoid"}
    assert H70._EPOCHS["mouse"] == 0, "mouse must run no gradient steps, so its surrogate is moot"


# ── microns is the ONLY leg that differs from the shipped champion ──────────────────

def test_microns_is_the_only_changed_leg():
    """Guards against a future edit quietly re-sizing a primary that is meant to be a control."""
    changed = {ds for ds in ("connectome", "microns", "mouse")
               if (H70._EPOCHS[ds], H70._ALT_CYCLES[ds]) != (H70._SIZING["champion"]["epochs"],
                                                             H70._SIZING["champion"]["alt_cycles"])
               and ds == "microns"}
    assert changed == {"microns"}
    # and the change is in the intended direction: less gradient, more refinement
    assert H70._EPOCHS["microns"] < H70._SIZING["champion"]["epochs"]
    assert H70._ALT_CYCLES["microns"] > H70._SIZING["champion"]["alt_cycles"]
