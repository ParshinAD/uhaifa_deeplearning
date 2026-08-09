"""Unit tests for the seed planner (autoresearch/seed_plan.py).

`seed_class.py` decides whether a variant draws from its seed; `seed_plan.py` turns that into the
seed list each dataset actually runs, and is therefore what CONTROLS COMPUTE. Its failure modes
are asymmetric and both are pinned here:

  * dropping seeds where they carry information (a false `deterministic`, or leaking the 1-seed
    rule into `confirm`) silently corrupts a verdict — the expensive kind of wrong;
  * keeping seeds that are inert only costs wall-clock — the acceptable kind.

So every degraded path (unknown variant, classifier error, policy disabled, unknown dataset key)
must fall back to MORE seeds, never fewer. `tests/test_seed_class.py` covers the classifier
itself; this file covers the policy applied on top of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "autoresearch"))

import seed_plan  # noqa: E402

pytest.importorskip("yaml", reason="seed_plan reads campaign.yaml")

PRIMARIES = ("connectome", "microns")


@pytest.fixture(scope="module")
def campaign():
    import yaml
    return yaml.safe_load((_ROOT / "autoresearch" / "campaign.yaml").read_text())


def test_deterministic_variant_screens_at_the_declared_seed_counts(campaign):
    """The plan must match the policy exactly — which datasets are reduced is campaign.yaml's
    call, not this test's. A dataset may only be declared reduced once its device determinism is
    verified; that coupling is enforced by
    test_campaign_yaml_declares_device_determinism_for_every_one_seed_dataset below.
    """
    policy = campaign["seed_policy"]["screen_seeds_by_class"]["deterministic"]
    p = seed_plan.plan("H35", "implement")
    assert p["classification"] == "deterministic"
    for ds in ("connectome", "microns", "mouse"):
        assert p["seeds"][ds] == list(policy[ds]), f"{ds}: got {p['seeds'][ds]}"


def test_the_amendment_is_not_inert(campaign):
    """At least one expensive primary must actually be reduced, or P02 bought nothing and the
    machinery is dead weight pretending to be a policy."""
    policy = campaign["seed_policy"]["screen_seeds_by_class"]["deterministic"]
    reduced = [ds for ds in PRIMARIES
               if len(policy[ds]) < len(campaign["datasets"][ds]["screen_seeds"])]
    assert reduced, "no primary screens at a reduced seed count; the 1-seed screen is inert"


def test_mouse_never_drops_below_three_seeds_on_the_screen():
    """Mouse is the standing tripwire: ~6 s/run, and three disagreeing runs falsify a
    `deterministic` classification before any expensive number is quoted."""
    for variant in ("H35", "H30", "H02", "H31", "baseline_passthrough"):
        p = seed_plan.plan(variant, "implement")
        assert len(p["seeds"]["mouse"]) >= 3, f"{variant} dropped mouse to {p['seeds']['mouse']}"


def test_deterministic_screen_carries_the_tripwire():
    assert seed_plan.plan("H35", "implement")["tripwire"]
    assert seed_plan.plan("H31", "implement")["tripwire"] is None   # rng: nothing to corroborate


def test_rng_variant_keeps_three_seeds_everywhere():
    p = seed_plan.plan("H31", "implement")          # RandomState(seed + 7919) in the LNS destroy
    assert p["classification"] == "rng"
    for ds in ("connectome", "microns", "mouse"):
        assert len(p["seeds"][ds]) == 3, f"{ds}: {p['seeds'][ds]}"


def test_confirm_role_is_never_touched_by_the_policy(campaign):
    """The 1-seed rule is screen-only. Every CI in a promotion path must come from a confirm pool:
    audit.protocol_se() is comparator-only (sqrt(2/n_c)), so on a 1-seed pool it would understate
    the SE by ~41% instead of widening it."""
    for variant in ("H35", "H30", "H31"):
        p = seed_plan.plan(variant, "confirm")
        for ds in ("connectome", "microns", "mouse"):
            assert p["seeds"][ds] == list(campaign["datasets"][ds]["confirm_seeds"])


def test_non_screen_roles_fall_back_to_screen_seeds(campaign):
    for role in ("verify", "prototype"):
        p = seed_plan.plan("H35", role)
        for ds in ("connectome", "microns", "mouse"):
            assert p["seeds"][ds] == list(campaign["datasets"][ds]["screen_seeds"])


def test_unknown_variant_fails_safe_to_three_seeds():
    p = seed_plan.plan("H_does_not_exist", "implement")
    assert p["classification"] == "rng"
    for ds in ("connectome", "microns", "mouse"):
        assert len(p["seeds"][ds]) == 3


def test_classifier_exception_fails_safe_to_rng(monkeypatch):
    monkeypatch.setattr(seed_plan.seed_class, "ModuleIndex",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert seed_plan.classify("H35")["verdict"] == "rng"


def test_disabled_policy_falls_back_to_screen_seeds(monkeypatch, campaign):
    import copy
    disabled = copy.deepcopy(campaign)
    disabled["seed_policy"]["enabled"] = False
    monkeypatch.setattr(seed_plan, "_campaign", lambda: disabled)
    p = seed_plan.plan("H35", "implement")
    for ds in ("connectome", "microns", "mouse"):
        assert p["seeds"][ds] == list(campaign["datasets"][ds]["screen_seeds"])


def test_missing_policy_entry_falls_back_to_screen_seeds(monkeypatch, campaign):
    import copy
    partial = copy.deepcopy(campaign)
    del partial["seed_policy"]["screen_seeds_by_class"]["deterministic"]["microns"]
    monkeypatch.setattr(seed_plan, "_campaign", lambda: partial)
    p = seed_plan.plan("H35", "implement")
    assert p["seeds"]["microns"] == list(campaign["datasets"]["microns"]["screen_seeds"])
    assert p["seeds"]["connectome"] == [42]          # the entry that IS present still applies


def test_plan_never_exceeds_the_flat_three_seed_cost():
    """The policy may only ever remove runs from the screen, never add them."""
    for variant in ("H35", "H30", "H02", "H31", "H01", "baseline_passthrough"):
        p = seed_plan.plan(variant, "implement")
        assert p["n_runs"] <= p["n_runs_at_3_seeds"], variant


def test_campaign_yaml_declares_device_determinism_for_every_one_seed_dataset(campaign):
    """A 1-seed screen is only licensed where the device is KNOWN to reproduce a repeated seed.
    If a port has not re-established that (PORTING.md § 6b), the flag must not say true."""
    policy = campaign["seed_policy"]
    verified = policy["device_determinism_verified"]
    for ds, seeds in policy["screen_seeds_by_class"]["deterministic"].items():
        if len(seeds) < len(campaign["datasets"][ds]["screen_seeds"]):
            assert verified.get(ds) is True, (
                f"{ds} screens at {len(seeds)} seed(s) but device determinism is not verified")
