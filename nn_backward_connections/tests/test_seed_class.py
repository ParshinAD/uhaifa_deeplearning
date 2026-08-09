"""Unit tests for the seed-dependence classifier (autoresearch/seed_class.py).

The classifier decides whether a variant may be screened at 1 seed instead of 3 (queue item
P02). A wrong ``deterministic`` verdict silently throws away real seed variance and corrupts
a screen verdict, so the classifier's behaviour is pinned here as a contract rather than left
as a one-off script whose output nobody re-checks.

Covers:
  (1) the ground-truth classification of all 16 current variants, cross-checked against the
      48-run mouse probe in experiments/outputs/proto_P02.json;
  (2) the two structural cases that broke naive implementations:
        - ``run_rocket`` is RNG-free when reached WITH ``init_positions`` and RNG-consuming
          without it (dead-branch pruning), which is what separates H30/H35 from the
          baseline;
        - ``mfas/refine/insertion.py`` holds both the RNG-free ``sift`` and the seeded
          ``sift_gauss_seidel_ref``, so file-level scanning misclassifies every champion;
  (3) the fail-safe direction: an unresolvable call forces ``rng``, never ``deterministic``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "autoresearch"))

import seed_class  # noqa: E402


# Ground truth. "det" = does not draw from `seed` on any path reachable from run().
EXPECTED = {
    "baseline_passthrough": "rng",   # run_rocket without init_positions -> torch.randn
    "baseline_multistart": "rng",
    "H01": "rng",                    # K restarts from different random inits
    "H02": "deterministic",          # greedy-FAS warm start, nothing draws
    "H03": "rng",
    "H04": "rng",
    "H05": "rng",
    "H06": "rng",
    "H09": "rng",
    "H11": "rng",
    "H13": "rng",                    # RandomState(seed + 104729) every step
    "H16": "deterministic",
    "H19": "deterministic",          # _init_positions() takes no seed (hand table got this wrong)
    "H30": "deterministic",
    "H31": "rng",                    # RandomState(seed + 7919) in the LNS destroy operator
    "H35": "deterministic",
}


@pytest.fixture(scope="module")
def idx():
    return seed_class.ModuleIndex()


@pytest.mark.parametrize("variant,expected", sorted(EXPECTED.items()))
def test_classification_matches_ground_truth(idx, variant, expected):
    got = seed_class.analyze(idx, variant)
    assert got["verdict"] == expected, (
        f"{variant}: expected {expected}, got {got['verdict']}. "
        f"rng_evidence={got['rng_evidence']} unresolved={got['unresolved_calls'][:3]}")


@pytest.mark.parametrize("variant", sorted(v for v, c in EXPECTED.items()
                                           if c == "deterministic"))
def test_deterministic_variants_need_one_seed(idx, variant):
    assert seed_class.analyze(idx, variant)["screen_seeds_required"] == 1


@pytest.mark.parametrize("variant", sorted(v for v, c in EXPECTED.items() if c == "rng"))
def test_rng_variants_keep_three_seeds(idx, variant):
    assert seed_class.analyze(idx, variant)["screen_seeds_required"] == 3


def test_champions_do_not_inherit_unused_rng_kernel(idx):
    """H30/H35 import from mfas.refine.insertion, which ALSO holds a seeded reference sift.

    A file-level scan flags that module and misclassifies both champions. The classifier must
    follow reachable functions, so the unused kernel must not appear in the evidence.
    """
    for variant in ("H30", "H35"):
        got = seed_class.analyze(idx, variant)
        assert got["verdict"] == "deterministic"
        assert not any("gauss_seidel" in e for e in got["rng_evidence"])


def test_run_rocket_branch_pruning_is_load_bearing(idx):
    """With init_positions bound, run_rocket cannot reach make_init_positions / torch.randn.

    Without it, it must. Both directions are asserted so a regression in the pruning rule
    cannot pass by making everything deterministic.
    """
    warm = seed_class.analyze(idx, "H35")          # passes init_positions=
    cold = seed_class.analyze(idx, "baseline_passthrough")   # does not
    assert warm["verdict"] == "deterministic"
    assert not any("make_init_positions" in e for e in warm["rng_evidence"])
    assert cold["verdict"] == "rng"
    assert any("make_init_positions" in e or "randn" in e for e in cold["rng_evidence"])


def test_unresolved_calls_fail_safe_to_rng(idx):
    """Strict mode must never answer 'deterministic' while a call path is unknown."""
    got = seed_class.analyze(idx, "H35", strict=True)
    if got["unresolved_calls"]:
        assert got["verdict"] == "rng", (
            "strict mode returned deterministic despite unresolved calls: "
            f"{got['unresolved_calls'][:3]}")


def test_missing_variant_reports_error(idx):
    got = seed_class.analyze(idx, "H_does_not_exist")
    assert got["verdict"] == "error"


def test_agrees_with_the_mouse_probe_where_the_probe_is_valid():
    """Cross-check against the 48-run empirical probe (experiments/outputs/proto_P02.json).

    Only one direction is a real error: the classifier calling a variant deterministic when
    the seed demonstrably changes its output. The reverse (static 'rng', mouse looked inert)
    is expected — H31 draws from its seed, but on mouse its LNS stage never beats the sift, so
    best-by-oracle returns the deterministic sift vector and the seed leaves no trace.
    """
    proto_path = _ROOT / "experiments" / "outputs" / "proto_P02.json"
    if not proto_path.exists():
        pytest.skip("proto_P02.json not generated (experiments/proto_p02_determinism.py)")
    proto = json.loads(proto_path.read_text())

    false_det = []
    for vid, obs in proto["variants"].items():
        if "observed_class" not in obs or vid not in EXPECTED:
            continue
        if EXPECTED[vid] == "deterministic" and not obs["seed_inert_positions"]:
            false_det.append(vid)
    assert not false_det, f"classified deterministic but seed-sensitive on mouse: {false_det}"
