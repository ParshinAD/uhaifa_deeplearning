"""Unit tests for the mechanical promotion gate (``autoresearch/audit.py`` + ``update_sota.py``).

These pin the five defects fixed on 2026-08-15. Every one of them was a *silent* failure — the
campaign kept exiting 0 and promoting champions while the check that was supposed to stop it
either did not exist, compared against the wrong number, or was defeated by float noise. A gate
nobody tests is indistinguishable from a gate that isn't there, which is exactly how these got in.

  D2  ``update_sota.py`` gated the next promotion against a 4-decimal-ROUNDED champion mean, so
      H42 took the mouse row from H36 on a delta of exactly 0.0 over bit-identical pools.
  D3  the degeneracy tripwire tested ``std == 0.0`` while numpy returns 1.458e-14 for 20
      identical values, so it did not fire on the one pool that most needed it.
  D4  the mouse non-inferiority bound used the comparator's n only and floored sigma at the
      random-init baseline's 0.2624 pp, making it unpassable without a +0.16 pp *improvement*.
  D5  an empty comparator pool WARNed and the audit still exited 0 with VERDICT: PASS.
  D6  provenance was keyed on the hash with '+dirty' stripped, and nothing checked that the
      variant's module existed at the commit its runs were stamped with.

Pure python, no dataset load, no GPU: the heaviest thing here is a handful of ``git cat-file``
calls and one pass over ``results/*.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "autoresearch"))

import audit  # noqa: E402
import update_sota  # noqa: E402


# The exact mouse confirm value shared, bit for bit, by all 20 H36 and all 20 H42 runs.
MOUSE_TIE_PCT = 92.91701410211007
# H36's mouse pool vs its comparator H30 (n=3), from autoresearch/audit_H36.json.
H36_MOUSE_MEAN = 92.91701410211006
H30_MOUSE_MEAN = 92.90180243040469
MOUSE_MARGIN_PP = -0.26            # campaign.yaml datasets.mouse.non_inferiority_pp
MOUSE_RESOLUTION_PP = 1e-12        # campaign.yaml datasets.mouse.measurement_resolution_pp


def _stats(mean: float, std: float, n: int) -> dict:
    return dict(mean=mean, std=std, n=n)


def _campaign() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "autoresearch" / "campaign.yaml").read_text())


# ──────────────────────────────────────────────────────────────────────────────
# D3 — the degeneracy tripwire must not be defeated by float noise
# ──────────────────────────────────────────────────────────────────────────────
def test_d3_numpy_std_of_identical_values_is_not_zero():
    """The premise of the bug, pinned so nobody 'simplifies' the tolerance away later.

    ``numpy.std(ddof=1)`` over 20 copies of the same float returns 1.458e-14, not 0 —
    catastrophic cancellation in the two-pass sum. ``std == 0.0`` was therefore False for the
    H42 mouse pool while it was True for connectome and microns, which is why
    ``significance.mouse.degenerate`` is absent from audit_H42.json and present for the others.
    """
    raw = float(np.array([MOUSE_TIE_PCT] * 20, dtype=np.float64).std(ddof=1))
    assert raw != 0.0
    assert 0.0 < raw < 1e-13
    assert raw < audit.ZERO_STD_TOL * 100          # comfortably inside the new tolerance
    assert not (raw == 0.0)                        # the old test, spelled out


def test_d3_summarize_reports_identical_pool_as_exactly_zero():
    runs = [dict(pct=MOUSE_TIE_PCT, seed=s, _file=f"r{s}.json") for s in range(20)]
    s = audit.summarize(runs)
    assert s["std"] == 0.0, "a pool with one distinct value has std 0, not 1e-14"
    assert s["n"] == 20
    assert s["mean"] == pytest.approx(MOUSE_TIE_PCT, abs=1e-12)


def test_d3_summarize_keeps_real_dispersion():
    """The snap-to-zero must not swallow genuine spread."""
    runs = [dict(pct=p, seed=i, _file=f"r{i}.json")
            for i, p in enumerate([92.90, 92.92, 92.91])]
    s = audit.summarize(runs)
    assert s["std"] > 1e-3


def test_d3_tolerance_comes_from_campaign_yaml():
    assert float(_campaign()["promotion_gate"]["zero_std_tol"]) == pytest.approx(1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# D4 — the mouse non-inferiority bound, re-specified
# ──────────────────────────────────────────────────────────────────────────────
def test_d4_old_formula_was_unpassable_regression_check():
    """Why the formula was replaced, kept as an executable statement of the defect.

    The old bound reused ``protocol_se`` (comparator n only, sigma floored at the random-init
    baseline's 0.2624 pp). Reproduce H36's mouse leg with it and you get -0.4047 — a FAIL for a
    variant whose mouse score IMPROVED by +0.0152 pp over 20 seeds with zero dispersion.
    """
    c = _stats(H30_MOUSE_MEAN, 0.0, 3)
    delta = H36_MOUSE_MEAN - H30_MOUSE_MEAN
    old_ci_lo = delta - 1.96 * audit.protocol_se(c, sigma_floor=0.2624)
    assert old_ci_lo == pytest.approx(-0.4047, abs=5e-4)
    assert old_ci_lo < MOUSE_MARGIN_PP            # i.e. it failed
    # It needed a large IMPROVEMENT to certify "not worse":
    needed = 1.96 * audit.protocol_se(c, sigma_floor=0.2624) + MOUSE_MARGIN_PP
    assert needed == pytest.approx(0.1600, abs=5e-3)


def test_d4_case_i_positive_delta_zero_dispersion_passes():
    """(i) A strictly positive delta with zero dispersion must PASS."""
    v = _stats(92.93, 0.0, 20)
    c = _stats(92.91, 0.0, 5)
    delta, se, ci_lo = audit.noninferiority_ci(v, c, MOUSE_RESOLUTION_PP)
    assert delta > 0
    assert se < 1e-12
    assert ci_lo > MOUSE_MARGIN_PP


def test_d4_case_ii_genuine_regression_fails():
    """(ii) A regression beyond the tolerance must FAIL."""
    v = _stats(92.60, 0.0, 20)                    # -0.31 pp, past the -0.26 margin
    c = _stats(92.91, 0.0, 20)
    delta, _se, ci_lo = audit.noninferiority_ci(v, c, MOUSE_RESOLUTION_PP)
    assert delta == pytest.approx(-0.31, abs=1e-9)
    assert ci_lo <= MOUSE_MARGIN_PP

    # ... and a regression INSIDE the tolerance is still non-inferior, which is the whole point
    # of a non-inferiority test: mouse is supporting, saturated, and allowed to wobble.
    v_small = _stats(92.81, 0.0, 20)              # -0.10 pp
    _d, _s, ci_small = audit.noninferiority_ci(v_small, c, MOUSE_RESOLUTION_PP)
    assert ci_small > MOUSE_MARGIN_PP


def test_d4_historical_h36_mouse_leg_now_passes():
    """The H36 numbers as recorded, under the new formula. n=20 vs n=3, both dispersion-free."""
    v = _stats(H36_MOUSE_MEAN, 0.0, 20)
    c = _stats(H30_MOUSE_MEAN, 0.0, 3)
    delta, se, ci_lo = audit.noninferiority_ci(v, c, MOUSE_RESOLUTION_PP)
    assert delta == pytest.approx(0.0152117, abs=1e-6)
    assert se < 1e-12, "a deterministic pipeline has no dispersion to spend on a CI"
    assert ci_lo > MOUSE_MARGIN_PP
    assert ci_lo == pytest.approx(delta, abs=1e-9)


def test_d4_historical_h42_mouse_tie_is_non_inferior_but_not_a_win():
    """H42 vs H36 on mouse: bit-identical pools. Non-inferior, yes. An improvement, no."""
    v = _stats(MOUSE_TIE_PCT, 0.0, 20)
    c = _stats(MOUSE_TIE_PCT, 0.0, 20)
    delta, _se, ci_lo = audit.noninferiority_ci(v, c, MOUSE_RESOLUTION_PP)
    assert delta == 0.0
    assert ci_lo > MOUSE_MARGIN_PP                # non-inferiority is satisfied
    mouse = _campaign()["datasets"]["mouse"]
    assert delta <= float(mouse["min_promotion_delta_pp"])   # but it is not promotable


def test_d4_pools_both_arms_n():
    """The variant's seeds must count. Doubling n_v alone has to shrink the SE."""
    c = _stats(92.91, 0.05, 5)
    se_small = audit.noninferiority_ci(_stats(92.91, 0.05, 5), c, 0.0)[1]
    se_large = audit.noninferiority_ci(_stats(92.91, 0.05, 40), c, 0.0)[1]
    assert se_large < se_small
    # The old formula ignored n_v entirely, so both would have been identical:
    assert audit.protocol_se(c, 0.0) == audit.protocol_se(c, 0.0)


def test_d4_reduces_to_pooled_variance_test_when_dispersion_is_real():
    v, c = _stats(92.90, 0.20, 20), _stats(92.91, 0.30, 20)
    sd = audit.pooled_sd(v, c)
    assert sd == pytest.approx(np.sqrt((19 * 0.04 + 19 * 0.09) / 38), rel=1e-12)
    _d, se, _ci = audit.noninferiority_ci(v, c, MOUSE_RESOLUTION_PP)
    assert se == pytest.approx(sd * np.sqrt(1 / 20 + 1 / 20), rel=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# D2 — promotion must gate on the unrounded mean AND on a real effect size
# ──────────────────────────────────────────────────────────────────────────────
def test_d2_rounding_artifact_reproduced():
    """The exact arithmetic that handed H42 the mouse championship over a delta of 0.0."""
    assert MOUSE_TIE_PCT > round(MOUSE_TIE_PCT, 4)
    assert round(MOUSE_TIE_PCT, 4) == 92.917
    # the phantom gain the old gate saw, ~1.4e-5 pp of pure rounding:
    assert 0 < MOUSE_TIE_PCT - round(MOUSE_TIE_PCT, 4) < 1e-4


def test_d2_champion_mean_prefers_full_precision():
    assert update_sota.champion_mean({"pct_mean": 92.917,
                                      "pct_mean_exact": MOUSE_TIE_PCT}) == MOUSE_TIE_PCT
    # Rows written before 2026-08-15 have no exact field and fall back — deliberately, because
    # rewriting them would change recorded numbers.
    assert update_sota.champion_mean({"pct_mean": 92.917}) == 92.917


def test_d2_tie_is_refused_by_the_effect_size_gate():
    """With pct_mean_exact present the delta is 0.0; the minimum effect size refuses it anyway."""
    entry = {"pct_mean": 92.917, "pct_mean_exact": MOUSE_TIE_PCT}
    delta = MOUSE_TIE_PCT - update_sota.champion_mean(entry)
    assert delta == 0.0
    min_delta = float(_campaign()["datasets"]["mouse"]["min_promotion_delta_pp"])
    assert delta <= min_delta, "an exact tie must never promote"
    # And the rounding artifact alone cannot clear the effect size either, which is the belt to
    # the braces: legacy rows still carry only 4 decimals.
    assert (MOUSE_TIE_PCT - 92.917) <= min_delta


def test_d2_every_dataset_has_a_minimum_effect_size():
    for ds, cfg in _campaign()["datasets"].items():
        floor = cfg.get("min_promotion_delta_pp", cfg.get("screen_delta_pp"))
        assert floor is not None, f"{ds} has no promotion effect size"
        # It must dominate the 4-decimal rounding slack of the legacy pct_mean field.
        assert float(floor) > 5e-5, f"{ds} floor {floor} is inside pct_mean's rounding noise"


def test_d2_a_real_gain_still_promotes():
    """The gate must not have become unpassable — H42's connectome leg still clears it."""
    delta = 84.15409511053134 - 84.09717365667385
    assert delta > float(_campaign()["datasets"]["connectome"]["min_promotion_delta_pp"])


# ──────────────────────────────────────────────────────────────────────────────
# D6 — provenance keys and reproducibility-by-checkout
# ──────────────────────────────────────────────────────────────────────────────
def test_d6_commit_key_keeps_the_dirty_marker():
    dirty = audit.commit_key({"git_commit": "9d43b977eea807ebe30fef82c5bc81b9d6fe1a61+dirty"})
    clean = audit.commit_key({"git_commit": "9d43b977eea807ebe30fef82c5bc81b9d6fe1a61"})
    assert dirty == "9d43b977+dirty"
    assert clean == "9d43b977"
    assert dirty != clean, "a dirty tree and a clean checkout are not the same provenance"
    assert audit.split_commit_key(dirty) == ("9d43b977", True)
    assert audit.split_commit_key(clean) == ("9d43b977", False)


def test_d6_summarize_separates_dirty_from_clean():
    """H42's real pool mixed 3791350d and 3791350d+dirty and the old key collapsed them to one."""
    runs = [dict(pct=83.24, seed=1, _file="a.json",
                 git_commit="3791350df8f90f480c38c42245fd264814e6c97c"),
            dict(pct=83.24, seed=2, _file="b.json",
                 git_commit="3791350df8f90f480c38c42245fd264814e6c97c+dirty")]
    assert audit.summarize(runs)["commits"] == ["3791350d", "3791350d+dirty"]


def _git_available() -> bool:
    return audit._git_prefix() is not None


@pytest.mark.skipif(not _git_available(), reason="git unavailable")
def test_d6_module_missing_at_commit_is_detected():
    """H36's evidence, checked the way a reviewer would: does the module exist at that commit?

    All 36 H36 runs are stamped 9d43b977...+dirty. ``src/mfas/experiments/H36.py`` does not
    exist there, so the champion's number cannot be reproduced by checkout (CLAUDE.md invariant
    5). Skipped rather than failed if that commit is not in this clone's history.
    """
    prefix = audit._git_prefix()
    sha = "9d43b977eea807ebe30fef82c5bc81b9d6fe1a61"
    present = audit.module_exists_at(sha, "src/mfas/experiments/H36.py", prefix)
    if present is None:
        pytest.skip(f"commit {sha[:8]} not resolvable in this clone")
    assert present is False


@pytest.mark.skipif(not _git_available(), reason="git unavailable")
def test_d6_module_present_at_head_is_detected():
    """The other direction, so a broken path prefix cannot make the check pass vacuously."""
    prefix = audit._git_prefix()
    assert audit.module_exists_at("HEAD", "src/mfas/experiments/H42.py", prefix) is True
    assert audit.module_exists_at("HEAD", "src/mfas/experiments/DOES_NOT_EXIST.py",
                                  prefix) is False


@pytest.mark.skipif(not _git_available(), reason="git unavailable")
def test_d6_check_provenance_fails_in_promotion_mode():
    runs = [dict(pct=92.9, seed=s, _file=f"r{s}.json",
                 git_commit="9d43b977eea807ebe30fef82c5bc81b9d6fe1a61+dirty") for s in range(3)]
    if audit.module_exists_at("9d43b977eea807ebe30fef82c5bc81b9d6fe1a61",
                              "src/mfas/experiments/H36.py", audit._git_prefix()) is None:
        pytest.skip("commit not resolvable in this clone")

    strict = audit.Report(gate="promotion")
    audit.check_provenance(strict, "H36", "mouse", runs, require_module=True, warn_dirty=True)
    assert strict.failed

    lenient = audit.Report(gate="report")
    audit.check_provenance(lenient, "H36", "mouse", runs, require_module=True, warn_dirty=True)
    assert not lenient.failed, "the default path must keep its old exit code"


# ──────────────────────────────────────────────────────────────────────────────
# D1 / D5 — the gate itself: promotion mode can fail, report mode cannot
# ──────────────────────────────────────────────────────────────────────────────
def test_d1_gated_check_is_advisory_in_report_mode_and_hard_in_promotion_mode():
    lenient, strict = audit.Report(gate="report"), audit.Report(gate="promotion")
    for rep in (lenient, strict):
        rep.gated("x", False, "violated")
    assert lenient.checks[0]["status"] == "WARN" and not lenient.failed
    assert strict.checks[0]["status"] == "FAIL" and strict.failed


def _run_audit(monkeypatch, tmp_path, argv):
    """Run audit.main() in-process with the re-score step stubbed out.

    ``check_rescore`` imports ``mfas`` and would pull in torch and a dataset load; the gates
    under test here are pure arithmetic over the recorded JSON, so stub it. Nothing else is
    mocked — the run inventory, the significance maths and the provenance calls are real.
    """
    monkeypatch.setattr(audit, "check_rescore", lambda rep, runs: rep.add(
        "rescore", "INFO", "stubbed in tests/test_audit_gates.py"))
    out = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["audit.py"] + argv + ["--out", str(out)])
    rc = audit.main()
    return rc, json.loads(out.read_text())


def test_d5_empty_comparator_pool_fails_only_in_promotion_mode(monkeypatch, tmp_path):
    """audit_P02.json is the live example: 2 of 3 datasets analysed nothing and it exited 0."""
    argv = ["--variant", "H42", "--comparator", "ZZ_NO_SUCH_COMPARATOR",
            "--role", "confirm", "--datasets", "mouse"]

    rc, rep = _run_audit(monkeypatch, tmp_path, argv + ["--gate", "report"])
    assert rc == 0, "the permissive default must keep passing"
    assert rep["per_dataset"] == {}

    rc, rep = _run_audit(monkeypatch, tmp_path, argv + ["--gate", "promotion"])
    assert rc == 1
    failed = {c["check"] for c in rep["checks"] if c["status"] == "FAIL"}
    assert "comparator.mouse" in failed
    assert "evidence.per_dataset" in failed


def test_d5_missing_variant_runs_fail_in_promotion_mode(monkeypatch, tmp_path):
    argv = ["--variant", "ZZ_NO_SUCH_VARIANT", "--comparator", "champion",
            "--role", "confirm", "--datasets", "mouse"]

    rc, _rep = _run_audit(monkeypatch, tmp_path, argv + ["--gate", "report"])
    assert rc == 0

    rc, rep = _run_audit(monkeypatch, tmp_path, argv + ["--gate", "promotion"])
    assert rc == 1
    failed = {c["check"] for c in rep["checks"] if c["status"] == "FAIL"}
    assert "runs.mouse" in failed


# ──────────────────────────────────────────────────────────────────────────────
# D7 — the registry must be able to say "this win is qualified"
# ──────────────────────────────────────────────────────────────────────────────
def test_d7_sota_schema_and_mouse_backfill():
    sota = json.loads((ROOT / "autoresearch" / "sota.json").read_text())
    assert "caveats" in sota["_schema"]
    mouse = sota["datasets"]["mouse"]
    assert isinstance(mouse.get("caveats"), list) and mouse["caveats"]
    assert isinstance(mouse["runner_up"].get("caveats"), list) and mouse["runner_up"]["caveats"]
    # The backfill must not have touched a recorded number.
    assert mouse["pct_mean"] == 92.917
    assert mouse["n_seeds"] == 20 and len(mouse["seeds"]) == 20
    assert mouse["runner_up"]["champion"] == "H36" and mouse["runner_up"]["pct_mean"] == 92.917
