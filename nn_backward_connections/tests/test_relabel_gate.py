"""Tests for the relabelling-robustness gate (P09).

These pin the rule pre-registered in ``experiments/outputs/proto_P09_prereg.json``, as amended by
the cycle-9 critic (D1, D2, D5):

    PASS iff  (a) delta_r > min_promotion_delta_pp for EVERY relabelling r,
              (b) the paired one-sided 95% lower bound on the mean delta also clears it,
              (c) at least ``min_relabellings`` non-identity relabellings are present, and
              (d) the identity row REPRODUCES the recorded confirm pools.

(d) is the one that nearly got away: the pre-registration made it a VOIDING condition and the
first implementation checked only that an identity row existed, leaving the reproduction claim as
English prose in the registry that no code read. That is the same shape of hole P09 was convened
to close.

The other property worth pinning is monotonicity of the min-rule: adding relabellings can only
make it harder. That is why a min-rule is used at all, and why it is not used alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autoresearch import relabel_gate  # noqa: E402

# A pool the identity row of every fixture below reproduces exactly.
REC_V, REC_C = 84.17502222088821, 84.15409511053134


def _study(deltas, with_identity=True, variant="V", comparator="C", dataset="connectome",
           identity_v=REC_V, identity_c=REC_C):
    """Build a minimal study: ``deltas[0]`` is the identity row when ``with_identity``."""
    rows = []
    for i, d in enumerate(deltas):
        identity = (i == 0 and with_identity)
        v = identity_v if identity else REC_V + 0.001 * i
        rows.append(dict(r=i, identity=identity,
                         comparator=dict(pct=(identity_c if identity else v - d)),
                         variant=dict(pct=v),
                         delta_pp=float(d)))
    return dict(item="TEST", dataset=dataset, comparator=comparator, variant=variant, rows=rows)


def _ev(deltas, min_delta=0.012, min_R=4, **kw):
    return relabel_gate.evaluate(_study(deltas, **kw), min_delta, min_R,
                                 variant_pct=REC_V, comparator_pct=REC_C)


# ── the happy path ────────────────────────────────────────────────────────────────────────
def test_passes_when_every_relabelling_and_the_paired_bound_clear_the_effect_size():
    ev = _ev([0.020, 0.021, 0.019, 0.025, 0.018])
    assert ev["passed"] is True, ev["reasons"]
    assert ev["n_relabellings"] == 4 and ev["identity_reproduces"] is True
    assert ev["min_rule_passed"] is True and ev["paired_bound_passed"] is True
    assert ev["delta_min_pp"] == pytest.approx(0.018)


def test_the_real_connectome_study_passes_its_own_rule():
    """The measured P09 numbers, so a regression in the arithmetic is caught by a real case."""
    ev = _ev([0.020927110356865340, 0.016810698, 0.016399, 0.015854594495650076, 0.017725])
    assert ev["passed"] is True, ev["reasons"]
    assert ev["paired_lower_bound_pp"] == pytest.approx(0.01562, abs=5e-5)


# ── (a) the min-rule ──────────────────────────────────────────────────────────────────────
def test_one_failing_relabelling_sinks_it():
    """A gain that survives four labellings and dies on the fifth is a labelling artefact."""
    ev = _ev([0.020, 0.021, 0.019, 0.025, 0.004])
    assert ev["passed"] is False and ev["min_rule_passed"] is False
    assert ev["n_failures"] == 1 and ev["failures"][0]["r"] == 4
    assert any("r=4" in r for r in ev["reasons"])


def test_a_delta_exactly_at_the_threshold_does_not_pass():
    """Strictly greater, matching audit.py's existing effect-size comparison."""
    assert _ev([0.012, 0.020, 0.020, 0.020, 0.020])["passed"] is False


def test_adding_relabellings_can_only_make_the_min_rule_harder():
    """The reason a min-rule is used: it is monotone in the direction that cannot be gamed.

    A ``z * sigma * sqrt(2/n)`` threshold FALLS as n rises, so a campaign could admit its own
    result by re-running an identical computation more times. Extending a passing study here can
    flip it to fail but never the reverse.
    """
    base = [0.020, 0.021, 0.019, 0.025, 0.018]
    assert _ev(base)["min_rule_passed"] is True
    assert _ev(base + [0.030])["min_rule_passed"] is True
    assert _ev(base + [0.001])["min_rule_passed"] is False


# ── (b) the paired lower bound (D5) ───────────────────────────────────────────────────────
def test_the_paired_bound_can_refuse_what_the_min_rule_allows():
    """Every point above the bar, but so scattered that the mean is not credibly above it."""
    ev = _ev([0.0121, 0.0400, 0.0125, 0.0450, 0.0122])
    assert ev["min_rule_passed"] is True
    assert ev["paired_bound_passed"] is False
    assert ev["passed"] is False
    assert any("lower bound" in r for r in ev["reasons"])


def test_the_paired_bound_tightens_as_relabellings_are_added():
    """Unlike a CI against zero, a CI against a FIXED effect size is consistent, not relaxing."""
    tight = [0.0176, 0.0175, 0.0177, 0.0174, 0.0178]
    few = relabel_gate.evaluate(_study(tight[:3]), 0.012, 2,
                                variant_pct=REC_V, comparator_pct=REC_C)
    many = relabel_gate.evaluate(_study(tight), 0.012, 2,
                                 variant_pct=REC_V, comparator_pct=REC_C)
    assert many["paired_lower_bound_pp"] > few["paired_lower_bound_pp"]


# ── (c) study size ────────────────────────────────────────────────────────────────────────
def test_too_few_relabellings_does_not_pass():
    """Absence of evidence is not evidence: an under-powered study fails closed."""
    ev = _ev([0.020, 0.021, 0.019])
    assert ev["passed"] is False
    assert "only 2 non-identity relabelling(s)" in " ".join(ev["reasons"])


def test_empty_study_does_not_pass():
    ev = relabel_gate.evaluate(dict(rows=[]), 0.012, 4, variant_pct=REC_V, comparator_pct=REC_C)
    assert ev["passed"] is False
    assert ev["delta_min_pp"] is None


# ── (d) the identity anchor (D1) ──────────────────────────────────────────────────────────
def test_missing_identity_row_does_not_pass():
    ev = _ev([0.02] * 5, with_identity=False)
    assert ev["passed"] is False
    assert any("identity" in r for r in ev["reasons"])


def test_identity_row_that_does_not_reproduce_the_confirm_pool_voids_the_study():
    """THE pre-registered voiding condition, now enforced by code rather than by prose."""
    ev = relabel_gate.evaluate(_study([0.02] * 5, identity_v=REC_V + 0.003), 0.012, 4,
                               variant_pct=REC_V, comparator_pct=REC_C)
    assert ev["passed"] is False and ev["identity_reproduces"] is False
    assert any("VOID" in r for r in ev["reasons"])


def test_without_the_recorded_means_the_study_cannot_be_anchored():
    ev = relabel_gate.evaluate(_study([0.02] * 5), 0.012, 4)
    assert ev["passed"] is False
    assert any("recorded confirm means" in r for r in ev["reasons"])


# ── the registry (D2) ─────────────────────────────────────────────────────────────────────
def test_registry_lookup_is_an_exact_triple(tmp_path):
    """A study is evidence only for the (variant, comparator, dataset) it actually ran."""
    study_path = tmp_path / "s.json"
    study_path.write_text(json.dumps(_study([0.02] * 5)))
    index = tmp_path / "relabel_index.json"
    index.write_text(json.dumps({"studies": {"V|C|connectome": {"path": str(study_path)}}}))
    assert relabel_gate.load_study("V", "C", "connectome", index) is not None
    assert relabel_gate.load_study("V", "C", "microns", index) is None
    assert relabel_gate.load_study("V", "OTHER", "connectome", index) is None
    assert relabel_gate.load_study("OTHER", "C", "connectome", index) is None


def test_a_mis_registered_study_is_refused_not_silently_used(tmp_path):
    """D2: the registry key alone was trusted about what a file contains. A typo would attach
    the wrong study to the wrong pair, which is exactly the moving-comparator error."""
    study_path = tmp_path / "s.json"
    study_path.write_text(json.dumps(_study([0.02] * 5, variant="WRONG")))
    index = tmp_path / "relabel_index.json"
    index.write_text(json.dumps({"studies": {"V|C|connectome": {"path": str(study_path)}}}))
    study = relabel_gate.load_study("V", "C", "connectome", index)
    ev = relabel_gate.evaluate(study, 0.012, 4, variant_pct=REC_V, comparator_pct=REC_C)
    assert ev["passed"] is False
    assert any("does not identify itself" in r for r in ev["reasons"])


def test_missing_index_is_not_an_error_but_yields_no_study(tmp_path):
    assert relabel_gate.load_index(tmp_path / "nope.json") == {}
    assert relabel_gate.load_study("V", "C", "connectome", tmp_path / "nope.json") is None


def test_t_quantiles_match_scipy_where_it_matters():
    scipy_stats = pytest.importorskip("scipy.stats")
    for df in (1, 4, 9, 19, 29):
        assert relabel_gate.t95_one_sided(df) == pytest.approx(
            float(scipy_stats.t.ppf(0.95, df)), abs=1e-5)
