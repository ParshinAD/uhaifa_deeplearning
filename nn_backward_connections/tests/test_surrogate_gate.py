"""Regression tests for the surrogate gate — the cases Phase 6.5 / 6.6 paid for.

Every assertion here encodes a conclusion that cost real compute to establish, so that a
future surrogate idea is checked against them in milliseconds instead of hours:

* `tanh` IS the sigmoid           -> the gate must REJECT it as a beta rescaling (H03/A-SCALE);
* the naive one-sided shape       -> the gate must REJECT it (collapse is the global maximum);
* H38's margin form               -> the gate must NOT reject it, and must flag its kink;
* H37's algebraic-tail form       -> the gate must NOT reject it, and must flag the z=0 defect;
* the sigmoid itself              -> sanity anchor: recognised as the sigmoid, no collapse.

Dataset-free by design (no connectome load), so this stays in the fast test path.
"""
from __future__ import annotations

import torch

from mfas.analysis.surrogate_gate import (
    autograd_check,
    cheap_gate,
    collapse_check,
    monotone_bounded_check,
    sigmoid_reparametrization_residual,
    transition_width,
)


# ── the shapes under test (kept local so the tests never import a variant) ────
def _sigmoid(z):
    return torch.sigmoid(z)


def _tanh_form(z):
    """The literal reading of TODO 7: (tanh(z/2)+1)/2. Provably == sigmoid(z)."""
    return 0.5 * (torch.tanh(z / 2.0) + 1.0)


def _tanh_slow(z):
    """'A tanh that decays more slowly' == sigmoid at beta/5 — still a rescaling."""
    return 0.5 * (torch.tanh(z / 10.0) + 1.0)


def _asym_naive(z, T=1.5):
    """H38's FIRST form: flat at 1 for z >= 0. Degenerate — collapse is the global max."""
    return torch.where(z >= 0, torch.ones_like(z), 1.0 + torch.tanh(z / T))


def _asym_margin(z, M=0.75, T=1.5):
    """H38 as shipped: the margin makes g(0) < sup g, removing the degeneracy."""
    return torch.where(z >= M, torch.ones_like(z), 1.0 + torch.tanh((z - M) / T))


def _poly_q4(z, q=4.0):
    """H37's algebraic-tail shape; `torch.sign` zeroes its autograd exactly at z = 0."""
    return 0.5 + 0.5 * torch.sign(z) * (1.0 - (1.0 + z.abs()) ** (-q))


# ── 1. a reparametrized sigmoid must be recognised and rejected ───────────────
def test_tanh_is_the_sigmoid_and_is_rejected():
    r = sigmoid_reparametrization_residual(_tanh_form)
    assert r["residual"] < 1e-9, f"(tanh(z/2)+1)/2 must equal sigmoid(z); got {r}"
    assert abs(r["a"] - 1.0) < 0.05

    rep = cheap_gate(_tanh_form, "tanh_half")
    assert rep["verdict"] == "REJECT"
    # a == 1, so it is not merely a rescaling: it is the baseline sigmoid itself.
    assert rep["is_baseline_sigmoid"] is True
    assert any("baseline sigmoid" in s for s in rep["reasons"])


def test_slower_decaying_tanh_is_also_just_a_rescaling():
    r = sigmoid_reparametrization_residual(_tanh_slow)
    assert r["residual"] < 1e-9
    # tanh(z/10) == sigmoid(z/5): the fitted scale must be ~0.2, not ~1.
    assert abs(r["a"] - 0.2) < 0.02, r
    assert cheap_gate(_tanh_slow, "tanh_slow")["verdict"] == "REJECT"


def test_sigmoid_itself_is_the_anchor():
    r = sigmoid_reparametrization_residual(_sigmoid)
    assert r["residual"] < 1e-12 and abs(r["a"] - 1.0) < 1e-6
    assert not collapse_check(_sigmoid)["collapse_is_global_max"]


# ── 2. the collapse degeneracy — the check that saves a wasted large-graph run ─
def test_naive_one_sided_shape_collapses_and_is_rejected():
    c = collapse_check(_asym_naive)
    assert c["collapse_is_global_max"], (
        "g(0) == sup g, so P=const attains the surrogate's global maximum; the gate must "
        f"detect it. Got {c}")
    rep = cheap_gate(_asym_naive, "asym_naive")
    assert rep["verdict"] == "REJECT"
    assert any("P=const" in s for s in rep["reasons"])


def test_margin_removes_the_collapse_degeneracy():
    c = collapse_check(_asym_margin)
    assert not c["collapse_is_global_max"]
    # g(0) = 1 + tanh(-M/T) = 1 + tanh(-0.5) with M=0.75, T=1.5  (float64!)
    expected = 1.0 + torch.tanh(torch.tensor(-0.5, dtype=torch.float64)).item()
    assert abs(c["g_at_0"] - expected) < 1e-9
    assert c["headroom_frac_of_range"] > 0.4
    assert cheap_gate(_asym_margin, "H38")["verdict"] != "REJECT"


# ── 3. monotonicity / argmax preservation ─────────────────────────────────────
def test_shipped_shapes_are_monotone_and_bounded():
    for name, fn in (("sigmoid", _sigmoid), ("H38", _asym_margin), ("H37", _poly_q4)):
        m = monotone_bounded_check(fn)
        assert m["monotone_non_decreasing"], f"{name} must be non-decreasing in Delta"
        assert m["bounded"], f"{name} must be bounded"
        assert m["min_value"] >= -1e-12 and m["max_value"] <= 1.0 + 1e-12


# ── 4. the two autograd defects, pinned so they stay documented ───────────────
def test_h37_autograd_is_zero_at_the_origin():
    """`torch.sign(0) == 0` silently zeroes the gradient at z = 0 — a real, documented defect."""
    a = autograd_check(_poly_q4, probes=[0.0])
    assert not a["all_match"] and a["mismatch_points"] == [0.0]
    assert a["rows"][0]["autograd"] == 0.0
    assert a["rows"][0]["central_diff"] > 1.0        # true slope is q/2 = 2


def test_h38_autograd_is_zero_exactly_at_the_margin():
    """`torch.where` zeroes the gradient exactly at the branch point z = M = 0.75."""
    a = autograd_check(_asym_margin, probes=[0.75])
    assert not a["all_match"] and a["mismatch_points"] == [0.75]
    assert a["rows"][0]["autograd"] == 0.0
    # away from the branch point autograd is correct
    assert autograd_check(_asym_margin, probes=[-3.0, -0.5, 0.0, 0.5, 2.0])["all_match"]


# ── 5. transition width — the comparator Q04 used ─────────────────────────────
def test_transition_width_ranks_shapes_as_measured_in_q04():
    w_sig = transition_width(_sigmoid)
    w_poly = transition_width(_poly_q4)
    assert abs(w_sig - 2.1973) < 0.01, w_sig
    assert abs(w_poly - 0.4954) < 0.01, w_poly
    # the width ratio is the SCALE constant H37 used to match cores (2.1973/0.4954)
    assert abs(w_sig / w_poly - 4.4361) < 0.05
