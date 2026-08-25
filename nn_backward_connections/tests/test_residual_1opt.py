"""Pinned regression test for the residual 1-opt diagnostic.

The numbers below are the instrument's calibration. If they move, the environment
moved (dataset, loader, insertion kernel, or the anchor artifact) and every
downstream reading is suspect -- so this test exists to make that failure loud
rather than silent.

Measured on connectome, this machine, 2026-08-25:
    parity anchor  82.916074%  ->  17,949 movers, naive_sum 0.5461 pp, best_single 1436
    reference      84.614678%  ->     131 movers, naive_sum 0.0010 pp, best_single   27
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from mfas.io import load_dataset

# experiments/diagnostics/ holds standalone scripts, not a package, so load by path
# rather than adding __init__.py files to a directory of one-off diagnostics.
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "residual_1opt",
    Path(__file__).resolve().parent.parent / "experiments" / "diagnostics" / "residual_1opt.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
# register before exec: @dataclass resolves cls.__module__ through sys.modules
sys.modules["residual_1opt"] = _MOD
_SPEC.loader.exec_module(_MOD)
load_order, residual_1opt = _MOD.load_order, _MOD.residual_1opt

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def connectome():
    return load_dataset("connectome")


def test_anchor_residual_is_pinned(connectome):
    rank, label = load_order(connectome, "anchor", None)
    r = residual_1opt(connectome, rank, dataset="connectome", label=label)
    assert r.pct == pytest.approx(82.916074, abs=1e-5)
    assert r.movers == 17_949
    assert r.naive_sum_pp == pytest.approx(0.5461, abs=5e-4)
    assert r.best_single_gain == pytest.approx(1436.0)


def test_reference_is_essentially_1opt_dry(connectome):
    """The load-bearing fact: 'more 1-opt' is not what closes the remaining gap."""
    rank, label = load_order(connectome, "reference", None)
    r = residual_1opt(connectome, rank, dataset="connectome", label=label)
    assert r.pct == pytest.approx(84.614678, abs=1e-5)
    assert r.movers == 131
    assert r.naive_sum_pp < 0.005
    # and it is two orders of magnitude drier than the anchor
    assert r.movers * 100 < 17_949


def test_rejects_a_non_permutation(connectome):
    bad = np.zeros(connectome.n_nodes, dtype=np.int64)
    with pytest.raises(ValueError):
        residual_1opt(connectome, bad, dataset="connectome", label="bad")


def test_naive_sum_is_an_upper_bound_not_an_estimate(connectome):
    """Guard the docstring's rule with a check, not a comment.

    The per-node gains are all computed against the same frozen order, so they
    overlap. On the anchor the sum is 0.5461 pp while the true remaining headroom to
    the reference is 1.6986 pp -- the sum being far BELOW that is not a contradiction,
    it is a reminder that the quantity is neither an upper nor a lower bound on what a
    SEQUENCE of applied moves recovers. It bounds only the one-shot, non-interacting
    total. This test pins the relationship so nobody re-reads it as a forecast.
    """
    rank, _ = load_order(connectome, "anchor", None)
    r = residual_1opt(connectome, rank, dataset="connectome", label="anchor")
    headroom_to_reference = 84.614678 - r.pct
    assert r.naive_sum_pp < headroom_to_reference
    assert r.naive_sum_pp > 0.0
