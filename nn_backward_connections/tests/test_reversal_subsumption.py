"""Tests for the Reversal Subsumption Lemma certificate (H56).

Two levels, deliberately:
  1. the algebraic identity, which is what the certificate re-runs;
  2. the OPERATIONAL corollary, brute-forced -- at a 1-opt local optimum no block
     reversal improves. The corollary is the claim that actually kills the move, so
     it gets its own independent check rather than riding on the identity.

Plus a negative control: away from a 1-opt local optimum an improving reversal DOES
exist, so the test suite cannot pass by the corollary being vacuous.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfas.theory.reversal_subsumption import (
    _ff_weight,
    _random_instance,
    check_certificate,
    relocation_gain_sum,
    reversal_gain,
)


def test_identity_holds_exactly():
    """gain(reverse B) == sum of individual relocation gains, to the last bit."""
    assert check_certificate(trials=200, seed=20260825) == 0.0


def test_identity_holds_under_a_different_seed():
    """The certificate is not tuned to one random stream."""
    assert check_certificate(trials=80, seed=1) == 0.0


def _best_relocation_gain(order: np.ndarray, A: np.ndarray) -> float:
    """Best gain over ALL single-node relocations to ALL gaps (full-range 1-opt)."""
    n = order.size
    base = _ff_weight(order, A)
    best = 0.0
    for p in range(n):
        rest = list(order)
        node = rest.pop(p)
        for q in range(n):
            cand = np.array(rest[:q] + [node] + rest[q:], dtype=order.dtype)
            best = max(best, _ff_weight(cand, A) - base)
    return best


def _descend_to_1opt(order: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Greedy full-range relocation descent until no single move improves."""
    n = order.size
    cur = order.copy()
    while True:
        base = _ff_weight(cur, A)
        best_gain, best_order = 1e-12, None
        for p in range(n):
            rest = list(cur)
            node = rest.pop(p)
            for q in range(n):
                cand = np.array(rest[:q] + [node] + rest[q:], dtype=cur.dtype)
                gain = _ff_weight(cand, A) - base
                if gain > best_gain:
                    best_gain, best_order = gain, cand
        if best_order is None:
            return cur
        cur = best_order


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_corollary_no_reversal_improves_at_a_1opt_optimum(seed):
    """THE OPERATIONAL CLAIM: at a 1-opt local optimum every block reversal is <= 0."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 9))
    A = _random_instance(rng, n, layered=bool(seed % 2))
    opt = _descend_to_1opt(rng.permutation(n), A)

    assert _best_relocation_gain(opt, A) <= 1e-9, "fixture is not a 1-opt local optimum"

    for i in range(n - 1):
        for L in range(2, n - i + 1):
            assert reversal_gain(opt, A, i, L) <= 1e-9, (
                f"a reversal improved at a 1-opt optimum: i={i} L={L} -- the lemma is false"
            )


def test_negative_control_reversal_can_improve_away_from_a_1opt_optimum():
    """Guard against a vacuous corollary: improving reversals must exist somewhere."""
    rng = np.random.default_rng(11)
    found = False
    for _ in range(40):
        n = int(rng.integers(6, 9))
        A = _random_instance(rng, n, layered=True)
        order = rng.permutation(n)
        if any(reversal_gain(order, A, i, L) > 1e-9
               for i in range(n - 1) for L in range(2, n - i + 1)):
            found = True
            break
    assert found, "no improving reversal found anywhere -- the corollary would be vacuous"


def test_relocation_sum_is_evaluated_on_the_original_order():
    """The terms must be independent; applying them in sequence breaks the identity."""
    rng = np.random.default_rng(7)
    A = _random_instance(rng, 8, layered=False)
    order = rng.permutation(8)
    i, L = 1, 5

    independent = relocation_gain_sum(order, A, i, L)

    sequential, cur = 0.0, order.copy()
    for p in range(L - 1):
        base = _ff_weight(cur, A)
        seg = list(cur[i:i + L])
        seg.append(seg.pop(p))
        cur[i:i + L] = seg
        sequential += _ff_weight(cur, A) - base

    assert independent == pytest.approx(reversal_gain(order, A, i, L))
    assert sequential != pytest.approx(independent), (
        "sequential and independent sums coincided; the test instance is degenerate"
    )
