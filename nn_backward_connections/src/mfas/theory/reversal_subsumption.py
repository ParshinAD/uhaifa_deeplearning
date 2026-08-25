"""The Reversal Subsumption Lemma — a T0 certificate.

This is a *proof*, checked exactly, not a measurement. It therefore holds on every
graph, at every size, forever, and it cost zero GPU time.

Context. Two independent routes pointed at "add a block-reversal move": an invariant
probe over the six discrete move classes found that no atomic move ever inverts the
relative order of the nodes it moves (at most 2-3 ascending runs, where a length-L
reversal would give L), and the linear-ordering literature reports insertion
alternated with reversal (Chanas-Kobylanski) as a strong hybrid. Both routes were
wrong *for the descent framing*, and this module says why. Convergence of two routes
on the same candidate is a reason to re-read the source, never evidence.

THE LEMMA. Let sigma be an ordering and B = (b_1, ..., b_L) a CONTIGUOUS block in it.
Reversing B changes no edge incident to a node outside B (the contiguous-block lemma),
so the gain is decided entirely inside B:

    gain_reverse(B) = sum_{p < q} ( w[b_q, b_p] - w[b_p, b_q] )                    (1)

because reversal flips the relative order of exactly the ordered pairs (p, q), p < q.

Now let r_p be the single-node relocation that moves b_p to the RIGHT END of B,
evaluated on the ORIGINAL sigma. It flips exactly the pairs (p, q) for q > p:

    gain_relocate(b_p) = sum_{q > p} ( w[b_q, b_p] - w[b_p, b_q] )                 (2)

Summing (2) over p = 1..L-1 telescopes into (1):

    gain_reverse(B) = sum_{p=1}^{L-1} gain_relocate(b_p)                           (3)

COROLLARY (the operational content). If sigma is a fixed point of the full-range
exact-gain insertion move -- i.e. every single-node relocation has gain <= 0, which
is exactly what stage 3 of the champion pipeline drives sigma to -- then every term
of (3) is <= 0, hence gain_reverse(B) <= 0 for EVERY contiguous block B.

    A 1-opt local optimum is already a block-reversal local optimum.

So block reversal, used as a DESCENT neighbourhood on a sift-converged order, cannot
find an improving move. Adding it is provably redundant work.

SCOPE, stated so this certificate cannot smother a live idea:
  * It kills reversal as `operator_role = descent_neighbourhood` only.
  * It says NOTHING about reversal as a PERTURBATION (a deliberately worsening kick
    used to escape a local optimum, as in an iterated-local-search wrapper). That is
    a different operator role and is filed separately as H57.
  * It assumes the block is CONTIGUOUS in the current order. A reversal of a
    non-contiguous selected set is not covered.
  * It assumes each relocation gain is evaluated on the original sigma. That is what
    makes (3) exact, and it is also why the sum is NOT an estimate of what a sequence
    of applied moves would recover.

Run this module directly to re-check the certificate.
"""
from __future__ import annotations

import numpy as np

__all__ = ["reversal_gain", "relocation_gain_sum", "check_certificate"]


def _ff_weight(order: np.ndarray, A: np.ndarray) -> float:
    """Total feedforward weight of ``order`` under dense weight matrix ``A``."""
    n = order.size
    total = 0.0
    for a in range(n):
        total += float(A[order[a], order[a + 1:]].sum())
    return total


def reversal_gain(order: np.ndarray, A: np.ndarray, i: int, L: int) -> float:
    """Exact gain of reversing the contiguous block at positions [i, i+L)."""
    rev = order.copy()
    rev[i:i + L] = order[i:i + L][::-1]
    return _ff_weight(rev, A) - _ff_weight(order, A)


def relocation_gain_sum(order: np.ndarray, A: np.ndarray, i: int, L: int) -> float:
    """Sum over p<L of the gain of relocating b_p to the right end of the block.

    Each term is evaluated on the ORIGINAL ``order`` -- the gains are *not* applied
    in sequence. That independence is what makes the identity exact.
    """
    total = 0.0
    for p in range(L - 1):
        moved = order.copy()
        seg = list(moved[i:i + L])
        seg.append(seg.pop(p))
        moved[i:i + L] = seg
        total += _ff_weight(moved, A) - _ff_weight(order, A)
    return total


def _random_instance(rng: np.random.Generator, n: int, layered: bool) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = (0.6 if i < j else 0.15) if layered else 0.45
            if rng.random() < p:
                A[i, j] = float(rng.integers(1, 12))
    return A


def check_certificate(trials: int = 200, seed: int = 20260825) -> float:
    """Re-check identity (3) on random instances. Returns the max absolute deviation.

    The deviation must be EXACTLY zero: both sides are sums of the same finite set of
    weights, so any nonzero value means the implementation, not the algebra, is wrong.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        n = int(rng.integers(5, 13))
        A = _random_instance(rng, n, layered=bool(rng.random() < 0.5))
        order = rng.permutation(n)
        i = int(rng.integers(0, n - 1))
        L = int(rng.integers(2, n - i + 1))
        worst = max(worst, abs(reversal_gain(order, A, i, L)
                               - relocation_gain_sum(order, A, i, L)))
    return worst


if __name__ == "__main__":
    dev = check_certificate()
    print(f"max |gain_reverse - sum gain_relocate| = {dev:.3e} over 200 trials, n in [5,12]")
    if dev != 0.0:
        raise SystemExit("CERTIFICATE FAILED — the lemma or this implementation is wrong")
    print("CERTIFICATE HOLDS EXACTLY.")
    print("Corollary: a 1-opt local optimum is already a block-reversal local optimum,")
    print("so reversal as a DESCENT neighbourhood is provably redundant (H56).")
    print("Reversal as a PERTURBATION is untouched by this and is filed as H57.")
