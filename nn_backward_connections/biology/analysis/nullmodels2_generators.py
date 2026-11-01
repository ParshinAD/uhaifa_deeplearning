"""Null-ladder v2 generators (study: biology/studies/null_ladder_v2/CHARTER.md).

Two rungs the reviewed G01 ladder (nullmodels_generators.py) does not have.
v1 files are imported, never edited (charter rule 5).

Rung W0 — ``w_topo_shuffle``: real binary topology, weights permuted
=====================================================================
Keep every (src, tgt) slot exactly as in the real graph; permute the weight
column with one uniform permutation. Underlying distribution: uniform over all
assignments of the real weight multiset to the real digraph (microcanonical
"real topology x exchangeable weights"). Preserves the full binary digraph —
degrees, reciprocal pair set, SCCs, motifs — plus the weight multiset/total.
Destroys every weight–edge coupling. The complementary axis to the whole v1
ladder, which always randomized topology: W0 asks how much of the real FF%
lives in WHERE the strong synapses sit. The 2-cycle PAIR SET is real but the
bound Sum min(w_uv, w_vu) changes with the permutation — reported per draw.

Rung L2R — ``dyad_fixedmut``: degrees + the REAL reciprocal backbone
=====================================================================
Split slots into MUTUAL (edges whose reverse exists in the real graph — the
reciprocal/2-cycle subgraph; the real fly is simple, so slots = edges) and
UNIDIRECTIONAL. Keep every mutual slot exactly real (both directions, both
weights). Permute the target column among unidirectional slots only; weights
stay on their slot (= travel with the source stub, the v1 ``w_source``
convention). Repairs, by pairwise target swaps within the uni class:

* self-loops (src == new tgt),
* uni slots landing on a dyad that is mutual in the real graph (either
  direction) — a parallel slot there would change min(W_uv, W_vu) and break
  the exact 2-cycle-bound guarantee,
* uni slots whose REVERSE exists among the current uni slots — that would
  create a NEW mutual dyad and change the dyad census.

Underlying distribution: uniform over digraphs with (i) the real mutual
subgraph fixed pointwise, (ii) the exact unidirectional in/out degree
sequence, conditioned on loop-freedom and no new mutual dyads. Preserves
exactly, per node: total in/out degree, uni in/out degree, mutual degree, and
total out-strength; globally: the real 2-cycle bound (validated bit-exactly in
the selftest). Destroys: which non-reciprocal target each neuron chooses.

Validation:  $PY biology/analysis/nullmodels2_generators.py --selftest
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

V2_VARIANTS = ("w_topo_shuffle", "dyad_fixedmut")


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────
def edge_keys(src: np.ndarray, tgt: np.ndarray, n: int) -> np.ndarray:
    """Ordered-pair keys src*n + tgt as int64 (n^2 for the fly needs int64)."""
    return src.astype(np.int64) * np.int64(n) + tgt.astype(np.int64)


def _in_sorted(sorted_keys: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Vectorized membership of ``queries`` in the sorted int64 array."""
    pos = np.searchsorted(sorted_keys, queries)
    pos = np.minimum(pos, sorted_keys.size - 1)
    return sorted_keys[pos] == queries


def two_cycle_bound(src: np.ndarray, tgt: np.ndarray, w: np.ndarray, n: int
                    ) -> float:
    """Sum over reciprocated unordered pairs of min(W(u->v), W(v->u)).

    Estimator-free lower bound on unavoidable feedback weight. Parallel slots
    are aggregated first, so the bound is a property of the weighted simple
    digraph the slots represent. float64 accumulation is exact here (fly
    weights are < 2^11 and sums stay far below 2^53).
    """
    key = edge_keys(src, tgt, n)
    uk, inv = np.unique(key, return_inverse=True)
    wagg = np.zeros(uk.size, dtype=np.float64)
    np.add.at(wagg, inv, np.asarray(w, dtype=np.float64))
    us, ut = np.divmod(uk, np.int64(n))
    rev = ut * np.int64(n) + us
    pos = np.searchsorted(uk, rev)
    pos = np.minimum(pos, uk.size - 1)
    has_rev = uk[pos] == rev
    wrev = np.where(has_rev, wagg[pos], 0.0)
    # each reciprocated pair contributes min() once per direction -> halve
    return float(np.minimum(wagg, wrev)[has_rev].sum() / 2.0)


def mutual_slot_mask(src: np.ndarray, tgt: np.ndarray, n: int) -> np.ndarray:
    """True for slots whose reverse ordered pair exists in the slot set."""
    key = edge_keys(src, tgt, n)
    skey = np.sort(key)
    rev = tgt.astype(np.int64) * np.int64(n) + src.astype(np.int64)
    return _in_sorted(skey, rev)


# ──────────────────────────────────────────────────────────────────────────────
# W0 — weights shuffled on the real topology
# ──────────────────────────────────────────────────────────────────────────────
def w_topo_shuffle_null(src: np.ndarray, tgt: np.ndarray, weight: np.ndarray,
                        *, seed: int
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Draw one W0 null: identical topology, one uniform weight permutation."""
    src = np.asarray(src)
    tgt = np.asarray(tgt)
    weight = np.asarray(weight)
    rng = np.random.default_rng(seed)
    w_new = weight[rng.permutation(weight.shape[0])].copy()
    stats = dict(variant="w_topo_shuffle", seed=int(seed),
                 n_selfloops_initial=0, n_selfloops_repaired=0,
                 repair_rounds=0, n_duplicate_slots=0)
    return src.copy(), tgt.copy(), w_new, stats


# ──────────────────────────────────────────────────────────────────────────────
# L2R — real reciprocal backbone + exact degrees, uni wiring randomized
# ──────────────────────────────────────────────────────────────────────────────
def dyad_fixedmut_null(src: np.ndarray, tgt: np.ndarray, weight: np.ndarray,
                       *, n: int, seed: int, max_repair_rounds: int = 1000
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Draw one L2R null. See module docstring for the exact ensemble."""
    src = np.asarray(src)
    tgt = np.asarray(tgt)
    weight = np.asarray(weight)
    m = src.shape[0]
    rng = np.random.default_rng(seed)

    mut = mutual_slot_mask(src, tgt, n)
    uni = np.flatnonzero(~mut)
    # ordered keys of the (fixed) mutual subgraph, sorted once
    mut_keys_sorted = np.sort(edge_keys(src[mut], tgt[mut], n))

    new_tgt = tgt.copy()
    new_tgt[uni] = tgt[uni][rng.permutation(uni.size)]

    usrc = src[uni].astype(np.int64)
    n64 = np.int64(n)

    def bad_uni_mask() -> np.ndarray:
        """Violations among uni slots, recomputed from the current state."""
        ut = new_tgt[uni].astype(np.int64)
        fwd = usrc * n64 + ut
        rev = ut * n64 + usrc
        loops = usrc == ut
        on_mut = _in_sorted(mut_keys_sorted, fwd) | _in_sorted(mut_keys_sorted, rev)
        uni_sorted = np.sort(fwd)
        new_mutual = _in_sorted(uni_sorted, rev)
        return loops | on_mut | new_mutual

    def slot_ok(u: int, t: int, uni_sorted: np.ndarray) -> bool:
        """Would a uni slot u->t violate any constraint (vs current uni keys)?"""
        if u == t:
            return False
        fwd = np.int64(u) * n64 + np.int64(t)
        rev = np.int64(t) * n64 + np.int64(u)
        if _in_sorted(mut_keys_sorted, np.array([fwd, rev])).any():
            return False
        if _in_sorted(uni_sorted, np.array([rev]))[0]:
            return False
        return True

    bad = np.flatnonzero(bad_uni_mask())
    n_initial = int(bad.size)
    rounds = 0
    while bad.size > 0 and rounds < max_repair_rounds:
        rounds += 1
        uni_sorted = np.sort(usrc * n64 + new_tgt[uni].astype(np.int64))
        partners = rng.integers(0, uni.size, size=bad.size)
        for bi, pj in zip(bad, partners):
            i, j = uni[bi], uni[pj]
            ti, tj = int(new_tgt[i]), int(new_tgt[j])
            ui, uj = int(src[i]), int(src[j])
            # swap targets only if BOTH slots are clean afterwards
            if slot_ok(ui, tj, uni_sorted) and slot_ok(uj, ti, uni_sorted):
                new_tgt[i], new_tgt[j] = tj, ti
        bad = np.flatnonzero(bad_uni_mask())
    if bad.size > 0:  # pragma: no cover - safety cap
        raise RuntimeError(
            f"dyad_fixedmut repair did not converge: {bad.size} bad uni slots "
            f"left after {max_repair_rounds} rounds")

    key = edge_keys(src, new_tgt, n)
    n_dup = int(m - np.unique(key).size)
    stats = dict(variant="dyad_fixedmut", seed=int(seed),
                 n_mutual_slots=int(mut.sum()), n_uni_slots=int(uni.size),
                 mutual_weight=float(weight[mut].sum()),
                 n_bad_initial=n_initial, n_bad_repaired=n_initial,
                 repair_rounds=rounds, n_duplicate_slots=n_dup)
    return src.copy(), new_tgt, weight.copy(), stats


# ──────────────────────────────────────────────────────────────────────────────
# Self-test: exact preservation guarantees + seed reproducibility
# ──────────────────────────────────────────────────────────────────────────────
def _validate_w0(src, tgt, w, s2, t2, w2) -> None:
    assert np.array_equal(src, s2) and np.array_equal(tgt, t2), \
        "W0 must not touch topology"
    assert np.array_equal(np.sort(w), np.sort(w2)), "weight multiset changed"
    assert w.sum() == w2.sum(), "total weight changed"


def _validate_dyad(src, tgt, w, s2, t2, w2, n) -> None:
    mut = mutual_slot_mask(src, tgt, n)
    # mutual slots bit-identical (topology AND weights)
    assert np.array_equal(src[mut], s2[mut]) and np.array_equal(tgt[mut], t2[mut])
    assert np.array_equal(w[mut], w2[mut]), "mutual weights changed"
    # weights stay on slots globally (w_source convention)
    assert np.array_equal(w, w2), "weights moved off their slots"
    # total in/out degree per node exact
    assert np.array_equal(np.bincount(src, minlength=n),
                          np.bincount(s2, minlength=n)), "out-degree changed"
    assert np.array_equal(np.bincount(tgt, minlength=n),
                          np.bincount(t2, minlength=n)), "in-degree changed"
    # uni in/out degree per node exact
    assert np.array_equal(np.bincount(src[~mut], minlength=n),
                          np.bincount(s2[~mut], minlength=n))
    assert np.array_equal(np.bincount(tgt[~mut], minlength=n),
                          np.bincount(t2[~mut], minlength=n))
    # out-strength exact (weights on slots, src untouched)
    sw = np.bincount(src, weights=np.asarray(w, np.float64), minlength=n)
    sw2 = np.bincount(s2, weights=np.asarray(w2, np.float64), minlength=n)
    assert np.array_equal(sw, sw2), "out-strength changed"
    # no self-loops
    assert int((s2 == t2).sum()) == 0, "self-loops remain"
    # dyad census: no NEW mutual dyads — every mutual slot of the null is real
    mut2 = mutual_slot_mask(s2, t2, n)
    assert np.array_equal(mut2, mut), "dyad census changed (new/lost mutuals)"
    # the decisive guarantee: 2-cycle bound preserved exactly
    b_real = two_cycle_bound(src, tgt, w, n)
    b_null = two_cycle_bound(s2, t2, w2, n)
    assert b_real == b_null, f"2-cycle bound changed: {b_real} != {b_null}"


def _selftest() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mfas.io import load_dataset

    for name in ("mouse", "connectome"):
        g = load_dataset(name)
        src = np.asarray(g.src); tgt = np.asarray(g.tgt); w = np.asarray(g.weight)
        n = g.n_nodes
        b_real = two_cycle_bound(src, tgt, w, n)
        print(f"{name}: n={n:,} m={src.size:,} 2-cycle bound "
              f"{100.0 * b_real / w.sum():.4f}% of W")
        # W0
        s2, t2, w2, stats = w_topo_shuffle_null(src, tgt, w, seed=7)
        _validate_w0(src, tgt, w, s2, t2, w2)
        _, _, w3, _ = w_topo_shuffle_null(src, tgt, w, seed=7)
        assert np.array_equal(w2, w3), "same seed must reproduce bit-identically"
        _, _, w4, _ = w_topo_shuffle_null(src, tgt, w, seed=8)
        assert not np.array_equal(w2, w4), "different seeds gave same draw"
        b0 = two_cycle_bound(s2, t2, w2, n)
        print(f"OK {name:10s} w_topo_shuffle   null 2-cycle bound "
              f"{100.0 * b0 / w.sum():.4f}% of W (real {100.0 * b_real / w.sum():.4f}%)")
        # L2R
        s2, t2, w2, stats = dyad_fixedmut_null(src, tgt, w, n=n, seed=7)
        _validate_dyad(src, tgt, w, s2, t2, w2, n)
        _, t3, _, _ = dyad_fixedmut_null(src, tgt, w, n=n, seed=7)
        assert np.array_equal(t2, t3), "same seed must reproduce bit-identically"
        _, t4, _, _ = dyad_fixedmut_null(src, tgt, w, n=n, seed=8)
        assert not np.array_equal(t2, t4), "different seeds gave same draw"
        print(f"OK {name:10s} dyad_fixedmut    mutual_slots={stats['n_mutual_slots']:,} "
              f"({100.0 * stats['n_mutual_slots'] / src.size:.2f}% of m, "
              f"{100.0 * stats['mutual_weight'] / w.sum():.2f}% of W) "
              f"bad_initial={stats['n_bad_initial']:,} "
              f"repair_rounds={stats['repair_rounds']} "
              f"dup_slots={stats['n_duplicate_slots']:,}")
    print("selftest passed")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="validate both v2 generators on mouse + connectome")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
