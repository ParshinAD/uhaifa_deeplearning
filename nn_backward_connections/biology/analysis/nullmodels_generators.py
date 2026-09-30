"""Null-model generators for Track C question G01 (real brains vs random graphs).

Self-contained (NumPy only), seedable, vectorized. Nothing here touches
``src/mfas/randomgraph/`` (that skeleton stays untouched by rule); these
generators live under ``biology/`` and are validated by ``--selftest``.

The study is a NESTED CONSTRAINT LADDER — each rung adds a restriction, so the
delta between adjacent rungs isolates one structural ingredient of the brain's
feedforward order:

  rung  generator                     matched exactly (microcanonical)
  L1    gnm_null                      n, m, weight multiset ("density only")
  L2    directed_configuration_null   + every node's in/out degree
                                        (w_source additionally: out-strength)
  L3a   sbm_null (non-nested ref.)    region-pair edge counts + weights,
                                        NOT degrees (SBM proper)
  L3    block_configuration_null      degrees AND each node's region-resolved
                                        in/out profile (+ per-region
                                        out-strength) — microcanonical
                                        degree-corrected SBM
  top   the real graph

Null family: DIRECTED CONFIGURATION MODEL by target-column permutation (L2)
===========================================================================
Given the edge list ``(src[i], tgt[i], w[i])`` we permute the TARGET column with
a uniform random permutation. This preserves, exactly:

* every node's out-degree (the src column is untouched),
* every node's in-degree as edge-endpoint counts (the tgt multiset is permuted,
  not changed),
* the multiset of edge weights (and hence the total weight).

Two WEIGHT variants (they support different claims):

* ``w_shuffled`` — weights are ADDITIONALLY shuffled across edge slots by an
  independent permutation. Preserves degrees + the weight multiset only; every
  degree–strength coupling (out-strength AND in-strength per node) is broken.
  Claim supported: "brains differ from graphs matched on degree sequence alone."
  NOTE: permuting only the target column while leaving ``w`` on its slot would
  trivially preserve out-strength (the slot keeps its source), so this variant
  shuffles weights explicitly to actually break the coupling.
* ``w_source`` — weights stay on their edge slot, i.e. they TRAVEL WITH THE
  SOURCE STUB. Preserves degrees + every node's exact OUT-STRENGTH (the weight
  a neuron sends is fixed; only where it lands is randomized). In-strength is
  randomized. Claim supported: "the difference is not explained by which
  neurons send how much weight" — the stricter, structure-matched null.

Self-loop / duplicate policy (both real graphs have ZERO of each — verified)
----------------------------------------------------------------------------
* Self-loops created by the permutation are REPAIRED by pairwise target swaps
  (swap the target of a loop slot with the target of a random partner slot,
  accepting only swaps that create no new loop). The target multiset is
  preserved exactly. Repair is necessary, not cosmetic: a self-loop can never
  be feedforward under any order, so leaving loops in would hand the null a
  guaranteed-feedback handicap the real graph does not have. The (small)
  distributional bias of swap-repair is standard for edge-swap null models and
  is reported per graph (``n_selfloops_repaired``).
* Duplicate (parallel) edges are LEFT IN PLACE as parallel slots. The oracle
  score, the greedy-FAS peel, the Rocket loss and the sift gains are all
  per-edge weighted sums, so parallel edges (u,v,w1),(u,v,w2) behave exactly
  like one edge (u,v,w1+w2) in every stage of the protocol; keeping them also
  preserves the weight multiset (collapsing would change ``max(w)`` and hence
  Rocket's weight normalization). The duplicate count is reported
  (``n_duplicate_slots``) as a property of the null, since the real graphs are
  simple.

Run the built-in validation (degree/strength/weight preservation, seed
reproducibility, loop-freedom) on both datasets:

    $PY biology/analysis/nullmodels_generators.py --selftest
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

VARIANTS = ("w_shuffled", "w_source")


def directed_configuration_null(
    src: np.ndarray,
    tgt: np.ndarray,
    weight: np.ndarray,
    *,
    variant: str,
    seed: int,
    max_repair_rounds: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Draw ONE directed-configuration-model null of the graph.

    Parameters
    ----------
    src, tgt : int arrays, shape (m,)
        Contiguous node indices of each edge's source / target.
    weight : array, shape (m,)
        Edge weights (int64 or float64; dtype is preserved).
    variant : {"w_shuffled", "w_source"}
        Weight handling — see module docstring.
    seed : int
        Seed for ``np.random.default_rng``; same seed => bit-identical output.
    max_repair_rounds : int
        Safety cap on the self-loop repair loop (never reached in practice).

    Returns
    -------
    (src_new, tgt_new, w_new, stats) where ``stats`` reports
    ``n_selfloops_initial``, ``n_selfloops_repaired``, ``n_duplicate_slots``,
    ``repair_rounds``.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    src = np.asarray(src)
    tgt = np.asarray(tgt)
    weight = np.asarray(weight)
    m = src.shape[0]
    rng = np.random.default_rng(seed)

    new_tgt = tgt[rng.permutation(m)].copy()

    # ── Self-loop repair by pairwise target swaps (target multiset preserved) ──
    n_initial = int((src == new_tgt).sum())
    rounds = 0
    loops = np.flatnonzero(src == new_tgt)
    while loops.size > 0 and rounds < max_repair_rounds:
        rounds += 1
        partners = rng.integers(0, m, size=loops.size)
        for i, j in zip(loops, partners):
            # Swap tgt[i] <-> tgt[j] only if it resolves i's loop and makes none.
            if src[i] != new_tgt[j] and src[j] != new_tgt[i]:
                new_tgt[i], new_tgt[j] = new_tgt[j], new_tgt[i]
        loops = np.flatnonzero(src == new_tgt)
    if loops.size > 0:  # pragma: no cover - safety cap, never hit in practice
        raise RuntimeError(
            f"self-loop repair did not converge: {loops.size} loops left after "
            f"{max_repair_rounds} rounds")

    # ── Weight handling ──
    if variant == "w_shuffled":
        w_new = weight[rng.permutation(m)].copy()
    else:  # "w_source": weight stays with its slot => travels with the source stub
        w_new = weight.copy()

    # ── Duplicate-slot count (kept as parallel edges; reported, not collapsed) ──
    n = int(max(src.max(), new_tgt.max())) + 1
    pair = src.astype(np.int64) * n + new_tgt.astype(np.int64)
    n_dup = int(m - np.unique(pair).size)

    stats = dict(
        variant=variant, seed=int(seed),
        n_selfloops_initial=n_initial,
        n_selfloops_repaired=n_initial,   # all initial loops are repaired
        repair_rounds=rounds,
        n_duplicate_slots=n_dup,
    )
    return src.copy(), new_tgt, w_new, stats


# ──────────────────────────────────────────────────────────────────────────────
# L1 — uniform random directed graph, G(n,m) with multi-edges ("density only")
# ──────────────────────────────────────────────────────────────────────────────
def gnm_null(src: np.ndarray, tgt: np.ndarray, weight: np.ndarray, *,
             n: int, seed: int, max_repair_rounds: int = 1000
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """L1: redraw BOTH endpoints of every edge uniformly over the n nodes.

    Underlying distribution: directed G(n, m) with i.i.d. endpoint sampling
    (multi-edges allowed, kept as parallel slots per the module policy), each
    edge conditioned on not being a self-loop (loop slots have their TARGET
    redrawn until src != tgt, which is exactly uniform over the n-1 non-source
    nodes). Weights stay with their edge slot, so the weight multiset and total
    are preserved; degrees, strengths and all wiring structure are destroyed.
    Matches: n, m, weight multiset. FF(L1) − 50% = what density alone allows
    the (identical, epoch-budgeted) optimizer.
    """
    src = np.asarray(src)
    weight = np.asarray(weight)
    m = src.shape[0]
    rng = np.random.default_rng(seed)
    src2 = rng.integers(0, n, size=m).astype(src.dtype)
    tgt2 = rng.integers(0, n, size=m).astype(src.dtype)
    rounds = 0
    loops = np.flatnonzero(src2 == tgt2)
    n_initial = int(loops.size)
    while loops.size > 0 and rounds < max_repair_rounds:
        rounds += 1
        tgt2[loops] = rng.integers(0, n, size=loops.size).astype(src.dtype)
        loops = np.flatnonzero(src2 == tgt2)
    if loops.size > 0:  # pragma: no cover
        raise RuntimeError("G(n,m) self-loop redraw did not converge")
    pair = src2.astype(np.int64) * n + tgt2.astype(np.int64)
    n_dup = int(m - np.unique(pair).size)
    stats = dict(variant="L1_gnm", seed=int(seed),
                 n_selfloops_initial=n_initial, n_selfloops_repaired=n_initial,
                 repair_rounds=rounds, n_duplicate_slots=n_dup)
    return src2, tgt2, weight.copy(), stats


# ──────────────────────────────────────────────────────────────────────────────
# L3 — block-constrained configuration model (microcanonical degree-corrected SBM)
# ──────────────────────────────────────────────────────────────────────────────
def block_configuration_null(src: np.ndarray, tgt: np.ndarray,
                             weight: np.ndarray, labels: np.ndarray, *,
                             seed: int, max_repair_rounds: int = 1000
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """L3: permute the target column WITHIN each (region(src) × region(tgt)) block.

    Every edge keeps its source and its target REGION; only which node inside
    the target region receives it is randomized. Weights stay with the slot.
    Preserves exactly, per node: out-degree AND out-strength to every target
    region, and in-degree from every source region — i.e. the full
    region-resolved degree/strength profile — plus every region-pair edge count
    and weight sum. This is the microcanonical degree-corrected SBM null.
    Self-loops (possible only inside diagonal blocks) are repaired by pairwise
    target swaps WITHIN the same block, so all block guarantees survive repair.
    """
    src = np.asarray(src)
    tgt = np.asarray(tgt)
    weight = np.asarray(weight)
    labels = np.asarray(labels, dtype=np.int64)
    m = src.shape[0]
    K = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    block = labels[src] * K + labels[tgt]
    order = np.argsort(block, kind="stable")
    bsorted = block[order]
    # segment boundaries of each occupied block in `order`
    seg_starts = np.flatnonzero(np.concatenate([[True], bsorted[1:] != bsorted[:-1]]))
    seg_ends = np.concatenate([seg_starts[1:], [m]])

    new_tgt = tgt.copy()
    slot_of_block: Dict[int, np.ndarray] = {}
    for a, b in zip(seg_starts, seg_ends):
        idx = order[a:b]
        slot_of_block[int(bsorted[a])] = idx
        new_tgt[idx] = tgt[idx][rng.permutation(b - a)]

    # self-loop repair by swaps WITHIN the loop's own block
    n_initial = int((src == new_tgt).sum())
    rounds = 0
    loops = np.flatnonzero(src == new_tgt)
    while loops.size > 0 and rounds < max_repair_rounds:
        rounds += 1
        for i in loops:
            seg = slot_of_block[int(block[i])]
            j = int(seg[rng.integers(0, seg.size)])
            if src[i] != new_tgt[j] and src[j] != new_tgt[i]:
                new_tgt[i], new_tgt[j] = new_tgt[j], new_tgt[i]
        loops = np.flatnonzero(src == new_tgt)
    if loops.size > 0:  # pragma: no cover
        raise RuntimeError("block-config self-loop repair did not converge")

    n = int(max(src.max(), new_tgt.max())) + 1
    pair = src.astype(np.int64) * n + new_tgt.astype(np.int64)
    n_dup = int(m - np.unique(pair).size)
    stats = dict(variant="L3_dcsbm", seed=int(seed), n_blocks=len(slot_of_block),
                 n_selfloops_initial=n_initial, n_selfloops_repaired=n_initial,
                 repair_rounds=rounds, n_duplicate_slots=n_dup)
    return src.copy(), new_tgt, weight.copy(), stats


# ──────────────────────────────────────────────────────────────────────────────
# L3a — SBM proper (block counts only; non-nested reference point)
# ──────────────────────────────────────────────────────────────────────────────
def sbm_null(src: np.ndarray, tgt: np.ndarray, weight: np.ndarray,
             labels: np.ndarray, *, seed: int, max_repair_rounds: int = 1000
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """L3a: within each (region × region) block redraw BOTH endpoints uniformly.

    Each edge keeps its (source-region, target-region) block; its endpoints are
    redrawn i.i.d. uniformly from the two regions' node sets (self-loops:
    target redrawn until src != tgt). Weights stay with the slot. Preserves
    every region-pair edge count and weight sum (the mesoscale wiring diagram)
    but NOT the degree sequence — the stochastic block model proper, kept as a
    NON-NESTED reference beside L3 (which additionally preserves degrees).
    """
    src = np.asarray(src)
    tgt = np.asarray(tgt)
    weight = np.asarray(weight)
    labels = np.asarray(labels, dtype=np.int64)
    m = src.shape[0]
    K = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    nodes_by_label = [np.flatnonzero(labels == k) for k in range(K)]
    src2 = np.empty_like(src)
    tgt2 = np.empty_like(tgt)
    block = labels[src] * K + labels[tgt]
    order = np.argsort(block, kind="stable")
    bsorted = block[order]
    seg_starts = np.flatnonzero(np.concatenate([[True], bsorted[1:] != bsorted[:-1]]))
    seg_ends = np.concatenate([seg_starts[1:], [m]])
    for a, b in zip(seg_starts, seg_ends):
        idx = order[a:b]
        r, s = divmod(int(bsorted[a]), K)
        src2[idx] = rng.choice(nodes_by_label[r], size=b - a, replace=True)
        tgt2[idx] = rng.choice(nodes_by_label[s], size=b - a, replace=True)

    rounds = 0
    loops = np.flatnonzero(src2 == tgt2)
    n_initial = int(loops.size)
    while loops.size > 0 and rounds < max_repair_rounds:
        rounds += 1
        lab = labels[src2[loops]]
        for i, k in zip(loops, lab):
            tgt2[i] = nodes_by_label[k][rng.integers(0, nodes_by_label[k].size)]
        loops = np.flatnonzero(src2 == tgt2)
    if loops.size > 0:  # pragma: no cover
        raise RuntimeError("SBM self-loop redraw did not converge")

    n = int(max(src2.max(), tgt2.max())) + 1
    pair = src2.astype(np.int64) * n + tgt2.astype(np.int64)
    n_dup = int(m - np.unique(pair).size)
    stats = dict(variant="L3a_sbm", seed=int(seed),
                 n_selfloops_initial=n_initial, n_selfloops_repaired=n_initial,
                 repair_rounds=rounds, n_duplicate_slots=n_dup)
    return src2, tgt2, weight.copy(), stats


# ──────────────────────────────────────────────────────────────────────────────
# Fly region labels = the 17 biological lanes of the layering study
# ──────────────────────────────────────────────────────────────────────────────
def fly_lane_labels() -> Tuple[np.ndarray, list]:
    """Per-node lane labels for the fly, via ``layering.tools.fly_streams.lane_of``.

    Reuses the EXACT lane definition of the layering study (super_class refined
    by cell_class; every node covered, the 13 unannotated ones fall into
    'central (other)'). The lane list here is sorted alphabetically for
    determinism — only the PARTITION matters for the block nulls, not the lane
    order. Returns ``(labels int8[n], lane_list)``.
    """
    import json
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "layering" / "tools"))
    from fly_streams import lane_of  # noqa: E402  (constants-only import)

    d = np.load(repo / "layering" / "outputs" / "fly_ann" / "node_ann.npz")
    meta = json.loads((repo / "layering" / "outputs" / "fly_ann"
                       / "node_ann_meta.json").read_text())
    scv = meta["super_class_values"]
    ccv = meta["cell_class_values"]
    names = [lane_of(scv[s], ccv[c])
             for s, c in zip(d["super_class"], d["cell_class"])]
    lane_list = sorted(set(names))
    idx = {L: i for i, L in enumerate(lane_list)}
    labels = np.array([idx[nm] for nm in names], dtype=np.int8)
    return labels, lane_list


# ──────────────────────────────────────────────────────────────────────────────
# Self-test: exact preservation properties + seed reproducibility
# ──────────────────────────────────────────────────────────────────────────────
def _validate_null(src, tgt, w, src2, tgt2, w2, variant: str, n: int) -> None:
    """Assert every preservation guarantee of the generator (exact, not approx)."""
    # out-degree exact (src untouched)
    assert np.array_equal(np.bincount(src, minlength=n),
                          np.bincount(src2, minlength=n)), "out-degree changed"
    # in-degree exact (tgt multiset permuted)
    assert np.array_equal(np.bincount(tgt, minlength=n),
                          np.bincount(tgt2, minlength=n)), "in-degree changed"
    # weight multiset + total exact
    assert np.array_equal(np.sort(w), np.sort(w2)), "weight multiset changed"
    assert w.sum() == w2.sum(), "total weight changed"
    # no self-loops
    assert int((src2 == tgt2).sum()) == 0, "self-loops remain"
    # variant-specific: exact per-node out-strength preservation
    if variant == "w_source":
        s_real = np.bincount(src, weights=np.asarray(w, dtype=np.float64),
                             minlength=n)
        s_null = np.bincount(src2, weights=np.asarray(w2, dtype=np.float64),
                             minlength=n)
        assert np.array_equal(s_real, s_null), "out-strength changed (w_source)"


def _validate_gnm(src, tgt, w, src2, tgt2, w2) -> None:
    """L1 guarantees: m, weight multiset/total, loop-freedom (degrees NOT kept)."""
    assert src2.shape == src.shape and tgt2.shape == tgt.shape
    assert np.array_equal(np.sort(w), np.sort(w2)) and w.sum() == w2.sum()
    assert int((src2 == tgt2).sum()) == 0, "self-loops remain (L1)"


def _validate_block(src, tgt, w, src2, tgt2, w2, labels, n, *, degrees: bool
                    ) -> None:
    """L3/L3a guarantees. ``degrees=True`` adds the L3 profile checks."""
    K = int(labels.max()) + 1
    lab = labels.astype(np.int64)
    blk = lab[src] * K + lab[tgt]
    blk2 = lab[src2] * K + lab[tgt2]
    wf = np.asarray(w, dtype=np.float64)
    # region-pair edge counts and weight sums exact (both L3 and L3a)
    assert np.array_equal(np.bincount(blk, minlength=K * K),
                          np.bincount(blk2, minlength=K * K)), "block counts changed"
    assert np.allclose(np.bincount(blk, weights=wf, minlength=K * K),
                       np.bincount(blk2, weights=np.asarray(w2, np.float64),
                                   minlength=K * K)), "block weights changed"
    assert int((src2 == tgt2).sum()) == 0, "self-loops remain (block null)"
    assert np.array_equal(np.sort(w), np.sort(w2)) and w.sum() == w2.sum()
    if degrees:
        # L3: full region-resolved degree + out-strength profile, per node
        s64, t64 = src.astype(np.int64), tgt.astype(np.int64)
        s64b, t64b = src2.astype(np.int64), tgt2.astype(np.int64)
        assert np.array_equal(np.bincount(s64 * K + lab[tgt], minlength=n * K),
                              np.bincount(s64b * K + lab[tgt2], minlength=n * K)), \
            "per-node out-profile changed"
        assert np.array_equal(np.bincount(t64 * K + lab[src], minlength=n * K),
                              np.bincount(t64b * K + lab[src2], minlength=n * K)), \
            "per-node in-profile changed"
        assert np.array_equal(
            np.bincount(s64 * K + lab[tgt], weights=wf, minlength=n * K),
            np.bincount(s64b * K + lab[tgt2], weights=np.asarray(w2, np.float64),
                        minlength=n * K)), "per-node region out-strength changed"


def _report(name, variant, m, stats):
    print(f"OK {name:10s} {variant:11s} loops_repaired="
          f"{stats['n_selfloops_repaired']:5d} "
          f"({100.0 * stats['n_selfloops_repaired'] / m:.4f}% of m) "
          f"dup_slots={stats['n_duplicate_slots']:6d} "
          f"({100.0 * stats['n_duplicate_slots'] / m:.4f}% of m) "
          f"repair_rounds={stats['repair_rounds']}")


def _selftest() -> None:
    """Validate every rung's generator on both datasets; print a compact report."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mfas.io import load_dataset

    for name in ("mouse", "connectome"):
        g = load_dataset(name)
        src = np.asarray(g.src); tgt = np.asarray(g.tgt); w = np.asarray(g.weight)
        # L2 — both weight variants
        for variant in VARIANTS:
            s2, t2, w2, stats = directed_configuration_null(
                src, tgt, w, variant=variant, seed=7)
            _validate_null(src, tgt, w, s2, t2, w2, variant, g.n_nodes)
            s3, t3, w3, _ = directed_configuration_null(
                src, tgt, w, variant=variant, seed=7)
            assert (np.array_equal(t2, t3) and np.array_equal(w2, w3)), \
                "same seed must reproduce bit-identically"
            _, t4, _, _ = directed_configuration_null(
                src, tgt, w, variant=variant, seed=8)
            assert not np.array_equal(t2, t4), "different seeds gave same draw"
            _report(name, variant, src.size, stats)
        # L1
        s2, t2, w2, stats = gnm_null(src, tgt, w, n=g.n_nodes, seed=7)
        _validate_gnm(src, tgt, w, s2, t2, w2)
        s3, t3, _, _ = gnm_null(src, tgt, w, n=g.n_nodes, seed=7)
        assert np.array_equal(s2, s3) and np.array_equal(t2, t3)
        _report(name, "L1_gnm", src.size, stats)
        # L3 / L3a — fly only (mouse nodes ARE regions; L3 degenerates)
        if name == "connectome":
            labels, lane_list = fly_lane_labels()
            assert labels.shape[0] == g.n_nodes
            print(f"   fly lanes: K={len(lane_list)}  sizes="
                  f"{np.bincount(labels).tolist()}")
            s2, t2, w2, stats = block_configuration_null(src, tgt, w, labels, seed=7)
            _validate_null(src, tgt, w, s2, t2, w2, "w_source", g.n_nodes)
            _validate_block(src, tgt, w, s2, t2, w2, labels, g.n_nodes, degrees=True)
            s3, t3, _, _ = block_configuration_null(src, tgt, w, labels, seed=7)
            assert np.array_equal(t2, t3)
            _report(name, "L3_dcsbm", src.size, stats)
            s2, t2, w2, stats = sbm_null(src, tgt, w, labels, seed=7)
            _validate_block(src, tgt, w, s2, t2, w2, labels, g.n_nodes, degrees=False)
            s3, t3, _, _ = sbm_null(src, tgt, w, labels, seed=7)
            assert np.array_equal(s2, s3) and np.array_equal(t2, t3)
            _report(name, "L3a_sbm", src.size, stats)
    print("selftest passed")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="validate generators on mouse + connectome")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
