"""Prototype gate for H60 — condense the NET digraph, not the RAW digraph.

The premise (verified, not claimed)
-----------------------------------
``SccRecursiveRefiner._refine`` (``src/mfas/refine/scc_recursive.py:222``) builds its
condensation matrix as

    mat = coo_matrix((np.ones(eidx.size, dtype=np.int8), (ls_pos, lt_pos)), ...)

i.e. over the RAW arc set, every edge counting the same. So a reciprocal pair
``u <-> v`` is an unbreakable 2-cycle: ``u`` and ``v`` land in the same SCC and the
refiner then preserves their relative order by construction (:248) — even when
``w_uv = 100`` and ``w_vu = 1``.

The reduction
-------------
For an unordered pair ``{u, v}`` the ordering contributes ``w_uv`` if ``u`` precedes
``v`` and ``w_vu`` otherwise. Writing that as

    contribution = min(w_uv, w_vu) + [the larger direction is forward] * |w_uv - w_vu|

splits the objective into an ORDER-INDEPENDENT constant

    C = sum over unordered pairs of min(w_uv, w_vu)

plus the forward weight of the **net digraph**

    D+ = {(u, v) : w_uv > w_vu},  d_uv = w_uv - w_vu.

Maximising raw feedforward weight is therefore *identical* to maximising feedforward
weight on ``D+``, up to ``C``. This is the Extended Condorcet Criterion's object: the
finest partition every optimum respects is the SCC condensation of the NET (majority)
digraph, not of the raw one (arXiv:2506.15097).

Why the substitution is still exactly monotone
----------------------------------------------
Run the refiner with ``D+`` as its structure graph and the contiguous-block lemma is
untouched — blocks are still contiguous position ranges, so no edge with an endpoint
outside a block can flip. Inside a block: every inter-component ``D+`` arc is laid
forward (gain ``+d``, and it is a *net* gain because the reciprocal half is inside the
constant ``C``), and every intra-component pair keeps its relative order (net change
0). So net-forward weight is non-decreasing, hence RAW forward weight is
non-decreasing. The script asserts this empirically every round against the frozen
scorer rather than trusting the argument.

``D+``'s arcs are a subset of ``G``'s, so ``D+``'s SCCs **strictly refine** ``G``'s:
the substitution can only expose decompositions, never hide one.

What is measured, and what is NOT
---------------------------------
M11 (``killed.json``) forbids justifying a move class by a capacity statistic. So the
gate here is the REALISED delta of an actual refiner pass from the champion's own
stored order, scored by the frozen oracle — not the weight sitting in reciprocal
pairs. ``C`` is reported for diagnosis and is explicitly labelled a constant, i.e. it
is NOT reclaimable by anything.

Three arms, all from the same champion start:

* ``raw``  — the champion's own structure graph. This is the CONTROL, and it should
  return ~0: the stored order is already a fixed point of it.
* ``net``  — ``D+`` substituted, nothing else changed. This is H60.
* ``both`` — net pass then raw pass each round, to show whether the two
  decompositions are complementary or the same ground.

Reproduce
---------
    PYTHONPATH=src python experiments/proto_H60_net_condensation.py --dataset connectome
    PYTHONPATH=src python experiments/proto_H60_net_condensation.py --dataset microns
    PYTHONPATH=src python experiments/proto_H60_net_condensation.py --dataset mouse
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mfas.io import load_dataset
from mfas.metrics import pct, score_from_positions
from mfas.refine.scc_recursive import DEFAULT_SPLIT_FRACS, scc_recursive_refine

# The champion order for each dataset, taken EXPLICITLY rather than by a name glob
# (H46's prototype picked its vector by glob and silently measured a killed run).
CHAMPION = {
    "connectome": {
        "variant": "H42",
        "positions": "experiments/evidence/"
                     "20260810T105922Z-H42-connectome-s31415-confirm-f41d7e_positions.npy",
        "expect_pct": 84.15409511,
    },
    "microns": {
        "variant": "H42",
        "positions": "experiments/evidence/"
                     "20260810T121650Z-H42-microns-s31415-confirm-f91b66_positions.npy",
        "expect_pct": 83.2409,
    },
    "mouse": {
        "variant": "H52",
        "positions": "experiments/evidence/"
                     "20260816T211722Z-H52-mouse-s1234-confirm-b458f1_positions.npy",
        "expect_pct": 93.1028,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# The net digraph
# ──────────────────────────────────────────────────────────────────────────────
def build_net_digraph(src: np.ndarray, tgt: np.ndarray, weight: np.ndarray,
                      n: int) -> dict:
    """Aggregate multi-edges, then reduce every reciprocal pair to its net arc.

    Returns a dict with the net arc arrays ``nsrc``/``ntgt``/``nw`` (``nw > 0``), the
    order-independent constant ``C = sum over pairs min(w_uv, w_vu)``, and pair counts.

    Self-loops are dropped: they are never feedforward under any ordering and the
    refiner drops them too.
    """
    keep = src != tgt
    s = np.asarray(src, dtype=np.int64)[keep]
    t = np.asarray(tgt, dtype=np.int64)[keep]
    w = np.asarray(weight)[keep]

    # Aggregate parallel edges into one weight per ORDERED pair.
    okey = s * np.int64(n) + t
    uo, inv = np.unique(okey, return_inverse=True)
    # bincount, not np.add.at: the latter takes the unbuffered path and costs minutes
    # at 5.7 M edges. float64 is exact here — the largest total is 4.2e7 << 2^53 — and
    # nothing from this aggregation ever reaches the frozen scorer, which re-reads the
    # original integer weights.
    ow = np.bincount(inv, weights=w.astype(np.float64), minlength=uo.size)
    ou = uo // np.int64(n)
    ov = uo % np.int64(n)

    # Canonical unordered-pair key, and which direction each ordered pair is.
    lo = np.minimum(ou, ov)
    hi = np.maximum(ou, ov)
    pkey = lo * np.int64(n) + hi
    forward = ou == lo                       # True => this is the (lo -> hi) direction

    up, pinv = np.unique(pkey, return_inverse=True)
    w_lo_hi = np.bincount(pinv[forward], weights=ow[forward], minlength=up.size)
    w_hi_lo = np.bincount(pinv[~forward], weights=ow[~forward], minlength=up.size)

    d = w_lo_hi - w_hi_lo
    const_c = float(np.minimum(w_lo_hi, w_hi_lo).sum())

    plo = up // np.int64(n)
    phi = up % np.int64(n)
    pos_d = d > 0
    neg_d = d < 0
    nsrc = np.concatenate([plo[pos_d], phi[neg_d]]).astype(np.int64)
    ntgt = np.concatenate([phi[pos_d], plo[neg_d]]).astype(np.int64)
    nw = np.concatenate([d[pos_d], -d[neg_d]]).astype(np.float64)

    reciprocal = (w_lo_hi > 0) & (w_hi_lo > 0)
    return {
        "nsrc": nsrc, "ntgt": ntgt, "nw": nw,
        "const_c": const_c,
        "n_ordered_pairs": int(uo.size),
        "n_unordered_pairs": int(up.size),
        "n_reciprocal_pairs": int(reciprocal.sum()),
        "n_tied_pairs": int((d == 0).sum()),
        "weight_in_reciprocal_pairs": float(
            (w_lo_hi[reciprocal] + w_hi_lo[reciprocal]).sum()),
        "n_net_arcs": int(nsrc.size),
    }


def top_level_scc(src: np.ndarray, tgt: np.ndarray, n: int) -> dict:
    """Component count and giant-component share of one whole-graph condensation."""
    mat = coo_matrix((np.ones(src.size, dtype=np.int8), (src, tgt)), shape=(n, n))
    n_lab, labels = connected_components(mat, directed=True, connection="strong")
    sizes = np.bincount(labels, minlength=n_lab)
    return {
        "n_scc": int(n_lab),
        "giant_size": int(sizes.max()),
        "giant_frac": float(sizes.max()) / float(n),
        "n_singletons": int((sizes == 1).sum()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Arms
# ──────────────────────────────────────────────────────────────────────────────
def _shim(src: np.ndarray, tgt: np.ndarray, n: int) -> SimpleNamespace:
    """A minimal stand-in for GraphData — ``scc_recursive_refine`` reads only these."""
    return SimpleNamespace(src=src, tgt=tgt, n_nodes=n)


def run_arm(arm: str, g, net, rank0: np.ndarray, rounds: int, total: float,
            base: float) -> dict:
    """Run ``rounds`` refiner rounds of one arm, scoring every round on the ORACLE."""
    n = g.n_nodes
    g_raw = _shim(np.asarray(g.src, dtype=np.int64),
                  np.asarray(g.tgt, dtype=np.int64), n)
    g_net = _shim(net["nsrc"], net["ntgt"], n)

    rank = rank0.copy()
    prev = base
    hist = []
    t0 = time.time()
    for r in range(rounds):
        sf = DEFAULT_SPLIT_FRACS[r % len(DEFAULT_SPLIT_FRACS)]
        if arm in ("net", "both"):
            rank = scc_recursive_refine(g_net, rank, split_frac=sf)
        if arm in ("raw", "both"):
            rank = scc_recursive_refine(g_raw, rank, split_frac=sf)
        sc = score_from_positions(rank.astype(np.float64), g.src, g.tgt, g.weight)
        # The monotonicity argument in the module docstring, tested rather than assumed.
        assert sc >= prev - 1e-9, (
            f"arm {arm} round {r}: score DECREASED {prev} -> {sc}; the monotonicity "
            f"argument for this structure graph is false")
        hist.append({
            "round": r, "split_frac": sf,
            "pct": pct(sc, total),
            "delta_pp": pct(sc, total) - pct(base, total),
            "cum_wall_s": time.time() - t0,
        })
        prev = sc
    return {
        "arm": arm,
        "rounds": rounds,
        "final_pct": pct(prev, total),
        "delta_pp_round1": hist[0]["delta_pp"] if hist else 0.0,
        "delta_pp_final": hist[-1]["delta_pp"] if hist else 0.0,
        "wall_s": time.time() - t0,
        "history": hist,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=sorted(CHAMPION))
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = args.dataset
    spec = CHAMPION[ds]
    t_start = time.time()

    g = load_dataset(ds)
    total = float(g.total_weight)
    pos = np.load(spec["positions"])
    base = score_from_positions(pos, g.src, g.tgt, g.weight)
    base_pct = pct(base, total)
    assert abs(base_pct - spec["expect_pct"]) < 1e-3, (
        f"stored order does not reproduce the champion: {base_pct} vs {spec['expect_pct']}")

    rank0 = np.argsort(np.argsort(pos, kind="stable"), kind="stable").astype(np.int64)
    assert score_from_positions(rank0.astype(np.float64), g.src, g.tgt,
                                g.weight) == base, "rank conversion changed the score"

    t_net = time.time()
    net = build_net_digraph(g.src, g.tgt, g.weight, g.n_nodes)
    t_net = time.time() - t_net

    scc_raw = top_level_scc(np.asarray(g.src, dtype=np.int64)[np.asarray(g.src) != np.asarray(g.tgt)],
                            np.asarray(g.tgt, dtype=np.int64)[np.asarray(g.src) != np.asarray(g.tgt)],
                            g.n_nodes)
    scc_net = top_level_scc(net["nsrc"], net["ntgt"], g.n_nodes)

    arms = {}
    for arm in ("raw", "net", "both"):
        print(f"[{ds}] arm {arm} ...", flush=True)
        arms[arm] = run_arm(arm, g, net, rank0, args.rounds, total, base)
        print(f"[{ds}] arm {arm}: round1 {arms[arm]['delta_pp_round1']:+.6f} pp, "
              f"final {arms[arm]['delta_pp_final']:+.6f} pp, "
              f"{arms[arm]['wall_s']:.1f} s", flush=True)

    out = {
        "hypothesis": "H60",
        "dataset": ds,
        "champion": spec["variant"],
        "champion_positions": spec["positions"],
        "base_pct": base_pct,
        "total_weight": total,
        "rounds": args.rounds,
        "net_build_s": t_net,
        "net_digraph": {k: v for k, v in net.items()
                        if not isinstance(v, np.ndarray)},
        "constant_c_pp": 100.0 * net["const_c"] / total,
        "weight_in_reciprocal_pairs_pp": 100.0 * net["weight_in_reciprocal_pairs"] / total,
        "scc_raw_toplevel": scc_raw,
        "scc_net_toplevel": scc_net,
        "arms": arms,
        "incremental_net_over_raw_pp": (arms["net"]["delta_pp_final"]
                                        - arms["raw"]["delta_pp_final"]),
        "wall_s_total": time.time() - t_start,
    }
    dest = Path(args.out or f"experiments/outputs/proto_H60_{ds}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("dataset", "base_pct", "constant_c_pp",
                       "scc_raw_toplevel", "scc_net_toplevel",
                       "incremental_net_over_raw_pp")}, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
