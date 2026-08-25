"""Prototype gate for H60, part 2 — the COMPOSED measurement, not the standalone one.

Why part 1 is not enough
------------------------
``proto_H60_net_condensation.py`` measures the refiner ALONE from the champion's stored
order. H36's own history says that number is not dispositive: the one-shot top-level
condensation was measured at **+0.00013 pp** (``scc_recursive.py`` module docstring,
citing ``dr_tmp/FINDINGS_underrelaxation.md`` E2) while the same mechanism ALTERNATED
with the sift became the campaign's largest score move, **+0.1837 pp**. The two move
classes re-open each other's exhausted moves, so a standalone refiner delta systematically
under-states the composed one.

Meta-rule **M8** demands the same thing from the other direction: judge a move class only
by the composed champion-vs-variant delta at constant-for-constant matched settings, and
report the REDUNDANCY FRACTION.

What this measures
------------------
``alternate_scc_sift`` with the structure graph made a parameter, run from the champion's
own stored order, at matched cycle counts, with H42's production stage-4 constants:

* arm ``raw`` — structure graph ``G``. This is exactly what the champion's stage 4 does,
  so it is the CONTROL: it says how much is left in the champion's own move classes when
  you simply keep going.
* arm ``net`` — structure graph ``D+`` (the net digraph, built by part 1's
  ``build_net_digraph``). Nothing else differs: same sift, same constants, same start,
  same number of cycles.

The H60 delta is ``net - raw``, i.e. what substituting the structure graph buys ON TOP OF
running the champion's own stage 4 for the same number of cycles. Scoring is by the frozen
oracle on the ORIGINAL graph throughout; ``D+`` is used only to decide the decomposition.

Redundancy fraction (M8)
------------------------
Each cycle logs the score after the block refiner and after the sift, so the two classes'
own credit is separable:

    refiner credit  = sum over cycles of (after_scc  - previous)
    sift credit     = sum over cycles of (after_sift - after_scc)
    redundancy      = (sift credit in raw arm - sift credit in net arm) / (net arm's
                      refiner credit - raw arm's refiner credit)

i.e. how much of the extra ground the net refiner takes was already reachable by the sift.

Reproduce
---------
    PYTHONPATH=src python experiments/proto_H60_alternation.py --dataset connectome --cycles 40
    PYTHONPATH=src python experiments/proto_H60_alternation.py --dataset microns --cycles 10
    PYTHONPATH=src python experiments/proto_H60_alternation.py --dataset mouse --cycles 32
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from mfas.io import load_dataset
from mfas.metrics import pct, score_from_order, score_from_positions
from mfas.refine.scc_recursive import DEFAULT_SPLIT_FRACS, scc_recursive_refine
from mfas.refine.underrelax import sift_underrelaxed

from proto_H60_net_condensation import CHAMPION, build_net_digraph

# H42's production stage-4 constants (src/mfas/experiments/H42.py:105-110). Matched
# constant-for-constant across both arms — the ONLY difference is the structure graph.
_ALT_SIFT_SWEEPS = 2
_ALT_K_FULL = 2
_ALPHA = 0.7
_MIN_BLOCK = 32


def alternate_with_structure_graph(
        g, g_struct, init_rank: np.ndarray, *, n_cycles: int,
        sift_sweeps: int = _ALT_SIFT_SWEEPS, k_full: int = _ALT_K_FULL,
        alpha: float = _ALPHA, min_block: int = _MIN_BLOCK,
        split_fracs: Sequence[float] = DEFAULT_SPLIT_FRACS,
) -> Tuple[np.ndarray, float, List[Dict]]:
    """``alternate_scc_sift`` with the CONDENSATION graph made a parameter.

    A verbatim copy of :func:`mfas.refine.scc_recursive.alternate_scc_sift` except that
    the block refiner reads its structure from ``g_struct`` while the sift and every
    score read ``g``. Copied rather than imported-and-patched so the production module
    stays untouched during a prototype (it backs three sitting champions).

    Passing ``g_struct is g`` reproduces the production function exactly.
    """
    src_o, tgt_o = np.asarray(g.src), np.asarray(g.tgt)
    total = g.total_weight
    rank = np.asarray(init_rank, dtype=np.int64).copy()
    best_score = score_from_order(rank, src_o, tgt_o, g.weight)
    best_rank = rank.copy()
    prev = best_score

    log: List[Dict] = []
    t0 = time.time()
    for c in range(n_cycles):
        ta = time.time()
        rank = scc_recursive_refine(g_struct, rank, min_block=min_block,
                                    split_frac=split_fracs[c % len(split_fracs)])
        s_scc = score_from_order(rank, src_o, tgt_o, g.weight)
        t_scc = time.time() - ta
        # The refiner is monotone for BOTH structure graphs (see part 1's docstring);
        # tested, not assumed, because it is the load-bearing claim of H60.
        assert s_scc >= prev - 1e-9, (
            f"cycle {c}: block refiner DECREASED the score {prev} -> {s_scc}")
        if s_scc > best_score:
            best_score, best_rank = s_scc, rank.copy()

        tb = time.time()
        rank, s_sift, _ = sift_underrelaxed(g, rank, k_full=k_full, alpha=alpha,
                                            max_sweeps=sift_sweeps)
        t_sift = time.time() - tb
        if s_sift > best_score:
            best_score, best_rank = s_sift, rank.copy()

        log.append(dict(cycle=c, split_frac=split_fracs[c % len(split_fracs)],
                        before_pct=pct(prev, total),
                        after_scc_pct=pct(s_scc, total),
                        after_sift_pct=pct(s_sift, total),
                        best_pct=pct(best_score, total),
                        scc_credit_pp=pct(s_scc, total) - pct(prev, total),
                        sift_credit_pp=pct(s_sift, total) - pct(s_scc, total),
                        t_scc_s=t_scc, t_sift_s=t_sift,
                        cum_wall_s=time.time() - t0))
        prev = s_sift
    return best_rank, float(best_score), log


def summarise(log: List[Dict], base_pct: float, wall: float) -> Dict:
    """Per-arm totals, including each move class's own credit (M8)."""
    return {
        "cycles": len(log),
        "final_best_pct": log[-1]["best_pct"] if log else base_pct,
        "delta_pp": (log[-1]["best_pct"] - base_pct) if log else 0.0,
        "scc_credit_pp": float(sum(r["scc_credit_pp"] for r in log)),
        "sift_credit_pp": float(sum(r["sift_credit_pp"] for r in log)),
        "t_scc_s": float(sum(r["t_scc_s"] for r in log)),
        "t_sift_s": float(sum(r["t_sift_s"] for r in log)),
        "wall_s": wall,
        "pp_per_s": ((log[-1]["best_pct"] - base_pct) / wall) if log and wall > 0 else 0.0,
        "history": log,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=sorted(CHAMPION))
    ap.add_argument("--cycles", type=int, required=True)
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

    net = build_net_digraph(g.src, g.tgt, g.weight, g.n_nodes)
    g_net = SimpleNamespace(src=net["nsrc"], tgt=net["ntgt"], n_nodes=g.n_nodes)

    arms = {}
    for arm, gs in (("raw", g), ("net", g_net)):
        print(f"[{ds}] alternation arm {arm}, {args.cycles} cycles ...", flush=True)
        t0 = time.time()
        _, _, log = alternate_with_structure_graph(g, gs, rank0, n_cycles=args.cycles)
        arms[arm] = summarise(log, base_pct, time.time() - t0)
        print(f"[{ds}] arm {arm}: {arms[arm]['delta_pp']:+.6f} pp in "
              f"{arms[arm]['wall_s']:.1f} s  (scc credit {arms[arm]['scc_credit_pp']:+.6f}, "
              f"sift credit {arms[arm]['sift_credit_pp']:+.6f})", flush=True)

    d_scc = arms["net"]["scc_credit_pp"] - arms["raw"]["scc_credit_pp"]
    d_sift = arms["raw"]["sift_credit_pp"] - arms["net"]["sift_credit_pp"]
    out = {
        "hypothesis": "H60",
        "part": "composed alternation (M8)",
        "dataset": ds,
        "champion": spec["variant"],
        "champion_positions": spec["positions"],
        "base_pct": base_pct,
        "cycles": args.cycles,
        "constants": {"sift_sweeps": _ALT_SIFT_SWEEPS, "k_full": _ALT_K_FULL,
                      "alpha": _ALPHA, "min_block": _MIN_BLOCK,
                      "split_fracs": list(DEFAULT_SPLIT_FRACS)},
        "arms": arms,
        "h60_composed_delta_pp": arms["net"]["delta_pp"] - arms["raw"]["delta_pp"],
        "extra_scc_credit_pp": d_scc,
        "sift_credit_given_up_pp": d_sift,
        "redundancy_fraction": (d_sift / d_scc) if d_scc > 0 else None,
        "wall_s_total": time.time() - t_start,
    }
    dest = Path(args.out or f"experiments/outputs/proto_H60_alt_{ds}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("dataset", "base_pct", "h60_composed_delta_pp",
                       "extra_scc_credit_pp", "sift_credit_given_up_pp",
                       "redundancy_fraction")}, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
