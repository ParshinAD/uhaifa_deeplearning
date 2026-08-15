"""H41 prototype gate - do exact-gain bounded-span SEGMENT moves buy anything?

Hypothesis (queue item H41)
---------------------------
The champion's two move classes are blind to the same failure. Stage 3
(:func:`mfas.refine.underrelax.sift_underrelaxed`) moves ONE node at a time and cannot
make a move that is only profitable jointly. Stage 4
(:func:`mfas.refine.scc_recursive.scc_recursive_refine`) relocates a group only when
that group is SCC-separable. ``experiments/diagnosis.md`` reports Kendall-tau 0.610 and
a median rank-distance of ~22,580 against the reference - the signature of whole REGIONS
sitting in the wrong place. :mod:`mfas.refine.segment` adds the missing class: relocate a
contiguous run of positions as a rigid unit, with an exact closed-form gain.

The control that matters
------------------------
This campaign's standing failure mode is "a gain that is really extra compute" (see
``experiments/proto_h36_control.py``). The segment sweep adds 0 gradient steps but real
CPU, so the only honest comparison is **matched wall-clock from the same starting order**:

    arm A  champion     = alternate_scc_sift            (block + short sift)
    arm B  +segment     = alternate_scc_sift_segment    (block + short sift + segment)
    arm C  segment only = segment_refine                (the new class alone)

all three given the SAME ``--budget`` seconds and the SAME ``rank0``. Arm A is not given
a fixed cycle count: its ``n_cycles`` is set far above what the budget allows, so the
budget - not the configuration - is what stops it. That way arm B cannot win merely by
running longer. Re-running at a much larger budget is the convergence check: if both arms
return the same numbers, the comparison is between two FIXED POINTS, not two truncations.

Credit attribution
------------------
``alternate_scc_sift_segment`` logs the exact score after each of its three stages, so
the per-cycle increments ``d_scc_pp / d_sift_pp / d_seg_pp`` sum to the cycle's total
increment with no double counting. The report gives each move class's cumulative pp.

Datasets
--------
``mouse`` (n=148, seconds) and ``gap.make_hard_synthetic_graph`` - the fixture built to
carry a verified Rocket-to-reference gap, and therefore the honest place for this move
class to show signal. CONNECTOME AND MICRONS ARE DELIBERATELY NOT RUN HERE.

Note on the hard synthetic's size. At ``n=400`` the default ladder reaches a window of
256 positions, i.e. 64% of the whole line, so that instance does NOT exercise the
*bounded*-span regime at all. ``hard6000`` (full 11-rung ladder, max window 2048 = 34% of
the line) is closer to the connectome's ratio (2048 / 136,648 = 1.5%), and no cheap proxy
reaches that ratio. This is the prototype's main external-validity limit.

Run
---
    PY=/c/ProgramData/anaconda3/envs/allen/python.exe
    PYTHONPATH=src $PY experiments/proto_H41.py --dataset mouse    --budget 10
    PYTHONPATH=src $PY experiments/proto_H41.py --dataset hard400  --budget 12
    PYTHONPATH=src $PY experiments/proto_H41.py --dataset hard6000 --budget 20 --no-reference
    PYTHONPATH=src $PY experiments/proto_H41.py --timing-probe 136648
    # -> experiments/outputs/proto_H41.json  (written incrementally, one arm at a time)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfas.io import GraphData, load_dataset                      # noqa: E402
from mfas.metrics import pct, score_from_order                   # noqa: E402
from mfas.refine.scc_recursive import alternate_scc_sift         # noqa: E402
from mfas.refine.segment import (                                # noqa: E402
    DEFAULT_OFFSETS,
    DEFAULT_SEG_LENGTHS,
    alternate_scc_sift_segment,
    segment_refine,
    segment_sweep,
)
from mfas.refine.underrelax import sift_underrelaxed             # noqa: E402

OUT = ROOT / "experiments" / "outputs" / "proto_H41.json"

# Stage-3 constants, copied from src/mfas/experiments/H42.py so rank0 is the order the
# champion actually hands to stage 4.
K_FULL = 6
ALPHA = 0.7
STAGE3_SWEEPS = 40
# Stage-4 constants, copied from H42 (the shipped allocation).
ALT_SIFT_SWEEPS = 2
ALT_K_FULL = 2
MIN_BLOCK = 32
# Far above anything the budget allows, so the WALL-CLOCK is what stops every arm.
BIG = 100_000


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------
def cache_dir() -> Path:
    """Where built synthetic fixtures are cached (OUTSIDE the repo)."""
    d = Path(os.environ.get("MFAS_PROTO_CACHE",
                            Path(tempfile.gettempdir()) / "mfas_proto_H41"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_fixture(name: str, reference: bool = True):
    """Return ``(GraphData, reference_pct or None)``.

    ``hard<N>`` builds :func:`mfas.analysis.gap.make_hard_synthetic_graph` at ``n=N``.
    Its reference order is expensive (a 6-run Rocket ensemble + an oracle-guided sift),
    so the built graph and its reference % are cached OUTSIDE the repo, keyed by ``n``.
    The reference is DIAGNOSTIC context only - no arm ever reads it.

    With ``reference=False`` the generator's ``_build_reference`` call is stubbed out.
    That leaves the GRAPH bit-identical: the generator finishes ``g`` completely and only
    then calls ``_build_reference(g)``, which reads ``g`` and mutates neither it nor the
    generator's RNG. Used for the large instances, where the Rocket ensemble costs
    minutes and the reference is not what the arms compare against.
    """
    if name == "mouse":
        return load_dataset("mouse"), None
    if not name.startswith("hard"):
        raise ValueError(f"unknown fixture {name!r} (use 'mouse' or 'hard<N>')")
    n = int(name[4:])
    cache = cache_dir() / f"hard_synth_n{n}.npz"
    if cache.exists():
        d = np.load(cache)
        g = GraphData(src=d["src"], tgt=d["tgt"], weight=d["weight"],
                      node_ids=d["node_ids"], name=f"hard_synthetic_n{n}")
        ref = float(d["ref_pct"])
        return g, (ref if ref == ref and ref > 0 else None)

    from mfas.analysis import gap as gap_mod
    if reference:
        g, _ref_order, ref_pct = gap_mod.make_hard_synthetic_graph(n=n, seed=0)
    else:
        real = gap_mod._build_reference
        gap_mod._build_reference = (
            lambda gg, **kw: (np.arange(gg.n_nodes, dtype=np.int64), 0.0))
        try:
            g, _ref_order, _ = gap_mod.make_hard_synthetic_graph(n=n, seed=0)
        finally:
            gap_mod._build_reference = real
        ref_pct = float("nan")
    np.savez(cache, src=g.src, tgt=g.tgt, weight=g.weight, node_ids=g.node_ids,
             ref_pct=np.array(ref_pct))
    out = GraphData(src=g.src, tgt=g.tgt, weight=g.weight, node_ids=g.node_ids,
                    name=f"hard_synthetic_n{n}")
    return out, (float(ref_pct) if reference else None)


def ladder(n: int):
    """Geometric ladder truncated to the graph: no rung may exceed ``n // 2``."""
    return tuple(v for v in DEFAULT_SEG_LENGTHS if v <= max(1, n // 2))


def stage123(g: GraphData):
    """The champion's stages 1-3 WITHOUT the GPU: greedy-FAS warm start + stage-3 sift.

    Rocket (stage 2) is skipped on purpose - a GPU measurement is in flight, and the
    comparison here is between STAGE-4 move classes, all of which start from the same
    ``rank0``. Skipping stage 2 shifts the starting point for every arm identically.
    """
    from mfas.experiments.H02 import greedy_fas_order
    t0 = time.time()
    rank_greedy = np.asarray(greedy_fas_order(g), dtype=np.int64)
    greedy_score = score_from_order(rank_greedy, np.asarray(g.src), np.asarray(g.tgt),
                                    g.weight)
    rank0, s0, log = sift_underrelaxed(g, rank_greedy, k_full=K_FULL, alpha=ALPHA,
                                       max_sweeps=STAGE3_SWEEPS)
    return rank0, dict(greedy_pct=pct(greedy_score, g.total_weight),
                       stage3_pct=pct(s0, g.total_weight),
                       stage3_sweeps=len(log),
                       stage123_wall_s=time.time() - t0)


# ------------------------------------------------------------------------------
# Incremental result file
# ------------------------------------------------------------------------------
def write_arm(key: str, payload: dict) -> None:
    """Merge one arm's payload into the results JSON (written after EVERY arm)."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {}
    if OUT.exists():
        try:
            doc = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            doc = {}
    doc.setdefault("_meta", {})["written_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc[key] = payload
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"  -> wrote {key}")


# ------------------------------------------------------------------------------
# Arms
# ------------------------------------------------------------------------------
def run_dataset(name: str, budget: float, reference: bool = True) -> None:
    g, ref_pct = build_fixture(name, reference=reference)
    total = g.total_weight
    n = g.n_nodes
    lad = ladder(n)
    # The arm key carries the matched budget, so runs at different budgets accumulate in
    # the same file instead of overwriting each other.
    key = f"{name}@{budget:g}s"

    rank0, base = stage123(g)
    base_pct = base["stage3_pct"]
    print(f"[{key}] n={n:,} m={g.n_edges:,} ladder={lad}")
    print(f"[{key}] greedy {base['greedy_pct']:.6f} -> stage3 {base_pct:.6f} "
          f"({base['stage123_wall_s']:.1f} s)"
          + (f"   [diagnostic reference {ref_pct:.6f}]" if ref_pct else ""))
    write_arm(f"{key}/_setup", dict(
        dataset=name, n_nodes=n, n_edges=g.n_edges, total_weight=total,
        ladder=list(lad), budget_s=budget, reference_pct=ref_pct, **base))

    # -- arm A: champion stage 4 alone, wall-clock bound --------------------
    t0 = time.time()
    _rA, sA, logA = alternate_scc_sift(g, rank0, n_cycles=BIG,
                                       sift_sweeps=ALT_SIFT_SWEEPS, k_full=ALT_K_FULL,
                                       alpha=ALPHA, min_block=MIN_BLOCK,
                                       time_budget_s=budget)
    wA = time.time() - t0
    pA = pct(sA, total)
    print(f"[{key}] A champion        {pA:.6f}  ({pA - base_pct:+.6f} pp)  "
          f"{len(logA)} cycles  {wA:.1f} s")
    write_arm(f"{key}/A_champion", dict(
        arm="champion (alternate_scc_sift)", final_pct=pA, delta_pp=pA - base_pct,
        n_cycles=len(logA), wall_s=wA, budget_s=budget, log_tail=logA[-3:]))

    # -- arm B: champion stage 4 + the segment sweep, SAME wall-clock -------
    t0 = time.time()
    _rB, sB, logB = alternate_scc_sift_segment(g, rank0, n_cycles=BIG,
                                               sift_sweeps=ALT_SIFT_SWEEPS,
                                               k_full=ALT_K_FULL, alpha=ALPHA,
                                               min_block=MIN_BLOCK, seg_sweeps=1,
                                               seg_lengths=lad, offsets=lad,
                                               time_budget_s=budget)
    wB = time.time() - t0
    pB = pct(sB, total)
    d_scc = float(sum(r["d_scc_pp"] for r in logB))
    d_sift = float(sum(r["d_sift_pp"] for r in logB))
    d_seg = float(sum(r["d_seg_pp"] for r in logB))
    t_scc = float(sum(r["t_scc_s"] for r in logB))
    t_sift = float(sum(r["t_sift_s"] for r in logB))
    t_seg = float(sum(r["t_seg_s"] for r in logB))
    n_seg_moves = int(sum(r["seg_moves"] for r in logB))
    print(f"[{key}] B +segment        {pB:.6f}  ({pB - base_pct:+.6f} pp)  "
          f"{len(logB)} cycles  {wB:.1f} s")
    print(f"[{key}]   credit split    block {d_scc:+.6f} pp | sift {d_sift:+.6f} pp | "
          f"segment {d_seg:+.6f} pp   ({n_seg_moves} segment moves)")
    print(f"[{key}]   wall split      block {t_scc:.1f} s | sift {t_sift:.1f} s | "
          f"segment {t_seg:.1f} s")
    write_arm(f"{key}/B_champion_plus_segment", dict(
        arm="champion + segment (alternate_scc_sift_segment)", final_pct=pB,
        delta_pp=pB - base_pct, n_cycles=len(logB), wall_s=wB, budget_s=budget,
        credit_pp=dict(block=d_scc, sift=d_sift, segment=d_seg),
        wall_split_s=dict(block=t_scc, sift=t_sift, segment=t_seg),
        n_segment_moves=n_seg_moves, log_tail=logB[-3:]))

    # -- arm C: the new move class ALONE, SAME wall-clock -------------------
    t0 = time.time()
    _rC, sC, logC = segment_refine(g, rank0, max_sweeps=BIG, seg_lengths=lad,
                                   offsets=lad, time_budget_s=budget)
    wC = time.time() - t0
    pC = pct(sC, total)
    print(f"[{key}] C segment only    {pC:.6f}  ({pC - base_pct:+.6f} pp)  "
          f"{len(logC)} sweeps  {wC:.1f} s")
    write_arm(f"{key}/C_segment_only", dict(
        arm="segment only (segment_refine)", final_pct=pC, delta_pp=pC - base_pct,
        n_sweeps=len(logC), n_moves=int(sum(r["n_moves"] for r in logC)),
        wall_s=wC, budget_s=budget, log_tail=logC[-3:]))

    # -- verdict row --------------------------------------------------------
    print(f"[{key}] VERDICT  B - A = {pB - pA:+.6f} pp at matched wall-clock "
          f"({wA:.1f} s vs {wB:.1f} s)")
    write_arm(f"{key}/VERDICT", dict(
        base_stage3_pct=base_pct, A_champion_pct=pA, B_plus_segment_pct=pB,
        C_segment_only_pct=pC, B_minus_A_pp=pB - pA,
        wall_A_s=wA, wall_B_s=wB, wall_C_s=wC, segment_credit_pp=d_seg))


def timing_probe(n: int) -> None:
    """Cost of ONE full-grid segment sweep at connectome-ish scale (CPU only).

    Connectome and MICrONS must NOT be touched while a GPU measurement is in flight, so
    the per-sweep cost at that scale is ESTIMATED on a synthetic of the same order of
    magnitude. Two starting orders bracket the answer, because the sweep's cost is driven
    by how many edges are SHORT-RANGE (only edges spanning < L+d positions contribute):

    * ``shuffled`` - a uniformly random order. Edge spans are uniform on [1, n), so only
      ~2048/n of the edges are ever in range: the CHEAP end.
    * ``identity`` - the generator's block order, where the 55% intra-block edges are
      local: the EXPENSIVE end, and the one a refined order resembles.

    This is an ESTIMATE, explicitly labelled as such in the JSON; it is not a connectome
    measurement and must not be quoted as one.
    """
    ladders = {
        "default_2^10": DEFAULT_SEG_LENGTHS,
        "extended_2^16": tuple(2 ** k for k in range(17)),
    }
    g, _ = build_fixture(f"hard{n}", reference=False)
    out = dict(n_nodes=g.n_nodes, n_edges=g.n_edges,
               ladders={k: list(v) for k, v in ladders.items()},
               caveat="ESTIMATE on a synthetic of the same order of magnitude; "
                      "NOT a connectome/microns measurement")
    for lname, lad in ladders.items():
        for label, rank in (("identity", np.arange(g.n_nodes, dtype=np.int64)),
                            ("shuffled", np.random.RandomState(0)
                             .permutation(g.n_nodes).astype(np.int64))):
            t0 = time.time()
            _new, info = segment_sweep(g, rank, seg_lengths=lad, offsets=lad)
            dt = time.time() - t0
            out[f"{lname}/{label}"] = dict(sweep_wall_s=dt, n_moves=info["n_moves"],
                                           n_candidates=info["n_candidates"],
                                           grid_points=len(lad) ** 2)
            print(f"[timing] n={g.n_nodes:,} m={g.n_edges:,} ladder={lname} "
                  f"order={label}: one full-grid sweep {dt:.2f} s, "
                  f"{info['n_moves']} moves, {info['n_candidates']} candidates")
    write_arm(f"_timing_probe/n{g.n_nodes}", out)


def main() -> None:
    ap = argparse.ArgumentParser(description="H41 prototype gate (cheap proxies only)")
    ap.add_argument("--dataset", default="mouse",
                    help="mouse | hard<N> (e.g. hard400). NEVER connectome/microns.")
    ap.add_argument("--budget", type=float, default=15.0,
                    help="matched wall-clock seconds given to EVERY arm")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the expensive diagnostic reference build (large hard<N>)")
    ap.add_argument("--timing-probe", type=int, default=0,
                    help="instead of the arms, time one full-grid sweep at this n")
    a = ap.parse_args()
    if a.dataset in ("connectome", "microns"):
        raise SystemExit("refusing: this prototype is for cheap proxies only")
    if a.timing_probe:
        timing_probe(a.timing_probe)
        return
    run_dataset(a.dataset, a.budget, reference=not a.no_reference)


if __name__ == "__main__":
    main()
