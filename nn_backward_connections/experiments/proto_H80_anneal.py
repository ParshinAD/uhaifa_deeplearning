"""H80 rung 1 - barrier-crossing search on cheap proxies (CPU, no GPU).

Pre-registered in ``experiments/outputs/proto_H80_prereg.json``, which was committed
BEFORE this script ran. Read that file first: it fixes the arms, the primary statistic
and the four kill conditions, and it records the counter-evidence (H17, M13/H66, M15,
M16) that already stands against this item.

What this measures
------------------
A 2x2 factorial separating the two ingredients H80 conflates:

                       acceptance = greedy        acceptance = Metropolis
    destroy = top-k    A1 (H31 re-implemented)    A4
    destroy = struct   A2 (= queue item H40)      A3  <- H80

plus ``A0`` (sift fixed point, the monotone baseline) and ``A1p`` (the production
``mfas.refine.ils_lns``, a POSITIVE CONTROL whose job is to reproduce
``experiments/outputs/proto_h31_lns.json``; if it does not, this harness is broken and
nothing else here may be quoted).

Every arm starts from the SAME sift fixed point, uses the SAME exact-gain rebuild, and
gets the SAME refinement wall budget, so the arms differ only in destroy and acceptance.

Leakage
-------
Destroy choices and acceptance decisions read input edge weights + current ranks ONLY,
via :func:`_ff_weight`, a pure function of the input. The frozen oracle
(:func:`mfas.metrics.score_from_order`) is called only to score whole candidate vectors
for best-by-oracle tracking and reporting - exactly as ``ils_lns`` already does.
``data/best_solution`` is never read. The synthetic ``reference_order`` is read only by
:func:`barrier_depth`, a diagnostic, and never by any arm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402

torch.set_num_threads(2)

from mfas import io  # noqa: E402
from mfas.analysis import gap as gapmod  # noqa: E402
from mfas.baseline.rocket import RocketConfig, run_rocket  # noqa: E402
from mfas.experiments.H02 import _init_positions_from_order, greedy_fas_order  # noqa: E402
from mfas.metrics import pct, score_from_order  # noqa: E402
from mfas.refine import ils_lns, sift  # noqa: E402
from mfas.refine.insertion import build_sift_edges  # noqa: E402
from mfas.refine.lns import apply_victim_reinsertions, back_edge_weight  # noqa: E402

CPU = torch.device("cpu")
OUT = _ROOT / "experiments" / "outputs"

# Grid of T0 multipliers applied to the instance's OWN measured delta_med (pre-registered).
T0_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
COOL_RATIO = 0.01          # geometric cooling T0 -> T0 * COOL_RATIO over the budget
N_CAL = 24                 # calibration proposals used to measure delta_med
INNER_SWEEPS = 2           # short production sift after every rebuild (same for all arms)


# ──────────────────────────────────────────────────────────────────────────────
# Target-blind objective (a pure function of the INPUT graph and a rank vector)
# ──────────────────────────────────────────────────────────────────────────────
def _ff_weight(rank: np.ndarray, src: np.ndarray, tgt: np.ndarray,
               w: np.ndarray) -> float:
    """Total FEEDFORWARD edge weight under ``rank`` (edge (u,v) forward iff r[u] < r[v]).

    Mathematically the same quantity the frozen oracle computes, but derived here from
    the input arrays so the search never calls into the scorer. Used only for the
    acceptance decision; the reported scores all come from the oracle.
    """
    return float(w[rank[src] < rank[tgt]].sum())


# ──────────────────────────────────────────────────────────────────────────────
# Destroy operators - both target-blind, both stochastic
# ──────────────────────────────────────────────────────────────────────────────
def _victims_topk(rank, src, tgt, w, n, k, rng, adj=None):
    """``k`` victims sampled WITHOUT replacement, prob. proportional to back-edge weight.

    The randomised form of H31's ``ruin`` selector: the same target-blind statistic
    (per-node current feedback weight), sampled rather than taken deterministically so
    successive rounds propose different neighbourhoods.
    """
    back = back_edge_weight(rank, src, tgt, w, n)
    tot = back.sum()
    if tot <= 0:
        return rng.choice(n, size=min(k, n), replace=False)
    p = back / tot
    nz = int((p > 0).sum())
    return rng.choice(n, size=min(k, nz), replace=False, p=p)


def _back_adjacency(rank, src, tgt, n):
    """Undirected adjacency (CSR-ish) over the CURRENT back-edge subgraph."""
    is_back = rank[src] > rank[tgt]
    a = np.concatenate([src[is_back], tgt[is_back]])
    b = np.concatenate([tgt[is_back], src[is_back]])
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, a + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, b


def _victims_struct(rank, src, tgt, w, n, k, rng, adj=None):
    """STRUCTURE-AWARE destroy: a cycle-dense, feedback-connected victim set.

    Seed a node with probability proportional to its current back-edge weight, then grow
    the set by breadth-first search over the BACK-EDGE subgraph (edges that are currently
    feedback, taken undirected) until ``k`` nodes are collected. Every node in the set is
    therefore reachable from the seed along edges the current order gets wrong - i.e. the
    set is a connected piece of the feedback tangle, not ``k`` unrelated bad nodes. If the
    seed's back-edge component is smaller than ``k``, reseed and continue.

    This is what ``killed.json`` H31's revival condition asks for ("SCC-, block- or
    cycle-guided rather than random"): a node is in a directed cycle with the seed only
    if it is connected to it in this subgraph, so the component is the smallest cheap
    superset of "the cycles the seed participates in".
    """
    back = back_edge_weight(rank, src, tgt, w, n)
    tot = back.sum()
    if tot <= 0:
        return rng.choice(n, size=min(k, n), replace=False)
    indptr, nbr = adj if adj is not None else _back_adjacency(rank, src, tgt, n)
    p = back / tot
    chosen: List[int] = []
    seen = np.zeros(n, dtype=bool)
    # `seen` is set when a node is PUSHED, but the back-edge subgraph has parallel edges,
    # so one `stack.extend` can push the same node twice (both copies pass the ~seen test
    # in the same vectorized call). `taken` deduplicates at POP time, which leaves the
    # traversal order untouched - the duplicate is skipped where it sits in the LIFO.
    taken = np.zeros(n, dtype=bool)
    guard = 0
    while len(chosen) < k and guard < 64:
        guard += 1
        seed = int(rng.choice(n, p=p))
        if seen[seed]:
            continue
        stack = [seed]
        seen[seed] = True
        while stack and len(chosen) < k:
            u = stack.pop()
            if taken[u]:
                continue
            taken[u] = True
            chosen.append(u)
            lo, hi = int(indptr[u]), int(indptr[u + 1])
            cand = nbr[lo:hi]
            if cand.size:
                cand = cand[~seen[cand]]
                if cand.size:
                    seen[cand] = True
                    stack.extend(int(v) for v in cand)
    if not chosen:
        return rng.choice(n, size=min(k, n), replace=False)
    return np.asarray(chosen, dtype=np.int64)


_DESTROY = {"topk": _victims_topk, "struct": _victims_struct}


# ──────────────────────────────────────────────────────────────────────────────
# TRUE ruin-and-recreate (the pre-registration AMENDMENT of 2026-08-28)
# ──────────────────────────────────────────────────────────────────────────────
def _csr(key: np.ndarray, val: np.ndarray, w: np.ndarray, n: int):
    """Group ``(val, w)`` by ``key`` into a CSR triple ``(indptr, val_sorted, w_sorted)``."""
    order = np.argsort(key, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, np.asarray(key, dtype=np.int64) + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, np.asarray(val)[order], np.asarray(w)[order]


def build_insert_csr(src, tgt, w, n):
    """``(out_csr, in_csr)``: out = u -> its targets, in = u -> its sources."""
    return _csr(src, tgt, w, n), _csr(tgt, src, w, n)


def ruin_and_recreate(rank, victims, out_csr, in_csr, n, rng):
    """REMOVE the victims from the ordering, then re-insert each at its exact-gain slot.

    This is the destroy-and-repair the acceptance schedule needs, and it is what
    ``apply_victim_reinsertions`` is NOT. The difference is what each victim can see:

    * ``apply_victim_reinsertions`` moves victim ``i`` to its optimal gap *with every
      other victim still in place*, applying the move only when the exact gain is > 0.
      Every proposal is therefore >= its input - monotone by construction, measured at
      0 downhill proposals out of 72 (``proto_H80_rebuild_monotone.json``).
    * here the victims are REMOVED first, so victim ``i`` is placed seeing only the
      ``n - k`` survivors plus victims ``0..i-1``. The early placements are blind to the
      later ones, so the recreated order genuinely can be - and usually is - worse than
      the one it came from. That is the point: an acceptance schedule that may keep a
      worse state needs worse states to exist.

    The per-victim placement is still EXACT-GAIN, as ``killed.json`` H31's revival
    condition requires. For a victim ``u`` inserted before index ``p`` of the current
    partial sequence, the feedforward weight it contributes is
    ``f(p) = sum(w(u,v) : pos(v) >= p) + sum(w(x,u) : pos(x) < p)``, which satisfies
    ``f(p+1) - f(p) = in_w_at[p] - out_w_at[p]``. So the whole profile is one cumulative
    sum over the placed neighbours' positions and the optimum is exact, not sampled.
    Ties take the LEFTMOST maximiser, matching the production sift's rule (M14 measured
    that 'first' is the better stage-3 rule).

    Target-blind: reads input edge weights and current ranks only.
    """
    out_indptr, out_nbr, out_w = out_csr
    in_indptr, in_nbr, in_w = in_csr
    seq = np.argsort(rank, kind="stable").astype(np.int64)   # position -> node

    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    seq_keep = seq[keep_mask[seq]]
    pos = np.full(n, -1, dtype=np.int64)
    pos[seq_keep] = np.arange(seq_keep.size, dtype=np.int64)

    order = np.asarray(victims, dtype=np.int64).copy()
    rng.shuffle(order)
    for u in order:
        u = int(u)
        L = seq_keep.size
        d = np.zeros(L, dtype=np.float64)
        lo, hi = int(out_indptr[u]), int(out_indptr[u + 1])
        pv = pos[out_nbr[lo:hi]]
        wv = out_w[lo:hi]
        mv = pv >= 0
        f0 = float(wv[mv].sum())
        if mv.any():
            np.add.at(d, pv[mv], -wv[mv].astype(np.float64))
        lo, hi = int(in_indptr[u]), int(in_indptr[u + 1])
        px = pos[in_nbr[lo:hi]]
        wx = in_w[lo:hi]
        mx = px >= 0
        if mx.any():
            np.add.at(d, px[mx], wx[mx].astype(np.float64))
        f = np.empty(L + 1, dtype=np.float64)
        f[0] = f0
        np.cumsum(d, out=f[1:])
        f[1:] += f0
        p = int(np.argmax(f))                     # leftmost maximiser
        seq_keep = np.insert(seq_keep, p, u)
        pos[seq_keep[p + 1:]] += 1
        pos[u] = p

    new_rank = np.empty(n, dtype=np.int64)
    new_rank[seq_keep] = np.arange(n, dtype=np.int64)
    return new_rank


# ──────────────────────────────────────────────────────────────────────────────
# The common search loop: destroy -> exact-gain rebuild -> short sift -> accept
# ──────────────────────────────────────────────────────────────────────────────
def search(g, init_rank, *, destroy: str, acceptance: str, k: int,
           time_budget_s: float, seed: int, t0_mult: float = 0.0,
           delta_med: Optional[float] = None) -> Dict:
    """One arm. ``acceptance`` is ``"greedy"`` (keep iff better) or ``"metropolis"``.

    Metropolis accepts a worse candidate with probability ``exp(-loss_pp / T)``, where
    ``T`` cools geometrically from ``t0_mult * delta_med`` to ``COOL_RATIO`` of that over
    the wall budget. The returned order is always best-by-oracle, so no arm can regress
    below its input.
    """
    n = g.n_nodes
    src_o, tgt_o = np.asarray(g.src, dtype=np.int64), np.asarray(g.tgt, dtype=np.int64)
    src, tgt, w = build_sift_edges(g)
    total = g.total_weight
    rng = np.random.RandomState(seed + 7919)
    pick = _DESTROY[destroy]

    out_csr, in_csr = build_insert_csr(src, tgt, w, n)

    cur_rank = np.asarray(init_rank, dtype=np.int64).copy()
    cur_ff = _ff_weight(cur_rank, src, tgt, w)
    best_rank = cur_rank.copy()
    best_ff = cur_ff
    best_score = score_from_order(best_rank, src_o, tgt_o, g.weight)
    best_pct = pct(best_score, total)
    ff_to_pct = 100.0 / float(total)

    def _propose(rank_in):
        """One destroy -> exact-gain recreate -> short sift proposal."""
        adj = _back_adjacency(rank_in, src, tgt, n)
        v = pick(rank_in, src, tgt, w, n, k, rng, adj=adj)
        r = ruin_and_recreate(rank_in, v, out_csr, in_csr, n, rng)
        if INNER_SWEEPS:
            r, _, _ = sift(g, r, max_sweeps=INNER_SWEEPS)
        return r

    t_start = time.time()

    # ── Calibration: measure the instance's OWN downhill scale (target-blind) ──────
    cal: List[float] = []
    if acceptance == "metropolis" and delta_med is None:
        for _ in range(N_CAL):
            r = _propose(cur_rank)
            d = (_ff_weight(r, src, tgt, w) - cur_ff) * ff_to_pct
            if d < 0:
                cal.append(-d)
        delta_med = float(np.median(cal)) if cal else 1e-6
    t_cal = time.time() - t_start

    T0 = float(t0_mult) * float(delta_med or 0.0)
    rounds = n_acc = n_uphill = n_sideways = n_proposed_down = 0
    max_excursion = 0.0
    deepest_uphill_step_pp = 0.0

    while True:
        elapsed = time.time() - t_start
        if elapsed >= time_budget_s:
            break
        frac = min(1.0, elapsed / max(time_budget_s, 1e-9))
        T = T0 * (COOL_RATIO ** frac) if T0 > 0 else 0.0

        cand = _propose(cur_rank)
        cand_ff = _ff_weight(cand, src, tgt, w)
        d_pp = (cand_ff - cur_ff) * ff_to_pct
        if d_pp < 0:
            n_proposed_down += 1

        if d_pp > 0:
            take = True
        elif d_pp == 0:
            # Sideways drift across an exact-tie plateau. NOT an uphill move, and the S1
            # data says it is 98.5-100% of everything the Metropolis arms accept - which
            # is why "sideways" exists as its own acceptance rule (amendment 4).
            take = acceptance in ("metropolis", "sideways")
            n_sideways += int(take)
        elif acceptance == "metropolis" and T > 0:
            take = bool(rng.random_sample() < np.exp(d_pp / T))
            if take:
                n_uphill += 1
                deepest_uphill_step_pp = max(deepest_uphill_step_pp, -d_pp)
        else:
            take = False

        if take:
            cur_rank, cur_ff = cand, cand_ff
            n_acc += 1
            if cand_ff > best_ff:
                best_ff = cand_ff
                cand_score = score_from_order(cand, src_o, tgt_o, g.weight)
                if cand_score > best_score:
                    best_score, best_rank = cand_score, cand.copy()
                    best_pct = pct(best_score, total)
            # How far BELOW the best-ever state the search is willing to sit. This is the
            # number that turns a null into a quantitative statement about the barrier.
            exc = (best_ff - cur_ff) * ff_to_pct
            if exc > max_excursion:
                max_excursion = exc
        rounds += 1

    return dict(
        destroy=destroy, acceptance=acceptance, t0_mult=float(t0_mult),
        delta_med_pp=float(delta_med or 0.0), T0_pp=T0,
        best_pct=float(best_pct), rounds=int(rounds), n_accepted=int(n_acc),
        n_uphill_accepted=int(n_uphill), n_sideways_accepted=int(n_sideways),
        n_proposed_downhill=int(n_proposed_down),
        deepest_uphill_step_pp=float(deepest_uphill_step_pp),
        max_excursion_pp=float(max_excursion), k=int(k),
        wall_s=round(time.time() - t_start, 2), cal_s=round(t_cal, 2),
        n_cal_downhill=len(cal),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic: how deep is the valley between this arm's order and the reference?
# ──────────────────────────────────────────────────────────────────────────────
def barrier_depth(g, rank_a: np.ndarray, ref_order: np.ndarray,
                  n_points: int = 21) -> Dict:
    """Minimum of the linear rank-blend path from ``rank_a`` to ``ref_order``.

    The same instrument H78 used on the connectome (``h78_path_to_reference.py``): score
    ``argsort(argsort((1-t)*rank_a + t*ref))`` on a grid of ``t`` and report the deepest
    point relative to the better endpoint. DIAGNOSTIC ONLY - no arm reads this.
    """
    src_o, tgt_o = np.asarray(g.src, dtype=np.int64), np.asarray(g.tgt, dtype=np.int64)
    a = np.asarray(rank_a, dtype=np.float64)
    b = np.asarray(ref_order, dtype=np.float64)
    pts = []
    for t in np.linspace(0.0, 1.0, n_points):
        key = (1.0 - t) * a + t * b
        r = np.argsort(np.argsort(key, kind="stable"), kind="stable").astype(np.int64)
        pts.append((float(t), pct(score_from_order(r, src_o, tgt_o, g.weight),
                                  g.total_weight)))
    endpoints = max(pts[0][1], pts[-1][1])
    t_min, p_min = min(pts, key=lambda z: z[1])
    return dict(path=[[round(t, 3), p] for t, p in pts],
                endpoint_max_pct=endpoints, valley_t=t_min, valley_pct=p_min,
                barrier_depth_pp=float(endpoints - p_min))


# ──────────────────────────────────────────────────────────────────────────────
# One instance: build the shared sift fixed point, then run every arm from it
# ──────────────────────────────────────────────────────────────────────────────
def run_instance(g, seed: int, *, epochs: int, sift_sweeps: int, budget_s: float,
                 full_grid: bool, ref_order=None, ref_pct=None) -> Dict:
    src_o, tgt_o = np.asarray(g.src, dtype=np.int64), np.asarray(g.tgt, dtype=np.int64)
    n, total = g.n_nodes, g.total_weight

    greedy = greedy_fas_order(g)
    t0 = time.time()
    if epochs > 0:
        res = run_rocket(g, RocketConfig(epochs=epochs), seed=seed, device=CPU,
                         init_positions=_init_positions_from_order(greedy, CPU))
        rank0 = np.argsort(np.argsort(res.best_positions, kind="stable"),
                           kind="stable").astype(np.int64)
        rocket_pct = float(res.best_pct)
    else:
        rank0, rocket_pct = np.asarray(greedy, dtype=np.int64), None
    t_rocket = time.time() - t0

    t0 = time.time()
    a0_rank, a0_score, slog = sift(g, rank0, max_sweeps=sift_sweeps)
    t_sift = time.time() - t0
    a0_pct = pct(a0_score, total)

    # Destroy size: 5% of the nodes, capped at 40. The cap is a COST bound, not a tuning
    # choice - the exact-gain rebuild costs one O(m log m) kernel call per victim, so an
    # uncapped k would spend the whole budget on a handful of rounds at n=4000. k is
    # identical across every arm of an instance, which is what the comparison needs.
    k = max(4, min(40, int(round(0.05 * n))))
    arms: Dict[str, Dict] = {}

    # A1p - production ils_lns, the harness positive control (H31 settings)
    if full_grid:
        t0 = time.time()
        _r, s_p, rlog = ils_lns(g, a0_rank, time_budget_s=budget_s, k=k, mode="ruin",
                                inner_sweeps=INNER_SWEEPS, sequential_victims=True,
                                seed=seed)
        arms["A1p_ils_lns_production"] = dict(
            best_pct=pct(s_p, total), rounds=len(rlog),
            n_accepted=int(sum(1 for r in rlog if r["accepted"])),
            n_uphill_accepted=0, max_excursion_pp=0.0, k=k,
            wall_s=round(time.time() - t0, 2))

    arms["A1_topk_greedy"] = search(g, a0_rank, destroy="topk", acceptance="greedy",
                                    k=k, time_budget_s=budget_s, seed=seed)
    arms["A2_struct_greedy"] = search(g, a0_rank, destroy="struct", acceptance="greedy",
                                      k=k, time_budget_s=budget_s, seed=seed)
    for m in T0_GRID:
        arms[f"A3_struct_anneal_T{m}"] = search(
            g, a0_rank, destroy="struct", acceptance="metropolis", k=k,
            time_budget_s=budget_s, seed=seed, t0_mult=m)
    if full_grid:
        for m in T0_GRID:
            arms[f"A4_topk_anneal_T{m}"] = search(
                g, a0_rank, destroy="topk", acceptance="metropolis", k=k,
                time_budget_s=budget_s, seed=seed, t0_mult=m)

    row = dict(seed=seed, n=int(n), m=int(g.n_edges), k=k,
               greedy_pct=pct(score_from_order(np.asarray(greedy, dtype=np.int64),
                                               src_o, tgt_o, g.weight), total),
               rocket_pct=rocket_pct, A0_sift_pct=a0_pct,
               t_rocket_s=round(t_rocket, 2), t_sift_s=round(t_sift, 2),
               n_sift_sweeps=len(slog), budget_s=budget_s, arms=arms)
    if ref_order is not None:
        row["reference_pct"] = ref_pct
        row["gap_A0_to_ref_pp"] = float(ref_pct - a0_pct)
        row["barrier"] = barrier_depth(g, a0_rank, ref_order)
    return row


def _summarise(rows: List[Dict], full_grid: bool) -> Dict:
    """Pre-registered statistics: delta_A3, the attribution terms, and the T0 profile."""
    def arm(r, key):
        return r["arms"][key]["best_pct"]

    per_seed = []
    for r in rows:
        mono = max(r["A0_sift_pct"], arm(r, "A1_topk_greedy"), arm(r, "A2_struct_greedy"))
        a3 = {m: arm(r, f"A3_struct_anneal_T{m}") for m in T0_GRID}
        row = dict(seed=r["seed"], monotone_best=mono, A0=r["A0_sift_pct"],
                   A1=arm(r, "A1_topk_greedy"), A2=arm(r, "A2_struct_greedy"),
                   A3_by_T={str(m): v for m, v in a3.items()},
                   A3_best=max(a3.values()),
                   delta_A3=max(a3.values()) - mono,
                   structure_alone=arm(r, "A2_struct_greedy") - arm(r, "A1_topk_greedy"))
        if full_grid:
            a4 = {m: arm(r, f"A4_topk_anneal_T{m}") for m in T0_GRID}
            row["A4_best"] = max(a4.values())
            row["acceptance_alone"] = max(a4.values()) - arm(r, "A1_topk_greedy")
            row["acceptance_given_structure"] = (max(a3.values())
                                                 - arm(r, "A2_struct_greedy"))
            row["interaction"] = row["acceptance_given_structure"] - row["acceptance_alone"]
        per_seed.append(row)

    def mean(key):
        vals = [p[key] for p in per_seed if key in p]
        return float(np.mean(vals)) if vals else None

    def _mono(r):
        return max(r["A0_sift_pct"], arm(r, "A1_topk_greedy"), arm(r, "A2_struct_greedy"))

    # Per-T0 mean delta vs the monotone best - K3 reads this.
    per_T = {}
    for m in T0_GRID:
        per_T[str(m)] = float(np.mean([arm(r, f"A3_struct_anneal_T{m}") - _mono(r)
                                       for r in rows]))

    # SELECTION-SAFE primary (amendment): every Metropolis cell's MEAN delta over seeds,
    # then pick the best CELL by that mean. Taking a per-seed max over 10 cells with 3
    # seeds would capitalise on chance; picking one cell by its seed-mean does not.
    families = ["A3_struct_anneal"] + (["A4_topk_anneal"] if full_grid else [])
    cell_mean = {f"{fam}_T{m}": float(np.mean([arm(r, f"{fam}_T{m}") - _mono(r)
                                               for r in rows]))
                 for fam in families for m in T0_GRID}
    best_cell = max(cell_mean, key=cell_mean.get)

    uphill = {str(m): int(sum(r["arms"][f"A3_struct_anneal_T{m}"]["n_uphill_accepted"]
                              for r in rows)) for m in T0_GRID}
    exc = {str(m): float(max(r["arms"][f"A3_struct_anneal_T{m}"]["max_excursion_pp"]
                             for r in rows)) for m in T0_GRID}
    n_adj = 0
    fam_best, m_best = best_cell.rsplit("_T", 1)
    fam_T = [cell_mean[f"{fam_best}_T{m}"] for m in T0_GRID]
    for i in range(len(T0_GRID) - 1):
        if fam_T[i] > 0 and fam_T[i + 1] > 0:
            n_adj += 1

    out = dict(per_seed=per_seed, mean_delta_A3=mean("delta_A3"),
               mean_structure_alone=mean("structure_alone"),
               mean_delta_A3_by_T0=per_T,
               mean_delta_by_cell=cell_mean,
               best_cell=best_cell, best_cell_mean_delta=cell_mean[best_cell],
               n_adjacent_T0_pairs_positive_in_best_family=n_adj,
               total_uphill_accepted_by_T0=uphill,
               max_excursion_pp_by_T0=exc,
               n_T0_beating_monotone=int(sum(1 for v in per_T.values() if v > 0)))
    if full_grid:
        out["mean_acceptance_alone"] = mean("acceptance_alone")
        out["mean_acceptance_given_structure"] = mean("acceptance_given_structure")
        out["mean_interaction"] = mean("interaction")
    if rows and "barrier" in rows[0]:
        out["mean_barrier_depth_pp"] = float(np.mean([r["barrier"]["barrier_depth_pp"]
                                                      for r in rows]))
    return out


def run_control(g, seed: int, *, epochs: int, sift_sweeps: int, budget_s: float) -> Dict:
    """AMENDMENT 4 control: does ZERO-temperature sideways drift explain the whole gain?

    Same instance, same start, same budget, same destroy operators - the only change is
    the acceptance rule, which becomes "accept iff not strictly worse". If this recovers
    the Metropolis arms' advantage, then the temperature is decorative and what the arms
    were actually buying is plateau drift, not barrier crossing.
    """
    n = g.n_nodes
    greedy = greedy_fas_order(g)
    if epochs > 0:
        res = run_rocket(g, RocketConfig(epochs=epochs), seed=seed, device=CPU,
                         init_positions=_init_positions_from_order(greedy, CPU))
        rank0 = np.argsort(np.argsort(res.best_positions, kind="stable"),
                           kind="stable").astype(np.int64)
    else:
        rank0 = np.asarray(greedy, dtype=np.int64)
    a0_rank, a0_score, _ = sift(g, rank0, max_sweeps=sift_sweeps)
    k = max(4, min(40, int(round(0.05 * n))))
    arms = {}
    for nm, dz, ac in (("A1_topk_greedy", "topk", "greedy"),
                       ("A2_struct_greedy", "struct", "greedy"),
                       ("A5_topk_sideways", "topk", "sideways"),
                       ("A6_struct_sideways", "struct", "sideways")):
        arms[nm] = search(g, a0_rank, destroy=dz, acceptance=ac, k=k,
                          time_budget_s=budget_s, seed=seed)
    return dict(seed=seed, n=int(n), m=int(g.n_edges), k=k,
                A0_sift_pct=pct(a0_score, g.total_weight), budget_s=budget_s, arms=arms)


def _summarise_control(rows: List[Dict]) -> Dict:
    def a(r, key):
        return r["arms"][key]["best_pct"]
    per_seed = [dict(seed=r["seed"], A0=r["A0_sift_pct"],
                     A1_topk_greedy=a(r, "A1_topk_greedy"),
                     A2_struct_greedy=a(r, "A2_struct_greedy"),
                     A5_topk_sideways=a(r, "A5_topk_sideways"),
                     A6_struct_sideways=a(r, "A6_struct_sideways"),
                     sideways_alone_topk=a(r, "A5_topk_sideways") - max(
                         r["A0_sift_pct"], a(r, "A1_topk_greedy"),
                         a(r, "A2_struct_greedy")),
                     sideways_alone_struct=a(r, "A6_struct_sideways") - max(
                         r["A0_sift_pct"], a(r, "A1_topk_greedy"),
                         a(r, "A2_struct_greedy")))
                for r in rows]
    return dict(per_seed=per_seed,
                mean_sideways_alone_topk=float(np.mean(
                    [p["sideways_alone_topk"] for p in per_seed])),
                mean_sideways_alone_struct=float(np.mean(
                    [p["sideways_alone_struct"] for p in per_seed])),
                total_sideways_accepted={
                    nm: int(sum(r["arms"][nm]["n_sideways_accepted"] for r in rows))
                    for nm in ("A5_topk_sideways", "A6_struct_sideways")},
                total_uphill_accepted={
                    nm: int(sum(r["arms"][nm]["n_uphill_accepted"] for r in rows))
                    for nm in ("A5_topk_sideways", "A6_struct_sideways")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["S1", "S2", "S3", "S1c", "S2c"])
    args = ap.parse_args()
    seeds = (42, 123, 999)
    rows: List[Dict] = []

    if args.stage in ("S1c", "S2c"):
        big = args.stage == "S2c"
        for s in seeds:
            g, _ref, _p = gapmod.make_hard_synthetic_graph(
                **({"n": 4000} if big else {}), seed=s)
            rows.append(run_control(g, s, epochs=4000, sift_sweeps=20,
                                    budget_s=120.0 if big else 30.0))
            print(f"[{args.stage} s{s}] done", flush=True)
        summary = _summarise_control(rows)
        payload = dict(stage=args.stage, seeds=list(seeds), inner_sweeps=INNER_SWEEPS,
                       rows=rows, summary=summary)
        path = OUT / f"proto_H80_{args.stage}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")
        print(json.dumps(summary, indent=2)[:3000])
        return

    if args.stage == "S1":
        for s in seeds:
            g, ref, ref_pct = gapmod.make_hard_synthetic_graph(seed=s)
            rows.append(run_instance(g, s, epochs=4000, sift_sweeps=20, budget_s=30.0,
                                     full_grid=True, ref_order=ref, ref_pct=ref_pct))
            print(f"[S1 s{s}] A0={rows[-1]['A0_sift_pct']:.4f} "
                  f"delta_A3={_summarise(rows[-1:], True)['mean_delta_A3']:+.4f}",
                  flush=True)
        summary = _summarise(rows, True)
    elif args.stage == "S2":
        for s in seeds:
            g, ref, ref_pct = gapmod.make_hard_synthetic_graph(n=4000, seed=s)
            rows.append(run_instance(g, s, epochs=4000, sift_sweeps=20, budget_s=120.0,
                                     full_grid=False, ref_order=ref, ref_pct=ref_pct))
            print(f"[S2 s{s}] A0={rows[-1]['A0_sift_pct']:.4f} "
                  f"delta_A3={_summarise(rows[-1:], False)['mean_delta_A3']:+.4f}",
                  flush=True)
        summary = _summarise(rows, False)
    else:
        g = io.load_dataset("mouse")
        for s in seeds:
            rows.append(run_instance(g, s, epochs=0, sift_sweeps=40, budget_s=30.0,
                                     full_grid=False))
            print(f"[S3 s{s}] A0={rows[-1]['A0_sift_pct']:.4f} "
                  f"delta_A3={_summarise(rows[-1:], False)['mean_delta_A3']:+.4f}",
                  flush=True)
        summary = _summarise(rows, False)

    payload = dict(stage=args.stage, seeds=list(seeds), t0_grid=list(T0_GRID),
                   cool_ratio=COOL_RATIO, n_cal=N_CAL, inner_sweeps=INNER_SWEEPS,
                   rows=rows, summary=summary)
    path = OUT / f"proto_H80_{args.stage}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    print(json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()
