"""H71 prototype gate — SUBSET-BIPARTITION interval repair (the GRaSP 'tuck').

The question this answers
-------------------------
H45/H52's pair relocation takes a backward edge ``u -> v`` (``rank[v] < rank[u]``) and rewrites
the interval as ``<prefix, u, v, suffix>``, where the cut is a POSITION. That is a prefix
bipartition of the interval interior. The general move — Definition 4.1 of GRaSP (Lam, Andrews,
Ramsey, UAI 2022, arXiv:2206.05421) — lets the bipartition be an ARBITRARY SUBSET ``gamma`` of the
interior:

    <d1, v, d2, u, d3>   ->   <d1, gamma, u, v, gamma_c, d3>

with each group keeping its internal relative order. A PREFIX ``gamma`` reproduces H45 exactly,
so the pair move is the prefix special case and ``max_gamma Delta >= max_cut Delta_pair`` by
construction. This is the one thing no move class in this campaign can do: reorder the interval
INTERIOR by a NON-POSITIONAL predicate.

The exact gain
--------------
Let ``a_z = w(z->v) - w(v->z)`` and ``b_z = w(u->z) - w(z->u)`` for ``z`` in the interior. Then

    Delta(gamma) = (w_uv - w_vu)
                 + sum_{z in gamma}   a_z
                 + sum_{z in gamma_c} b_z
                 + sum_{z in gamma, y in gamma_c, pos(y) < pos(z)} [ w(z->y) - w(y->z) ]

**Proof.** The interval ``[rank[v], rank[u]]`` is a contiguous position range and every node that
moves stays inside it, so by the contiguous-block lemma (proved in :mod:`mfas.refine.segment`) no
edge with an endpoint outside the interval can flip. Inside, the relative order of exactly four
kinds of pair changes: ``(u,v)`` itself (``+w_uv - w_vu``); ``v`` against each ``z in gamma``
(``v`` passes to their right, so ``z->v`` becomes feedforward: ``+a_z``); ``u`` against each
``z in gamma_c`` (``u`` passes to their left: ``+b_z``); and each ``z in gamma`` against each
``y in gamma_c`` that ORIGINALLY PRECEDED it (``z`` passes to ``y``'s left: ``+w(z->y) -
w(y->z)``). Nothing inside ``gamma`` or inside ``gamma_c`` flips, since each keeps its internal
relative order. The fourth term is identically zero when ``gamma`` is a prefix, which is why H45
never had to compute it.

Writing ``B = sum_{z in interior} b_z`` (finite support: only neighbours of ``u``) this is

    Delta(gamma) = base + B + sum_{z in gamma} (a_z - b_z) + coupling(gamma)

so the SEPARABLE part is maximised by the per-node preference sign ``a_z > b_z`` — rule (b)
below — and the coupling term is the whole of what makes the problem non-trivial.

Cost. ``a_z`` and ``b_z`` have support only on ``N(v)`` and ``N(u)`` intersected with the
interval, so the separable part costs ``O(d(u) + d(v))`` — the same as H45, and independent of
the interval length. The coupling term costs ``O(sum_{z in gamma} deg(z))``: for each member of
``gamma`` we visit its own neighbours and keep those that lie earlier inside the interval and
outside ``gamma``. That is data-sized, not interval-sized, and STAGE 1 measures it rather than
assuming it.

The three gamma rules (pre-registered in autoresearch/queue.json, item H71)
--------------------------------------------------------------------------
(a) **best PREFIX** — H45's exact kernel, unmodified, imported from
    :mod:`mfas.refine.pair_relocate`. This is the CONTROL: it is exactly what the campaign
    already has, so everything H71 can claim is the difference against it.
(b) **preference sign** — ``gamma = {z : a_z > b_z}``. Maximises the separable part exactly;
    the coupling term is then evaluated, not ignored, so the reported ``Delta`` is EXACT.
(c) **forward reachability** — ``gamma = {z : z reaches u by currently-feedforward edges that
    stay inside the interval}``, the literal GRaSP rule ("the ancestors of k"). Restricted to
    intervals shorter than ``_REACH_MAX_SPAN`` so the reverse BFS stays affordable.

Rules (b) and (c) are heuristics for a subset problem that is not obviously tractable, so
``Delta(b)`` and ``Delta(c)`` are LOWER bounds on the subset optimum and may be worse than the
prefix control on any given candidate. Every arm therefore reports ``max`` over the rules it
computed, and the per-candidate winner is recorded.

What STAGE 1 measures
---------------------
The pre-registered FREE KILL, run before anything expensive, on the champion's own connectome
order:

1. **Is the subset freedom used at all?** The fraction of positive-gain candidates whose winning
   ``gamma`` is a strict NON-PREFIX. If rule (b)'s gamma is a prefix for >95% of candidates, this
   is H45 under another name and the item dies for free.
2. **Is it worth anything?** ``Delta(best rule) - Delta(prefix)`` per candidate — the ONLY
   quantity H71 can be credited with, per meta-rule M8 (a class is judged by what it ADDS).
3. **Exactness.** Predicted ``Delta`` versus the frozen oracle's realised delta on >= 25 applied
   candidates, zero mismatches required before any total is quoted. H45's own gate scored 1/18
   on its first run because it aggregated per-EDGE instead of per-POSITION events; this kernel
   aggregates per NODE, which is immune to that particular bug, and the cross-check is what
   establishes that rather than the argument.
4. **The cost constants** that size STAGE 2: coupling edge visits and wall clock per candidate.

STAGE 2 (``--stage 2``) is the realised-gain arm: sequential application, H52-style, of the
prefix control and of the tuck, at a MATCHED pop budget, plus the composed arm that gives the
M8 redundancy fraction.

Leakage-safety: every quantity is computed from edge weights and current ranks. The frozen
oracle is called only to CHECK a predicted gain after the fact, never to choose a move.
``data/best_solution`` is never read. The champion order is identified through
``autoresearch/sota.json`` and its exact score is ASSERTED against ``sota.json``'s recorded
figure before anything is measured — never by a bare name glob (H46's prototype silently
measured a killed variant that way).

Run:
    PYTHONPATH=src python experiments/proto_H71_tuck.py --stage 1 --dataset connectome --topk 2000
Out:
    experiments/outputs/proto_H71_stage1_<dataset>.json
"""
from __future__ import annotations

import argparse
import glob
import heapq
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mfas import io                                                      # noqa: E402
from mfas.metrics import pct, score_from_order                           # noqa: E402
from mfas.refine.pair_relocate import build_direction_csr, pair_move_gain  # noqa: E402

# Rule (c) runs a reverse BFS over the interval interior, which is the only interval-sized cost
# in this file. Bounded per the pre-registration.
_REACH_MAX_SPAN = 5_000
# Exactness cross-checks against the frozen oracle before any total may be quoted.
_N_ORACLE_CHECKS = 30


# ──────────────────────────────────────────────────────────────────────────────
# Champion order, identified through sota.json and verified by its exact score
# ──────────────────────────────────────────────────────────────────────────────
def load_champion_order(dataset: str) -> Tuple[np.ndarray, str, str, float]:
    """``(rank, variant_id, positions_path, sota_pct)`` for ``dataset``'s champion.

    The variant id, role and expected exact score all come from ``autoresearch/sota.json``.
    A run file is admissible only if its ``pct`` equals ``sota.json``'s recorded exact figure,
    so a stale or mislabelled artifact cannot be measured by accident.
    """
    sota = json.loads((_ROOT / "autoresearch" / "sota.json").read_text())
    entry = sota["datasets"][dataset]
    variant = entry["champion"]
    role = entry.get("role", "confirm")
    want = float(entry.get("pct_mean_exact", entry["pct_mean"]))

    pattern = str(_ROOT / "results" / f"*-{variant}-{dataset}-*-{role}-*.json")
    best: Optional[Tuple[str, str]] = None
    for f in sorted(glob.glob(pattern)):
        rec = json.loads(Path(f).read_text())
        if rec.get("algo") not in (None, variant) and variant not in str(rec.get("algo")):
            continue
        if abs(float(rec["pct"]) - want) > 1e-9:
            continue
        p = rec.get("best_positions_path")
        if p and (_ROOT / p).exists():
            best = (f, str(_ROOT / p))
            break
    if best is None:
        raise SystemExit(
            f"no {variant}/{dataset}/{role} run on disk whose pct equals sota.json's {want!r}")
    positions = np.load(best[1])
    rank = np.argsort(np.argsort(positions, kind="stable"), kind="stable").astype(np.int64)
    return rank, variant, best[1], want


# ──────────────────────────────────────────────────────────────────────────────
# The subset kernel
# ──────────────────────────────────────────────────────────────────────────────
class TuckKernel:
    """Exact subset-bipartition gains for one graph, against a mutable ``rank``."""

    def __init__(self, g, rank: np.ndarray):
        src = np.asarray(g.src, dtype=np.int64)
        tgt = np.asarray(g.tgt, dtype=np.int64)
        w = np.asarray(g.weight, dtype=np.float64)
        keep = src != tgt
        self.n = g.n_nodes
        self.src, self.tgt, self.w = src, tgt, w
        self.out_ptr, self.out_idx, self.out_w = build_direction_csr(
            src[keep], tgt[keep], w[keep], self.n)
        self.in_ptr, self.in_idx, self.in_w = build_direction_csr(
            tgt[keep], src[keep], w[keep], self.n)
        self.rank = rank
        self.coupling_visits = 0          # cost counter: edges touched by the coupling term

    # -- the two endpoint terms, per NODE (parallel edges summed, as the scorer does) ------
    def endpoint_terms(self, u: int, v: int, lo: int, hi: int
                       ) -> Tuple[Dict[int, float], Dict[int, float], float, float]:
        rank = self.rank
        a: Dict[int, float] = {}          # a_z = w(z->v) - w(v->z)
        b: Dict[int, float] = {}          # b_z = w(u->z) - w(z->u)
        w_uv = 0.0
        w_vu = 0.0
        for k in range(self.in_ptr[v], self.in_ptr[v + 1]):
            z = int(self.in_idx[k])
            if z == u:
                w_uv += float(self.in_w[k])
                continue
            p = int(rank[z])
            if lo < p < hi:
                a[z] = a.get(z, 0.0) + float(self.in_w[k])
        for k in range(self.out_ptr[v], self.out_ptr[v + 1]):
            z = int(self.out_idx[k])
            if z == u:
                w_vu += float(self.out_w[k])
                continue
            p = int(rank[z])
            if lo < p < hi:
                a[z] = a.get(z, 0.0) - float(self.out_w[k])
        for k in range(self.out_ptr[u], self.out_ptr[u + 1]):
            z = int(self.out_idx[k])
            if z == v:
                continue
            p = int(rank[z])
            if lo < p < hi:
                b[z] = b.get(z, 0.0) + float(self.out_w[k])
        for k in range(self.in_ptr[u], self.in_ptr[u + 1]):
            z = int(self.in_idx[k])
            if z == v:
                continue
            p = int(rank[z])
            if lo < p < hi:
                b[z] = b.get(z, 0.0) - float(self.in_w[k])
        return a, b, w_uv, w_vu

    def coupling(self, gamma: Set[int], lo: int, hi: int) -> float:
        """``sum_{z in gamma, y in gamma_c, pos(y) < pos(z)} [ w(z->y) - w(y->z) ]``.

        Costs ``O(sum_{z in gamma} deg(z))`` — the members' own adjacency, never the interval.
        """
        rank = self.rank
        total = 0.0
        visits = 0
        for z in gamma:
            pz = int(rank[z])
            for k in range(self.out_ptr[z], self.out_ptr[z + 1]):
                y = int(self.out_idx[k])
                py = int(rank[y])
                if lo < py < pz and y not in gamma:
                    total += float(self.out_w[k])
            visits += int(self.out_ptr[z + 1] - self.out_ptr[z])
            for k in range(self.in_ptr[z], self.in_ptr[z + 1]):
                y = int(self.in_idx[k])
                py = int(rank[y])
                if lo < py < pz and y not in gamma:
                    total -= float(self.in_w[k])
            visits += int(self.in_ptr[z + 1] - self.in_ptr[z])
        self.coupling_visits += visits
        return total

    def gain_of_gamma(self, gamma: Set[int], a: Dict[int, float], b: Dict[int, float],
                      base: float, b_total: float, lo: int, hi: int) -> float:
        """EXACT ``Delta(gamma)`` — separable part plus the evaluated coupling term."""
        sep = 0.0
        for z in gamma:
            sep += a.get(z, 0.0) - b.get(z, 0.0)
        return base + b_total + sep + self.coupling(gamma, lo, hi)

    # -- the pre-registered gamma rules ---------------------------------------------------
    def gamma_preference(self, a: Dict[int, float], b: Dict[int, float]) -> Set[int]:
        """Rule (b): ``{z : a_z > b_z}``. Maximises the SEPARABLE part exactly."""
        out: Set[int] = set()
        for z in a:
            if a[z] > b.get(z, 0.0):
                out.add(z)
        for z in b:
            if z not in out and 0.0 > b[z]:
                out.add(z)
        return out

    def gamma_reachable(self, u: int, lo: int, hi: int) -> Optional[Set[int]]:
        """Rule (c): interior nodes that reach ``u`` along currently-FEEDFORWARD interior edges.

        ``None`` when the interval is longer than ``_REACH_MAX_SPAN`` (not attempted).
        """
        if hi - lo > _REACH_MAX_SPAN:
            return None
        rank = self.rank
        seen: Set[int] = set()
        stack: List[int] = []
        for k in range(self.in_ptr[u], self.in_ptr[u + 1]):
            y = int(self.in_idx[k])
            p = int(rank[y])
            if lo < p < hi and y not in seen:
                seen.add(y)
                stack.append(y)
        while stack:
            z = stack.pop()
            pz = int(rank[z])
            for k in range(self.in_ptr[z], self.in_ptr[z + 1]):
                y = int(self.in_idx[k])
                py = int(rank[y])
                if lo < py < pz and y not in seen:      # y -> z is feedforward, inside d2
                    seen.add(y)
                    stack.append(y)
        return seen

    # -- applying a move -------------------------------------------------------------------
    @staticmethod
    def apply_gamma(seq: np.ndarray, rank: np.ndarray, u: int, v: int,
                    gamma: Set[int], lo: int, hi: int) -> None:
        """Rewrite ``seq[lo:hi+1]`` as ``gamma, u, v, gamma_c`` and repair ``rank`` in place."""
        head: List[int] = []
        tail: List[int] = []
        for x in seq[lo:hi + 1]:
            xi = int(x)
            if xi == u or xi == v:
                continue
            (head if xi in gamma else tail).append(xi)
        block = head + [u, v] + tail
        seq[lo:hi + 1] = np.asarray(block, dtype=np.int64)
        rank[seq[lo:hi + 1]] = np.arange(lo, hi + 1, dtype=np.int64)


def is_prefix(gamma: Set[int], rank: np.ndarray, lo: int, hi: int) -> bool:
    """Is ``gamma`` exactly the interior positions ``lo+1 .. lo+|gamma|``?

    That is the ONLY shape H45's prefix kernel can express, so a gamma failing this test is
    a move the campaign's existing pair class provably cannot make.
    """
    if not gamma:
        return True
    ps = sorted(int(rank[z]) for z in gamma)
    return ps == list(range(lo + 1, lo + 1 + len(ps)))


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — the free kill, the distinctness measurement, and the cost constants
# ──────────────────────────────────────────────────────────────────────────────
def stage1(dataset: str, topk: int) -> int:
    g = io.load_dataset(dataset)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    total = g.total_weight

    rank, variant, pos_path, sota_pct = load_champion_order(dataset)
    base_score = score_from_order(rank, src, tgt, g.weight)
    base_pct = pct(base_score, total)
    if abs(base_pct - sota_pct) > 1e-9:
        raise SystemExit(f"champion order scores {base_pct!r}, sota.json says {sota_pct!r}")
    print(f"champion {variant}/{dataset}: {base_pct:.11f}  ({pos_path})", flush=True)

    K = TuckKernel(g, rank)
    n = g.n_nodes
    seq = np.empty(n, dtype=np.int64)
    seq[rank] = np.arange(n, dtype=np.int64)

    back = np.flatnonzero(rank[src] > rank[tgt])
    cand = back[np.argsort(-w[back], kind="stable")][:topk]
    print(f"backward edges {back.size:,}; examining the top {cand.size:,} by weight", flush=True)

    rows: List[Dict] = []
    t0 = time.time()
    n_reach_attempted = 0
    for ei in cand:
        u = int(src[ei]); v = int(tgt[ei])
        lo = int(rank[v]); hi = int(rank[u])
        if hi <= lo:
            continue

        # (a) the control: H45's exact prefix kernel, imported unmodified
        g_a, cut, _lo, _hi = pair_move_gain(rank, u, v, K.out_ptr, K.out_idx, K.out_w,
                                            K.in_ptr, K.in_idx, K.in_w)

        a, b, w_uv, w_vu = K.endpoint_terms(u, v, lo, hi)
        base = w_uv - w_vu
        b_total = 0.0
        for z in b:
            b_total += b[z]

        # (b) preference sign
        gam_b = K.gamma_preference(a, b)
        g_b = K.gain_of_gamma(gam_b, a, b, base, b_total, lo, hi)

        # (c) forward reachability, bounded span
        gam_c = K.gamma_reachable(u, lo, hi)
        if gam_c is None:
            g_c = None
        else:
            n_reach_attempted += 1
            g_c = K.gain_of_gamma(gam_c, a, b, base, b_total, lo, hi)

        opts = [("prefix", g_a, None), ("preference", g_b, gam_b)]
        if g_c is not None:
            opts.append(("reachable", g_c, gam_c))
        win_rule, win_gain, win_gamma = max(opts, key=lambda t: t[1])

        rows.append(dict(
            edge=int(ei), u=u, v=v, lo=lo, hi=hi, span=hi - lo,
            w_uv=w_uv, w_vu=w_vu,
            gain_prefix=float(g_a), prefix_cut=int(cut),
            gain_preference=float(g_b), size_gamma_b=len(gam_b),
            gain_reachable=(None if g_c is None else float(g_c)),
            size_gamma_c=(None if gam_c is None else len(gam_c)),
            win_rule=win_rule, win_gain=float(win_gain),
            win_is_prefix=(True if win_gamma is None
                           else is_prefix(win_gamma, rank, lo, hi)),
            b_is_prefix=is_prefix(gam_b, rank, lo, hi),
        ))
    t_scan = time.time() - t0

    pos_rows = [r for r in rows if r["win_gain"] > 1e-9]
    pos_prefix = [r for r in rows if r["gain_prefix"] > 1e-9]
    strictly_better = [r for r in rows if r["win_gain"] > r["gain_prefix"] + 1e-9]
    nonprefix_wins = [r for r in strictly_better if not r["win_is_prefix"]]

    print(f"scan {t_scan:.1f}s for {len(rows):,} candidates "
          f"({1e3 * t_scan / max(len(rows), 1):.2f} ms each); "
          f"coupling edge visits {K.coupling_visits:,} "
          f"({K.coupling_visits / max(len(rows), 1):.0f} per candidate)", flush=True)
    print(f"positive gain: prefix {len(pos_prefix):,} | best-of-rules {len(pos_rows):,}",
          flush=True)
    print(f"FREE KILL: candidates where a subset beats the best prefix: "
          f"{len(strictly_better):,} ({100.0 * len(strictly_better) / max(len(rows), 1):.2f}% of "
          f"scanned); of those, strict NON-prefix gamma: {len(nonprefix_wins):,}", flush=True)

    sum_prefix = sum(max(r["gain_prefix"], 0.0) for r in rows)
    sum_best = sum(max(r["win_gain"], 0.0) for r in rows)
    print(f"capacity (NOT achievable, per M11 - overlapping intervals): "
          f"prefix +{100.0 * sum_prefix / total:.6f} pp | "
          f"best-of-rules +{100.0 * sum_best / total:.6f} pp | "
          f"subset increment +{100.0 * (sum_best - sum_prefix) / total:.6f} pp", flush=True)

    # ── EXACTNESS: the frozen oracle checks the predicted Delta, one move at a time ──
    checks: List[Dict] = []
    order_by_gain = sorted(rows, key=lambda r: -r["win_gain"])
    for r in order_by_gain:
        if len(checks) >= _N_ORACLE_CHECKS:
            break
        if r["win_gain"] <= 1e-9:
            break
        u, v, lo, hi = r["u"], r["v"], r["lo"], r["hi"]
        if r["win_rule"] == "prefix":
            gam = {int(x) for x in seq[lo + 1:r["prefix_cut"] + 1]}
        elif r["win_rule"] == "preference":
            a, b, _wuv, _wvu = K.endpoint_terms(u, v, lo, hi)
            gam = K.gamma_preference(a, b)
        else:
            gam = K.gamma_reachable(u, lo, hi) or set()
        trial_seq = seq.copy()
        trial_rank = rank.copy()
        TuckKernel.apply_gamma(trial_seq, trial_rank, u, v, gam, lo, hi)
        realised = float(score_from_order(trial_rank, src, tgt, g.weight) - base_score)
        checks.append(dict(u=u, v=v, rule=r["win_rule"], predicted=r["win_gain"],
                           realised=realised,
                           match=bool(abs(realised - r["win_gain"]) < 1e-6)))
    n_match = sum(c["match"] for c in checks)
    print(f"EXACTNESS: {n_match}/{len(checks)} predicted == frozen-oracle realised", flush=True)
    for c in checks:
        if not c["match"]:
            print(f"   MISMATCH {c['rule']} u={c['u']} v={c['v']} "
                  f"pred={c['predicted']} real={c['realised']}", flush=True)

    out = dict(
        item="H71", stage=1, dataset=dataset,
        champion_variant=variant, champion_positions=pos_path,
        champion_pct=base_pct, sota_pct=sota_pct,
        n_backward_edges=int(back.size), n_candidates=len(rows),
        reach_max_span=_REACH_MAX_SPAN, n_reach_attempted=n_reach_attempted,
        t_scan_s=t_scan, ms_per_candidate=1e3 * t_scan / max(len(rows), 1),
        coupling_edge_visits=int(K.coupling_visits),
        coupling_visits_per_candidate=K.coupling_visits / max(len(rows), 1),
        n_positive_prefix=len(pos_prefix), n_positive_best=len(pos_rows),
        n_subset_strictly_better=len(strictly_better),
        frac_subset_strictly_better=len(strictly_better) / max(len(rows), 1),
        n_nonprefix_wins=len(nonprefix_wins),
        capacity_prefix_pp=100.0 * sum_prefix / total,
        capacity_best_pp=100.0 * sum_best / total,
        capacity_subset_increment_pp=100.0 * (sum_best - sum_prefix) / total,
        win_rule_counts={k: sum(1 for r in pos_rows if r["win_rule"] == k)
                         for k in ("prefix", "preference", "reachable")},
        exactness_checks=checks, exactness_all_match=bool(n_match == len(checks)),
        top_rows=order_by_gain[:50],
    )
    op = _ROOT / "experiments" / "outputs" / f"proto_H71_stage1_{dataset}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"wrote {op}", flush=True)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — realised gain: sequential application, matched pop budgets
# ──────────────────────────────────────────────────────────────────────────────
def _sequential(g, rank0: np.ndarray, mode: str, max_pops: int, n_passes: int,
                coupling_budget: Optional[int] = None) -> Dict:
    """Sequential best-by-weight application. ``mode`` is ``'prefix'`` or ``'tuck'``.

    ``'prefix'`` is H52's production kernel, unmodified — the CONTROL.
    ``'tuck'`` evaluates the prefix AND rule (b), and applies whichever is better, so it is a
    superset of the control by construction and the difference is exactly what the subset
    freedom buys.
    """
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    w = np.asarray(g.weight, dtype=np.float64)
    n, total = g.n_nodes, g.total_weight

    rank = np.asarray(rank0, dtype=np.int64).copy()
    seq = np.empty(n, dtype=np.int64)
    seq[rank] = np.arange(n, dtype=np.int64)
    K = TuckKernel(g, rank)

    start_score = score_from_order(rank, src, tgt, g.weight)
    predicted_total = 0.0
    popped = applied = applied_subset = 0
    subset_credit = 0.0
    t0 = time.time()

    for _p in range(n_passes):
        if popped >= max_pops:
            break
        back = np.flatnonzero(rank[src] > rank[tgt])
        heap = [(-float(w[e]), int(e)) for e in back]
        heapq.heapify(heap)
        while heap and popped < max_pops:
            _, ei = heapq.heappop(heap)
            popped += 1
            u = int(src[ei]); v = int(tgt[ei])
            if rank[u] <= rank[v]:
                continue
            lo = int(rank[v]); hi = int(rank[u])
            g_a, cut, _lo, _hi = pair_move_gain(rank, u, v, K.out_ptr, K.out_idx, K.out_w,
                                                K.in_ptr, K.in_idx, K.in_w)
            use_gamma: Optional[Set[int]] = None
            gain = g_a
            if mode == "tuck" and (coupling_budget is None or K.coupling_visits < coupling_budget):
                # BOTH subset rules, as pre-registered. Stage 1 measured that on connectome
                # rule (b) is almost always far WORSE than the prefix (the coupling term it
                # ignores dominates) while rule (c) is the one that wins, so evaluating only
                # (b) here would be a strawman of the hypothesis.
                a, b, w_uv, w_vu = K.endpoint_terms(u, v, lo, hi)
                b_total = 0.0
                for z in b:
                    b_total += b[z]
                base_uv = w_uv - w_vu
                for gam in (K.gamma_preference(a, b), K.gamma_reachable(u, lo, hi)):
                    if gam is None:
                        continue
                    g_s = K.gain_of_gamma(gam, a, b, base=base_uv, b_total=b_total,
                                          lo=lo, hi=hi)
                    if g_s > gain + 1e-9:
                        gain, use_gamma = g_s, gam
            if gain <= 1e-9:
                continue
            if use_gamma is None:
                use_gamma = {int(x) for x in seq[lo + 1:cut + 1]}
            else:
                applied_subset += 1
                subset_credit += gain - max(g_a, 0.0)
            TuckKernel.apply_gamma(seq, rank, u, v, use_gamma, lo, hi)
            applied += 1
            predicted_total += gain

    wall = time.time() - t0
    end_score = score_from_order(rank, src, tgt, g.weight)
    realised = float(end_score - start_score)
    return dict(mode=mode, max_pops=max_pops, n_passes=n_passes,
                popped=popped, applied=applied, applied_subset=applied_subset,
                subset_credit_pp=100.0 * subset_credit / total,
                predicted_pp=100.0 * predicted_total / total,
                realised_pp=100.0 * realised / total,
                exact=bool(abs(realised - predicted_total) < 1e-6),
                start_pct=pct(start_score, total), end_pct=pct(end_score, total),
                wall_s=wall, pp_per_s=(100.0 * realised / total) / max(wall, 1e-9),
                coupling_edge_visits=int(K.coupling_visits),
                rank=rank)


def stage2(dataset: str, max_pops: int, n_passes: int, arms: List[str]) -> int:
    g = io.load_dataset(dataset)
    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    total = g.total_weight
    rank, variant, pos_path, sota_pct = load_champion_order(dataset)
    base_score = score_from_order(rank, src, tgt, g.weight)
    base_pct = pct(base_score, total)
    if abs(base_pct - sota_pct) > 1e-9:
        raise SystemExit(f"champion order scores {base_pct!r}, sota.json says {sota_pct!r}")
    print(f"champion {variant}/{dataset}: {base_pct:.11f}", flush=True)

    res: Dict[str, Dict] = {}
    if "prefix" in arms:
        print("ARM prefix (H52 control) ...", flush=True)
        res["prefix"] = _sequential(g, rank, "prefix", max_pops, n_passes)
        print(f"  {res['prefix']['realised_pp']:+.6f} pp  "
              f"{res['prefix']['applied']:,} moves  {res['prefix']['wall_s']:.0f}s", flush=True)
    if "tuck" in arms:
        print("ARM tuck (prefix + subset, best of the two per candidate) ...", flush=True)
        res["tuck"] = _sequential(g, rank, "tuck", max_pops, n_passes)
        print(f"  {res['tuck']['realised_pp']:+.6f} pp  {res['tuck']['applied']:,} moves "
              f"({res['tuck']['applied_subset']:,} subset)  {res['tuck']['wall_s']:.0f}s",
              flush=True)
    if "composed" in arms and "prefix" in res:
        print("ARM composed (prefix, then tuck on its output) ...", flush=True)
        res["composed"] = _sequential(g, res["prefix"]["rank"], "tuck", max_pops, n_passes)
        print(f"  {res['composed']['realised_pp']:+.6f} pp on top of prefix", flush=True)

    out = dict(item="H71", stage=2, dataset=dataset,
               champion_variant=variant, champion_positions=pos_path,
               champion_pct=base_pct, max_pops=max_pops, n_passes=n_passes,
               arms={k: {kk: vv for kk, vv in v.items() if kk != "rank"}
                     for k, v in res.items()})
    if "prefix" in res and "tuck" in res:
        inc = res["tuck"]["realised_pp"] - res["prefix"]["realised_pp"]
        out["subset_increment_over_prefix_pp"] = inc
        print(f"SUBSET INCREMENT over the prefix control: {inc:+.6f} pp", flush=True)
    if "composed" in res and "tuck" in res:
        # M8 redundancy: what the tuck finds ALONE versus what it still finds after the
        # prefix class has taken its ground.
        solo = res["tuck"]["realised_pp"] - res["prefix"]["realised_pp"]
        after = res["composed"]["realised_pp"]
        out["redundancy_fraction"] = (1.0 - after / solo) if abs(solo) > 1e-12 else None
        out["composed_total_pp"] = res["prefix"]["realised_pp"] + after
    op = _ROOT / "experiments" / "outputs" / f"proto_H71_stage2_{dataset}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"wrote {op}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--dataset", default="connectome")
    ap.add_argument("--topk", type=int, default=2000)
    ap.add_argument("--max-pops", type=int, default=200_000)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--arms", default="prefix,tuck,composed")
    a = ap.parse_args()
    if a.stage == 1:
        return stage1(a.dataset, a.topk)
    return stage2(a.dataset, a.max_pops, a.passes, [s for s in a.arms.split(",") if s])


if __name__ == "__main__":
    raise SystemExit(main())
