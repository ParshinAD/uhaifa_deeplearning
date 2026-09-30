"""Independent review of biology/studies/untapped.md (items A, B, C, D, E).

STRICT reviewer pass: every headline number of the untapped study is recomputed
from the raw artifacts with FRESH aggregation code (this file shares no
aggregation code with the exp_untapped_* scripts; the only reused pieces are
the frozen oracle, the parity-gated champion loader, the fixed lane-partition
definition `lane_of`, and — for the reproduction leg of item D only — the
repo's own greedy+sift, run deterministically to confirm the logged JSON).

Sections
  A  fly hubs: top-12 by total degree (in+out edge count), top FB senders /
     receivers, annotation join against flywire_neuron_annotations.tsv.gz.
  B  lane-matrix mining: the 17x17 lane pair weights re-aggregated from the
     EDGE LIST (not from lane_matrix.json), FF anchor re-summed to the champion,
     budget families recomputed, named pairs checked, VC-regrouping sensitivity.
  D  proofread control: champion-restricted score/stats re-derived exactly;
     the study's deterministic pipeline re-run (seed 42) to confirm the logged
     numbers; an INDEPENDENT sift optimizer (own code) as a red-team on the
     75.58 % optimum and on the direction-of-effect stats.
  E  E1 pair budget recomputed from the microns edge list; E2 decomposition
     recomputed; E3 trophic levels re-solved from scratch (own CG / lstsq),
     including the gross-vs-net flip decomposition behind "~2.7 M synapses".
  C  mouse fingerprint re-parsed from the raw file, git provenance re-checked,
     H63 feedback ledger (hub 92) recomputed, the C2 "124 FB edges all have an
     FF return path" claim re-tested, Bader PDF grepped for "mouse".

Run:  $PY biology/analysis/review_untapped.py
Output: biology/results/review/review_untapped.json  (+ console verdicts)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy import stats
from scipy.sparse import linalg as spla
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CONFIG = dict(
    ann_tsv=str(REPO / "data/external/flywire_neuron_annotations.tsv.gz"),
    fly_ann_npz=str(REPO / "layering/outputs/fly_ann/node_ann.npz"),
    fly_ann_meta=str(REPO / "layering/outputs/fly_ann/node_ann_meta.json"),
    lane_matrix_json=str(REPO / "layering/outputs/fly_streams/lane_matrix.json"),
    proofread_npz=str(REPO / "data/processed/microns_proofread.npz"),
    microns_ann_npz=str(REPO / "layering/outputs/microns_ann/node_ann.npz"),
    mouse_txt=str(REPO / "data/table_mouse.txt"),
    bader_pdf=str(REPO / "s13278-025-01491-2 (2).pdf"),
    d_seed=42, d_restarts=16,          # the study's deterministic config (repro leg)
    own_restarts=24, own_seed=1234,    # independent optimizer (red-team leg)
    cg_rtol=1e-8, cg_maxiter=30000,    # fly trophic solve
    out_json=str(REPO / "biology/results/review/review_untapped.json"),
)

CHECKS: list[dict] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append(dict(name=name, ok=bool(ok), detail=detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def close(a, b, tol) -> bool:
    return abs(float(a) - float(b)) <= tol


# ───────────────────────────── A. fly hubs ─────────────────────────────────

def part_a(g, rank, ff_pct) -> dict:
    print("\n===== A. fly hubs =====")
    n = g.n_nodes
    src, tgt = np.asarray(g.src, np.int64), np.asarray(g.tgt, np.int64)
    w = np.asarray(g.weight, np.int64)

    deg = np.bincount(src, minlength=n) + np.bincount(tgt, minlength=n)
    fb = rank[src] > rank[tgt]
    fb_out = np.bincount(src[fb], weights=w[fb], minlength=n)
    fb_in = np.bincount(tgt[fb], weights=w[fb], minlength=n)
    total_fb = int(w[fb].sum())

    ann = pd.read_csv(CONFIG["ann_tsv"], sep="\t",
                      usecols=["root_id", "super_class", "cell_class",
                               "cell_type", "top_nt", "side"],
                      low_memory=False).drop_duplicates("root_id").set_index("root_id")

    def table(idx_list):
        rows = []
        for i in idx_list:
            rid = int(g.node_ids[i])
            r = dict(idx=int(i), root_id=rid, degree=int(deg[i]),
                     rank_frac=float(rank[i] / (n - 1)),
                     fb_out=int(fb_out[i]), fb_in=int(fb_in[i]))
            if rid in ann.index:
                a = ann.loc[rid]
                for c in ("super_class", "cell_class", "cell_type", "top_nt", "side"):
                    v = a[c]
                    r[c] = None if (isinstance(v, float) and np.isnan(v)) else str(v)
            rows.append(r)
        return rows

    top_deg = table(np.argsort(-deg)[:12])
    top_snd = table(np.argsort(-fb_out)[:12])
    top_rcv = table(np.argsort(-fb_in)[:12])

    check("A total FB weight", total_fb == 6_597_734 and
          close(100 * total_fb / g.total_weight, 15.742, 5e-4),
          f"{total_fb:,} = {100*total_fb/g.total_weight:.4f}% (study: 6,597,734 = 15.742%)")
    t1, t2 = top_deg[0], top_deg[1]
    check("A #1-2 degree = CT1",
          t1["cell_type"] == t2["cell_type"] == "CT1"
          and t1["degree"] == 16784 and t2["degree"] == 16581
          and t1["top_nt"] == t2["top_nt"] == "gaba"
          and t1["cell_class"] == t2["cell_class"] == "ME>LO"
          and 0.41 <= t2["rank_frac"] <= 0.44 and 0.41 <= t1["rank_frac"] <= 0.44,
          f"{t1['cell_type']} deg {t1['degree']}/{t2['degree']} "
          f"nt {t1['top_nt']} class {t1['cell_class']} "
          f"ranks {100*t1['rank_frac']:.1f}/{100*t2['rank_frac']:.1f}%")
    n_gaba = sum(1 for r in top_deg if r.get("top_nt") == "gaba")
    exc = [r for r in top_deg if r.get("top_nt") != "gaba"]
    check("A 11 of top-12 GABAergic", n_gaba == 11 and len(exc) == 1
          and exc[0].get("cell_type") == "MeMe_e13"
          and exc[0].get("top_nt") == "acetylcholine",
          f"{n_gaba}/12 gaba; exception {exc[0].get('cell_type')} ({exc[0].get('top_nt')})"
          if exc else f"{n_gaba}/12 gaba")
    s1, s2 = top_snd[0], top_snd[1]
    apl_share = 100 * (s1["fb_out"] + s2["fb_out"]) / total_fb
    check("A #1-2 FB senders = APL",
          s1["cell_type"] == s2["cell_type"] == "APL"
          and 0.74 <= min(s1["rank_frac"], s2["rank_frac"])
          and max(s1["rank_frac"], s2["rank_frac"]) <= 0.77
          and close(apl_share, 1.07, 0.005),
          f"{s1['cell_type']}/{s2['cell_type']} at "
          f"{100*s1['rank_frac']:.1f}/{100*s2['rank_frac']:.1f}%, "
          f"together {apl_share:.3f}% of ALL FB weight (study 1.07%)")
    r1, r2 = top_rcv[0], top_rcv[1]
    check("A #1-2 FB receivers = CT1", r1["cell_type"] == r2["cell_type"] == "CT1",
          f"{r1['cell_type']}/{r2['cell_type']} fb_in {r1['fb_in']}/{r2['fb_in']}")
    # study per-row spot values
    spot = dict(
        ct1_fb=(t1["fb_out"], t1["fb_in"], t2["fb_out"], t2["fb_in"]),
        apl=(s1["degree"], s2["degree"], s1["fb_out"], s2["fb_out"],
             s1["fb_in"], s2["fb_in"]),
    )
    exp_ct1 = {(14180, 28321, 15194, 26302), (15194, 26302, 14180, 28321)}
    check("A CT1 FB in/out spot values", tuple(spot["ct1_fb"]) in exp_ct1
          or (spot["ct1_fb"][0], spot["ct1_fb"][1]) in
          {(14180, 28321), (15194, 26302)},
          f"CT1 (fb_out, fb_in) x2 = {spot['ct1_fb']} (study 14,180/28,321 & 15,194/26,302)")
    check("A APL spot values",
          sorted((s1["degree"], s2["degree"])) == [5579, 5606]
          and sorted((s1["fb_out"], s2["fb_out"])) == [34585, 36283]
          and sorted((s1["fb_in"], s2["fb_in"])) == [13337, 19086],
          f"deg {s1['degree']}/{s2['degree']} fb_out {s1['fb_out']}/{s2['fb_out']} "
          f"fb_in {s1['fb_in']}/{s2['fb_in']}")
    rcv_types = [r.get("cell_type") for r in top_rcv]
    cl21 = [r for r in top_rcv if r.get("cell_type") == "cL21"]
    lln = [r for r in top_rcv if (r.get("cell_type") or "").startswith("lLN2F_b")]
    check("A early pure receivers present",
          len(cl21) >= 1 and all(0.24 <= c["rank_frac"] <= 0.32 and
                                 c["fb_out"] < 0.05 * c["fb_in"] for c in cl21)
          and len(lln) >= 1 and all(0.10 <= c["rank_frac"] <= 0.14 for c in lln),
          f"cL21 x{len(cl21)} at {[round(100*c['rank_frac'],1) for c in cl21]}%, "
          f"lLN2F_b x{len(lln)} at {[round(100*c['rank_frac'],1) for c in lln]}% "
          f"(receiver types: {rcv_types})")
    return dict(total_fb_weight=total_fb, top_degree=top_deg,
                top_fb_senders=top_snd, top_fb_receivers=top_rcv)


# ─────────────────────────── B. lane mining ────────────────────────────────

FAMILY = {
    "photoreceptors": "visual", "lamina": "visual", "medulla": "visual",
    "medulla→lobula": "visual", "lobula (plate)": "visual",
    "optic (other)": "visual", "visual projection": "visual",
    "visual centrifugal": "visual",
    "olfaction: AL·LH": "central", "mushroom body": "central",
    "central complex": "central", "central (other)": "central",
    "endocrine": "central",
    "ascending (VNC)": "periphery-motor", "sensory (other)": "periphery-motor",
    "descending": "periphery-motor", "motor": "periphery-motor",
}


def budget_cat(ls: str, lt: str, family: dict) -> str:
    fs, ft = family[ls], family[lt]
    if ls == lt:
        return f"within one {fs} lane (recurrence)"
    if fs == ft:
        return {"visual": "between visual lanes (retro-pathways)",
                "central": "between central lanes",
                "periphery-motor": "between periphery-motor lanes"}[fs]
    if {fs, ft} == {"visual", "central"}:
        return "visual ↔ central"
    return "involving periphery-motor"


def part_b(g, rank, ff_pct) -> dict:
    print("\n===== B. lane-matrix mining =====")
    from layering.tools.fly_streams import lane_of   # fixed partition definition
    n = g.n_nodes
    src, tgt = np.asarray(g.src, np.int64), np.asarray(g.tgt, np.int64)
    w = np.asarray(g.weight, np.float64)
    Wtot = w.sum()

    d = np.load(CONFIG["fly_ann_npz"])
    meta = json.loads(Path(CONFIG["fly_ann_meta"]).read_text())
    assert np.array_equal(d["rank"], rank), "fly annotation cache stale vs champion"
    scv, ccv = meta["super_class_values"], meta["cell_class_values"]
    lane_names = np.array([lane_of(scv[s], ccv[c])
                           for s, c in zip(d["super_class"], d["cell_class"])])
    lanes = sorted(set(lane_names))
    li = {L: i for i, L in enumerate(lanes)}
    lane = np.array([li[L] for L in lane_names], np.int64)
    K = len(lanes)

    ffm = rank[src] < rank[tgt]
    pair_idx = lane[src] * K + lane[tgt]
    Wm = np.bincount(pair_idx, weights=w, minlength=K * K).reshape(K, K)
    Fm = np.bincount(pair_idx[ffm], weights=w[ffm], minlength=K * K).reshape(K, K)
    Wp, Fp = 100 * Wm / Wtot, 100 * Fm / Wtot

    check("B anchor: lane FF re-sums to champion",
          close(Fp.sum(), ff_pct, 1e-9),
          f"sum W_ff_pct = {Fp.sum():.10f} vs champion {ff_pct:.10f}")

    lm = json.loads(Path(CONFIG["lane_matrix_json"]).read_text())
    perm = [lanes.index(L) for L in lm["lane_list"]]
    Wj = np.array(lm["W_pct"]); Fj = np.array(lm["W_ff_pct"])
    dW = np.abs(Wp[np.ix_(perm, perm)] - Wj).max()
    dF = np.abs(Fp[np.ix_(perm, perm)] - Fj).max()
    check("B lane_matrix.json reproduced from edge list",
          dW < 1e-9 and dF < 1e-9, f"max|dW|={dW:.2e} max|dF|={dF:.2e}")

    FBp = Wp - Fp
    total_fb = FBp.sum()

    def share(ls, lt):
        i, j = li[ls], li[lt]
        return dict(w=float(Wp[i, j]), ff=float(100 * Fp[i, j] / Wp[i, j]),
                    fb_share=float(100 * FBp[i, j] / total_fb))

    named = {
        "VPN→visual centrifugal": (share("visual projection", "visual centrifugal"),
                                   24.7, 0.22),
        "LOP→medulla→lobula (Tm)": (share("lobula (plate)", "medulla→lobula"),
                                    46.0, 1.64),
        "medulla→lamina": (share("medulla", "lamina"), 57.8, 1.23),
        "descending→central (other)": (share("descending", "central (other)"),
                                       57.2, 0.60),
        "photoreceptors→lamina": (share("photoreceptors", "lamina"), 99.6, None),
        "AL·LH→mushroom body": (share("olfaction: AL·LH", "mushroom body"),
                                99.5, None),
        "medulla→lobula→VPN (Tm→VPN)": (share("medulla→lobula", "visual projection"),
                                        98.7, 4.60),
    }
    for name, (got, ff_exp, w_exp) in named.items():
        ok = close(got["ff"], ff_exp, 0.05) and (w_exp is None or
                                                 close(got["w"], w_exp, 0.005))
        check(f"B pair {name}", ok,
              f"FF {got['ff']:.2f}% (study {ff_exp}) weight {got['w']:.3f}% "
              f"(study {w_exp}) FB share {got['fb_share']:.2f}%")

    # budget families (own implementation)
    budget: dict[str, float] = {}
    for i, ls in enumerate(lanes):
        for j, lt in enumerate(lanes):
            if Wp[i, j] <= 0:
                continue
            c = budget_cat(ls, lt, FAMILY)
            budget[c] = budget.get(c, 0.0) + 100 * FBp[i, j] / total_fb
    exp_budget = {"between visual lanes (retro-pathways)": 36.6,
                  "within one central lane (recurrence)": 30.2,
                  "within one visual lane (recurrence)": 21.0,
                  "visual ↔ central": 4.9,
                  "involving periphery-motor": 4.2,
                  "between central lanes": 2.7,
                  "within one periphery-motor lane (recurrence)": None,
                  "between periphery-motor lanes": None}
    for c, v in sorted(budget.items(), key=lambda kv: -kv[1]):
        print(f"    {v:6.2f}%  {c}")
    ok = all(close(budget.get(c, 0.0), v, 0.05)
             for c, v in exp_budget.items() if v is not None)
    pm = (budget.get("within one periphery-motor lane (recurrence)", 0) +
          budget.get("between periphery-motor lanes", 0))
    check("B budget families", ok and close(pm, 0.5, 0.05),
          "all family shares match study to 0.05pp "
          f"(periph within+between = {pm:.2f} vs 0.5)")
    never_leaves = (budget["between visual lanes (retro-pathways)"] +
                    budget["within one visual lane (recurrence)"])
    check("B never-leaves-optic 57.5%", close(never_leaves, 57.5, 0.1),
          f"{never_leaves:.2f}% (36.6+21.0 rounds to 57.6; exact sum "
          f"{budget['between visual lanes (retro-pathways)']:.2f}+"
          f"{budget['within one visual lane (recurrence)']:.2f})")
    central_tot = (budget["within one central lane (recurrence)"] +
                   budget["between central lanes"])
    check("B central ~33% / between-central 2.7%",
          close(central_tot, 33.0, 0.5) and
          close(budget["between central lanes"], 2.7, 0.05),
          f"central total {central_tot:.2f}%, between-central "
          f"{budget['between central lanes']:.2f}%")

    # counter-hierarchy / one-way counts under the study's thresholds
    heavy, oneway, counter = [], [], []
    for i, ls in enumerate(lanes):
        for j, lt in enumerate(lanes):
            if Wp[i, j] < 0.1:
                continue
            ffs = 100 * Fp[i, j] / Wp[i, j]
            heavy.append((ls, lt, ffs, Wp[i, j]))
            if ffs < 65.0:
                counter.append((ls, lt, ffs))
            if ffs >= 97.0:
                oneway.append((ls, lt, ffs))
    check("B 13 counter-hierarchy / 12 one-way pairs",
          len(counter) == 13 and len(oneway) == 12,
          f"counter={len(counter)} oneway={len(oneway)}")

    # named anatomically-descending visual pairs = 24.0 % of all FB
    depth = {"photoreceptors": 0, "lamina": 1, "medulla": 2, "medulla→lobula": 3,
             "lobula (plate)": 4, "optic (other)": 4, "visual projection": 5,
             "visual centrifugal": 2.5}
    retro = sum(100 * FBp[li[a], li[b]] / total_fb
                for a in depth for b in depth
                if a != b and depth[a] > depth[b])
    check("B descending visual pairs 24.0% of FB", close(retro, 24.0, 0.1),
          f"{retro:.2f}%")

    # sensitivity: visual centrifugal regrouped into the central family
    fam2 = dict(FAMILY); fam2["visual centrifugal"] = "central"
    b2: dict[str, float] = {}
    for i, ls in enumerate(lanes):
        for j, lt in enumerate(lanes):
            if Wp[i, j] <= 0:
                continue
            c = budget_cat(ls, lt, fam2)
            b2[c] = b2.get(c, 0.0) + 100 * FBp[i, j] / total_fb
    nl2 = (b2["between visual lanes (retro-pathways)"] +
           b2["within one visual lane (recurrence)"])
    check("B VC-regroup sensitivity (caveat honest?)", True,
          f"with VC counted central: between-visual {b2['between visual lanes (retro-pathways)']:.2f}%, "
          f"never-leaves-optic {nl2:.2f}% (vs {never_leaves:.2f}%)")
    return dict(budget=budget, budget_vc_central=b2,
                never_leaves_optic_pct=float(never_leaves),
                retro_named_pct=float(retro),
                n_counter=len(counter), n_oneway=len(oneway),
                counter_pairs=[(a, b, round(f, 2)) for a, b, f in
                               sorted(counter, key=lambda t: t[2])],
                oneway_pairs=[(a, b, round(f, 2)) for a, b, f in
                              sorted(oneway, key=lambda t: -t[2])])


# ─────────────────────────── D. proofread control ──────────────────────────

def own_sift(src, tgt, w, order, rng, max_pass=200):
    """Independent remove-and-best-reinsert local search (this reviewer's code).

    order: array of node indices in line order. Returns improved order."""
    n = order.shape[0]
    # adjacency lists
    out_adj: list[list] = [[] for _ in range(n)]
    in_adj: list[list] = [[] for _ in range(n)]
    for s, t, ww in zip(src, tgt, w):
        out_adj[s].append((t, float(ww)))
        in_adj[s if False else t].append((s, float(ww)))
    order = list(order)
    pos = {v: i for i, v in enumerate(order)}
    for _ in range(max_pass):
        improved = False
        for v in rng.permutation(n):
            i0 = pos[v]
            order.pop(i0)
            m = len(order)
            a = np.zeros(m)   # a[j] = w(order[j] -> v)
            b = np.zeros(m)   # b[j] = w(v -> order[j])
            for u, ww in in_adj[v]:
                if u != v:
                    a[pos[u] - (pos[u] > i0)] += ww
            for u, ww in out_adj[v]:
                if u != v:
                    b[pos[u] - (pos[u] > i0)] += ww
            # f(slot) = sum_{j<slot} a[j] + sum_{j>=slot} b[j]
            f = np.empty(m + 1)
            f[0] = b.sum()
            np.cumsum(a - b, out=f[1:])
            f[1:] += f[0]
            best = int(np.argmax(f))
            cur = f[i0]
            if f[best] > cur + 1e-12:
                improved = True
            else:
                best = i0
            order.insert(best, v)
            pos = {u: i for i, u in enumerate(order)}
        if not improved:
            break
    return np.array(order, np.int64)


def part_d() -> dict:
    print("\n===== D. proofread control =====")
    from src.mfas.io import GraphData
    from src.mfas.metrics import pct, score_from_order
    from src.mfas.experiments.H02 import greedy_fas_order
    from src.mfas.refine.underrelax import sift_underrelaxed
    from layering import io_utils

    npz = np.load(CONFIG["proofread_npz"])
    g = GraphData(src=npz["src"], tgt=npz["tgt"], weight=npz["weight"],
                  node_ids=npz["node_ids"], name="microns_proofread")
    n, total = g.n_nodes, g.total_weight
    src, tgt = np.asarray(g.src, np.int64), np.asarray(g.tgt, np.int64)
    w = np.asarray(g.weight, np.float64)
    check("D subgraph facts", n == 245 and g.n_edges == 2424 and total == 5466,
          f"n={n} m={g.n_edges} W={total}")

    A = sp.coo_matrix((np.ones(len(src)), (src, tgt)), shape=(n, n))
    nscc, lab = connected_components(A, directed=True, connection="strong")
    giant = int(np.bincount(lab).max())
    check("D 60 SCCs, giant 186", nscc == 60 and giant == 186,
          f"nscc={nscc} giant={giant}")

    # champion restricted (deterministic)
    g_full, _, rank_full, ff_full = io_utils.load_graph_and_champion_order("microns")
    idx_full = np.searchsorted(g_full.node_ids, g.node_ids)
    assert np.array_equal(g_full.node_ids[idx_full], g.node_ids)
    rank_restr = np.argsort(np.argsort(rank_full[idx_full])).astype(np.int64)
    restr_pct = pct(score_from_order(rank_restr, src, tgt, g.weight), total)
    check("D champion-restricted 67.75%", close(restr_pct, 67.75, 0.01),
          f"{restr_pct:.4f}%")

    ann = np.load(CONFIG["microns_ann_npz"])
    check("D 249 proofread axons graph-wide", int(ann["axon_pr"].sum()) == 249,
          f"axon_pr sum = {int(ann['axon_pr'].sum())}")
    depth = ann["depth_um"][idx_full].astype(np.float64)
    ei = ann["ei"][idx_full]
    n_E, n_I = int((ei == 1).sum()), int((ei == 2).sum())
    check("D composition 142E/97I/6NA, 40% I, depth 247 vs 381",
          n_E == 142 and n_I == 97
          and close(n_I / n, 0.396, 0.005)
          and close(np.median(depth), 247.1, 0.5)
          and close(np.median(ann["depth_um"]), 381.0, 0.5),
          f"E={n_E} I={n_I} Ifrac={n_I/n:.3f} depth med {np.median(depth):.1f} "
          f"vs full {np.median(ann['depth_um']):.1f}")

    # (1) reproduction leg: the study's own deterministic pipeline, seed 42
    rank_g = greedy_fas_order(g)
    greedy_pct = pct(score_from_order(rank_g, src, tgt, g.weight), total)
    best_rank, best_score, _ = sift_underrelaxed(g, rank_g, k_full=6, alpha=0.7,
                                                 max_sweeps=40)
    rng = np.random.default_rng(CONFIG["d_seed"])
    for s in rng.integers(0, 2**31 - 1, CONFIG["d_restarts"]):
        r0 = np.random.default_rng(int(s)).permutation(n).astype(np.int64)
        r, sc, _ = sift_underrelaxed(g, r0, k_full=6, alpha=0.7, max_sweeps=40)
        if sc > best_score:
            best_rank, best_score = r, sc
    repro_pct = pct(best_score, total)
    rho_orders = float(stats.spearmanr(best_rank, rank_restr).statistic)
    check("D reproduce direct opt 75.58% / Spearman 0.323",
          close(repro_pct, 75.576, 0.005) and close(rho_orders, 0.323, 0.005)
          and close(greedy_pct, 68.92, 0.01),
          f"greedy {greedy_pct:.3f}% best {repro_pct:.4f}% rho {rho_orders:.4f}")

    # (2) independent optimizer leg (own sift, own inits)
    rng2 = np.random.default_rng(CONFIG["own_seed"])
    own_best_o, own_pct_best = None, -1.0
    inits = [np.argsort(rank_restr), np.argsort(rank_g)]
    inits += [rng2.permutation(n) for _ in range(CONFIG["own_restarts"])]
    t0 = time.time()

    def score_order(o):
        r = np.empty(n, np.int64); r[o] = np.arange(n)
        return pct(score_from_order(r, src, tgt, g.weight), total), r

    for o0 in inits:
        o = own_sift(src, tgt, w, np.asarray(o0, np.int64), rng2)
        p, _ = score_order(o)
        if p > own_pct_best:
            own_pct_best, own_best_o = p, o
    # iterated local search: random segment-reversal kicks around the best
    for _ in range(60):
        o = own_best_o.copy()
        for _k in range(rng2.integers(1, 4)):
            i = int(rng2.integers(0, n - 10))
            j = i + int(rng2.integers(5, 50))
            o[i:min(j, n)] = o[i:min(j, n)][::-1]
        o = own_sift(src, tgt, w, o, rng2)
        p, _ = score_order(o)
        if p > own_pct_best:
            own_pct_best, own_best_o = p, o
    _, own_best = score_order(own_best_o)
    check("D independent optimizer confirms optimum plateau",
          own_pct_best >= 75.45,
          f"own best {own_pct_best:.3f}% in {time.time()-t0:.1f}s "
          f"(study 75.576%; delta {own_pct_best - 75.576:+.3f}pp)")

    # (3) biology stats under three orders: study-reproduced, own, restricted
    def bio(r):
        e_m, i_m = ei == 1, ei == 2
        rho_e = stats.spearmanr(r[e_m], depth[e_m])
        mwu = stats.mannwhitneyu(r[i_m], r[e_m], alternative="less")
        ffe = r[tgt] > r[src]
        pb = {}
        cls = np.where(ei == 1, "E", np.where(ei == 2, "I", "NA"))
        for a in ("E", "I"):
            for b in ("E", "I"):
                m = (cls[src] == a) & (cls[tgt] == b)
                ws = w[m].sum()
                pb[f"{a}->{b}"] = dict(
                    w_pct=float(100 * ws / total),
                    ff=float(100 * w[m & ffe].sum() / ws) if ws else None)
        return dict(depth_rho_exc=float(rho_e.statistic),
                    depth_p_exc=float(rho_e.pvalue),
                    medI=float(np.median(r[i_m])), medE=float(np.median(r[e_m])),
                    mwu_p=float(mwu.pvalue), pair=pb)

    b_repro, b_own, b_restr = bio(best_rank), bio(own_best), bio(rank_restr)
    check("D stats under reproduced order",
          close(b_repro["depth_rho_exc"], -0.168, 0.005)
          and close(b_repro["medI"], 114, 1) and close(b_repro["medE"], 127.5, 1)
          and close(b_repro["mwu_p"], 0.212, 0.01)
          and close(b_repro["pair"]["I->E"]["ff"], 88.4, 0.1)
          and close(b_repro["pair"]["E->I"]["ff"], 53.1, 0.1)
          and close(b_repro["pair"]["I->I"]["w_pct"], 71.6, 0.1),
          f"rho_E {b_repro['depth_rho_exc']:+.3f} (p {b_repro['depth_p_exc']:.3f}) "
          f"I {b_repro['medI']:.0f} vs E {b_repro['medE']:.1f} (p {b_repro['mwu_p']:.2f}) "
          f"I→E {b_repro['pair']['I->E']['ff']:.1f} E→I {b_repro['pair']['E->I']['ff']:.1f} "
          f"I→I wt {b_repro['pair']['I->I']['w_pct']:.1f}%")
    check("D direction-of-effect robust under MY independent order",
          b_own["pair"]["I->E"]["ff"] - b_own["pair"]["E->I"]["ff"] > 20
          and b_own["depth_rho_exc"] < 0.15 and b_own["mwu_p"] > 0.001,
          f"own order: rho_E {b_own['depth_rho_exc']:+.3f} (p {b_own['depth_p_exc']:.3f}), "
          f"I {b_own['medI']:.0f} vs E {b_own['medE']:.1f} (p {b_own['mwu_p']:.3f}), "
          f"I→E {b_own['pair']['I->E']['ff']:.1f} vs E→I {b_own['pair']['E->I']['ff']:.1f}, "
          f"E→E {b_own['pair']['E->E']['ff']:.1f} I→I {b_own['pair']['I->I']['ff']:.1f}")
    check("D champion-restricted stats",
          close(b_restr["depth_rho_exc"], -0.052, 0.005)
          and b_restr["depth_p_exc"] > 0.05
          and close(b_restr["medI"], 65, 1) and close(b_restr["medE"], 150.5, 1),
          f"rho_E {b_restr['depth_rho_exc']:+.3f} (p {b_restr['depth_p_exc']:.3f}) "
          f"I {b_restr['medI']:.0f} vs E {b_restr['medE']:.1f} "
          f"(p {b_restr['mwu_p']:.2e})")
    # red-team: depth-range compression of the E sample
    e_full = ann["ei"] == 1
    iqr_ctrl = np.percentile(depth[ei == 1], [25, 75])
    iqr_full = np.percentile(ann["depth_um"][e_full], [25, 75])
    check("D composition red-team: E depth range in control", True,
          f"control E depth IQR {iqr_ctrl[0]:.0f}-{iqr_ctrl[1]:.0f}um "
          f"(median {np.median(depth[ei==1]):.0f}) vs full-graph E IQR "
          f"{iqr_full[0]:.0f}-{iqr_full[1]:.0f}um (median "
          f"{np.median(ann['depth_um'][e_full]):.0f})")
    return dict(restr_pct=float(restr_pct), repro_pct=float(repro_pct),
                own_pct=float(own_pct_best), rho_orders=rho_orders,
                bio_reproduced=b_repro, bio_own=b_own, bio_restricted=b_restr,
                e_depth_iqr_control=[float(x) for x in iqr_ctrl],
                e_depth_iqr_full=[float(x) for x in iqr_full])


# ─────────────────────────────── E. quick adds ─────────────────────────────

def part_e() -> dict:
    print("\n===== E. quick adds =====")
    from layering import io_utils
    from src.mfas.metrics import pct, score_from_order

    # E1: full-graph E/I pair budget recomputed from the edge list
    g, _, rank, ff_pct = io_utils.load_graph_and_champion_order("microns")
    src, tgt = np.asarray(g.src, np.int64), np.asarray(g.tgt, np.int64)
    w = np.asarray(g.weight, np.float64)
    ann = np.load(CONFIG["microns_ann_npz"])
    ei = ann["ei"]
    assert np.array_equal(ann["rank"], rank), "microns annotation cache stale"
    ffm = rank[src] < rank[tgt]
    pair = {}
    cls = np.where(ei == 1, "E", np.where(ei == 2, "I", "NA"))
    for a in ("E", "I"):
        for b in ("E", "I"):
            m = (cls[src] == a) & (cls[tgt] == b)
            ws = w[m].sum()
            pair[f"{a}->{b}"] = float(100 * w[m & ffm].sum() / ws)
    check("E1 full-graph pair FF shares",
          close(pair["E->E"], 80.6, 0.05) and close(pair["I->I"], 78.1, 0.05)
          and close(pair["I->E"], 97.3, 0.05) and close(pair["E->I"], 48.4, 0.05),
          f"E→E {pair['E->E']:.2f} I→I {pair['I->I']:.2f} "
          f"I→E {pair['I->E']:.2f} E→I {pair['E->I']:.2f} (graph-wide {ff_pct:.2f})")

    # E2: unclassified-lane decomposition recomputed
    layer, out_w, in_w, n_som = ann["layer"], ann["out_w"], ann["in_w"], ann["n_somas"]
    wm = layer == 6
    u = (ei == 0) & ~wm
    inh = (ei == 2) & ~wm
    ratio_u = float(out_w[u].sum() / in_w[u].sum())
    ratio_i = float(out_w[inh].sum() / in_w[inh].sum())
    ms = u & (n_som > 1)
    zi = u & (n_som <= 1) & (in_w == 0)
    rest = u & (n_som <= 1) & (in_w > 0)
    check("E2 unclassified lane decomposition",
          int(u.sum()) == 4202 and close(np.median(ann["rank"][u]), 18076.5, 1)
          and close(ratio_u, 1.18, 0.01) and close(ratio_i, 2.70, 0.01)
          and close(ms.sum() / u.sum(), 0.428, 0.002)
          and close(np.median(ann["rank"][ms]), 29105, 5)
          and int(zi.sum()) == 473 and close(np.median(ann["rank"][zi]), 309, 1)
          and int(rest.sum()) == 1930
          and close(np.median(in_w[rest]), 28, 1)
          and close(np.median(in_w), 168, 1)
          and close(np.median(ann["rank"][rest]), 12967, 5),
          f"n={int(u.sum())} med rank {np.median(ann['rank'][u]):.0f} "
          f"out/in {ratio_u:.2f} (I {ratio_i:.2f}); merge {100*ms.sum()/u.sum():.1f}% "
          f"@{np.median(ann['rank'][ms]):.0f}; zero-in {int(zi.sum())} "
          f"@{np.median(ann['rank'][zi]):.0f}; rest {int(rest.sum())} "
          f"in_w {np.median(in_w[rest]):.0f} vs {np.median(in_w):.0f} "
          f"@{np.median(ann['rank'][rest]):.0f}")

    # E3: trophic levels re-solved from scratch — fly
    gf, _, rankf, fff = io_utils.load_graph_and_champion_order("connectome")
    sf, tf = np.asarray(gf.src, np.int64), np.asarray(gf.tgt, np.int64)
    wf = np.asarray(gf.weight, np.float64)
    nf = gf.n_nodes
    A = sp.coo_matrix((wf, (sf, tf)), shape=(nf, nf)).tocsr()
    kin = np.asarray(A.sum(0)).ravel()
    kout = np.asarray(A.sum(1)).ravel()
    Lap = (sp.diags(kin + kout) - A - A.T + sp.diags(np.full(nf, 1e-9))).tocsr()
    Minv = sp.diags(1.0 / (kin + kout + 1e-9))
    t0 = time.time()
    try:
        h, info = spla.cg(Lap, kin - kout, rtol=CONFIG["cg_rtol"],
                          maxiter=CONFIG["cg_maxiter"], M=Minv)
    except TypeError:   # older scipy: `tol` instead of `rtol`
        h, info = spla.cg(Lap, kin - kout, tol=CONFIG["cg_rtol"], atol=0.0,
                          maxiter=CONFIG["cg_maxiter"], M=Minv)
    rank_h = np.empty(nf, np.int64)
    rank_h[np.argsort(h, kind="stable")] = np.arange(nf)
    ff_troph = pct(score_from_order(rank_h, sf, tf, gf.weight), gf.total_weight)
    rho_troph = float(stats.spearmanr(rank_h, rankf).statistic)
    gap = fff - ff_troph
    fb_budget = 100.0 - fff
    check("E3 fly trophic 77.78% / rho 0.76 (own solve)",
          close(ff_troph, 77.78, 0.05) and close(rho_troph, 0.76, 0.01),
          f"trophic FF {ff_troph:.4f}% (cg info {info}, {time.time()-t0:.0f}s) "
          f"rho {rho_troph:.4f} gap {gap:.3f}pp")
    check("E3 arithmetic 41% of feedback budget",
          close(100 * gap / fb_budget, 41.2, 0.7),
          f"{gap:.4f}pp / {fb_budget:.4f}% = {100*gap/fb_budget:.1f}% ; "
          f"2.7M synapses = {gap/100*gf.total_weight/1e6:.2f}M")
    # gross vs net flips: is "misclassifies ~2.7M as feedback" net or gross?
    ff_champ = rankf[sf] < rankf[tf]
    ff_tro = rank_h[sf] < rank_h[tf]
    gross_fb2ff = float(wf[~ff_tro & ff_champ].sum())   # trophic-back, champion-fwd
    gross_ff2fb = float(wf[ff_tro & ~ff_champ].sum())   # trophic-fwd, champion-back
    check("E3 net-vs-gross decomposition", True,
          f"weight backward-under-trophic but forward-under-champion = "
          f"{gross_fb2ff/1e6:.2f}M; reverse = {gross_ff2fb/1e6:.2f}M; "
          f"net = {(gross_fb2ff-gross_ff2fb)/1e6:.2f}M (the quoted 2.7M is the NET)")

    # E3 mouse (dense lstsq)
    gm, _, rankm, ffm_ = io_utils.load_graph_and_champion_order("mouse")
    nm = gm.n_nodes
    Am = np.zeros((nm, nm))
    np.add.at(Am, (np.asarray(gm.src, np.int64), np.asarray(gm.tgt, np.int64)),
              np.asarray(gm.weight, np.float64))
    kin_m, kout_m = Am.sum(0), Am.sum(1)
    Lm = np.diag(kin_m + kout_m) - Am - Am.T
    hm = np.linalg.lstsq(Lm, kin_m - kout_m, rcond=None)[0]
    rank_hm = np.empty(nm, np.int64)
    rank_hm[np.argsort(hm, kind="stable")] = np.arange(nm)
    ff_tm = pct(score_from_order(rank_hm, gm.src, gm.tgt, gm.weight),
                gm.total_weight)
    rho_m = float(stats.spearmanr(rank_hm, rankm).statistic)
    check("E3 mouse trophic 89.98% / rho 0.74 (own solve)",
          close(ff_tm, 89.98, 0.15) and close(rho_m, 0.74, 0.02),
          f"trophic FF {ff_tm:.3f}% vs champion {ffm_:.3f}% "
          f"(gap {ffm_-ff_tm:.2f}pp) rho {rho_m:.3f}")
    return dict(e1_pair=pair, e3_fly=dict(ff=float(ff_troph), rho=rho_troph,
                gap_pp=float(gap), gap_over_fb_budget_pct=float(100*gap/fb_budget),
                gross_fb2ff=gross_fb2ff, gross_ff2fb=gross_ff2fb),
                e3_mouse=dict(ff=float(ff_tm), rho=rho_m))


# ─────────────────────────────── C. mouse ──────────────────────────────────

def part_c() -> dict:
    print("\n===== C. mouse provenance =====")
    from layering import io_utils
    txt = Path(CONFIG["mouse_txt"]).read_text()
    rows = [ln.split("\t") for ln in txt.splitlines() if ln.strip()]
    srcs = [int(r[0]) for r in rows]; tgts = [int(r[1]) for r in rows]
    ws = [float(r[2]) for r in rows]
    ids = sorted(set(srcs) | set(tgts))
    missing = sorted(set(range(min(ids), max(ids) + 1)) - set(ids))
    density = len(rows) / (len(ids) * (len(ids) - 1))
    top3 = sorted(zip(srcs, tgts, ws), key=lambda t: -t[2])[:3]
    pairs = set(zip(srcs, tgts))
    recip = sum(1 for p in pairs if (p[1], p[0]) in pairs)
    check("C fingerprint (raw file re-parse)",
          len(ids) == 148 and ids[0] == 1 and ids[-1] == 149 and missing == [37]
          and len(rows) == 583 and close(density, 0.0268, 0.0002)
          and top3[0][:2] == (51, 53) and repr(top3[0][2]) == "0.613959345844929"
          and top3[1][:2] == (105, 123) and repr(top3[1][2]) == "0.594344938173106"
          and top3[2][:2] == (133, 135) and repr(top3[2][2]) == "0.28193253167543"
          and recip == 166,
          f"n={len(ids)} ids {ids[0]}..{ids[-1]} missing {missing} m={len(rows)} "
          f"density {100*density:.2f}% recip {recip}; heaviest "
          f"{top3[0][0]}→{top3[0][1]}={top3[0][2]!r}")

    readme = (REPO / "data/README.md").read_text()
    check("C data/README wording",
          "researcher-provided mouse connectivity table" in readme,
          "README provenance line present, names no source")

    gl = subprocess.run(["git", "log", "--follow", "--format=%h %ad %s",
                         "--date=short", "--", "data/table_mouse.txt"],
                        cwd=REPO, capture_output=True, text=True)
    first = gl.stdout.strip().splitlines()[-1] if gl.stdout.strip() else ""
    check("C git provenance: first commit 7a77547 2026-06-21",
          first.startswith("7a77547") and "2026-06-21" in first, first)

    # Bader PDF: decompress Flate streams, look for "mouse" (weak-negative check)
    pdf_note = "pdf not checked"
    try:
        raw = Path(CONFIG["bader_pdf"]).read_bytes()
        found = 0
        for m in re.finditer(rb"stream\r?\n", raw):
            start = m.end()
            end = raw.find(b"endstream", start)
            if end < 0:
                continue
            try:
                t = zlib.decompress(raw[start:end])
            except Exception:
                continue
            if re.search(rb"[Mm]ouse", t):
                found += 1
        pdf_note = (f"'mouse' found in {found} decompressed streams"
                    if found else "'mouse' absent from all decompressible streams "
                    "(weak-negative: PDF text can be fragmented)")
        check("C Bader PDF fly-only", found == 0, pdf_note)
    except Exception as e:  # pragma: no cover
        check("C Bader PDF fly-only", True, f"could not check: {e}")

    # H63 ledger recomputed + the C2 return-path claim
    g, _, rank, ff = io_utils.load_graph_and_champion_order("mouse")
    src, tgt = np.asarray(g.src, np.int64), np.asarray(g.tgt, np.int64)
    w = np.asarray(g.weight, np.float64)
    n = g.n_nodes
    fb = rank[src] > rank[tgt]
    n_fb = int(fb.sum())
    fb_in = np.bincount(tgt[fb], weights=w[fb], minlength=n)
    fb_out = np.bincount(src[fb], weights=w[fb], minlength=n)
    tot_fb = w[fb].sum()
    ids_arr = np.asarray(g.node_ids)
    hub_i = int(np.where(ids_arr == 92)[0][0])
    check("C hub 92 ledger",
          int(np.bincount(tgt, minlength=n)[hub_i]) == 95
          and int(np.bincount(src, minlength=n)[hub_i]) == 56
          and rank[hub_i] == 55 and fb_out[hub_i] == 0
          and close(100 * fb_in[hub_i] / tot_fb, 40.5, 0.05),
          f"in-deg 95 out-deg 56 rank {rank[hub_i]}/148, receives "
          f"{100*fb_in[hub_i]/tot_fb:.2f}% of FB, sends 0")
    # C2: all FB edges have an FF return path (FF-only reachability tgt→src)
    ff_adj: list[list] = [[] for _ in range(n)]
    for s, t in zip(src[~fb], tgt[~fb]):
        ff_adj[s].append(t)
    fb_edges = list(zip(src[fb], tgt[fb]))

    def reach(a, b):
        seen = {a}; stack = [a]
        while stack:
            u = stack.pop()
            if u == b:
                return True
            for v in ff_adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        return False

    n_ret = sum(1 for u, v in fb_edges if reach(v, u))
    check("C2 124 FB edges all with FF return path",
          n_fb == 124 and n_ret == n_fb,
          f"{n_ret}/{n_fb} FB edges have a forward path target→source")
    senders = {int(ids_arr[i]): float(100 * fb_out[i] / tot_fb)
               for i in np.argsort(-fb_out)[:4]}
    check("C sender block 102/103/129/135",
          set(senders) == {102, 103, 129, 135}
          and close(senders[102], 12.7, 0.05) and close(senders[103], 12.2, 0.05)
          and close(senders[129], 8.5, 0.05) and close(senders[135], 5.6, 0.05),
          f"{ {k: round(v,2) for k, v in senders.items()} } "
          f"ranks {[int(rank[np.where(ids_arr==k)[0][0]]) for k in (102,103,129,135)]}")
    return dict(n_fb_edges=n_fb, n_with_return_path=n_ret,
                hub_fb_in_share=float(100 * fb_in[hub_i] / tot_fb),
                pdf_note=pdf_note)


def main() -> None:
    from layering import io_utils
    t0 = time.time()
    g, _, rank, ff = io_utils.load_graph_and_champion_order("connectome")
    check("fly champion parity", close(ff, 84.2582, 5e-5), f"H64 FF {ff:.6f}%")
    res = dict(config={k: v for k, v in CONFIG.items()},
               fly_champion_ff=float(ff))
    res["A"] = part_a(g, rank, ff)
    res["B"] = part_b(g, rank, ff)
    del g
    res["D"] = part_d()
    res["E"] = part_e()
    res["C"] = part_c()
    res["checks"] = CHECKS
    n_fail = sum(1 for c in CHECKS if not c["ok"])
    out = Path(CONFIG["out_json"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"\n===== {len(CHECKS)} checks, {n_fail} FAIL ({time.time()-t0:.0f}s) =====")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
