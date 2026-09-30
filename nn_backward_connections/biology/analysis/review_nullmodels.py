"""Independent review of the null-models study (biology/studies/nullmodels.md).

Reviewer recomputation layer — NOTHING here reuses the study's analysis
functions. The only study code deliberately reused is the *generators*
(the null graphs were never archived, so regeneration from the logged
``gen_seed`` is the only way to reconstruct them; seed-reproducibility is
itself verified below). Every statistic, bound, and matrix is recomputed
with independent implementations; every discrete score goes through the
frozen oracle ``mfas.metrics.score_from_order`` on CPU.

Checks (numbered as in the review brief):
  1. 2-cycle lower bounds — own implementation (lexsort/pair-merge, no
     n-based key encoding) on the real graphs and every regenerated null;
     the mouse "provably less feedback-bound" counts and the one-sided
     proof logic; the 42.7% / ~80% arithmetic.
  2. Every ladder mean ± sd, z, beyond-count and empirical p recomputed
     straight from runs/*.json; decomposition deltas; probe arithmetic.
  3. Generator validation — one fresh draw per rung checked with the
     reviewer's own assertions (including negative controls the selftest
     does not do: w_shuffled must BREAK out-strength, L1 must break
     degrees); duplicate-slot score-invariance verified numerically.
  4. Spot re-scores — saved rank vectors scored by the frozen oracle
     against regenerated null graphs / the real graph.
  5. Lane z-matrix — real lane-pair FB matrix recomputed from the saved
     real order; claimed hotspot z values read back from the npz; lane-pair
     total-weight preservation under L3 verified on a regenerated draw;
     stability of the extremes across the 3 real pipeline seeds.

Results: biology/results/review/review_nullmodels.json (+ console report).

    $PY biology/analysis/review_nullmodels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "biology" / "analysis"))

from mfas.io import load_dataset                              # noqa: E402
from mfas.metrics import pct, score_from_order                # noqa: E402

from nullmodels_generators import (directed_configuration_null,  # noqa: E402
                                   block_configuration_null, gnm_null,
                                   sbm_null, fly_lane_labels)
from nullmodels_run import _draw_null                         # noqa: E402

RUNS = _REPO / "biology" / "results" / "nullmodels" / "runs"
ORDERS = _REPO / "biology" / "results" / "nullmodels" / "orders"
OUT_DIR = _REPO / "biology" / "results" / "review"
OUT_JSON = OUT_DIR / "review_nullmodels.json"
LANE_NPZ = _REPO / "biology" / "results" / "nullmodels" / "lane_fb_zmatrix.npz"

REPORT: dict = {}


def say(*a):
    print(*a, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Independent 2-cycle bound (different algorithm from the study's searchsorted
# implementation: collapse by lexsort-grouping, pair-merge on (min,max) keys).
# ──────────────────────────────────────────────────────────────────────────────
def my_two_cycle_bound_pct(src, tgt, w) -> float:
    """Reviewer's own exact reciprocity lower bound, % of total weight."""
    s = np.asarray(src, np.int64)
    t = np.asarray(tgt, np.int64)
    wf = np.asarray(w, np.float64)
    total = wf.sum()
    # collapse parallel directed edges by (s,t) grouping
    o = np.lexsort((t, s))
    s, t, wf2 = s[o], t[o], wf[o]
    new = np.concatenate([[True], (s[1:] != s[:-1]) | (t[1:] != t[:-1])])
    grp = np.cumsum(new) - 1
    cs, ct = s[new], t[new]
    cw = np.bincount(grp, weights=wf2)
    # merge the two directions of each unordered pair
    lo = np.minimum(cs, ct)
    hi = np.maximum(cs, ct)
    fwd = cs < ct
    po = np.lexsort((hi, lo))
    lo, hi, fwd, cw = lo[po], hi[po], fwd[po], cw[po]
    same = (lo[1:] == lo[:-1]) & (hi[1:] == hi[:-1])
    both = np.flatnonzero(same)
    assert np.all(fwd[both] != fwd[both + 1]), "pair-merge direction error"
    forced = np.minimum(cw[both], cw[both + 1]).sum()
    return float(100.0 * forced / total)


def load_runs():
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        r = json.loads(f.read_text())
        r["file"] = f.name
        rows.append(r)
    return rows


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def mstats(vals):
    v = np.asarray(vals, float)
    return dict(K=int(v.size), mean=float(v.mean()),
                std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                min=float(v.min()), max=float(v.max()))


# ──────────────────────────────────────────────────────────────────────────────
# Check 2 — summary statistics straight from runs/*.json
# ──────────────────────────────────────────────────────────────────────────────
def check_summary(rows):
    say("\n== [2] summary statistics recomputed from runs/*.json ==")
    out = {"n_runs_total": len(rows)}
    for dataset, std_epochs in (("connectome", 4000), ("mouse", 2000)):
        for epochs in sorted({r["epochs"] for r in sel(rows, dataset=dataset)}):
            reals = sel(rows, dataset=dataset, kind="real", epochs=epochs)
            rf = np.array([r["pct_final"] for r in reals])
            rg = np.array([r["pct_greedy"] for r in reals])
            key = f"{dataset}_e{epochs}"
            d = dict(real_final=mstats(rf), real_greedy=mstats(rg),
                     real_seeds=sorted(r["pipeline_seed"] for r in reals))
            for kind in ("w_shuffled", "w_source", "L1_gnm", "L3a_sbm",
                         "L3_dcsbm"):
                nulls = sel(rows, dataset=dataset, kind=kind, epochs=epochs)
                if not nulls:
                    continue
                nf = np.array([r["pct_final"] for r in nulls])
                ng = np.array([r["pct_greedy"] for r in nulls])
                rmean = float(rf.mean())
                z = (rmean - nf.mean()) / nf.std(ddof=1) if nf.size > 1 else None
                if rmean >= nf.mean():
                    beyond = int((nf >= rmean).sum())
                else:
                    beyond = int((nf <= rmean).sum())
                d[kind] = dict(final=mstats(nf), greedy=mstats(ng),
                               z_final=(float(z) if z is not None else None),
                               beyond=beyond,
                               empirical_p=(beyond + 1) / (nf.size + 1),
                               gap_pp=rmean - float(nf.mean()),
                               t_total_range=[float(min(r["t_total_s"]
                                                        for r in nulls)),
                                              float(max(r["t_total_s"]
                                                        for r in nulls))],
                               sift_sweeps=sorted({r["sift_max_sweeps"]
                                                   for r in nulls}))
            out[key] = d
            say(f"  {key}: real {mstats(rf)['mean']:.4f}±{mstats(rf)['std']:.4f}"
                f" (K={rf.size})")
            for kind in d:
                if isinstance(d[kind], dict) and "final" in d[kind]:
                    f_ = d[kind]["final"]
                    zs = (f"{d[kind]['z_final']:+8.2f}"
                          if d[kind]["z_final"] is not None else "     n/a")
                    say(f"    {kind:11s} K={f_['K']:3d} {f_['mean']:8.4f}"
                        f"±{f_['std']:.4f}  z={zs}"
                        f"  beyond={d[kind]['beyond']}"
                        f"  p={d[kind]['empirical_p']:.3f}"
                        f"  greedy {d[kind]['greedy']['mean']:.3f}")
    # nested-ladder deltas (fly, standard arm)
    s = out["connectome_e4000"]
    chain = [("chance", 50.0), ("L1_gnm", s["L1_gnm"]["final"]["mean"]),
             ("w_source", s["w_source"]["final"]["mean"]),
             ("L3_dcsbm", s["L3_dcsbm"]["final"]["mean"]),
             ("real", s["real_final"]["mean"])]
    deltas = {f"{a}->{b}": chain[i + 1][1] - chain[i][1]
              for i, ((a, _), (b, _)) in enumerate(zip(chain[:-1], chain[1:]))}
    out["fly_ladder_deltas_pp"] = deltas
    say("  fly nested deltas:", {k: round(v, 3) for k, v in deltas.items()})
    # fly total nulls in the standard arm
    out["fly_standard_null_count"] = sum(
        len(sel(rows, dataset="connectome", kind=k, epochs=4000))
        for k in ("w_shuffled", "w_source", "L1_gnm", "L3a_sbm", "L3_dcsbm"))
    # w_shuffled vs w_source coupling delta
    out["fly_wsource_minus_wshuffled_pp"] = (
        s["w_source"]["final"]["mean"] - s["w_shuffled"]["final"]["mean"])
    REPORT["summary"] = out


# ──────────────────────────────────────────────────────────────────────────────
# Check 2b — the 5× budget probe & mouse strong arm, straight from the JSONs
# ──────────────────────────────────────────────────────────────────────────────
def check_probe(rows):
    say("\n== [2b] budget probe (fly e20000) & mouse strong arm (e20000) ==")
    out = {}
    probe = {}
    for r in sel(rows, dataset="connectome", epochs=20000):
        tag = (f"{r['kind']}_gs{r['gen_seed']}" if r["kind"] != "real"
               else f"real_ps{r['pipeline_seed']}")
        probe[tag] = dict(pct_final=r["pct_final"], sweeps=r["sift_max_sweeps"])
    out["fly_e20000"] = probe
    # matching e4000 baselines for the same graphs
    base = {}
    for kind, gs in (("w_source", 2000), ("L3_dcsbm", 5000), ("L1_gnm", 3000)):
        m = sel(rows, dataset="connectome", kind=kind, epochs=4000, gen_seed=gs)
        if m:
            base[f"{kind}_gs{gs}"] = m[0]["pct_final"]
    m = sel(rows, dataset="connectome", kind="real", epochs=4000,
            pipeline_seed=42)
    base["real_ps42"] = m[0]["pct_final"]
    out["fly_e4000_same_graphs"] = base
    for tag in probe:
        if tag in base:
            say(f"  {tag}: {base[tag]:.4f} -> {probe[tag]['pct_final']:.4f} "
                f"(+{probe[tag]['pct_final'] - base[tag]:.3f}) "
                f"sweeps={probe[tag]['sweeps']}")
    if "real_ps42" in probe:
        for kind, gs in (("L3_dcsbm", 5000), ("w_source", 2000),
                         ("L1_gnm", 3000)):
            t = f"{kind}_gs{gs}"
            if t in probe:
                out[f"probe_gap_real_minus_{kind}_pp"] = (
                    probe["real_ps42"]["pct_final"] - probe[t]["pct_final"])
    say("  probe gaps:", {k: round(v, 3) for k, v in out.items()
                          if k.startswith("probe_gap")})
    REPORT["probe"] = out


# ──────────────────────────────────────────────────────────────────────────────
# Check 1 — 2-cycle bounds: real graphs + every regenerated null
# ──────────────────────────────────────────────────────────────────────────────
def check_two_cycle(rows, g_fly, g_mouse, labels):
    say("\n== [1] 2-cycle bounds — reviewer's own implementation ==")
    out = {}
    for name, g in (("connectome", g_fly), ("mouse", g_mouse)):
        src, tgt, w = (np.asarray(g.src), np.asarray(g.tgt),
                       np.asarray(g.weight))
        real_b = my_two_cycle_bound_pct(src, tgt, w)
        d = {"real": real_b}
        epochs = 4000 if name == "connectome" else 2000
        lab = labels if name == "connectome" else None
        for kind in ("w_shuffled", "w_source", "L1_gnm", "L3a_sbm",
                     "L3_dcsbm"):
            nulls = sel(rows, dataset=name, kind=kind, epochs=epochs)
            if not nulls:
                continue
            vals, prov_below, prov_above = [], 0, 0
            reals = sel(rows, dataset=name, kind="real", epochs=epochs)
            real_meas_fb = 100.0 - float(np.mean([r["pct_final"]
                                                  for r in reals]))
            for r in nulls:
                s2, t2, w2, _ = _draw_null(kind, src, tgt, w, g.n_nodes,
                                           r["gen_seed"], lab)
                b = my_two_cycle_bound_pct(s2, t2, w2)
                vals.append(b)
                null_meas_fb = 100.0 - r["pct_final"]
                # null measured FB (upper bd on its unavoidable FB) below the
                # real 2-cycle LOWER bound => null provably less FB-bound.
                if null_meas_fb < real_b:
                    prov_below += 1
                # the reverse proof needs the null's LOWER bound above the
                # real's measured FB (upper bd on real unavoidable FB).
                if b > real_meas_fb:
                    prov_above += 1
            d[kind] = mstats(vals)
            d[kind + "_n_provably_below_real"] = prov_below
            d[kind + "_n_provably_above_real"] = prov_above
            d[kind + "_real_measured_fb_pct"] = real_meas_fb
            say(f"  {name} {kind:11s} bound {d[kind]['mean']:.4f}"
                f"±{d[kind]['std']:.4f} [{d[kind]['min']:.4f},"
                f"{d[kind]['max']:.4f}]  provably_below={prov_below}"
                f"/{len(nulls)}  provably_above={prov_above}")
        out[name] = d
        say(f"  {name} REAL 2-cycle bound = {real_b:.4f}% of W")
    # headline arithmetic
    ch = REPORT.get("champion", {}).get("connectome_pct")
    if ch:
        fb_champ = 100.0 - ch
        out["fly_share_of_champion_fb"] = out["connectome"]["real"] / fb_champ
        say(f"  fly bound / champion FB = {out['connectome']['real']:.4f}"
            f"/{fb_champ:.4f} = {out['fly_share_of_champion_fb']:.4f}")
    mouse_fb = 100.0 - float(np.mean(
        [r["pct_final"] for r in sel(rows, dataset="mouse", kind="real",
                                     epochs=2000)]))
    out["mouse_share_of_measured_fb"] = out["mouse"]["real"] / mouse_fb
    say(f"  mouse bound / measured FB = {out['mouse']['real']:.4f}"
        f"/{mouse_fb:.4f} = {out['mouse_share_of_measured_fb']:.4f}")
    # fly excess ratios vs each rung
    out["fly_excess_ratios"] = {
        k: out["connectome"]["real"] / out["connectome"][k]["mean"]
        for k in ("w_shuffled", "w_source", "L1_gnm", "L3a_sbm", "L3_dcsbm")
        if k in out["connectome"]}
    say("  fly real/null bound ratios:",
        {k: round(v, 1) for k, v in out["fly_excess_ratios"].items()})
    REPORT["two_cycle"] = out


# ──────────────────────────────────────────────────────────────────────────────
# Check 3 — generators: reviewer's own fresh-draw validation (+ negatives)
# ──────────────────────────────────────────────────────────────────────────────
def check_generators(g_fly, g_mouse, labels):
    say("\n== [3] generator checks — one fresh draw per rung, own assertions ==")
    out = {}
    SEED = 20260930
    for name, g in (("connectome", g_fly), ("mouse", g_mouse)):
        src, tgt, w = (np.asarray(g.src), np.asarray(g.tgt),
                       np.asarray(g.weight))
        n = g.n_nodes
        wf = np.asarray(w, np.float64)
        res = {}
        # L2 w_source
        s2, t2, w2, st = directed_configuration_null(src, tgt, w,
                                                     variant="w_source",
                                                     seed=SEED)
        ok = (np.array_equal(np.bincount(src, minlength=n),
                             np.bincount(s2, minlength=n))
              and np.array_equal(np.bincount(tgt, minlength=n),
                                 np.bincount(t2, minlength=n))
              and np.array_equal(np.sort(wf),
                                 np.sort(np.asarray(w2, np.float64)))
              and np.array_equal(
                  np.bincount(src, weights=wf, minlength=n),
                  np.bincount(s2, weights=np.asarray(w2, np.float64),
                              minlength=n))
              and int((s2 == t2).sum()) == 0)
        s3, t3, w3, _ = directed_configuration_null(src, tgt, w,
                                                    variant="w_source",
                                                    seed=SEED)
        ok = ok and np.array_equal(t2, t3) and np.array_equal(w2, w3)
        res["w_source"] = dict(ok=bool(ok), stats=st)
        # L2 w_shuffled — must preserve degrees but BREAK out-strength
        s2, t2, w2, st = directed_configuration_null(src, tgt, w,
                                                     variant="w_shuffled",
                                                     seed=SEED)
        deg_ok = (np.array_equal(np.bincount(src, minlength=n),
                                 np.bincount(s2, minlength=n))
                  and np.array_equal(np.bincount(tgt, minlength=n),
                                     np.bincount(t2, minlength=n))
                  and np.array_equal(np.sort(wf),
                                     np.sort(np.asarray(w2, np.float64)))
                  and int((s2 == t2).sum()) == 0)
        strength_broken = not np.array_equal(
            np.bincount(src, weights=wf, minlength=n),
            np.bincount(s2, weights=np.asarray(w2, np.float64), minlength=n))
        res["w_shuffled"] = dict(ok=bool(deg_ok),
                                 out_strength_actually_broken=bool(
                                     strength_broken), stats=st)
        # L1 — totals kept, degrees actually destroyed
        s2, t2, w2, st = gnm_null(src, tgt, w, n=n, seed=SEED)
        tot_ok = (s2.size == src.size
                  and np.array_equal(np.sort(wf),
                                     np.sort(np.asarray(w2, np.float64)))
                  and int((s2 == t2).sum()) == 0)
        deg_broken = not np.array_equal(np.bincount(src, minlength=n),
                                        np.bincount(s2, minlength=n))
        res["L1_gnm"] = dict(ok=bool(tot_ok),
                             degrees_actually_broken=bool(deg_broken),
                             stats=st)
        if name == "connectome":
            K = int(labels.max()) + 1
            lab = labels.astype(np.int64)
            # L3 — per-node per-block profiles via np.add.at (own method)
            s2, t2, w2, st = block_configuration_null(src, tgt, w, labels,
                                                      seed=SEED)
            out_prof = np.zeros((n, K)); out_prof2 = np.zeros((n, K))
            in_prof = np.zeros((n, K)); in_prof2 = np.zeros((n, K))
            ws_prof = np.zeros((n, K)); ws_prof2 = np.zeros((n, K))
            np.add.at(out_prof, (src, lab[tgt]), 1.0)
            np.add.at(out_prof2, (s2, lab[t2]), 1.0)
            np.add.at(in_prof, (tgt, lab[src]), 1.0)
            np.add.at(in_prof2, (t2, lab[s2]), 1.0)
            np.add.at(ws_prof, (src, lab[tgt]), wf)
            np.add.at(ws_prof2, (s2, lab[t2]), np.asarray(w2, np.float64))
            ok3 = (np.array_equal(out_prof, out_prof2)
                   and np.array_equal(in_prof, in_prof2)
                   and np.array_equal(ws_prof, ws_prof2)
                   and int((s2 == t2).sum()) == 0)
            res["L3_dcsbm"] = dict(ok=bool(ok3), stats=st)
            # L3a — block counts + weights preserved, degrees broken
            s2, t2, w2, st = sbm_null(src, tgt, w, labels, seed=SEED)
            blk = lab[src] * K + lab[tgt]
            blk2 = lab[s2] * K + lab[t2]
            ok3a = (np.array_equal(np.bincount(blk, minlength=K * K),
                                   np.bincount(blk2, minlength=K * K))
                    and np.allclose(
                        np.bincount(blk, weights=wf, minlength=K * K),
                        np.bincount(blk2, weights=np.asarray(w2, np.float64),
                                    minlength=K * K))
                    and int((s2 == t2).sum()) == 0)
            deg_broken3a = not np.array_equal(np.bincount(src, minlength=n),
                                              np.bincount(s2, minlength=n))
            res["L3a_sbm"] = dict(ok=bool(ok3a),
                                  degrees_actually_broken=bool(deg_broken3a),
                                  stats=st)
        out[name] = res
        for k, v in res.items():
            say(f"  {name} {k:11s} ok={v['ok']} "
                + " ".join(f"{kk}={vv}" for kk, vv in v.items()
                           if kk not in ("ok", "stats")))
    REPORT["generators"] = out


def check_duplicate_invariance(g_fly, labels):
    """Duplicates: collapse parallel slots and re-score the SAME order —
    the oracle score must be identical (the study's score-invariance claim)."""
    say("\n== [3b] duplicate-slot score-invariance (numerical) ==")
    src, tgt, w = (np.asarray(g_fly.src), np.asarray(g_fly.tgt),
                   np.asarray(g_fly.weight))
    r = json.loads((RUNS / "connectome__L3_dcsbm__gs5003__ps42__e4000.json")
                   .read_text())
    s2, t2, w2, st = _draw_null("L3_dcsbm", src, tgt, w, g_fly.n_nodes,
                                r["gen_seed"], labels)
    rank = np.load(_REPO / r["order_npy"]).astype(np.int64)
    sc_par = score_from_order(rank, s2, t2, w2)
    key = s2.astype(np.int64) * g_fly.n_nodes + t2.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    uw = np.bincount(inv, weights=np.asarray(w2, np.float64))
    us, ut = (uk // g_fly.n_nodes).astype(np.int64), \
        (uk % g_fly.n_nodes).astype(np.int64)
    sc_col = score_from_order(rank, us, ut, uw)
    ok = abs(sc_par - sc_col) < 1e-6
    say(f"  parallel={sc_par:.1f} collapsed={sc_col:.1f} equal={ok} "
        f"(dup slots={st['n_duplicate_slots']})")
    REPORT["duplicate_invariance"] = dict(score_parallel=sc_par,
                                          score_collapsed=sc_col, ok=bool(ok),
                                          n_duplicate_slots=int(
                                              st["n_duplicate_slots"]))


# ──────────────────────────────────────────────────────────────────────────────
# Check 4 — spot re-scores through the frozen oracle
# ──────────────────────────────────────────────────────────────────────────────
def check_spot_rescores(g_fly, labels):
    say("\n== [4] spot re-scores (frozen oracle, CPU) ==")
    src, tgt, w = (np.asarray(g_fly.src), np.asarray(g_fly.tgt),
                   np.asarray(g_fly.weight))
    total = g_fly.total_weight
    picks = [
        "connectome__real__ps42__e4000",
        "connectome__real__ps123__e4000",
        "connectome__real__ps999__e4000",
        "connectome__real__ps42__e20000",
        "connectome__w_shuffled__gs1013__ps42__e4000",
        "connectome__w_source__gs2007__ps42__e4000",
        "connectome__w_source__gs2000__ps42__e4000",
        "connectome__L1_gnm__gs3002__ps42__e4000",
        "connectome__L3a_sbm__gs4006__ps42__e4000",
        "connectome__L3_dcsbm__gs5003__ps42__e4000",
        "connectome__L3_dcsbm__gs5000__ps42__e20000",
        "connectome__w_source__gs2000__ps42__e20000",
        "connectome__L1_gnm__gs3000__ps42__e20000",
    ]
    out = []
    for stem in picks:
        jf = RUNS / f"{stem}.json"
        of = ORDERS / f"{stem}.npy"
        if not jf.exists() or not of.exists():
            out.append(dict(stem=stem, status="MISSING",
                            json=jf.exists(), order=of.exists()))
            say(f"  {stem}: MISSING (json={jf.exists()} order={of.exists()})")
            continue
        r = json.loads(jf.read_text())
        rank = np.load(of).astype(np.int64)
        if r["kind"] == "real":
            s2, t2, w2 = src, tgt, w
        else:
            s2, t2, w2, _ = _draw_null(r["kind"], src, tgt, w, g_fly.n_nodes,
                                       r["gen_seed"], labels)
        sc = score_from_order(rank, s2, t2, w2)
        match = abs(sc - r["score_final"]) < 0.5
        out.append(dict(stem=stem, rescored=sc, logged=r["score_final"],
                        rescored_pct=pct(sc, total),
                        logged_pct=r["pct_final"], match=bool(match)))
        say(f"  {stem}: rescored {sc:.0f} ({pct(sc, total):.4f}%) "
            f"logged {r['score_final']:.0f} ({r['pct_final']:.4f}%) "
            f"match={match}")
    REPORT["spot_rescores"] = out


def check_champion(g_fly):
    say("\n== [champion] fly champion re-score (frozen oracle) ==")
    rank = np.load(_REPO / "layering" / "outputs" / "fly_ann"
                   / "node_ann.npz")["rank"]
    sc = score_from_order(np.asarray(rank, np.int64), np.asarray(g_fly.src),
                          np.asarray(g_fly.tgt), np.asarray(g_fly.weight))
    p = pct(sc, g_fly.total_weight)
    say(f"  champion score {sc:.0f} = {p:.5f}% (study claims 84.25818)")
    REPORT["champion"] = dict(connectome_score=sc, connectome_pct=p)


# ──────────────────────────────────────────────────────────────────────────────
# Check 5 — lane z-matrix
# ──────────────────────────────────────────────────────────────────────────────
def check_lane(rows, g_fly, labels):
    say("\n== [5] lane z-matrix ==")
    d = np.load(LANE_NPZ, allow_pickle=False)
    lane_list = [str(x) for x in d["lane_list"]]
    K = len(lane_list)
    src, tgt, w = (np.asarray(g_fly.src), np.asarray(g_fly.tgt),
                   np.asarray(g_fly.weight))
    wf = np.asarray(w, np.float64)
    total = float(g_fly.total_weight)
    lab = labels.astype(np.int64)

    def my_fb_matrix(rank, s, t, ww):
        back = ~(rank[t] > rank[s])
        M = np.zeros((K, K))
        np.add.at(M, (lab[s][back], lab[t][back]),
                  np.asarray(ww, np.float64)[back])
        return M

    # real matrix from the seed-42 real order (as the study used)
    rk42 = np.load(ORDERS / "connectome__real__ps42__e4000.npy").astype(
        np.int64)
    M_mine = my_fb_matrix(rk42, src, tgt, w)
    real_fb_match = bool(np.allclose(M_mine, d["real_fb"]))
    say(f"  real_fb matrix matches reviewer recomputation: {real_fb_match}")

    # lane-pair TOTALS preserved by L3? regenerate one draw and compare
    s2, t2, w2, _ = _draw_null("L3_dcsbm", src, tgt, w, g_fly.n_nodes, 5000,
                               labels)
    Wr = np.zeros((K, K)); Wn = np.zeros((K, K))
    np.add.at(Wr, (lab[src], lab[tgt]), wf)
    np.add.at(Wn, (lab[s2], lab[t2]), np.asarray(w2, np.float64))
    totals_ok = bool(np.allclose(Wr, Wn))
    wpair_ok = bool(np.allclose(Wr, d["w_pair"]))
    say(f"  lane-pair totals real==L3-null(gs5000): {totals_ok}; "
        f"npz w_pair==recomputed: {wpair_ok}")

    z = d["z"]; mu = d["null_mean"]; sd = d["null_std"]; Mreal = d["real_fb"]
    idx = {L: i for i, L in enumerate(lane_list)}
    claims = {
        "MB->MB": ("mushroom body", "mushroom body"),
        "lamina->lamina": ("lamina", "lamina"),
        "central(other)->visual projection": ("central (other)",
                                              "visual projection"),
        "photoreceptors->medulla": ("photoreceptors", "medulla"),
        "visual centrifugal->lobula (plate)": ("visual centrifugal",
                                               "lobula (plate)"),
        "central(other)->central(other)": ("central (other)",
                                           "central (other)"),
        "medulla->medulla": ("medulla", "medulla"),
    }
    lane_out = dict(real_fb_match=real_fb_match, totals_ok=totals_ok,
                    wpair_ok=wpair_ok, mask_sum=int(d["mask"].sum()),
                    K=K, hotspots={})
    for name, (a, b) in claims.items():
        i, j = idx[a], idx[b]
        lane_out["hotspots"][name] = dict(
            z=float(z[i, j]), real_pct_W=100 * float(Mreal[i, j]) / total,
            null_pct_W=100 * float(mu[i, j]) / total,
            ratio=float(Mreal[i, j] / mu[i, j]) if mu[i, j] else None)
        h = lane_out["hotspots"][name]
        say(f"  {name:38s} z={h['z']:+7.1f} real {h['real_pct_W']:.3f}%W "
            f"null {h['null_pct_W']:.3f}%W ratio {h['ratio']:.2f}")
    # optic-chain negatives quoted as a range
    negs = {}
    for a, b in (("visual projection", "medulla"),
                 ("medulla", "medulla→lobula"),
                 ("medulla→lobula", "medulla"),
                 ("visual projection", "medulla→lobula"),
                 ("medulla→lobula", "lobula (plate)")):
        if a in idx and b in idx:
            negs[f"{a}->{b}"] = float(z[idx[a], idx[b]])
    lane_out["optic_negatives"] = negs
    say("  optic negatives:", {k: round(v, 1) for k, v in negs.items()
                               if np.isfinite(v)})
    # stability of the extremes across the 3 real pipeline seeds
    stab = {}
    for ps in (42, 123, 999):
        rk = np.load(ORDERS / f"connectome__real__ps{ps}__e4000.npy").astype(
            np.int64)
        M = my_fb_matrix(rk, src, tgt, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = (M - mu) / sd
        stab[str(ps)] = {name: float(zz[idx[a], idx[b]])
                         for name, (a, b) in claims.items()}
    lane_out["z_across_real_seeds"] = stab
    say("  z across real seeds (MB->MB):",
        {k: round(v["MB->MB"], 1) for k, v in stab.items()})
    REPORT["lane"] = lane_out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_runs()
    say(f"loaded {len(rows)} run records (study claims 689)")
    g_fly = load_dataset("connectome")
    g_mouse = load_dataset("mouse")
    labels, lane_list = fly_lane_labels()
    check_summary(rows)
    check_probe(rows)
    check_champion(g_fly)
    check_two_cycle(rows, g_fly, g_mouse, labels)
    check_generators(g_fly, g_mouse, labels)
    check_duplicate_invariance(g_fly, labels)
    check_spot_rescores(g_fly, labels)
    check_lane(rows, g_fly, labels)
    OUT_JSON.write_text(json.dumps(REPORT, indent=1, default=float))
    say(f"\nreview results -> {OUT_JSON}")


if __name__ == "__main__":
    main()
