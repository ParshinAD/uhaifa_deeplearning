"""INDEPENDENT REVIEW of biology/studies/label_nulls.md (strict pass, 2026-09-30).

Everything here is re-derived with code written for this review — no function of
biology/analysis/exp_label_nulls.py is imported or called (the only shared code
is the parity-gated champion loader layering.io_utils, which is the task's own
definition of the order, and numpy itself).

Sections
  1. GATES  — recompute every observed statistic with a per-EDGE pandas groupby
              (the study used per-node bincounts): fly per-NT backward shares,
              MICrONS E/I shares + per-layer leads (incl. WM), T3 medians.
  2. NPZ    — internal consistency of the study's stored null draws: recompute
              mean/sd/z/p/n_le/n_ge from label_nulls_*_nulls.npz and compare to
              every number in label_nulls_*.json.
  3. NULLS  — small independent nulls (K=300, seed 20260930, own stratified-
              permutation implementation):
              (a) fly T1 within-super_class (all 7 NTs; focus 5-HT, OA);
              (b) MICrONS T2 within-layer (leads L1-L6, mean, E/I shares);
              (c) T3 as an HONEST label permutation (permute the type-label
                  vector, NOT the rank vector) — empirically tests the study's
                  claimed equivalence as well as its numbers.
  4. REDTEAM
              (3a) T1 stratified by cell_class and by super_class x cell_class
                   (finer composition control) — does 5-HT survive?
              (3d) drop the KC->ACh override (raw top_nt from the Schlegel TSV)
                   — observed shares + within-super_class null; verdict flips?
              (3e) include WM in the T2 lead statistic — observed WM lead and
                   the 7-layer mean lead under the within-layer null.

Run:  $PY biology/analysis/review_label_nulls.py
Out:  biology/results/review/review_label_nulls.json (+ _draws.npz)

Determinism re-run of the study script itself is a separate wrapper (scratch);
see the review section appended to the study.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RSEED = 20260930          # deliberately different from the study's 42
K = 300                   # independent permutations per null
OUT_DIR = REPO_ROOT / "biology/results/review"
TSV = REPO_ROOT / "data/external/flywire_neuron_annotations.tsv.gz"

REPORT: dict = {"seed": RSEED, "n_perm": K}
DRAWS: dict = {}


def emp(null, obs):
    """Two-sided add-one empirical p + z (review's own copy of the convention)."""
    null = np.asarray(null, float)
    n = len(null)
    n_le = int((null <= obs).sum())
    n_ge = int((null >= obs).sum())
    p = min(1.0, 2.0 * min((1 + n_le) / (n + 1), (1 + n_ge) / (n + 1)))
    sd = float(null.std(ddof=1))
    return dict(observed=float(obs), null_mean=float(null.mean()), null_sd=sd,
                z=float((obs - null.mean()) / sd) if sd > 0 else float("inf"),
                p_two_sided=p, n_le=n_le, n_ge=n_ge, n_perm=n)


def shuffle_within(labels, strata, rng):
    """Own stratified-permutation implementation (differs from the study's:
    strata given as a code vector, permuted via argsort-free per-stratum draw)."""
    out = labels.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = labels[idx[rng.permutation(idx.size)]]
    return out


# ===========================================================================
# FLY
# ===========================================================================
def fly_section():
    from layering import io_utils
    g, order, rank, ff = io_utils.load_graph_and_champion_order("connectome")
    ann = np.load(REPO_ROOT / "layering/outputs/fly_ann/node_ann.npz")
    meta = json.loads((REPO_ROOT / "layering/outputs/fly_ann/node_ann_meta.json")
                      .read_text())
    nt = ann["top_nt"].astype(np.int64)
    sc = ann["super_class"].astype(np.int64)
    cc = ann["cell_class"].astype(np.int64)
    nt_names = meta["top_nt_values"]
    w = g.weight.astype(np.int64)
    back = rank[g.src] > rank[g.tgt]

    # ---- 1. GATES: per-EDGE groupby (independent route) -------------------
    df = pd.DataFrame({"nt": nt[g.src], "w": w, "bw": np.where(back, w, 0)})
    gb = df.groupby("nt").sum()
    obs_share = (100.0 * gb["bw"] / gb["w"]).reindex(range(len(nt_names))).to_numpy()
    brainwide = 100.0 * df["bw"].sum() / df["w"].sum()
    expected = {"acetylcholine": 11.360, "gaba": 20.897, "glutamate": 22.249,
                "dopamine": 20.107, "serotonin": 28.826, "octopamine": 16.724}
    gates = {nm: dict(got=round(float(obs_share[i]), 3), expected=expected.get(nm))
             for i, nm in enumerate(nt_names)}
    gates["_brain_wide"] = dict(got=round(float(brainwide), 3), expected=15.742)
    gates["_champion_ff_pct"] = float(ff)
    REPORT["fly_gates"] = gates
    for nm, v in expected.items():
        assert abs(gates[nm]["got"] - v) < 5e-4, (nm, gates[nm])
    assert abs(brainwide - 15.742) < 5e-4
    print("[gates fly] all per-NT shares + brain-wide reproduced:", gates)

    # per-node accumulators for the permutation nulls (np.add.at, own route)
    n = rank.shape[0]
    ow = np.zeros(n, np.int64)
    bw = np.zeros(n, np.int64)
    np.add.at(ow, g.src, w)
    np.add.at(bw, g.src[back], w[back])
    assert abs(100.0 * bw.sum() / ow.sum() - brainwide) < 1e-12

    def shares_of(lab):
        num = np.bincount(lab, weights=bw, minlength=len(nt_names))
        den = np.bincount(lab, weights=ow, minlength=len(nt_names))
        return 100.0 * num / den

    assert np.allclose(shares_of(nt), obs_share)  # per-node == per-edge route

    # ---- 3a. independent null: within super_class -------------------------
    rng = np.random.default_rng(RSEED)
    draws = np.empty((K, len(nt_names)))
    for k in range(K):
        draws[k] = shares_of(shuffle_within(nt, sc, rng))
    REPORT["null_T1_super_class"] = {nm: emp(draws[:, i], obs_share[i])
                                    for i, nm in enumerate(nt_names)}
    DRAWS["T1_super_class"] = draws
    print("[null T1/super_class] done")

    # ---- REDTEAM 3a: cell_class and crossed strata ------------------------
    for tag, strat in [("cell_class", cc), ("super_x_cell", sc * 1000 + cc)]:
        rng = np.random.default_rng([RSEED, hash(tag) % 2**31])
        d2 = np.empty((K, len(nt_names)))
        for k in range(K):
            d2[k] = shares_of(shuffle_within(nt, strat, rng))
        REPORT[f"redteam_T1_{tag}"] = {nm: emp(d2[:, i], obs_share[i])
                                       for i, nm in enumerate(nt_names)}
        DRAWS[f"T1_{tag}"] = d2
        print(f"[redteam T1/{tag}] done")

    # ---- REDTEAM 3d: no KC->ACh override ----------------------------------
    raw_nt_str = {}
    with gzip.open(TSV, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            raw_nt_str[int(row["root_id"])] = row["top_nt"].strip() or "NA"
    code = {nm: i for i, nm in enumerate(nt_names)}
    raw = np.array([code[raw_nt_str.get(int(i), "NA")] for i in g.node_ids],
                   np.int64)
    kc = cc == meta["cell_class_values"].index("Kenyon_Cell")
    diff = raw != nt
    REPORT["redteam_kc_override"] = dict(
        n_differing=int(diff.sum()), n_kc=int(kc.sum()),
        all_diffs_are_kc=bool(np.all(kc[diff])),
        kc_raw_nt_counts={nt_names[v]: int((raw[kc] == v).sum())
                          for v in np.unique(raw[kc])})
    obs_raw = shares_of(raw)
    rng = np.random.default_rng([RSEED, 77])
    d3 = np.empty((K, len(nt_names)))
    for k in range(K):
        d3[k] = shares_of(shuffle_within(raw, sc, rng))
    REPORT["redteam_T1_no_override"] = {
        nm: dict(emp(d3[:, i], obs_raw[i]), n_neurons=int((raw == i).sum()))
        for i, nm in enumerate(nt_names)}
    DRAWS["T1_no_override"] = d3
    print("[redteam T1/no_override] done:", REPORT["redteam_kc_override"])

    # ---- T3: types, gates, honest label-permutation null ------------------
    ctype = {}
    with gzip.open(TSV, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = row["cell_type"].strip()
            if not t:
                c = row["cell_class"].strip() or row["super_class"].strip()
                t = f"{c} (class)" if c else ""
            ctype[int(row["root_id"])] = t or "unannotated"
    types = np.array([ctype.get(int(i), "unannotated") for i in g.node_ids])
    uniq, inv, counts = np.unique(types, return_inverse=True,
                                  return_counts=True)

    def iqr_over_n(v):
        q1, q3 = np.percentile(v, [25, 75])
        return (q3 - q1) / n

    keep100 = [i for i in range(len(uniq))
               if counts[i] >= 100 and uniq[i] != "unannotated"]
    med100 = float(np.median([iqr_over_n(rank[inv == i]) for i in keep100]))
    keep = np.array([i for i in range(len(uniq))
                     if counts[i] >= 20 and uniq[i] != "unannotated"])
    obs_iqr = np.array([iqr_over_n(rank[inv == i]) for i in keep])
    obs_med = float(np.median(obs_iqr))
    REPORT["fly_gates_t3"] = dict(n_types_min20=int(len(keep)),
                                  observed_median=obs_med,
                                  n_types_min100=len(keep100),
                                  median_min100=med100)
    assert len(keep) == 437 and abs(obs_med - 0.0441) < 5e-4
    assert abs(med100 - 0.049) < 5e-4
    print(f"[gates T3] 437 types, median {obs_med:.5f}; gate(>=100) {med100:.5f}")

    # honest label permutation: permute the TYPE-CODE vector across neurons
    rng = np.random.default_rng([RSEED, 3])
    starts = np.concatenate([[0], np.cumsum(counts)])
    kmed = np.empty(K)
    per_type_null = np.empty((K, len(keep)), np.float32)
    for kk in range(K):
        lab_p = inv[rng.permutation(n)]           # label perm, NOT rank perm
        sr = rank[np.argsort(lab_p, kind="stable")]
        vals = np.array([iqr_over_n(sr[starts[i]:starts[i + 1]]) for i in keep])
        per_type_null[kk] = vals
        kmed[kk] = np.median(vals)
    REPORT["null_T3_label_perm"] = emp(kmed, obs_med)
    pct1 = np.percentile(per_type_null.astype(np.float64), 1, axis=0)
    REPORT["null_T3_per_type"] = dict(
        n_tighter_than_pct1=int((obs_iqr < pct1).sum()),
        n_below_all_draws=int((obs_iqr < per_type_null.min(axis=0)).sum()),
        n_types=int(len(keep)))
    DRAWS["T3_null_median"] = kmed
    print("[null T3/label_perm] done:", REPORT["null_T3_label_perm"])


# ===========================================================================
# MICRONS
# ===========================================================================
def microns_section():
    from layering import io_utils
    g, order, rank, ff = io_utils.load_graph_and_champion_order("microns")
    ann = np.load(REPO_ROOT / "layering/outputs/microns_ann/node_ann.npz")
    ei = ann["ei"].astype(np.int64)      # 0 NA / 1 E / 2 I
    layer = ann["layer"].astype(np.int64)  # 0 NA / 1 L1..5 L6 / 6 WM
    lnames = ["NA", "L1", "L2/3", "L4", "L5", "L6", "WM"]
    w = g.weight.astype(np.int64)
    back = rank[g.src] > rank[g.tgt]

    # ---- 1. GATES: per-edge groupby + per-layer medians -------------------
    df = pd.DataFrame({"ei": ei[g.src], "w": w, "bw": np.where(back, w, 0)})
    gb = df.groupby("ei").sum()
    sE = 100.0 * gb.loc[1, "bw"] / gb.loc[1, "w"]
    sI = 100.0 * gb.loc[2, "bw"] / gb.loc[2, "w"]
    leads = {}
    for L in range(1, 7):
        m = layer == L
        leads[lnames[L]] = dict(
            lead=float(np.median(rank[m & (ei == 1)])
                       - np.median(rank[m & (ei == 2)])),
            n_exc=int((m & (ei == 1)).sum()), n_inh=int((m & (ei == 2)).sum()))
    REPORT["microns_gates"] = dict(
        champion_ff_pct=float(ff), E_backward=round(float(sE), 3),
        I_backward=round(float(sI), 3), leads=leads)
    assert abs(sE - 26.622) < 5e-4 and abs(sI - 5.210) < 5e-4
    for nm, v in [("L1", 6640.0), ("L2/3", 6936.0), ("L4", 12935.0),
                  ("L5", 6672.0), ("L6", 6843.5)]:
        assert leads[nm]["lead"] == v, (nm, leads[nm])
    print("[gates microns] E/I shares + leads reproduced;",
          "WM lead:", leads["WM"])

    # per-node accumulators
    n = rank.shape[0]
    ow = np.zeros(n, np.int64)
    bw = np.zeros(n, np.int64)
    np.add.at(ow, g.src, w)
    np.add.at(bw, g.src[back], w[back])

    tested = [1, 2, 3, 4, 5]           # L1..L6 (per study); WM tracked extra
    lidx = {L: np.flatnonzero(layer == L) for L in range(1, 7)}
    strat = np.where(ei != 0, layer, -1)   # WM stratum included, like study

    def leads_of(lab, layers):
        return np.array([np.median(rank[lidx[L]][lab[lidx[L]] == 1])
                         - np.median(rank[lidx[L]][lab[lidx[L]] == 2])
                         for L in layers])

    def shares_of(lab):
        num = np.bincount(lab, weights=bw, minlength=3)
        den = np.bincount(lab, weights=ow, minlength=3)
        s = 100.0 * num / den
        return np.array([s[1], s[2], s[1] - s[2]])

    obs_l = leads_of(ei, tested)
    obs_wm = leads_of(ei, [6])[0]
    obs_s = shares_of(ei)
    assert np.isclose(obs_s[0], sE) and np.isclose(obs_s[1], sI)

    # ---- 3b. independent within-layer null (own implementation) -----------
    rng = np.random.default_rng([RSEED, 2])
    dl = np.empty((K, len(tested)))
    dwm = np.empty(K)
    ds = np.empty((K, 3))
    for k in range(K):
        lab = ei.copy()
        for s in range(1, 7):                       # never permutes NA labels
            idx = np.flatnonzero((strat == s))
            lab[idx] = ei[idx[rng.permutation(idx.size)]]
        dl[k] = leads_of(lab, tested)
        dwm[k] = leads_of(lab, [6])[0]
        ds[k] = shares_of(lab)
    res = {f"lead_{lnames[L]}": emp(dl[:, j], obs_l[j])
           for j, L in enumerate(tested)}
    res["lead_mean_L1_L6"] = emp(dl.mean(axis=1), obs_l.mean())
    res["backward_share_exc"] = emp(ds[:, 0], obs_s[0])
    res["backward_share_inh"] = emp(ds[:, 1], obs_s[1])
    res["backward_share_diff"] = emp(ds[:, 2], obs_s[2])
    REPORT["null_T2_within_layer"] = res
    DRAWS["T2_leads"] = dl
    DRAWS["T2_shares"] = ds
    print("[null T2] mean-lead:", res["lead_mean_L1_L6"])

    # ---- REDTEAM 3e: include WM in the lead statistic ---------------------
    mean7_obs = float(np.append(obs_l, obs_wm).mean())
    mean7_null = np.column_stack([dl, dwm]).mean(axis=1)
    REPORT["redteam_T2_with_WM"] = dict(
        observed_lead_WM=float(obs_wm),
        n_exc_WM=int(((layer == 6) & (ei == 1)).sum()),
        n_inh_WM=int(((layer == 6) & (ei == 2)).sum()),
        lead_WM=emp(dwm, obs_wm),
        lead_mean_incl_WM=dict(emp(mean7_null, mean7_obs)))
    print("[redteam T2/WM]", REPORT["redteam_T2_with_WM"]["lead_mean_incl_WM"])


# ===========================================================================
# NPZ internal consistency: study's stored draws vs study's JSON
# ===========================================================================
def npz_section():
    ok, bad = 0, []

    def cmp(tag, st, null, obs):
        nonlocal ok
        r = emp(null, obs)
        for f in ("null_mean", "null_sd", "z", "p_two_sided", "n_le", "n_ge"):
            a, b = r[f], st[f]
            if not np.isclose(a, b, rtol=1e-9, atol=1e-12):
                bad.append((tag, f, a, b))
                return
        ok += 1

    fj = json.loads((REPO_ROOT / "biology/results/label_nulls_fly.json").read_text())
    fn = np.load(REPO_ROOT / "biology/results/label_nulls_fly_nulls.npz")
    meta = json.loads((REPO_ROOT / "layering/outputs/fly_ann/node_ann_meta.json")
                      .read_text())
    nt_names = meta["top_nt_values"]
    for null_name, key in [("global", "t1_global"),
                           ("stratified_super_class",
                            "t1_stratified_super_class")]:
        for i, nm in enumerate(nt_names):
            cmp(f"t1/{null_name}/{nm}",
                fj["t1_nt_backward_share"]["tables"][null_name][nm],
                fn[key][:, i], fn["t1_observed"][i])
    t3 = fj["t3_type_contiguity"]
    cmp("t3/median", t3["median_statistic"], fn["t3_null_median"],
        t3["median_statistic"]["observed"])
    n_pct1 = int((fn["t3_obs_iqr"] < fn["t3_null_pct1"]).sum())
    if n_pct1 != t3["n_tighter_than_null_pct1"]:
        bad.append(("t3/n_tighter_pct1", n_pct1, t3["n_tighter_than_null_pct1"]))
    else:
        ok += 1

    mj = json.loads((REPO_ROOT / "biology/results/label_nulls_microns.json")
                    .read_text())
    mn = np.load(REPO_ROOT / "biology/results/label_nulls_microns_nulls.npz")
    t2 = mj["t2_ei_shuffle"]
    layers = t2["tested_layers"]
    for null_name, lk, sk in [("global", "t2_leads_global", "t2_shares_global"),
                              ("stratified_layer", "t2_leads_stratified_layer",
                               "t2_shares_stratified_layer")]:
        for j, nm in enumerate(layers):
            cmp(f"t2/{null_name}/lead_{nm}",
                t2["tables"][null_name][f"lead_{nm}"],
                mn[lk][:, j], mn["t2_observed_leads"][j])
        cmp(f"t2/{null_name}/lead_mean", t2["tables"][null_name]
            ["lead_mean_over_layers"], mn[lk].mean(axis=1),
            mn["t2_observed_leads"].mean())
        for i, nm in enumerate(["backward_share_exc", "backward_share_inh",
                                "backward_share_diff_E_minus_I"]):
            cmp(f"t2/{null_name}/{nm}", t2["tables"][null_name][nm],
                mn[sk][:, i], mn["t2_observed_shares"][i])
    REPORT["npz_internal_consistency"] = dict(n_checked=ok + len(bad),
                                              n_ok=ok, mismatches=bad)
    print(f"[npz] {ok} stats consistent, {len(bad)} mismatches", bad or "")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_section()
    fly_section()
    microns_section()
    REPORT["seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / "review_label_nulls.json").write_text(
        json.dumps(REPORT, indent=1, default=float))
    np.savez_compressed(OUT_DIR / "review_label_nulls_draws.npz", **DRAWS)
    print(f"wrote {OUT_DIR}/review_label_nulls.json ({REPORT['seconds']}s)")


if __name__ == "__main__":
    main()
