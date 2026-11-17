"""Biological readings of the two fly orders, same code for FlyWire (female) and Male CNS.

Ports the FlyWire stream readings of layering/FLY_FINDINGS.md that had no statistic yet, and
adds comparisons that need two flies:
  hemi   per-lane ipsilateral share of weight (FLY_FINDINGS 4, "two eyes", was by eye only)
  vc     visual centrifugal: FF share of inputs vs outputs (FLY_FINDINGS 7)
  recip  where reciprocity (2-cycle floor) lives, by lane pair
  types  per-type feedback share of output, cross-sex correlation and biggest changes
  paths  lane pairs whose FF share changed most between the two flies
  sex    position and feedback share of sex-specific / fru-dsx neurons vs same-lane controls
  mb     mushroom-body roles in order (olfactory PN -> KC -> MBON, DAN, APL)
  vnc    male only: neuromere order and motor subclasses along the order

    $PY male_cns/analysis/biology_diff.py
Output: male_cns/results/biology_diff.json, male_cns/results/type_fb_share_cross_sex.csv,
        male_cns/results/lane_pair_change.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "male_cns" / "results"
sys.path.insert(0, str(ROOT / "male_cns" / "analysis"))
from analyze import load  # noqa: E402

CONFIG = dict(seed=42, n_perm=2000, min_type_out_weight=200, min_lane_pair_share=0.002,
              datasets=["flywire", "malecns_brain", "malecns"],
              optic_lanes=["photoreceptors", "lamina", "medulla", "medulla→lobula", "lobula (plate)",
                           "optic (other)"],
              mb_roles={"ALPN": "olfactory PN", "Kenyon_Cell": "Kenyon cell", "MBON": "MBON",
                        "DAN": "DAN", "MBIN": "MBIN"})
rng = np.random.default_rng(CONFIG["seed"])


def node_table(ds):
    """Graph arrays plus per-node frame with rank, lane, side, type, NT, sex flags, class."""
    src, tgt, w, rank, a = load(ds, None)
    a = a.copy()
    a["rn"] = rank / (len(rank) - 1)
    if ds == "flywire":
        f = pd.read_csv(ROOT / "data/external/flywire_neuron_annotations.tsv.gz", sep="\t", low_memory=False,
                        usecols=["root_id", "dimorphism", "fru_dsx", "cell_class"]).drop_duplicates("root_id")
        a = a.drop(columns=["cell_class"]).merge(f.rename(columns={"root_id": "node"}), on="node", how="left")
        a["sexflag"] = a.dimorphism.isin(["sexually dimorphic", "female-specific", "potentially sexually dimorphic",
                                          "potentially female-specific"]) | a.fru_dsx.notna()
        a["role_class"] = a.cell_class
        a["neuromere"] = np.nan
        a["motor_sub"] = np.nan
    else:
        nodes = pd.read_csv(ROOT / "data/processed/malecns_nodes.csv.gz", low_memory=False,
                            usecols=["bodyId", "subclass", "somaNeuromere", "dimorphism", "fruDsx"])
        a = a.drop(columns=["dimorphism", "fruDsx"], errors="ignore").merge(nodes, on="bodyId", how="left")
        a["sexflag"] = a.dimorphism.notna() | a.fruDsx.notna()
        a["role_class"] = a["class"]
        a["neuromere"] = a.somaNeuromere
        a["motor_sub"] = np.where(a.superclass == "vnc_motor", a.subclass, np.nan)
    assert len(a) == len(rank)
    return src, tgt, w, rank, a


def hemi(src, tgt, w, a):
    """Per source lane: share of weight to targets on the same side (both sides annotated L/R)."""
    side = a.side.to_numpy()
    lr = np.isin(side, ["left", "right"])
    m = lr[src] & lr[tgt]
    same = side[src] == side[tgt]
    lane = a.lane.to_numpy()
    d = pd.DataFrame({"lane": lane[src[m]], "w": w[m], "same": w[m] * same[m]})
    g = d.groupby("lane").sum()
    out = (g.same / g.w).round(4).to_dict()
    opt = np.isin(lane[src], CONFIG["optic_lanes"]) & np.isin(lane[tgt], CONFIG["optic_lanes"]) & m
    cen = (lane[src] == "central (other)") & (lane[tgt] == "central (other)") & m
    return dict(ipsi_share_by_source_lane=out,
                optic_to_optic_ipsi=float(w[opt & same].sum() / w[opt].sum()),
                central_to_central_ipsi=float(w[cen & same].sum() / w[cen].sum()))


def visual_centrifugal(src, tgt, w, a, fb):
    """FF share of the inputs and outputs of the visual centrifugal lane."""
    vc = (a.lane == "visual centrifugal").to_numpy()
    i, o = vc[tgt], vc[src]
    return dict(n=int(vc.sum()), median_rank=float(np.median(a.rn[vc])),
                input_ff_share=float(1 - w[i & fb].sum() / w[i].sum()),
                output_ff_share=float(1 - w[o & fb].sum() / w[o].sum()))


def reciprocity_by_lane(src, tgt, w, a, n):
    """2-cycle floor min(w_uv, w_vu) summed per unordered lane pair, as share of the floor."""
    key = src * n + tgt
    order = np.argsort(key)
    ks = key[order]
    pos = np.searchsorted(ks, tgt * n + src)
    pos[pos >= len(ks)] = 0
    has = ks[pos] == tgt * n + src
    rw = np.zeros(len(w), dtype=np.int64)
    rw[has] = w[order][pos[has]]
    m = has & (src < tgt)
    fl = np.minimum(w, rw)[m]
    lane = a.lane.to_numpy()
    la, lb = lane[src[m]], lane[tgt[m]]
    pair = np.where(la <= lb, la + " | " + lb, lb + " | " + la)
    s = pd.Series(fl).groupby(pair).sum().sort_values(ascending=False)
    same = lane[src] == lane[tgt]
    in_lane_w = pd.Series(w[same]).groupby(lane[src][same]).sum()
    out_lane_w = pd.Series(w).groupby(lane[src]).sum()
    within = {l: float(s.get(f"{l} | {l}", 0) / in_lane_w[l]) for l in in_lane_w.index}
    return dict(floor_share_by_lane_pair=(s / s.sum()).round(4).head(10).to_dict(),
                within_lane_floor_per_within_lane_weight=dict(sorted(within.items(), key=lambda kv: -kv[1])),
                within_lane_weight_share_of_lane_output=(in_lane_w / out_lane_w.reindex(in_lane_w.index)).round(4).to_dict())


def per_type_fb(src, tgt, w, a, fb):
    """Per type: share of output weight that is feedback; type keyed by FlyWire type name."""
    t = a.flywire_type.to_numpy()
    out = pd.DataFrame({"t": t[src], "w": w, "fb": w * fb}).dropna().groupby("t").sum()
    out = out[out.w >= CONFIG["min_type_out_weight"]]
    lanes = a.groupby("flywire_type").lane.agg(lambda s: s.mode().iat[0])
    return pd.DataFrame({"fb_share": out.fb / out.w, "out_w": out.w, "lane": lanes.reindex(out.index)})


def sex_test(src, tgt, w, a, fb):
    """Feedback share of output: sex-flagged neurons vs same-lane controls.

    The flag is a type property, so the null shuffles the flag over whole types within each lane
    (a neuron-level shuffle would understate the spread).
    """
    node_out = np.bincount(src, weights=w.astype(float), minlength=len(a))
    node_fb = np.bincount(src[fb], weights=w[fb].astype(float), minlength=len(a))
    flag = a.sexflag.to_numpy()
    key = a.cell_type.fillna(pd.Series("untyped_" + a.index.astype(str), index=a.index)).to_numpy()
    tdf = pd.DataFrame({"key": key, "lane": a.lane.to_numpy(), "out": node_out, "fb": node_fb, "flag": flag})
    t = tdf.groupby("key").agg(lane=("lane", lambda s: s.mode().iat[0]), out=("out", "sum"), fb=("fb", "sum"),
                               flag=("flag", "max"))
    tflag = t.flag.to_numpy()
    tout, tfb = t.out.to_numpy(), t.fb.to_numpy()
    groups = [np.where(t.lane.to_numpy() == l)[0] for l in t.lane.unique()]
    obs = node_fb[flag].sum() / node_out[flag].sum()
    null = []
    for _ in range(CONFIG["n_perm"]):
        p = tflag.copy()
        for g in groups:
            p[g] = tflag[rng.permutation(g)]
        null.append(tfb[p].sum() / tout[p].sum())
    null = np.array(null)
    return dict(n_flagged=int(flag.sum()), n_flagged_types=int(tflag.sum()), fb_share_flagged=float(obs),
                null_mean=float(null.mean()), z=float((obs - null.mean()) / null.std()),
                p_two_sided=float(np.mean(np.abs(null - null.mean()) >= abs(obs - null.mean()))), median_rank_flagged=float(np.median(a.rn[flag])),
                lanes_flagged=a.lane[flag].value_counts().head(5).to_dict())


def mb_roles(a):
    """Median position of olfactory PN, KC, MBON, DAN, MBIN, APL."""
    out = {}
    for k, v in CONFIG["mb_roles"].items():
        m = (a.role_class == k).to_numpy()
        if m.any():
            out[v] = dict(n=int(m.sum()), median=round(float(np.median(a.rn[m])), 4))
    apl = (a.cell_type == "APL").to_numpy()
    out["APL"] = dict(n=int(apl.sum()), median=round(float(np.median(a.rn[apl])), 4) if apl.any() else None)
    return out


def vnc(a):
    """Male only: median position by soma neuromere (VNC lanes) and by motor subclass."""
    v = a[a.region == "VNC"]
    nm = v.groupby("neuromere").rn.agg(["median", "size"])
    nm = nm[nm["size"] >= 50].sort_values("median")
    ms = a.dropna(subset=["motor_sub"]).groupby("motor_sub").rn.agg(["median", "size"]).sort_values("median")
    lane_nm = v.groupby(["lane", "neuromere"]).rn.median().unstack()
    return dict(by_neuromere={k: dict(median=round(float(r["median"]), 4), n=int(r["size"])) for k, r in nm.iterrows()},
                motor_by_subclass={k: dict(median=round(float(r["median"]), 4), n=int(r["size"])) for k, r in ms.iterrows()},
                vnc_intrinsic_by_thoracic_neuromere={k: round(float(lane_nm.loc["VNC intrinsic", k]), 4)
                                                     for k in ["T1", "T2", "T3"] if k in lane_nm.columns})


R, types, lanes_mat = {}, {}, {}
for ds in CONFIG["datasets"]:
    src, tgt, w, rank, a = node_table(ds)
    n = len(rank)
    fb = rank[src] > rank[tgt]
    r = dict(hemi=hemi(src, tgt, w, a), visual_centrifugal=visual_centrifugal(src, tgt, w, a, fb),
             reciprocity=reciprocity_by_lane(src, tgt, w, a, n), sex=sex_test(src, tgt, w, a, fb),
             mb=mb_roles(a))
    if ds == "malecns":
        r["vnc"] = vnc(a)
    R[ds] = r
    types[ds] = per_type_fb(src, tgt, w, a, fb)
    print(ds, "done", flush=True)

# Per-type feedback share, female vs male brain
tf, tm = types["flywire"], types["malecns_brain"]
j = tf.join(tm, lsuffix="_f", rsuffix="_m", how="inner")
j["delta"] = j.fb_share_m - j.fb_share_f
j.sort_values("delta").to_csv(RES / "type_fb_share_cross_sex.csv")
R["types_cross_sex"] = dict(
    types=int(len(j)), spearman=float(j.fb_share_f.corr(j.fb_share_m, method="spearman")),
    pearson=float(j.fb_share_f.corr(j.fb_share_m)),
    both_fb_heavy_top=j[(j.fb_share_f > 0.5) & (j.fb_share_m > 0.5)].sort_values("out_w_m", ascending=False)
    .head(12)[["lane_f", "fb_share_f", "fb_share_m"]].round(3).reset_index().to_dict("records"),
    more_fb_in_male=j.nlargest(10, "delta")[["lane_f", "fb_share_f", "fb_share_m"]].round(3).reset_index().to_dict("records"),
    more_fb_in_female=j.nsmallest(10, "delta")[["lane_f", "fb_share_f", "fb_share_m"]].round(3).reset_index().to_dict("records"),
    by_lane_spearman={l: round(float(d.fb_share_f.corr(d.fb_share_m, method="spearman")), 3)
                      for l, d in j.groupby("lane_f") if len(d) > 20})

# Lane pairs whose FF share changed most (weighted by their share of weight in both)
lf = pd.read_csv(RES / "lane_matrix_flywire.csv").set_index(["pre", "post"])
lm = pd.read_csv(RES / "lane_matrix_malecns_brain.csv").set_index(["pre", "post"])
p = lf[["ff_share", "share_of_W", "share_of_FB"]].join(lm[["ff_share", "share_of_W", "share_of_FB"]],
                                                     lsuffix="_f", rsuffix="_m", how="inner")
p = p[(p.share_of_W_f >= CONFIG["min_lane_pair_share"]) & (p.share_of_W_m >= CONFIG["min_lane_pair_share"])]
p["ff_change_pp"] = (p.ff_share_m - p.ff_share_f) * 100
p.sort_values("ff_change_pp").to_csv(RES / "lane_pair_change.csv")
R["lane_pairs"] = dict(n_pairs=int(len(p)),
                       less_ff_in_male=p.nsmallest(8, "ff_change_pp").round(4).reset_index().to_dict("records"),
                       more_ff_in_male=p.nlargest(8, "ff_change_pp").round(4).reset_index().to_dict("records"))
(RES / "biology_diff.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps({k: R[k] for k in ["types_cross_sex"]}, indent=1, default=str)[:3000])
