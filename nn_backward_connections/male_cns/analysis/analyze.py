"""Order-based analyses of one fly connectome, same code for FlyWire (female) and Male CNS.

Re-implements, in one dataset-agnostic script, the FlyWire analyses behind biology/FINDINGS.md
(A2, A3, A4, A5 swap, B1, B2, B5, B7, 2-cycle floor of A1). Running it on FlyWire re-checks the
port against the published FlyWire numbers; running it on Male CNS gives the comparison.

    $PY male_cns/analysis/analyze.py --dataset flywire
    $PY male_cns/analysis/analyze.py --dataset malecns [--rank male_cns/results/malecns_H64_s42_rank.npy]
    $PY male_cns/analysis/analyze.py --dataset malecns_brain   # brain-only subgraph, restricted order

Output: male_cns/results/analysis_<dataset>.json (+ lane matrix CSV, per-type ranks CSV)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "layering" / "tools"))
from mfas.io import load_connectome  # noqa: E402
from mfas.metrics import pct, score_from_order  # noqa: E402

CONFIG = dict(
    seed=42,
    n_perm=2000,           # NT and flow permutation nulls (as fly_streams / exp_label_nulls)
    n_type_draws=1000,     # type contiguity null draws (exp_label_nulls T3)
    min_type_size=20,
    top_k_nodes=(1, 10, 100, 1000),
    near_frac=0.017,       # "weight within +-1.7 % of n" (gallery_fly report)
    hub_top=12,
    flywire_rank="results/champions/H64_connectome_s42_cuda.npz",
    malecns_rank="male_cns/results/malecns_H64_s42_rank.npy",
    flywire_ann="data/external/flywire_neuron_annotations.tsv.gz",
    malecns_ann="data/processed/malecns_ann.csv.gz",
)
NT_SHORT = {"acetylcholine": "ACh", "gaba": "GABA", "glutamate": "Glu", "dopamine": "DA",
            "serotonin": "5-HT", "octopamine": "OA"}


def load(dataset, rank_path):
    """Return src, tgt, w (int64), rank (int64) and a per-node annotation frame."""
    if dataset == "flywire":
        from fly_streams import lane_of
        g = load_connectome()
        src, tgt, w, ids = g.src.astype(np.int64), g.tgt.astype(np.int64), g.weight, g.node_ids
        rank = np.load(ROOT / (rank_path or CONFIG["flywire_rank"]))["rank"].astype(np.int64)
        f = pd.read_csv(ROOT / CONFIG["flywire_ann"], sep="\t", low_memory=False,
                        usecols=["root_id", "flow", "super_class", "cell_class", "cell_type", "side",
                                 "top_nt"]).drop_duplicates("root_id")
        a = pd.DataFrame({"node": np.asarray(ids, dtype=np.int64)}).merge(
            f.rename(columns={"root_id": "node"}), on="node", how="left")
        a["lane"] = [lane_of(s if isinstance(s, str) else "", c if isinstance(c, str) else "")
                     for s, c in zip(a.super_class, a.cell_class)]
        a["top_nt"] = a.top_nt.map(lambda x: NT_SHORT.get(x, x))
        a.loc[a.cell_class == "Kenyon_Cell", "top_nt"] = "ACh"
        a["superclass"] = a.super_class
        a["region"] = "brain"
        a["side"] = a.side.fillna("na")
        a["flywire_type"] = a.cell_type
        return src, tgt, w, rank, a
    z = np.load(ROOT / "data/processed/malecns.npz")
    src, tgt, w = z["src"].astype(np.int64), z["tgt"].astype(np.int64), z["weight"]
    rank = np.load(ROOT / (rank_path or CONFIG["malecns_rank"])).astype(np.int64)
    a = pd.read_csv(ROOT / CONFIG["malecns_ann"], low_memory=False)
    a["flywire_type"] = a.flywireType
    if dataset == "malecns_brain":
        keep = (a.region == "brain").to_numpy()
        new = -np.ones(len(a), dtype=np.int64)
        new[keep] = np.arange(keep.sum())
        e = keep[src] & keep[tgt]
        src, tgt, w = new[src[e]], new[tgt[e]], w[e]
        sub = rank[keep]
        rank = np.argsort(np.argsort(sub, kind="stable"), kind="stable").astype(np.int64)
        a = a[keep].reset_index(drop=True)
    return src, tgt, w, rank, a


def backward_share_by_label(lab, src, tgt, w, fb):
    """Share of each presynaptic label's output weight that is feedback."""
    out = {}
    s_lab = lab[src]
    for v in pd.unique(lab[~pd.isna(lab)]):
        m = s_lab == v
        tot = w[m].sum()
        out[v] = float(w[m & fb].sum() / tot) if tot > 0 else float("nan")
    return out


def perm_nt_test(nt, strata, src, tgt, w, fb, rng, n_perm):
    """Within-stratum label permutation null for per-NT backward share (exp_label_nulls T1)."""
    nodes_out = np.bincount(src, weights=w.astype(np.float64), minlength=len(nt))
    nodes_fb = np.bincount(src[fb], weights=w[fb].astype(np.float64), minlength=len(nt))
    labs = [v for v in pd.unique(nt[~pd.isna(nt)])]
    obs = {v: nodes_fb[nt == v].sum() / nodes_out[nt == v].sum() for v in labs}
    valid = ~pd.isna(nt)
    groups = [np.where(valid & (strata == s))[0] for s in pd.unique(strata[valid])]
    null = {v: [] for v in labs}
    for _ in range(n_perm):
        perm = nt.copy()
        for gi in groups:
            perm[gi] = nt[rng.permutation(gi)]
        for v in labs:
            m = perm == v
            null[v].append(nodes_fb[m].sum() / nodes_out[m].sum())
    res = {}
    for v in labs:
        x = np.array(null[v])
        res[v] = dict(n=int((nt == v).sum()), obs=float(obs[v]), null_mean=float(x.mean()),
                      z=float((obs[v] - x.mean()) / x.std()) if x.std() > 0 else float("nan"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["flywire", "malecns", "malecns_brain"], required=True)
    ap.add_argument("--rank", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(CONFIG["seed"])
    src, tgt, w, rank, a = load(args.dataset, args.rank)
    n, m, W = len(rank), len(w), int(w.sum())
    rn = rank / (n - 1)
    fb = rank[src] > rank[tgt]
    ff_score = score_from_order(rank, src, tgt, w)
    R = dict(dataset=args.dataset, n=n, m=m, total_weight=W, ff_pct=pct(ff_score, W),
             fb_weight_pct=float(w[fb].sum() / W * 100))

    # 2-cycle floor: every reciprocal pair forces at least min(w_uv, w_vu) into feedback.
    key = src * n + tgt
    rkey = tgt * n + src
    order = np.argsort(key)
    pos = np.searchsorted(key[order], rkey)
    pos[pos >= m] = 0
    has = key[order][pos] == rkey
    rw = np.zeros(m, dtype=np.int64)
    rw[has] = w[order][pos[has]]
    floor = np.minimum(w, rw)[src < tgt].sum()
    R["reciprocal_edge_share"] = float(has.mean())
    R["two_cycle_floor_pct_of_W"] = float(floor / W * 100)
    R["two_cycle_floor_share_of_fb"] = float(floor / w[fb].sum())

    # A2: feedback concentration on nodes (FB weight touching a node, top-k share)
    fbw = w[fb].astype(np.float64)
    node_fb = (np.bincount(src[fb], weights=fbw, minlength=n) + np.bincount(tgt[fb], weights=fbw, minlength=n)) / 2
    srt = np.sort(node_fb)[::-1]
    R["fb_top_k_node_share_pct"] = {k: float(srt[:k].sum() / fbw.sum() * 100) for k in CONFIG["top_k_nodes"]}
    span = np.abs(rank[src] - rank[tgt]) / n
    R["fb_span_median_n"] = float(np.median(span[fb]))
    R["fb_weighted_span_median_n"] = float(np.median(np.repeat(span[fb], np.minimum(w[fb], 50))))
    R["weight_within_near_frac_pct"] = float(w[span <= CONFIG["near_frac"]].sum() / W * 100)

    # B1: strongly connected components
    A = coo_matrix((np.ones(m, dtype=np.int8), (src, tgt)), shape=(n, n)).tocsr()
    ncomp, lab = connected_components(A, directed=True, connection="strong")
    sizes = np.bincount(lab)
    giant = sizes.argmax()
    in_g = lab == giant
    R["scc_count"] = int(ncomp)
    R["giant_scc_size"] = int(sizes[giant])
    R["giant_scc_frac"] = float(sizes[giant] / n)
    R["fb_weight_inside_giant_scc_pct"] = float(w[fb & in_g[src] & in_g[tgt]].sum() / w[fb].sum() * 100)
    R["giant_scc_rank_span_frac"] = [float(rn[in_g].min()), float(rn[in_g].max())]

    # B5: zero-output sinks at the tail
    outdeg = np.bincount(src, minlength=n)
    sinks = outdeg == 0
    R["zero_output_sinks"] = int(sinks.sum())
    R["sinks_median_rank_frac"] = float(np.median(rn[sinks])) if sinks.any() else None
    R["sinks_lanes_top"] = {k: int(v) for k, v in a.lane[sinks].value_counts().head(6).items()}

    # A4: lane medians and flow test
    lane = a.lane.to_numpy()
    lanes = pd.DataFrame({"lane": lane, "r": rn}).groupby("lane").r.agg(["median", "size"]).sort_values("median")
    R["lane_median_rank_frac"] = {k: dict(median=float(v["median"]), n=int(v["size"])) for k, v in lanes.iterrows()}
    flow = a.flow.map({"afferent": 0, "intrinsic": 1, "efferent": 2}).to_numpy()
    ok = ~np.isnan(flow)

    def spearman(x, y):
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        return float(np.corrcoef(rx, ry)[0, 1])
    rho = spearman(flow[ok], rn[ok])
    fr = pd.Series(flow[ok]).rank().to_numpy()
    rr = pd.Series(rn[ok]).rank().to_numpy()
    null = [float(np.corrcoef(fr, rng.permutation(rr))[0, 1]) for _ in range(CONFIG["n_perm"])]
    R["flow_rho"] = rho
    R["flow_rho_null_max_abs"] = float(np.max(np.abs(null)))
    R["flow_median_rank_frac"] = {k: float(np.median(rn[(a.flow == k).to_numpy()]))
                                  for k in ["afferent", "intrinsic", "efferent"] if (a.flow == k).any()}

    # A4: type contiguity (IQR of normalized rank within type vs random sets of same size)
    t = a.cell_type.to_numpy()
    vc = pd.Series(t).value_counts()
    big = vc[vc >= CONFIG["min_type_size"]]
    iqr_obs, iqr_null = [], []
    for ty, k in big.items():
        r = rn[t == ty]
        iqr_obs.append(np.subtract(*np.percentile(r, [75, 25])))
    sizes_ = big.to_numpy()
    for _ in range(CONFIG["n_type_draws"]):
        for k in sizes_:
            r = rn[rng.choice(n, k, replace=False)]
            iqr_null.append(np.subtract(*np.percentile(r, [75, 25])))
    iqr_obs = np.array(iqr_obs)
    R["types_ge_min"] = int(len(big))
    R["type_iqr_median"] = float(np.median(iqr_obs))
    R["type_iqr_null_median"] = float(np.median(iqr_null))
    R["types_tighter_than_null_5pct"] = float((iqr_obs < np.percentile(iqr_null, 5)).mean())

    # A3: NT backward share, permutation nulls global and within superclass
    nt = a.top_nt.to_numpy(dtype=object)
    nt[pd.isna(nt)] = None
    nt_arr = np.array([x if x is not None else np.nan for x in nt], dtype=object)
    R["nt_backward_share_global_null"] = perm_nt_test(nt_arr, np.zeros(n), src, tgt, w, fb, rng, CONFIG["n_perm"])
    sc = a.superclass.fillna("na").to_numpy()
    R["nt_backward_share_within_superclass_null"] = perm_nt_test(nt_arr, sc, src, tgt, w, fb, rng, CONFIG["n_perm"])
    s_nt = nt_arr[src]
    inh = np.isin(s_nt, ["GABA", "Glu"])
    R["inhibitory_share_of_fb_pct"] = float(w[fb & inh].sum() / w[fb].sum() * 100)
    R["inhibitory_share_of_ff_pct"] = float(w[~fb & inh].sum() / w[~fb].sum() * 100)

    # B2: lane x lane feedback budget
    lane_codes, lane_idx = np.unique(lane, return_inverse=True)
    L = len(lane_codes)
    pair = lane_idx[src] * L + lane_idx[tgt]
    tot = np.bincount(pair, weights=w.astype(np.float64), minlength=L * L).reshape(L, L)
    fbm = np.bincount(pair[fb], weights=fbw, minlength=L * L).reshape(L, L)
    rows = []
    for i in range(L):
        for j in range(L):
            if tot[i, j] > 0:
                rows.append(dict(pre=lane_codes[i], post=lane_codes[j], weight=tot[i, j], fb=fbm[i, j],
                                 ff_share=1 - fbm[i, j] / tot[i, j], share_of_W=tot[i, j] / W,
                                 share_of_FB=fbm[i, j] / fbw.sum()))
    lm = pd.DataFrame(rows).sort_values("share_of_FB", ascending=False)
    out = ROOT / "male_cns" / "results"
    lm.to_csv(out / f"lane_matrix_{args.dataset}.csv", index=False)
    R["fb_within_same_lane_pct"] = float(np.trace(fbm) / fbw.sum() * 100)
    R["lane_pairs_top_fb"] = lm.head(8)[["pre", "post", "share_of_FB", "ff_share"]].round(4).to_dict("records")
    optic = {"lamina", "medulla", "medulla→lobula", "lobula (plate)", "optic (other)", "photoreceptors"}
    o = np.isin(lane[src], list(optic)) & np.isin(lane[tgt], list(optic))
    R["fb_inside_optic_lanes_pct"] = float(w[fb & o].sum() / fbw.sum() * 100)
    if (a.region == "VNC").any():
        vnc = (a.region == "VNC").to_numpy()
        R["fb_inside_vnc_pct"] = float(w[fb & vnc[src] & vnc[tgt]].sum() / fbw.sum() * 100)
        R["fb_brain_vnc_crossing_pct"] = float(w[fb & (vnc[src] != vnc[tgt])].sum() / fbw.sum() * 100)
        R["vnc_median_rank_frac"] = float(np.median(rn[vnc]))

    # B7: hubs (total degree) and top feedback senders
    deg = np.bincount(src, minlength=n) + np.bincount(tgt, minlength=n)
    top = np.argsort(deg)[::-1][:CONFIG["hub_top"]]
    R["degree_hubs"] = [dict(type=str(a.cell_type.iat[i]), nt=str(a.top_nt.iat[i]), lane=str(lane[i]),
                             degree=int(deg[i]), rank_frac=round(float(rn[i]), 4)) for i in top]
    R["degree_hubs_inhibitory_count"] = int(sum(str(a.top_nt.iat[i]) in ("GABA", "Glu") for i in top))
    fbs = np.bincount(src[fb], weights=fbw, minlength=n)
    tops = np.argsort(fbs)[::-1][:CONFIG["hub_top"]]
    R["top_fb_senders"] = [dict(type=str(a.cell_type.iat[i]), nt=str(a.top_nt.iat[i]),
                                fb_share_pct=round(float(fbs[i] / fbw.sum() * 100), 3)) for i in tops]

    # A5: mirror swap. Pair left[i] <-> right[i] within type (sorted by rank), swap their ranks.
    side = a.side.to_numpy()
    swapped = rank.copy()
    n_pairs = 0
    left_earlier = []
    for ty, idx in pd.Series(np.arange(n)).groupby(t).groups.items():
        idx = np.asarray(idx)
        l = idx[side[idx] == "left"]
        r = idx[side[idx] == "right"]
        k = min(len(l), len(r))
        if k == 0:
            continue
        l = l[np.argsort(rank[l])][:k]
        r = r[np.argsort(rank[r])][:k]
        swapped[l], swapped[r] = rank[r], rank[l]
        n_pairs += k
        if k >= 5:
            left_earlier.append(np.mean(rank[l] < rank[r]))  # ties (exactly 0.5) are dropped below
    sw = score_from_order(swapped, src, tgt, w)
    R["mirror_pairs"] = int(n_pairs)
    R["mirror_paired_neuron_frac"] = float(2 * n_pairs / n)
    R["mirror_swap_ff_pct"] = pct(sw, W)
    R["mirror_swap_cost_pp"] = pct(sw, W) - R["ff_pct"]
    le = np.array(left_earlier)
    le = le[le != 0.5]
    k_left = int((le > 0.5).sum())
    R["mirror_types_ge5_pairs_without_ties"] = int(len(le))
    R["mirror_types_left_earlier"] = k_left
    R["mirror_left_earlier_rate_types_ge5_pairs"] = float(k_left / len(le))
    R["mirror_left_earlier_binom_p"] = float(binomtest(k_left, len(le), 0.5).pvalue)

    # per-type median rank for cross-dataset comparison
    ft = a.flywire_type.to_numpy()
    pt = pd.DataFrame({"flywire_type": ft, "r": rn, "lane": lane}).dropna(subset=["flywire_type"])
    pt = pt.groupby("flywire_type").agg(median_rank=("r", "median"), n=("r", "size"),
                                         lane=("lane", lambda s: s.mode().iat[0]))
    pt.to_csv(out / f"type_ranks_{args.dataset}.csv")

    (out / f"analysis_{args.dataset}.json").write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps({k: v for k, v in R.items() if not isinstance(v, (dict, list))}, indent=1))


if __name__ == "__main__":
    main()
