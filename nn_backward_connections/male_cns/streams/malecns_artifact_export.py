"""Data bundle for the interactive Male CNS streams page (counterpart of
layering/tools/fly_artifact_export.py, same JSON schema).

Differences from the FlyWire export: lanes come from the `lane` column, the two raw groupings
are the Male CNS `class` and `superclass` columns (kept under the FlyWire keys `cell_class` and
`super_class` so the page code is unchanged), transmitter values include histamine, and two
extra blocks are added: `fw_lane_median_frac` (FlyWire median position of each lane, from
male_cns/results/analysis_flywire.json) and `compare` (female vs male headline numbers from
male_cns/results/analysis_*.json and vnc_loop.json).

    $PY male_cns/streams/malecns_artifact_export.py   (after malecns_streams.py)
Output: male_cns/results/streams/artifact_data.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from layering.tools.fly_artifact_export import b64  # noqa: E402
from layering.tools.fly_streams import build_layout, layout_stats  # noqa: E402
from malecns_streams import CONFIG as SCFG, LANE_COLORS, assign_lanes  # noqa: E402

CONFIG = {
    "graph": ROOT / "data/processed/malecns.npz",
    "ann_csv": ROOT / "data/processed/malecns_ann.csv.gz",
    "ann_npz": ROOT / "male_cns/results/streams/node_ann.npz",
    "ann_meta": ROOT / "male_cns/results/streams/node_ann_meta.json",
    "streams_dir": ROOT / "male_cns/results/streams",
    "analysis_flywire": ROOT / "male_cns/results/analysis_flywire.json",
    "analysis_malecns": ROOT / "male_cns/results/analysis_malecns.json",
    "vnc_loop": ROOT / "male_cns/results/vnc_loop.json",
    "out": ROOT / "male_cns/results/streams/artifact_data.json",
    "bins": [8, 16, 32],
    "top_types_per_cell": 5,
    "band_halfwidth": 0.40,
    # male lane -> FlyWire lane with the same neurons, where the name differs
    "fw_lane_alias": {"motor (brain)": "motor"},
}


def ordered_grouping(rank, codes, names, display=None):
    """Group values reordered by median rank -> (glane, glist)."""
    present = sorted(set(int(c) for c in codes), key=lambda i: float(np.median(rank[codes == i])))
    remap = {old: new for new, old in enumerate(present)}
    glane = np.array([remap[int(c)] for c in codes], dtype=np.int64)
    glist = [(display(names[i]) if display else names[i]) for i in present]
    return glane, glist


def family_colors(glane, glist, lane, lane_list):
    """Each group inherits its members' modal stream colour."""
    cols = []
    for i in range(len(glist)):
        vals, cnts = np.unique(lane[glane == i], return_counts=True)
        cols.append(LANE_COLORS[lane_list[vals[np.argmax(cnts)]]])
    return cols


def main():
    t0 = time.time()
    z = np.load(CONFIG["graph"])
    src, tgt = z["src"].astype(np.int64), z["tgt"].astype(np.int64)
    w = z["weight"].astype(np.float64)
    d = np.load(CONFIG["ann_npz"])
    meta = json.loads(CONFIG["ann_meta"].read_text(encoding="utf-8"))
    rank = d["rank"]
    n = rank.shape[0]
    Wtot = float(w.sum())
    ff_pct = meta["champion_ff_pct"]
    out_w = np.bincount(src, weights=w, minlength=n)
    lane, lane_list = assign_lanes(rank, d["lane"].astype(np.int64), meta["lane_values"])
    K = len(lane_list)

    # cell_type per node; untyped neurons get a marked class-level label (as FlyWire)
    a = pd.read_csv(CONFIG["ann_csv"], low_memory=False, usecols=["idx", "cell_type", "class", "superclass"])
    assert np.array_equal(a.idx.to_numpy(), np.arange(n))
    types_by_node = []
    for t, c, s in zip(a.cell_type, a["class"], a.superclass):
        if isinstance(t, str) and t:
            types_by_node.append(t)
        else:
            c = c if isinstance(c, str) else s
            types_by_node.append(f"{c} (class)" if isinstance(c, str) else "unannotated")

    grp_defs = {"streams": (lane, lane_list, [LANE_COLORS[nm] for nm in lane_list])}
    cl_lane, cl_list = ordered_grouping(rank, d["class"].astype(np.int64), meta["class_values"],
                                        display=lambda s: "NA (no class)" if s == "NA" else s)
    grp_defs["cell_class"] = (cl_lane, cl_list, family_colors(cl_lane, cl_list, lane, lane_list))
    su_lane, su_list = ordered_grouping(rank, d["superclass"].astype(np.int64), meta["superclass_values"])
    grp_defs["super_class"] = (su_lane, su_list, family_colors(su_lane, su_list, lane, lane_list))

    fw_med = json.loads(CONFIG["analysis_flywire"].read_text())["lane_median_rank_frac"]
    fw_lane_median_frac = {}
    for L in lane_list:
        key = CONFIG["fw_lane_alias"].get(L, L)
        if key in fw_med:
            fw_lane_median_frac[L] = fw_med[key]["median"]

    thr_ink, thr_exp = SCFG["min_bundle_frac"], SCFG["export_bundle_frac"]
    hw = CONFIG["band_halfwidth"]
    spec = d["spec_y"].astype(np.float64)
    layouts, cloud_groups = {}, {}
    for gname, (glane, glist, gcols) in grp_defs.items():
        Kg = len(glist)
        ycl = np.zeros(n)
        for i in range(Kg):
            m = glane == i
            r = np.argsort(np.argsort(spec[m], kind="stable"), kind="stable")
            ycl[m] = i + hw * (2.0 * r / max(int(m.sum()) - 1, 1) - 1.0)
        y01 = (ycl - ycl.min()) / (ycl.max() - ycl.min())
        cloud_groups[gname] = {"K": Kg, "y_min": float(ycl.min()), "y_max": float(ycl.max()),
                               "y_u16": b64(np.round(y01 * 65535).astype(np.uint16)),
                               "lane_u8": b64(glane.astype(np.uint8))}
        layouts[gname] = {}
        for B in CONFIG["bins"]:
            L = build_layout(rank, glane, glist, src, tgt, w, B, out_w)
            stats = layout_stats(L, Wtot, thr_ink * Wtot)
            dbin = np.minimum((rank * B) // n, B - 1).astype(np.int64)
            cellid = glane * B + dbin
            per_cell = {}
            for ci, t in zip(cellid, types_by_node):
                per_cell.setdefault(int(ci), Counter())[t] += 1
            NC = Kg * B
            tot = {k: np.zeros(NC) for k in ("out_ff", "out_fb", "in_ff", "in_fb")}
            for s_, t_, wf, wb in L["bundles"]:
                tot["out_ff"][s_] += wf
                tot["out_fb"][s_] += wb
                tot["in_ff"][t_] += wf
                tot["in_fb"][t_] += wb
            cells = []
            for c in np.nonzero(L["counts"])[0]:
                top = per_cell.get(int(c), Counter()).most_common(CONFIG["top_types_per_cell"])
                cells.append({"lane": int(c // B), "bin": int(c % B), "count": int(L["counts"][c]),
                              "intra_ff": float(L["intra_ff"][c]), "intra_fb": float(L["intra_fb"][c]),
                              "sink0": int(L["sink0"][c]),
                              "out_ff": float(tot["out_ff"][c]), "out_fb": float(tot["out_fb"][c]),
                              "in_ff": float(tot["in_ff"][c]), "in_fb": float(tot["in_fb"][c]),
                              "top_types": [[t, int(cnt)] for t, cnt in top]})
            lanes = []
            for i in range(Kg):
                lj = {"name": glist[i], "color": gcols[i], "n": int((glane == i).sum()),
                      "med_rank": float(np.median(rank[glane == i]))}
                if gname == "streams" and glist[i] in fw_lane_median_frac:
                    lj["fw_med_frac"] = fw_lane_median_frac[glist[i]]
                lanes.append(lj)
            layouts[gname][f"B{B}"] = {
                "n_bins": B, "ink_threshold_frac": thr_ink, "export_floor_frac": thr_exp,
                "lanes": lanes, "cells": cells,
                "bundles": [{"s": s_, "t": t_, "ff": round(wf, 2), "fb": round(wb, 2)}
                            for s_, t_, wf, wb in L["bundles"] if wf + wb >= thr_exp * Wtot],
                "stats": stats,
            }

    type_names = sorted(set(types_by_node))
    type_idx = {t: i for i, t in enumerate(type_names)}
    assert len(type_names) < 65536
    type_u16 = np.array([type_idx[t] for t in types_by_node], dtype=np.uint16)
    type_counts = np.bincount(type_u16, minlength=len(type_names))

    nt_codes = d["top_nt"].astype(np.int64)
    ntv = meta["top_nt_values"]
    NNT = len(ntv)
    ff_edge = rank[src] < rank[tgt]
    nt_src = nt_codes[src]
    lane_nt = lane[src] * NNT + nt_src
    nt_stats = {
        "values": ntv,
        "n_neurons": np.bincount(nt_codes, minlength=NNT).tolist(),
        "w_ff": np.bincount(nt_src[ff_edge], weights=w[ff_edge], minlength=NNT).tolist(),
        "w_fb": np.bincount(nt_src[~ff_edge], weights=w[~ff_edge], minlength=NNT).tolist(),
        "lane_out_ff": np.bincount(lane_nt[ff_edge], weights=w[ff_edge], minlength=K * NNT).reshape(K, NNT).tolist(),
        "lane_out_fb": np.bincount(lane_nt[~ff_edge], weights=w[~ff_edge], minlength=K * NNT).reshape(K, NNT).tolist(),
        "note": "absolute synapse weight by SOURCE neuron's predicted transmitter (Male CNS "
                "consensus_nt); ff by the node-level order",
    }

    sidev = meta["side_values"]
    side_map = np.array([1 if v == "left" else (2 if v == "right" else 0) for v in sidev], dtype=np.uint8)
    cloud = {"n": int(n), "x_u32": b64(rank.astype(np.uint32)),
             "sink_u8": b64((out_w == 0).astype(np.uint8)),
             "side_u8": b64(side_map[d["side"].astype(np.int64)]),
             "type_u16": b64(type_u16), "nt_u8": b64(nt_codes.astype(np.uint8)),
             "groups": cloud_groups}

    report = json.loads((CONFIG["streams_dir"] / "report_streams.json").read_text(encoding="utf-8"))
    lane_matrix = json.loads((CONFIG["streams_dir"] / "lane_matrix.json").read_text(encoding="utf-8"))
    assert lane_matrix["lane_list"] == lane_list
    af = json.loads(CONFIG["analysis_flywire"].read_text())
    am = json.loads(CONFIG["analysis_malecns"].read_text())
    vl = json.loads(CONFIG["vnc_loop"].read_text())["blocks"]
    keys = ["n", "m", "total_weight", "ff_pct", "reciprocal_edge_share", "two_cycle_floor_pct_of_W",
            "two_cycle_floor_share_of_fb", "inhibitory_share_of_fb_pct", "inhibitory_share_of_ff_pct",
            "zero_output_sinks"]
    compare = {"flywire": {k: af[k] for k in keys}, "malecns": {k: am[k] for k in keys},
               "brain_cord_blocks": vl,
               "source": "male_cns/results/analysis_flywire.json, analysis_malecns.json, vnc_loop.json"}

    data = {
        "generated": time.strftime("%Y-%m-%d"),
        "dataset": "malecns",
        "champion_ff_pct": ff_pct,
        "ff_weight": meta["ff_weight"],
        "W_total": Wtot, "n": int(n), "m": int(src.shape[0]),
        "lane_list": lane_list,
        "layouts": layouts, "cloud": cloud,
        "type_names": type_names, "type_counts": type_counts.tolist(),
        "lane_matrix": lane_matrix,
        "flow_hierarchy_test": report["flow_hierarchy_test"],
        "nt_direction": report["nt_direction"],
        "nt_stats": nt_stats,
        "sink_audit": report["sink_audit"],
        "lane_order_check": report["lane_order_check"],
        "spectral": meta["spectral"],
        "fw_lane_median_frac": fw_lane_median_frac,
        "compare": compare,
        "provenance": {"order": meta["order"], "annotation": meta["annotation"],
                       "graph": "data/processed/malecns.npz (male_cns/pipeline/build_graph.py, BUILD.md)"},
    }
    CONFIG["out"].write_text(json.dumps(data), encoding="utf-8")
    print(f"wrote {CONFIG['out']} ({round(CONFIG['out'].stat().st_size / 1e6, 1)} MB, {round(time.time() - t0)} s)")


if __name__ == "__main__":
    main()
