"""Assemble the data bundle for the interactive fly-connectome artifact.

Reads (read-only): fly_ann outputs, fly_streams layouts, fly_cloud positions,
the annotation TSV (for cell_type hover cards), and the graph itself.
Writes: layering/outputs/fly_artifact/artifact_data.json  — one JSON with
  * layouts for B in {8,16,32}: lanes, cells (+ top cell_types per cell),
    bundles, stats
  * cloud: per-neuron x (rank), y (quantized uint16 over lane bands), lane id,
    packed as base64 arrays; plus per-neuron flags (zero-out-weight)
  * global stats: hierarchy test, NT x direction, sink audit, provenance
  * edges: meta of the edge payload (see edge_payload)
and layering/outputs/fly_artifact/edges.zlib — the full edge list for the
page's what-if scorer (gitignored, regenerable).

Run:  $PY layering/tools/fly_artifact_export.py
"""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONFIG = {
    "ann_npz": HERE.parent / "outputs" / "fly_ann" / "node_ann.npz",
    "ann_meta": HERE.parent / "outputs" / "fly_ann" / "node_ann_meta.json",
    "ann_tsv": REPO / "data" / "external" / "flywire_neuron_annotations.tsv.gz",
    "streams_dir": HERE.parent / "outputs" / "fly_streams",
    "cloud_npz": HERE.parent / "outputs" / "fly_cloud" / "cloud_xy.npz",
    "out_dir": HERE.parent / "outputs" / "fly_artifact",
    "bins": [8, 16, 32],
    "top_types_per_cell": 5,
    "edges_zlib_level": 9,
}


def b64(arr) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def edge_payload(g, rank):
    """Encode the edge list for the page's exact what-if scorer.

    Endpoints are written as H64 ranks, edges sorted by (rank[src], rank[tgt]).
    Uncompressed layout, every value little-endian and byte-plane split
    (plane b = byte b of every value): deg uint32 (out-degree of the node with
    rank r, r = 0..N-1) | dtgt uint32 (target rank for the first edge of a
    source run, else the gap to the previous target, > 0) | w uint16.
    Returns (zlib bytes, meta); see layering/VIZ_V2_DESIGN.md section 2.
    """
    from src.mfas.metrics import score_from_order              # frozen oracle
    n, m = int(g.n_nodes), int(g.n_edges)
    rs = rank[g.src.astype(np.int64)]
    rt = rank[g.tgt.astype(np.int64)]
    w = g.weight
    assert not np.any(rs == rt), "self-loop in the edge list"
    assert np.all(w == np.round(w)) and w.min() >= 1 and w.max() < 2**16, \
        "weights do not fit uint16"
    o = np.lexsort((rt, rs))
    rs, rt, w16 = rs[o], rt[o], w[o].astype(np.uint16)
    deg = np.bincount(rs, minlength=n).astype(np.uint32)
    first = np.ones(m, dtype=bool)
    first[1:] = rs[1:] != rs[:-1]
    dtgt = np.diff(rt, prepend=0)
    dtgt[first] = rt[first]
    assert np.all(dtgt[~first] > 0), "duplicate (src, tgt) pair"

    def planes(a):
        return a.view(np.uint8).reshape(-1, a.itemsize).T.copy().tobytes()

    raw = planes(deg) + planes(dtgt.astype(np.uint32)) + planes(w16)
    assert len(raw) == 4 * n + 6 * m
    z = zlib.compress(raw, CONFIG["edges_zlib_level"])
    meta = {
        "n": n, "m": m,
        "layout": "deg_u32|dtgt_u32|w_u16, byte planes, LE, zlib",
        "raw_bytes": len(raw), "zlib_bytes": len(z),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "ff_h64_syn": int(score_from_order(rank, g.src, g.tgt, g.weight)),
    }
    return z, meta


def main():
    out = CONFIG["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from layering import io_utils
    from layering.tools.fly_streams import (assign_lanes, build_layout,
                                            layout_stats, lane_of, LANE_COLORS,
                                            CONFIG as SCFG)
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order("connectome")
    n = g.n_nodes
    src = g.src.astype(np.int64)
    w = g.weight.astype(np.float64)
    out_w = np.bincount(src, weights=w, minlength=n)

    d = np.load(CONFIG["ann_npz"])
    meta = json.loads(Path(CONFIG["ann_meta"]).read_text(encoding="utf-8"))
    assert np.array_equal(d["rank"], rank), "annotation cache is stale"
    lane, lane_list = assign_lanes(rank, d["super_class"], d["cell_class"],
                                   meta["super_class_values"],
                                   meta["cell_class_values"])
    K = len(lane_list)

    # cell_type per node (hover cards) — read once from the TSV
    node_ids = g.node_ids.astype(np.int64)
    ctype = {}
    opener = gzip.open if str(CONFIG["ann_tsv"]).endswith(".gz") else open
    with opener(CONFIG["ann_tsv"], "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = row["cell_type"].strip()
            if not t:
                # fallback labels are marked so the type search cannot pass a
                # class-level bucket off as the full class population
                c = row["cell_class"].strip() or row["super_class"].strip()
                t = f"{c} (class)" if c else ""
            ctype[int(row["root_id"])] = t or "unannotated"
    types_by_node = [ctype.get(int(i), "unannotated") for i in node_ids]

    # ---- three Y-axis groupings: our streams, raw cell classes, super classes
    d_meta = meta
    scv, ccv = d_meta["super_class_values"], d_meta["cell_class_values"]

    def ordered_grouping(codes, names, display=None):
        """Group values reordered by median champion rank -> (glane, glist)."""
        present = sorted(set(int(c) for c in codes),
                         key=lambda i: float(np.median(rank[codes == i])))
        remap = {old: new for new, old in enumerate(present)}
        glane = np.array([remap[int(c)] for c in codes], dtype=np.int64)
        glist = [(display(names[i]) if display else names[i]) for i in present]
        return glane, glist

    def family_colors(glane, glist):
        """Each group inherits its members' modal stream colour."""
        stream_by_node = np.array(
            [lane_of(scv[s], ccv[c]) for s, c in zip(d["super_class"],
                                                     d["cell_class"])])
        cols = []
        for i in range(len(glist)):
            vals, cnts = np.unique(stream_by_node[glane == i],
                                   return_counts=True)
            cols.append(LANE_COLORS[vals[np.argmax(cnts)]])
        return cols

    grp_defs = {"streams": (lane, lane_list,
                            [LANE_COLORS[nm] for nm in lane_list])}
    cl_lane, cl_list = ordered_grouping(
        d["cell_class"].astype(np.int64), ccv,
        display=lambda s: "NA (no fine label)" if s == "NA" else s)
    grp_defs["cell_class"] = (cl_lane, cl_list, family_colors(cl_lane, cl_list))
    su_lane, su_list = ordered_grouping(d["super_class"].astype(np.int64), scv)
    grp_defs["super_class"] = (su_lane, su_list,
                               family_colors(su_lane, su_list))

    thr_ink = SCFG["min_bundle_frac"]
    thr_exp = SCFG["export_bundle_frac"]
    Wtot = float(w.sum())
    layouts = {}
    cloud_groups = {}
    hw = 0.40                                   # band halfwidth, as fly_cloud.py
    spec = d["spec_y"].astype(np.float64)
    for gname, (glane, glist, gcols) in grp_defs.items():
        Kg = len(glist)
        # per-neuron cloud y for this grouping (ordinal spectral rank in band)
        ycl = np.zeros(n)
        for i in range(Kg):
            m = glane == i
            r = np.argsort(np.argsort(spec[m], kind="stable"), kind="stable")
            ycl[m] = i + hw * (2.0 * r / max(int(m.sum()) - 1, 1) - 1.0)
        y01 = (ycl - ycl.min()) / (ycl.max() - ycl.min())
        cloud_groups[gname] = {
            "K": Kg, "y_min": float(ycl.min()), "y_max": float(ycl.max()),
            "y_u16": b64(np.round(y01 * 65535).astype(np.uint16)),
            "lane_u8": b64(glane.astype(np.uint8)),
        }
        layouts[gname] = {}
        for B in CONFIG["bins"]:
            L = build_layout(rank, glane, glist, src, g.tgt.astype(np.int64),
                             w, B, out_w)
            stats = layout_stats(L, Wtot, thr_ink * Wtot)
            dbin = np.minimum((rank * B) // n, B - 1).astype(np.int64)
            cellid = glane * B + dbin
            per_cell: dict[int, Counter] = {}
            for ci, t in zip(cellid, types_by_node):
                per_cell.setdefault(int(ci), Counter())[t] += 1
            NC = Kg * B
            tot = {k: np.zeros(NC) for k in ("out_ff", "out_fb",
                                             "in_ff", "in_fb")}
            for s_, t_, wf, wb in L["bundles"]:
                tot["out_ff"][s_] += wf; tot["out_fb"][s_] += wb
                tot["in_ff"][t_] += wf; tot["in_fb"][t_] += wb
            cells = []
            for c in np.nonzero(L["counts"])[0]:
                top = per_cell.get(int(c), Counter()).most_common(
                    CONFIG["top_types_per_cell"])
                cells.append({"lane": int(c // B), "bin": int(c % B),
                              "count": int(L["counts"][c]),
                              "intra_ff": float(L["intra_ff"][c]),
                              "intra_fb": float(L["intra_fb"][c]),
                              "sink0": int(L["sink0"][c]),
                              "out_ff": float(tot["out_ff"][c]),
                              "out_fb": float(tot["out_fb"][c]),
                              "in_ff": float(tot["in_ff"][c]),
                              "in_fb": float(tot["in_fb"][c]),
                              "top_types": [[t, int(cnt)] for t, cnt in top]})
            layouts[gname][f"B{B}"] = {
                "n_bins": B,
                "ink_threshold_frac": thr_ink,
                "export_floor_frac": thr_exp,
                "lanes": [{"name": glist[i], "color": gcols[i],
                           "n": int((glane == i).sum()),
                           "med_rank": float(np.median(rank[glane == i]))}
                          for i in range(Kg)],
                "cells": cells,
                "bundles": [{"s": s_, "t": t_, "ff": round(wf, 2),
                             "fb": round(wb, 2)}
                            for s_, t_, wf, wb in L["bundles"]
                            if wf + wb >= thr_exp * Wtot],
                "stats": stats,
            }

    # cloud positions, quantized: x = rank (uint32); per-grouping y/lane above
    # per-neuron cell-type codes for the artifact's type-highlight search
    type_names = sorted(set(types_by_node))
    type_idx = {t: i for i, t in enumerate(type_names)}
    assert len(type_names) < 65536
    type_u16 = np.array([type_idx[t] for t in types_by_node], dtype=np.uint16)
    type_counts = np.bincount(type_u16, minlength=len(type_names))

    # transmitter stats in ABSOLUTE synapse weight, for the Chemistry view's
    # visible reconciliation (percentages elsewhere must re-derive from these)
    nt_codes = d["top_nt"].astype(np.int64)
    ntv = meta["top_nt_values"]
    NNT = len(ntv)
    tgt = g.tgt.astype(np.int64)
    ff_edge = rank[src] < rank[tgt]
    nt_src = nt_codes[src]
    w_ff_nt = np.bincount(nt_src[ff_edge], weights=w[ff_edge], minlength=NNT)
    w_fb_nt = np.bincount(nt_src[~ff_edge], weights=w[~ff_edge], minlength=NNT)
    lane_nt = lane[src] * NNT + nt_src
    lane_ff = np.bincount(lane_nt[ff_edge], weights=w[ff_edge],
                          minlength=K * NNT).reshape(K, NNT)
    lane_fb = np.bincount(lane_nt[~ff_edge], weights=w[~ff_edge],
                          minlength=K * NNT).reshape(K, NNT)
    nt_stats = {
        "values": ntv,
        "n_neurons": np.bincount(nt_codes, minlength=NNT).tolist(),
        "w_ff": w_ff_nt.tolist(),
        "w_fb": w_fb_nt.tolist(),
        "lane_out_ff": lane_ff.tolist(),
        "lane_out_fb": lane_fb.tolist(),
        "note": "absolute synapse weight by SOURCE neuron's predicted "
                "transmitter (Eckstein et al. 2024); ff by the node-level "
                "champion order",
    }

    side_codes = d["side"]
    sidev = meta["side_values"]
    side_map = np.zeros(len(sidev), dtype=np.uint8)          # 0 = other/center
    for i, v in enumerate(sidev):
        side_map[i] = 1 if v == "left" else (2 if v == "right" else 0)
    cloud = {
        "n": int(n),
        "x_u32": b64(rank.astype(np.uint32)),
        "sink_u8": b64((out_w == 0).astype(np.uint8)),
        "side_u8": b64(side_map[side_codes]),                # 0 other, 1 L, 2 R
        "type_u16": b64(type_u16),
        "nt_u8": b64(nt_codes.astype(np.uint8)),
        "groups": cloud_groups,
    }

    report = json.loads((CONFIG["streams_dir"] / "report_streams.json")
                        .read_text(encoding="utf-8"))
    lane_matrix = json.loads((CONFIG["streams_dir"] / "lane_matrix.json")
                             .read_text(encoding="utf-8"))
    data = {
        "generated": time.strftime("%Y-%m-%d"),
        "champion_ff_pct": ff_pct,
        "W_total": float(w.sum()),
        "n": int(n), "m": int(g.n_edges),
        "lane_list": lane_list,
        "layouts": layouts,
        "cloud": cloud,
        "type_names": type_names,
        "type_counts": type_counts.tolist(),
        "lane_matrix": lane_matrix,
        "flow_hierarchy_test": report["flow_hierarchy_test"],
        "nt_direction": report["nt_direction"],
        "nt_stats": nt_stats,
        "sink_audit": report["sink_audit"],
        "lane_order_check": report["lane_order_check"],
        "spectral": meta["spectral"],
        "provenance": meta["join"],
    }
    edges_z, data["edges"] = edge_payload(g, rank)
    pe = out / "edges.zlib"
    pe.write_bytes(edges_z)
    print(f"wrote {pe}  ({len(edges_z)} B, H64 FF {data['edges']['ff_h64_syn']} syn)")
    p = out / "artifact_data.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    print(f"wrote {p}  ({p.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
