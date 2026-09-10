"""Assemble the data bundle for the interactive fly-connectome artifact.

Reads (read-only): fly_ann outputs, fly_streams layouts, fly_cloud positions,
the annotation TSV (for cell_type hover cards), and the graph itself.
Writes: layering/outputs/fly_artifact/artifact_data.json  — one JSON with
  * layouts for B in {8,16,32}: lanes, cells (+ top cell_types per cell),
    bundles, stats
  * cloud: per-neuron x (rank), y (quantized uint16 over lane bands), lane id,
    packed as base64 arrays; plus per-neuron flags (zero-out-weight)
  * global stats: hierarchy test, NT x direction, sink audit, provenance

Run:  $PY layering/tools/fly_artifact_export.py
"""
from __future__ import annotations

import base64
import csv
import gzip
import json
import sys
import time
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
}


def b64(arr) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def main():
    out = CONFIG["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from layering import io_utils
    from layering.tools.fly_streams import assign_lanes
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

    layouts = {}
    for B in CONFIG["bins"]:
        lay = json.loads((CONFIG["streams_dir"] / f"layout_bio{K}_B{B}.json")
                         .read_text(encoding="utf-8"))
        dbin = np.minimum((rank * B) // n, B - 1).astype(np.int64)
        cellid = lane * B + dbin
        # top cell_types per cell
        per_cell: dict[int, Counter] = {}
        for ci, t in zip(cellid, types_by_node):
            per_cell.setdefault(int(ci), Counter())[t] += 1
        for cell in lay["cells"]:
            cid = cell["lane"] * B + cell["bin"]
            top = per_cell.get(cid, Counter()).most_common(
                CONFIG["top_types_per_cell"])
            cell["top_types"] = [[t, int(c)] for t, c in top]
        layouts[f"B{B}"] = lay

    # cloud positions, quantized: x = rank (uint32), y in [0,1] -> uint16
    cl = np.load(CONFIG["cloud_npz"], allow_pickle=False)
    assert [str(x) for x in cl["lane_list"]] == lane_list, \
        "cloud_xy.npz is stale: its lane list disagrees with assign_lanes"
    y = cl["y"].astype(np.float64)
    y01 = (y - y.min()) / (y.max() - y.min())
    # per-neuron cell-type codes for the artifact's type-highlight search
    type_names = sorted(set(types_by_node))
    type_idx = {t: i for i, t in enumerate(type_names)}
    assert len(type_names) < 65536
    type_u16 = np.array([type_idx[t] for t in types_by_node], dtype=np.uint16)
    type_counts = np.bincount(type_u16, minlength=len(type_names))

    side_codes = d["side"]
    sidev = meta["side_values"]
    side_map = np.zeros(len(sidev), dtype=np.uint8)          # 0 = other/center
    for i, v in enumerate(sidev):
        side_map[i] = 1 if v == "left" else (2 if v == "right" else 0)
    cloud = {
        "n": int(n),
        "K": K,
        "y_min": float(y.min()), "y_max": float(y.max()),
        "x_u32": b64(cl["x"].astype(np.uint32)),
        "y_u16": b64(np.round(y01 * 65535).astype(np.uint16)),
        "lane_u8": b64(cl["lane"].astype(np.uint8)),
        "sink_u8": b64((out_w == 0).astype(np.uint8)),
        "side_u8": b64(side_map[side_codes]),                # 0 other, 1 L, 2 R
        "type_u16": b64(type_u16),
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
        "sink_audit": report["sink_audit"],
        "lane_order_check": report["lane_order_check"],
        "spectral": meta["spectral"],
        "provenance": meta["join"],
    }
    p = out / "artifact_data.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    print(f"wrote {p}  ({p.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
