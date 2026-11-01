"""Assemble the data bundle for the interactive MICrONS artifact.

Reads (read-only): microns_ann outputs, microns_streams layouts + lane matrix,
diagnostics.json, gallery report (span-spectrum curves), and the graph itself.
Writes: layering/outputs/microns_artifact/artifact_data.json — one JSON with
  * layouts for B in {8,16,32}: lanes, cells (+ per-cell median depth and
    proofread count), bundles, stats
  * cloud: per-neuron packed arrays (base64) for BOTH spaces:
      x = rank (u32); depth_um, x_um, z_um quantized u16; lane u8; ei u8;
      flags u8 (bit0 zero-out, bit1 proofread-axon, bit2 multi-soma);
      out_w, in_w u32 (clipped)
  * global stats: diagnostics D1–D6, the E<->I pair budget, span-spectrum
    curves (fly vs microns), layer boundaries + soma-density profile,
    provenance

Run:  $PY layering/tools/microns_artifact_export.py
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONFIG = {
    "ann_npz": HERE.parent / "outputs" / "microns_ann" / "node_ann.npz",
    "ann_meta": HERE.parent / "outputs" / "microns_ann" / "node_ann_meta.json",
    "diagnostics": HERE.parent / "outputs" / "microns_ann" / "diagnostics.json",
    "streams_dir": HERE.parent / "outputs" / "microns_streams",
    "gallery_report": HERE.parent / "outputs" / "microns_gallery" / "report_gallery.json",
    "out_dir": HERE.parent / "outputs" / "microns_artifact",
    "bins": [8, 16, 32],
    "depth_q_max_um": 1100.0,     # depth quantization ceiling (u16 over 0..this)
}


def b64(arr) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def q16(a, lo, hi):
    """Quantize float array to u16 over [lo, hi]; NaN -> 65535."""
    out = np.full(a.shape, 65535, dtype=np.uint16)
    m = np.isfinite(a)
    x = np.clip((a[m] - lo) / (hi - lo), 0, 1)
    out[m] = np.round(x * 65534).astype(np.uint16)
    return out


def main():
    cfg = CONFIG
    out = cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from layering import io_utils
    from microns_streams import assign_lanes, LANE_COLORS
    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order("microns")
    n = g.n_nodes
    src = g.src.astype(np.int64)
    tgt = g.tgt.astype(np.int64)
    w = g.weight.astype(np.float64)
    Wtot = float(w.sum())
    out_w = np.bincount(src, weights=w, minlength=n)
    in_w = np.bincount(tgt, weights=w, minlength=n)

    d = np.load(cfg["ann_npz"])
    meta = json.loads(Path(cfg["ann_meta"]).read_text(encoding="utf-8"))
    diag = json.loads(Path(cfg["diagnostics"]).read_text(encoding="utf-8"))
    gal = json.loads(Path(cfg["gallery_report"]).read_text(encoding="utf-8"))
    assert np.array_equal(d["rank"], rank), "annotation cache is stale"
    lane, lane_list = assign_lanes(d["ei"], d["layer"])
    K = len(lane_list)

    # ---- layouts (with per-cell extras) -------------------------------------
    layouts = {}
    for B in cfg["bins"]:
        Lj = json.loads((Path(cfg["streams_dir"]) / f"layout_ana{K}_B{B}.json")
                        .read_text(encoding="utf-8"))
        dbin = np.minimum((rank * B) // n, B - 1).astype(np.int64)
        cellid = lane * B + dbin
        pr = d["axon_pr"].astype(bool)
        for c in Lj["cells"]:
            cid = c["lane"] * B + c["bin"]
            m = cellid == cid
            dep = d["depth_um"][m]
            dep = dep[np.isfinite(dep)]
            c["med_depth_um"] = float(np.median(dep)) if dep.size else None
            c["n_proofread"] = int(pr[m].sum())
        layouts["B%d" % B] = Lj

    lane_matrix = json.loads((Path(cfg["streams_dir"]) / "lane_matrix.json")
                             .read_text(encoding="utf-8"))

    # ---- per-neuron cloud arrays --------------------------------------------
    flags = (d["out_w"] == 0).astype(np.uint8)
    flags |= (d["axon_pr"].astype(np.uint8) << 1)
    flags |= ((d["n_somas"] > 1).astype(np.uint8) << 2)
    xmax = float(np.nanmax(d["x_um"]))
    zmax = float(np.nanmax(d["z_um"]))
    cloud = {
        "x_u32": b64(rank.astype(np.uint32)),
        "depth_u16": b64(q16(d["depth_um"].astype(np.float64), 0.0,
                             cfg["depth_q_max_um"])),
        "xum_u16": b64(q16(d["x_um"].astype(np.float64), 0.0, xmax)),
        "zum_u16": b64(q16(d["z_um"].astype(np.float64), 0.0, zmax)),
        "lane_u8": b64(lane.astype(np.uint8)),
        "ei_u8": b64(d["ei"].astype(np.uint8)),
        "flags_u8": b64(flags),
        "outw_u32": b64(np.clip(out_w, 0, 2**32 - 1).astype(np.uint32)),
        "inw_u32": b64(np.clip(in_w, 0, 2**32 - 1).astype(np.uint32)),
        "depth_max_um": cfg["depth_q_max_um"],
        "x_max_um": xmax,
        "z_max_um": zmax,
        "x_min_um": float(np.nanmin(d["x_um"])),
        "z_min_um": float(np.nanmin(d["z_um"])),
        "note": "q16: value 65535 = missing; else lo + v/65534*(hi-lo)",
    }

    data = {
        "dataset": "MICrONS minnie65 (mouse visual cortex), v117 public release",
        "champion": "H42",
        "champion_ff_pct": ff_pct,
        "n": n, "m": int(src.shape[0]), "W_total": Wtot,
        "lane_list": lane_list,
        "lane_colors": {k: LANE_COLORS[k] for k in lane_list},
        "layouts": layouts,
        "lane_matrix": lane_matrix,
        "cloud": cloud,
        "layers": meta["layers"],
        "pia_fit": meta["pia_fit"],
        "join": meta["join"],
        "density_profile": {
            "centers_um": d["density_centers"].astype(float).tolist(),
            "smoothed": d["density_smoothed"].astype(float).tolist()},
        "diagnostics": {k: diag[k] for k in diag if k.startswith("d")},
        "fb_pct": diag["fb_pct"],
        "ei_pair_budget": gal["ei_pair_budget"],
        "depth_z_leak_check": gal["depth_z_leak_check"],
        "span_curves": gal["m6"],
        "generated_by": "layering/tools/microns_ann_prep.py -> microns_diagnostics.py "
                        "-> microns_streams.py -> microns_gallery.py -> "
                        "microns_artifact_export.py",
    }
    (out / "artifact_data.json").write_text(json.dumps(data), encoding="utf-8")
    size = (out / "artifact_data.json").stat().st_size
    print(f"artifact_data.json: {size/1e6:.1f} MB  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
