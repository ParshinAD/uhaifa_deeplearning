"""Join MICrONS anatomy onto the minnie65 graph — prep for the (layer, depth)
representation of the mouse-cortex connectome.

The fly pipeline (fly_ann_prep.py) had FlyWire's curated stream labels; MICrONS
has something the fly never had: REAL 3-D soma positions and an ANATOMY-BASED
excitatory/inhibitory classification. This tool joins them in:

  input   data/processed/microns.npz                 (67,534 nodes, root IDs, v117)
          data/raw/microns/nucleus_neuron_svm.csv    (nucleus id, root_id, x/y/z)
          data/raw/microns/allen_soma_ei_class_model_v1.csv
                (E/I per nucleus id; joined nucleus-id -> root_id -> node.
                 Verified: all 69,957 rows join the nucleus table by id with an
                 identical position, so the id join is exact.)
          data/raw/microns/proofreading_status_public_release.csv
                (col[6]=root_id, col[9]=status_axon — same parse as
                 experiments/build_microns.py)
          results/champions/H42_microns_s42_cuda.npz (via io_utils, parity-gated)

  output  layering/outputs/microns_ann/node_ann.npz   per-node arrays aligned to
                                                      contiguous graph indices:
            rank         int64    champion pi rank (H42, 83.2409 %)
            x_um,z_um    float32  soma position, µm (4/4/40 nm voxels)
            y_um         float32  raw vertical position, µm
            depth_um     float32  depth below the local pia proxy, µm (NaN if no soma)
            layer        int8     0 NA / 1 L1 / 2 L2/3 / 3 L4 / 4 L5 / 5 L6 / 6 WM
            ei           int8     0 NA / 1 excitatory / 2 inhibitory
            n_somas      int16    neuron nuclei mapped to this root (>1 = merge)
            axon_pr      uint8    1 if status_axon in {clean, extended}
            out_w,in_w   int64    total outgoing / incoming synapse weight
            spec_y(2)    float32  spectral coordinates (same operator as the fly)
          layering/outputs/microns_ann/node_ann_meta.json  join stats, pia fit,
                                                      layer boundaries, licenses

Depth and layers. minnie65's pia surface is tilted ~5 deg relative to the raw
y axis (documented in the MICrONS standard_transform tutorial; we measure a
119-µm drop across x), so depth is computed as y minus a pia PROXY: the 1st
percentile of excitatory soma y on a 2-D (x, z) grid, gaussian-smoothed and
hole-filled (loops + np.convolve only — no matrix ops, repo convention P26).
Layer boundaries are then anchored on LANDMARKS of the smoothed
excitatory-soma density along corrected depth — the L4 peak, the L3 and L5a
troughs, the L6 peak — using half-height flank crossings, the standard way
laminae show up in soma-density profiles. The full profile is saved so the
calibration can be inspected. Exact published minnie65 boundary values are
not in any public flat file (checked: standard_transform bundles streamlines,
not layer arrays; the cell_types CSVs carry only coarse E/I), so these bands
are OUR calibration: display annotations, not ground truth, and they never
touch the optimizer or the frozen scorer (invariant 6).

Run:  $PY layering/tools/microns_ann_prep.py
"""
from __future__ import annotations

import csv
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

from layering import io_utils                                  # noqa: E402
from fly_ann_prep import spectral_coords                       # noqa: E402

CONFIG = {
    "dataset": "microns",
    "nucleus_csv": REPO / "data" / "raw" / "microns" / "nucleus_neuron_svm.csv",
    "ei_csv": REPO / "data" / "raw" / "microns" / "allen_soma_ei_class_model_v1.csv",
    "proofread_csv": REPO / "data" / "raw" / "microns" / "proofreading_status_public_release.csv",
    "out_dir": HERE.parent / "outputs" / "microns_ann",
    "voxel_um": (0.004, 0.004, 0.040),   # 4 x 4 x 40 nm voxels -> µm
    # pia proxy: per-(x,z)-cell low percentile of EXCITATORY soma y (E somas
    # start at the L1/L2 border; L1 holds almost no E somas, so the 1st
    # percentile of E tracks the L1/L2 surface — we subtract a fixed L1
    # thickness to reach a nominal pia). The grid is smoothed and hole-filled.
    "pia_grid_um": 100.0,
    "pia_percentile": 1.0,
    "pia_min_count": 20,       # grid cells with fewer E somas become holes
    "pia_grid_smooth_sigma": 1.0,   # in grid cells
    "l1_thickness_um": 90.0,   # nominal L1 thickness added above the E-soma surface
    "density_bin_um": 5.0,
    "density_smooth_sigma_um": 12.0,
    # landmark windows (µm below pia) used only to LOCATE the density landmarks;
    # boundaries come from half-height flank crossings between landmarks.
    "landmark_windows_um": {
        "L4_peak": (250.0, 400.0),
        "L3_trough": (120.0, 250.0),
        "L5a_trough": (400.0, 560.0),
        "L6_peak": (560.0, 720.0),
    },
    "l6_wm_frac": 0.15,        # L6/WM where density falls below this frac of the L6 peak
    "spec_iters": 300,
    "spec_seed": 42,
    "spec_tol": 1e-9,
    "spec_tau_factor": 1.0,
    # provenance of the two downloaded cell_types CSVs (public BossDB, no token):
    "ei_url": "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
              "minnie65/cell_types/allen_soma_ei_class_model_v1.csv",
    "ei_sha256": "0d5114cea032a5fd27fa098a15e7481d759513914a83d0f8bf5703e6b6cea4ce",
    "visp_note": "allen_visp_column_soma_coarse_types_v1.csv was also fetched "
                 "(sha256 4afd12cee608e20db52019da38d2c1fb739d7decb68a51056c5c8e847bf77c63) "
                 "but contains only coarse excitatory/inhibitory/Unsure labels, "
                 "no layer types — unused.",
    "license": "MICrONS minnie65 public release (CC-BY 4.0) — cite MICrONS "
               "Consortium 2025 (Nature 640) wherever a figure built on this "
               "join appears.",
}


def gaussian_smooth(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    """1-D gaussian smoothing via np.convolve (no scipy)."""
    half = int(np.ceil(4 * sigma_bins))
    xs = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-0.5 * (xs / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(y.astype(np.float64), k, mode="same")


def main():
    cfg = CONFIG
    out = cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rx, ry, rz = cfg["voxel_um"]

    g, order, rank, ff_pct = io_utils.load_graph_and_champion_order(cfg["dataset"])
    n = g.n_nodes
    node_ids = g.node_ids.astype(np.int64)
    print(f"loaded {n} nodes, {g.n_edges} edges, champion FF {ff_pct:.4f}%  "
          f"({time.time() - t0:.0f}s)")

    # ---- nucleus table: root_id -> neuron somas ------------------------------
    t1 = time.time()
    somas: dict[int, list] = {}          # root -> [(nucleus_id, x, y, z), ...]
    with open(cfg["nucleus_csv"], "rt") as f:
        for r in csv.reader(f):
            if r[8] != "neuron":
                continue
            root = int(r[3])
            if root == 0:
                continue
            somas.setdefault(root, []).append(
                (int(r[0]), int(r[4]), int(r[5]), int(r[6])))
    for v in somas.values():
        v.sort()                          # deterministic: by nucleus id
    print(f"nucleus table: {sum(len(v) for v in somas.values())} neuron somas on "
          f"{len(somas)} roots  ({time.time() - t1:.0f}s)")

    # ---- E/I table: nucleus id -> class -------------------------------------
    ei_by_nuc: dict[int, int] = {}
    with open(cfg["ei_csv"], "rt") as f:
        for r in csv.reader(f):
            cls = r[3]
            if cls == "excitatory":
                ei_by_nuc[int(r[0])] = 1
            elif cls == "inhibitory":
                ei_by_nuc[int(r[0])] = 2

    # ---- proofread axons -----------------------------------------------------
    pr_roots = set()
    with open(cfg["proofread_csv"], "rt") as f:
        for r in csv.reader(f):
            if r[9] in ("clean", "extended"):
                pr_roots.add(int(r[6]))

    # ---- per-node join -------------------------------------------------------
    x_um = np.full(n, np.nan, dtype=np.float32)
    y_um = np.full(n, np.nan, dtype=np.float32)
    z_um = np.full(n, np.nan, dtype=np.float32)
    n_somas = np.zeros(n, dtype=np.int16)
    ei = np.zeros(n, dtype=np.int8)
    axon_pr = np.zeros(n, dtype=np.uint8)
    ei_conflicts = 0
    for i, root in enumerate(node_ids):
        root = int(root)
        lst = somas.get(root)
        if lst:
            n_somas[i] = len(lst)
            nid, xx, yy, zz = lst[0]
            x_um[i], y_um[i], z_um[i] = xx * rx, yy * ry, zz * rz
            classes = {ei_by_nuc[s[0]] for s in lst if s[0] in ei_by_nuc}
            if len(classes) == 1:
                ei[i] = classes.pop()
            elif len(classes) > 1:
                ei_conflicts += 1        # merged root with conflicting E/I -> NA
        if root in pr_roots:
            axon_pr[i] = 1
    has_pos = ~np.isnan(x_um)
    print(f"join: pos {int(has_pos.sum())}/{n} "
          f"({100.0 * has_pos.sum() / n:.2f}%), "
          f"E {int((ei == 1).sum())}, I {int((ei == 2).sum())}, "
          f"EI-NA {int((ei == 0).sum())} (conflicts {ei_conflicts}), "
          f"multi-soma roots {int((n_somas > 1).sum())}, "
          f"proofread axon {int(axon_pr.sum())}")

    # ---- pia proxy (2-D grid over x,z) and depth -----------------------------
    exc = (ei == 1) & has_pos
    gb = cfg["pia_grid_um"]
    ex, ez, ey = x_um[exc], z_um[exc], y_um[exc]
    gx = np.floor(ex / gb).astype(np.int64)
    gz = np.floor(ez / gb).astype(np.int64)
    gx0, gz0 = int(gx.min()), int(gz.min())
    nx, nz = int(gx.max()) - gx0 + 1, int(gz.max()) - gz0 + 1
    grid = np.full((nx, nz), np.nan)
    for ix in range(nx):
        selx = gx - gx0 == ix
        if not selx.any():
            continue
        zc = gz[selx] - gz0
        yc = ey[selx]
        for iz in np.unique(zc):
            sel = zc == iz
            if int(sel.sum()) >= cfg["pia_min_count"]:
                grid[ix, iz] = np.percentile(yc[sel], cfg["pia_percentile"])
    n_cells = int(np.isfinite(grid).sum())
    # hole-fill by iterative neighbour means, then smooth each axis with the
    # 1-D gaussian kernel (rows, then columns — separable, np.convolve only)
    for _ in range(nx + nz):
        holes = ~np.isfinite(grid)
        if not holes.any():
            break
        filled = grid.copy()
        for ix in range(nx):
            for iz in range(nz):
                if holes[ix, iz]:
                    nb = [grid[a, b]
                          for a, b in ((ix-1, iz), (ix+1, iz), (ix, iz-1), (ix, iz+1))
                          if 0 <= a < nx and 0 <= b < nz and np.isfinite(grid[a, b])]
                    if nb:
                        filled[ix, iz] = float(np.mean(nb))
        grid = filled
    sig = cfg["pia_grid_smooth_sigma"]
    grid = np.stack([gaussian_smooth(grid[ix, :], sig) for ix in range(nx)])
    grid = np.stack([gaussian_smooth(grid[:, iz], sig) for iz in range(nz)], axis=1)
    node_ix = np.clip(np.floor(x_um / gb).astype(np.int64) - gx0, 0, nx - 1)
    node_iz = np.clip(np.floor(z_um / gb).astype(np.int64) - gz0, 0, nz - 1)
    node_ix[~has_pos] = 0
    node_iz[~has_pos] = 0
    pia_y = grid[node_ix, node_iz] - cfg["l1_thickness_um"]
    depth_um = np.where(has_pos, y_um - pia_y, np.nan).astype(np.float32)
    tilt = float(np.nanmax(grid) - np.nanmin(grid))
    print(f"pia proxy: {n_cells} filled grid cells of {nx}x{nz}, surface relief "
          f"{tilt:.0f} µm  (corrected)")

    # ---- layer bands anchored on density landmarks ---------------------------
    db = cfg["density_bin_um"]
    d = depth_um[exc]
    d = d[np.isfinite(d)]
    edges_ = np.arange(0.0, float(np.nanmax(d)) + db, db)
    hist, _ = np.histogram(d, bins=edges_)
    dens = gaussian_smooth(hist, cfg["density_smooth_sigma_um"] / db)
    centers = 0.5 * (edges_[:-1] + edges_[1:])

    def landmark(name, kind):
        lo, hi = cfg["landmark_windows_um"][name]
        idx = np.where((centers >= lo) & (centers <= hi))[0]
        seg = dens[idx]
        k = int(np.argmax(seg) if kind == "max" else np.argmin(seg))
        return float(centers[idx[k]]), float(seg[k])

    lm = {"L3_trough": landmark("L3_trough", "min"),
          "L4_peak": landmark("L4_peak", "max"),
          "L5a_trough": landmark("L5a_trough", "min"),
          "L6_peak": landmark("L6_peak", "max")}

    def flank_cross(d_from, d_to, v_from, v_to):
        """Depth where dens crosses the half-height between two landmarks."""
        half = 0.5 * (v_from + v_to)
        idx = np.where((centers >= min(d_from, d_to)) & (centers <= max(d_from, d_to)))[0]
        rising = v_to > v_from
        for i in (idx if d_to > d_from else idx[::-1]):
            if (dens[i] >= half) == rising:
                return float(centers[i])
        return 0.5 * (d_from + d_to)

    boundaries = {
        "L1/L23": float(np.percentile(d, 0.5)),
        "L23/L4": flank_cross(lm["L3_trough"][0], lm["L4_peak"][0],
                              lm["L3_trough"][1], lm["L4_peak"][1]),
        "L4/L5": flank_cross(lm["L4_peak"][0], lm["L5a_trough"][0],
                             lm["L4_peak"][1], lm["L5a_trough"][1]),
        "L5/L6": flank_cross(lm["L5a_trough"][0], lm["L6_peak"][0],
                             lm["L5a_trough"][1], lm["L6_peak"][1]),
    }
    after_l6 = np.where(centers > lm["L6_peak"][0])[0]
    thr = cfg["l6_wm_frac"] * lm["L6_peak"][1]
    below = after_l6[dens[after_l6] < thr]
    boundaries["L6/WM"] = float(centers[below[0]]) if len(below) else float(centers[-1])
    print("density landmarks (µm, count):",
          {k: (round(v[0]), round(v[1])) for k, v in lm.items()})
    print("layer boundaries (µm below pia):",
          {k: round(v, 1) for k, v in boundaries.items()})

    bvals = [boundaries["L1/L23"], boundaries["L23/L4"], boundaries["L4/L5"],
             boundaries["L5/L6"], boundaries["L6/WM"]]
    layer = np.zeros(n, dtype=np.int8)
    fin = np.isfinite(depth_um)
    layer[fin] = 1 + np.searchsorted(np.array(bvals), depth_um[fin])
    layer[~fin] = 0
    for code, nm in enumerate(["NA", "L1", "L2/3", "L4", "L5", "L6", "WM"]):
        print(f"  layer {nm}: {int((layer == code).sum())}")

    # ---- per-node weight totals ---------------------------------------------
    w = g.weight.astype(np.int64)
    out_w = np.bincount(g.src, weights=w, minlength=n).astype(np.int64)
    in_w = np.bincount(g.tgt, weights=w, minlength=n).astype(np.int64)

    # ---- spectral coordinates ------------------------------------------------
    t2 = time.time()
    ((sy1, it1), (sy2, it2)), tau = spectral_coords(
        g.src, g.tgt, g.weight.astype(np.float64), n, k=2,
        iters=cfg["spec_iters"], seed=cfg["spec_seed"],
        tol=cfg["spec_tol"], tau_factor=cfg["spec_tau_factor"])
    print(f"spectral coords: {it1} + {it2} iterations, tau={tau:.3f}  "
          f"({time.time() - t2:.0f}s)")

    np.savez_compressed(
        out / "node_ann.npz",
        rank=rank.astype(np.int64),
        x_um=x_um, y_um=y_um, z_um=z_um, depth_um=depth_um,
        layer=layer, ei=ei, n_somas=n_somas, axon_pr=axon_pr,
        out_w=out_w, in_w=in_w,
        spec_y=sy1, spec_y2=sy2,
        density_centers=centers.astype(np.float32),
        density_smoothed=dens.astype(np.float32))
    meta = {
        "dataset": cfg["dataset"],
        "n": n, "m": g.n_edges,
        "champion_ff_pct": ff_pct,
        "join": {
            "with_position": int(has_pos.sum()),
            "excitatory": int((ei == 1).sum()),
            "inhibitory": int((ei == 2).sum()),
            "ei_na": int((ei == 0).sum()),
            "ei_conflicts": ei_conflicts,
            "multi_soma_roots": int((n_somas > 1).sum()),
            "proofread_axon": int(axon_pr.sum()),
            "ei_source": cfg["ei_url"],
            "ei_sha256": cfg["ei_sha256"],
            "visp_note": cfg["visp_note"],
            "license": cfg["license"],
        },
        "pia_fit": {
            "method": f"{cfg['pia_percentile']}th percentile of excitatory soma y "
                      f"per {cfg['pia_grid_um']:.0f}-µm (x,z) grid cell, "
                      f"hole-filled + gaussian-smoothed, minus "
                      f"{cfg['l1_thickness_um']:.0f} µm nominal L1 thickness",
            "n_grid_cells": n_cells,
            "grid_shape": [nx, nz],
            "surface_relief_um": tilt,
        },
        "layers": {
            "boundaries_um": {k: float(v) for k, v in boundaries.items()},
            "landmarks_um": {k: [float(v[0]), float(v[1])] for k, v in lm.items()},
            "landmark_windows_um": {k: list(v) for k, v in
                                    cfg["landmark_windows_um"].items()},
            "counts": {nm: int((layer == code).sum()) for code, nm in
                       enumerate(["NA", "L1", "L2/3", "L4", "L5", "L6", "WM"])},
            "note": "Layers are depth bands below a pia PROXY, anchored on "
                    "landmarks of the smoothed excitatory-soma density (saved in "
                    "node_ann.npz as density_centers/density_smoothed) via "
                    "half-height flank crossings. Display annotation, not ground "
                    "truth; never touches the optimizer or the oracle.",
        },
        "spectral": {"iters": [it1, it2], "seed": cfg["spec_seed"], "tau": tau,
                     "tau_factor": cfg["spec_tau_factor"],
                     "weights": "log1p of undirected collapsed synapse counts",
                     "operator": "tau-regularized random-walk (same as fly_ann_prep)"},
    }
    (out / "node_ann_meta.json").write_text(json.dumps(meta, indent=1),
                                            encoding="utf-8")
    print(f"saved -> {out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
