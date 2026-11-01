"""Per-neuron cloud view of the fly connectome: every neuron one dot.

Companion to fly_streams.py (same lanes, same depth axis), but at per-neuron
granularity: x = champion pi rank, y = lane band + a within-lane offset.
The within-lane offset is the RANK of the tau-regularized spectral coordinate
inside the lane (rank-normalised, i.e. ORDINAL, not metric) — neurons that
are graph-close sit at similar heights, so bilateral / sub-community structure
appears inside each band without any claim beyond "affinity". The coordinate
is produced by fly_ann_prep.py (spec_y in node_ann.npz: tau-regularized
random-walk operator, log1p weights, deflated power iteration, seed 42 —
all recorded in node_ann_meta.json).

Outputs -> layering/outputs/fly_cloud/
  C1_fly_cloud_lanes.png   dots colored by lane
  C1_fly_cloud_side.png    same geometry, colored by hemisphere (validation:
                           the unsupervised coordinate splits left/right
                           inside the optic lanes)
  cloud_xy.npz             per-node x (rank), y (float), lane — for the artifact

Staleness gate: the annotation cache's rank vector must equal the pinned
champion rank (results/champions/H64_connectome_s42_cuda.npz) — a stale cache
fails loudly.

Run:  $PY layering/tools/fly_cloud.py   (after fly_ann_prep.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from layering.tools.fly_streams import LANE_COLORS, lane_of   # noqa: E402
from layering.io_utils import CHAMPION_ORDER_NPZ              # noqa: E402

CONFIG = {
    "ann_npz": HERE.parent / "outputs" / "fly_ann" / "node_ann.npz",
    "ann_meta": HERE.parent / "outputs" / "fly_ann" / "node_ann_meta.json",
    "champion_npz": REPO / CHAMPION_ORDER_NPZ["connectome"],
    "out_dir": HERE.parent / "outputs" / "fly_cloud",
    "band_halfwidth": 0.40,     # within-lane offset range
    "dpi": 130,
    "dot_size": 0.6,
    "dot_alpha": 0.35,
}


def main():
    out = CONFIG["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(CONFIG["ann_npz"])
    meta = json.loads(Path(CONFIG["ann_meta"]).read_text(encoding="utf-8"))
    rank = d["rank"]
    with np.load(CONFIG["champion_npz"]) as npz:
        assert np.array_equal(rank, npz["rank"].astype(np.int64)), \
            "node_ann.npz is stale: its rank vector is not the pinned champion"
    n = rank.shape[0]
    scv, ccv = meta["super_class_values"], meta["cell_class_values"]
    sidev = meta["side_values"]

    lane_names_by_node = np.array(
        [lane_of(scv[s], ccv[c]) for s, c in zip(d["super_class"], d["cell_class"])])
    lane_list = sorted(set(lane_names_by_node),
                       key=lambda L: float(np.median(rank[lane_names_by_node == L])))
    lane = np.array([lane_list.index(L) for L in lane_names_by_node])
    K = len(lane_list)

    # within-lane offset: rank of spec_y inside the lane, centred (ORDINAL)
    y = np.zeros(n)
    hw = CONFIG["band_halfwidth"]
    y1 = d["spec_y"].astype(np.float64)
    for i in range(K):
        m = lane == i
        r = np.argsort(np.argsort(y1[m], kind="stable"), kind="stable")
        nl = int(m.sum())
        y[m] = i + hw * (2.0 * r / max(nl - 1, 1) - 1.0)

    np.savez_compressed(out / "cloud_xy.npz", x=rank.astype(np.int32),
                        y=y.astype(np.float32), lane=lane.astype(np.int8),
                        lane_list=np.array(lane_list))

    for tag, colors, title in (
        ("lanes",
         [LANE_COLORS[lane_list[i]] for i in lane],
         "every neuron one dot — color = stream"),
        ("side",
         np.where(np.isin(d["side"], [sidev.index(v) for v in ("left",)]), "#c96fae",
                  np.where(np.isin(d["side"], [sidev.index(v) for v in ("right",)]),
                           "#3f7fb5", "#8a94a6")),
         "same geometry — color = hemisphere (left pink / right blue / other grey; "
         "crimson stays reserved for feedback)"),
    ):
        fig, ax = plt.subplots(figsize=(22, 11), dpi=CONFIG["dpi"])
        ax.scatter(rank, y, s=CONFIG["dot_size"], c=colors,
                   alpha=CONFIG["dot_alpha"], linewidths=0, rasterized=True)
        for i, name in enumerate(lane_list):
            ax.annotate(name, (-0.008, 1.0 - (i + 0.62) / (K + 0.6)),
                        xycoords="axes fraction", fontsize=10, ha="right",
                        va="center", color=LANE_COLORS[name], fontweight="bold",
                        annotation_clip=False)
        ax.set_xlim(0, n)
        ax.set_ylim(K - 0.3, -0.9)
        ax.set_xlabel("champion order pi (H64) — MFAS depth")
        ax.set_title(f"The fly connectome, {title}; y within a stream = "
                     "rank of the tau-regularized spectral coordinate "
                     "(graph affinity; ordinal, not metric)",
                     loc="left", fontsize=12)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.subplots_adjust(left=0.10)
        fig.savefig(out / f"C1_fly_cloud_{tag}.png", facecolor="#faf9f5")
        plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
