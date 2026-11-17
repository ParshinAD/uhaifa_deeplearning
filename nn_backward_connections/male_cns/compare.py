"""Female (FlyWire) vs male (Male CNS) comparison table and figures.

Reads male_cns/results/analysis_{flywire,malecns,malecns_brain}.json, type_ranks_*.csv,
the null-ladder JSONs (FlyWire: biology/results/nullmodels/runs; male: male_cns/results/nulls)
and writes male_cns/results/comparison.json + male_cns/outputs/*.png.

    $PY male_cns/compare.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "male_cns" / "results"
OUT = ROOT / "male_cns" / "outputs"
CONFIG = dict(dpi=120, min_type_n=1, ladder=["L1_gnm", "w_source", "L3_dcsbm", "real"],
              nt_order=["ACh", "GABA", "Glu", "HA", "DA", "5-HT", "OA"],
              colors=dict(female="#c0392b", male="#2471a3", male_brain="#7fb3d5"))
OUT.mkdir(parents=True, exist_ok=True)

A = {k: json.loads((RES / f"analysis_{k}.json").read_text()) for k in ["flywire", "malecns", "malecns_brain"]
     if (RES / f"analysis_{k}.json").exists()}
C = {}

scalar_keys = ["n", "m", "total_weight", "ff_pct", "reciprocal_edge_share", "two_cycle_floor_pct_of_W",
               "two_cycle_floor_share_of_fb", "fb_span_median_n", "giant_scc_frac",
               "fb_weight_inside_giant_scc_pct", "zero_output_sinks", "flow_rho", "type_iqr_median",
               "type_iqr_null_median", "types_tighter_than_null_5pct", "inhibitory_share_of_fb_pct",
               "inhibitory_share_of_ff_pct", "fb_within_same_lane_pct", "fb_inside_optic_lanes_pct",
               "degree_hubs_inhibitory_count", "mirror_paired_neuron_frac", "mirror_swap_cost_pp",
               "mirror_left_earlier_rate_types_ge5_pairs", "fb_inside_vnc_pct", "fb_brain_vnc_crossing_pct",
               "vnc_median_rank_frac"]
table = pd.DataFrame({k: {s: A[k].get(s) for s in scalar_keys} for k in A})
for k in A:
    for kk, v in A[k]["fb_top_k_node_share_pct"].items():
        table.loc[f"fb_top_{kk}_nodes_pct", k] = v
table.to_csv(RES / "comparison_table.csv")
C["table"] = table.to_dict()

# NT backward share (within-superclass null z)
nt = {k: {t: A[k]["nt_backward_share_within_superclass_null"].get(t) for t in CONFIG["nt_order"]} for k in A}
C["nt"] = nt

# Null ladder: mean pct_final per rung
def ladder(runs_dir, prefix):
    rows = []
    for p in sorted(Path(runs_dir).glob(f"{prefix}__*__e4000.json")):
        r = json.loads(p.read_text())
        if r.get("sift_max_sweeps", 8) != 8:
            continue
        rows.append(dict(kind=r["kind"], pct=r["pct_final"], greedy=r["pct_greedy"]))
    d = pd.DataFrame(rows)
    return d.groupby("kind").agg(mean=("pct", "mean"), std=("pct", "std"), k=("pct", "size"),
                                 greedy=("greedy", "mean")) if len(d) else None
lad_f = ladder(ROOT / "biology/results/nullmodels/runs", "connectome")
lad_m = ladder(RES / "nulls", "malecns")
lad_b = ladder(RES / "nulls", "malecns_brain")
C["ladder_female"] = None if lad_f is None else lad_f.to_dict("index")
C["ladder_male"] = None if lad_m is None else lad_m.to_dict("index")
C["ladder_male_brain_real"] = None if lad_b is None else lad_b.to_dict("index")

# Cross-sex type ranks (matched on FlyWire type name)
tf = pd.read_csv(RES / "type_ranks_flywire.csv", index_col=0)
tm = pd.read_csv(RES / "type_ranks_malecns_brain.csv", index_col=0)
j = tf.join(tm, lsuffix="_f", rsuffix="_m", how="inner")
j = j[(j.n_f >= CONFIG["min_type_n"]) & (j.n_m >= CONFIG["min_type_n"])]
rho = float(j.median_rank_f.rank().corr(j.median_rank_m.rank()))
j["delta"] = j.median_rank_m - j.median_rank_f
j.sort_values("delta").to_csv(RES / "type_rank_cross_sex.csv")
w = j.n_f + j.n_m
C["cross_sex"] = dict(types_matched=int(len(j)), spearman_type_median_rank=rho,
                      neurons_weighted_abs_delta_median=float(np.median(np.repeat(j.delta.abs(), w))),
                      biggest_movers_earlier_in_male=j.nsmallest(10, "delta")[["lane_f", "median_rank_f", "median_rank_m"]].round(3).reset_index().to_dict("records"),
                      biggest_movers_later_in_male=j.nlargest(10, "delta")[["lane_f", "median_rank_f", "median_rank_m"]].round(3).reset_index().to_dict("records"))
lane_rho = j.groupby("lane_f").apply(lambda d: d.median_rank_f.corr(d.median_rank_m, method="spearman") if len(d) > 10 else np.nan)
C["cross_sex"]["within_lane_spearman"] = {k: (None if pd.isna(v) else float(v)) for k, v in lane_rho.items()}
(RES / "comparison.json").write_text(json.dumps(C, indent=1, default=float))

# ---- Figure 1: lane medians, female vs male (brain-only order and full CNS)
lanes_f = A["flywire"]["lane_median_rank_frac"]
order = sorted(set(lanes_f) | set(A["malecns"]["lane_median_rank_frac"]),
               key=lambda l: A["malecns"]["lane_median_rank_frac"].get(l, lanes_f.get(l, {"median": 1}))["median"])
fig, ax = plt.subplots(figsize=(8, 7))
y = np.arange(len(order))
for key, lab, col, mk in [("flywire", "female (FlyWire, brain)", CONFIG["colors"]["female"], "o"),
                          ("malecns_brain", "male (brain only)", CONFIG["colors"]["male_brain"], "s"),
                          ("malecns", "male (full CNS)", CONFIG["colors"]["male"], "D")]:
    if key not in A:
        continue
    d = A[key]["lane_median_rank_frac"]
    ax.scatter([d[l]["median"] if l in d else np.nan for l in order], y, label=lab, color=col, marker=mk, s=40)
ax.set_yticks(y)
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xlabel("median position in the MFAS order (0 = first, 1 = last)")
ax.set_title("Where each cell group sits in the feedforward order")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig1_lane_medians_female_vs_male.png", dpi=CONFIG["dpi"])
plt.close(fig)

# ---- Figure 2: matched types, female vs male median rank
fig, ax = plt.subplots(figsize=(7, 7))
for lane_name, d in j.groupby("lane_f"):
    ax.scatter(d.median_rank_f, d.median_rank_m, s=6, alpha=0.6, label=lane_name)
ax.plot([0, 1], [0, 1], color="k", lw=0.8)
ax.set_xlabel("female (FlyWire): median position of the type")
ax.set_ylabel("male (brain only): median position of the type")
ax.set_title(f"Same cell type, two brains: {len(j)} matched types, Spearman {round(rho, 3)}")
ax.legend(fontsize=7, markerscale=2, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(OUT / "fig2_type_rank_female_vs_male.png", dpi=CONFIG["dpi"])
plt.close(fig)

# ---- Figure 3: headline metrics side by side
metrics = [("ff_pct", "feedforward weight, %"), ("two_cycle_floor_pct_of_W", "2-cycle floor, % of weight"),
           ("inhibitory_share_of_fb_pct", "GABA+Glu share of feedback, %"),
           ("inhibitory_share_of_ff_pct", "GABA+Glu share of feedforward, %"),
           ("fb_within_same_lane_pct", "feedback inside one lane, %")]
fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.6))
keys = [k for k in ["flywire", "malecns_brain", "malecns"] if k in A]
names = {"flywire": "female", "malecns_brain": "male brain", "malecns": "male CNS"}
cols = {"flywire": CONFIG["colors"]["female"], "malecns_brain": CONFIG["colors"]["male_brain"],
        "malecns": CONFIG["colors"]["male"]}
for ax, (m, lab) in zip(axes, metrics):
    vals = [A[k][m] for k in keys]
    bars = ax.bar([names[k] for k in keys], vals, color=[cols[k] for k in keys])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(round(float(v), 1)), ha="center", va="bottom", fontsize=8)
    ax.set_title(lab, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Feedback structure, female vs male (same analysis code)")
fig.tight_layout()
fig.savefig(OUT / "fig3_headline_metrics.png", dpi=CONFIG["dpi"])
plt.close(fig)

# ---- Figure 4: NT backward share
fig, ax = plt.subplots(figsize=(8, 4))
xs = np.arange(len(CONFIG["nt_order"]))
for i, k in enumerate(keys):
    v = [nt[k][t]["obs"] * 100 if nt[k].get(t) else np.nan for t in CONFIG["nt_order"]]
    ax.bar(xs + (i - 1) * 0.27, v, width=0.27, color=cols[k], label=names[k])
ax.set_xticks(xs)
ax.set_xticklabels(CONFIG["nt_order"])
ax.set_ylabel("share of output weight that is feedback, %")
ax.set_title("Feedback share by presynaptic transmitter")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig4_nt_feedback_share.png", dpi=CONFIG["dpi"])
plt.close(fig)

# ---- Figure 5: null ladder
if lad_f is not None and lad_m is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    rungs = [r for r in CONFIG["ladder"] if r in lad_f.index and r in lad_m.index]
    for lad, lab, col in [(lad_f, "female", CONFIG["colors"]["female"]), (lad_m, "male CNS", CONFIG["colors"]["male"])]:
        ax.errorbar(range(len(rungs)), [lad.loc[r, "mean"] for r in rungs],
                    yerr=[0 if pd.isna(lad.loc[r, "std"]) else lad.loc[r, "std"] for r in rungs],
                    marker="o", color=col, label=lab, capsize=3)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels(["G(n,m)", "degree-matched\n(config model)", "degree + lane\nblocks (dcSBM)", "real graph"][:len(rungs)])
    ax.set_ylabel("feedforward weight, % (same protocol)")
    ax.set_title("How much order each random-graph rung explains")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_null_ladder.png", dpi=CONFIG["dpi"])
    plt.close(fig)

print(json.dumps({k: v for k, v in C.items() if k != "table"}, indent=1, default=float)[:4000])
print(table.round(4).to_string())

# ---- Figure 6: streams x depth: where each lane's neurons sit along the order, female vs male CNS
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "male_cns"))
from analyze import load  # noqa: E402
bins = 40
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)
for ax, ds, title in [(axes[0], "flywire", "female (FlyWire, brain only)"), (axes[1], "malecns", "male CNS (brain + nerve cord)")]:
    _, _, _, rank, a = load(ds, None)
    rn = rank / (len(rank) - 1)
    lanes_here = A[ds]["lane_median_rank_frac"]
    ln_order = sorted(lanes_here, key=lambda l: lanes_here[l]["median"])
    M = np.array([np.histogram(rn[(a.lane == l).to_numpy()], bins=bins, range=(0, 1))[0] for l in ln_order], dtype=float)
    M = M / M.sum(axis=1, keepdims=True)
    im = ax.imshow(M, aspect="auto", cmap="Greys", extent=(0, 1, len(ln_order) - 0.5, -0.5), vmin=0, vmax=0.5)
    ax.set_yticks(range(len(ln_order)))
    ax.set_yticklabels([f"{l} ({lanes_here[l]['n']})" for l in ln_order], fontsize=8)
    ax.set_xlabel("position in the MFAS order (0 = first, 1 = last)")
    ax.set_title(title)
fig.colorbar(im, ax=axes, shrink=0.6, label="share of the lane's neurons in the bin")
fig.suptitle("Streams x depth: each row is one cell group, dark = where its neurons sit")
fig.savefig(OUT / "fig6_streams_depth_female_vs_male.png", dpi=CONFIG["dpi"], bbox_inches="tight")
plt.close(fig)
