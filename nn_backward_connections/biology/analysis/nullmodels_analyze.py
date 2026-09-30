"""G01 analysis — tables, z-scores, ladder decomposition, lane matrix, figures.

Consumes the per-run JSON records written by ``nullmodels_run.py`` (every number
below traces to one of those files), re-verifies the fly champion score from the
layering artifact with the frozen oracle, and writes:

  biology/results/nullmodels/summary.json        all tables, machine-readable
  biology/results/nullmodels/lane_fb_zmatrix.npz lane-pair FB z-matrix (fly)
  biology/outputs/nullmodels_fly_ladder.png      the constraint-ladder figure
  biology/outputs/nullmodels_mouse_nulls.png     mouse null distributions vs real
  biology/outputs/nullmodels_fly_lane_z.png      lane-pair FB deviation heatmap

Safe to run on PARTIAL results (skips rungs with no runs yet); rerun any time.

    $PY biology/analysis/nullmodels_analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "biology" / "analysis"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mfas.io import load_dataset                                   # noqa: E402
from mfas.metrics import pct, score_from_order                     # noqa: E402

from nullmodels_generators import fly_lane_labels                  # noqa: E402
from nullmodels_run import ALL_VARIANTS, _draw_null                # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = dict(
    runs_dir=_REPO / "biology" / "results" / "nullmodels" / "runs",
    orders_dir=_REPO / "biology" / "results" / "nullmodels" / "orders",
    out_summary=_REPO / "biology" / "results" / "nullmodels" / "summary.json",
    out_lane_npz=_REPO / "biology" / "results" / "nullmodels" / "lane_fb_zmatrix.npz",
    outputs_dir=_REPO / "biology" / "outputs",
    # fly champion order (layering artifact) — re-scored here with the frozen oracle
    champion_npz=_REPO / "layering" / "outputs" / "fly_ann" / "node_ann.npz",
    # standard (headline) epoch budgets; the mouse also has a 20k robustness arm
    standard_epochs=dict(mouse=2_000, connectome=4_000),
    # nested ladder (delta-chain) per dataset; L3a is a non-nested reference
    ladder=dict(
        mouse=["L1_gnm", "w_source"],
        connectome=["L1_gnm", "w_source", "L3_dcsbm"],
    ),
    ladder_labels={"L1_gnm": "L1 density", "w_source": "L2 +degrees",
                   "w_shuffled": "L2 (w shuffled)", "L3a_sbm": "L3a SBM (ref)",
                   "L3_dcsbm": "L3 +regions"},
    lane_null="L3_dcsbm",        # null used for the lane-pair FB z-matrix
    lane_min_pairs_w_frac=2e-4,  # only z-score lane pairs carrying >= this of total
    dpi=120,
    # dataviz reference palette (light mode)
    col=dict(null="#2a78d6", null_light="#9ec5f4", real="#0b0b0b",
             champ="#52514e", accent="#eb6834", surface="#fcfcfb",
             grid="#e5e4e0", text="#0b0b0b", text2="#52514e"),
)


# ──────────────────────────────────────────────────────────────────────────────
# Load runs
# ──────────────────────────────────────────────────────────────────────────────
def load_runs() -> list:
    """Read every per-run JSON record into a list of dicts."""
    rows = []
    for f in sorted(CONFIG["runs_dir"].glob("*.json")):
        r = json.loads(f.read_text())
        r["file"] = f.name
        rows.append(r)
    return rows


def _sel(rows, **kw):
    """Filter rows by exact key/value pairs."""
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def _stats(vals: np.ndarray) -> dict:
    return dict(K=int(vals.size), mean=float(vals.mean()),
                std=float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                min=float(vals.min()), max=float(vals.max()))


# ──────────────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────────────
def summarize(rows) -> dict:
    """Per dataset × epoch-budget × rung: null stats, real value, z, empirical p."""
    out = {}
    for dataset in ("mouse", "connectome"):
        for epochs in sorted({r["epochs"] for r in _sel(rows, dataset=dataset)}):
            reals = _sel(rows, dataset=dataset, kind="real", epochs=epochs)
            if not reals:
                continue
            real_final = np.array([r["pct_final"] for r in reals])
            real_greedy = np.array([r["pct_greedy"] for r in reals])
            key = f"{dataset}_e{epochs}"
            out[key] = dict(
                real=dict(final=_stats(real_final), greedy=_stats(real_greedy),
                          seeds=[r["pipeline_seed"] for r in reals]),
                nulls={},
            )
            for kind in ALL_VARIANTS:
                nulls = _sel(rows, dataset=dataset, kind=kind, epochs=epochs)
                if not nulls:
                    continue
                nf = np.array([r["pct_final"] for r in nulls])
                ng = np.array([r["pct_greedy"] for r in nulls])
                rf, rg = float(real_final.mean()), float(real_greedy.mean())
                z_f = (rf - nf.mean()) / nf.std(ddof=1) if nf.size > 1 else np.nan
                z_g = (rg - ng.mean()) / ng.std(ddof=1) if ng.size > 1 else np.nan
                # one-sided empirical p in the observed direction, add-one rule
                if rf >= nf.mean():
                    n_beyond_f = int((nf >= rf).sum())
                else:
                    n_beyond_f = int((nf <= rf).sum())
                out[key]["nulls"][kind] = dict(
                    final=_stats(nf), greedy=_stats(ng),
                    real_minus_null_final_pp=rf - float(nf.mean()),
                    real_minus_null_greedy_pp=rg - float(ng.mean()),
                    z_final=float(z_f), z_greedy=float(z_g),
                    n_nulls_beyond_real_final=n_beyond_f,
                    empirical_p_final=(n_beyond_f + 1) / (nf.size + 1),
                    mean_runtime_s=float(np.mean([r["t_total_s"] for r in nulls])),
                )
    return out


def champion_pct_fly() -> float:
    """Re-score the layering champion order with the frozen oracle (logged here)."""
    g = load_dataset("connectome")
    rank = np.load(CONFIG["champion_npz"])["rank"]
    s = score_from_order(rank, np.asarray(g.src), np.asarray(g.tgt),
                         np.asarray(g.weight))
    return pct(s, g.total_weight)


def ladder_table(summary: dict, champion: dict) -> dict:
    """FF% at each rung of the nested ladder + labeled deltas between rungs."""
    out = {}
    for dataset, rungs in CONFIG["ladder"].items():
        key = f"{dataset}_e{CONFIG['standard_epochs'][dataset]}"
        if key not in summary:
            continue
        s = summary[key]
        steps = [("chance", 50.0, 0.0, 0)]
        ok = True
        for kind in rungs:
            if kind not in s["nulls"]:
                ok = False
                break
            st = s["nulls"][kind]["final"]
            steps.append((kind, st["mean"], st["std"], st["K"]))
        if not ok:
            continue
        steps.append(("real_protocol", s["real"]["final"]["mean"],
                      s["real"]["final"]["std"], s["real"]["final"]["K"]))
        if dataset in champion:
            steps.append(("real_champion", champion[dataset], 0.0, 1))
        deltas = [dict(frm=steps[i][0], to=steps[i + 1][0],
                       delta_pp=steps[i + 1][1] - steps[i][1])
                  for i in range(len(steps) - 1)]
        ref = {}
        if "L3a_sbm" in s["nulls"]:
            st = s["nulls"]["L3a_sbm"]["final"]
            ref["L3a_sbm"] = dict(mean=st["mean"], std=st["std"], K=st["K"])
        out[dataset] = dict(
            steps=[dict(rung=a, ff_pct_mean=b, ff_pct_std=c, K=k)
                   for a, b, c, k in steps],
            deltas=deltas, non_nested_reference=ref)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Estimator-free structural bound: 2-cycle (reciprocity) lower bound
# ──────────────────────────────────────────────────────────────────────────────
def two_cycle_bound_pct(src, tgt, w, n: int) -> float:
    """EXACT lower bound on unavoidable feedback, as % of total weight.

    Every reciprocated pair {u,v} with weights w(u->v) and w(v->u) forces
    min(w(u->v), w(v->u)) into feedback under ANY linear order (one of the two
    arcs must point backward). Summing the lighter arc over all unordered pairs
    is therefore a TRUE, estimator-free lower bound on unavoidable feedback —
    no heuristic involved. Parallel slots are collapsed (summed) first.
    """
    key = src.astype(np.int64) * n + tgt.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    uw = np.bincount(inv, weights=np.asarray(w, np.float64))
    ru, rv = uk // n, uk % n
    rev = rv * n + ru
    pos = np.searchsorted(uk, rev)
    pos = np.clip(pos, 0, uk.size - 1)
    has_rev = uk[pos] == rev
    lighter = np.minimum(uw, uw[pos])
    # each reciprocated unordered pair is seen twice (once from each arc)
    lb = 0.5 * lighter[has_rev].sum()
    return float(100.0 * lb / np.asarray(w, np.float64).sum())


def structural_bounds(rows) -> dict:
    """2-cycle lower bound for each real graph and each null draw (standard arm).

    Nulls are regenerated bit-identically from their logged ``gen_seed`` (the
    generator selftest asserts seed-reproducibility), so the bound refers to
    exactly the graphs the protocol ran on.
    """
    out = {}
    for dataset in ("mouse", "connectome"):
        epochs = CONFIG["standard_epochs"][dataset]
        if not _sel(rows, dataset=dataset, epochs=epochs):
            continue
        g = load_dataset(dataset)
        src = np.asarray(g.src); tgt = np.asarray(g.tgt); w = np.asarray(g.weight)
        labels = None
        if any(_sel(rows, dataset=dataset, kind=k, epochs=epochs)
               for k in ("L3a_sbm", "L3_dcsbm")):
            labels, _ = fly_lane_labels()
        d = dict(real=two_cycle_bound_pct(src, tgt, w, g.n_nodes))
        for kind in ALL_VARIANTS:
            nulls = _sel(rows, dataset=dataset, kind=kind, epochs=epochs)
            if not nulls:
                continue
            vals = []
            for r in nulls:
                s2, t2, w2, _ = _draw_null(kind, src, tgt, w, g.n_nodes,
                                           r["gen_seed"], labels)
                vals.append(two_cycle_bound_pct(s2, t2, w2, g.n_nodes))
            d[kind] = _stats(np.array(vals))
            # RIGOROUS comparison, no estimator involved on the decisive side:
            # a null whose MEASURED feedback (an upper bound on its true
            # unavoidable FB) is below the real graph's 2-cycle LOWER bound is
            # PROVABLY less feedback-bound than the real graph.
            null_fb_meas = np.array([100.0 - r["pct_final"] for r in nulls])
            d[kind + "_n_nulls_provably_below_real"] = int(
                (null_fb_meas < d["real"]).sum())
            d[kind + "_n_nulls_total"] = len(nulls)
        out[dataset] = d
        print(f"[2cycle] {dataset}: real lower bound {d['real']:.4f}% of weight")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Lane-pair FB decomposition (fly)
# ──────────────────────────────────────────────────────────────────────────────
def lane_fb_matrix(rank, src, tgt, w, labels, K) -> np.ndarray:
    """K×K matrix: feedback weight from lane_row to lane_col under ``rank``."""
    fb = ~(rank[tgt] > rank[src])
    pairidx = labels[src].astype(np.int64) * K + labels[tgt]
    return np.bincount(pairidx[fb], weights=np.asarray(w, np.float64)[fb],
                       minlength=K * K).reshape(K, K)


def lane_analysis(rows) -> dict | None:
    """z-matrix of real vs L3-null lane-pair feedback weight (lanes are node
    attributes, so they survive rewiring; under L3 every lane-pair's TOTAL
    weight is preserved exactly, so FB weight differences are pure ordering)."""
    epochs = CONFIG["standard_epochs"]["connectome"]
    kind = CONFIG["lane_null"]
    nulls = _sel(rows, dataset="connectome", kind=kind, epochs=epochs)
    reals = _sel(rows, dataset="connectome", kind="real", epochs=epochs)
    if len(nulls) < 10 or not reals:
        print(f"[lane] skipped: {len(nulls)} {kind} nulls, {len(reals)} real runs")
        return None
    g = load_dataset("connectome")
    src = np.asarray(g.src); tgt = np.asarray(g.tgt); w = np.asarray(g.weight)
    labels, lane_list = fly_lane_labels()
    K = len(lane_list)
    total = g.total_weight

    def order_of(r):
        return np.load(_REPO / r["order_npy"]).astype(np.int64)

    r0 = _sel(reals, pipeline_seed=42)[0]
    M_real = lane_fb_matrix(order_of(r0), src, tgt, w, labels, K)

    mats = []
    for r in nulls:
        s2, t2, w2, _ = _draw_null(kind, src, tgt, w, g.n_nodes,
                                   r["gen_seed"], labels)
        mats.append(lane_fb_matrix(order_of(r), s2, t2, w2, labels, K))
    mats = np.stack(mats)                       # (Knulls, K, K)
    mu, sd = mats.mean(axis=0), mats.std(axis=0, ddof=1)

    # total weight per lane pair (identical in real and every L3 null — verified
    # by the generator selftest), used to mask trivial pairs
    pairidx = labels[src].astype(np.int64) * K + labels[tgt]
    Wpair = np.bincount(pairidx, weights=w.astype(np.float64),
                        minlength=K * K).reshape(K, K)
    mask = Wpair >= CONFIG["lane_min_pairs_w_frac"] * total
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (M_real - mu) / sd
    z[~mask] = np.nan
    np.savez_compressed(CONFIG["out_lane_npz"], z=z, real_fb=M_real,
                        null_mean=mu, null_std=sd, w_pair=Wpair,
                        lane_list=np.array(lane_list), mask=mask,
                        n_nulls=len(nulls), null_kind=kind)
    print(f"[lane] z-matrix saved ({len(nulls)} {kind} nulls, "
          f"{int(mask.sum())}/{K * K} pairs above weight floor)")
    return dict(lane_list=lane_list, n_nulls=len(nulls), null_kind=kind,
                n_pairs_scored=int(mask.sum()),
                real_fb_total_pct=float(100 * M_real.sum() / total),
                null_fb_total_pct_mean=float(100 * mats.sum(axis=(1, 2)).mean()
                                             / total))


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────
def _style_ax(ax):
    c = CONFIG["col"]
    ax.set_facecolor(c["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(c["grid"])
    ax.tick_params(colors=c["text2"], labelsize=9)
    ax.grid(True, axis="x", color=c["grid"], lw=0.7)
    ax.set_axisbelow(True)


def fig_ladder(rows, summary, champion) -> None:
    """The fly constraint ladder: nested chain with labeled deltas + references."""
    c = CONFIG["col"]
    epochs = CONFIG["standard_epochs"]["connectome"]
    key = f"connectome_e{epochs}"
    if key not in summary:
        return
    s = summary[key]
    chain = [k for k in ("L1_gnm", "w_source", "L3_dcsbm") if k in s["nulls"]]
    refs = [k for k in ("L3a_sbm", "w_shuffled") if k in s["nulls"]]
    if not chain:
        return
    delta_names = {("chance", "L1_gnm"): "density",
                   ("L1_gnm", "w_source"): "degrees",
                   ("w_source", "L3_dcsbm"): "regional diagram",
                   ("L3_dcsbm", "real"): "fine wiring",
                   ("real", "champion"): "estimator slack"}
    rng = np.random.default_rng(0)
    n_rows = len(chain) + len(refs) + 2
    fig, ax = plt.subplots(figsize=(8.8, 0.85 * n_rows + 1.8))
    fig.patch.set_facecolor(c["surface"])
    _style_ax(ax)
    yt, yl = [], []
    ax.axvline(50, color=c["grid"], lw=1.2, ls=(0, (4, 3)))
    ax.text(50.4, -0.86, "chance 50%", ha="left", va="top", fontsize=8,
            color=c["text2"])

    def strip(vals, y, color, alpha):
        jitter = rng.uniform(-0.14, 0.14, vals.size)
        ax.scatter(vals, y + jitter, s=26, color=color, alpha=alpha,
                   edgecolors=c["surface"], linewidths=1.2, zorder=3)
        m = float(vals.mean())
        ax.plot([m, m], [y - 0.3, y + 0.3], color=color, lw=2.4, zorder=4)
        ax.annotate(f"{m:.2f}", (m, y + 0.36), ha="center", fontsize=8.5,
                    color=c["text"], zorder=5)
        return m

    # reference rows (non-nested / secondary weight variant) at the bottom
    y = 0
    for kind in refs:
        vals = np.array([r["pct_final"] for r in
                         _sel(rows, dataset="connectome", kind=kind,
                              epochs=epochs)])
        strip(vals, y, c["null_light"], 0.9)
        yt.append(y)
        yl.append(f"{CONFIG['ladder_labels'][kind]}\nK={vals.size} — reference")
        y += 1
    # the nested chain
    pts = [("chance", 50.0, None)]
    for kind in chain:
        vals = np.array([r["pct_final"] for r in
                         _sel(rows, dataset="connectome", kind=kind,
                              epochs=epochs)])
        m = strip(vals, y, c["null"], 0.55)
        pts.append((kind, m, y))
        yt.append(y); yl.append(f"{CONFIG['ladder_labels'][kind]}\nK={vals.size}")
        y += 1
    rv = np.array([r["pct_final"] for r in
                   _sel(rows, dataset="connectome", kind="real", epochs=epochs)])
    ax.scatter(rv, np.full(rv.size, y), s=36, color=c["real"], zorder=4)
    ax.axvline(rv.mean(), color=c["real"], lw=1.4, alpha=0.85, zorder=2)
    ax.annotate(f"real (protocol) {rv.mean():.2f}  ", (rv.mean(), y + 0.36),
                ha="right", fontsize=8.5, color=c["text"], weight="bold")
    pts.append(("real", float(rv.mean()), y))
    yt.append(y); yl.append(f"real graph\nprotocol, K={rv.size} seeds")
    y += 1
    if "connectome" in champion:
        cv = champion["connectome"]
        ax.scatter([cv], [y], s=36, marker="D", color=c["champ"], zorder=4)
        ax.annotate(f"champion {cv:.2f}  ", (cv, y + 0.36), ha="right",
                    fontsize=8.5, color=c["text2"])
        pts.append(("champion", cv, y))
        yt.append(y); yl.append("real graph\nchampion budget")
    # labeled deltas along the chain (chance row is the x=50 guide line)
    for (k0, x0, y0), (k1, x1, y1) in zip(pts[:-1], pts[1:]):
        name = delta_names.get((k0, k1), "")
        if y0 is None:
            y0 = pts[1][2] - 0.85
        ax.annotate("", xy=(x1, (y0 + y1) / 2 + 0.5), xytext=(x0, (y0 + y1) / 2),
                    arrowprops=dict(arrowstyle="->", color=c["accent"], lw=1.3,
                                    shrinkA=2, shrinkB=2))
        label = (f"{name} +{x1 - x0:.2f} pp" if x1 >= x0
                 else f"{name} {x1 - x0:.2f} pp")
        if abs(x1 - x0) < 2.0:   # short arrow: label to its left, off the marks
            ax.annotate(label + "  ", (min(x0, x1) - 0.2, (y0 + y1) / 2 + 0.30),
                        ha="right", fontsize=8, color=c["accent"], style="italic")
        else:
            ax.annotate(label, ((x0 + x1) / 2, (y0 + y1) / 2 + 0.30), ha="center",
                        fontsize=8, color=c["accent"], style="italic")
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9, color=c["text"])
    ax.set_ylim(-1.2, y + 0.8)
    ax.set_xlabel("feedforward weight after identical epoch-budgeted protocol (%)",
                  fontsize=10, color=c["text"])
    ax.set_title("Fly connectome vs the constraint ladder of matched random graphs",
                 fontsize=11.5, color=c["text"], loc="left", pad=12)
    fig.tight_layout()
    out = CONFIG["outputs_dir"] / "nullmodels_fly_ladder.png"
    fig.savefig(out, dpi=CONFIG["dpi"], facecolor=c["surface"])
    plt.close(fig)
    print(f"[fig] {out.name}")


def fig_mouse(rows, summary) -> None:
    """Mouse null distributions (histograms) with the real value marked."""
    c = CONFIG["col"]
    epochs = CONFIG["standard_epochs"]["mouse"]
    kinds = [k for k in ("L1_gnm", "w_source", "w_shuffled")
             if _sel(rows, dataset="mouse", kind=k, epochs=epochs)]
    if not kinds:
        return
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.3),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(c["surface"])
    axes = np.atleast_1d(axes)
    rv = np.array([r["pct_final"] for r in
                   _sel(rows, dataset="mouse", kind="real", epochs=epochs)])
    for ax, kind in zip(axes, kinds):
        _style_ax(ax)
        ax.grid(True, axis="y", color=c["grid"], lw=0.7)
        ax.grid(False, axis="x")
        vals = np.array([r["pct_final"] for r in
                         _sel(rows, dataset="mouse", kind=kind, epochs=epochs)])
        ax.hist(vals, bins=18, color=c["null"], edgecolor=c["surface"],
                linewidth=1.0, zorder=3)
        ax.axvline(rv.mean(), color=c["real"], lw=1.8, zorder=4)
        ax.annotate("real", (rv.mean(), ax.get_ylim()[1] * 0.92),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=c["text"], weight="bold")
        ax.set_title(f"{CONFIG['ladder_labels'][kind]}  (K={vals.size})",
                     fontsize=10, color=c["text"], loc="left")
        ax.set_xlabel("feedforward weight (%)", fontsize=9, color=c["text2"])
    axes[0].set_ylabel("null realizations", fontsize=9, color=c["text2"])
    fig.suptitle("Mouse meso-connectome vs matched random graphs "
                 "(identical protocol, 2,000 epochs)", fontsize=11,
                 color=c["text"], x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = CONFIG["outputs_dir"] / "nullmodels_mouse_nulls.png"
    fig.savefig(out, dpi=CONFIG["dpi"], facecolor=c["surface"])
    plt.close(fig)
    print(f"[fig] {out.name}")


def fig_lane_z() -> None:
    """Diverging heatmap of the lane-pair FB z-matrix (blue = less FB than null,
    red = more FB than null; gray midpoint = as-null)."""
    if not CONFIG["out_lane_npz"].exists():
        return
    c = CONFIG["col"]
    d = np.load(CONFIG["out_lane_npz"], allow_pickle=False)
    z = d["z"]; lane_list = [str(x) for x in d["lane_list"]]
    K = len(lane_list)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#7a1413"])
    cmap.set_bad("#ffffff")
    lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    fig.patch.set_facecolor(c["surface"])
    im = ax.imshow(z, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_xticks(range(K)); ax.set_xticklabels(lane_list, rotation=45,
                                                ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels(lane_list, fontsize=8)
    ax.set_xlabel("target lane", fontsize=10, color=c["text"])
    ax.set_ylabel("source lane", fontsize=10, color=c["text"])
    ax.set_title("Fly feedback weight per lane pair: real minus L3-null, in null σ\n"
                 f"(K={int(d['n_nulls'])} degree-corrected block nulls; white = "
                 "pair below weight floor)", fontsize=10.5, color=c["text"],
                 loc="left", pad=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("z (red = more feedback than null)", fontsize=9, color=c["text2"])
    cb.outline.set_visible(False)
    fig.tight_layout()
    out = CONFIG["outputs_dir"] / "nullmodels_fly_lane_z.png"
    fig.savefig(out, dpi=CONFIG["dpi"], facecolor=c["surface"])
    plt.close(fig)
    print(f"[fig] {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    rows = load_runs()
    print(f"loaded {len(rows)} run records")
    summary = summarize(rows)
    champion = {}
    if any(r["dataset"] == "connectome" for r in rows):
        champion["connectome"] = champion_pct_fly()
        print(f"fly champion (re-scored by frozen oracle): "
              f"{champion['connectome']:.5f}%")
    # mouse champion: quoted from the registry (autoresearch/sota.json, H63,
    # 20 seeds, measured on CUDA) — no local order artifact to re-score.
    sota = json.loads((_REPO / "autoresearch" / "sota.json").read_text())
    champion["mouse"] = float(sota["datasets"]["mouse"]["pct_mean"])
    ladder = ladder_table(summary, champion)
    lane = lane_analysis(rows)
    bounds = structural_bounds(rows)
    out = dict(summary=summary, ladder=ladder, champion=champion, lane=lane,
               two_cycle_bounds=bounds, n_runs=len(rows))
    CONFIG["out_summary"].write_text(json.dumps(out, indent=1))
    print(f"summary -> {CONFIG['out_summary']}")

    CONFIG["outputs_dir"].mkdir(parents=True, exist_ok=True)
    fig_ladder(rows, summary, champion)
    fig_mouse(rows, summary)
    fig_lane_z()

    # console table for the study report
    for key, s in summary.items():
        print(f"\n== {key} ==  real final {s['real']['final']['mean']:.4f}% "
              f"(± {s['real']['final']['std']:.4f}, K={s['real']['final']['K']}) "
              f"greedy {s['real']['greedy']['mean']:.4f}%")
        for kind, t in s["nulls"].items():
            print(f"  {kind:11s} K={t['final']['K']:3d} "
                  f"final {t['final']['mean']:8.4f}±{t['final']['std']:.4f} "
                  f"[{t['final']['min']:.4f},{t['final']['max']:.4f}] "
                  f"z={t['z_final']:+6.2f} beyond={t['n_nulls_beyond_real_final']} "
                  f"| greedy {t['greedy']['mean']:8.4f}±{t['greedy']['std']:.4f} "
                  f"z={t['z_greedy']:+6.2f} | {t['mean_runtime_s']:.0f}s/run")


if __name__ == "__main__":
    main()
