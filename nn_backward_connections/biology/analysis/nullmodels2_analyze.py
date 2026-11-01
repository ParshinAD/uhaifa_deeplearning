"""Null-ladder v2 analysis (study: biology/studies/null_ladder_v2/CHARTER.md).

Builds the EXTENDED fly ladder — the reviewed G01 rungs (loaded from
``biology/results/nullmodels/runs/``) plus the v2 rungs W0 ``w_topo_shuffle``
and L2R ``dyad_fixedmut`` (from ``biology/results/nullmodels_v2/runs/``) — and
reports, per rung: K, protocol FF% mean +- sd, min/max, z of the real graph,
draws at/beyond the real value, greedy FF%, and the estimator-free 2-cycle
lower bound on unavoidable feedback.

2-cycle bounds for the v1 rungs are not in the v1 run JSONs; they are
recomputed here by REGENERATING each null from its logged ``gen_seed`` (the
G01 review demonstrated 13/13 exact reconstructions from seed + variant) and
cached in ``bounds_cache.json``. v2 runs log their bound directly.

Outputs
-------
* ``biology/results/nullmodels_v2/summary_v2.json`` — every number below.
* ``biology/outputs/nullmodels2_fly_ladder.png`` — two panels: the FF% ladder
  and the 2-cycle bound per rung (log axis).
* a markdown table on stdout, ready for REPORT.md.

Run:  $PY biology/analysis/nullmodels2_analyze.py            # full
      $PY biology/analysis/nullmodels2_analyze.py --no-bounds  # skip regen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "biology" / "analysis"))

from mfas.io import load_dataset                                    # noqa: E402

from nullmodels_generators import (block_configuration_null,        # noqa: E402
                                   directed_configuration_null,
                                   fly_lane_labels, gnm_null, sbm_null)
from nullmodels2_generators import two_cycle_bound                  # noqa: E402

CONFIG = dict(
    dataset="connectome",
    epochs=4_000,                      # the protocol budget (filters out probes)
    sift_max_sweeps=8,
    v1_runs=_REPO / "biology" / "results" / "nullmodels" / "runs",
    v2_runs=_REPO / "biology" / "results" / "nullmodels_v2" / "runs",
    bounds_cache=_REPO / "biology" / "results" / "nullmodels_v2" / "bounds_cache.json",
    summary_out=_REPO / "biology" / "results" / "nullmodels_v2" / "summary_v2.json",
    figure_out=_REPO / "biology" / "outputs" / "nullmodels2_fly_ladder.png",
    sota=_REPO / "autoresearch" / "sota.json",
    # ladder order, bottom -> top (constraint strength / narrative order)
    rungs=["L1_gnm", "L3a_sbm", "w_shuffled", "w_source", "L3_dcsbm",
           "dyad_fixedmut", "type_dcsbm", "w_topo_shuffle"],
    rung_label=dict(
        L1_gnm="L1  G(n,m) — density only",
        L3a_sbm="L3a SBM — lane blocks, no degrees",
        w_shuffled="L2  config — degrees",
        w_source="L2  config — degrees + out-strengths",
        L3_dcsbm="L3  dc-SBM — + 17-lane profiles",
        dyad_fixedmut="L2R + REAL reciprocal backbone (v2)",
        type_dcsbm="L4  dc-SBM — 8.8k cell-type profiles (v2)",
        w_topo_shuffle="W0  REAL topology, weights shuffled (v2)",
    ),
    v2_rungs={"dyad_fixedmut", "w_topo_shuffle", "type_dcsbm"},
    # palette (dataviz reference instance, light mode)
    col=dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e",
             muted="#898781", grid="#e1e0d9", baseline="#c3c2b7",
             v1="#2a78d6", v2="#eb6834"),
    dpi=120,
)


# ──────────────────────────────────────────────────────────────────────────────
# Loading runs
# ──────────────────────────────────────────────────────────────────────────────
def _load_runs(runs_dir: Path) -> list[dict]:
    out = []
    for p in sorted(runs_dir.glob("*.json")):
        r = json.loads(p.read_text())
        if (r.get("dataset") == CONFIG["dataset"]
                and r.get("epochs") == CONFIG["epochs"]
                and r.get("sift_max_sweeps") == CONFIG["sift_max_sweeps"]):
            r["_file"] = p.name
            out.append(r)
    return out


def _champion_pct() -> float:
    sota = json.loads(CONFIG["sota"].read_text())
    row = sota["datasets"][CONFIG["dataset"]] if "datasets" in sota else sota[CONFIG["dataset"]]
    return float(row.get("pct_mean_exact", row["pct_mean"]))


# ──────────────────────────────────────────────────────────────────────────────
# 2-cycle bounds for v1 rungs (regenerate from logged gen_seed; cached)
# ──────────────────────────────────────────────────────────────────────────────
def _v1_bound(kind: str, gen_seed: int, g, labels) -> float:
    src = np.asarray(g.src); tgt = np.asarray(g.tgt); w = np.asarray(g.weight)
    if kind in ("w_shuffled", "w_source"):
        s2, t2, w2, _ = directed_configuration_null(src, tgt, w, variant=kind,
                                                    seed=gen_seed)
    elif kind == "L1_gnm":
        s2, t2, w2, _ = gnm_null(src, tgt, w, n=g.n_nodes, seed=gen_seed)
    elif kind == "L3a_sbm":
        s2, t2, w2, _ = sbm_null(src, tgt, w, labels, seed=gen_seed)
    elif kind == "L3_dcsbm":
        s2, t2, w2, _ = block_configuration_null(src, tgt, w, labels,
                                                 seed=gen_seed)
    else:
        raise ValueError(kind)
    return two_cycle_bound(s2, t2, w2, g.n_nodes)


def collect_bounds(runs_v1: list[dict], compute: bool) -> dict:
    cache = {}
    if CONFIG["bounds_cache"].exists():
        cache = json.loads(CONFIG["bounds_cache"].read_text())
    if not compute:
        return cache
    need = [(r["kind"], r["gen_seed"]) for r in runs_v1
            if r["kind"] != "real" and f"{r['kind']}__gs{r['gen_seed']}" not in cache]
    need = sorted(set(need))
    if need:
        g = load_dataset(CONFIG["dataset"])
        W = float(np.asarray(g.weight).sum())
        labels = None
        if any(k in ("L3a_sbm", "L3_dcsbm") for k, _ in need):
            labels, _ = fly_lane_labels()
        for kind, gs in need:
            b = _v1_bound(kind, gs, g, labels)
            cache[f"{kind}__gs{gs}"] = 100.0 * b / W
            print(f"[bound] {kind} gs{gs}: {cache[f'{kind}__gs{gs}']:.4f}% of W",
                  flush=True)
        CONFIG["bounds_cache"].write_text(json.dumps(cache, indent=1))
    return cache


# ──────────────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────────────
def rung_stats(runs: list[dict], bounds: dict, real_mean: float) -> dict:
    out = {}
    for kind in CONFIG["rungs"]:
        rs = [r for r in runs if r["kind"] == kind]
        if not rs:
            continue
        ff = np.array([r["pct_final"] for r in rs])
        gr = np.array([r["pct_greedy"] for r in rs])
        if kind in CONFIG["v2_rungs"]:
            b = np.array([r["bound_2cycle_pct"] for r in rs])
        else:
            b = np.array([bounds.get(f"{kind}__gs{r['gen_seed']}", np.nan)
                          for r in rs])
        sd = float(ff.std(ddof=1)) if ff.size > 1 else 0.0
        out[kind] = dict(
            K=int(ff.size), ff_mean=float(ff.mean()), ff_sd=sd,
            ff_min=float(ff.min()), ff_max=float(ff.max()),
            ff_draws=[float(x) for x in ff],
            greedy_mean=float(gr.mean()),
            z_real=float((real_mean - ff.mean()) / sd) if sd > 0 else None,
            n_beyond=int((ff >= real_mean).sum()),
            bound_mean=float(np.nanmean(b)) if b.size else None,
            bound_sd=float(np.nanstd(b, ddof=1)) if b.size > 1 else 0.0,
            bound_draws=[None if np.isnan(x) else float(x) for x in b],
            files=[r["_file"] for r in rs],
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Figure (light mode; palette = dataviz reference instance)
# ──────────────────────────────────────────────────────────────────────────────
def make_figure(stats: dict, real_mean: float, real_sd: float,
                champion: float, real_bound_pct: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = CONFIG["col"]
    rungs = [k for k in CONFIG["rungs"] if k in stats]
    ys = np.arange(len(rungs))
    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(13.5, 0.62 * len(rungs) + 2.4), width_ratios=[2.1, 1.0],
        facecolor=C["surface"])

    for a in (ax, axb):
        a.set_facecolor(C["surface"])
        a.tick_params(colors=C["muted"], labelsize=9)
        for s in a.spines.values():
            s.set_color(C["baseline"])
        a.spines[["top", "right"]].set_visible(False)
        a.set_yticks(ys)
        a.grid(axis="x", color=C["grid"], lw=0.8)
        a.set_axisbelow(True)

    # ── panel A: the FF% ladder ──
    for y, kind in zip(ys, rungs):
        st = stats[kind]
        c = C["v2"] if kind in CONFIG["v2_rungs"] else C["v1"]
        ax.scatter(st["ff_draws"], [y] * st["K"], s=14, color=c, alpha=0.35,
                   zorder=3, linewidths=0)
        ax.errorbar(st["ff_mean"], y, xerr=max(st["ff_sd"], 1e-9), fmt="o",
                    ms=7, color=c, ecolor=c, elinewidth=2, capsize=3, zorder=4)
        ax.annotate(f"{st['ff_mean']:.2f}", (st["ff_mean"], y),
                    xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8.5, color=C["secondary"])
    ax.axvline(real_mean, color=C["ink"], lw=1.6, zorder=2)
    ax.axvline(champion, color=C["ink"], lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.set_ylim(-0.7, len(rungs) - 0.3)
    ax.annotate(f"real graph,\nsame protocol\n{real_mean:.3f}%",
                (real_mean, 1.0), ha="right", va="center",
                fontsize=8.5, color=C["ink"], xytext=(-6, 0),
                textcoords="offset points")
    ax.annotate(f"champion (H64)\n{champion:.3f}%",
                (champion, len(rungs) - 2.0), ha="left", va="center",
                fontsize=8.5, color=C["ink"], xytext=(6, 0),
                textcoords="offset points")
    ax.set_yticklabels([CONFIG["rung_label"][k] for k in rungs], fontsize=9.5,
                       color=C["ink"])
    ax.set_xlabel("feedforward weight after the identical protocol (% of total)",
                  color=C["secondary"], fontsize=10)
    ax.set_title("Fly connectome vs the extended null ladder — "
                 "every rung, identical estimator",
                 loc="left", fontsize=12, color=C["ink"], pad=12)

    # ── panel B: 2-cycle lower bound (estimator-free), log axis ──
    for y, kind in zip(ys, rungs):
        st = stats[kind]
        bd = [b for b in st["bound_draws"] if b is not None]
        if not bd:
            continue
        c = C["v2"] if kind in CONFIG["v2_rungs"] else C["v1"]
        axb.scatter(bd, [y] * len(bd), s=14, color=c, alpha=0.35, zorder=3,
                    linewidths=0)
        axb.scatter([float(np.mean(bd))], [y], s=46, color=c, zorder=4)
    axb.axvline(real_bound_pct, color=C["ink"], lw=1.6, zorder=2)
    axb.annotate(f"real\n{real_bound_pct:.2f}%", (real_bound_pct, -0.35),
                 ha="right", va="bottom", fontsize=8.5, color=C["ink"],
                 xytext=(-6, 0), textcoords="offset points")
    axb.set_xscale("log")
    axb.set_yticklabels([])
    axb.set_xlabel("2-cycle lower bound (% of W, log)",
                   color=C["secondary"], fontsize=10)
    axb.set_title("Forced reciprocal feedback", loc="left", fontsize=12,
                  color=C["ink"], pad=12)

    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=C["v1"], marker="o", lw=0, ms=7,
                      label="G01 rungs (reviewed)"),
        mlines.Line2D([], [], color=C["v2"], marker="o", lw=0, ms=7,
                      label="v2 rungs (this study)"),
        mlines.Line2D([], [], color=C["ink"], lw=1.6,
                      label="real graph (protocol)"),
        mlines.Line2D([], [], color=C["ink"], lw=1.2, ls=(0, (4, 3)),
                      label="champion budget"),
    ]
    ax.legend(handles=handles, loc="center left", fontsize=8.5, frameon=False,
              labelcolor=C["ink"])
    fig.tight_layout()
    CONFIG["figure_out"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONFIG["figure_out"], dpi=CONFIG["dpi"],
                facecolor=C["surface"], bbox_inches="tight")
    print(f"[figure] {CONFIG['figure_out'].relative_to(_REPO)}")


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-bounds", action="store_true",
                    help="skip regenerating v1 nulls for 2-cycle bounds")
    args = ap.parse_args()

    runs_v1 = _load_runs(CONFIG["v1_runs"])
    runs_v2 = _load_runs(CONFIG["v2_runs"])
    real = [r for r in runs_v1 if r["kind"] == "real"]
    assert len(real) >= 3, "expected the 3 G01 real-graph protocol runs"
    real_ff = np.array([r["pct_final"] for r in real])
    real_mean, real_sd = float(real_ff.mean()), float(real_ff.std(ddof=1))
    champion = _champion_pct()

    g = load_dataset(CONFIG["dataset"])
    W = float(np.asarray(g.weight).sum())
    real_bound_pct = 100.0 * two_cycle_bound(
        np.asarray(g.src), np.asarray(g.tgt), np.asarray(g.weight),
        g.n_nodes) / W

    bounds = collect_bounds(runs_v1, compute=not args.no_bounds)
    stats = rung_stats(runs_v1 + runs_v2, bounds, real_mean)

    summary = dict(
        dataset=CONFIG["dataset"], epochs=CONFIG["epochs"],
        real_protocol=dict(mean=real_mean, sd=real_sd, K=len(real),
                           draws=[float(x) for x in real_ff]),
        champion_pct=champion, real_bound_2cycle_pct=real_bound_pct,
        rungs=stats,
    )
    CONFIG["summary_out"].write_text(json.dumps(summary, indent=1))
    print(f"[summary] {CONFIG['summary_out'].relative_to(_REPO)}")

    # markdown table for REPORT.md
    print("\n| rung | K | FF% (protocol) | z of real | beyond | greedy FF% | "
          "2-cycle bound % of W |")
    print("|---|---|---|---|---|---|---|")
    for kind in CONFIG["rungs"]:
        if kind not in stats:
            continue
        st = stats[kind]
        z = f"{st['z_real']:+.1f}" if st["z_real"] is not None else "—"
        bm = (f"{st['bound_mean']:.4f} ± {st['bound_sd']:.4f}"
              if st["bound_mean"] is not None and not np.isnan(st["bound_mean"])
              else "—")
        print(f"| {CONFIG['rung_label'][kind]} | {st['K']} | "
              f"{st['ff_mean']:.3f} ± {st['ff_sd']:.3f} | {z} | "
              f"{st['n_beyond']}/{st['K']} | {st['greedy_mean']:.3f} | {bm} |")
    print(f"| **real graph (protocol)** | {len(real)} | "
          f"**{real_mean:.3f} ± {real_sd:.3f}** | — | — | "
          f"{float(np.mean([r['pct_greedy'] for r in real])):.3f} | "
          f"{real_bound_pct:.4f} |")
    print(f"| real graph (champion budget) | 1 | {champion:.3f} | — | — | — | "
          f"{real_bound_pct:.4f} |")

    make_figure(stats, real_mean, real_sd, champion, real_bound_pct)


if __name__ == "__main__":
    main()
