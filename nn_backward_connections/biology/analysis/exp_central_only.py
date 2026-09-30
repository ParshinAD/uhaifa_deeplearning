"""Central-brain-only MFAS: is the central brain's internal order self-determined?

Research question
-----------------
The fly-connectome champion order (H64, 84.258% feedforward) was optimized on the
WHOLE brain (136,648 neurons). Here we run the SAME champion pipeline on the induced
subgraph of neurons with ``super_class == 'central'`` only (all edges to/from
non-central neurons removed) and ask: how much does the internal order of the central
neurons change relative to their order within the full-brain champion? I.e. is the
center's order dictated by its own recurrent wiring, or by its periphery interfaces
(optic lobes, sensory afferents, descending outputs)?

Pipeline (production functions, imported - never copied)
--------------------------------------------------------
    stage 1  greedy-FAS warm start          src.mfas.experiments.H02.greedy_fas_order
    stage 2  Rocket + ASYM surrogate        src.mfas.experiments.H64._rocket_asym
    stage 3  under-relaxed exact-gain sift  src.mfas.refine.sift_underrelaxed
    stage 4  SCC-block <-> sift alternation src.mfas.refine.alternate_scc_sift
    oracle   frozen exact scorer            src.mfas.metrics.score_from_order (CPU)

Champion constants are H64's connectome constants (see CONFIG below and
docs/h64_minimal.py). MPS kernels are non-deterministic, so the pipeline is run
TWICE at the same seed; Spearman(run1, run2) is the noise floor against which the
"order changed vs champion" claim is judged.

Stages of this script (run from the repo root):
    PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
    $PY biology/analysis/exp_central_only.py --stage subgraph
    $PY biology/analysis/exp_central_only.py --stage smoke
    $PY biology/analysis/exp_central_only.py --stage run --run-id 1
    $PY biology/analysis/exp_central_only.py --stage run --run-id 2
    $PY biology/analysis/exp_central_only.py --stage analyze
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# CONFIG - every constant of the study lives here (no magic numbers below)
# ---------------------------------------------------------------------------
CONFIG = dict(
    dataset="connectome",
    super_class_keep="central",          # induced subgraph = this super_class only
    seed=42,                             # champion seed (pipeline is greedy-init: seed
                                         # is never drawn from; MPS noise remains)
    device_prefer="auto",                # MPS > CUDA > CPU (select_device)
    n_runs=2,                            # same-seed repeats = MPS noise floor
    # -- stage 2: Rocket + ASYM surrogate (H64 connectome champion constants) --
    epochs=20_000,
    beta_cycles=5,
    lr=0.05,
    lr_decay_start=0.5,
    lr_end_factor=0.1,
    grad_clip=1.0,
    log_interval=100,
    asym_M=0.75,
    asym_T=1.5,
    # -- stage 3: under-relaxed exact-gain sift --
    sift_max_sweeps=40,
    sift_k_full=6,
    sift_alpha=0.7,
    # -- stage 4: SCC-block <-> sift alternation --
    alt_cycles=77,
    alt_sift_sweeps=2,
    alt_k_full=2,
    alt_alpha=0.7,
    min_block=32,
    split_fracs=(0.5, 0.382, 0.618, 0.25, 0.75),
    # -- smoke-test overrides (wiring check only; results never reported) --
    smoke_epochs=200,
    smoke_sift_max_sweeps=4,
    smoke_alt_cycles=2,
    # -- analysis --
    move_threshold_frac=0.05,            # "a neuron moved" = >5% of the line length
    min_class_n=20,                      # per-class stats only for classes this big
    top_movers=12,                       # rows in the top-mover table
    # -- paths (all outputs under biology/ only) --
    ann_npz="layering/outputs/fly_ann/node_ann.npz",
    ann_meta="layering/outputs/fly_ann/node_ann_meta.json",
    results_dir="biology/results",
    outputs_dir="biology/outputs",
    subgraph_npz="biology/results/central_subgraph.npz",
    subgraph_meta="biology/results/central_subgraph_meta.json",
    champ_restricted_npy="biology/results/central_only_champ_restricted_rank.npy",
    run_rank_npy="biology/results/central_only_run{run_id}_rank.npy",
    run_json="biology/results/central_only_run{run_id}.json",
    final_json="biology/results/central_only.json",
    fig_scatter="biology/outputs/central_only_rank_vs_champion_hexbin.png",
    fig_classes="biology/outputs/central_only_class_displacement.png",
    fig_dpi=120,
    # -- figure palette (dataviz skill reference palette; one-hue sequential) --
    seq_ramp=("#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"),
    bar_color="#2a78d6",
    accent_color="#eb6834",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _p(rel: str) -> Path:
    """Absolute path for a repo-relative path string."""
    return REPO_ROOT / rel


def now_iso() -> str:
    """Current local timestamp, ISO-8601 with seconds."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_full_graph_and_champion():
    """Load the full connectome + pinned H64 champion order (parity-gated).

    Returns ``(g, order, rank, ff_pct)`` exactly as ``layering.io_utils`` does.
    """
    from layering import io_utils
    return io_utils.load_graph_and_champion_order(CONFIG["dataset"])


def load_annotations(champion_rank: np.ndarray):
    """Load node annotations and their meta; assert they match the champion rank.

    The staleness gate: ``node_ann.npz['rank']`` was written against the pinned
    champion; if it no longer equals the loaded champion rank the annotation file
    is stale and every per-class number would be garbage.
    """
    ann = np.load(_p(CONFIG["ann_npz"]))
    meta = json.loads(_p(CONFIG["ann_meta"]).read_text(encoding="utf-8"))
    assert np.array_equal(np.asarray(ann["rank"], dtype=np.int64),
                          np.asarray(champion_rank, dtype=np.int64)), (
        "STALENESS GATE FAILED: node_ann.npz['rank'] != pinned champion rank; "
        "the annotation join is stale - rebuild it before using per-class stats.")
    return ann, meta


def compress_rank(rank_subset: np.ndarray) -> np.ndarray:
    """Compress a rank vector restricted to a subset into a dense 0..m-1 permutation.

    Preserves the relative order (stable), so every internal edge keeps its
    feedforward/feedback status - the FF% of the subgraph under the compressed
    rank equals the FF% of the subgraph's edges inside the full order.
    """
    r = np.asarray(rank_subset)
    return np.argsort(np.argsort(r, kind="stable"), kind="stable").astype(np.int64)


# ---------------------------------------------------------------------------
# Stage: subgraph - build + cache the induced central-brain graph
# ---------------------------------------------------------------------------
def stage_subgraph() -> None:
    """Build the induced 'central' subgraph, cache it, and log its statistics.

    Keeps nodes with ``super_class == 'central'`` and edges with BOTH endpoints
    central; node indices are relabeled to ``0..m-1`` (ascending original index,
    so ``sub_node_ids`` maps back). Also saves the champion order restricted to
    the central nodes (compressed ranks) - comparator (b) of the study.
    """
    from src.mfas.io import GraphData, save_processed
    from src.mfas.metrics import score_from_order, pct

    g, _order, rank, ff_pct = load_full_graph_and_champion()
    ann, meta = load_annotations(rank)
    sc_values = meta["super_class_values"]
    central_code = sc_values.index(CONFIG["super_class_keep"])
    mask = np.asarray(ann["super_class"]) == central_code
    central_nodes = np.flatnonzero(mask)                     # original indices, ascending
    m = int(central_nodes.size)

    new_index = np.full(g.n_nodes, -1, dtype=np.int64)
    new_index[central_nodes] = np.arange(m)

    src = np.asarray(g.src, dtype=np.int64)
    tgt = np.asarray(g.tgt, dtype=np.int64)
    keep = mask[src] & mask[tgt]
    sub_src = new_index[src[keep]].astype(np.int32)
    sub_tgt = new_index[tgt[keep]].astype(np.int32)
    sub_w = np.asarray(g.weight)[keep]

    gc = GraphData(src=sub_src, tgt=sub_tgt, weight=sub_w,
                   node_ids=np.asarray(g.node_ids)[central_nodes],
                   name="central_sub")
    save_processed(gc, _p(CONFIG["subgraph_npz"]))

    # Comparator (b): champion order restricted to central nodes, ranks compressed.
    champ_restricted = compress_rank(np.asarray(rank)[central_nodes])
    np.save(_p(CONFIG["champ_restricted_npy"]), champ_restricted)
    score_b = score_from_order(champ_restricted, sub_src, sub_tgt, sub_w)

    stats = dict(
        built_at=now_iso(),
        super_class_keep=CONFIG["super_class_keep"],
        central_code=central_code,
        n_nodes=m,
        n_nodes_full=g.n_nodes,
        n_nodes_pct_of_full=100.0 * m / g.n_nodes,
        n_edges=int(sub_w.shape[0]),
        n_edges_full=g.n_edges,
        n_edges_pct_of_full=100.0 * sub_w.shape[0] / g.n_edges,
        total_weight=float(sub_w.sum()),
        total_weight_full=g.total_weight,
        total_weight_pct_of_full=100.0 * float(sub_w.sum()) / g.total_weight,
        champion_ff_pct_full=ff_pct,
        champ_restricted_score=score_b,
        champ_restricted_ff_pct=pct(score_b, float(sub_w.sum())),
        ann_unmatched_note=("13 of 136,648 nodes have no annotation row (coded NA) "
                            "and are excluded from 'central' by construction"),
    )
    _p(CONFIG["subgraph_meta"]).write_text(json.dumps(stats, indent=1),
                                           encoding="utf-8")
    print(json.dumps(stats, indent=1))


# ---------------------------------------------------------------------------
# Stage: run - the champion pipeline on the central subgraph
# ---------------------------------------------------------------------------
def run_champion_pipeline(gc, cfg: dict, seed: int, device) -> dict:
    """Run stages 1-4 of the H64 connectome champion on graph ``gc``.

    Uses the PRODUCTION functions (H02 warm start, H64's ``_rocket_asym``,
    ``sift_underrelaxed``, ``alternate_scc_sift``) with the constants in ``cfg``;
    best-by-oracle across stages 2-4, exactly as ``H64.run`` composes them
    (stages 5/6 have zero budget on the connectome champion and are omitted).
    Returns a dict with per-stage scores, times, and the final rank vector.
    """
    from src.mfas.baseline.rocket import RocketConfig
    from src.mfas.experiments.H02 import greedy_fas_order, _init_positions_from_order
    from src.mfas.experiments.H38 import M as ASYM_M, T as ASYM_T
    from src.mfas.experiments.H64 import _rocket_asym
    from src.mfas.metrics import pct, score_from_order
    from src.mfas.refine import alternate_scc_sift, sift_underrelaxed

    # The ASYM surrogate constants live in H38; assert they are the champion's.
    assert (ASYM_M, ASYM_T) == (cfg["asym_M"], cfg["asym_T"]), (
        "H38's surrogate constants changed - CONFIG no longer matches production.")
    rocket_cfg = RocketConfig(epochs=cfg["epochs"])
    assert (rocket_cfg.cycles, rocket_cfg.lr, rocket_cfg.lr_decay_start,
            rocket_cfg.lr_end_factor, rocket_cfg.grad_clip,
            rocket_cfg.log_interval) == (
        cfg["beta_cycles"], cfg["lr"], cfg["lr_decay_start"],
        cfg["lr_end_factor"], cfg["grad_clip"], cfg["log_interval"]), (
        "RocketConfig defaults changed - CONFIG no longer matches production.")

    src, tgt, w = np.asarray(gc.src), np.asarray(gc.tgt), np.asarray(gc.weight)
    total = gc.total_weight
    log = dict(stages={})

    t0 = time.time()
    print(f"[{now_iso()}] stage 1: greedy-FAS warm start", flush=True)
    order1 = greedy_fas_order(gc)
    s1 = score_from_order(order1, src, tgt, w)
    t1 = time.time() - t0
    log["stages"]["1_greedy"] = dict(score=s1, pct=pct(s1, total), wall_s=t1)
    print(f"  greedy {pct(s1, total):.4f}%  ({t1:.1f}s)", flush=True)

    print(f"[{now_iso()}] stage 2: Rocket + ASYM surrogate "
          f"({cfg['epochs']} epochs)", flush=True)
    init_positions = _init_positions_from_order(order1, device)
    rocket = _rocket_asym(gc, rocket_cfg, seed=seed, device=device,
                          init_positions=init_positions)
    s2 = rocket.best_score
    log["stages"]["2_rocket_asym"] = dict(score=s2, pct=pct(s2, total),
                                          wall_s=rocket.wall_clock_s,
                                          n_epochs_done=rocket.n_epochs_done)
    print(f"  pure Rocket {pct(s2, total):.4f}%  ({rocket.wall_clock_s:.1f}s)",
          flush=True)
    rank2 = compress_rank(rocket.best_positions)

    print(f"[{now_iso()}] stage 3: under-relaxed exact-gain sift", flush=True)
    t3a = time.time()
    rank3, s3, sweep_log = sift_underrelaxed(
        gc, rank2, k_full=cfg["sift_k_full"], alpha=cfg["sift_alpha"],
        max_sweeps=cfg["sift_max_sweeps"])
    t3 = time.time() - t3a
    log["stages"]["3_sift"] = dict(score=s3, pct=pct(s3, total), wall_s=t3,
                                   n_sweeps=len(sweep_log))
    print(f"  after sift {pct(s3, total):.4f}%  ({t3:.1f}s, "
          f"{len(sweep_log)} sweeps)", flush=True)

    print(f"[{now_iso()}] stage 4: SCC-block <-> sift alternation "
          f"({cfg['alt_cycles']} cycles)", flush=True)
    t4a = time.time()
    rank4, s4, alt_log = alternate_scc_sift(
        gc, rank3, n_cycles=cfg["alt_cycles"],
        sift_sweeps=cfg["alt_sift_sweeps"], k_full=cfg["alt_k_full"],
        alpha=cfg["alt_alpha"], min_block=cfg["min_block"],
        split_fracs=cfg["split_fracs"])
    t4 = time.time() - t4a
    log["stages"]["4_alt_scc"] = dict(score=s4, pct=pct(s4, total), wall_s=t4,
                                      n_cycles=len(alt_log))
    print(f"  final {pct(s4, total):.4f}%  ({t4:.1f}s)", flush=True)

    # Best-by-oracle across stages 2-4 (H64.run's composition rule).
    cands = [(s2, rank2), (s3, np.asarray(rank3, dtype=np.int64)),
             (s4, np.asarray(rank4, dtype=np.int64))]
    best_score, best_rank = max(cands, key=lambda t: t[0])
    # Frozen-oracle recheck of the returned rank.
    recheck = score_from_order(best_rank, src, tgt, w)
    assert recheck == best_score, "oracle recheck mismatch on the final rank"
    log["final"] = dict(score=best_score, pct=pct(best_score, total),
                        wall_s_total=time.time() - t0)
    log["rank"] = best_rank
    return log


def stage_run(run_id: int, smoke: bool = False) -> None:
    """Execute one full central-only champion run and persist its outputs.

    ``smoke=True`` shrinks epochs/sweeps/cycles (CONFIG smoke_* values) to verify
    the plumbing end-to-end; smoke results are printed only, never persisted.
    """
    import torch
    from src.mfas.io import load_processed
    from src.mfas.utils.seeding import select_device

    cfg = dict(CONFIG)
    if smoke:
        cfg["epochs"] = cfg["smoke_epochs"]
        cfg["sift_max_sweeps"] = cfg["smoke_sift_max_sweeps"]
        cfg["alt_cycles"] = cfg["smoke_alt_cycles"]

    gc = load_processed(_p(CONFIG["subgraph_npz"]))
    device, device_name = select_device(CONFIG["device_prefer"])
    print(f"[{now_iso()}] run_id={run_id} smoke={smoke} seed={cfg['seed']} "
          f"device={device_name} torch={torch.__version__} "
          f"graph: n={gc.n_nodes:,} m={gc.n_edges:,} "
          f"W={gc.total_weight:,.0f}", flush=True)

    log = run_champion_pipeline(gc, cfg, seed=cfg["seed"], device=device)
    if smoke:
        print("SMOKE OK - wiring verified; nothing persisted.")
        return

    rank = log.pop("rank")
    np.save(_p(CONFIG["run_rank_npy"].format(run_id=run_id)), rank)
    record = dict(
        run_id=run_id, seed=cfg["seed"], device=device_name,
        torch=torch.__version__, python=platform.python_version(),
        machine=platform.platform(), finished_at=now_iso(),
        config={k: cfg[k] for k in
                ("epochs", "beta_cycles", "lr", "lr_decay_start", "lr_end_factor",
                 "grad_clip", "log_interval", "asym_M", "asym_T",
                 "sift_max_sweeps", "sift_k_full", "sift_alpha", "alt_cycles",
                 "alt_sift_sweeps", "alt_k_full", "alt_alpha", "min_block",
                 "split_fracs")},
        **log)
    out = _p(CONFIG["run_json"].format(run_id=run_id))
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"[{now_iso()}] saved {out}")


# ---------------------------------------------------------------------------
# Stage: analyze - comparisons, per-class stats, figures, final JSON
# ---------------------------------------------------------------------------
def order_comparison(rank_a: np.ndarray, rank_b: np.ndarray,
                     line_len: int = 0) -> dict:
    """Spearman/Kendall correlation + normalized rank-displacement stats a vs b.

    Displacement of node i is ``|rank_a[i] - rank_b[i]| / (line_len - 1)`` - the
    fraction of the whole line the node moved. ``line_len`` defaults to the
    vector length; pass it explicitly when comparing a SUBSET of a longer line.
    """
    from scipy import stats
    m = line_len if line_len else rank_a.shape[0]
    rho = stats.spearmanr(rank_a, rank_b).statistic
    tau = stats.kendalltau(rank_a, rank_b).statistic
    disp = np.abs(rank_a.astype(np.float64) - rank_b) / (m - 1)
    q25, q50, q75 = np.percentile(disp, (25, 50, 75))
    return dict(spearman=float(rho), kendall_tau=float(tau),
                disp_median=float(q50), disp_iqr=[float(q25), float(q75)],
                disp_mean=float(disp.mean()),
                frac_moved_gt_threshold=float(
                    (disp > CONFIG["move_threshold_frac"]).mean()),
                move_threshold_frac=CONFIG["move_threshold_frac"])


def per_class_stats(rank_new: np.ndarray, rank_champ: np.ndarray,
                    cell_class_codes: np.ndarray, class_values: list) -> list:
    """Per-cell_class displacement stats (new vs champion-restricted order).

    Returns a list of dicts sorted by median displacement (desc), classes with
    at least CONFIG['min_class_n'] members only. ``median_signed_shift`` > 0
    means the class moved toward the sink (back) end in the central-only order.
    """
    m = rank_new.shape[0]
    disp = (rank_new.astype(np.float64) - rank_champ) / (m - 1)   # signed
    rows = []
    for code in np.unique(cell_class_codes):
        sel = cell_class_codes == code
        n = int(sel.sum())
        if n < CONFIG["min_class_n"]:
            continue
        d = disp[sel]
        rows.append(dict(
            cell_class=class_values[int(code)], n=n,
            median_abs_disp=float(np.median(np.abs(d))),
            iqr_abs_disp=[float(v) for v in np.percentile(np.abs(d), (25, 75))],
            median_signed_shift=float(np.median(d)),
            frac_moved_gt_threshold=float(
                (np.abs(d) > CONFIG["move_threshold_frac"]).mean()),
            median_champ_rank_frac=float(np.median(rank_champ[sel]) / (m - 1)),
            median_new_rank_frac=float(np.median(rank_new[sel]) / (m - 1)),
        ))
    rows.sort(key=lambda r: -r["median_abs_disp"])
    return rows


def make_figures(rank_champ: np.ndarray, rank_new: np.ndarray,
                 class_rows: list) -> None:
    """Save (a) hexbin champion-restricted vs central-only rank, (b) class bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    m = rank_champ.shape[0]
    cmap = LinearSegmentedColormap.from_list("seq_blue", CONFIG["seq_ramp"])

    # -- (a) hexbin: where did every central neuron go? -----------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    hb = ax.hexbin(rank_champ, rank_new, gridsize=90, cmap=cmap, mincnt=1,
                   linewidths=0.1)
    ax.plot([0, m - 1], [0, m - 1], color=CONFIG["accent_color"], lw=1.2,
            label="identity (no change)")
    ax.set_xlabel("rank within full-brain champion (central neurons, compressed)")
    ax.set_ylabel("rank in central-only optimization (run 1)")
    ax.set_title("Central-brain neurons: full-brain order vs central-only order")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.15)
    fig.colorbar(hb, ax=ax, label="neurons per hex")
    fig.tight_layout()
    fig.savefig(_p(CONFIG["fig_scatter"]), dpi=CONFIG["fig_dpi"])
    plt.close(fig)

    # -- (b) per-class median displacement bars -------------------------------
    rows = class_rows[:CONFIG["top_movers"]][::-1]        # biggest at top
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.6))
    names = [f"{r['cell_class']} (n={r['n']})" for r in rows]
    vals = [r["median_abs_disp"] for r in rows]
    ax.barh(names, vals, color=CONFIG["bar_color"], height=0.62)
    ax.axvline(CONFIG["move_threshold_frac"], color=CONFIG["accent_color"],
               lw=1.2, ls="--",
               label=f"move threshold ({CONFIG['move_threshold_frac']:.0%} of line)")
    ax.set_xlabel("median |rank displacement| / line length")
    ax.set_title("Which cell classes move most when the periphery is removed?")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(_p(CONFIG["fig_classes"]), dpi=CONFIG["fig_dpi"])
    plt.close(fig)


def stage_analyze() -> None:
    """Compute all comparison metrics + figures and write the final JSON."""
    from src.mfas.io import load_processed
    from src.mfas.metrics import pct, score_from_order

    gc = load_processed(_p(CONFIG["subgraph_npz"]))
    sub_meta = json.loads(_p(CONFIG["subgraph_meta"]).read_text(encoding="utf-8"))
    src, tgt, w = np.asarray(gc.src), np.asarray(gc.tgt), np.asarray(gc.weight)
    total = gc.total_weight

    rank_champ = np.load(_p(CONFIG["champ_restricted_npy"]))
    runs, ranks = {}, {}
    for k in range(1, CONFIG["n_runs"] + 1):
        runs[k] = json.loads(
            _p(CONFIG["run_json"].format(run_id=k)).read_text(encoding="utf-8"))
        ranks[k] = np.load(_p(CONFIG["run_rank_npy"].format(run_id=k)))

    # Frozen-oracle FF% of every order on the central subgraph.
    score_b = score_from_order(rank_champ, src, tgt, w)
    ff = dict(champ_restricted=dict(score=score_b, pct=pct(score_b, total)))
    for k, r in ranks.items():
        s = score_from_order(r, src, tgt, w)
        assert s == runs[k]["final"]["score"], (
            f"run{k}: saved rank does not reproduce the logged score")
        ff[f"run{k}"] = dict(score=s, pct=pct(s, total))
    gap_pp = {f"run{k}": ff[f"run{k}"]["pct"] - ff["champ_restricted"]["pct"]
              for k in ranks}

    comparisons = {
        "run1_vs_champ": order_comparison(ranks[1], rank_champ),
        "run2_vs_champ": order_comparison(ranks[2], rank_champ),
        "run1_vs_run2_noise_floor": order_comparison(ranks[1], ranks[2]),
    }

    # Per-class breakdown on run 1 (annotations re-gated against the champion).
    g, _order, full_rank, _ffp = load_full_graph_and_champion()
    ann, meta = load_annotations(full_rank)
    central_code = meta["super_class_values"].index(CONFIG["super_class_keep"])
    central_nodes = np.flatnonzero(np.asarray(ann["super_class"]) == central_code)
    assert np.array_equal(np.asarray(g.node_ids)[central_nodes],
                          np.asarray(gc.node_ids)), (
        "cached subgraph node set != current annotation central set")
    cell_codes = np.asarray(ann["cell_class"])[central_nodes]
    class_rows = per_class_stats(ranks[1], rank_champ, cell_codes,
                                 meta["cell_class_values"])
    # How much of the global order change is the CX block alone? Recompute the
    # run1-vs-champ comparison restricted to non-CX central neurons (their ranks
    # on the SHARED line, not re-compressed - displacement keeps its meaning).
    non_cx = cell_codes != meta["cell_class_values"].index("CX")
    comparisons["run1_vs_champ_excluding_CX"] = dict(
        n=int(non_cx.sum()),
        **{k: v for k, v in order_comparison(
            ranks[1][non_cx], rank_champ[non_cx],
            line_len=ranks[1].shape[0]).items()})
    # Same statistic between the two same-seed runs = per-class NOISE floor:
    # a class whose champ-displacement exceeds its run1-vs-run2 displacement
    # genuinely sits elsewhere in the central-only solution.
    class_rows_noise = per_class_stats(ranks[1], ranks[2], cell_codes,
                                       meta["cell_class_values"])

    make_figures(rank_champ, ranks[1], class_rows)

    # -- Full-brain context for interpreting the movers (logged, not estimated) --
    # (i) where each super_class sits in the FULL champion order; (ii) where the
    # out-weight of CX cells and the in-weight of TuBu cells goes/comes from,
    # across the central/non-central boundary. Pure numpy group-bys on g + ann.
    sc = np.asarray(ann["super_class"])
    cc_full = np.asarray(ann["cell_class"])
    fr = np.asarray(full_rank, dtype=np.float64) / (g.n_nodes - 1)
    sc_names = meta["super_class_values"]
    per_super = {sc_names[c]: dict(n=int((sc == c).sum()),
                                   median_rank_frac=float(np.median(fr[sc == c])))
                 for c in np.unique(sc)}

    def boundary_flow(node_mask: np.ndarray, direction: str) -> dict:
        """Weight of ``node_mask``'s out- ('src') or in-edges ('tgt') grouped by
        the OTHER endpoint's super_class, as fractions of the group's total."""
        s_full = np.asarray(g.src, dtype=np.int64)
        t_full = np.asarray(g.tgt, dtype=np.int64)
        w_full = np.asarray(g.weight, dtype=np.int64)
        if direction == "src":
            emask, other = node_mask[s_full], sc[t_full]
        else:
            emask, other = node_mask[t_full], sc[s_full]
        tot = float(w_full[emask].sum())
        out = {}
        for c in np.unique(other[emask]):
            wsum = float(w_full[emask & (other == c)].sum())
            out[sc_names[int(c)]] = round(wsum / tot, 4)
        return dict(total_weight=tot,
                    frac_by_other_super_class=dict(sorted(
                        out.items(), key=lambda kv: -kv[1])))

    cc_names = meta["cell_class_values"]
    central_mask_full = sc == central_code
    cx_mask = central_mask_full & (cc_full == cc_names.index("CX"))
    tubu_mask = central_mask_full & (cc_full == cc_names.index("TuBu"))
    context = dict(
        full_champion_median_rank_frac_by_super_class=per_super,
        cx_out_weight_by_target_super_class=boundary_flow(cx_mask, "src"),
        cx_in_weight_by_source_super_class=boundary_flow(cx_mask, "tgt"),
        tubu_in_weight_by_source_super_class=boundary_flow(tubu_mask, "tgt"),
        tubu_out_weight_by_target_super_class=boundary_flow(tubu_mask, "src"),
    )

    result = dict(
        study="central_only", created_at=now_iso(),
        question=("Is the central brain's internal MFAS order self-determined by "
                  "its own wiring, or dictated by its periphery interfaces?"),
        config=dict(CONFIG),        # tuples serialize as JSON lists
        subgraph=sub_meta,
        ff_pct_central_subgraph=ff,
        gap_central_only_minus_champ_restricted_pp=gap_pp,
        comparisons=comparisons,
        per_class_run1_vs_champ=class_rows,
        per_class_run1_vs_run2_noise=class_rows_noise,
        full_brain_context=context,
        runs={k: {kk: vv for kk, vv in v.items()} for k, v in runs.items()},
        figures=[CONFIG["fig_scatter"], CONFIG["fig_classes"]],
        rerun_commands=[
            "PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python",
            "$PY biology/analysis/exp_central_only.py --stage subgraph",
            "$PY biology/analysis/exp_central_only.py --stage run --run-id 1"
            " 2>&1 | tee biology/results/central_only_run1.log",
            "$PY biology/analysis/exp_central_only.py --stage run --run-id 2"
            " 2>&1 | tee biology/results/central_only_run2.log",
            "$PY biology/analysis/exp_central_only.py --stage analyze",
        ],
        caveats=[
            "MPS kernels are non-deterministic: same-seed repeats differ; the "
            "run1-vs-run2 block is the noise floor for every order comparison.",
            "The pipeline is deterministic-greedy-initialized (seed never drawn), "
            "so 'seed 42' is one sample of device noise, not of an RNG.",
            "Full-brain champion was produced on CUDA; central-only runs are MPS. "
            "FF%% comparisons here are within-machine (frozen oracle, CPU), but "
            "the champion ORDER itself is a CUDA artifact.",
            "cell_class annotation is NA for many central neurons; per-class rows "
            "cover only classes with >= {} members.".format(CONFIG["min_class_n"]),
        ],
    )
    out = _p(CONFIG["final_json"])
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("ff_pct_central_subgraph",
                       "gap_central_only_minus_champ_restricted_pp",
                       "comparisons")}, indent=1))
    print(f"[{now_iso()}] wrote {out}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["subgraph", "smoke", "run", "analyze"])
    ap.add_argument("--run-id", type=int, default=1, choices=[1, 2])
    args = ap.parse_args()

    for d in (CONFIG["results_dir"], CONFIG["outputs_dir"]):
        _p(d).mkdir(parents=True, exist_ok=True)

    if args.stage == "subgraph":
        stage_subgraph()
    elif args.stage == "smoke":
        stage_run(run_id=0, smoke=True)
    elif args.stage == "run":
        stage_run(run_id=args.run_id, smoke=False)
    elif args.stage == "analyze":
        stage_analyze()


if __name__ == "__main__":
    main()
