"""Q02 — How far apart are Rocket solutions across seeds (no greedy warm-start)?

TRACK-B DIAGNOSTIC. Answers: Rocket's SCORE is known to be stable across seeds
(~82.9% connectome), but is the ORDERING stable? We run plain Rocket from a RANDOM
N(0,1) init (deliberately NO greedy warm-start — a shared greedy seed would make all
runs artificially similar and hide the basin structure we want to measure) across 5
seeds and quantify how far apart the resulting orderings are.

Three ordering-distance lenses, per dataset:
  * Spearman rank correlation (global order agreement; Pearson on the rank vectors,
    which is exact Spearman because argsort-of-argsort produces tie-free permutations).
  * Kendall-tau (pairwise concordance; scipy's O(n log n) kernel is fast enough to run
    in FULL even on the 136k-node connectome — measured ~0.02 s/pair).
  * Jaccard@k of the FRONT set (k lowest-rank nodes = sources) vs the BACK set (k
    highest-rank nodes = sinks). This front/back split is the headline: the prior
    notebook (experiments/rocket_base.ipynb) found sinks far stabler than sources
    (Jaccard@1000 ~0.65 vs ~0.33). We reproduce and extend it across k.

Score dispersion (mean +/- std of the discrete feedforward %) is reported alongside the
order dispersion so "score stable but order not" can be read off directly.

The exact discrete feedforward % is computed only by the frozen scorer
(mfas.metrics.score_from_positions, on CPU, int64/float64). Rocket optimizes on
MPS/CUDA if available. MPS kernels are non-deterministic, so this script is
reproducible-by-checkout in DISTRIBUTION, not bit-identically — which is itself part
of what Q02 measures.

TRACK-B rules honoured: does NOT read data/best_solution or import mfas.analysis.gap
(no answer key needed); writes NOTHING to results/ (all artifacts -> experiments/outputs/).

Run (env `allen`, ~8 min on Apple MPS: connectome 5x20k ~= 7.5 min, mouse 5x5k ~= 5 s):
    PYTHONPATH=src /opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python \
        experiments/diagnostics/q02_seed_distance.py

Artifacts:
    experiments/outputs/q02_seed_distance.json   (all numbers: per-seed scores,
        per-dataset Spearman/Kendall mean+/-std, Jaccard@k front/back table, config,
        deferred-microns note)
    experiments/outputs/q02_seed_distance.png    (Jaccard@k front vs back grouped bars
        per dataset — the source/sink asymmetry made visual)
"""
from __future__ import annotations

import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from mfas.io import load_dataset
from mfas.baseline.rocket import RocketConfig, run_rocket
from mfas.metrics import score_from_positions, pct

# ── CONFIG (no magic numbers) ───────────────────────────────────────────────────
CONFIG = dict(
    SEEDS        = [42, 123, 999, 7, 31415],          # 5 seeds -> C(5,2)=10 pairs
    # PRIMARY: connectome (20k epochs, mirrors baseline_passthrough). SECONDARY: mouse
    # (5k epochs, sanity). microns is DEFERRED (see DEFERRED_NOTE): 80k x 5 ~= 46 min.
    DATASET_EPOCHS = {"connectome": 20_000, "mouse": 5_000},
    # Jaccard@k front/back radii, per dataset (mouse is tiny: n=148 -> small k).
    JACCARD_K    = {"connectome": [100, 1000, 10000], "mouse": [5, 20, 50]},
    INIT_MODE    = "random",     # plain Rocket, random N(0,1) init — NO greedy warm-start
    KENDALL_FULL = True,         # scipy kendalltau is O(n log n); ~0.02 s even at n=136k
    OUT_DIR      = Path("experiments/outputs"),
    OUT_JSON     = "q02_seed_distance.json",
    OUT_PNG      = "q02_seed_distance.png",
    DEFERRED_NOTE = ("microns DEFERRED: 80k epochs x 5 seeds ~= 46 min is too expensive "
                     "for a diagnostic; run separately if the connectome asymmetry needs "
                     "a second large-connectome confirmation."),
    REPRO_CMD = ("PYTHONPATH=src /opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/"
                 "python experiments/diagnostics/q02_seed_distance.py"),
)


def get_device() -> torch.device:
    """Pick the optimizer device (MPS > CUDA > CPU); discrete scoring stays on CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def git_commit() -> str:
    """Short git commit for provenance (empty string if git is unavailable)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - provenance is best-effort
        return ""


def positions_to_rank(positions: np.ndarray) -> np.ndarray:
    """Integer rank vector: rank[node] = its position on the line (0 = front/source).

    Stable argsort-of-argsort of the continuous positions. The node with the smallest
    position gets rank 0; the largest gets rank n-1. The result is a tie-free
    permutation of [0, n), so Pearson on it equals exact Spearman.
    """
    order = np.argsort(positions, kind="stable")          # nodes sorted by position
    rank = np.empty(order.shape[0], dtype=np.int64)
    rank[order] = np.arange(order.shape[0], dtype=np.int64)
    return rank


def run_seed(dataset: str, g, seed: int, epochs: int, device) -> dict:
    """Run plain Rocket (random init) once and return its discrete % + rank vector."""
    cfg = RocketConfig(epochs=epochs, init_mode=CONFIG["INIT_MODE"])
    res = run_rocket(g, cfg, seed=seed, device=device, init_positions=None)
    rank = positions_to_rank(res.best_positions)
    # Re-score through the frozen oracle from the returned positions for a clean,
    # audit-independent number (identical to res.best_pct; belt-and-braces).
    score = score_from_positions(np.asarray(res.best_positions, dtype=np.float64),
                                 np.asarray(g.src), np.asarray(g.tgt), g.weight)
    return dict(seed=seed, pct=pct(score, g.total_weight),
                score=float(score), rank=rank, wall_clock_s=res.wall_clock_s,
                n_epochs_done=res.n_epochs_done)


def pairwise_spearman(ranks: list[np.ndarray]) -> tuple[float, float, list[float]]:
    """Mean/std of pairwise Spearman over all seed pairs (Pearson on rank vectors)."""
    vals = [float(np.corrcoef(ranks[i], ranks[j])[0, 1])
            for i, j in combinations(range(len(ranks)), 2)]
    return float(np.mean(vals)), float(np.std(vals)), vals


def pairwise_kendall(ranks: list[np.ndarray]) -> tuple[float, float, list[float]]:
    """Mean/std of pairwise Kendall-tau over all seed pairs (full, O(n log n))."""
    vals = [float(kendalltau(ranks[i], ranks[j]).correlation)
            for i, j in combinations(range(len(ranks)), 2)]
    return float(np.mean(vals)), float(np.std(vals)), vals


def _jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Jaccard overlap of two equal-size boolean node masks."""
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return inter / union if union else 0.0


def jaccard_front_back(ranks: list[np.ndarray], k: int, n: int
                       ) -> tuple[dict, dict]:
    """Mean/std pairwise Jaccard of the FRONT (k lowest-rank) and BACK (k highest) sets.

    front = sources (rank < k); back = sinks (rank >= n-k). Returns two dicts each with
    mean/std over the C(seeds,2) pairs, so the source-vs-sink asymmetry is directly
    comparable at each k.
    """
    fronts = [r < k for r in ranks]
    backs = [r >= (n - k) for r in ranks]
    fv = [_jaccard(fronts[i], fronts[j]) for i, j in combinations(range(len(ranks)), 2)]
    bv = [_jaccard(backs[i], backs[j]) for i, j in combinations(range(len(ranks)), 2)]
    return (dict(mean=float(np.mean(fv)), std=float(np.std(fv)), pairs=fv),
            dict(mean=float(np.mean(bv)), std=float(np.std(bv)), pairs=bv))


def analyse_dataset(dataset: str, epochs: int, device) -> dict:
    """Run all seeds on one dataset and compute score + ordering dispersion."""
    print(f"\n=== {dataset}  (epochs={epochs}, seeds={CONFIG['SEEDS']}) ===")
    g = load_dataset(dataset)
    n = g.n_nodes
    runs = []
    for seed in CONFIG["SEEDS"]:
        r = run_seed(dataset, g, seed, epochs, device)
        print(f"  seed={seed:>6}  pct={r['pct']:.4f}%  "
              f"wall={r['wall_clock_s']:.1f}s  epochs={r['n_epochs_done']}")
        runs.append(r)

    pcts = [r["pct"] for r in runs]
    ranks = [r["rank"] for r in runs]

    sp_mean, sp_std, sp_pairs = pairwise_spearman(ranks)
    kt_mean, kt_std, kt_pairs = pairwise_kendall(ranks) if CONFIG["KENDALL_FULL"] \
        else (None, None, None)

    jaccard = {}
    for k in CONFIG["JACCARD_K"][dataset]:
        front, back = jaccard_front_back(ranks, k, n)
        jaccard[str(k)] = dict(front=front, back=back)
        print(f"  Jaccard@{k:<6} FRONT(sources)={front['mean']:.3f}"
              f"+/-{front['std']:.3f}   BACK(sinks)={back['mean']:.3f}"
              f"+/-{back['std']:.3f}")

    print(f"  score:   {np.mean(pcts):.4f} +/- {np.std(pcts):.4f} %")
    print(f"  Spearman:{sp_mean:.5f} +/- {sp_std:.5f}   "
          f"Kendall:{kt_mean if kt_mean is None else f'{kt_mean:.5f}'} "
          f"+/- {kt_std if kt_std is None else f'{kt_std:.5f}'}")

    return dict(
        dataset=dataset, n_nodes=n, n_edges=g.n_edges, epochs=epochs,
        total_weight=g.total_weight,
        per_seed=[dict(seed=r["seed"], pct=r["pct"], score=r["score"],
                       wall_clock_s=r["wall_clock_s"], n_epochs_done=r["n_epochs_done"])
                  for r in runs],
        score_pct=dict(values=pcts, mean=float(np.mean(pcts)), std=float(np.std(pcts))),
        spearman=dict(mean=sp_mean, std=sp_std, pairs=sp_pairs),
        kendall=(dict(mean=kt_mean, std=kt_std, pairs=kt_pairs, method="full_scipy_kendalltau")
                 if CONFIG["KENDALL_FULL"] else dict(note="skipped")),
        jaccard=jaccard,
    )


def make_plot(results: dict, out_png: Path) -> None:
    """Grouped bars: Jaccard@k FRONT (sources) vs BACK (sinks), one panel per dataset."""
    datasets = list(results.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5.2),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        r = results[ds]
        ks = sorted((int(k) for k in r["jaccard"]))
        x = np.arange(len(ks))
        width = 0.38
        front_m = [r["jaccard"][str(k)]["front"]["mean"] for k in ks]
        front_s = [r["jaccard"][str(k)]["front"]["std"] for k in ks]
        back_m = [r["jaccard"][str(k)]["back"]["mean"] for k in ks]
        back_s = [r["jaccard"][str(k)]["back"]["std"] for k in ks]
        ax.bar(x - width / 2, front_m, width, yerr=front_s, capsize=4,
               color="tab:orange", label="FRONT set (sources)")
        ax.bar(x + width / 2, back_m, width, yerr=back_s, capsize=4,
               color="tab:blue", label="BACK set (sinks)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_ylim(0, 1)
        ax.set_ylabel("mean pairwise Jaccard overlap")
        ax.set_xlabel("set radius k")
        sp = r["spearman"]
        ax.set_title(f"{ds}  (n={r['n_nodes']:,})\n"
                     f"score {r['score_pct']['mean']:.4f}+/-{r['score_pct']['std']:.4f}%   "
                     f"Spearman {sp['mean']:.3f}")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Q02 — Rocket seed-to-seed ordering distance (random init, no greedy)\n"
                 "sinks (back) reproduce far more stably than sources (front)",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"saved {out_png}")


def main() -> None:
    out_dir = CONFIG["OUT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device={device}  seeds={CONFIG['SEEDS']}")

    results = {ds: analyse_dataset(ds, epochs, device)
               for ds, epochs in CONFIG["DATASET_EPOCHS"].items()}

    payload = dict(
        question="Q02 — how far apart are Rocket solutions across seeds (no greedy warm-start)?",
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in CONFIG.items()},
        device=str(device),
        git_commit=git_commit(),
        mps_nondeterminism_note=(
            "Rocket optimizes on MPS/CUDA (non-deterministic kernels); discrete scores "
            "computed on CPU (float64) by the frozen scorer. Reproducible-by-checkout in "
            "DISTRIBUTION, not bit-identically — the seed-to-seed spread is Q02's subject."),
        deferred=CONFIG["DEFERRED_NOTE"],
        reproduce_command=CONFIG["REPRO_CMD"],
        results=results,
    )
    out_json = out_dir / CONFIG["OUT_JSON"]
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out_json}")
    make_plot(results, out_dir / CONFIG["OUT_PNG"])


if __name__ == "__main__":
    main()
