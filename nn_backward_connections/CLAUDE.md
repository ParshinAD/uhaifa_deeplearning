# Project: Rocket/MFAS — improving the algorithm & studying feedback in brains

> Note: this directory has its own focus, separate from the parent
> `deeplearning_thesis/CLAUDE.md` (which is about NN initialization & data geometry).
> The instructions in *this* file take precedence for work inside
> `nn_backward_connections/`.

> **⚙ IF THIS CHECKOUT IS AN AUTONOMOUS CAMPAIGN SANDBOX** — i.e. a worktree on a branch named
> `auto/campaign*` — then read **`autoresearch/CAMPAIGN.md` first**. It is the standing brief and
> it overrides ordinary defaults about scope and pace:
> - variants are judged against the **champion** in `autoresearch/sota.json`, not the baseline;
> - every run must finish within **3600 s**;
> - `autoresearch/audit.py` must pass before any champion changes;
> - nothing is ever written outside that worktree, pushed, or merged into another branch.
>
> Current state: `autoresearch/DASHBOARD.md`. One cycle: `/research-cycle`.
> Unattended: `bash autoresearch/driver.sh`.
>
> On `main` those rules do not apply — `main` is the **unified record**, not a running campaign.

> **📍 `main` carries the merged record of two parallel tracks** (merged 2026-08-25): the Phase-7
> autonomous campaign, run on Windows/CUDA, and the Phase-6 surrogate track, run on macOS/MPS.
> `experiments/log.md` is split into "Track 1" and "Track 2" from 2026-08-09 onward for exactly
> that reason. **Scores measured on different devices are not directly comparable** — see
> "Hardware notes" below before quoting a delta across the two.

## Research context

The task is **Maximum Feedforward Arc Set (MFAS)** on the **fly connectome**
(FlyWire MFAS Challenge): order all neurons on a line so that the total weight of
**feedforward** edges is maximized. An edge `(u, v)` is feedforward iff
`position[u] < position[v]`. Everything not feedforward is a **feedback** edge.

Baseline reference is the paper we reproduce:
**Bader et al. (2025). _Rocket-crane algorithm for the Feedback Arc Set problem._**
Social Network Analysis and Mining 15:68. DOI: 10.1007/s13278-025-01491-2
(PDF lives in this folder). The current notebooks already reproduce its **Rocket**
sub-algorithm (see "Current state" below).

## Two research goals

1. **Improve the Rocket algorithm.** Push past the reproduced plateau
   (~82.9% feedforward weight). Ideas to explore: better initialization, optimizer
   / β-schedule variants, momentum/annealing tricks, stronger local search,
   hybrid discrete+continuous refinement, partial Crane-style MIP on subgraphs.
   Always report the **pure Rocket score separately** from any post-processing.

2. **Random graphs vs. feedback in real brains.** Study how the minimum amount of
   **feedback** (= total weight − best feedforward weight) depends on graph
   structure. Compare the real fly connectome (and other real connectomes, e.g.
   `data/table_mouse.txt`) against random graph models matched on size/degree
   (Erdős–Rényi, configuration model, degree-preserving rewiring, etc.).
   Core question: **do real brains have more or less unavoidable feedback than
   random graphs with the same statistics?** What structural features explain it?

## Current state (what already exists)

- `create_notebook.py` → generates `rocket_mfas_reproduction.ipynb` (full study:
  data validation, graph stats, baselines, Rocket, ablations, local search).
- `experiments/` → `create_notebook.py` + `rocket_base.ipynb` (minimal Rocket +
  a **seed-stability analysis**: scores are stable ~82.9% across seeds, but the
  ordering only correlates Spearman ≈ 0.96; **sinks are far more stable than
  sources**, Jaccard@1000 ≈ 0.65–0.74 vs ≈ 0.33–0.40).
- `results/` → CSVs (`all_results.csv`, ablations, `rocket_baseline_history.csv`)
  + `rocket_best_positions.npy` (best position vector found so far).
- `outputs/` → PNG figures.

### Reproduced numbers (dual-A100 paper → our Apple MPS / PyTorch)
| Algorithm | Ours | Paper |
|---|---|---|
| Simple | 50.05% | 50.17% |
| Greedy | 68.07% | 69.25% |
| GreedyAbs | 72.04% | 73.12% |
| **Rocket** | **~82.92%** (74 s) | 80.05% (4.2 s) / 82.87% plateau |
| Rocket + Crane | not reproduced | 84.60% (~20 days, needs Gurobi MIP) |

### Dataset facts (validated, don't re-derive)
- `data/connectome_graph.csv.gz`: columns `Source Node  ID,Target Node ID,Edge Weight`
  (note the **double space** in "Source Node  ID").
- 136,648 vertices, 5,657,719 edges, total edge weight 41,912,141 (max 2,405).
- `data/table_mouse.txt`: `src tgt weight` (float), mouse connectome — not yet used.

## Tech stack
- Python 3.9 (conda env `allen`). The interpreter path is machine-specific — see
  "Run everything via" at the bottom of this file.
- PyTorch (Rocket: continuous positions as a learnable Parameter + Adam)
- NumPy / Pandas, Matplotlib, SciPy (stats), nbformat (notebook generation)
- networkx / igraph are reasonable choices for the random-graph study (confirm
  what's installed before relying on them).

## Hardware notes

**Two machines produced the record on `main`. Know which one a number came from.**

| | MacBook (Apple Silicon) | Windows 10 laptop |
|---|---|---|
| device | MPS | CUDA, NVIDIA RTX 4060 Laptop (8 GB) |
| interpreter | `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python` | `/c/ProgramData/anaconda3/envs/allen/python.exe` |
| torch | 2.8.0 | 2.8.0+cu128 (CUDA 12.8) |
| python | 3.9.23 | 3.9.25 |
| shell | zsh | Git Bash (not WSL) |
| produced | the baseline reproduction, H01–H35, the Q-track diagnostics, H37/H38, the surrogate gate | the whole Phase-7 campaign: P01–P09, H36, H41–H52, `autoresearch/sota.json` |

The champion registry in `autoresearch/sota.json` was measured on **CUDA**. Re-measuring those
champions on MPS is queue item **P01** and it has NOT been done on this side of the merge — until
it is, any delta computed here against a `sota.json` number is a moving comparator in a form
`config_hash` cannot detect, because the device is not part of it.

What travels between machines is the **deterministic scorer**, not a training trajectory:
`tests/test_metrics.py` scoring `results/rocket_best_positions.npy` to exactly 34,751,902.

- `select_device("auto")` prefers MPS → CUDA → CPU, so the same code runs on both.
- **Compute the discrete score on CPU**, not on the accelerator: large-magnitude float32
  index ops on MPS can occasionally return garbage and corrupt best-score tracking.
  (`torch_score` already does this — keep it, it is also what makes the score portable.)
- **The champion pipelines are deterministic on CUDA.** H02/H30/H35 take a `seed` but never
  draw from it: `init_positions` comes from greedy-FAS, so `make_init_positions` — the only
  RNG consumer — is never called. Seeds 42/123/999 produce **bit-identical** position vectors,
  so σ = 0 and the seed-to-seed spread seen on MPS was pure device non-determinism. Consequence:
  a Welch CI computed from these seeds is degenerate (SE = 0 makes any positive delta look
  significant). Use the PROTOCOL CI (`audit.py` floors its σ at `baseline_sigma_pp`) together
  with `screen_delta_pp` as a minimum effect size. Verified in P01 — see `experiments/log.md`.
- A connectome run is ~2.7× slower on the RTX 4060 than on the MacBook (591 s vs 219 s) — the loop is
  gather/scatter over 5.7 M edges with atomic accumulation, i.e. memory-bound, where a 128-bit
  8 GB laptop card has no advantage over unified memory. Still far inside the 3600 s budget.

## Workflow conventions
- **Notebooks are generated from `create_notebook.py`** via nbformat — they are
  build artifacts. Edit the `.py` generator, then regenerate; do not hand-edit the
  `.ipynb` as the source of truth. Regenerate with `$PY create_notebook.py`, where `$PY` is
  this machine's `allen` interpreter (see "Run everything via").
- Use **time limits** (`time_limit=`) on Rocket runs rather than fixed epoch counts
  when comparing — runtime differs a lot from the paper's GPUs.
- Keep the reproduction notebook intact; put new experiments under `experiments/`
  (own generator + notebook) so the baseline study stays clean.

## Code style rules
- All hyperparameters in a `CONFIG` dict at the top — **no magic numbers**.
- Set seeds globally and locally (`GLOBAL_SEED = 42`) for reproducibility.
- Functions need docstrings; reference the paper's algorithm/equation where relevant.
- Plots need titles, axis labels, legends; save to `./outputs/` (or
  `experiments/outputs/`) with descriptive filenames, dpi≈120.
- A single canonical scoring function used everywhere (feedforward weight).
- Save every experiment's results to `./results/` as CSV; persist orderings as `.npy`.

## Honest constraints (already documented, keep stating them)
- Crane phase needs Gurobi (MIP) + ~20 days → 84.60% is not reproducible here.
- Hardware mismatch (PyTorch/MPS vs JAX/XLA on 2×A100): expect different runtimes.
- When a result comes from post-processing/local search, label it as such and keep
  the raw Rocket number separate.

---

## Hard invariants (infrastructure / oracle) — DO NOT VIOLATE

These govern the evaluation harness built under `src/mfas/`, `eval/`, `tests/`.

1. **Never modify the scorer or eval harness once its unit test passes.** `src/mfas/metrics.py`
   and `eval/harness.py` define ground truth. They are frozen after `tests/test_metrics.py` is green.
2. **Always evaluate on BOTH datasets** (`connectome` and `mouse`); report per-dataset.
3. **Every reported metric must come from an actual logged run** (`results/*.json`) with a
   re-runnable command. NEVER fabricate, estimate, or hand-edit a number.
4. **Pin and log seeds.** Use ≥3 seeds for any comparison; report mean ± std. A gain within
   noise (overlapping std) is NOT a gain.
   *Qualified 2026-08-09 (P02), screen stage only.* The rule exists so no verdict rests on a
   single sample of a **stochastic** process. A variant that never draws from its `seed` is not
   such a process: `init_positions` comes from deterministic greedy-FAS, so `make_init_positions`
   is unreachable and seeds 42/123/999 are one sample taken three times, bit-identically. For
   those, and only on the screen, the primaries run 1 seed — decided mechanically by
   `autoresearch/seed_class.py`, which is fail-safe toward "stochastic", and corroborated by the
   3 mouse runs, which stay. **`confirm` still runs ≥5 seeds for every variant**, so no promoted
   number ever rests on one run. See `experiments/PROTOCOL.md § Phase-7.4`.
5. **One git commit per experiment**; results must be reproducible via `git checkout` + the
   logged command.
6. **Never hardcode or peek at the target metric inside any algorithm.** The discrete
   feedforward score is computed only by the oracle, after the algorithm produces positions.
7. **This is a thesis: correctness and reproducibility over iteration speed.**
8. **All repo content in English.**

### Oracle facts (verified)
- Exact scorer accumulates weights in **int64 (connectome) / float64 (mouse)**, never float32.
- Parity anchor: scoring `results/rocket_best_positions.npy` yields **34,751,902** on total
  weight **41,912,141** (= 82.9161%). The notebook's old 34,751,904 / 41,912,104 were float32 noise.
- Discrete score is always computed on **CPU** (MPS float32 index ops can return garbage).
- MPS kernels are not deterministic → exact training-trajectory parity is impossible; the
  authoritative reproduction claim rests on the deterministic **scorer-parity test**.

### Run everything via
The conda env `allen`, whichever machine you are on:

```bash
# MacBook / MPS
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
# Windows laptop / CUDA (Git Bash, not WSL)
PY=/c/ProgramData/anaconda3/envs/allen/python.exe
```

One-command reproduction: `bash scripts/reproduce_baseline.sh`.

**The interpreter path in the older documents is the MacBook one.** Everything written before
2026-08-09 — `findings.md`, `experiments/backlog.md`, `dr_tmp/`, the diagnostics docstrings —
quotes `/opt/homebrew/...`; the Phase-7 campaign entries quote `/c/ProgramData/...`. Those
documents are left as written because they record how those numbers were actually produced.
Substitute your own `$PY` when re-running any of them. See `experiments/log.md` (P01).
