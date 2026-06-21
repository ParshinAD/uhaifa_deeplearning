# Project: Rocket/MFAS — improving the algorithm & studying feedback in brains

> Note: this directory has its own focus, separate from the parent
> `deeplearning_thesis/CLAUDE.md` (which is about NN initialization & data geometry).
> The instructions in *this* file take precedence for work inside
> `nn_backward_connections/`.

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
- Python 3.9 (conda env `allen`:
  `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`)
- PyTorch (Rocket: continuous positions as a learnable Parameter + Adam)
- NumPy / Pandas, Matplotlib, SciPy (stats), nbformat (notebook generation)
- networkx / igraph are reasonable choices for the random-graph study (confirm
  what's installed before relying on them).

## Hardware notes
- Run on **Apple MPS** (`torch.backends.mps`) when available, else CUDA, else CPU.
- **Compute the discrete score on CPU**, not MPS: large-magnitude float32 index
  ops on MPS can occasionally return garbage and corrupt best-score tracking.
  (`torch_score` already does this — keep it.)

## Workflow conventions
- **Notebooks are generated from `create_notebook.py`** via nbformat — they are
  build artifacts. Edit the `.py` generator, then regenerate; do not hand-edit the
  `.ipynb` as the source of truth. Regenerate with:
  `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python create_notebook.py`
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
`/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python` (conda env `allen`).
One-command reproduction: `bash scripts/reproduce_baseline.sh`.
