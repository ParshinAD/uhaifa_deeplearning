# Research Brief — Improving the Rocket MFAS sub-algorithm

## Goal
Improve the **Rocket** sub-algorithm for the **Minimum Feedback Arc Set (MFAS)** problem from
Bader et al. (2025), *"Rocket-Crane algorithm for the Feedback Arc Set problem"*
(Social Network Analysis and Mining 15:68). MFAS on a directed weighted graph = order the
vertices on a line to **maximize the total weight of feedforward edges** (edge `(u,v)` is
feedforward iff `pos(u) < pos(v)`); equivalently, minimize the feedback weight
`total_weight − feedforward_weight`.

This document covers the **infrastructure phase only**: a clean repo, an exact evaluation
oracle, and a reproduced baseline. **No algorithmic improvements are introduced here.**

## Metric
The single ground-truth metric is the **exact discrete feedforward weight** and its percentage
of total edge weight, computed by `src/mfas/metrics.py` (int64/float64 accumulation,
deterministic). The sigmoid surrogate used to *train* Rocket is never used for *reporting*.

## Baseline to beat (per dataset)
| Dataset | Baseline (this repo) | Paper reference |
|---|---|---|
| connectome (FlyWire) | Rocket ≈ **82.9%** feedforward weight (see `experiments/log.md`) | Rocket 80.05% (TTE) / 82.87% plateau; Rocket+Crane 84.60% |
| mouse (own) | first measured here (no prior reference) | — |

The authoritative anchor is the deterministic scorer-parity test: scoring the saved
`results/rocket_best_positions.npy` returns exactly **34,751,902 / 41,912,141 = 82.9161%**.

## Constraints
- **Hardware:** Apple Silicon (MPS) via PyTorch 2.8 in conda env `allen` (Python 3.9). The paper
  used dual A100 GPUs with JAX — wall-clock differs; the **score** is the anchor, not the time.
- **MPS nondeterminism:** training trajectories are not bit-reproducible; reproduction rests on
  the exact scorer + ≥3 seeds (mean ± std), not on identical runs.
- **Crane phase not in scope:** it needs Gurobi (MIP) + ~20 days; 84.60% is not reproducible here.

## Definition of Done (infrastructure phase)
- [x] Clean documented repo; tree in `README.md`.
- [x] Exact scorer with a passing known-answer unit test.
- [x] Ported Rocket matches the notebook (scorer-parity == 34,751,902; CPU-determinism).
- [ ] Baseline reproduced on both datasets, ≥3 seeds, logged with mean ± std; connectome ≈ paper.
- [x] One command reproduces the baseline end-to-end (`scripts/reproduce_baseline.sh`).
- [x] No fabricated numbers; every number traces to a `results/*.json` + command.
- [x] No algorithmic changes.

## Working agreement
I (the researcher) review the output of each phase. In later (improvement) phases you may run
autonomously: propose a hypothesis in `experiments/backlog.md`, run it on **both** datasets across
≥3 seeds through the **unchanged** harness, log to `results/*.json`, and record the ranked
conclusion in `experiments/findings.md`. A change is only a "win" if it beats the baseline beyond
seed noise on the agreed metric.
