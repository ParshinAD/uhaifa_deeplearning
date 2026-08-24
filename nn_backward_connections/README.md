# MFAS Rocket — Thesis Infrastructure & Oracle

Infrastructure for an MSc thesis (Statistics, University of Haifa) aimed at **improving the
"Rocket" sub-algorithm** for the Minimum Feedback Arc Set (MFAS) problem (Bader et al., 2025).

This repository provides (A) a clean reproducible layout and (B) the **evaluation oracle** — an
exact, deterministic feedforward-weight scorer plus a CLI harness — and reproduces the paper's
Rocket baseline as a sanity anchor.

> **Where the project stands.** The paragraphs below describe the Phase-0 infrastructure, which
> deliberately introduced no algorithmic change. That is history: the algorithm has since moved
> from the reproduced **82.90%** to **84.15%** on the fly connectome against a reference solution
> of 84.61%. For the current picture read **`reports/STATUS.md`** (10 min, thesis-facing) or
> **`WIKI.md`** (5 min, plain language); for the ranked conclusions, `experiments/findings.md`.

> Built **in place** inside `nn_backward_connections/`, which is part of the existing
> `uhaifa_deeplearning` git repository. The original reproduction notebook and datasets are
> reused, not duplicated.

## Quick start

```bash
# conda env `allen` (Python 3.9, torch 2.8). Pick your machine's interpreter:
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python   # macOS / MPS
# PY=/c/ProgramData/anaconda3/envs/allen/python.exe              # Windows / CUDA (Git Bash)

# 1. Run the oracle gate (exact parity + known-answer; ~10s, no training)
$PY -m pytest tests/test_metrics.py -v

# 2. Reproduce the full baseline (both datasets x 3 seeds) + aggregate to experiments/log.md
bash scripts/reproduce_baseline.sh            # add --quick for a 30s/run smoke test

# 3. A single run
$PY -m eval.harness --algo baseline_rocket --dataset connectome --seed 42 --out results/
```

## The oracle

- **`src/mfas/metrics.py`** — the canonical scorer. An edge `(u,v)` is feedforward iff
  `pos[u] < pos[v]`; the score is the total feedforward weight. Weights are summed in
  **int64 / float64** (never float32) so the result is exact and deterministic.
- **Parity anchor:** scoring the saved `results/rocket_best_positions.npy` returns exactly
  **34,751,902** on total weight **41,912,141** (= **82.9161%**). This is asserted in
  `tests/test_metrics.py::test_scorer_parity_connectome` and is the authoritative reproduction
  claim (MPS makes bit-identical *training* impossible).
- **`eval/harness.py`** — runs an algorithm, extracts positions, re-scores them with the exact
  scorer, and writes a fixed-schema JSON record (`exp_id, algo, dataset, seed, score, pct,
  wall_clock_s, git_commit, env, config_hash, timestamp, …`) to `results/`.

## Repository layout

```
nn_backward_connections/
  README.md  CLAUDE.md  RESEARCH_BRIEF.md  WIKI.md
  pyproject.toml  requirements.txt  environment.yml  .gitignore
  src/mfas/
    metrics.py            # EXACT feedforward scorer (the oracle core) — FROZEN
    io.py                 # load all datasets -> canonical GraphData (+ node-id remap)
    graph.py              # SCC + degree stats (scipy.csgraph)
    baseline/             # rocket.py (port of the notebook's run_rocket), ratio_greedy.py
    refine/               # the DISCRETE move classes — where 97% of the gain came from:
                          #   insertion.py (full-range exact-gain sift), underrelax.py,
                          #   scc_recursive.py (block refinement), segment.py,
                          #   pair_relocate.py, lns.py
    experiments/          # one isolated module per hypothesis: H01..H52, A_INIT, baselines
    analysis/             # gap.py (gated read of the reference solution), surrogate_gate.py
    randomgraph/          # Track C scaffolding (generators.py)
    utils/{seeding,logging}.py
  eval/
    harness.py            # CLI: run -> score -> JSON record        [FROZEN]
    run_variant.py        # the variant runner — the ONLY writer of results/*.json
    aggregate.py          # results/*.json -> mean±std table in experiments/log.md  [FROZEN]
    frozen_guard.py  runtime_guard.py
  autoresearch/           # the Phase-7 autonomous campaign: CAMPAIGN.md (constitution),
                          # campaign.yaml, driver.sh, audit.py, sota/queue/killed registries,
                          # DASHBOARD.md, lit/ (literature notes)
  configs/baseline_rocket.yaml
  tests/                  # 344 tests; test_metrics.py holds the parity anchor
  scripts/reproduce_baseline.sh
  experiments/            # PROTOCOL.md (governance) + the record: log.md, findings.md,
                          # backlog.md, diagnosis.md, questions.md, roadmap.md,
                          # randomgraph.md, TODO_STATUS.md, diagnostics/, evidence/
  results/                # per-run JSON + rocket_best_positions.npy (parity anchor)
  data/{raw/, processed/, best_solution/, README.md}
  notebooks/              # walkthrough notebooks + their nbformat generators
  reports/                # STATUS.md (the 10-minute cross-track status), methods_walkthrough.md
```

## Datasets

Three datasets. Both large ones are **primary** — a result on one graph is not a result.

| | connectome (FlyWire) | microns (mouse cortex) | mouse |
|---|---|---|---|
| role | primary | primary | supporting only |
| nodes / edges | 136,648 / 5,657,719 | 67,534 / 10,436,569 | 148 / 583 |
| weights | integer (synapse counts) | integer (synapse counts) | float |
| total weight | 41,912,141 | 15,400,557 | computed from data |
| seed noise floor | 0.0189 pp | 0.0006 pp | 0.2624 pp |

`data/processed/microns.npz` is **gitignored** (55 MB, rebuildable — see
`data/processed/microns_BUILD.md`); it must be copied by hand onto a new machine. Its statistics
are tracked in `microns_stats.json`. See `data/README.md` for schema and provenance.

## Environment

conda env `allen`: Python 3.9, PyTorch 2.8.0, numpy 1.23.5, scipy 1.10.1, networkx 3.2.1,
pandas 1.5.3, pyyaml 6.0.2, pytest. Recreate via `environment.yml` (or
`pip install -r requirements.txt` into a Python 3.9 env). We keep conda (not uv) because the
reproduction numbers were produced there; a different torch build would not be bit-identical.

The record spans **two machines** — a MacBook on Apple MPS and a Windows laptop on CUDA
(RTX 4060). Neither device reproduces the other's *training trajectory*; what is portable is the
deterministic scorer-parity test. `CLAUDE.md` (Hardware notes) says which results came from which,
and porting instructions are in `autoresearch/PORTING.md`.

## Governance

Hard invariants (frozen scorer, both datasets, ≥3 seeds with mean±std, no fabricated numbers,
one commit per experiment) are in `CLAUDE.md`. Goal, baseline, metric, and Definition of Done
are in `RESEARCH_BRIEF.md`.
