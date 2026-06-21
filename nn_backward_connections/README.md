# MFAS Rocket — Thesis Infrastructure & Oracle

Infrastructure for an MSc thesis (Statistics, University of Haifa) aimed at **improving the
"Rocket" sub-algorithm** for the Minimum Feedback Arc Set (MFAS) problem (Bader et al., 2025).

This repository provides (A) a clean reproducible layout and (B) the **evaluation oracle** — an
exact, deterministic feedforward-weight scorer plus a CLI harness — and reproduces the paper's
Rocket baseline as a sanity anchor. **No algorithmic changes are introduced.**

> Built **in place** inside `nn_backward_connections/`, which is part of the existing
> `uhaifa_deeplearning` git repository. The original reproduction notebook and datasets are
> reused, not duplicated.

## Quick start

```bash
# conda env `allen` (Python 3.9, torch 2.8, Apple MPS)
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python

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
  README.md  CLAUDE.md  RESEARCH_BRIEF.md
  pyproject.toml  requirements.txt  environment.yml  .gitignore
  src/mfas/
    metrics.py            # EXACT feedforward scorer (the oracle core) — frozen after tests pass
    io.py                 # load both datasets -> canonical GraphData (+ node-id remap)
    graph.py              # SCC + degree stats (scipy.csgraph)
    baseline/rocket.py    # behavior-preserving port of the notebook's run_rocket
    utils/{seeding,logging}.py
  eval/
    harness.py            # CLI: run -> score -> JSON record
    aggregate.py          # results/*.json -> mean±std table in experiments/log.md
  configs/baseline_rocket.yaml
  tests/{test_metrics.py, test_rocket.py, conftest.py}
  scripts/reproduce_baseline.sh
  experiments/{log.md, backlog.md, findings.md}
  results/                # per-run JSON + rocket_best_positions.npy (parity anchor)
  data/{raw/, processed/, README.md}   # connectome_graph.csv.gz, table_mouse.txt (reused in place)
  notebooks/reference/    # read-only copy of the original reproduction notebook
  reports/                # thesis-facing figures/tables (later)
```

## Datasets

| | connectome (FlyWire) | mouse |
|---|---|---|
| nodes / edges | 136,648 / 5,657,719 | 148 / 583 |
| weights | integer (synapse counts) | float |
| total weight | 41,912,141 | computed from data |

See `data/README.md` for schema, provenance, and full statistics.

## Environment

conda env `allen`: Python 3.9.23, PyTorch 2.8.0 (Apple MPS), numpy 1.23.5, scipy 1.10.1,
networkx 3.2.1, pandas 1.5.3, pyyaml 6.0.2, pytest. Recreate via `environment.yml` (or
`pip install -r requirements.txt` into a Python 3.9 env). We keep conda (not uv) because the
reproduction numbers were produced there; a different torch/MPS build would not be bit-identical.

## Governance

Hard invariants (frozen scorer, both datasets, ≥3 seeds with mean±std, no fabricated numbers,
one commit per experiment) are in `CLAUDE.md`. Goal, baseline, metric, and Definition of Done
are in `RESEARCH_BRIEF.md`.
