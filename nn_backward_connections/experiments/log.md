# Experiment Log

Append-only lab notebook. Each entry: date, hypothesis, command, result (traced to
`results/*.json`), conclusion. The auto-generated aggregate block below is rewritten by
`python -m eval.aggregate`; manual notes outside the markers are preserved.

---

## Baseline reproduction (infrastructure phase)

- **Setup:** ported Rocket (`src/mfas/baseline/rocket.py`), exact scorer (`src/mfas/metrics.py`),
  conda env `allen` (Python 3.9.23, torch 2.8.0, Apple MPS).
- **Command:** `bash scripts/reproduce_baseline.sh`
- **Anchor (deterministic):** scoring `results/rocket_best_positions.npy` →
  **34,751,902 / 41,912,141 = 82.9161%** (`tests/test_metrics.py::test_scorer_parity_connectome`).

<!-- The table below is auto-generated; do not edit by hand. -->

<!-- BEGIN AGGREGATED RESULTS (auto-generated) -->
_Generated 2026-06-21T00:21:01Z from 6 run(s)._

| algo | dataset | n_seeds | pct mean±std | score mean±std | wall_clock_s (mean) | seeds | config_hash | git_commit |
|---|---|---|---|---|---|---|---|---|
| baseline_rocket | connectome | 3 | 82.8958 ± 0.0189 | 34,743,386 ± 7,927 | 75.9 | [42, 123, 999] | 5ec3ce | 7a77547dee9 |
| baseline_rocket | mouse | 3 | 92.0696 ± 0.2624 | 8.4327 ± 0.0240 | 1.9 | [42, 123, 999] | 5cf05a | 7a77547dee9 |

<!-- END AGGREGATED RESULTS -->
