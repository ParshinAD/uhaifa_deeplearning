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
_Generated 2026-06-21T03:25:48Z from 18 run(s)._

| algo | dataset | n_seeds | pct mean±std | score mean±std | wall_clock_s (mean) | seeds | config_hash | git_commit |
|---|---|---|---|---|---|---|---|---|
| baseline_passthrough | connectome | 6 | 82.8958 ± 0.0167 | 34,743,388 ± 6,980 | 1199.5 | [42, 42, 123, 123, 999, 999] | f8cb3c | 8f0e5066211 |
| baseline_passthrough | mouse | 6 | 92.0696 ± 0.2347 | 8.4327 ± 0.0215 | 151.6 | [42, 42, 123, 123, 999, 999] | 7b7cba | 8f0e5066211 |
| baseline_rocket | connectome | 3 | 82.8958 ± 0.0189 | 34,743,386 ± 7,927 | 75.9 | [42, 123, 999] | 5ec3ce | 7a77547dee9 |
| baseline_rocket | mouse | 3 | 92.0696 ± 0.2624 | 8.4327 ± 0.0240 | 1.9 | [42, 123, 999] | 5cf05a | 7a77547dee9 |

<!-- END AGGREGATED RESULTS -->

---

## 2026-06-21 — baseline_passthrough: dry run (pipeline wiring smoke test)
- Hypothesis: none — the UNCHANGED baseline Rocket wrapped as a variant
  (`src/mfas/experiments/baseline_passthrough.py`). Purpose: prove the research-loop wiring
  (`eval/run_variant.py` → screen → confirm → log) reproduces the frozen baseline within the
  noise floor. Not a research hypothesis.

#### Implementer (screen)
`python -m eval.run_variant --exp baseline_passthrough --dataset {connectome,mouse}
--seed {42,123,999} --out results/ --role implement` (full epochs: connectome 20k, mouse 5k).
- connectome: 82.8958% (n=3)  Δ vs baseline = **+0.0000 pp**  → screen improvement = NO (expected; 2σ=0.04)
- mouse:      92.0696% (n=3)  Δ vs baseline = **+0.0000 pp**  → screen improvement = NO (expected; 2σ=0.52)
- Frozen baseline (baseline_rocket): connectome 82.8958 ± 0.0189, mouse 92.0696 ± 0.2624.

#### Verifier (confirm)
Independent re-run, `--role verify`, standard seeds (screen reproduction):
- connectome: 82.8957% (n=3)  Δ = −0.0001 pp  → within noise.
- mouse:      92.0696% (n=3)  Δ = +0.0000 pp  → within noise.
- CONFIRM stage not triggered (no screen pass — correct: a passthrough must NOT register as an
  improvement). Numbers match the frozen baseline to 4 decimals.

#### Critic verdict
- Frozen-file integrity: PASS — `git diff` shows none of the 4 frozen files changed; guardrail
  hook tested (blocks Edit/Write to frozen files via exit 2).
- Metric leakage: PASS — variant imports only `baseline.rocket`; never imports/peeks at the
  discrete metric; `eval/run_variant.py` scores externally with the frozen oracle.
- Reproducibility: PASS — all 12 numbers trace to `results/*baseline_passthrough*.json`
  (roles implement/verify) with re-runnable commands.
- Significance: PASS — Δ within noise on both datasets (the intended result).
- Robustness: PASS — holds on BOTH datasets.
- Recommendation: machinery validated (not a candidate win — it IS the baseline).

#### Decision: kill (no improvement — expected). **Pipeline wiring CONFIRMED: the new research
loop reproduces the frozen baseline within the noise floor on both datasets.**

> Note: some `wall_clock_s` values in the auto table are inflated because these runs executed as
> a throttled background batch; `pct`/`score` numbers are unaffected.
