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
_Generated 2026-06-21T08:52:13Z from 30 run(s)._

| algo | dataset | n_seeds | pct mean±std | score mean±std | wall_clock_s (mean) | seeds | config_hash | git_commit |
|---|---|---|---|---|---|---|---|---|
| H01 | connectome | 3 | 82.0510 ± 0.0290 | 34,389,329 ± 12,134 | 78.7 | [42, 123, 999] | fdb0b2 | 704221ab778 |
| H01 | mouse | 3 | 92.1625 ± 0.0242 | 8.4412 ± 0.0022 | 2.0 | [42, 123, 999] | 434027 | 704221ab778 |
| baseline_multistart | connectome | 3 | 82.0507 ± 0.0293 | 34,389,211 ± 12,284 | 77.2 | [42, 123, 999] | 330c58 | 704221ab778 |
| baseline_multistart | mouse | 3 | 92.1625 ± 0.0242 | 8.4412 ± 0.0022 | 1.9 | [42, 123, 999] | c6938b | 704221ab778 |
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

---

## 2026-06-21 — Cleanup: discard contaminated H01 (subagent-registration bug)
- **Issue:** A prior session ran H01 while the custom subagents were not registered, so the loop
  silently fell back to the **general-purpose** agent (full tool access). This collapses the
  integrity design (read-only verifier; implementer ≠ verifier), so any H01 output from it is
  **UNVERIFIED** and must not be trusted or promoted.
- **Discarded** (partial, fallback-produced, `git_commit = …+dirty`, only 1 of 6 screen runs):
  - `results/20260621T082219Z-H01-mouse-s42-implement-434027.json`
  - `results/20260621T082219Z-H01-mouse-s42-implement-434027_positions.npy`
  (For the record, that run reported mouse 92.135% — within the 0.52 pp mouse noise floor — but it
  is discarded on **provenance** grounds regardless of its value.)
- **Retained:** `src/mfas/experiments/H01.py` — reviewed and correct (K=4 restarts from distinct
  sub-seeds `seed + j*7919`; best-of-K by the existing oracle tracker → leakage-safe;
  `n_epochs_done` = total steps → compute-matched to `baseline_multistart` K=4). It will be re-run
  **from scratch** through the real subagents (implementer → verifier → critic) so every H01 number
  is produced under the intended tool scopes. The comparator `baseline_multistart` has no logged
  runs yet and will be generated during the clean H01 cycle.
- **Pre-flight:** `tests/test_metrics.py` green (8 passed); `eval/frozen.sha256` matches all four
  frozen files. Infra fixes that restored correct subagent registration (`.claude/settings.json`,
  `.gitignore`) committed separately as a cleanup commit (not an experiment).

---

## 2026-06-21 — H01: multi-start best-of-K restarts (clean re-run)
- Hypothesis: running K=4 independent short Rocket optimizations from different random inits and
  keeping the best discrete-scoring result beats one long run at equal total compute. (Restart
  variant → per PROTOCOL §Compute-matched, screened against `baseline_multistart` at K=4, NOT the
  global passthrough table.)
- Pre-flight: `eval/frozen_guard.verify_frozen_manifest()` → OK (gated by `run_variant` on every run).

#### Implementer (screen)
Compute-matched comparator = `baseline_multistart` (plain baseline run as K=4 naive restarts ×
total/K epochs, best-of-K), at matched `total_grad_steps` (connectome 20000, mouse 5000).

- **connectome** (n=3, seeds 42/123/999):
  - H01: **82.0510 ± 0.0290** pct
  - baseline_multistart K=4: **82.0507 ± 0.0293** pct
  - Δmean (H01 − comparator) = **+0.0003 pp**;  matched `total_grad_steps = 20000` (both)
  - screen gate (Δ > 0.04 pp): **FAIL**
- **mouse** (n=3, seeds 42/123/999):
  - H01: **92.1625 ± 0.0242** pct
  - baseline_multistart K=4: **92.1625 ± 0.0242** pct
  - Δmean (H01 − comparator) = **+0.0000 pp**;  matched `total_grad_steps = 5000` (both)
  - screen gate (Δ > 0.52 pp): **FAIL**

- **SCREEN verdict: FAIL on both datasets.** Note: H01 and the equal-budget comparator are
  algorithmically identical here — both draw the same distinct sub-seeds (`seed + j*7919`) and call
  the unchanged `run_rocket` for total/K epochs with best-of-K. So H01's "diverse-init restart
  strategy" reduces to plain naive restarts, and the comparator (designed to isolate strategy from
  mere sampling) correctly shows ~zero gain (mouse identical to 4 dp). Both restart schemes also
  sit *below* the single-long-run baseline (connectome 82.896, mouse 92.070) on connectome
  (−0.84 pp): splitting the budget into 4×(total/K) epochs leaves each restart far short of the
  plateau, and best-of-4 short runs cannot recover it. Honest non-improvement.

- Commands (role=implement; via the non-frozen `eval/run_variant.py`, frozen oracle scores):
  - `python -m eval.run_variant --exp H01 --dataset {connectome,mouse} --seed {42,123,999} --out results/ --role implement`
  - `MFAS_MULTISTART_K=4 python -m eval.run_variant --exp baseline_multistart --dataset {connectome,mouse} --seed {42,123,999} --out results/ --role implement`
- Result file exp_ids:
  - H01 connectome: `20260621T084204Z-H01-connectome-s42-implement-fdb0b2`,
    `20260621T084331Z-H01-connectome-s123-implement-fdb0b2`,
    `20260621T084457Z-H01-connectome-s999-implement-fdb0b2`
  - H01 mouse: `20260621T084155Z-H01-mouse-s42-implement-434027`,
    `20260621T084158Z-H01-mouse-s123-implement-434027`,
    `20260621T084201Z-H01-mouse-s999-implement-434027`
  - baseline_multistart connectome: `20260621T084704Z-baseline_multistart-connectome-s42-implement-330c58`,
    `20260621T084830Z-baseline_multistart-connectome-s123-implement-330c58`,
    `20260621T084957Z-baseline_multistart-connectome-s999-implement-330c58`
  - baseline_multistart mouse: `20260621T084655Z-baseline_multistart-mouse-s42-implement-c6938b`,
    `20260621T084658Z-baseline_multistart-mouse-s123-implement-c6938b`,
    `20260621T084701Z-baseline_multistart-mouse-s999-implement-c6938b`

#### Verifier (confirm)
_(not run — screen FAILED on both datasets, so the CONFIRM stage is not triggered, per PROTOCOL.)_

#### Critic verdict
_(full critic red-team is reserved for CONFIRMED findings; not run on a screen-fail kill. Orchestrator
integrity spot-check at decision time: frozen manifest re-verified — all 4 frozen files match
`eval/frozen.sha256` (`FROZEN OK`); `git status` shows no frozen file modified; H01 imports only
`baseline.rocket` and tracks best-by-oracle exactly as the baseline does → no metric leakage; all 12
screen numbers trace to logged `results/*.json` at matched `total_grad_steps`.)_

#### Decision: **kill.** H01 (multi-start best-of-K=4 at equal compute) does NOT beat one long run —
it scores **−0.84 pp** below the single-run baseline on connectome (82.05 vs 82.896) and ties its
equal-budget comparator to 4 dp on mouse (Δ = +0.0000 pp) and within noise on connectome (Δ = +0.0003
pp ≪ 0.04 pp gate). Root cause: splitting the budget into K short runs leaves each restart short of
the plateau, and best-of-K cannot recover the loss; the "diverse-init" strategy is also algorithmically
identical to naive restarts. Not salvageable by retuning K (lower K → approaches baseline from below;
higher K → worse). Hypothesis falsified. Backlog status → killed.
