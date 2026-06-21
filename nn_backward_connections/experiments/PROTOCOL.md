# Research-Loop Protocol

How the autonomous campaign to improve the **Rocket** MFAS sub-algorithm runs. Subagents
coordinate **purely via the filesystem** (the files in the contracts below). The exact scorer
and harness are **FROZEN** and define ground truth; nothing here may change them.

## The loop
```
plan
  → pick top-ranked backlog item (experiments/backlog.md)
  → implementer   : build isolated variant + run (both datasets ×3 seeds) + SCREEN
  → verifier      : independent re-run; SCREEN then CONFIRM (more seeds, CI test)
  → critic        : red-team (leakage / overfit / noise / reproducibility / frozen files)
  → log decision  : keep / kill / iterate  (experiments/log.md)
  → re-prioritize backlog
  → repeat until budget or stop-criteria
```
One full cycle = one invocation of `/run-experiment <backlog-id>`.

## Definition of a CONFIRMED improvement (two-stage)
A variant is a real win ONLY if it passes BOTH stages, on BOTH datasets (connectome AND mouse).

**Stage 1 — SCREEN** (cheap gate, standard loop seeds = 3: 42/123/999):
- Promote iff `Δmean > 2 × std` on **both** datasets, where `std` is the per-dataset baseline
  noise floor and `Δmean = mean(variant) − mean(baseline)`.
- Thresholds: **connectome Δ > 0.04 pp**, **mouse Δ > 0.52 pp**.
- This is a gate, not a verdict. Failing the screen → kill (or iterate).

**Stage 2 — CONFIRM** (only for screened variants; more seeds):
- Reseed: **connectome = 5 seeds** (42,123,999,7,31415), **mouse = 20–30 seeds**.
- Difference of means `Δ = mean_variant − mean_baseline`; standard error of the difference
  `SE = std · sqrt(2/n)` (per dataset, n = seeds/group); 95% CI lower bound `= Δ − 1.96 · SE`.
- **CONFIRMED iff the 95% CI lower bound > 0 on BOTH datasets.** Because the bar is on the
  **standard error of the mean**, more seeds tighten it — a small true effect needs more seeds.
- Record **both** the screen and the confirm outcomes in `experiments/log.md`.

> Rationale: the screen is a fast, lenient filter to avoid wasting confirm-compute; the confirm
> stage is the statistically defensible verdict and is what may be promoted to a thesis finding.

## Noise floor (significance reference)
From the frozen baseline, 3 seeds (42/123/999):
| dataset | baseline mean (pct) | std (σ) | 2σ screen threshold |
|---|---|---|---|
| connectome | 82.8958 | **0.0189 pp** | 0.0378 pp (use 0.04) |
| mouse | 92.0696 | **0.2624 pp** | 0.5248 pp (use 0.52) |
Mouse is ~14× noisier than connectome — hence the larger confirm seed count for mouse.

## Budget & stop-criteria  (TODO — set these before launching the campaign)
- `MAX_EXPERIMENTS` = **TODO**  (hard cap on cycles)
- `MAX_WALLCLOCK` / `MAX_COMPUTE` = **TODO**  (e.g. hours of MPS time)
- `EARLY_EXIT_K` = **TODO**  (stop after K consecutive non-improving cycles)
- `CONFIRM_SEEDS_MOUSE` = **TODO in [20,30]**  (pick a value; connectome fixed at 5)

## File contracts (who writes what)
Single-writer per file; agents run sequentially within a cycle, so appends do not clobber.

| file | written by | content |
|---|---|---|
| `experiments/backlog.md` | **ideator** (create/rank), **implementer** (flip status) | ranked hypotheses; `status: proposed → screened → confirmed/killed/iterate` |
| `src/mfas/experiments/<id>.py` | **implementer** | the isolated variant (`ID`, `HYPOTHESIS`, `run(g, seed, device, time_limit=None) -> RocketResult`) |
| `results/<run_id>.json` | **run_variant.py** (invoked by implementer & verifier) | per-run record (schema below) |
| `experiments/log.md` (narrative) | **implementer** (Hypothesis+Screen), **verifier** numbers (via orchestrator), **critic** (verdict block) | one append-only section per cycle |
| `experiments/log.md` (auto numbers block) | **frozen `eval/aggregate.py`** | mean±std table between its `<!-- … AGGREGATED RESULTS -->` markers |
| `experiments/findings.md` | orchestrator at decision time | ranked, evidence-backed confirmed wins |

### `results/<run_id>.json` schema
Identical to the baseline harness record, plus variant provenance:
```
exp_id, algo(=variant id), dataset, seed, score, pct, wall_clock_s, git_commit,
env{python, jax_or_torch, cuda, gpu}, config_hash, timestamp,
total_weight, n_epochs_done, best_positions_path,
experiment_id, role(implement|verify|confirm), hypothesis
```
Produced ONLY by `eval/run_variant.py` (frozen oracle scores the positions). `config_hash`
excludes seed/device so seeds of one (variant, dataset) share a hash for aggregation.

### `experiments/log.md` per-cycle entry format
```
## <run_id or YYYY-MM-DD> — <variant id>: <short hypothesis>
- Hypothesis: <from backlog>
#### Implementer (screen)
  - connectome: mean±std (n=3), Δ vs baseline, screen pass/fail
  - mouse:      mean±std (n=3), Δ vs baseline, screen pass/fail
  - commands + result file ids
#### Verifier (confirm)
  - per dataset: mean±std (n), Δ, SE, 95% CI lower bound, CONFIRMED/NOT
#### Critic verdict
  - frozen-integrity / leakage / reproducibility / significance / robustness: PASS/FAIL
  - recommendation: keep | kill | iterate
#### Decision: keep | kill | iterate
```

## Invariants (see CLAUDE.md)
- Never modify frozen files (a PreToolUse hook blocks edits to them).
- Always evaluate BOTH datasets; ≥3 seeds; report mean ± std.
- Every reported number traces to a `results/*.json` + a re-runnable command — never fabricated.
- Never hardcode/peek at the target metric inside a variant.
- One git commit per experiment; reproducible via `git checkout` + the logged command.
- A gain within noise is not a gain.

## Run the campaign
For each ranked hypothesis: `/run-experiment <backlog-id>` (e.g. `/run-experiment H01`).
Stop when a budget/stop-criterion above is hit.
