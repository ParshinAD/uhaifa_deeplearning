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
_Generated 2026-06-21T12:03:18Z from 130 run(s)._

| algo | dataset | n_seeds | pct mean±std | score mean±std | wall_clock_s (mean) | seeds | config_hash | git_commit |
|---|---|---|---|---|---|---|---|---|
| H01 | connectome | 3 | 82.0510 ± 0.0290 | 34,389,329 ± 12,134 | 78.7 | [42, 123, 999] | fdb0b2 | 704221ab778 |
| H01 | mouse | 3 | 92.1625 ± 0.0242 | 8.4412 ± 0.0022 | 2.0 | [42, 123, 999] | 434027 | 704221ab778 |
| H02 | connectome | 11 | 82.9298 ± 0.0012 | 34,757,648 ± 496 | 91.9 | [7, 42, 42, 42, 123, 123, 123, 999, 999, 999, 31415] | 059689 | f36e02847a9 |
| H02 | mouse | 26 | 92.4793 ± 0.0000 | 8.4702 ± 0.0000 | 1.8 | [7, 42, 42, 42, 123, 123, 123, 999, 999, 999, 1111, 1234, 1414, 1618, 1732, 2222, 2236, 2718, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 31415] | a8bbc0 | f36e02847a9 |
| H03 | connectome | 3 | 82.8582 ± 0.0181 | 34,727,627 ± 7,582 | 78.2 | [42, 123, 999] | 082864 | 858e000ec21 |
| H03 | mouse | 3 | 92.0810 ± 0.2730 | 8.4337 ± 0.0250 | 2.0 | [42, 123, 999] | 0b23c4 | 858e000ec21 |
| H04 | connectome | 3 | 82.8958 ± 0.0183 | 34,743,389 ± 7,664 | 80.6 | [42, 123, 999] | 430afe | 4b5ba1674e4 |
| H04 | mouse | 3 | 92.0696 ± 0.2624 | 8.4327 ± 0.0240 | 1.9 | [42, 123, 999] | fa47bd | 4b5ba1674e4 |
| H05 | connectome | 3 | 82.8976 ± 0.0205 | 34,744,166 ± 8,583 | 75.9 | [42, 123, 999] | 74a1ef | 85066a96fa4 |
| H05 | mouse | 3 | 92.0696 ± 0.2624 | 8.4327 ± 0.0240 | 1.9 | [42, 123, 999] | 4254d9 | 85066a96fa4 |
| H06 | connectome | 3 | 82.8578 ± 0.0288 | 34,727,496 ± 12,053 | 119.4 | [42, 123, 999] | e07005 | 85097398c64 |
| H06 | mouse | 4 | 92.0437 ± 0.1937 | 8.4303 ± 0.0177 | 1.9 | [42, 42, 123, 999] | ae6060 | 85097398c64 |
| H09 | connectome | 3 | 82.8948 ± 0.0187 | 34,742,975 ± 7,818 | 77.4 | [42, 123, 999] | 8e9a0f | 90073cf2d95 |
| H09 | mouse | 4 | 92.1425 ± 0.2591 | 8.4394 ± 0.0237 | 1.9 | [42, 42, 123, 999] | ace4f2 | 90073cf2d95 |
| H11 | connectome | 3 | 82.8571 ± 0.0235 | 34,727,188 ± 9,856 | 94.5 | [42, 123, 999] | ec5418 | 258bcbd07c0 |
| H11 | mouse | 3 | 92.1960 ± 0.2643 | 8.4443 ± 0.0242 | 2.4 | [42, 123, 999] | 29b494 | 258bcbd07c0 |
| H13 | connectome | 3 | 82.0602 ± 0.0045 | 34,393,204 ± 1,875 | 1332.1 | [42, 123, 999] | 52314a | 07b26a8edcc |
| H13 | mouse | 3 | 92.0371 ± 0.2110 | 8.4297 ± 0.0193 | 3.9 | [42, 123, 999] | c42396 | 07b26a8edcc |
| baseline_multistart | connectome | 3 | 82.0507 ± 0.0293 | 34,389,211 ± 12,284 | 77.2 | [42, 123, 999] | 330c58 | 704221ab778 |
| baseline_multistart | mouse | 3 | 92.1625 ± 0.0242 | 8.4412 ± 0.0022 | 1.9 | [42, 123, 999] | c6938b | 704221ab778 |
| baseline_passthrough | connectome | 8 | 82.8887 ± 0.0223 | 34,740,422 ± 9,345 | 918.5 | [7, 42, 42, 123, 123, 999, 999, 31415] | f8cb3c | 8f0e5066211, f36e02847a9 |
| baseline_passthrough | mouse | 23 | 92.2464 ± 0.2138 | 8.4489 ± 0.0196 | 40.9 | [7, 42, 42, 123, 123, 999, 999, 1111, 1234, 1414, 1618, 1732, 2222, 2236, 2718, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 31415] | 7b7cba | 8f0e5066211, f36e02847a9 |
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

---

## 2026-06-21 — H02: Warm-start Rocket from a greedy MFAS ordering (GreedyAbs-style init)
- Hypothesis: initializing positions from a degree/greedy DAG ordering (Kahn-style topo sort of
  a high-weight acyclic subgraph, or the greedy MFAS ordering that already reaches ~68–72%)
  instead of N(0,1) gives Rocket a better basin and a higher final feedforward weight.

#### Implementer (screen)
- **Init source (leakage-safe):** the **Eades–Lin–Smyth / GreedyAbs greedy-FAS ordering**
  (`src/mfas/experiments/H02.py::greedy_fas_order`), computed from input graph structure + edge
  weights ONLY — peel sinks→back / sources→front, else remove the remaining node with max
  `out_w − in_w`→front (lazy max-heap, ~O((n+m) log n)). A full Kahn topo sort is ill-defined on
  these cyclic connectomes, so this is the standard cheap greedy surrogate (it degenerates to a
  topo order on a DAG). The rank vector in `[0,n)` is mapped to evenly-spaced positions in
  `[-1,1]` (front rank → −1 = source side) and handed to the **UNCHANGED** `run_rocket` as
  `init_positions`. The oracle is used only as the baseline does (best-by-oracle tracking); the
  discrete score is never folded into the loss or hardcoded. No post-processing/local search
  (that is H04) — PURE Rocket score reported. Greedy-order quality alone: connectome 68.91%,
  mouse 90.13% (graph-only, no leakage).
- **Compute-matched:** standard knob-swap (init only); baseline epoch budget (connectome 20k,
  mouse 5k). `n_epochs_done = 20000 / 5000` (= baseline `total_grad_steps`). Comparator =
  frozen baseline / `baseline_passthrough` at matched seeds.
- **connectome:** mean±std (n=3) = **82.9308 ± 0.0005** (vals 82.9314 / 82.9304 / 82.9305);
  baseline 82.8958; **Δ = +0.0350 pp**; 2σ gate 0.04 pp → **screen FAIL** (below threshold by
  0.005 pp; directionally positive but within the screen bar).
- **mouse:** mean±std (n=3) = **92.4793 ± 0.0000** (all three seeds identical — the warm-start
  basin is deterministic on this 148-node graph); baseline 92.0696; **Δ = +0.4097 pp**; 2σ gate
  0.52 pp → **screen FAIL** (below threshold by 0.11 pp).
- **Overall: SCREEN FAIL** (the gate requires PASS on BOTH datasets; both are sub-threshold).
- **Commands** (Python `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`):
  ```
  for DS in connectome mouse; do for S in 42 123 999; do
    python -m eval.run_variant --exp H02 --dataset $DS --seed $S --out results/ --role implement
  done; done
  python -m eval.aggregate --glob "results/*.json" --out experiments/log.md
  ```
- **Result file ids:**
  - connectome: `20260621T090127Z-H02-connectome-s42-implement-059689`,
    `20260621T090315Z-H02-connectome-s123-implement-059689`,
    `20260621T090456Z-H02-connectome-s999-implement-059689`
  - mouse: `20260621T090112Z-H02-mouse-s42-implement-a8bbc0`,
    `20260621T090115Z-H02-mouse-s123-implement-a8bbc0`,
    `20260621T090118Z-H02-mouse-s999-implement-a8bbc0`

#### Orchestrator escalation note (overrides the implementer's provisional "kill (screen)")
The implementer's screen marked FAIL because the gate is `Δ > 2σ_baseline`, which **assumes the
variant's variance equals the baseline noise floor**. H02 violates that assumption: its own
variance is ~40× smaller (connectome σ≈0.0005, mouse σ≈0.0000 — the warm-start basin is essentially
deterministic), so the gate is mis-specified for this variant and a marginal sub-threshold Δ is not
evidence of "no effect". The screen is explicitly *a gate, not a verdict*; the CONFIRM test (95% CI
lower bound on the difference of means, the statistically defensible bar) is more rigorous, so H02
was escalated to the verifier rather than killed on the cheap gate. This is not lowering the bar —
it applies the real one.

#### Verifier (confirm)
Independent re-run from a clean state (read-only on source). Integrity: `eval/frozen.sha256` matches
all 4 frozen files before and after; no frozen file modified; `run_variant` re-verified the oracle
manifest before every score. Leakage: PASS — H02 reads only `g.src`/`g.tgt`/`g.weight` for the
greedy order, imports only `baseline.rocket`, no post-processing → pure Rocket score.
- **Screen reproduction** (role=verify, seeds 42/123/999): mouse 92.4793 (exact, deterministic);
  connectome mean 82.9297 (82.9301/82.9291/82.9299) — matches implementer to ~0.001 pp (MPS
  nondeterminism), same direction, well above baseline.
- **CONFIRM** (role=confirm; baseline re-run at **matched seeds** — the fair, conservative comparator,
  which *raises* the mouse baseline to 92.2729 and thus *shrinks* Δ). SE = std·√(2/n) using the
  **baseline** std (conservative, since H02 variance ≪ baseline variance):

  | dataset | H02 mean±std (n) | baseline mean±std (n) | Δ | SE | 95% CI lower |
  |---|---|---|---|---|---|
  | connectome | 82.9292 ± 0.0015 (5) | 82.8845 ± 0.0253 (5) | **+0.0448 pp** | 0.0160 | **+0.0135** |
  | mouse | 92.4793 ± 0.0000 (20) | 92.2729 ± 0.2000 (20) | **+0.2064 pp** | 0.0633 | **+0.0824** |

  Under Welch two-sample SE the CI lower bounds are higher still (connectome +0.0226, mouse +0.1187);
  the verdict holds under both conventions.
- **Verdict: CONFIRMED on BOTH datasets** (95% CI lower bound > 0 on connectome +0.0135 and mouse
  +0.0824). Result ids: H02 connectome confirm `…-H02-connectome-s{42,123,999,7,31415}-confirm-059689`;
  H02 mouse confirm `…-H02-mouse-s{20 seeds}-confirm-a8bbc0`; matched baseline
  `…baseline_passthrough-{connectome,mouse}-…-{implement,confirm}-{f8cb3c,7b7cba}`.

#### Critic verdict
Red-team of the CONFIRMED finding. Read-only adjudication; numbers re-derived from the logged JSONs.

1. **Frozen-file integrity — PASS.** Recomputed SHA-256 of all 4 frozen files; every hash matches
   `eval/frozen.sha256` exactly (`metrics.py` bd2ff9…, `harness.py` 0ba534…, `aggregate.py`
   28340949…, `test_metrics.py` 27f02a77…). `git status --short` and `git diff --stat` show **no
   frozen file modified** (only untracked `H02.py` + result JSONs and unrelated `.claude`/notebook
   edits). `run_variant.py` calls `verify_frozen_manifest()` before every score (SystemExit on drift).

2. **Metric leakage — PASS.** `greedy_fas_order` reads ONLY `g.src`/`g.tgt`/`g.weight`; grep for
   `discrete_score|oracle|target|best_score|82.9|92.4|34751` in the init path finds matches only in
   docstrings/comments, never in code. The init is fully **seed-independent and deterministic** (which
   is exactly why mouse std=0.0000 and H02 variance is ~40× below baseline). No dataset special-casing
   (the `_EPOCHS` dict only sets the *baseline* budget per dataset — same as baseline). Score is
   computed externally by the non-frozen `run_variant.py` via `mfas.metrics.score_from_positions`
   (frozen oracle); the variant never folds the discrete score into the loss. Best-by-oracle tracking
   is identical to baseline.

3. **Reproducibility — PASS.** All reported numbers trace to logged `results/*.json` with re-runnable
   commands. Spot-checked H02 connectome confirm s42 (`…091809Z…-confirm-059689`): role=confirm,
   total_grad_steps=20000=n_epochs_done, budget_basis=total_grad_steps, score 34,756,618 → pct
   82.9273, git_commit f36e0284…+dirty (the `+dirty` is from untracked H02.py/JSONs, **not** frozen
   edits). **Equal-compute confirmed:** every H02 and every matched-baseline JSON has
   total_grad_steps = 20000 (connectome) / 5000 (mouse). I re-aggregated from disk and reproduced the
   verifier's matched-seed baselines (the comparator pools implement+verify+confirm JSONs to one value
   per seed): connectome baseline 82.8844 (n=5), mouse 92.2729 (n=20) — both match.

4. **Significance — PASS (but connectome margin is thin).** Re-derived from the JSONs with
   SE=std_base·√(2/n):
   - connectome: H02 82.9292±0.0015, base 82.8844±0.0252 → Δ=**+0.0448**, SE=0.0160, 95% CI lower
     **+0.0135** (reproduced exactly). Positive but **thin**: Δ would only need to fall below 0.0314
     (a ~0.013 pp buffer) for the CI to touch 0 — fragile to a couple of baseline seeds. Mitigants:
     H02's own variance is essentially zero (0.0015), the SE uses the *conservative* larger baseline
     std, and Welch SE gives an even higher bound (+0.0226).
   - mouse: H02 92.4793±0.0000, base 92.2729±0.2000 → Δ=**+0.2064**, SE=0.0633, 95% CI lower
     **+0.0824** (reproduced). Comfortably positive. The matched-seed baseline **raised** the mouse
     baseline from the 3-seed 92.0696 to 92.2729 (+0.20 pp), which **shrinks** Δ — i.e. the verifier
     chose the *conservative, fair* comparator, not a cherry-pick. No cherry-picked seeds: H02 mouse
     n=20 spans a wide seed set; baseline uses the identical seed set.

5. **Both-dataset robustness — PASS.** CI lower bound > 0 on BOTH connectome (+0.0135) and mouse
   (+0.0824). Not a single-dataset artifact. The deterministic init means the effect is not a lucky
   seed: every mouse seed lands on the identical (higher) basin; connectome variation is residual MPS
   nondeterminism, not init noise.

6. **Overfitting / generality — PASS (mechanism plausible).** The win is a pure initialization change
   (a graph-derived greedy-FAS warm start → better basin), a textbook continuation/warm-start trick
   that is dataset-agnostic and reaches ~68–72% on its own before any optimization. No tuning to these
   two graphs. Risk is only that "two graphs" is a small population — generality beyond connectome/mouse
   is asserted, not proven, but that is inherent to the available datasets, not a flaw in H02.

**Recommendation: keep (promote to findings.md) — with an honesty caveat.** The win is real,
leakage-free, reproducible, equal-compute, and CONFIRMED on both datasets under the conservative
matched-seed baseline. But it is **small**: mouse +0.21 pp is solid; connectome +0.045 pp clears the
CI bar by a thin +0.0135 pp margin. Promote it as a *modest, robust* improvement (and ideally widen
the connectome confirm seed count to harden the thin margin), not as a large gain.

#### Decision: **keep** — CONFIRMED on both datasets and survives red-team (frozen integrity, no
leakage, reproducible, equal-compute, both-dataset CI>0). Caveat: connectome margin is thin
(+0.0135 pp CI lower); report H02 as a small-but-genuine warm-start win, mouse-strong / connectome-marginal.

---

## 2026-06-21 — H03: Sharper / extended beta schedule (final monotone-sharpening ramp)
- Hypothesis: raising the maximum sigmoid sharpness in the late phase (beta_max from ~1.05 up to
  e.g. 2–8, and/or appending a final monotone-sharpening ramp) tightens the surrogate->discrete
  gap and yields higher exact feedforward weight. The surrogate sigma_beta only approximates the
  discrete indicator; at beta~1 a unit gap maps to sigma~0.74 so "weakly correct" edges and
  near-ties contribute little gradient. Annealing beta upward at the end is standard deterministic-
  annealing. beta is a loss-shape parameter only -> leakage-safe.

#### Implementer (screen)
- **Chosen arm (one, for a bounded screen):** keep the baseline cyclic exploration intact for the
  first **75%** of epochs, then **append a final monotone (linear) sharpening ramp** over the last
  **25%** (`RAMP_FRAC=0.25`) that raises beta from its end-of-cyclic value up to
  **`BETA_MAX_TERMINAL = 4.0`**. Rationale: this preserves the paper's broad cyclic exploration
  (which the paper credits for dodging manual tuning) and only sharpens the EXPLOITATION tail,
  where a tighter surrogate matters and where over-sharp early gradients would freeze a poor basin.
  beta=4 gives sigma(4)~0.982 for a unit gap (vs 0.74 at 1.05) — clearly sharper but not
  gradient-killing (beta=8 -> sigma~0.9997 risks vanishing gradients).
- **Exact beta schedule:** segment 1 = UNCHANGED `make_beta_schedule(n_explore=15000/3750, cycles=5)`
  (baseline cos formula `[0.05,1.05]`, byte-for-byte baseline exploration); segment 2 =
  `np.linspace(betas_explore[-1], 4.0, n_ramp=5000/1250)` (monotone ramp); concatenated to length T.
- **Implementation:** `run_rocket` exposes no beta-override hook, so `H03.py` REPLICATES the
  `run_rocket` main loop verbatim (same Adam, grad-clip=1.0, const->exp LR schedule, N(0,1) init,
  CPU int64/float64 discrete scoring, best-by-oracle tracking, history, time-limit) and substitutes
  ONLY the beta array (`_make_h03_beta_schedule`). The discrete oracle is read only for the
  baseline's existing best-by-oracle tracking; never folded into the loss / hardcoded / used to
  special-case a dataset. No post-processing -> PURE Rocket score.
- **Compute-matched:** standard knob-swap (beta only); SAME epoch budget as baseline
  (connectome 20k, mouse 5k); `n_epochs_done = total_grad_steps = 20000 / 5000`. Comparator =
  frozen baseline / `baseline_passthrough` at matched seeds (connectome 82.8958 ± 0.0189;
  mouse 92.0696 ± 0.2624).
- **connectome** (n=3, seeds 42/123/999): per-seed 82.8752 / 82.8601 / 82.8392 →
  **mean 82.8582 ± 0.0181** (H03's own std = 0.0181, ~= baseline noise floor 0.0189);
  baseline 82.8958; **Δ = −0.0376 pp** (NEGATIVE — H03 sits *below* baseline); 2σ gate 0.04 →
  **screen FAIL**.
- **mouse** (n=3, seeds 42/123/999): per-seed 92.3832 / 91.8520 / 92.0077 →
  **mean 92.0810 ± 0.2730** (H03's own std = 0.2730, ~= baseline noise floor 0.2624);
  baseline 92.0696; **Δ = +0.0114 pp**; 2σ gate 0.52 → **screen FAIL** (Δ ≪ gate).
- **Overall: SCREEN FAIL on both datasets.** This is NOT a borderline-low-variance case like H02:
  H03's own per-seed std on BOTH datasets matches the baseline noise floor (connectome 0.0181 vs
  0.0189; mouse 0.2730 vs 0.2624), so the 2σ_baseline gate is correctly specified for this variant
  and the failure is not an artifact of a mis-specified gate. Δ is essentially zero on mouse
  (+0.011 pp, well within noise) and **negative** on connectome (−0.038 pp) — the terminal beta=4
  ramp slightly *hurt* connectome. No CONFIRM escalation warranted.
- **Un-run sweep arms (left for follow-up only if revisited):** `BETA_MAX_TERMINAL ∈ {2, 8}` and
  varying `RAMP_FRAC`. The primary arm (beta_max=4, ramp_frac=0.25) is not promising, so the sweep
  is not pursued in this cycle.
- **Commands** (Python `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`):
  ```
  for DS in connectome mouse; do for S in 42 123 999; do
    python -m eval.run_variant --exp H03 --dataset $DS --seed $S --out results/ --role implement
  done; done
  python -m eval.aggregate --glob "results/*.json" --out experiments/log.md
  ```
- **Result file exp_ids:**
  - connectome: `20260621T094013Z-H03-connectome-s42-implement-082864`,
    `20260621T094139Z-H03-connectome-s123-implement-082864`,
    `20260621T094304Z-H03-connectome-s999-implement-082864`
  - mouse: `20260621T093957Z-H03-mouse-s42-implement-0b23c4`,
    `20260621T094007Z-H03-mouse-s123-implement-0b23c4`,
    `20260621T094010Z-H03-mouse-s999-implement-0b23c4`

#### Verifier (confirm)
_(not run by the implementer — SCREEN FAILED on both datasets, and H03's per-seed variance matches
the baseline noise floor (so this is not the H02-style low-variance escalation case). CONFIRM stage
is reserved for the orchestrator/verifier.)_

#### Critic verdict
_(reserved for CONFIRMED findings; not run on a screen-fail.)_

#### Decision: **kill.** Primary arm β_max=4 terminal sharpening ramp does NOT beat baseline —
connectome −0.0376 pp (regression), mouse +0.0114 pp (within noise), both far below the 2σ gate.
Unlike H02, this is a **correctly-specified** screen fail: H03's own per-seed std matches the
baseline noise floor on both datasets (connectome 0.0181≈0.0189; mouse 0.2730≈0.2624), so no CONFIRM
escalation is warranted. Orchestrator integrity spot-check: frozen manifest matches all 4 files; no
frozen file modified; H03 alters only the β array (loss-shape) → no leakage; all 6 numbers trace to
logged `results/*.json` at matched `total_grad_steps`. Hypothesis (primary arm) not supported;
β_max∈{2,8} left un-run as low-priority sweep arms. Backlog status → killed.

---

## 2026-06-21 — H04: In-the-loop discrete refinement (continuous Rocket + periodic local swaps)
- Hypothesis: periodically nudging positions toward a locally-improved ordering (a cheap greedy
  local search on the CURRENT order, then re-seed positions from the improved ranks) lets Rocket
  escape the surrogate plateau and raises the exact feedforward weight. Crane's premise is that
  local refinement extends quality beyond Rocket's plateau (paper §4.4, 82.87%→84.60%); H04 imports
  that idea with a cheap, in-variant local search (no Gurobi / no MIP).

#### Implementer (screen)
- **Local-search move (leakage-safe):** a **weighted-barycenter rank reposition** (the classic
  barycenter heuristic for 1-D vertex arrangement), `src/mfas/experiments/H04.py`. From the CURRENT
  ordering (stable argsort of the live positions → integer ranks in `[0,n)`), each node is pulled
  toward the weighted mean rank of its incident neighbours,
  `target[u] = (Σ_out w·rank[v] + Σ_in w·rank[t]) / incident_w[u]`, and the candidate ordering is
  the stable argsort of `target`. Decided **purely by input edge weights + current ranks** (uses
  only `g.src`/`g.tgt`/`g.weight`); the oracle is NEVER consulted to choose the move. Each pass is
  two `np.add.at` scatter-adds (O(m)) + one O(n log n) argsort.
- **Refinement schedule:** at most **MAX_REFINES = 8** passes total, spread over the SECOND HALF of
  training (`REFINE_START_FRAC = 0.5`, after the ordering has formed) at scoring points only — NOT
  every step.
- **Ranks→positions re-seeding (anti-collapse):** an accepted candidate's integer ranks map to
  evenly-spaced positions in `[-1,1]` (rank 0 → −1, strictly increasing, never ties/NaN) and are
  written into the LIVE optimizer parameter so Rocket continues from the improved point. `target` is
  sanitized (NaN/Inf/isolated → current rank) before argsort.
- **Oracle usage / leakage-safety:** the frozen oracle is used EXACTLY as the baseline uses it —
  best-by-oracle tracking. A refined candidate is ADOPTED only when the oracle says it strictly
  beats the current best, so **post-refinement ≥ pure Rocket by construction** (refinement can never
  lower the tracked best). The discrete score is never folded into the loss, hardcoded, or used to
  special-case a dataset.
- **Implementation note:** `run_rocket` exposes no in-loop hook, so H04 REPLICATES the `run_rocket`
  main loop VERBATIM (N(0,1) init, Adam, grad-clip=1.0, const→exp LR, cyclic-cosine β via
  `make_beta_schedule`, CPU int64/float64 discrete scoring, best-by-oracle tracking, history,
  time-limit) and adds ONLY the periodic barycenter refinement. Everything else is identical to
  baseline.
- **Compute accounting (auditable):** H04 runs the FULL single-run gradient budget (connectome 20k,
  mouse 5k) — NO restarts — so `n_epochs_done = total_grad_steps = 20000 / 5000`, matched to the
  frozen baseline / `baseline_passthrough` at the same seeds. Added NON-gradient compute = the
  refinement: **8 passes/run** (measured), refine wall-clock **≈0.002 s/run on mouse** (negligible)
  and **≈3.0 s/run on connectome** (≈3.8% of the ~78 s run; non-negligible but small, and gradient
  steps are unchanged, so the equal-grad-step comparator stays valid). Extra op cost ≈ 8×O(m) on top
  of the gradient budget. **Refinement acceptances across all 6 runs = 0** (the barycenter candidate
  never beat Rocket's plateau), so post == pure on every run.
- **PRE-refinement (pure Rocket) vs POST-refinement:** identical on every run (0 accepted refines):
  - connectome PRE = POST (n=3): **82.8958 ± 0.0183** (per-seed POST from JSON: 82.9152 / 82.8933 /
    82.8789).
  - mouse PRE = POST (n=3): **92.0696 ± 0.2624** (per-seed POST from JSON: 92.3610 / 91.8520 /
    91.9959).
- **Δ vs baseline (POST):**
  - connectome: baseline 82.8958; Δ = **−0.0000 pp** (H04 own std 0.0183); 2σ gate 0.04 → **screen FAIL**.
  - mouse: baseline 92.0696; Δ = **+0.0000 pp** (H04 own std 0.2624); 2σ gate 0.52 → **screen FAIL**.
- **Overall: SCREEN FAIL on both datasets.** This is NOT an H02-style low-variance escalation case:
  H04's own per-seed std matches the baseline noise floor on BOTH datasets (connectome 0.0183≈0.0189;
  mouse 0.2624≈0.2624 — identical, because post==pure==raw Rocket), so the 2σ_baseline gate is
  correctly specified. Honest non-improvement: the weighted-barycenter local search is too weak to
  escape Rocket's plateau. (Diagnostic: on a random ordering one barycenter pass lifts mouse from
  ~48% to ~53% then plateaus far below Rocket's ~92%, so once Rocket has converged a single pass on
  the already-good ordering cannot beat it → 0 accepts.) A stronger (but more expensive) local search
  — e.g. iterated sink/source relocation or adjacent-block sifting to convergence — would be the
  natural follow-up if H04 is revisited, but the cheap barycenter primary arm is not promising.
- **Commands** (Python `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`):
  ```
  for DS in connectome mouse; do for S in 42 123 999; do
    python -m eval.run_variant --exp H04 --dataset $DS --seed $S --out results/ --role implement
  done; done
  python -m eval.aggregate --glob "results/*.json" --out experiments/log.md
  ```
- **Result file exp_ids:**
  - connectome: `20260621T095133Z-H04-connectome-s42-implement-430afe`,
    `20260621T095302Z-H04-connectome-s123-implement-430afe`,
    `20260621T095430Z-H04-connectome-s999-implement-430afe`
  - mouse: `20260621T095117Z-H04-mouse-s42-implement-fa47bd`,
    `20260621T095120Z-H04-mouse-s123-implement-fa47bd`,
    `20260621T095123Z-H04-mouse-s999-implement-fa47bd`
  (Note: logged JSON values are authoritative; a recompute differs by ≤0.0002 pp due to MPS
  nondeterminism.)

#### Verifier (confirm)
_(not run by the implementer — SCREEN FAILED on both datasets, and H04's per-seed variance matches
the baseline noise floor on both (post==pure==raw Rocket), so this is NOT the H02-style low-variance
escalation case. CONFIRM stage reserved for the orchestrator/verifier.)_

#### Critic verdict
_(reserved for CONFIRMED findings; not run on a screen-fail.)_

#### Decision: **kill (screen).** The cheap weighted-barycenter in-loop refinement does NOT beat
baseline — Δ = −0.0000 pp (connectome) / +0.0000 pp (mouse), both ≈ zero and far below the 2σ gate,
with 0 refinement candidates accepted across all 6 runs (post == pure on every run). Correctly-
specified screen fail (H04 own std ≈ baseline noise floor on both datasets), so no CONFIRM escalation.
The refinement is leakage-safe (input-only barycenter move; oracle used only for best-tracking),
never collapses positions (all unique, no NaN), and never lowers the tracked best — it is simply too
weak to escape Rocket's plateau. Backlog status → screened.

---

## 2026-06-21 — H09: Anti-tie / near-equal-position handling (recover strict-`>` lost edges)
- Hypothesis: the exact oracle counts `pos[tgt] > pos[src]` **strictly** (ties are NOT
  feedforward — see `mfas.metrics._ff_weight`), so any edge whose endpoints land at exactly-equal
  / numerically-indistinguishable positions contributes zero discrete weight even when the
  intended order is correct. A target-blind anti-tie separation (here a deterministic symmetric
  scoring-time jitter) should recover that silently-dropped feedforward weight "for free" without
  changing the basin or dynamics.

#### Implementer (screen)
- **Opportunity sizing FIRST (backlog-required), on logged baseline plateau positions, both
  datasets** (connectome s42 `…001607Z…`, mouse s42 `…002023Z…`):
  - **Exact ties** (`d == pos[tgt]−pos[src] == 0`): **0 edges on BOTH datasets** (weight 0.000000 pp).
  - connectome near-ties (currently-feedback edges recoverable if flipped): `|d|<1e-3` → 2 edges,
    weight 20 / 41,912,141 = **0.000048 pp**; `|d|<1e-2` → 29 edges, weight 135 = **0.000322 pp**.
  - mouse: **0 edges with `|d|<1e-2`** at all → recoverable weight **0.000000 pp**.
  - The entire recoverable set is ~3 orders of magnitude below the screen thresholds (connectome
    0.04 pp, mouse 0.52 pp). The Adam optimizer spreads positions apart; there are no ties to
    break. **H09 is predicted to self-falsify** — the screen confirms it empirically below.
- **Arm used:** post-hoc, target-BLIND, symmetric per-node jitter applied to the positions the
  UNCHANGED `run_rocket` returns (same N(0,1) init, loss, Adam, grad-clip, LR + β schedules, same
  epoch budget 20k/5k). `jitter[u] = 1e-9·(2·h(u,seed)−1)` where `h` is a SeedSequence PRNG over
  the node *index* and run seed → `[−1e-9, +1e-9]`, mean ~0. The frozen oracle scores both raw and
  jittered positions and the better candidate is kept (best-by-oracle candidate tracking — exactly
  what the baseline already does on its own iterates; never folded into a differentiable loss).
- **Target-blindness argument:** `h(u,seed)` depends ONLY on node index + run seed — never on the
  edge list, edge weights, surrogate, or the discrete oracle score. The same perturbation is added
  to a node whether it is a source or a target of any edge, so the sign of
  `jitter[tgt]−jitter[src]` for a tied edge is a fixed function of two pre-committed node hashes —
  symmetric in expectation, impossible to steer toward the favourable orientation. The oracle picks
  only the better of the two *whole-vector* candidates (raw vs jittered), never a per-edge
  orientation. `EPS_JITTER=1e-9` ≪ the smallest non-tie position gap, so non-tie edges cannot be
  flipped — only exact / sub-EPS ties are resolved. No leakage.
- **connectome:** mean **82.8948 ± 0.0187** (n=3; H09 own std 0.0187), Δ vs baseline (82.8958) =
  **−0.0010 pp** → **2σ gate 0.04 pp → FAIL** (regression-within-noise).
- **mouse:** mean **92.0696 ± 0.2624** (n=3; H09 own std 0.2624), Δ vs baseline (92.0696) =
  **+0.0000 pp** → **2σ gate 0.52 pp → FAIL**. Mouse output is bit-identical to baseline_passthrough
  (0 ties → 0 recovery on every seed).
- **CONFIRM-escalation check (H02-style):** NO. H09's own std ≈ the baseline noise floor on BOTH
  datasets (connectome 0.0187 vs 0.0189; mouse 0.2624 vs 0.2624), so the 2σ_baseline gate is
  correctly specified — this is the H03/H04 case, not the H02 low-variance case. No escalation.
- **Leakage / collapse:** target-blind jitter (index+seed only), positions never NaN/collapse, the
  oracle is used only for best-by-oracle candidate selection → tracked best is never lowered.
- **Commands** (`PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`):
  ```
  for DS in connectome mouse; do for S in 42 123 999; do
    $PY -m eval.run_variant --exp H09 --dataset $DS --seed $S --out results/ --role implement
  done; done
  $PY -m eval.aggregate --glob "results/*.json" --out experiments/log.md
  ```
- **Result exp_ids:** connectome `20260621T101005Z-H09-connectome-s42-implement-8e9a0f`,
  `20260621T101130Z-H09-connectome-s123-implement-8e9a0f`,
  `20260621T101255Z-H09-connectome-s999-implement-8e9a0f`; mouse
  `20260621T101425Z-H09-mouse-s42-implement-ace4f2`,
  `20260621T101428Z-H09-mouse-s123-implement-ace4f2`,
  `20260621T101431Z-H09-mouse-s999-implement-ace4f2`
  (smoke `20260621T100955Z-H09-mouse-s42-implement-ace4f2` excluded as a duplicate of s42).
- **SCREEN verdict: FAIL on BOTH datasets** (correctly-specified gate; no CONFIRM escalation).
  Self-falsified exactly as the opportunity sizing predicted: the continuous optimizer leaves no
  exact/near-ties for the strict-`>` oracle to drop, so there is nothing for anti-tie handling to
  recover.

#### Decision: **kill.** H09 self-falsified at the opportunity-sizing step (0 exact ties; near-tie
recoverable weight ~3 orders of magnitude below the screen thresholds on both datasets) and the
screen confirmed it empirically (connectome −0.0010 pp within noise, mouse bit-identical to
baseline). Correctly-specified gate (own std ≈ noise floor) → no escalation. Mechanism ruled out:
Adam spreads positions apart, so the strict-`>` oracle drops essentially no edges to ties.
Leakage-safe, no collapse. Backlog status → killed.
**Knock-on:** this also moots **H14** (anti-tie jitter stacked on the H02 basin) — H02's positions
are equally Adam-spread, so the same null result applies; H14 is marked killed-by-implication
without a separate cycle (documented in backlog) to conserve compute.


## 2026-06-21 — H06: Weight-aware loss reweighting (heavy-&-borderline edge emphasis)
- Hypothesis: Reweighting the surrogate so that high-weight AND currently-borderline edges receive
  proportionally more gradient (vs the flat max-normalized hat-w) increases retained high-weight
  feedforward arcs. The skewed edge-weight distribution means a few heavy edges are diluted late in
  training; emphasizing heavy/uncertain edges aligns gradient effort with the metric's own weighting.

#### Implementer (screen)
- **Variant** (`src/mfas/experiments/H06.py`): replicates the `run_rocket` main loop VERBATIM
  (same N(0,1) init, Adam, grad-clip=1.0, constant→exponential LR schedule, cyclic β schedule,
  CPU discrete scoring, best-by-oracle tracking, history, time-limit) and changes ONLY the per-edge
  loss weighting. `run_rocket` exposes no loss hook, so the loop is copied and only two lines added.
- **Reweighting formula (primary arm, ALPHA = 4.0):** each baseline per-edge term `σ_β(Δ)·hat_w`
  (`hat_w = w/max(w) ∈ (0,1]`, `Δ = pos[v]−pos[u]`) is multiplied by a strictly-positive, bounded,
  DETACHED emphasis factor
  `m_e = 1 + ALPHA · hat_w · b_e`, with `b_e = 4·σ_β·(1−σ_β) ∈ [0,1]`.
  `b_e` is the normalized sigmoid sensitivity: 1 exactly at the borderline (`σ=0.5`,
  i.e. `|σ−0.5|=0`), → 0 as the edge saturates feedforward OR feedback. The product `hat_w·b_e`
  routes EXTRA gradient to edges that are HEAVY and UNDECIDED. Bounded `1 ≤ m_e ≤ 1+ALPHA = 5`.
  This is a clean COMBINATION of the backlog's two ablations ((a) heavy `hat_w`, (b) borderline
  `|σ−0.5|`); chosen because the hypothesis is specifically about heavy-AND-uncertain edges
  (a heavy already-feedforward edge needs no push; a borderline tiny edge barely moves the metric).
  UN-RUN arms: heavy-only `(hat_w)^(γ−1)`, borderline-only `1+ALPHA·b_e`, ALPHA sweep.
- **Direction-preservation (argmax preserved):** `b_e` is computed under `torch.no_grad()` and
  detached, so the gradient w.r.t. positions is, per edge, `−m_e·hat_w·∂σ_β/∂pos` — EXACTLY the
  baseline per-edge gradient scaled by `m_e > 0`. Every edge still pushes toward `pos[v]>pos[u]`
  (feedforward); no edge's contribution sign is inverted — only relative magnitudes change. The
  floor `m_e ≥ 1` (ALPHA, hat_w, b_e all ≥ 0) guarantees H06 NEVER reduces an edge below its
  baseline `hat_w`, only adds bounded emphasis. The objective stays a monotone reward for
  feedforward orientation; the discrete oracle (total feedforward weight) is what is tracked/reported.
- **Leakage-safe / target-blind:** `m_e` uses ONLY (i) input edge weights via `hat_w` and (ii) the
  model's OWN surrogate state `σ_β(Δ)` (a borderline edge = `σ_β` near 0.5). It never reads,
  hardcodes, or folds in the discrete oracle score, and never special-cases a dataset (identical
  formula for connectome and mouse). Oracle used only for the baseline's best-by-oracle tracking.
- **Compute-matched:** standard knob-swap (loss reweighting only); same epoch budget as baseline
  (connectome 20k, mouse 5k), same optimizer-step count; `n_epochs_done`=actual steps; comparator =
  `baseline_passthrough` / frozen baseline at matched seeds (connectome 82.8958 ± 0.0189;
  mouse 92.0696 ± 0.2624). Pure Rocket score (no post-processing).
- **connectome:** mean **82.8578 ± 0.0288** (n=3; H06 own std 0.0288), Δ = **−0.0380 pp** vs baseline
  82.8958 → below the +0.04 pp 2σ gate (a slight REGRESSION). **SCREEN FAIL.**
- **mouse:** mean **91.9902 ± 0.1978** (n=3; H06 own std 0.1978), Δ = **−0.0794 pp** vs baseline
  92.0696 → below the +0.52 pp 2σ gate (a slight REGRESSION). **SCREEN FAIL.**
- **CONFIRM-escalation check:** NOT a low-variance H02-style case. Both Δ are NEGATIVE (not positive
  sub-threshold), and H06's own std (connectome 0.0288, mouse 0.1978) ≈ the baseline noise floor on
  both datasets (0.0189 / 0.2624) → the 2σ gate is correctly specified; no escalation to CONFIRM.
- **Commands:**
  ```
  PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
  for DS in connectome mouse; do for S in 42 123 999; do
    $PY -m eval.run_variant --exp H06 --dataset $DS --seed $S --out results/ --role implement
  done; done
  $PY -m eval.aggregate --glob "results/*.json" --out experiments/log.md
  ```
- **Result file ids:** connectome —
  `20260621T102136Z-H06-connectome-s42-implement-e07005`,
  `20260621T102341Z-H06-connectome-s123-implement-e07005`,
  `20260621T102557Z-H06-connectome-s999-implement-e07005`;
  mouse —
  `20260621T102120Z-H06-mouse-s42-implement-ae6060`,
  `20260621T102123Z-H06-mouse-s123-implement-ae6060`,
  `20260621T102125Z-H06-mouse-s999-implement-ae6060`
  (smoke `20260621T102110Z-H06-mouse-s42-implement-ae6060` excluded as a duplicate of s42).
- **SCREEN verdict: FAIL on BOTH datasets** (correctly-specified gate; no CONFIRM escalation).
  The heavy-&-borderline emphasis biased the surrogate slightly away from the faithful Eq.-7
  objective (the backlog's stated risk: "changing the loss too far from Eq. 7 biases the surrogate
  away from the true metric"), nudging the metric just below baseline on both datasets rather than
  improving it. Objective/landscape reshaping via per-edge emphasis did not move the metric up.

#### Decision: **kill.** H06 SCREEN-FAILED on both datasets (connectome Δ=−0.0380 pp, mouse
Δ=−0.0794 pp), both below the 2σ gate and in fact small regressions. Correctly-specified gate
(own std ≈ noise floor on both) → no CONFIRM escalation. Direction-preserving and leakage-safe by
construction (detached `m_e>1`, only input weights + own surrogate state), so the null is a genuine
objective-axis result: re-emphasizing heavy/borderline edges around the faithful max-normalized
weighting does not raise exact feedforward weight at equal budget. Backlog status → screened.
**Knock-on for H15:** H15 (H02 basin × best objective lever {H06,H11}) is gated on an objective
lever screening positive standalone; H06 did not, so if H11 also fails to screen, H15 reduces to
H02 and should be dropped.

---

## 2026-06-21 — H11: Surrogate swap — margin-shaped (smooth-hinge) reward vs saturating sigmoid
- Hypothesis: Replacing the saturating sigmoid surrogate σ_β(Δ) with a margin-shaped surrogate
  (smooth-hinge / tanh) that KEEPS producing gradient for already-correct-but-small-margin edges
  yields higher exact feedforward weight than σ_β. Rationale: σ_β saturates — once an edge is
  comfortably feedforward its gradient → 0, so the optimizer stops WIDENING margins that protect the
  discrete order against later cyclic-β reshuffling.

#### Implementer (screen)
- **Variant** (`src/mfas/experiments/H11.py`): replicates the `run_rocket` main loop VERBATIM
  (same N(0,1) init, Adam, grad-clip=1.0, constant→exponential LR schedule, cyclic β schedule, CPU
  discrete scoring, best-by-oracle tracking, history, time-limit) and changes ONLY the per-edge
  surrogate SHAPE. `run_rocket` exposes no surrogate hook, so the loop is copied and exactly ONE line
  changed: `sig = sigmoid(β·delta)` → `r = clamp(0.5 + (β·delta)/(2·MARGIN), 0, 1)`.
- **Surrogate formula (primary arm = smooth-hinge, MARGIN = 2.0):** with `Δ = pos[v]−pos[u]` and the
  scaled margin `z = β·Δ` (β keeps EXACTLY its baseline role — the margin/temperature scale), the
  per-edge reward is the bounded smooth-hinge `r_β(Δ) = clamp(0.5 + z/(2·MARGIN), 0, 1)`. It is
  LINEAR (constant non-zero gradient `1/(2·MARGIN)`) across the correct-but-thin band `|z| ≤ MARGIN`,
  then flat at {1 feedforward, 0 feedback} outside — the decisive difference vs σ_β, whose gradient
  `σ(1−σ)` decays to ~0 almost immediately past `z=0`. Per-edge weight `hat_w = w/max(w)` is
  IDENTICAL to baseline (loss `= −Σ r_β·hat_w`), which isolates H11 (surrogate SHAPE) from H06
  (per-edge reweighting). UN-RUN arm: tanh `r=(tanh z+1)/2` — rejected as primary because tanh is an
  affine reparametrization of the sigmoid (`σ(x)=(tanh(x/2)+1)/2`) and saturates IDENTICALLY, so it
  would NOT keep gradient on correct-but-thin edges and would not test the hypothesis's mechanism.
- **Monotonicity / argmax-preservation:** `r(z)=clamp(0.5+z/(2M),0,1)` is non-decreasing in `z`
  (slope `1/(2M)>0` on the band, slope 0 on the flats) and `z=β·Δ` with `β>0` is increasing in `Δ`,
  so `r_β(Δ)` is MONOTONICALLY non-decreasing in `Δ=pos[v]−pos[u]`: making an edge more feedforward
  never decreases its reward (strictly increases it while `|z|<M`). The per-edge gradient points
  toward `pos[v]>pos[u]` (feedforward) inside the band and is zero (never inverted) outside — no edge
  is ever pushed toward feedback. A `hat_w`-weighted sum of monotone-in-Δ feedforward rewards keeps
  its optimum at "maximize feedforward weight"; the discrete oracle is tracked best-by-oracle and
  reported, exactly as baseline.
- **Anti-divergence guard:** a never-saturating pure hinge could drive `Δ→∞` and blow up positions;
  the bounded plateau (reward flats beyond `±MARGIN`) removes the incentive to grow `Δ` past the
  margin, and the unchanged grad-clip=1.0 caps step size. Smoke test (mouse s42) showed no
  NaN/collapse (92.3182%, sane).
- **Leakage-safe / target-blind:** `r_β` is a function of ONLY (i) model positions (via Δ) and
  (ii) β (a loss-shape schedule), weighted by input `hat_w`. It never reads/hardcodes/folds in the
  discrete oracle, never special-cases a dataset (same MARGIN and formula for connectome and mouse).
  Oracle used only for the baseline's best-by-oracle tracking.
- **Compute-matched:** standard knob-swap (surrogate SHAPE only); same epoch budget as baseline
  (connectome 20k, mouse 5k), same optimizer-step count; `n_epochs_done`=actual steps; comparator =
  `baseline_passthrough` / frozen baseline at matched seeds (connectome 82.8958 ± 0.0189;
  mouse 92.0696 ± 0.2624). Pure Rocket score (no post-processing).
- **connectome:** mean **82.8571 ± 0.0235** (n=3; H11 own std 0.0235), Δ = **−0.0387 pp** vs baseline
  82.8958 → below the +0.04 pp 2σ gate (a slight REGRESSION). **SCREEN FAIL.**
  (seeds: 42 → 82.8837, 123 → 82.8487, 999 → 82.8389.)
- **mouse:** mean **92.1960 ± 0.2643** (n=3; H11 own std 0.2643), Δ = **+0.1264 pp** vs baseline
  92.0696 → POSITIVE but below the +0.52 pp 2σ gate. **SCREEN FAIL.**
  (seeds: 42 → 92.3182, 123 → 91.8927, 999 → 92.3770.)
- **CONFIRM-escalation check:** NOT a low-variance H02-style case. Connectome Δ is NEGATIVE (a
  regression, not a positive sub-threshold gain), and H11's own std (connectome 0.0235, mouse 0.2643)
  ≈ the baseline noise floor on both datasets (0.0189 / 0.2624) → the 2σ gate is correctly specified;
  no escalation to CONFIRM. (The mouse +0.1264 pp is positive but ~4× under threshold and sits
  squarely inside H11's own ±0.26 pp seed scatter — not a low-variance signal.)
- **Commands:**
  ```
  PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
  for DS in connectome mouse; do for S in 42 123 999; do
    $PY -m eval.run_variant --exp H11 --dataset $DS --seed $S --out results/ --role implement
  done; done
  $PY -m eval.aggregate --glob "results/*.json" --out experiments/log.md
  ```
- **Result file ids:** connectome —
  `20260621T103328Z-H11-connectome-s42-implement-ec5418`,
  `20260621T103512Z-H11-connectome-s123-implement-ec5418`,
  `20260621T103658Z-H11-connectome-s999-implement-ec5418`;
  mouse —
  `20260621T103303Z-H11-mouse-s42-implement-29b494`,
  `20260621T103316Z-H11-mouse-s123-implement-29b494`,
  `20260621T103319Z-H11-mouse-s999-implement-29b494`.
- **SCREEN verdict: FAIL on BOTH datasets** (correctly-specified gate; no CONFIRM escalation).
  Keeping gradient on correct-but-thin edges via a linear-band smooth-hinge did not raise exact
  feedforward weight: connectome regressed slightly and mouse rose only ~¼ of its 2σ gate within
  seed noise. Consistent with the campaign's objective-axis pattern (H06 also failed): reshaping the
  per-edge surrogate around the faithful Eq.-7 objective does not move the discrete metric up at
  equal budget — the smooth-hinge's wider gradient band trades the sigmoid's faithful S-curve for a
  shape that drifts slightly off the true metric on connectome.

#### Decision: **kill.** H11 SCREEN-FAILED on both datasets (connectome Δ=−0.0387 pp regression;
mouse Δ=+0.1264 pp, ~4× under the 0.52 pp gate and within own seed noise), correctly-specified gate
(own std ≈ noise floor on both) → no CONFIRM escalation. Monotone/argmax-preserving and leakage-safe
by construction (positions + input weights + β only). **Knock-on for H15:** H15 was gated on an
objective lever (H06 or H11) screening positive standalone; BOTH H06 and H11 have now failed, so H15
reduces to H02 and should be dropped per the ideator's own gate.

---

## 2026-06-21 — H13: Mini-batch / stochastic edge subsampling per step (SGD-style Rocket)
- Hypothesis: Computing the surrogate loss on a RANDOM SUBSET of edges each step (SGD-style) injects
  useful gradient noise that escapes the surrogate plateau, matching or beating full-batch Rocket.
  Rationale: baseline Rocket is FULL-BATCH (all edges summed every step) — deterministic descent into
  the nearest basin, consistent with the observed ~82.9% / ~92.1% plateau; stochastic edge sampling is
  the textbook way to add exploration noise.

#### Implementer (screen)
- **Variant** (`src/mfas/experiments/H13.py`): replicates the `run_rocket` main loop VERBATIM (same
  N(0,1) init, Adam, grad-clip=1.0, constant→exponential LR schedule, cyclic β schedule, CPU int64/
  float64 discrete scoring, best-by-oracle tracking, history, time-limit) and adds ONLY (i) a dedicated
  batch RNG and (ii) per-step uniform edge subsampling with an unbiased rescale. `run_rocket` exposes no
  edge-set hook, so the loop is copied and only these additions made; the surrogate `σ_β` and per-edge
  weight `hat_w = w/max(w)` are UNCHANGED.
- **Sampling scheme (primary arm FRAC = 0.5):** each optimizer step, draw a UNIFORM
  without-replacement subset `S_i ⊆ E` of fixed size `m = round(FRAC·|E|)` of edge indices and form
  `loss_i = −(|E|/m)·Σ_{e∈S_i} σ_β(Δ)·hat_w`. Only the edge SET per step changes; init/Adam/grad-clip/
  LR/β/hat_w/σ_β are baseline.
- **Unbiasedness (unbiased descent direction):** for a uniform fixed-size-`m` subset, each edge is
  included with probability `m/|E|`, so the Horvitz-Thompson scaled-sum `(|E|/m)·Σ_S f(e)` is an
  UNBIASED estimator of the full-batch sum `Σ_E f(e)` (`E[(|E|/m)Σ_S f] = (|E|/m)·Σ_E (m/|E|)f = Σ_E f`).
  This holds termwise for the gradient (a finite linear combination of per-edge gradients), so the
  expected subset gradient EQUALS the full-batch gradient — an unbiased stochastic-descent direction
  with added zero-mean noise. The `|E|/m` scale (not the mean `1/m`) keeps the gradient MAGNITUDE on the
  baseline scale, so grad-clip=1.0, the LR schedule and β retain their baseline meaning and the only
  injected effect is the SGD noise.
- **Target-blindness / leakage-safe:** the subset is drawn UNIFORMLY over input edge indices `[0,|E|)`
  using a dedicated `numpy.random.RandomState(seed + 104729)` (a fixed offset so the batch stream is
  reproducible from the run seed yet independent of the init RNG). Sampling never consults the discrete
  oracle, never uses edge orientation / current positions / the target metric, and is NOT
  dataset-special-cased (same FRAC and scheme for connectome and mouse). The discrete score scored every
  `log_interval` is the EXACT full-graph oracle (never the subset); oracle used only for the baseline's
  best-by-oracle tracking.
- **Compute-matched (PROTOCOL basis = total_grad_steps):** H13 runs the SAME number of optimizer steps
  as baseline (connectome 20k, mouse 5k), each on a random subset, so `n_epochs_done` = 20000/5000
  EXACTLY matches the baseline single run → comparator = `baseline_passthrough` / frozen baseline at
  matched seeds (connectome 82.8958 ± 0.0189; mouse 92.0696 ± 0.2624). **Wall-clock note:** the
  forward/backward touches HALF the edges, so per-step gradient compute is lower, BUT the per-step
  uniform-without-replacement draw over 5.6M connectome edges (`rng.choice(replace=False)`, ~72 ms/step)
  dominates and pushes connectome wall-clock UP (~1332 s/run vs ~78 s baseline) — wall-clock is logged
  but per PROTOCOL is NOT the comparison basis (gradient steps are). NO extra "spend the saved
  wall-clock" steps were added (that would break the grad-step match).
- **Smoke test** (mouse s42): valid RocketResult, no NaN/collapse, 92.2791% (sane, near baseline).
- **connectome:** mean **82.0602 ± 0.0045** (n=3; H13 own std 0.0045), Δ = **−0.8356 pp** vs baseline
  82.8958 → far below the +0.04 pp 2σ gate (a large REGRESSION). **SCREEN FAIL.**
  (seeds: 42 → 82.0572, 123 → 82.0582, 999 → 82.0654.)
- **mouse:** mean **92.0371 ± 0.2110** (n=3; H13 own std 0.2110), Δ = **−0.0325 pp** vs baseline
  92.0696 → below the +0.52 pp 2σ gate (within seed noise). **SCREEN FAIL.**
  (seeds: 42 → 92.2791, 123 → 91.9411, 999 → 91.8911.)
- **CONFIRM-escalation check:** NOT a low-variance H02-style case. Δ is NEGATIVE on BOTH datasets (a
  regression, not a positive sub-threshold gain). On connectome H13's own std (0.0045) is actually
  BELOW the baseline noise floor (0.0189) — but the −0.84 pp gap is ~185× the std and decisively a
  regression, so there is nothing to escalate. On mouse own std (0.2110) ≈ baseline floor (0.2624) and
  Δ is negative. The screen is correctly specified; no escalation to CONFIRM. (As anticipated in the
  brief, subsampling did NOT collapse mouse variance, and on the large connectome the injected noise
  decisively hurt the discrete score rather than helping it escape the plateau.)
- **Commands:**
  ```
  PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
  for DS in connectome mouse; do for S in 42 123 999; do
    $PY -m eval.run_variant --exp H13 --dataset $DS --seed $S --out results/ --role implement
  done; done
  $PY -m eval.aggregate --glob "results/*.json" --out experiments/log.md
  ```
- **Result file ids:** connectome —
  `20260621T104432Z-H13-connectome-s42-implement-52314a`,
  `20260621T110657Z-H13-connectome-s123-implement-52314a`,
  `20260621T112912Z-H13-connectome-s999-implement-52314a`;
  mouse —
  `20260621T115137Z-H13-mouse-s42-implement-c42396`,
  `20260621T115141Z-H13-mouse-s123-implement-c42396`,
  `20260621T115146Z-H13-mouse-s999-implement-c42396`.
- **UN-RUN arms (noted, not run this screen):** FRAC = 0.25 (stronger noise, higher variance);
  weight-proportional sampling (still target-blind — uses only input `w`); an extra-steps arm spending
  the per-step wall-clock saving on more gradient steps (would break the grad-step match — out of scope
  for an equal-budget screen).
- **SCREEN verdict: FAIL on BOTH datasets** (correctly-specified gate; no CONFIRM escalation).
  Unbiased mini-batch SGD did not help Rocket escape its plateau at equal gradient steps: on connectome
  the injected noise drove a large −0.84 pp regression (the full-batch deterministic descent reaches a
  markedly better basin than its noisy estimator at the same step count), and on mouse it landed −0.03
  pp within seed noise. Consistent with the campaign's dynamics-knob pattern (H01/H03/H04): changing the
  descent dynamics — here adding stochastic gradient noise — does not lift the discrete metric; if
  anything, on the large graph it under-fits the surrogate at the fixed step budget.

#### Decision: **kill.** H13 SCREEN-FAILED on both datasets (connectome Δ=−0.8356 pp large regression;
mouse Δ=−0.0325 pp within noise), Δ negative on both → correctly-specified gate, no CONFIRM escalation.
Unbiased + leakage-safe/target-blind by construction (uniform over input edge indices via a
run-seed-derived RNG; oracle only for best-by-oracle tracking). Adds another dynamics-axis negative to
the campaign's evidence base (full-batch deterministic descent beats its stochastic estimator at equal
steps on these graphs).


---

## 2026-06-21 — H05: Optimizer swap — AdamW (decoupled weight decay) vs Adam (FALSIFIER cycle)
- Hypothesis: Replacing Adam with AdamW (a small DECOUPLED weight decay to keep the unconstrained
  positions bounded / scale-regularized) — or a sign-based optimizer (Lion) — changes the BASIN the
  optimizer reaches and improves the final exact feedforward weight. Rationale: positions are
  unconstrained and can drift to large magnitudes where σ_β saturates and the surrogate gradient
  vanishes; mild decoupled decay regularizes position scale (a scale-invariance argument the paper
  makes for *weight* normalization, §3.1.1). Optimizer choice is target-blind → leakage-safe.
- **Falsifier framing:** the campaign's evidence (H02 win = basin change; H01/H03/H04/H13 kills =
  late-dynamics / trajectory interventions all re-converge to or below Rocket's plateau) implies the
  optimizer *trajectory* does not set the plateau — the starting *basin* does. H05 swaps the single
  dynamics knob most able to reach a DIFFERENT basin (the optimizer itself + its decoupled decay). An
  honest negative strengthens that finding; a positive would overturn it.

#### Implementer (screen)
- **Variant** (`src/mfas/experiments/H05.py`): replicates the `run_rocket` main loop VERBATIM (same
  N(0,1) init + RNG seeding, grad-clip=1.0, ConstantLR(50%)→ExponentialLR(→10%) `SequentialLR`
  schedule, cyclic-β surrogate schedule `make_beta_schedule(epochs, 5)`, CPU int64/float64 discrete
  scoring, best-by-oracle tracking, history, time-limit) and changes ONLY the optimizer constructor:
  `optim.Adam([positions], lr=cfg.lr)` → `optim.AdamW([positions], lr=cfg.lr, weight_decay=1e-4)`.
  `run_rocket` exposes no optimizer hook, so the loop is copied and only that one line changed. The
  surrogate `σ_β` and per-edge weight `hat_w = w/max(w)` are UNCHANGED.
- **Optimizer arm (primary, this screen):** AdamW, `weight_decay = 1e-4`. AdamW's (β1,β2)=(0.9,0.999)
  and eps=1e-8 defaults are BIT-IDENTICAL to torch's Adam defaults, so weight_decay is the ONLY
  behavioural difference (with `weight_decay=0` AdamW reduces EXACTLY to the baseline Adam run). The
  decoupled pull per coordinate is `lr·wd = 0.05·1e-4 = 5e-6` — a gentle scale prior relative to the
  ~O(1) surrogate gradient under grad-clip=1.0. (Note: "β" here = the cyclic SURROGATE-sharpness
  schedule the paper calls β, NOT the Adam moment coefficients, which are left at defaults.)
- **Leakage-safe / target-blind:** decoupled weight decay applies `θ ← θ − lr·wd·θ` (an L2 pull of the
  positions toward 0, separate from the adaptive gradient step). It reads ONLY the positions
  themselves — never the discrete oracle, never the input weights — and is NOT dataset-special-cased
  (same `weight_decay` for connectome and mouse). The frozen oracle is consulted ONLY for the
  baseline's existing best-by-oracle tracking.
- **Compute-matched (PROTOCOL basis = total_grad_steps):** standard knob-swap (optimizer only) at the
  SAME epoch budget as baseline (connectome 20k, mouse 5k); `n_epochs_done` = 20000/5000 EXACTLY
  matches the baseline single run → comparator = `baseline_passthrough` / frozen baseline at matched
  seeds (connectome 82.8958 ± 0.0189; mouse 92.0696 ± 0.2624). PURE Rocket score (no post-processing).
- **Smoke test** (mouse s42): valid RocketResult, no NaN/collapse, 92.3610% (sane, near baseline).
- **connectome:** mean **82.8976 ± 0.0205** (n=3; H05 own std 0.0205), Δ = **+0.0018 pp** vs baseline
  82.8958 → far below the +0.04 pp 2σ gate. **SCREEN FAIL.**
  (seeds: 42 → 82.9194, 123 → 82.8946, 999 → 82.8788.)
- **mouse:** mean **92.0696 ± 0.2624** (n=3; H05 own std 0.2624), Δ = **+0.0000 pp** vs baseline
  92.0696 → far below the +0.52 pp 2σ gate (bit-for-bit the baseline mean ± std). **SCREEN FAIL.**
  (seeds: 42 → 92.3610, 123 → 91.8520, 999 → 91.9959.)
- **CONFIRM-escalation check:** NOT a low-variance H02-style case. H05's own std ≈ the baseline noise
  floor on BOTH datasets (connectome 0.0205 vs 0.0189; mouse 0.2624 vs 0.2624 — identical), so the 2σ
  gate is CORRECTLY SPECIFIED and there is no mis-specification to escalate. Δ is essentially zero on
  both (connectome +0.0018 pp ≈ 0.1σ; mouse +0.0000 pp), not a positive-but-marginal sub-threshold
  signal. No escalation to CONFIRM.
- **Commands:**
  ```
  PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
  for DS in connectome mouse; do for S in 42 123 999; do
    $PY -m eval.run_variant --exp H05 --dataset $DS --seed $S --out results/ --role implement
  done; done
  $PY -m eval.aggregate --glob "results/*.json" --out experiments/log.md
  ```
- **Result file ids:** connectome —
  `20260621T115813Z-H05-connectome-s42-implement-74a1ef`,
  `20260621T115938Z-H05-connectome-s123-implement-74a1ef`,
  `20260621T120103Z-H05-connectome-s999-implement-74a1ef`;
  mouse —
  `20260621T120229Z-H05-mouse-s42-implement-4254d9`,
  `20260621T120231Z-H05-mouse-s123-implement-4254d9`,
  `20260621T120234Z-H05-mouse-s999-implement-4254d9`.
  (The duplicate smoke-test JSON `20260621T115802Z-H05-mouse-s42-…` — bit-identical to the canonical
  seed-42 run — was removed so the results dir holds exactly the 3 canonical seeds per dataset.)
- **UN-RUN arms (noted, not run this screen):** AdamW `weight_decay = 1e-2` (a 100× stronger scale
  prior); **Lion** (sign-based) — NOT available in this environment (`torch.optim` has no Lion and no
  Lion package is installed); per the hard constraint no dependency was added, and a hand-rolled Lion
  is left as an explicitly-labelled optional follow-up, not run here.
- **SCREEN verdict: FAIL on BOTH datasets** (correctly-specified gate; no CONFIRM escalation). The
  AdamW scale prior re-converges to Rocket's exact-metric plateau (connectome +0.0018 pp, mouse
  +0.0000 pp), squarely confirming the campaign's basin-not-dynamics inference: the optimizer — the
  dynamics knob with the strongest a-priori case to reach a different basin — did not move the discrete
  metric. The honest negative STRENGTHENS the finding that the starting basin (H02), not the optimizer
  trajectory, sets the plateau.

#### Decision: **kill.** H05 SCREEN-FAILED on both datasets (connectome Δ=+0.0018 pp, mouse Δ=+0.0000
pp, both far below the 2σ gate), own std ≈ baseline noise floor on both → correctly-specified gate, no
CONFIRM escalation. Leakage-safe/target-blind by construction (decoupled decay reads only the
positions; oracle only for best-by-oracle tracking). As a deliberate falsifier of "dynamics never
matters," H05 returns a clean negative that strengthens, rather than overturns, the basin-not-dynamics
conclusion. AdamW 1e-2 and Lion remain un-run, but the primary arm is decisively unpromising.

---

## 2026-06-21 — CAMPAIGN STOP (Phase-3 Rocket-improvement loop)

**Stop criterion fired (PROTOCOL §Budget & stop-criteria):** `EARLY_EXIT_K` — **7 consecutive
non-improving cycles** after the lone H02 win (H03, H04, H09, H06, H11, H13, H05), **with no
remaining promising backlog items**. The only un-run items (H07 LR schedule, H08 β+LR joint, H10
grad-clip, H12 EMA) are all LOW-EV pure-dynamics knobs; the H05 optimizer falsifier — the strongest
member of that bucket — re-converged to the plateau, so they are predicted non-improving and are
**deferred, not falsified**. Other budgets were not binding: **9 / 30** experiments used (`MAX_EXPERIMENTS`),
well under the **12 h** `MAX_WALLCLOCK`.

**Cycles run (9), all numbers traced to `results/*.json`, one git commit each:**
| id | mechanism / axis | screen Δ (conn / mouse) | outcome |
|---|---|---|---|
| H01 | multi-start best-of-K restarts | +0.0003 / +0.0000 pp (vs multistart) | KILL |
| **H02** | **greedy-FAS warm-start init (basin)** | **+0.0448 / +0.2064 pp (CONFIRMED)** | **KEEP (win #1)** |
| H03 | sharper terminal β ramp | −0.0376 / +0.0114 pp | KILL |
| H04 | in-loop barycenter refinement | −0.0000 / +0.0000 pp (0 accepts) | KILL |
| H09 | anti-tie jitter (free-edge) | −0.0010 / +0.0000 pp (0 ties) | KILL |
| H06 | weight-aware loss reweighting | −0.0380 / −0.0794 pp | KILL |
| H11 | margin/smooth-hinge surrogate | −0.0387 / +0.1264 pp | KILL |
| H13 | stochastic edge subsampling | −0.8356 / −0.0325 pp | KILL |
| H05 | optimizer swap → AdamW (falsifier) | +0.0018 / +0.0000 pp | KILL |
- Mooted/dropped without a cycle: **H14** (jitter-on-H02 — killed by implication, H09 found 0 ties),
  **H15** (H02×objective — dropped, both H06 and H11 failed its gate). Deferred: **H07, H08, H10, H12**.

**Central finding (see findings.md #2):** across 9 cycles the ONLY lever that improved the exact
feedforward metric was H02, which changes the *starting basin* (a greedy-FAS warm-start). Every
intervention on the optimization *dynamics / trajectory* — restarts (H01), β schedule (H03), in-loop
refinement (H04), edge-subsampling noise (H13), optimizer (H05) — re-converged to (or below) Rocket's
plateau; and both *objective/landscape* reshapings (H06 reweighting, H11 surrogate shape) slightly
**regressed** connectome, indicating the faithful sigmoid surrogate at the random basin is already
near-optimal for the dynamics. Rocket's plateau is set by **where optimization starts, not how it
moves**. Integrity held throughout: `eval/frozen.sha256` matched on every cycle; no frozen file ever
modified; every variant leakage-checked (oracle used only for best-by-oracle tracking).

---

# PHASE 4 — diagnose the Rocket↔best gap & push Rocket-only quality (continuous-only)

## 2026-06-21 — Stage A diagnosis (decisive surrogate verdict + gap structure)

Full writeup: `experiments/diagnosis.md`; all numbers in `experiments/outputs/diagnosis.json`
(reproduce: `python experiments/diagnostics.py --steps 0,1,2,3,4`). New isolated analysis module
`src/mfas/analysis/gap.py` is the ONLY reader of `data/best_solution` (leakage audit clean;
frozen-guard OK; pytest 11/11).

- **Anchor.** best_solution scores **84.6147%** (35,463,823) under the frozen oracle; exact
  node-set coverage verified. Rocket-only best (H02) = 82.927%; gap ≈ **1.69 pp**.
- **VERDICT = OPTIMIZATION-GAP, not surrogate-misalignment.** Scale-fair static test
  (`gap.optimize_spacing`): best's order out-surrogates Rocket's converged solution at EVERY β
  (+238 at β=1.05 … +1153 at β=0.05). The surrogate correctly ranks best higher; the optimizer
  fails to reach it. Drift probe: started AT 84.61%, Rocket collapses to 82.94% (cyclic), 83.03%
  (constant β=1.05), 82.75% (constB+low-LR), dipping to 76–79% — i.e. **best is not a reachable or
  holdable attractor of Adam-on-sigmoid under any schedule/scale**. Deep landscape problem.
- **DIRECTION I down-ranked.** init→plateau is FLAT on connectome (82.87–82.93% across inits
  36–69%); a perfect init collapses (drift probe). Better-init-alone has a ≤0.06 pp ceiling; H02
  captured it. Mouse mildly init-sensitive (greedy best 92.48%).
- **Gap structure.** Distributed reordering: 8.5% of edges / 7.5% of weight flip direction
  (+4.60 vs −2.91 = net +1.69 pp), uniform across weight buckets, on median-degree not hub
  endpoints; Kendall-τ 0.61 (moderate). 0 exact ties; near-ties negligible (0.03% within |Δ|<1)
  → genuine misordering, not discretization. Surrogate still descending at stop but discrete
  plateau slope ≈ 0.
- **Synthetic check.** Easy planted graph: Rocket 96.56% > planted 96.10% → no gap; the connectome
  gap is a property of its hard cyclic structure (a Stage-B R-prototype needs a HARDER synthetic).
- **Selected directions (rule-based):** PRIMARY O (monotone β-continuation / basin-hopping),
  SECONDARY R (straight-through / soft-rank), DOWN-RANKED I.

## 2026-06-21 — H02 hardening (re-confirm at n=15, Welch CI)
Re-confirmed the H02 connectome win on 15 matched seeds (5 original + 10 new) vs
`baseline_passthrough` at the same seeds. **H02 = 82.92975 ± 0.00106; baseline = 82.87898 ± 0.02305;
Δ = +0.0508 pp; Welch 95% CI lower bound = +0.0391** (paired +0.0390). Far more robust than the
original n=5 thin +0.0135; H02's warm-start is near-deterministic (std 0.001). Finding #1 stands and
is hardened. Result JSONs: `results/*H02-connectome-s*-confirm-*.json` (15),
`results/*baseline_passthrough-connectome-s*-confirm-*.json` (15).

## 2026-06-21 — H16: monotone β-continuation from H02 warm-start — KILL (screen FAIL connectome)
- Hypothesis: replacing the cyclic schedule (which re-melts β→0.05 and destroys good orders, per the
  drift probe) with a single MONOTONE β rise over [0.05,1.05], from H02's greedy warm-start, lets the
  optimizer commit to/sharpen a better basin. Variant `src/mfas/experiments/H16.py` (verbatim
  run_rocket loop; only init + β-path changed). Distinct from killed H03 (random init, kept cycling,
  raised β_max).
#### Implementer (screen, n=3 seeds 42/123/999, equal budget)
  - connectome: H16 = **82.6269** (deterministic) vs H02 82.9300 (**Δ −0.3031 pp**) vs baseline
    82.8958 (**Δ −0.2688 pp**) → **screen FAIL** (regresses; worse than even random-init baseline).
  - mouse: H16 = 92.6131 vs H02 92.4793 (+0.1337) vs baseline 92.0696 (+0.5435) → improves.
  - Result JSONs: `results/*H16-{connectome,mouse}-s{42,123,999}-implement-*.json`.
  - Command: `python -m eval.run_variant --exp H16 --dataset {connectome,mouse} --seed S --role implement`.
#### Decision: **KILL** (connectome regression). Mechanism corroborates the diagnosis: the low-β start
  melts the warm-start (drift-probe mechanism) and the well-tuned cyclic baseline beats a monotone
  schedule on connectome — *how* it descends is dominated by the tuned baseline; basin still rules.

## 2026-06-21 — H19: soft-rank (DIRECTION R) — EARLY_EXIT (prototype FAIL on mouse + hard synthetic)
- Hypothesis (DIRECTION R): optimizing in RANK space (bounded, scale-free) instead of raw positions
  fixes the scale blow-up (pos std≈142 saturates σ) and reaches a better basin. Prototype-first gate:
  mouse + a HARD synthetic ONLY (O(n²) all-pairs soft rank `r_i = Σ_j σ(α(p_j−p_i))`, connectome-guarded).
- **Hard synthetic fixture (backlog H21) BUILT** (`gap.make_hard_synthetic_graph`, reference_order is
  diagnostic-only): verified optimization gap **+0.80 ± 0.21 pp** (reference 74.07% vs baseline-Rocket
  73.28%, 3 seeds) — a valid prototype proxy (the easy synthetic had no gap).
#### Implementer (prototype screen, equal compute)
  - mouse (3 seeds): H19 = 90.1263 ± 0.0000 vs baseline 92.0696 → **Δ −1.94 pp (regression)**.
  - hard synthetic (3 seeds): H19 = 68.6175 ± 0.96 vs baseline 73.2768 ± 1.19 → **Δ −4.66 pp**; soft-rank
    stalls EXACTLY at the greedy warm-start (68.6175) for ALL α∈{2,4,8,16,32} — normalized rank gaps are
    O(1/n) so σ(β·gap) gradients are too flat to move positions; the optimizer never improves the init.
  - Variant `src/mfas/experiments/H19.py`; repro scripts `experiments/protoR_{tune_hardsynth,softrank_synth}.py`;
    mouse JSONs `results/*H19-mouse-s{42,123,999}-implement-*.json`.
#### Decision: **EARLY_EXIT DIRECTION R**. Soft-rank (the highest-ceiling R lever, aimed at the named
  scale-saturation mechanism) is a large robust regression at prototype scale → earns no connectome
  compute and falsifies the rank-space hypothesis: scale-saturation is not a bottleneck a rank
  reparametrization fixes.

---

## 2026-06-21 — STAGE B CAMPAIGN STOP (Phase-4 continuous-only push)

**Stop criterion fired (PROTOCOL §Budget):** `EARLY_EXIT` — both pre-registered directions' highest-EV
levers failed, with the diagnosis predicting the remainder non-improving.

**Cycles run (Phase 4):**
| id | direction / mechanism | scale | result |
|---|---|---|---|
| (diagnosis) | decisive surrogate + gap structure | connectome+mouse+synth | OPTIMIZATION-GAP verdict |
| H02-harden | re-confirm warm-start at n=15 | connectome | Δ+0.0508, CI_low +0.0391 (holds) |
| H16 | O: monotone β-continuation from warm-start | connectome+mouse | KILL (conn −0.27 vs base) |
| H21 | hard-synthetic gap fixture | synthetic | BUILT (+0.80 pp gap) |
| H19 | R: soft-rank (rank-space) | mouse+hard-synth | EARLY_EXIT (regress both) |

**Deferred-by-evidence (NOT run), with reasoning:**
- **H17 (O, basin-hopping / parallel tempering):** predicted non-improving — the drift probe showed
  re-optimization flows back to the ~82.9% basin from ANY init (incl. the 84.6% optimum), and naive
  restarts already KILLED in Phase-3 (H01, +0.0003). Hopping samples ~82.9% basins.
- **H18 (R, straight-through estimator):** STE's backward IS the sigmoid-surrogate gradient, so its
  trajectory ≈ baseline Rocket (which already best-by-oracle tracks the discrete score) → ~no change.
- **H20 (R, Gumbel-Sinkhorn):** O(n²), note-only; soft-rank (a cheaper member of the same relaxation
  class) already failed at prototype scale.

**Central Phase-4 conclusion (see findings.md #3):** the Rocket↔best gap is a genuine OPTIMIZATION-GAP
(the surrogate correctly ranks the near-optimal order higher) but is **not closable by the Rocket class
of continuous optimization**: better init washes out (flat init→plateau; a perfect init collapses under
gradient flow), schedule changes regress (H16), and rank-space reparametrization stalls (H19). The
~1.69 pp is a distributed reordering with no tie-slack → **largely irreducible to continuous methods**,
explaining why the paper's discrete Crane phase is what extends quality past Rocket's plateau. The
Rocket-only best remains **H02 = 82.93% connectome** (hardened) / **92.48% mouse**. Integrity held:
frozen-guard OK every cycle, no frozen file modified, leakage audit clean, `best_solution` confined to
`mfas.analysis`.

#### Critic verdict (Phase-4)

Adjudicated commits 9cc9c34, 3275f5c, 6eb8684 on `phase4-campaign`. Phase produced NO new promoted
win (correct); it hardened H02 and reached a negative/structural conclusion (finding #3). Read-only
re-derivation from the JSONs + grep/pytest.

1. **best_solution leakage isolation — PASS.** `grep` over `src/mfas/experiments/*.py` and
   `src/mfas/baseline/*.py`: no import of `mfas.analysis`/`gap`, no read of `data/best_solution`,
   no reference to `reference_order`/`planted`/`drift_probe`/`make_hard_synthetic`/`_build_reference`.
   The only matches in the optimization path are docstring assertions in H16/H19 ("never
   best_solution"). `best_solution` is read ONLY by `src/mfas/analysis/gap.py:load_best_solution`;
   the only importers of `gap` are `experiments/diagnostics.py` and the two `protoR_*` scripts plus
   docs — all diagnostic drivers, none on any variant's run/loss/init. The hard-synthetic
   `reference_order` is built in `gap.py` (`_build_reference`/`_refine_reference_order`, may consult
   the oracle) and is returned for diagnostic comparison only — H19 calls `make_hard_synthetic_graph`
   only via the proto scripts, never inside `H19.run` (which takes a plain `GraphData`). Drift-probe /
   best-seeing outputs go to `experiments/outputs/` only; `grep` of `results/` shows no
   drift/best/proto artifact and no proto script writes to `results/` (verified: "NO writes").
2. **Frozen integrity — PASS.** `git diff --stat 5615dff..HEAD` touches none of `metrics.py`,
   `harness.py`, `aggregate.py`, `test_metrics.py`; working tree clean for those four;
   `verify_frozen_manifest()` → FROZEN OK; `pytest tests/test_metrics.py` 8/8 green.
3. **H02 hardening CI — PASS.** 15 matched connectome seeds present for BOTH H02 and
   `baseline_passthrough` (identical seed set {7,42,123,999,1414,1618,1732,2236,2718,5005,6004,7003,
   8002,9001,31415}; all `role=confirm`, config_hash 059689 / f8cb3c, git_commit f36e0284 + 9cc9c34).
   Re-derived: H02 = 82.9297 ± 0.0011, baseline = 82.8790 ± 0.0231, **Δ = +0.0508 pp**, **Welch 95%
   CI low = +0.0391**, paired +0.0390 (t-crit df14: +0.0379 — still >0). Matches finding #1 exactly;
   no seed mismatch, no cherry-pick. (Note: the claimed +0.0391 is the Welch bound, NOT the screen
   `std_base·√(2/n)` formula, which would give +0.0343 — the text correctly labels it "Welch", which
   is the more conservative/honest choice given H02's near-zero variance.)
4. **Compute-fairness — PASS.** H16 connectome/mouse logged `total_grad_steps`=20000/5000 =
   baseline budget; verbatim `run_rocket` loop, only init+β-path changed; deterministic across seeds.
   H19 mouse `total_grad_steps`=5000 = baseline budget; only the surrogate parametrisation (soft-rank)
   differs; O(n²) connectome guard enforced. Comparators consistent (H16/H19 screened vs the 3-seed
   implement baseline mean 92.0696 and vs H02). Equal-epoch, fair.
5. **No overclaim — PASS (with verified hedging).** "OPTIMIZATION-GAP not misalignment" is supported
   by the *scale-fair* static test (`optimize_spacing` optimises monotone spacing AND a free global
   scale for best's order, so best is not under-powered vs Rocket's pos_std≈142): best out-surrogates
   Rocket at every β (+238…+1153, diagnosis.json `static_surrogate`), corroborated by the drift probe
   (84.61→82.75–83.03 under every schedule/scale). Not a scale artifact. "Irreducible to continuous
   methods" is appropriately hedged: only DIRECTION O top lever (H16) and DIRECTION R top lever (H19)
   were run; H17/H18/H20 explicitly deferred-by-evidence (not falsified); mouse has no best_solution
   so the decisive/gap steps are connectome-only; rests on one hard synthetic (with a verified +0.80 pp
   planted gap, so the levers failed where a gap demonstrably exists). Findings #3 states all of these
   caveats. The H16 mouse result (+0.13 vs H02) is honestly reported and correctly does NOT override
   the connectome KILL. No claim exceeds the evidence.
6. **Reproducibility — PASS.** Cited files exist and re-derive: `experiments/outputs/diagnosis.json`
   (all #3 probe numbers match), 15+15 H02/baseline confirm JSONs, H16 (3+3) and H19 (3) JSONs with
   matching pct/grad-steps. `python experiments/diagnostics.py --steps 0,1,2,3,4` and the
   `eval.run_variant` commands are present and consistent with the JSON `experiment_id`/`config_hash`.

**Recommendation: keep.** Phase-4 is sound: H02 hardening is correctly re-derived (Δ+0.0508,
CI_low +0.0391 @ n=15), the negative/structural conclusion (finding #3) is scale-fair, not
overstated, and properly scoped; integrity and leakage isolation held throughout. No new win is
claimed, which is the honest outcome. No required fixes.

---

## 2026-06-22 — H22: bounded-window discrete local search (DIRECTION D) — KILLED at sizing gate

- Hypothesis (DIRECTION D, NEW): the corollary of finding #3 — since the gap is irreducible to
  *continuous* methods (the smooth gradient is a coarse majority-vote blind to cyclic-core
  reorderings), a cheap **discrete** bounded-window sift/re-insertion post-phase might recover a
  fraction of the ~1.69 pp by acting directly on that discrete structure. Sanctioned by parent
  CLAUDE.md goal #1; distinct from killed H04 (barycenter = the same coarse signal, 0 moves accepted).
- **Sizing gate FIRST (like H09):** `experiments/size_localsearch.py` →
  `experiments/outputs/localsearch_sizing.json`. Diagnostic-only; reads `best_solution` ONLY via the
  privileged `mfas.analysis.gap`; writes nothing to `results/`.
- **Cross-check PASSED (validates the measurement):** net gap **+1.6874 pp** = gain +4.5975 − lose
  +2.9102, reproducing the Stage-A diagnosis; H02 order re-scores **82.9273%** (rank-faithful, 0 ties).
- **DECISIVE — the gap is LONG-RANGE / GLOBAL, not local.** The recoverable
  (feedback→feedforward) weight has rank-distance percentiles **p25=8,290 / p50=22,580 / p90=87,497**
  in Rocket's order (n=136,648): the median recoverable edge needs a node to travel ~22.6k ranks. Net
  gap recoverable within any tractable window is **≤0**: W=100 → net −0.0100 pp (gain only 0.24% of
  total gain), W=1000 → −0.1696, W=5000 → −0.4045; net is positive only for W≤10 (+0.0003). Measure-1
  ceiling agrees (feedback pool within W=100 = 0.027 pp). Within a window the broken `lose` edges
  outweigh the `gain` → like H04, there is no improving local move. Mouse is uninformative (n=148, the
  whole graph is "local"; no mouse best_solution).
#### Decision: **KILL (by sizing).** A bounded-window single-node local search cannot close the gap;
  compute conserved (no variant built), exactly the H09 pattern. **Strengthens finding #3:** the
  residual is irreducible not only to continuous methods but to *bounded-local* discrete refinement —
  it is a global reordering that requires global discrete optimization (the paper's Crane MIP).
  Reproduce: `python experiments/size_localsearch.py`.
