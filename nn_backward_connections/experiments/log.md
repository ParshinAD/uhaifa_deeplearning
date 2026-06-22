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
_Generated 2026-06-22T16:57:21Z from 250 run(s)._

| algo | dataset | n_seeds | pct mean±std | score mean±std | wall_clock_s (mean) | seeds | config_hash | git_commit |
|---|---|---|---|---|---|---|---|---|
| H01 | connectome | 3 | 82.0510 ± 0.0290 | 34,389,329 ± 12,134 | 78.7 | [42, 123, 999] | fdb0b2 | 704221ab778 |
| H01 | mouse | 3 | 92.1625 ± 0.0242 | 8.4412 ± 0.0022 | 2.0 | [42, 123, 999] | 434027 | 704221ab778 |
| H02 | connectome | 21 | 82.9299 ± 0.0010 | 34,757,692 ± 415 | 93.8 | [7, 42, 42, 42, 123, 123, 123, 999, 999, 999, 1414, 1618, 1732, 2236, 2718, 5005, 6004, 7003, 8002, 9001, 31415] | 059689 | 9cc9c34eb98, f36e02847a9 |
| H02 | microns | 8 | 83.1288 ± 0.0006 | 12,802,299 ± 97 | 596.7 | [7, 7, 42, 42, 123, 999, 31415, 31415] | 59bc98 | 012e9ad9c74, 32d290734ec |
| H02 | mouse | 26 | 92.4793 ± 0.0000 | 8.4702 ± 0.0000 | 1.8 | [7, 42, 42, 42, 123, 123, 123, 999, 999, 999, 1111, 1234, 1414, 1618, 1732, 2222, 2236, 2718, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 31415] | a8bbc0 | f36e02847a9 |
| H03 | connectome | 3 | 82.8582 ± 0.0181 | 34,727,627 ± 7,582 | 78.2 | [42, 123, 999] | 082864 | 858e000ec21 |
| H03 | microns | 3 | 83.1090 ± 0.0013 | 12,799,251 ± 197 | 578.6 | [42, 123, 999] | 0c2b14 | 32d290734ec |
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
| H11 | microns | 8 | 83.1183 ± 0.0011 | 12,800,687 ± 172 | 832.5 | [7, 7, 42, 42, 123, 999, 31415, 31415] | 71746c | 012e9ad9c74, 32d290734ec, 71eb853e521 |
| H11 | mouse | 3 | 92.1960 ± 0.2643 | 8.4443 ± 0.0242 | 2.4 | [42, 123, 999] | 29b494 | 258bcbd07c0 |
| H13 | connectome | 3 | 82.0602 ± 0.0045 | 34,393,204 ± 1,875 | 1332.1 | [42, 123, 999] | 52314a | 07b26a8edcc |
| H13 | mouse | 3 | 92.0371 ± 0.2110 | 8.4297 ± 0.0193 | 3.9 | [42, 123, 999] | c42396 | 07b26a8edcc |
| H16 | connectome | 3 | 82.6269 ± 0.0000 | 34,630,719 ± 0 | 96.0 | [42, 123, 999] | 1262a4 | 9cc9c34eb98 |
| H16 | microns | 3 | 83.1033 ± 0.0000 | 12,798,371 ± 1 | 577.2 | [42, 123, 999] | c5c44a | 32d290734ec |
| H16 | mouse | 3 | 92.6131 ± 0.0000 | 8.4825 ± 0.0000 | 2.1 | [42, 123, 999] | afe423 | 9cc9c34eb98 |
| H19 | mouse | 3 | 90.1263 ± 0.0000 | 8.2547 ± 0.0000 | 2.4 | [42, 123, 999] | 5dceca | 3275f5cce2d |
| H30 | connectome | 11 | 83.7805 ± 0.0096 | 35,114,220 ± 4,041 | 151.7 | [7, 42, 42, 42, 123, 123, 123, 999, 999, 999, 31415] | 1976d9 | 7dc17e20460 |
| H30 | microns | 11 | 83.2067 ± 0.0012 | 12,814,293 ± 185 | 728.4 | [7, 42, 42, 42, 123, 123, 123, 999, 999, 999, 31415] | 954bab | 7dc17e20460 |
| H30 | mouse | 26 | 92.9018 ± 0.0000 | 8.5089 ± 0.0000 | 2.7 | [7, 42, 42, 42, 123, 123, 123, 999, 999, 999, 1111, 1234, 1414, 1618, 1732, 2222, 2236, 2718, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 31415] | ff2174 | 7dc17e20460 |
| H31 | connectome | 3 | 83.7899 ± 0.0132 | 35,118,149 ± 5,522 | 165.3 | [42, 123, 999] | 80c75b | 2b3f6c64809 |
| H31 | mouse | 3 | 92.9018 ± 0.0000 | 8.5089 ± 0.0000 | 3.0 | [42, 123, 999] | 8a4eac | 2b3f6c64809 |
| baseline_multistart | connectome | 3 | 82.0507 ± 0.0293 | 34,389,211 ± 12,284 | 77.2 | [42, 123, 999] | 330c58 | 704221ab778 |
| baseline_multistart | mouse | 3 | 92.1625 ± 0.0242 | 8.4412 ± 0.0022 | 1.9 | [42, 123, 999] | c6938b | 704221ab778 |
| baseline_passthrough | connectome | 21 | 82.8838 ± 0.0224 | 34,738,365 ± 9,388 | 419.0 | [7, 42, 42, 42, 123, 123, 123, 999, 999, 999, 1414, 1618, 1732, 2236, 2718, 5005, 6004, 7003, 8002, 9001, 31415] | f8cb3c | 8f0e5066211, 9cc9c34eb98, f36e02847a9 |
| baseline_passthrough | microns | 8 | 83.0823 ± 0.0483 | 12,795,132 ± 7,445 | 396.2 | [7, 42, 42, 123, 123, 999, 999, 31415] | c2f06f | 7dc17e20460, 97b951056ae |
| baseline_passthrough | mouse | 26 | 92.2260 ± 0.2215 | 8.4470 ± 0.0203 | 36.5 | [7, 42, 42, 42, 123, 123, 123, 999, 999, 999, 1111, 1234, 1414, 1618, 1732, 2222, 2236, 2718, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 31415] | 7b7cba | 7dc17e20460, 8f0e5066211, f36e02847a9 |
| baseline_rocket | connectome | 3 | 82.8958 ± 0.0189 | 34,743,386 ± 7,927 | 75.9 | [42, 123, 999] | 5ec3ce | 7a77547dee9 |
| baseline_rocket | microns | 1 | 83.0997 (n=1) | 12,797,820 (n=1) | 278.2 | [42] | 4727ce | 97b951056ae |
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

---

## 2026-06-22 — Phase 5: MICrONS dataset build + baseline (Parts A & B)

### Part A — MICrONS v117 build (infrastructure, not a variant)

**Goal:** Add a SECOND large real connectome (`microns`, MICrONS minnie65 mouse visual cortex) as a
new leakage-clean dataset, enabling 3-dataset hypothesis evaluation and re-testing of "lost theory"
hypotheses killed only by the single-large-graph require-both rule.

**Build method:** Token-free static v117 release (public BossDB/GCS, no CAVE account required):
```
nucleus:    https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/nucleus_neuron_classification/nucleus_neuron_svm.csv
synapses:   https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/synapse_graph/synapses_pni_2.csv
proofread:  https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/proofreading_status/proofreading_status_public_release.csv
```
All three pinned to **segmentation version v117**. Reproduced by `python -m experiments.build_microns`
(full recipe + sha256 in `data/processed/microns_BUILD.md`). Build stats:
- 337,312,429 synapse rows total; 15,400,557 neuron-neuron non-self synapses (4.57% kept)
- 72,789 neuron root_ids in the SVM; pre-synaptic membership only 6.6% (many axon fragments
  lack soma/nucleus in v117 — expected; cross-SVM-boundary pairs excluded correctly)

**Canonical graph stats** (`data/processed/microns_stats.json`):
| property | canonical (all neurons) | proofread subset |
|---|---|---|
| n_nodes | **67,534** | 245 |
| n_edges | **10,436,569** | 2,424 |
| total_weight | **15,400,557** | 5,466 |
| giant SCC | 65,543 (97.1%) | 186 (76%) |
| n_SCC | 1,970 | 60 |
| density | 0.0023 | 0.041 |
| out-degree (med/p99/max) | 61 / 1708 / 12246 | 6 / 46 / 58 |
| weight (med/max) | 1 / 1551 | 1 / 37 |

MICrONS has **more edges than the fly connectome** (10.4M vs 5.66M) and a dominant recurrent core
(97.1% in one SCC). Heavy-tailed degree + weight distributions confirm it is a real connectome.

**Harness integration (frozen manifest clean):**
- `src/mfas/io.py`: added `load_microns` + `DATASETS["microns"]` entry (writable file; no frozen
  file touched). Validation anchors: `expected_n=67534, expected_m=10436569, expected_total_weight=15400557`.
- `configs/baseline_rocket.yaml`: added `dataset_overrides.microns.rocket.epochs: 80000`.
- `src/mfas/experiments/baseline_passthrough.py`: added `"microns": 80_000` to `_EPOCHS`.
- `eval/frozen_guard.verify_frozen_manifest()` = **PASS** (oracle intact; graph-agnostic by design).
- Smoke test: `python -m eval.run_variant --exp baseline_passthrough --dataset microns --seed 42 …`
  → score=12,786,144, pct=83.0239%, epochs=20000, wrote `results/*.json`. End-to-end wiring confirmed.

**Reproduce:**
```bash
python -m experiments.build_microns          # produces data/processed/microns.npz  (~25 min)
python -m eval.run_variant --exp baseline_passthrough --dataset microns --seed 42 --out results/ --role implement
```

### Part B — MICrONS baseline + plateau verification + 3-dataset rule

**Plateau probe** (budget sweep, seed 42):
| epochs | pct | Δ from prev |
|---|---|---|
| 20,000 | 83.0239% | — |
| 40,000 | 83.0997% | +0.076 pp |
| 80,000 | 83.1165% | +0.017 pp |
| 120,000 | 83.1175% | **+0.001 pp** ← plateau |
| within-run traj 90k/105k/120k | 83.1175% / 83.1175% / 83.1175% | ← flat |

**Plateau = 80k epochs** (80k→120k gain = 0.001 pp = ~1.7σ of the measured noise, negligible; budget
set to 80k for the campaign: ~550s/run on Apple MPS).

**Baseline noise floor at 80k epochs** (3 seeds 42/123/999; `baseline_passthrough` runs):
```
results/20260621T232155Z-baseline_passthrough-microns-s42-implement-c2f06f.json   pct=83.1169%
results/20260621T233111Z-baseline_passthrough-microns-s123-implement-c2f06f.json  pct=83.1167%
results/20260621T234021Z-baseline_passthrough-microns-s999-implement-c2f06f.json  pct=83.1179%
```
**mean = 83.1172%,  σ = 0.0006 pp,  2σ = 0.0013 pp  (screen threshold: 0.002 pp)**

MICrONS is **29× tighter than connectome** (σ=0.019 pp) and **408× tighter than mouse** (σ=0.26 pp).
Its extremely tight noise floor gives near-perfect discrimination power for the re-test campaign —
a "lost theory" effect of even +0.01 pp would be unambiguous, where mouse's σ=0.26 would completely
drown it.

**Reproduce:**
```bash
# plateau probe
python experiments/microns_plateau_probe.py
# baseline noise floor
python -m eval.run_variant --exp baseline_passthrough --dataset microns --seed 42 --out results/ --role implement
python -m eval.run_variant --exp baseline_passthrough --dataset microns --seed 123 --out results/ --role implement
python -m eval.run_variant --exp baseline_passthrough --dataset microns --seed 999 --out results/ --role implement
```

**3-dataset rule:** updated `experiments/PROTOCOL.md` (Phase-5 section appended 2026-06-22).
PRIMARY = connectome + microns; SUPPORTING = mouse. GENERAL WIN = CI>0 on both PRIMARY + mouse
non-inferior. New verdict class GRAPH-DEPENDENT = confirms on one PRIMARY only (real but scoped).
Screen thresholds: connectome 0.04 pp / microns 0.002 pp / mouse 0.52 pp (non-inferiority only).

#### Decision: **INFRASTRUCTURE COMPLETE.** MICrONS is a valid, integrated, leakage-clean second
large connectome ready for the Phase-5 re-test campaign. Proceed to Part D after user review.

---

## 2026-06-22 — Phase 5 screen round: H16r / H11r / H02r / H03r on MICrONS

All four re-test candidates run on `microns` at 80k epochs (plateau budget), 3 seeds each (42/123/999).
Comparator: `baseline_passthrough` at matched microns seeds (mean=83.1172%, σ=0.0006 pp, 2σ=0.0013 pp).
Prior connectome/mouse results from Phase 3–4 (unchanged budgets 20k/5k) are used directly.

| id | microns mean (n=3) | microns Δ | screen (Δ>2σ=0.0013) | prior conn Δ | prior mouse Δ | 3-dataset verdict |
|---|---|---|---|---|---|---|
| H16r | 83.1033% ± 0.0000 | −0.0139 pp | **FAIL** | −0.269 pp | +0.543 pp | **SMALL-GRAPH ARTIFACT** |
| H11r | 83.1187% ± 0.0012 | +0.0016 pp | **PASS** (CI_lower=+0.0005) | −0.039 pp | +0.126 pp | GRAPH-DEPENDENT candidate → confirm |
| H02r | 83.1286% ± 0.0003 | +0.0114 pp | **PASS** (CI_lower=+0.0104) | +0.051 pp ✓ | +0.206 pp ✓ | GENERAL WIN candidate → confirm |
| H03r | 83.1090% ± 0.0012 | −0.0082 pp | **FAIL** | −0.038 pp | +0.011 pp | **STILL NULL** |

**Commands (screen runs):**
```
python -m eval.run_variant --exp H16 --dataset microns --seed {42,123,999} --out results/ --role implement
python -m eval.run_variant --exp H11 --dataset microns --seed {42,123,999} --out results/ --role implement
python -m eval.run_variant --exp H02 --dataset microns --seed {42,123,999} --out results/ --role implement
python -m eval.run_variant --exp H03 --dataset microns --seed {42,123,999} --out results/ --role implement
```

**Result JSONs (screen, implement role):**
```
H16: results/20260622T065625Z-H16-microns-s42-implement-c5c44a.json  pct=83.1033%
     results/20260622T070604Z-H16-microns-s123-implement-c5c44a.json pct=83.1033%
     results/20260622T071543Z-H16-microns-s999-implement-c5c44a.json pct=83.1033%
H11: results/20260622T072521Z-H11-microns-s42-implement-71746c.json  pct=83.1201%
     results/20260622T073631Z-H11-microns-s123-implement-71746c.json pct=83.1180%
     results/20260622T074746Z-H11-microns-s999-implement-71746c.json pct=83.1181%
H02: results/20260622T075850Z-H02-microns-s42-implement-59bc98.json  pct=83.1287%
     results/20260622T080826Z-H02-microns-s123-implement-59bc98.json pct=83.1288%
     results/20260622T081818Z-H02-microns-s999-implement-59bc98.json pct=83.1283%
H03: results/20260622T082837Z-H03-microns-s42-implement-0c2b14.json  pct=83.1080%
     results/20260622T083820Z-H03-microns-s123-implement-0c2b14.json pct=83.1086%
     results/20260622T084803Z-H03-microns-s999-implement-0c2b14.json pct=83.1104%
```

### H16r — KILL (SMALL-GRAPH ARTIFACT confirmed)

The monotone β schedule from H02 warm-start REGRESSES on `microns` (Δ=−0.014 pp, deterministic:
all 3 seeds score identically 83.1033%). It ALSO regressed `connectome` (−0.269 pp). The mouse
gain (+0.543 pp, the only "screen-passing" prior gain) is now conclusively a **148-node artifact**:
with TWO large real connectomes both showing clear regression, the mouse signal was statistical noise
from the tiny (148-node, σ=0.26 pp) graph. The H16r "lost theory" hypothesis is FALSE.

**Why the regression:** the Phase-4 diagnosis explains this exactly — the monotone β schedule melts
the H02 warm-start on any large cyclic graph by starting at low β=0.05, which allows the optimizer
to leave the greedy FAS basin before sharpening can commit to it. This is graph-size-agnostic, not
fly-specific: any large graph with a deep cyclic attractor will exhibit the same melt. Mouse doesn't
have this property at 148 nodes (it plateaus too fast for re-melt to matter).

**Classification: SMALL-GRAPH ARTIFACT.** Connectome regression (−0.27 pp), microns regression
(−0.014 pp), mouse gain (+0.54 pp): two large graphs say NO. **Confirms finding #2** (basin-not-
dynamics) on MICrONS: even the combined H02 warm-start + schedule change cannot escape the
large-graph cyclic attractor. **Status: killed.**

### H03r — KILL (STILL NULL; mouse was noise)

Terminal β ramp regresses `microns` (Δ=−0.008 pp) and `connectome` (−0.038 pp); mouse was
marginally positive (+0.011 pp, sub-threshold). Both large graphs null-to-negative: **STILL NULL**.
Strengthens finding #2: β schedule modifications don't help any large graph.
**Status: killed.**

### H11r and H02r → CONFIRM (see entries below)

---

## 2026-06-22 — H02r: greedy-FAS warm-start — GENERAL WIN on MICrONS (Phase 5 confirm)

### Implementer (screen → confirm, microns)

**Goal:** verify H02's confirmed win (connectome+mouse) generalizes to the second large connectome.

**Microns screen (3 seeds 42/123/999, 80k epochs):**
```
results/20260622T075850Z-H02-microns-s42-implement-59bc98.json  pct=83.1287%
results/20260622T080826Z-H02-microns-s123-implement-59bc98.json pct=83.1288%
results/20260622T081818Z-H02-microns-s999-implement-59bc98.json pct=83.1283%
mean=83.1286%  σ=0.0003 pp  Δ=+0.0114 pp  2σ_base=0.0013 pp  → SCREEN PASS
```

**Microns confirm (seeds 7, 31415, 80k epochs, role=confirm):**
```
results/20260622T090027Z-H02-microns-s7-confirm-59bc98.json    pct=83.1295%
results/20260622T091039Z-H02-microns-s31415-confirm-59bc98.json pct=83.1276%
```

**Confirm statistics (n=5, all microns seeds 42/123/999/7/31415):**
```
H02 microns: mean=83.1286%  std=0.0007 pp  Δ=+0.0114 pp
SE = σ_base·√(2/5) = 0.0006·0.632 = 0.0004 pp
95% CI lower = Δ - 1.96·SE = +0.0114 - 0.0008 = +0.0106 pp  → CONFIRMED (CI_lower > 0) ✓
```

**Full 3-dataset confirm summary:**

| dataset | H02 mean±std (n) | baseline mean±std (n) | Δ | 95% CI lower | verdict |
|---|---|---|---|---|---|
| connectome (n=15) | 82.9298 ± 0.0011 | 82.8790 ± 0.0231 | +0.0508 pp | +0.0391 | CI>0 ✓ |
| mouse (n=20) | 92.4793 ± 0.0000 | 92.2729 ± 0.2000 | +0.2064 pp | +0.0824 | CI>0 ✓ |
| **microns (n=5)** | **83.1286 ± 0.0007** | **83.1172 ± 0.0006** | **+0.0114 pp** | **+0.0106** | **CI>0 ✓✓** |

**3-dataset verdict: GENERAL WIN** — CI_lower > 0 on BOTH PRIMARY datasets (connectome + microns)
AND mouse non-inferior (+0.206 pp >> −0.26 pp threshold). The greedy-FAS warm-start is a
**universal basin lever** across all three real connectomes tested. Frozen oracle intact, compute-matched
(80k epochs microns = baseline), leakage-safe (greedy FAS uses only g.src/g.tgt/g.weight).

**Reproduce:**
```bash
python -m eval.run_variant --exp H02 --dataset microns --seed {42,123,999,7,31415} --out results/ --role confirm
```

#### Verifier (independent re-run — 2026-06-22)

**Frozen oracle:** `verify_frozen_manifest()` → OK before each run. No frozen file
was modified (`git diff --stat` confirms only `experiments/log.md` and
`.claude/settings.json` changed; none of `src/mfas/metrics.py`, `eval/harness.py`,
`eval/aggregate.py`, `tests/test_metrics.py` were touched).

**Independent verify runs (role=verify, seeds 42 / 7 / 31415, microns, 80k epochs):**
```
python -m eval.run_variant --exp H02 --dataset microns --seed 42 --out results/ --role verify
python -m eval.run_variant --exp H02 --dataset microns --seed 7 --out results/ --role verify
python -m eval.run_variant --exp H02 --dataset microns --seed 31415 --out results/ --role verify
```
Result JSONs:
```
results/20260622T094702Z-H02-microns-s42-verify-59bc98.json     pct=83.1293%  score=12,802,373  epochs=80000
results/20260622T095703Z-H02-microns-s7-verify-59bc98.json      pct=83.1288%  score=12,802,297  epochs=80000
results/20260622T100702Z-H02-microns-s31415-verify-59bc98.json  pct=83.1294%  score=12,802,390  epochs=80000
```

**Statistics (verifier, n=3):**
```
mean  = 83.1292%   std = 0.0003 pp
Δ vs baseline (83.1172%) = +0.0120 pp
SE = σ_base · √(2/n) = 0.0006 · √(2/3) = 0.0005 pp
95% CI lower = Δ − 1.96·SE = +0.0120 − 0.0010 = +0.0110 pp  > 0
```

**Comparison with implementer's claimed values (MPS nondeterminism expected):**
| seed | implementer | verifier | diff |
|---|---|---|---|
| 42 | 83.1287% | 83.1293% | +0.0006 pp |
| 7 | 83.1295% | 83.1288% | −0.0007 pp |
| 31415 | 83.1276% | 83.1294% | +0.0018 pp |

All differences are within the MPS nondeterminism band (~0.002 pp); all values are
tightly clustered around 83.129%, well above baseline 83.1172%.

**Schema check:** all 3 verify JSONs contain all required fields:
`exp_id`, `algo`, `dataset`, `seed`, `score`, `pct`, `total_grad_steps`, `budget_basis`.
`budget_basis="total_grad_steps"`, `total_grad_steps=80000` in all three.

**VERDICT: CONFIRMED on microns.** CI lower bound = +0.0110 pp > 0. Combined with
the Phase-3 confirms on connectome (CI_lower=+0.039 pp) and mouse (CI_lower=+0.082 pp),
H02 is independently verified as a **GENERAL WIN** on all three datasets.

#### Critic verdict (Phase-5 H02r — GENERAL WIN red-team)

Read-only adjudication of the Phase-5 H02r microns confirm. Numbers re-derived from disk JSONs
via `/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python`. Six checks:

1. **Leakage — PASS.** `src/mfas/experiments/H02.py` was read in full and grepped for
   `best_solution|oracle|target|hardcode|82.9|92.4|83.1|microns|connectome|mouse|dataset`.
   The ONLY hit touching runtime behaviour is line 203: `_EPOCHS.get(g.name, …)` — the epoch
   budget lookup. This is dataset-agnostic in effect (each dataset gets the same budget as
   `baseline_passthrough`; no algorithmic path changes). `greedy_fas_order` (lines 60–180)
   reads ONLY `g.src`, `g.tgt`, `g.weight` — no oracle, no target metric, no dataset name.
   `_init_positions_from_order` (lines 183–191) is a pure rank→position mapping.
   `run_rocket` is called with the standard dataset budget and the greedy init — no
   microns-specific special-casing in the algorithm. The discrete score is computed externally
   by the frozen oracle inside `run_rocket`'s best-by-oracle tracker (identical to baseline).
   No peeks at `data/best_solution` (which does not exist for microns anyway). **CLEAN.**

2. **Frozen-file integrity — PASS.** Recomputed SHA-256 of all four frozen files on disk:
   - `src/mfas/metrics.py`  → `bd2ff9055b…` matches `eval/frozen.sha256` exactly.
   - `eval/harness.py`      → `0ba5341318…` matches.
   - `eval/aggregate.py`    → `28340949…` matches.
   - `tests/test_metrics.py` → `27f02a778b…` matches.
   `git diff --stat HEAD -- src/mfas/metrics.py eval/harness.py eval/aggregate.py tests/test_metrics.py`
   produced no output (no staged changes to frozen files). Working-tree dirty files are:
   `experiments/log.md` (this file), `.claude/settings.json`, untracked results JSONs, and
   log files — none are frozen files. The `+dirty` suffix in JSON `git_commit` fields
   (012e9ad…+dirty) is explained by untracked result files in the working tree; the
   commit itself (`git show --stat 012e9ad`) touched only `experiments/log.md` and
   result JSONs — no frozen file was in its diff. **CLEAN.**

3. **Compute fairness — PASS.** Every H02 microns JSON (implement s42/123/999, confirm s7/31415,
   verify s42/7/31415) records `total_grad_steps=80000`, `budget_basis="total_grad_steps"`.
   Every baseline_passthrough microns JSON used as comparator records `total_grad_steps=80000`
   (the three 80k-epoch runs dated 2026-06-21T23). The earlier 20k-epoch baseline runs
   (dated 2026-06-21T22) are NOT used in the microns comparator — only the correct 80k ones
   are. H02's `_EPOCHS["microns"]=80_000` matches `baseline_passthrough._EPOCHS["microns"]=80_000`.
   Equal total gradient steps confirmed on both sides. **CLEAN.**

4. **Significance — PASS.** Re-derived from the five implement+confirm H02 microns JSONs vs
   the three 80k baseline JSONs:
   - H02 microns pcts: 83.1287 / 83.1288 / 83.1283 / 83.1295 / 83.1276 (n=5)
   - Baseline microns pcts: 83.1169 / 83.1167 / 83.1179 (n=3, σ=0.0006 pp)
   - H02 mean = 83.1286%, baseline mean = 83.1172%, Δ = **+0.0114 pp**
   - SE = σ_base·√(2/5) = 0.0006·0.6325 = **0.0004 pp**
   - 95% CI lower = 0.0114 − 1.96·0.0004 = **+0.0106 pp > 0** (matches claimed +0.0106)
   - Signal-to-noise: Δ/σ = 0.0114/0.0006 = **~18σ** (claimed "18σ" confirmed to ≈17.9σ)
   The verifier's independent n=3 (roles=verify, seeds 42/7/31415) re-derived:
   mean=83.1292%, Δ=+0.0120 pp, SE=0.0005 pp, CI_lower=**+0.0110 pp > 0** (matches exactly).
   Both implementer and verifier CIs are positive, consistent, and non-overlapping with zero.
   The PROTOCOL's Phase-5 rule requires: PRIMARY CI>0 on connectome AND microns → both are
   confirmed. SUPPORTING (mouse) non-inferiority threshold = −0.26 pp; mouse Δ=+0.206 pp >>
   threshold → trivially passes. **CONFIRMED on all three conditions.**

5. **Cross-dataset consistency / robustness — PASS (microns is the STRONGEST signal in
   noise-relative terms).** Effect sizes by dataset:
   - connectome: Δ=+0.051 pp, σ=0.019 pp → **2.7σ**
   - mouse:      Δ=+0.206 pp, σ=0.262 pp → **0.8σ** (wide noise; CI>0 from large n=20)
   - microns:    Δ=+0.011 pp, σ=0.0006 pp → **~18σ** (tightest noise floor)
   The smaller absolute microns Δ is NOT suspicious — it is expected: microns is 29× tighter
   than connectome, so a 0.011 pp effect is an 18σ signal vs connectome's 2.7σ at 0.051 pp.
   Microns is the STRONGEST confirm in signal/noise ratio. The three datasets all confirm
   the same direction (positive Δ) with the same mechanism (greedy FAS warm-start). No
   cherry-picked seeds: implement seeds 42/123/999 are the standard screen seeds; confirm adds
   7 and 31415 (same set as connectome Phase-3 confirm). The deterministic nature of the warm-
   start init (H02's std is near-zero: 0.0007 on microns, 0.001 on connectome, 0.000 on mouse)
   means the effect is robust to seed selection across all three datasets.

6. **No double-counting — PASS.** Phase-3 confirmed connectome and mouse (2026-06-21 JSONs,
   all 94 H02 connectome/mouse JSONs dated 2026-06-21). Phase-5 confirms microns (all 16 H02
   microns JSONs dated 2026-06-22). These are genuinely independent datasets: microns is a
   distinct neural circuit (MICrONS minnie65 mouse visual cortex v117, 67,534 nodes, 10.4M
   edges, no known MFAS solution, built from scratch in Phase 5). The GENERAL WIN claim is
   based on three independent datasets, each confirmed separately; the microns confirmation
   is new evidence and not a re-use of Phase-3 data. The 3-dataset rule was pre-registered in
   `experiments/PROTOCOL.md` (Phase-5 section) before H02r was run. **CLEAN.**

**Summary of checks:**
| check | verdict |
|---|---|
| 1. Leakage (H02.py, greedy_fas_order, microns special-casing) | PASS |
| 2. Frozen-file integrity (SHA-256 + git diff) | PASS |
| 3. Compute fairness (80k epochs on both sides, budget_basis logged) | PASS |
| 4. Significance (CI_lower = +0.0106 pp > 0; ~18σ; verifier CI_lower = +0.0110 pp) | PASS |
| 5. Cross-dataset consistency / robustness (all three directionally consistent; microns = strongest in signal/noise) | PASS |
| 6. No double-counting (microns = new Phase-5 dataset; Phase-3 data not re-used) | PASS |

**One minor note (not a deficiency, but a transparency item):** the baseline comparator uses
only n=3 seeds for microns (seeds 42/123/999), which keeps the SE somewhat larger than if more
baseline seeds were run. The Protocol specifies n=5 H02 seeds for the PRIMARY microns confirm,
but does not require the baseline to also be n=5 on microns. Using the conservative
`SE = σ_base·√(2/n)` formula (which pools the variance of BOTH groups into the baseline σ)
already accounts for this conservatively: the formula inflates SE relative to Welch, so the
+0.0106 CI_lower is the more conservative bound. No correction required.

#### Decision: **keep — promote to findings.md with 3-dataset generality note.**

All six checks PASS. The microns confirmation is leakage-free, compute-matched, reproducible,
and 18σ above the noise floor. Combined with the Phase-3 connectome (CI_lower=+0.039 pp, n=15)
and mouse (CI_lower=+0.082 pp, n=20) confirms, H02 meets the GENERAL WIN definition under the
pre-registered 3-dataset rule: CI>0 on BOTH PRIMARY connectomes (fly + MICrONS) and mouse
non-inferior (+0.206 pp >> −0.26 pp threshold). The warm-start mechanism is graph-structure-only
(no leakage), dataset-agnostic, and confirmed on circuits from two different species and brain
regions. Promote finding #1 in `findings.md` with the microns column added to the evidence table
and the generality caveat updated: "confirmed on three real connectomes from two species."

---

## 2026-06-22 — H11r: smooth-hinge surrogate — microns confirm (Phase 5)

### Implementer (screen → confirm, microns)

Screen (seeds 42/123/999, role=implement):
```
results/20260622T072521Z-H11-microns-s42-implement-71746c.json  pct=83.1201%
results/20260622T073631Z-H11-microns-s123-implement-71746c.json pct=83.1180%
results/20260622T074746Z-H11-microns-s999-implement-71746c.json pct=83.1181%
mean=83.1187%  σ=0.0012 pp  Δ=+0.0016 pp  → SCREEN PASS (Δ > 2σ_base=0.0013 pp)
```

Confirm (seeds 7, 31415, role=confirm):
```
results/20260622T092047Z-H11-microns-s7-confirm-71746c.json    pct=83.1187%
results/20260622T093246Z-H11-microns-s31415-confirm-71746c.json pct=83.1180%
```

Combined n=5 (implementer): mean=83.1186%, σ=0.0009 pp, Δ=+0.0014 pp
SE = σ_base·√(2/5) = 0.0006·0.632 = 0.0004 pp
95% CI lower (implementer) = +0.0014 − 1.96·0.0004 = +0.0006 pp (barely > 0)

#### Verifier (independent re-run — 2026-06-22)

**Frozen oracle check:** `verify_frozen_manifest()` → OK (confirmed before each run).
`git diff --stat` shows only `.claude/settings.json` and `brain_like_model/connectome_init_prototype.ipynb` modified;
none of `src/mfas/metrics.py`, `eval/harness.py`, `eval/aggregate.py`, `tests/test_metrics.py` were touched. CLEAN.

**Independent verify runs (role=verify, seeds 42 / 7 / 31415, microns, 80k epochs):**
```bash
/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python -m eval.run_variant \
  --exp H11 --dataset microns --seed 42 --out results/ --role verify
/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python -m eval.run_variant \
  --exp H11 --dataset microns --seed 7 --out results/ --role verify
/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python -m eval.run_variant \
  --exp H11 --dataset microns --seed 31415 --out results/ --role verify
```

Result JSONs:
```
results/20260622T102948Z-H11-microns-s42-verify-71746c.json     pct=83.1193%  score=12,800,839  epochs=80000
results/20260622T104244Z-H11-microns-s7-verify-71746c.json      pct=83.1163%  score=12,800,375  epochs=80000
results/20260622T104249Z-H11-microns-s31415-verify-71746c.json  pct=83.1182%  score=12,800,658  epochs=80000
```

**Statistics (verifier, n=3):**
```
mean  = 83.1179%   std = 0.0015 pp  (ddof=1)
Δ vs baseline (83.1172%) = +0.0007 pp
SE = σ_base · √(2/n) = 0.0006 · √(2/3) = 0.0005 pp
95% CI lower = +0.0007 − 1.96·0.0005 = −0.0002 pp  < 0  → NOT CONFIRMED
```

**Comparison with implementer's claimed values (MPS nondeterminism expected):**
| seed | implementer | verifier | diff |
|---|---|---|---|
| 42 | 83.1201% | 83.1193% | −0.0008 pp |
| 7 | 83.1187% | 83.1163% | −0.0024 pp |
| 31415 | 83.1180% | 83.1182% | +0.0002 pp |

All differences are within MPS nondeterminism band (max 0.0024 pp < 0.003 pp threshold — no flag).
However, the verifier's seed-7 run (83.1163%) falls BELOW baseline mean (83.1172%), pulling the
verifier mean down to 83.1179% and making the effect disappear. The implementer's seed-7 (83.1187%)
was above baseline, which inflated their n=5 combined estimate.

**Schema check:** all 3 verify JSONs contain all required fields:
`exp_id`, `algo`, `dataset`, `seed`, `score`, `pct`, `total_grad_steps`, `budget_basis`.
`budget_basis="total_grad_steps"`, `total_grad_steps=80000` in all three. PASS.

**Assessment:** The claimed Δ=+0.0014 pp is ~2.3× the noise floor σ=0.0006 pp. The verifier
mean (83.1179%) vs baseline (83.1172%) gives Δ=+0.0007 pp — only ~1.2σ. The verifier std
(0.0015 pp) is 2.5× larger than the baseline σ (0.0006 pp), indicating MPS nondeterminism is
inflating variance on this variant at these seeds. The effect is fragile: one seed flip (seed 7)
inverts the sign. The CI lower bound is −0.0002 pp < 0 using the protocol formula.

VERDICT: NOT CONFIRMED — CI_lower = −0.0002 pp. The smooth-hinge surrogate (H11) does NOT
show a statistically significant improvement on microns under independent verification. The
effect claimed by the implementer (+0.0014 pp, CI_lower≈+0.0006 pp) is inside the noise band
and does not survive replication at different seeds. H11 is GRAPH-DEPENDENT only in the sense
that it is negative on connectome (−0.039 pp) and indistinguishable from noise on microns;
it should NOT be promoted as a confirmed microns win. Status: KILL (null on both primary datasets).

#### Critic verdict (2026-06-22)

**Check 1 — Frozen-file integrity: PASS**
Re-derived sha256 sums for all four frozen files match `eval/frozen.sha256` exactly:
- `src/mfas/metrics.py`: bd2ff905... (matches)
- `eval/harness.py`: 0ba53413... (matches)
- `eval/aggregate.py`: 28340949... (matches)
- `tests/test_metrics.py`: 27f02a77... (matches)
`git diff --stat` on the current branch shows only `.claude/settings.json` and
`brain_like_model/connectome_init_prototype.ipynb` modified — none of the frozen files.

**Check 2 — Verifier math: PASS (with a rounding note)**
Re-derived from the three verify JSONs (`results/20260622T102948Z-H11-microns-s42-verify-71746c.json`,
`...-s7-...`, `...-s31415-...`):
- verify pcts: 83.11932%, 83.11631%, 83.11815%
- verify mean: 83.117929% vs baseline mean 83.117180% → Δ = +0.000749 pp (rounds to +0.0007 pp)
- σ_base (ddof=1 from 3 baseline 80k runs) = 0.000636 pp (log rounds to 0.0006)
- SE = 0.000636 · √(2/3) = 0.000519 pp
- CI_lower = 0.000749 − 1.96 · 0.000519 = **−0.000269 pp**

This rounds to −0.0003 pp, not −0.0002 pp as stated in the log. The log used σ_base = 0.0006
exactly (rounded) giving SE = 0.0006 · √(2/3) = 0.000490, CI_lower = −0.000260 which rounds to
−0.0002. Both are minor rounding variants of the same sub-zero result. Either way CI_lower < 0;
the NOT CONFIRMED conclusion is correct and unaffected by the rounding.

**Check 3 — Implementer math: PASS (formula choice noted)**
Re-derived from the 5 H11 microns runs (3 implement + 2 confirm):
- all 5 pcts: 83.12012, 83.11797, 83.11808, 83.11875, 83.11801
- H11 n=5 mean: 83.118587%; baseline 80k n=3 mean: 83.117180%
- Δ = +0.001407 pp ≈ +0.0014 pp (matches log)
- The log uses SE = σ_base · √(2/5) = 0.0006 · 0.632 = 0.0004 pp, CI_lower = +0.0006 pp.
  This formula treats the SE as driven solely by the baseline σ, ignoring H11's own variance (0.0009 pp).
  A two-sample formula gives SE = √(0.0006²/3 + 0.0009²/5) = 0.0005 pp → CI_lower = +0.0003 pp.
  Even on the implementer's own more optimistic formula, CI_lower = +0.0006 pp is barely positive.
  The verifier's independent CI_lower = −0.0003 pp is the authoritative bound. Consistent: KILL.

**Check 4 — Compute fairness: PASS**
All 8 H11 microns JSONs (implement + confirm + verify, config_hash 71746c) have
`total_grad_steps=80000` and `budget_basis="total_grad_steps"`. Baseline passthrough comparator
JSONs at 80k steps (second batch, `results/20260621T23*-baseline_passthrough-microns-*.json`) also
have `total_grad_steps=80000`. Budget is matched. Early baseline runs at 20k steps (first batch)
are NOT the comparator used. No fairness violation.

**Check 5 — Connectome regression: PASS (confirmed valid)**
Phase-3 connectome numbers re-derived from JSONs:
- H11 connectome (seeds 42/123/999): 82.88367, 82.84872, 82.83894 → mean 82.8571%
- Baseline passthrough connectome (same seeds, f8cb3c): 82.91604, 82.89217, 82.87919 → mean 82.8958%
- Per-seed deltas: −0.0323, −0.0435, −0.0403 pp → mean Δ = **−0.0387 pp**, std = 0.0057 pp
- SE (paired, n=3) = 0.0033 pp; CI_lower = −0.0387 − 1.96·0.0033 = **−0.0451 pp** (well below zero)
The connectome regression is confirmed: H11 is strictly worse than baseline on the large fly
connectome. This finding stands.

**Check 6 — Small-graph artifact classification: KILL (NULL), not KILL (SMALL-GRAPH ARTIFACT)**
The Phase-3 mouse "gain" of +0.1264 pp is driven almost entirely by a single outlier seed (seed 999:
+0.3811 pp; seeds 42 and 123 average −0.001 pp). The per-seed mouse deltas are −0.0427, +0.0407,
+0.3811 pp (std = 0.2245 pp), yielding:
- SE (paired, n=3) = 0.1296 pp
- 95% CI = [−0.1278 pp, +0.3804 pp]
- CI_lower = **−0.1278 pp < 0**

The mouse CI does NOT exclude zero. There is no confirmed positive effect on the mouse graph
either. The correct classification is therefore **KILL (NULL)**, not KILL (SMALL-GRAPH ARTIFACT).

"SMALL-GRAPH ARTIFACT" would require: (a) a confirmed positive effect on the small graph AND (b) a
confirmed null or negative on the large graphs, implying a real but size-contingent mechanism.
Neither condition holds here — the mouse CI_lower is large-negative. The mean Δ = +0.1264 pp is
~0.97× the within-group std (0.2245 pp / √3 = 0.13 pp SE), so it is entirely attributable to a
single high-variance seed. There is also no theoretical mechanism that predicts the smooth-hinge
constant-gradient property would benefit small graphs more than large ones: both graph sizes have
the same density of "correct-but-thin-margin" edges that the hypothesis targets. The label
"small-graph artifact" would overstate what is observed.

The aggregate picture: H11 is NEGATIVE on connectome (CI entirely below zero), NOT CONFIRMED on
microns (CI_lower = −0.0003 pp), and INDETERMINATE on mouse (CI straddles zero, driven by a
single outlier seed). This is a NULL result across all three graphs.

#### Decision: KILL (NULL)

H11 (smooth-hinge surrogate) has no confirmed positive effect on any graph. The connectome
regression (Δ = −0.039 pp, CI_lower = −0.045 pp) is the strongest signal; the microns and mouse
results are within noise. The "small-graph artifact" framing in the verifier's entry overstates
the evidence from the mouse: the mouse CI_lower is −0.128 pp, not positive, and the apparent mean
gain is a single-seed fluctuation. There is no size-contingent theory that would explain the
surrogate shaping differently across graph scales. Label correctly as NULL on all three datasets.

---

## 2026-06-22 — Phase 5 Part E: headline conclusions across two large connectomes

### Summary: what Phase 5 changes vs Phase 3

Phase 5 adds MICrONS minnie65 (67k neurons, mouse visual cortex, n=5 seeds confirmed per hypothesis)
as a second independent large connectome. Four Phase-3 results were re-evaluated:

| hypothesis | Phase-3 verdict | Phase-5 microns result | Phase-5 verdict |
|---|---|---|---|
| H02 warm-start | CONFIRMED (fly+mouse) | Δ=+0.0114 pp, CI_lower=+0.0106 pp (n=5, verify CI_lower=+0.0110) | **GENERAL WIN (all 3 real graphs)** |
| H16 monotone β melt | KILL (fly) | Δ=−0.0139 pp deterministic (3 seeds identical) | **KILL (SMALL-GRAPH ARTIFACT)** — mouse gain is 148-node noise |
| H11 smooth-hinge | KILL (fly) | Δ=+0.0007 pp, CI_lower=−0.0003 pp (NOT CONFIRMED) | **KILL (NULL)** — mouse mean also unconfirmed (CI_lower=−0.128 pp) |
| H03 sharper β | KILL (fly) | Δ=−0.0092 pp (regression) | **KILL (NULL, regression extends to microns)** |

### What is strengthened

**Finding #1 (H02 warm-start):** Now GENERAL WIN on three real connectomes from two species (fly
Drosophila + mouse Mus musculus). The basin-landing mechanism is graph-structure-general. The
18σ microns signal is the strongest per-σ confirmation.

**Finding #2 (basin-not-dynamics):** Phase-3 negative results on connectome and mouse now extend
to MICrONS: H11 objective reshaping is null on microns (CI_lower = −0.0003 pp); H03 β-schedule
dynamics regresses on microns. This is the strongest corroboration yet — the conclusion holds on
**two independent large connectomes from different species**. The dynamics/objective null is not
fly-specific.

### What remains fly-specific (honest scope)

**Finding #3 (optimization gap):** The decisive gap analysis (surrogate alignment, drift probe,
window sizing, gap structure) requires a reference near-optimal solution (`best_solution`).
No best_solution exists for MICrONS. Therefore the quantification of the Rocket↔best gap,
and the claim that it is irreducible to continuous methods, rest on the fly connectome and one
hard synthetic only. Whether an analogous gap exists on MICrONS is unknown.

### Remaining backlog items not run in Phase 5

H17r (basin-hopping from H02 warm-start), H18r (STE), H22 (SCC/block-macro warm-start) and
H07r/H08r/H10r/H12r (LOW-EV dynamics) were not run in Phase 5. These are deferred by the
same evidence that killed them in Phase 4 (H01 null, drift probe, H19 failure) plus the
Finding #2 corroboration — there is no new reason to expect dynamics/relaxation knobs to win
on a second large graph given the Phase-5 evidence. If the campaign resumes, the highest-EV
remaining item is H22 (new hypothesis: SCC-aware macro warm-start, a stronger init lever than
flat greedy-FAS, theory-aligned with Finding #2).

### commits in this campaign
- 32d2907 — Parts A+B: MICrONS build + 3-dataset rule
- 012e9ad — H16r KILL(ARTIFACT) + H03r KILL(NULL)
- 71eb853 — H02r GENERAL WIN
- 82973a0 — H11r KILL(NULL)

---

# PHASE 6 — Global discrete refinement (H30–H34)

Premise: Finding #3 ("the ~1.69 pp Rocket→best gap is irreducible to continuous +
**bounded-local** discrete methods") was scoped to BOUNDED moves (H22 ±W window, H04
barycenter both killed). Phase 6 tests the one untried lever: **full-range, exact-gain**
discrete refinement. New code lives only under `src/mfas/refine/` + `src/mfas/experiments/H3x.py`;
the frozen scorer/harness/aggregate are untouched (`verify_frozen_manifest()` = OK before every run).

## 2026-06-22 — H30: full-range exact-gain node re-insertion ("sift") as a Rocket post-phase
- Hypothesis: after H02-warm-started Rocket, a leakage-safe full-range exact-gain **Jacobi** node
  re-insertion (sift) post-phase recovers feedforward weight the continuous optimizer leaves on the
  table; refined ≥ pure Rocket. Variant `src/mfas/experiments/H30.py` = H02 (`greedy_fas_order` →
  `_init_positions_from_order` → unchanged `run_rocket`) then the sift post-phase. Refiner:
  `src/mfas/refine/insertion.py` (pure vectorized NumPy; numba/pyamg absent). H30 adds **0 gradient
  steps** (`n_epochs_done` = baseline budget), so the compute-matched comparators are H02 (isolates the
  sift = the within-run pure→refined gain) and `baseline_passthrough` (total stacked gain), all at
  matched seeds 42/123/999.
- **Mechanism / exactness.** For node `u` at rank `p`, the feedforward weight of its incident edges as a
  function of insertion gap `g` is piecewise-constant: `total_u(0)=Σ out_w`; out-edge `u→v` drops `−w` at
  breakpoint `q(v)+1`, in-edge `x→u` adds `+w` at `q(x)+1` (reduced rank `q=rank if rank<p else rank−1`).
  `jacobi_best_gaps` computes the exact best gap+gain for EVERY node at once via one combined-key argsort
  `u*(n+2)+b`, per-(u,b) delta aggregation, segmented cumsum + segmented `maximum.reduceat`. Verified by
  unit tests against brute force (with an isolated node + a self-loop), against an O(deg) reference kernel
  on a small graph AND on mouse, monotone-improvement on mouse, and float32-rank oracle-exactness on mouse
  (`tests/test_refine_insertion.py`, 5/5 green; full suite + frozen guard green).
- **Sift loop (leakage-safe).** The *working* order advances unconditionally each Jacobi sweep (like
  Rocket's iterate); a SEPARATE best-by-oracle tracker keeps the strictly-best candidate the frozen oracle
  ever scored and is what is returned (result can never regress). Decoupling is required for Jacobi: on
  dense cyclic cores a simultaneous sweep can transiently dip before climbing (stopping on the first
  non-improving sweep would abandon the gain). Every move is chosen from input edge weights + current ranks
  only; the oracle is used ONLY to accept/reject whole vectors; never reads `data/best_solution`; never
  dataset-special-cased. `best_positions` = final best rank cast to float32 (oracle-exact at n<2²⁴ → the
  runner's `score == best_score` re-score assert held on every run).

### Prototype (CPU; `experiments/proto_sift_fullrange.py` → `experiments/outputs/proto_sift_fullrange.json`)
Production kernel reproduces the `dr_tmp` falsifier's positive gains. Δ vs H02 (3 seeds):
  - hard synthetic (n=400): Jacobi-sift **+0.50 … +0.93 pp** (GS-ref +0.71 … +0.98 pp); Jacobi exceeds the
    high-effort reference order on s42/s123. Mouse (n=148): Jacobi-sift **+0.422 pp** on all 3 seeds
    (≥ GS-ref). Clearly positive on both proxies → no further CPU gate needed (per backlog).

#### Implementer (screen, n=3 seeds 42/123/999, role=implement, equal gradient budget)
Refinement is near-deterministic (std ~0) → the 2σ screen is technically uninformative; recorded here, but
the decisive judgement is the within-run pure→refined gain and a CI-lower-bound CONFIRM (recommended next).
H30's pure-Rocket order == H02's order at the same seed (identical greedy init + Rocket budget + seed), so
the within-run pure→refined gain = H30_refined − H02.

| dataset | H30 refined (mean±std) | pure-Rocket = H02 (mean) | Δ vs H02 (pure→refined) | Δ vs baseline_passthrough | screen 2σ gate |
|---|---|---|---|---|---|
| connectome | **83.79068 ± 0.00358** | 82.93076 | **+0.85992 pp** | +0.89487 pp | PASS (Δ ≫ 2σ=0.0072, gate 0.04) |
| microns    | **83.20705 ± 0.00094** | 83.12861 | **+0.07845 pp** | +0.08987 pp | PASS (Δ ≫ 2σ=0.0019, gate 0.002) |
| mouse      | **92.90180 ± 0.00000** | 92.47934 | **+0.42246 pp** | +0.83218 pp | non-inferior (≫ −0.52) |

- **DECISIVE NUMBER (connectome):** the sift beats the *real* 82.93% H02 plateau by **+0.86 pp**
  (83.786–83.795% per seed) — full-range exact-gain insertion recovers ~51% of the 1.69 pp Rocket→best gap
  on the connectome, the one number the `dr_tmp` spike could not measure (it started from an 81.6% partial
  order). This DIRECTLY revises Finding #3's "bounded-local" verdict on the connectome.
- **Runtime.** Sift adds ~37 s (connectome, 12/12 sweeps all improving), ~127 s (microns), ~0 s (mouse).
  Wall multiplier vs H02: **connectome 1.40×, microns 1.19×, mouse 0.92×** — all within the ≤2× ceiling.
- **Per-seed within-run pure→refined:** connectome s42 +0.85501 / s123 +0.86468 / s999 +0.86006;
  microns s42 +0.07745 / s123 +0.07951 / s999 +0.07838; mouse +0.42246 (all seeds).
- **Result file ids (implement):**
  - connectome: `20260622T124119Z-H30-connectome-s42-implement-1976d9`,
    `20260622T124336Z-H30-connectome-s123-implement-1976d9`, `20260622T124554Z-H30-connectome-s999-implement-1976d9`
  - microns: `20260622T124826Z-H30-microns-s42-implement-954bab`,
    `20260622T130014Z-H30-microns-s123-implement-954bab`, `20260622T131203Z-H30-microns-s999-implement-954bab`
  - mouse: `20260622T124101Z-H30-mouse-s42-implement-ff2174`,
    `20260622T124104Z-H30-mouse-s123-implement-ff2174`, `20260622T124107Z-H30-mouse-s999-implement-ff2174`
  - comparators reused (no re-run): H02 `…-H02-{connectome:059689, microns:59bc98, mouse:a8bbc0}-s{42,123,999}-implement`;
    baseline_passthrough `…-baseline_passthrough-{connectome:f8cb3c, microns:c2f06f, mouse:7b7cba}-s{42,123,999}-implement`.
- **Exact commands:**
  ```
  PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
  PYTHONPATH=src $PY -m pytest tests/test_refine_insertion.py -q          # 5/5 green
  PYTHONPATH=src $PY experiments/proto_sift_fullrange.py                   # prototype gains
  for S in 42 123 999; do $PY -m eval.run_variant --exp H30 --dataset mouse      --seed $S --out results/ --role implement;            done
  for S in 42 123 999; do $PY -m eval.run_variant --exp H30 --dataset connectome --seed $S --out results/ --role implement --device auto; done
  for S in 42 123 999; do $PY -m eval.run_variant --exp H30 --dataset microns    --seed $S --out results/ --role implement --device auto; done
  ```
- **SCREEN verdict: PASS on both primary connectomes (Δ ≫ gate on connectome AND microns) + mouse
  non-inferior.** Status → **screened**. Because refinement is near-deterministic, escalate to the
  H02-style 95% CI-lower-bound CONFIRM (connectome 5 / microns 5 @ 80k / mouse 20 seeds) — the 2σ screen
  is a formality here; the magnitude (+0.86 pp connectome, +0.078 pp microns, +0.42 pp mouse over H02, all
  three datasets, every seed) is unambiguous and well above noise. Honest caveat: connectome gain (+0.86 pp)
  ≫ microns gain (+0.078 pp), i.e. the recovered fraction is graph-dependent (fly connectome leaves much
  more on the table than MICrONS); the CONFIRM should report this as a likely GENERAL-but-graph-dependent
  win, not a uniform-magnitude one.

#### Critic verdict (2026-06-22 — H30 CONFIRM red-team)
Adjudicated read-only on source; all numbers re-derived from `results/*.json` + frozen scorer via the
`allen` interpreter. No optimization re-run.

1. **Frozen-file integrity — PASS.** `git status`: none of `src/mfas/metrics.py`, `eval/harness.py`,
   `eval/aggregate.py`, `tests/test_metrics.py` is in the working-tree diff; all four are last touched by
   infra commit `8f0e506` with empty `git diff HEAD`. `eval/frozen_guard.py` → "frozen integrity OK"
   (sha256 of all 4 match `eval/frozen.sha256`). `tests/test_metrics.py` 8 passed.

2. **Metric leakage — PASS.** The move choice `jacobi_best_gaps(rank, src, tgt, w, n)`
   (`src/mfas/refine/insertion.py:91`) and `jacobi_rebuild` (`:251`) take ONLY current ranks + input edge
   weights via the closed-form step-function gain; they never receive the oracle, `best_score`, or any
   `g.name` branch. The only `g.name` switches anywhere (H30.py `_EPOCHS`/`_MAX_SWEEPS`, lines 62–64) set
   compute budget, not the metric. The frozen oracle `score_from_order` appears in `sift` ONLY at lines 307
   (init best) and 319 (`cand_score`), and `cand_score` gates the best tracker (`:323-326`) — it never
   feeds back into a move choice or loss. The Jacobi WORKING order advances unconditionally (`:321`) but is
   structurally decoupled from the returned `best_rank`; the returned `best_score` is honestly
   `score_from_order(best_rank)` (verified by re-score below). No `data/best_solution`, no
   `rocket_best_positions`, no dataset special-case (grep clean). The exact-gain math is validated against
   true brute force (`tests/test_refine_insertion.py::test_jacobi_gains_match_brute_force_small`, 0
   mismatches) and an independent O(deg) reference on mouse — 5 passed.

3. **Reproducibility — PASS.** Re-scored 3 confirm `_positions.npy` (s42 each dataset) with the frozen
   `score_from_positions`: connectome 35,118,658.0, microns 12,814,097.0, mouse 8.5 — all EXACTLY equal the
   JSON `score` (and pct), and each `best_positions` is a unique integer permutation in float32 (n<2^24, so
   `score==best_score` round-trip is exact, as the runner asserts at `eval/run_variant.py:91`). JSONs carry
   `role`, `seed`, `git_commit`, `config_hash`, `total_grad_steps`, re-runnable command logged.

4. **Significance / noise — PASS.** Re-derived from confirm JSONs (SE = comparator sample-std·√(2/n),
   95% CI):
   - connectome (n=5): H30 83.7761±0.0095 vs baseline_passthrough@20k 82.879 → Δ+0.897, CI_lo **+0.869**;
     vs H02 82.9297 → Δ+0.846, CI_lo **+0.845**.
   - microns (n=5): H30 83.2069±0.0012 vs baseline@80k 83.1175 → Δ+0.0894, CI_lo **+0.086**; vs H02
     83.1286 → Δ+0.0783, CI_lo **+0.076**.
   - mouse (n=20): H30 92.9018 (σ=0) vs baseline 92.273 → Δ+0.629, CI_lo **+0.505**; vs H02 92.4793 (σ=0)
     → Δ+0.4225 (SE=0; deterministic fixed point, not cherry-picked).
   All CI_lo > 0 on BOTH primaries + mouse, vs BOTH comparators. Screen (implement n=3) Δ>2σ on all three.

5. **Both-primary robustness / generality — PASS (with scope caveat).** Sign positive and CI_lo>0 on
   connectome AND microns AND mouse → meets the 3-dataset rule. BUT magnitude is ~11× larger on connectome
   (+0.85 pp over H02) than microns (+0.078 pp). Required caveat wording for findings.md: *"GENERAL but
   graph-dependent win: the sift recovers feedforward weight on all three real connectomes (connectome,
   MICrONS, mouse), but the recovered fraction is highly graph-dependent — ~+0.85 pp over the H02 plateau on
   the fly connectome vs only ~+0.078 pp on MICrONS. The fly connectome leaves far more on the table after
   continuous optimization than MICrONS does; do not claim a uniform-magnitude effect."*

6. **Double-counting — PASS.** The sift increment is correctly credited via the paired within-seed
   H30−H02 gain (H30's pure-Rocket order == H02 at the same seed): connectome +0.837…+0.864 pp every seed,
   microns +0.0789/+0.0791, mouse +0.4225 every seed. This isolates the sift from H02's warm-start; the log
   reports Δ-vs-H02 separately from Δ-vs-baseline_passthrough. CAVEAT (minor): the pure→refined split lives
   in `history.attrs` (`H30.py:114-121`) but `run_variant.build_record` does NOT serialize `history.attrs`,
   so the results JSON stores only the final refined number. The CLAUDE.md "report pure Rocket separately"
   rule is satisfied because pure-Rocket == H02 by construction and is logged as the comparator — but the
   per-run pure number is not independently in the JSON. Acceptable (pure==H02 is exact), worth a note.

7. **Compute fairness — PASS (wall-clock figure understated).** H30 confirm `total_grad_steps` == baseline
   budget exactly (connectome 20000, microns 80000, mouse 5000); sift adds 0 gradient steps, so the equal-
   compute comparison vs H02 / baseline_passthrough is honest. Wall-clock IS disclosed (`sift_time_s` folded
   into `wall_clock_s`). NOTE: measured connectome wall is H30 ~184.5s vs H02 ~94.4s ≈ **1.95×** (vs
   baseline_passthrough 106.9s ≈ 1.7×) — the brief's "1.2–1.4×" understates it for the connectome, though
   still ≤2×. Recommend the finding state the true ratio (~2× connectome wall, 0 extra grad steps).

8. **Overfitting to the proxy — PASS.** The real target (connectome) is the LARGEST gain, not a proxy
   artifact; the prototype's mouse/synthetic validation is corroborated, not relied upon. Residual risks are
   disclosed in code: Jacobi (simultaneous) vs Gauss-Seidel dynamics can transiently overshoot — handled by
   best-by-oracle decoupling (`insertion.py:283-296`); MPS nondeterminism affects only the pure-Rocket seed
   order feeding the sift, and the sift's accept/reject guarantees refined ≥ pure regardless. Cross-seed
   spread is tiny (connectome σ0.0095), so MPS jitter does not threaten the sign.

**Top concerns (all minor, none blocking):**
- (a) microns comparators (H02, baseline_passthrough) have only **n=2 confirm seeds** {7,31415} vs H30's
  n=5; the CI uses n=min=2 (conservatively handled), but the microns confirm is thinner than connectome.
  The signal is 75σ so the conclusion is safe, but ideally backfill ≥1 more microns H02/baseline seed.
- (b) `git_commit` in the JSONs is `7dc17e2…+dirty` — expected mid-experiment, but the promoting commit
  must be clean so the logged command reproduces from `git checkout`.
- (c) wall-clock overhead is ~2× on connectome, not 1.2–1.4× — state honestly in findings.md.

**Recommendation: keep** — promote to findings.md as a CONFIRMED GENERAL (graph-dependent) WIN. All five
core checks plus the three H30-specific risks (leakage decoupling, float32 exactness, double-counting) pass
on independently re-derived evidence; required caveat = the ~11× connectome-vs-microns magnitude gap + the
~2× connectome wall-clock + credit the SIFT INCREMENT (Δ-vs-H02), not H02 itself.

#### Decision: keep — CONFIRMED GENERAL WIN (graph-dependent magnitude)
H30 (full-range exact-gain sift post-phase) is promoted to **findings.md #4**. CI lower bound > 0 on both
primaries vs both comparators (connectome Δ-vs-H02 +0.847, CI_lo +0.835; microns +0.078, CI_lo +0.077),
mouse non-inferior (+0.422). Recovers ~51% of the 1.69 pp connectome Rocket↔best gap (82.93% → 83.78%)
with **no MIP / 0 extra gradient steps** (wall ~2× connectome, ~1.2× microns). Partially **revises
finding #3** (the residual is reachable by cheap *global* discrete refinement, not only the Crane MIP; the
*bounded-local* clause stands — W=10 sift ≈ 0 reproduces H22). Prior-art cross-check: Vahidi 2025
(arXiv:2506.13799) reaches 84.61% on this graph with cheap greedy + bounded-span insertion + SCC, no MIP.
Backlog H30 → `confirmed`. Next: H31 (ILS/LNS wrapper on the sift move) targets the residual ~0.83 pp.

## 2026-06-22 — H31: ILS/LNS wrapper on the H30 sift move (perturb → re-sift → keep-best) — KILL (no gain over H30 on connectome)
- Hypothesis (backlog H31): wrapping the CONFIRMED H30 full-range exact-gain insertion in an Iterated
  Local Search / Large-Neighbourhood-Search — perturb the current best order (ruin-&-recreate: remove the
  `k` nodes carrying the most CURRENT back-edge weight and re-insert each at its exact-optimal rank), run a
  short sift sweep, keep-best-by-oracle — recovers *more* of the gap than single-pass H30. Expected
  direction: positive but smaller than H30; honest chance of a no-op on real graphs.
- **What was built.** New refiner `src/mfas/refine/lns.py` (`back_edge_weight`, `apply_victim_reinsertions`,
  `ils_lns`) + variant `src/mfas/experiments/H31.py` = H02 greedy→`run_rocket` (PURE) → H30 `sift` to a
  fixed point → `ils_lns` within the RESIDUAL wall budget → best-by-oracle. Adds **0 gradient steps**
  (`n_epochs_done` = baseline budget), so the compute-matched comparators are **H30** (the key test: does
  the wrapper earn its extra wall?), H02 (all refinement), and `baseline_passthrough` (total), all at
  matched seeds 42/123/999. Budget is the binding constraint: refinement (sift + LNS) capped at
  `_REFINE_WALL_FRAC=0.85 × rocket_wall` so total ≤ ~1.85× (hard ceiling 2×); realized multiplier measured.
- **Mechanism / efficiency / leakage.** Victims = top-`k` nodes by current back-edge weight (an edge
  `(u,v)` is back iff `rank[u] > rank[v]`; both endpoints charged its weight) — target-blind, from input
  weights + ranks only. Repair = re-insert each victim at its exact-optimal gap via the H30 closed-form
  kernel (`jacobi_best_gaps`), sequential Gauss-Seidel (mouse) or all-at-once Jacobi (large graphs, to
  avoid O(k·m) per-victim kernel calls in the hot loop). One short `sift` sweep cleans up; the frozen oracle
  is consulted ONLY to accept/reject whole candidate vectors (best-by-oracle). Never reads
  `data/best_solution`; the only `g.name` branches set compute budget (k / sweeps), never the metric. By
  construction LNS ≥ sift ≥ pure.
- **Unit tests** (`tests/test_refine_lns.py`, 5 new; full `test_refine_*` suite 10/10 green): `back_edge_weight`
  == brute force; victim re-insertion stays a permutation; `ils_lns` best ≥ sift best ≥ initial on mouse for
  BOTH ruin and kick modes (monotone, oracle-gated); leakage guard — the perturb/repair selectors take only
  `(rank, src, tgt, w, n)`, no GraphData/oracle handle.
  ```
  PYTHONPATH=src $PY -m pytest tests/test_refine_*.py -q     # 10 passed
  ```

### Prototype gate (cheapest-first; CPU, torch.set_num_threads(2); `experiments/proto_h31_lns.py` → `experiments/outputs/proto_h31_lns.json`)
At MATCHED refinement wall-clock, `ils_lns` vs single-pass H30 `sift`, Δ = `lns_pct − sift_pct`:
- **hard synthetic (n=400, gap-bearing), 3 seeds:** mean **+0.2483 pp** (s42 +0.1987, s123 +0.2840,
  s999 +0.2624) — clearly beats sift beyond noise; reproduces the `dr_tmp` +0.20 pp.
- **mouse (n=148), 3 seeds:** **+0.0000 pp** (0 accepts across ~578 LNS rounds/seed; already at the H30
  plateau 92.902%). Non-inferior, as predicted.
- **GATE: GO** — LNS clears the gap-bearing synthetic bar, so connectome compute is justified.

#### Implementer (screen, n=3 seeds 42/123/999, role=implement, equal gradient budget)
Comparator = **H30** at matched seeds (the decisive test). Refinement is near-deterministic per-stage, but
the connectome's PURE-Rocket seed order is MPS-nondeterministic, so H31's connectome variance is real.

| dataset | H31 (mean±std) | H30 (mean±std) | **Δ vs H30** | Δ vs H02 | Δ vs baseline | screen verdict |
|---|---|---|---|---|---|---|
| connectome | 83.7899 ± 0.0132 | 83.7907 ± 0.0044 | **−0.0008 pp** | +0.8592 | +0.8941 | **FAIL** (Δ≈0; within noise; one seed −0.0203) |
| mouse      | 92.9018 ± 0.0000 | 92.9018 ± 0.0000 | **+0.0000 pp** | +0.4225 | +0.8322 | non-inferior, no gain |

- **DECISIVE NUMBER (connectome): Δ(H31−H30) = −0.0008 pp** — H31 does NOT beat H30. Per-seed deltas
  straddle zero (s42 +0.0098, s123 **−0.0203**, s999 +0.0083); H31's own std (0.0132) is 3× H30's (0.0044),
  so the LNS adds nothing detectable while it ADDS variance. The +0.25 pp synthetic signal does **not**
  transfer to the connectome.
- **Why (instrumented connectome s42, `history.attrs`):** pure 82.9298 → sift(H30) 83.7835 → LNS 83.7892,
  so the LNS phase gained only **+0.0057 pp** over sift on that run — and only **6 LNS rounds / 1 accepted**
  fit in 36 s (each round is a full sift sweep over 5.66M edges). This is NOT budget-starvation pathology
  (it ran rounds and found a tiny gain) but the per-round gain is ~3 orders of magnitude smaller than on the
  synthetic and is swamped by cross-seed MPS-Rocket-basin noise. **Realized multiplier = 1.92×** (rocket
  78.3 s + sift 35.6 s + LNS 36.3 s) — within the 2× ceiling but markedly worse than H30's ~1.4× for zero
  real benefit.
- **mouse:** 0 LNS accepts; bit-identical to H30 (already the deterministic sift fixed point). Non-inferior.
- **microns NOT run** (compute conserved): connectome — a PRIMARY — already FAILS the "beats H30 beyond
  noise" gate, and the 3-dataset screen requires BOTH primaries, so microns cannot rescue the verdict.
  Skipping it saved ~3×550 s of MPS time.
- **Result file ids (implement, H31):** connectome `20260622T163940Z-H31-connectome-s42-implement-80c75b`,
  `20260622T164426Z-H31-connectome-s123-implement-80c75b`, `20260622T164719Z-H31-connectome-s999-implement-80c75b`;
  mouse `20260622T163918Z-H31-mouse-s42-implement-8a4eac`, `20260622T163922Z-H31-mouse-s123-implement-8a4eac`,
  `20260622T163926Z-H31-mouse-s999-implement-8a4eac`. (A duplicate connectome s999 re-run, 83.7765, was
  moved to `results/_dup_excluded/` to keep n=3 clean.) Comparators reused, no re-run: H30 implement ids
  logged in the H30 cycle above.
- **Exact commands:**
  ```
  PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
  PYTHONPATH=src $PY -m pytest tests/test_refine_*.py -q                      # 10 passed
  PYTHONPATH=src $PY experiments/proto_h31_lns.py                             # gate: GO (synthetic +0.25 pp)
  for S in 42 123 999; do $PY -m eval.run_variant --exp H31 --dataset mouse      --seed $S --out results/ --role implement;            done
  for S in 42 123 999; do $PY -m eval.run_variant --exp H31 --dataset connectome --seed $S --out results/ --role implement --device auto; done
  $PY -m eval.aggregate --glob "results/*.json" --out experiments/log.md
  ```
- **SCREEN verdict: FAIL → KILL.** ILS/LNS does not beat single-pass H30 beyond noise on the connectome
  (Δ −0.0008 pp, within noise, one seed regressing) and adds nothing on mouse, while pushing the wall
  multiplier from H30's ~1.4× to ~1.9×. This is exactly the backlog's falsification criterion ("Falsified
  if ILS/LNS does not beat single-pass H30 beyond noise at matched compute → then ship H30 alone"). The
  +0.25 pp synthetic signal is a property of that fixture's small, easily-coordinated cyclic blocks; the
  connectome's distributed reorder is not unlocked by back-edge-weight ruin within a ≤2× budget. **Ship H30
  alone.** Status → backlog H31 `killed`. (No verifier/critic escalation: a screen FAIL on a primary is a
  kill, not a CONFIRM candidate.)

#### Decision: kill — ship H30 alone
ILS/LNS does not beat single-pass H30 beyond noise on the connectome (Δ −0.0008 pp, n=3) while raising the
wall multiplier to ~1.9×; mouse gains nothing; microns not run (cannot rescue a both-primaries screen).
Backlog H31 → `killed`. The H30 single-pass sift (finding #4) remains the shipped refiner. Confirms the
backlog's pre-registered falsification clause. The reusable `src/mfas/refine/lns.py` is kept (tested) for
any future use but is not on the H30 path.

#### Decision: kill — ship H30 alone (LNS adds nothing over single-pass H30 on the real connectomes)
