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

## Budget & stop-criteria  (set for the Phase-3 campaign)
- `MAX_EXPERIMENTS` = **30**  (hard cap on total cycles, including iterations)
- `MAX_WALLCLOCK` = **12h**  (wall-clock ceiling for the whole campaign)
- `EARLY_EXIT_K` = **2**  (stop iterating a single hypothesis after 2 consecutive non-improvements)
- `CONFIRM_SEEDS_CONNECTOME` = **5**  (42, 123, 999, 7, 31415)
- `CONFIRM_SEEDS_MOUSE` = **20**  (mouse is ~14× noisier → more seeds to tighten the CI)

Stop the campaign when ANY fires: backlog exhausted / `MAX_EXPERIMENTS` reached /
`MAX_WALLCLOCK` reached / `EARLY_EXIT_K` consecutive non-improving cycles with no remaining
promising backlog items.

## Compute-matched comparison (fairness)
Every variant is compared to baseline under an **EQUAL compute budget**. The basis is
**total gradient steps** = the number of `optimizer.step()` calls, **summed across any restarts**
(chosen over wall-clock because MPS timing is non-deterministic and machine-dependent;
gradient-steps are exact, reproducible, auditable). Wall-clock is still logged (`wall_clock_s`) but
is not the comparison basis.
- Every `results/*.json` records `budget_basis = "total_grad_steps"` and `total_grad_steps`
  (= `RocketResult.n_epochs_done`). **Variants MUST set `n_epochs_done` to the total optimizer
  steps actually performed.**
- **Standard knob-swap variants** (init / loss / β / optimizer / LR / grad handling — H02, H03,
  H05–H12) run at the *same* epoch budget as baseline (connectome 20k, mouse 5k); the existing
  `baseline_passthrough` at matched seeds is therefore already the equal-budget comparator.
- **Multi-start / restart / subsample variants** (H01, H13) and added-compute refinement (H04) are
  compared to `src/mfas/experiments/baseline_multistart.py` — the *plain* baseline run as K naive
  restarts × (total/K) epochs, best-of-K, with `n_epochs_done = total`. Set `MFAS_MULTISTART_K` to
  the variant's K so total steps match. This isolates algorithmic merit from mere extra sampling.
  The verifier computes Δ from the *specific* matched-budget result JSONs (not the global table).

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
experiment_id, role(implement|verify|confirm), hypothesis,
budget_basis(="total_grad_steps"), total_grad_steps(=n_epochs_done)
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
- Never modify frozen files. Three layers enforce this: a PreToolUse hook blocks Edit/Write,
  the 4 files are chmod 0444, and `eval/frozen_guard.verify_frozen_manifest()` (called by
  `run_variant.py` before any score) aborts if their SHA-256 differs from `eval/frozen.sha256`.
- Always evaluate BOTH datasets; ≥3 seeds; report mean ± std.
- Every reported number traces to a `results/*.json` + a re-runnable command — never fabricated.
- Never hardcode/peek at the target metric inside a variant.
- One git commit per experiment; reproducible via `git checkout` + the logged command.
- A gain within noise is not a gain.

## Run the campaign
For each ranked hypothesis: `/run-experiment <backlog-id>` (e.g. `/run-experiment H01`).
Stop when a budget/stop-criterion above is hit.

---

## Phase-5 update: 3-dataset rule (added 2026-06-22)

Phase 5 adds the **MICrONS minnie65** mouse-visual-cortex connectome as a THIRD dataset and a
SECOND large real connectome (~67,534 nodes, 10.4M edges). The evaluation rule is updated by
character of dataset:

### Dataset character classification
| dataset | character | role in verdict |
|---|---|---|
| `connectome` | FlyWire fly, 136k nodes, 5.66M edges — LARGE, hard cyclic core | **PRIMARY** |
| `microns` | MICrONS minnie65, 67,534 nodes, 10.4M edges — LARGE, 97% in giant SCC | **PRIMARY** |
| `mouse` | tiny 148-node graph, σ≈0.26 pp (14× noisier) — near-saturated | **SUPPORTING / non-inferiority only** |

### Updated noise floor and budget table (≥3 seeds, 42/123/999)
| dataset | baseline mean (pct) | σ (std) | 2σ screen threshold | epoch budget | wall/run |
|---|---|---|---|---|---|
| connectome | 82.8958% | 0.0189 pp | 0.04 pp | 20,000 | ~90s |
| mouse | 92.0696% | 0.2624 pp | 0.52 pp | 5,000 | ~1s |
| microns | **83.1172%** | **0.0006 pp** | **0.002 pp** | **80,000** | ~550s |

> MICrONS is **29× tighter than connectome** and **408× tighter than mouse** — exceptional
> discriminating power; even a 0.002 pp gain is reliably detectable.
> **Plateau verification:** 20k=83.024%, 40k=83.100%, 80k=83.117%, 120k=83.118% (+0.001 pp);
> within-run trajectory flatlines at ~90k iter inside 120k schedule → 80k is at plateau.
> **Baseline runs (80k):** s42=83.1169%, s123=83.1167%, s999=83.1179% (mean 83.1172%, σ=0.0006 pp).

### Updated CONFIRMED-improvement rule (3 datasets)

A variant is a CONFIRMED improvement only if it passes BOTH stages across ALL datasets
in the following way:

**Stage 1 — SCREEN** (cheap gate, 3 seeds 42/123/999):
- Pass iff `Δmean > 2σ` on BOTH PRIMARY datasets (`connectome` AND `microns`).
- Mouse check: `Δmean > −σ_mouse` (non-inferior; a large mouse regression can still kill).
- The screen on `microns` uses its own per-dataset noise floor.

**Stage 2 — CONFIRM** (only for screened variants; more seeds):
- PRIMARY datasets: 95% Welch CI lower bound `> 0` on BOTH (`connectome` and `microns`).
  Seeds: connectome = 5 (42,123,999,7,31415); microns = 5 (same seeds, budget-matched at 80k).
- SUPPORTING non-inferiority: mouse 95% CI lower bound `> −σ_mouse` (i.e. > −0.26 pp).
- **CONFIRMED GENERAL WIN** iff all three conditions pass.

### New verdict class: GRAPH-DEPENDENT
A variant that **confirms on ONE primary connectome but not the other** is classified as:

**GRAPH-DEPENDENT** (= "recovered lost theory") — a real effect scoped to one connectome's
structure. Report with explicit scope: e.g. "gains on microns+mouse, not fly connectome —
connectome-specific suppression." Neither promoted to GENERAL WIN nor dismissed as noise.

Full decision table:
| microns CI>0? | connectome CI>0? | mouse non-inf? | verdict |
|---|---|---|---|
| ✓ | ✓ | ✓ | **GENERAL WIN** → findings.md |
| ✓ | ✗ | ✓ | **GRAPH-DEPENDENT** (microns+mouse, not fly) → log + scope |
| ✗ | ✓ | ✓ | **GRAPH-DEPENDENT** (fly+mouse, not microns) → log + scope |
| ✓ | ✓ | ✗ | **GENERAL WIN with mouse caveat** → findings.md + caveats |
| ✗ | ✗ | — | STILL NULL (or SMALL-GRAPH ARTIFACT if mouse was positive) |
| ✗ or ✓ | ✗ or ✓ | — (any) | SMALL-GRAPH ARTIFACT if ONLY mouse is positive |

### Compute-matched rule (unchanged, extended to microns)
`budget_basis = "total_grad_steps"`. Variants on `microns` run at 80k epochs (= `_EPOCHS["microns"]`
in `baseline_passthrough.py` and `dataset_overrides.microns.rocket.epochs` in
`configs/baseline_rocket.yaml`). Comparator for knob-swap variants on microns = `baseline_passthrough`
at matched seeds and 80k epochs. Multi-start variants → `baseline_multistart` at matched total steps.

### Frozen oracle (unchanged)
`eval/frozen_guard.verify_frozen_manifest()` called before every scored run. `src/mfas/io.py` is
writable (adding `microns` needed no change to any frozen file). MICrONS is leakage-clean (no
known MFAS solution; `data/best_solution` does not apply).

---

## Multi-track research map (added 2026-08-01)

The campaign now spans **three tracks**. Track A (everything above) is unchanged. Tracks B and C
reuse the same frozen oracle and the shared reproducibility contract below, but each has its own
verdict rule and file contract — they are **not** run through SCREEN/CONFIRM. The live cross-track
plan is `experiments/roadmap.md` (priority + pointers; it never restates a queue's status).

| track | question | queue (owns status) | code | results | conclusions | verdict rule |
|---|---|---|---|---|---|---|
| **A — improvement (H)** | beat the baseline on the exact metric | `backlog.md` | `src/mfas/experiments/H*.py` | `results/*.json` | `findings.md` | SCREEN + CONFIRM (above) |
| **B — diagnostics (Q)** | *explain* Rocket's behaviour | `questions.md` | `experiments/diagnostics/` + `src/mfas/analysis/gap.py` | *(none in `results/`)* | `diagnosis.md` | reproducible-explanation |
| **C — random graphs (G)** | real vs random unavoidable feedback | `randomgraph.md` | `src/mfas/randomgraph/` | `results/randomgraph/*.json` | `randomgraph.md` (findings §) | N-realization mean±std |

### Shared reproducibility contract (ALL tracks)
- Every reported number traces to a **committed artifact** (a `results/*.json`, an
  `experiments/outputs/*.json`, a plot, or a table) **plus a re-runnable command**. Never fabricated.
- Never modify frozen files; never hardcode/peek at the discrete target metric inside any
  algorithm **or generator**.
- Fixed, logged seeds; report dispersion (std / CI), not point values.
- One git commit per experiment/answer; scratch stays in `dr_tmp/` (gitignored) until **promoted**
  to its track's home.

### Track B — Diagnostics (Q-series) — verdict + contract
- **Purpose:** answer a question (mechanism / measurement), not to win — **no SCREEN/CONFIRM**.
- **ANSWERED** when: claim + method + cited artifact + re-runnable command are written to
  `diagnosis.md` under a `## Q0x — <question>` anchor, **and** any finding it contradicts is
  corrected/annotated.
- **Leakage firewall:** any read of `data/best_solution` goes through `mfas.analysis.gap`;
  diagnostics **never write to `results/`** (outputs → `experiments/outputs/` or `dr_tmp/`;
  conclusions → `diagnosis.md`). Enforced by convention + the critic + the Edit/Write guard hook
  (`data/best_solution` and `results/**/*.json` are write-protected). *Convention still covers what
  the hook can't see — e.g. a raw `best_solution` read from inside a running script.*
- **File contract:** `questions.md` (owns Q status) ; stable scripts in `experiments/diagnostics/`
  (promoted from `dr_tmp/`) ; answers appended to `diagnosis.md`.
- **Numbering:** `Q01`+ for new questions. Historical diagnostics that shipped in the H-series
  (**H21** hard-synthetic fixture, **H22** window/SCC sizing) keep their H IDs — they are cited by
  commit-hash across `findings.md`/`diagnosis.md`/`log.md`; do **not** renumber.

### Track C — Random graphs vs brains (G-series) — verdict + contract
- **Purpose:** quantify `unavoidable_feedback = total_weight − best_feedforward` and compare the
  real connectome(s) to **structure-matched** null models (ER, configuration model, stochastic
  block model, degree-preserving rewiring).
- **Estimator contract** (best_feedforward is *estimated*, so be explicit — full version in
  `randomgraph.md`):
  1. Use a **Track-C runner**, not `eval/run_variant.py` (it only accepts registered datasets, and
     `H35.run` keys its budget on `g.name` → a generated graph silently gets defaults).
  2. **Size-scaled, family-invariant budget** held constant across graph families; log epochs/sweeps.
  3. Report a **greedy-FAS lower bound** beside the H35 estimate:
     `unavoidable_feedback ∈ [total − H35, total − greedy]`; trends must hold for **both** bounds,
     else it's an estimator artefact, not structure.
  4. Score every ordering with the frozen `mfas.metrics`; the runner writes provenance JSON
     (`git_commit`, `config_hash`, seeds, `n_nodes`, generator params, `estimator_budget`) to
     `results/randomgraph/*.json`.
- **Leakage:** none — random graphs have no reference solution (the `gap.py` firewall does not
  apply). The only risk is **estimator bias**, handled by the lower-bound control (#3). Generators
  live in `src/mfas/randomgraph/` and must be **answer-free** — distinct from the answer-carrying
  synthetic generators in the privileged `mfas.analysis.gap`.
- **Verdict:** report mean ± std over ≥K graph realizations per model at fixed seeds; a real-vs-null
  difference counts only if it exceeds realization noise (report a CI). State K.

---

## Phase-7 addendum: the autonomous campaign (added 2026-08-09, branch `auto/campaign`)

Phase 7 runs the Track-A loop **unattended** in an isolated git worktree. Everything above still
holds. Three rules change or are added; the constitution is `autoresearch/CAMPAIGN.md`.

### 1. The comparator is the CHAMPION, not the baseline

Phases 3–6 compared each variant to `baseline_passthrough` (random-init Rocket). By Phase 7 the
pipeline is three mechanisms deep (H02 warm-start → Rocket → sift), so beating the original
baseline is no longer evidence of progress. The comparator is now the per-dataset champion in
`autoresearch/sota.json`, at a matched role.

Δ versus `baseline_passthrough` and `H02` is still reported, so every number stays comparable with
`findings.md`; only the **verdict** moved.

**Screen thresholds (champion comparator).** The Phase-3 thresholds were 2σ of the *baseline*
noise floor. Against a lower-variance champion that gate is mis-scaled, so Phase 7 uses 2σ of the
**champion's** dispersion, per dataset, from `campaign.yaml`:

| dataset | champion σ | screen threshold | note |
|---|---|---|---|
| connectome | 0.0060 pp (H35, n=3) | **0.012 pp** | was 0.04 pp against the baseline floor |
| microns | 0.0012 pp (H30, n=5) | **0.002 pp** | unchanged; already champion-scaled |
| mouse | 0.0000 pp (saturated) | non-inferiority only, > −0.26 pp | unchanged |

The screen remains a lenient gate, not a verdict. CONFIRM (Welch 95% CI lower bound > 0 on both
primaries) is unchanged and is still what may be promoted.

### 2. Runtime is a hard constraint, not a free variable

Any run exceeding **3600 s wall-clock per (dataset, seed)** fails on usability grounds regardless
of score: an algorithm that cannot answer within an hour is not a usable result for this thesis.
`total_grad_steps` remains the equal-compute basis for fairness; wall-clock is now additionally a
constraint. Phase 2 of the campaign turns wall-clock into an objective — that phase does not begin
without human sign-off.

### 3. Mechanical audit before promotion

`autoresearch/audit.py` re-derives every claimed number directly from `results/*.json` and
re-scores stored positions with the frozen oracle. It must exit 0 before any champion changes, and
`autoresearch/sota.json` is writable only by `autoresearch/update_sota.py` (which runs the audit
itself and refuses on failure). The critic adjudicates; the auditor computes.

Two failure modes it exists to catch, both already observed in this repo:
- **Moving comparator** — `config_hash` covers only (algo, dataset), so it cannot distinguish
  `H30@12` from `H30@40`. Pooling them silently biases every delta. The auditor warns and breaks
  the comparator down by commit.
- **Prose drift** — a number quoted in a document that no longer matches the JSONs behind it.

### 4. The screen's seed count is a property of the VARIANT, not a constant (P02, 2026-08-09)

Phases 3–6 screened every variant at 3 seeds (42/123/999). That was correct when every pipeline
started from a random init. It stopped being correct at H02: the champion pipelines take a `seed`
and **never draw from it** — `init_positions` comes from deterministic greedy-FAS, so
`make_init_positions`, the only RNG consumer in `run_rocket`, is never reached
(`src/mfas/baseline/rocket.py:168`). On CUDA, where the device also reproduces bit-identically,
seeds 42/123/999 are the *same computation three times*. On microns (3240 s/run) that is 108
minutes of screen wall-clock buying nothing.

**The rule.** A variant is classified once, mechanically, by
`autoresearch/seed_class.py --variant <id>`, which walks every function reachable from the
variant's `run()` and looks for an RNG draw:

| classification | screen seeds, connectome + microns | screen seeds, mouse | confirm seeds |
|---|---|---|---|
| `deterministic` | **1** (seed 42) | 3 | unchanged (full `confirm_seeds`) |
| `rng` | 3 (42/123/999) | 3 | unchanged (full `confirm_seeds`) |

Four constraints make this safe, and none of them is optional:

1. **Static, not empirical, and fail-safe.** The classifier answers `rng` on any call path it
   cannot resolve. A false `rng` costs wall-clock; a false `deterministic` silently discards real
   variance and corrupts a verdict, so every ambiguity resolves toward the expensive answer. Its
   behaviour is pinned by `tests/test_seed_class.py`, not left to a script nobody re-checks.
2. **Static classification OVERRIDES the empirical probe, in one direction only.** A variant may
   look seed-inert on a proxy and still draw — H31 does (`RandomState(seed + 7919)`), but on mouse
   its LNS stage never beats the sift, so best-by-oracle returns the deterministic sift vector and
   the seed leaves no trace. The mouse probe may therefore *escalate* a variant to 3 seeds; it may
   never de-escalate one to 1.
3. **Mouse stays at 3 seeds as a standing tripwire.** It costs ~6 s/run, so there is nothing to
   save, and three mouse runs that are not identical falsify a `deterministic` classification
   before any expensive number is quoted. A 1-seed screen whose mouse runs disagree is void.
4. **`confirm_seeds` is unchanged.** Confirm is where a promotion happens, and re-running a
   deterministic variant at 5 seeds is not information-free: it re-establishes *device*
   repeatability, which is a genuinely non-zero source of variance and the only one left here.

**What is and is not proven equivalent.** The screen verdict is the point-estimate rule
`Δ > screen_delta_pp`, a function of the pool mean; for a variant with bit-identical runs the
1-seed and 3-seed means are equal exactly, so the screen verdict is provably unchanged. That is
the whole claim. It does **not** extend to confidence intervals: `audit.protocol_se()` is
`max(σ_c, floor)·√(2/n_c)` — comparator-only — so its value is unchanged at n_v=1 by construction,
not by evidence. Were a CI ever computed on a 1-seed pool its SE would be ~41% wider
(`√(1/1+1/3)` vs `√(2/3)`). Keeping the full `confirm_seeds` is what guarantees no CI in a
promotion path is ever computed on a 1-seed pool.

**Device-determinism is a precondition, established per machine.** The rule rests on the device
reproducing a repeated seed bit-identically. That is a property of the hardware, not of the code,
so it is re-established on every port — the procedure is `autoresearch/PORTING.md § 6b`. If it
fails, the screen stays at 3 seeds and the observed spread is device noise, not seed variance.

**Every log entry must record the classification** of the variant it reports, so no later reader
has to guess which regime a number came from.

### Isolation

The campaign runs in a worktree at `../../mfas_autoresearch` on branch `auto/campaign`, with
`data/` symlinked to the original repository (read-only). A PreToolUse hook blocks writes outside
the sandbox, to frozen files, to `data/best_solution`, to `results/*.json` and to
`autoresearch/sota.json`. The campaign never pushes and never merges into another branch —
integration is a human action.
