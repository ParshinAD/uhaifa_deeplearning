# Phase 7 — the autonomous campaign constitution

**Read this file at the start of every cycle.** It is the standing brief; `campaign.yaml` holds
the numbers, `state.json` holds where we are, and `experiments/PROTOCOL.md` (§ Phase-7) holds the
statistics. If this file and the code disagree, the code that produced a logged number wins —
then fix this file.

## Mission

Raise the exact feedforward percentage on the fly connectome from the current champion
**83.9101% (H35)** to at least **84.6147%** — the score of the downloaded reference solution,
independently reached by Vahidi 2025 with cheap greedy + bounded-span insertion + SCC and **no
MIP** — and then past it. Do not regress MICrONS or mouse. Keep every run under one hour.

Then, and only then, Phase 2: hold that quality and make it fast.

The gap is **0.70 pp**. It is known to be reachable by combinatorial means on this exact graph.
This is not a fishing expedition.

## The five things this campaign must never do

1. **Never touch a frozen file.** `src/mfas/metrics.py`, `eval/harness.py`, `eval/aggregate.py`,
   `tests/test_metrics.py` (+ `eval/frozen.sha256`, `eval/frozen_guard.py`). A hook blocks edits,
   the files are 0444, and `verify_frozen_manifest()` aborts any scored run if a hash moved.
2. **Never write outside the sandbox.** Everything lives under
   `/Users/abed359/IdeaProjects/university/mfas_autoresearch`. The original repo is read-only
   reference. Never `git push`, never merge into `main` or `phase6-global-discrete`.
3. **Never fabricate a number.** Every figure in every document traces to a `results/*.json` (or
   `experiments/outputs/*.json`) plus a re-runnable command. `autoresearch/audit.py` re-derives
   them mechanically; if the audit and the prose disagree, the prose is wrong.
4. **Never let the target metric into an algorithm.** The discrete score may be used exactly as
   the baseline uses it — track the best-scoring positions seen — and never inside a loss, a move
   choice, or a dataset special-case. The reference solution is for *measurement only*, read only
   through `mfas.analysis.gap`.
5. **Never claim a win inside noise.** Overlapping mean ± std is not a gain. One dataset is not
   three. A screen pass is not a verdict.

## The cycle

One cycle = one hypothesis carried as far as the evidence justifies, then written down.

```
preflight   git clean? frozen OK? pytest green? disk/budget OK? lock held?
plan        read state.json + queue.json + killed.json + sota.json -> pick ONE item
novelty     distinct from killed.json? if it shares an axis, name the revival condition
prototype   cheap proxies (mouse, hard synthetic, SCC subgraph). Minutes, CPU. Most ideas die here.
implement   isolated module src/mfas/experiments/<id>.py; nothing else changes
screen      3 seeds x 3 datasets, role=implement, vs the CHAMPION
verify      independent re-run, confirm seeds, Welch CI            [only if screen passes]
critic      red-team + autoresearch/audit.py                       [only if confirm passes]
decide      keep | kill | iterate -> experiments/log.md
record      update queue.json, killed.json, sota.json, state.json, DASHBOARD.md
commit      ONE commit per cycle, message: "phase7 <id>: <verdict> — <one line>"
```

A cycle that ends in **kill** is a successful cycle. Most will. Write the kill down properly —
with the revival condition — because the kill index is what keeps the campaign from looping.

## Gates, in order (never skip a rung)

| gate | question | cost |
|---|---|---|
| novelty | is this materially different from what is already dead? | free |
| prototype | does the mechanism show signal on a cheap proxy? | minutes |
| screen | does it beat the champion on 3 seeds x 3 datasets? | ~1 h |
| confirm | does the 95% CI lower bound clear 0 on both primaries? | ~3-6 h |
| critic | does it survive an adversary who wants it to be wrong? | minutes |

Phase 6 killed four hypotheses for almost no compute because the prototype gate ran first
(H32 −2.12 pp, H33 −4.97 pp and 300-500 s eigensolve, H34 −0.64 pp). Keep doing that.

## Comparator policy (Phase-7 change)

Variants are judged against the **current champion** in `autoresearch/sota.json`, per dataset —
not the original random-init baseline. Beating a superseded baseline is not progress.

Report the delta versus `baseline_passthrough` and `H02` as well, so the numbers stay comparable
with `findings.md`, but the **verdict** is against the champion.

**Comparator hygiene:** a variant id can denote different configurations at different commits
(`H30` at 12 vs 40 sweeps is the known case, and `config_hash` cannot see the difference — it
only covers algo+dataset). Always restrict the comparator by role, and check
`comparator_homogeneity` in the audit before quoting a delta.

## Escalation — what to do when it stops working

After **3 consecutive kills**, incremental mode is over. Do not propose a fourth variant of the
same design. Instead:

1. Run a literature scan (`scout`). Read something new: newer FAS heuristics, minimum linear
   arrangement, rank aggregation, DAG structure learning, anything with a transferable *move
   class*. Write it to `autoresearch/lit/`.
2. Re-read `diagnosis.md` Q01/Q02 and ask what the mechanism *predicts* should work — Q01 is a
   sharp instrument: it says the continuous relaxation is scale-blind at the operating scale,
   which is why discrete refinement works and why every continuous lever failed.
3. Attack a different level: the move class (single node → segment → block), the decomposition
   (flat line → SCC-recursive), the objective's structure, or the problem formulation itself.
4. Only then propose.

Also force a literature scan every 8 cycles regardless of outcomes, and ideate whenever the
queue has fewer than 3 viable items.

## How not to fool yourself

These are the failure modes this project has already survived once; they are cheap to repeat.

- **A moving comparator.** See comparator hygiene above.
- **A gain that lives on one dataset.** MICrONS has a 0.0006 pp noise floor — 29× tighter than
  connectome. If an effect is real and general it will show there. If it only shows on mouse
  (σ = 0.26 pp), it is almost certainly noise (this is what the SMALL-GRAPH ARTIFACT class is for).
- **A gain that is really extra compute.** Multi-start and refinement variants must be matched on
  `total_grad_steps`, not on wall-clock.
- **Double-counting an increment.** If a pipeline chains A → B, credit B with (A+B) − A, not with
  the whole thing. H30's sift increment was credited this way.
- **A scale artefact.** Q01 is the cautionary tale: a probe started 200× below the operating
  scale produced a "collapse" that got written into a finding and stood for weeks. When a result
  surprises you, check the scales before you believe the mechanism.
- **Believing your own summary.** Run `autoresearch/audit.py` and read its numbers, not your
  memory of them.

## File contracts

| file | who writes | what |
|---|---|---|
| `autoresearch/queue.json` | planner / ideator / cycle | live hypothesis queue + status |
| `autoresearch/killed.json` | cycle (on kill) | do-not-repeat index + revival conditions |
| `autoresearch/sota.json` | `update_sota.py`, after critic PASS | champion per dataset |
| `autoresearch/state.json` | cycle (end) + driver | phase, cycle no., streaks, budget |
| `autoresearch/lit/*.md` | scout | literature notes + transfer ideas |
| `autoresearch/DASHBOARD.md` | `dashboard.py` | human-facing status (regenerate, never hand-write) |
| `experiments/log.md` | cycle | the narrative record, one section per cycle |
| `experiments/findings.md` | cycle, on a confirmed win | ranked, evidence-backed conclusions |
| `src/mfas/experiments/<id>.py` | implementer | the isolated variant |
| `results/*.json` | `eval/run_variant.py` ONLY | per-run records |

Track B (diagnostics, `questions.md` → `diagnosis.md`) and Track C (random graphs,
`randomgraph.md`) keep their existing contracts. Diagnostics never write to `results/`.

## Hardware — the champion registry is machine-specific

`sota.json` records scores, not truths about the algorithm alone: they were produced by a
particular device (originally Apple MPS) with its own kernels and its own non-determinism. What
travels between machines is the **deterministic scorer** — `tests/test_metrics.py` scoring
`results/rocket_best_positions.npy` to exactly 34,751,902 — not a training trajectory.

So on **any change of machine or device** (MPS → CUDA, new GPU, different torch build):

1. `pytest tests/ -q` must be fully green first. If scorer parity fails, stop — nothing measured
   on that environment is admissible.
2. Re-measure the champions at the screen seeds on all three datasets (queue item **P01**), and
   update `sota.json` and the `screen_delta_pp` values in `campaign.yaml` to 2× the *new* σ.
3. Log the device name and torch version in the cycle entry.

Until P01 is done, every delta is against a foreign-hardware comparator — the moving-comparator
error, in a form the auditor cannot detect, because the device is not part of `config_hash`.

**One machine at a time.** Two machines running `auto/campaign` will diverge: both write
`sota.json`, `state.json`, `queue.json` and `experiments/log.md` every cycle, and the merge is
not mechanical. If you genuinely want two, give them different branches
(`auto/campaign-mps`, `auto/campaign-cuda`) and treat each as an independent replication — which
is scientifically useful, but never merge their `sota.json` automatically.

## Human checkpoints

The campaign runs unattended, but it stops and waits for a human when:

- the Phase-1 exit criteria are met (a champion at or past 84.6147%) — do not start Phase 2 alone;
- an audit FAIL cannot be explained by anything other than a bug in the harness;
- the frozen manifest fails;
- the budget in `campaign.yaml` is exhausted.

Write the reason into `state.json` (`mode: "blocked"`, with a note) and stop. `DASHBOARD.md` is
what the human reads first — keep it honest and current.
