# Questions — Track B (diagnostics / understanding)

Open questions about *why* Rocket behaves as it does. Unlike Track A, the goal is to **explain**,
not to win — there is no SCREEN/CONFIRM gate. This file **owns the live status** of each question
(the roadmap only points here).

Contract: `experiments/PROTOCOL.md` § "Track B — Diagnostics". In one line:
- Every claim = statement + method + **cited artifact** (`experiments/outputs/*.json`, a plot, a
  table) + a **re-runnable command**. Never fabricated.
- **Leakage firewall:** any read of `data/best_solution` goes through `mfas.analysis.gap`;
  diagnostics **never write to `results/`** (that namespace is for the frozen runner only).
- A question is **ANSWERED** when the answer is written to `experiments/diagnosis.md` under a
  `## Q0x — <question>` anchor, and any finding it contradicts is corrected/annotated.
- **Numbering:** `Q01`+ for new questions. Historical H-series diagnostics (**H21**, **H22**) keep
  their IDs — do not renumber.

---

## Entry template (copy for a new question)

```
## Q0x — <one-line question>
- **status:** open | in-progress | answered
- **why it matters:** <what conclusion / finding depends on the answer>
- **method:** <how we'll answer it: script, probe, measurement — reuse mfas.analysis.gap for
  any best_solution read>
- **artifacts:** <experiments/outputs/*.json | plot paths — NEVER results/*.json>
- **answer:** <link to the `## Q0x` section in diagnosis.md, once answered>
- **touches:** <findings.md / diagnosis.md sections this may correct>
```

---

## Q01 — Why does starting Rocket from the best solution drift the score DOWN?
- **status:** open  *(first task — ~80% pre-worked in `dr_tmp/`, needs consolidation)*
- **why it matters:** `findings.md` #3 currently claims the best order is "unreachable/unholdable
  by Adam-on-σ under **every** schedule/scale". Prior scratch work contradicts the "every scale"
  strength: it is **holdable at the right scale**. The finding needs correcting for thesis
  correctness (this is a rigor issue, not a new win).
- **method:** consolidate the two existing probes and make the scale dependence explicit:
  - `dr_tmp/drift_from_optimal_spacing.py` — from the true low-loss point P* (surrogate-optimal
    spacing of the best order, std≈141), constant β, small lr → does the discrete score hold?
    (result so far: **holds 84.61%** for lr ∈ {5e-4, 5e-3, 5e-2}; ‖∇F(P*)‖≈5e-4, a real local max).
  - `dr_tmp/drift_scale_sweep.py` — same shape rescaled across std ∈ {0.58 … 5e4}: quantify the
    drop vs scale (drop 4.55 at std 0.6 → 0.44 at std 141 → ~0 at std 5e3). The logged "collapse"
    used even-spacing init at std≈0.58, ~200× below Rocket's operating scale.
  - fill the intermediate std grid (2/10/50/500) and save a disc-vs-step plot.
- **artifacts:** promote the two scripts to `experiments/diagnostics/`, fix their hardcoded
  `results/*_positions.npy` path (point at the committed `results/rocket_best_positions.npy` or
  regenerate), write a `experiments/outputs/drift_scale.json` + plot. **No writes to `results/`.**
- **answer:** → `diagnosis.md` `## Q01` (to be written); then annotate `findings.md` #3.
- **touches:** `findings.md` #3 (soften "every scale"; the reachability-from-cold-start and the
  optimization-gap sub-claims are unaffected and stay).

## Q02 — How far apart are Rocket solutions across seeds (no greedy warm-start)?
- **status:** open
- **why it matters:** characterises the basin structure (motivated the killed multi-start H01) and
  feeds the Track-C narrative on structural determinism of feedback. Partly done already — this is
  an **extension**, not a fresh start.
- **method:** run baseline Rocket (random init, no greedy) across ≥5 seeds; measure ordering
  distance — pick and justify one of Kendall-τ / Spearman / Jaccard@k — and split **sources vs
  sinks** (the existing `experiments/rocket_base.ipynb` analysis found sinks far stabler:
  Jaccard@1000 ≈ 0.65 vs 0.33). Report score dispersion alongside ordering dispersion.
- **artifacts:** `experiments/outputs/seed_distance.json` + plot. **No writes to `results/`.**
- **answer:** → `diagnosis.md` `## Q02` (to be written).
- **touches:** none yet (new measurement); may inform the H01 kill rationale.
