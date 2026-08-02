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
- **status:** **answered (2026-08-01)** — `experiments/diagnostics/q01_drift_from_optimum.py` →
  `diagnosis.md` § Q01; `findings.md` #3 corrected. **The intuition holds:** from the true low-loss
  point P\* (best order's surrogate-optimal spacing, std≈53,626, a critical point with F(P\*) >
  F(Rocket)+307), small-lr **and Rocket-default-lr** Adam **hold 84.6147% exactly**. The logged
  "collapse" was a **scale artefact** (even-spacing init, std≈0.58, ~200× too small). Barrier is
  **reachability**, not stability.
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
- **status:** **answered (2026-08-03)** — `experiments/diagnostics/q02_seed_distance.py` →
  `diagnosis.md` § Q02. Score **stable** (connectome 82.8845 ± 0.0225 pp) but the **order is not**
  (Spearman 0.9577 ± 0.0012): a degenerate set of near-equivalent orderings. Genuine source/sink
  asymmetry — **extreme sinks stabler than extreme sources** (Jaccard@1000 back 0.68 vs front 0.37) —
  but a **tail effect that closes by k=10000** (0.783 vs 0.794). Retroactively explains the H01
  multi-start kill (no score tail ⇒ best-of-K harvests nothing).
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
