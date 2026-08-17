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

## Q01 — Why does starting Rocket from the best solution lose score?
- **status:** **answered (2026-08-17)** — `experiments/diagnostics/q01_surrogate_ranking.py` →
  `experiments/outputs/q01_surrogate_ranking.json` → `diagnosis.md` § Q01; `findings.md` #3 rewritten.
  **Answer:** Rocket optimizes coordinates, not an order, and its surrogate is a different objective at
  every scale (everything depends on the product β·std).
  - **β·std → 0:** the edge sum telescopes, `F = Ŵ/2 − (β/4)·⟨c,P⟩ + O((β·std)³)` — the surrogate stops
    being a relaxation of the feedforward objective and becomes "sort by weight imbalance". Measured at
    std = 0.001 it ranks the imbalance sort (69.63% discrete) **above** Rocket (82.92%) **above** best
    (84.61%) — exactly inverted — and the linear model reproduces F to 3.1e-10 relative.
  - **β·std ≈ 148 (Rocket's operating point):** best's true advantage +296.02 is outweighed by its
    8.9× larger smoothing loss (530.44 vs 59.54) → net **−174.88**: the surrogate prefers the worse
    order. Narrow wins are discounted toward σ = 0.5.
  - **β·std ≈ 470:** the crossover where best finally wins — invariant across β ∈ {0.05, 0.3, 1.05},
    which independently verifies that F sees only β·std. Rocket never gets there (β caps at 1.05).
  - **Not the sigmoid's shape:** sigmoid, slow tanh, hard clip and a cusped |x|^0.5 shape give the
    *identical* ranking at both scales. The bias belongs to the family `Σ_e ŵ_e g(Δ_e)`, not to σ.
  - **Explicitly NOT established:** anything about the optimum's stability. A "hold" at very large
    scale is a frozen optimizer (σ′ underflows; Rocket's own worse order holds equally, and a
    scale-matched lr reproduces a −0.39 pp drop).
- **why it matters:** it is the mechanism behind `findings.md` #3 and the reason rank-space discrete
  refinement (H30/H35) recovers what no continuous lever does.
- **artifacts:** `experiments/outputs/q01_surrogate_ranking.json` (primary), `q01_drift.json` +
  `q01_scale_sweep.png`, `q01_F_vs_scale.png` (scale sweep, β analysis). **No writes to `results/`.**
- **touches:** `findings.md` #3 (rewritten), `diagnosis.md` Step 1 + Selected-directions (scope of the
  "surrogate aligned" table now stated), `roadmap.md` Q01 row, `todo_origin.md` TODO 4.

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

## Q03 — How far is our current best order from the reference near-optimal one, and how degenerate is the near-optimal set?
- **status:** **open** — queued behind the S1/S2 collective-move sizing (`experiments/size_collective_moves.py`),
  because that sizing tells us which move class to characterise the residual with.
- **why it matters:** two separate reasons.
  1. **Every existing gap-structure number is stale.** The Step-2 measurements in
     `diagnosis.md` (8.47% of edges flip / 7.51% of weight; Spearman 0.757, Kendall-τ 0.610;
     flip rank-distance p25=8,290 / p50=22,580 / p90=87,497 from `localsearch_sizing.json`)
     were all taken against **Rocket/H02's 82.93% order**. H30+H35 have since closed **0.98 pp**
     of the 1.69 pp. Whether the sift removed the *short-range* part of the disagreement and
     left a purely long-range residual — or shrank it uniformly — is unknown, and it decides
     what the next move class must look like.
  2. **"How many variants are there?"** Q02 established that plain Rocket's ~82.9% level is a
     **degenerate set** of near-equivalent orderings (score std 0.0225 pp, Spearman 0.958).
     Whether the ~84.6% level is *also* degenerate is the open question: if many mutually
     distant orderings score ~84.61%, then "reach that specific order" is the wrong framing
     and the target is a *region*; if the top is essentially unique, the residual is a single
     hard reordering. This also bounds what any best-of-K / multi-basin scheme could ever buy.
- **method:**
  1. **Re-measure the gap structure against H35** (not H02) and put the two side by side:
     flip fraction and its gain/lose decomposition, Spearman / Kendall-τ, and the flip
     rank-distance percentiles. Reuse the machinery in `experiments/size_localsearch.py`.
  2. **Localise the residual:** which nodes carry it — the extreme sources Q02 found unstable,
     the giant SCC's interior, median-degree nodes (Step 2's finding)? Report the residual's
     concentration, not just its size.
  3. **Degeneracy at the top:** produce several distinct orderings scoring within ε of the
     reference (perturb → re-sift under the frozen oracle, keep those above a threshold) and
     report their pairwise Spearman / Jaccard@k. Compare that spread with Q02's ~82.9% spread.
     State K and ε; a null result ("we could not find a second distant 84.6% order") must be
     reported as such and not as evidence of uniqueness.
  4. **Reconcile the reference's provenance.** `data/best_solution` scores **84.6147%**
     (≈35,463,832) but Vahidi 2025 publishes **35,462,925 = 84.6125%** — ~907 weight units
     apart, so they are **not the same solution** even though `findings.md` #4 equates them.
     Verify both numbers through the frozen oracle and correct the docs.
- **artifacts:** `experiments/outputs/q03_gap_to_reference.json` + plots. **No writes to `results/`.**
  Privileged: reads `data/best_solution` **only** via `mfas.analysis.gap.load_best_solution`.
  Nothing produced here may feed a variant's init, loss or move choice.
- **answer:** → `diagnosis.md` `## Q03` (to be written).
- **touches:** `diagnosis.md` § "Gap structure (Step 2)" (annotate as measured-vs-H02);
  `findings.md` #3 (same), #4 (the Vahidi 35,462,925 vs 84.6147% conflation), #5 (the
  "~17% of the residual closed" framing depends on which reference is meant).
