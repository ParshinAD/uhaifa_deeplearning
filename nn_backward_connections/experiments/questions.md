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

## Q04 — Is the continuous family's failure a property of *any* g(Delta), or only of exponentially-tailed g?
- **status:** **answered (2026-08-17)** — `experiments/diagnostics/q04_surrogate_tails.py` →
  `diagnosis.md` § Q04; artifact `experiments/outputs/q04_surrogate_tails.json`.
  **Answer: it is a property of the whole family, and the sub-question the shape axis really
  poses is a trade-off, not an escape.** Every monotone bounded `g` aligns (ranks the better
  order higher) only above `beta*std ~ 200 * width(g)`, where `width(g)` is the `z` at which
  `g` reaches 0.9 — the ratio is 179–233 across eleven shapes, i.e. changing the shape is
  ~85% a rescaling of `beta` (Q01's already-closed axis). The genuinely new degree of freedom
  is the TAIL EXPONENT, which controls what a narrow core *costs*: at matched width 0.4954 a
  sigmoid freezes 42.5% of node gradients while an algebraic `z^-4` tail freezes 0.006%.
  Measured on the fly connectome; `tanh` is proved to BE the sigmoid (2.2e-16).
- **why it matters:** `diagnosis.md` Q01 asserted "it is not the SHAPE of the sigmoid" on the
  strength of a four-shape table that (a) had **no committed artifact** and (b) used
  `tanh(x/10)` as its "slower-decaying" arm — which is `sigmoid(x/5)`, i.e. a pure `beta`
  rescaling, so the tail axis it was meant to test was never varied. Q04 re-measures that
  table into a committed artifact and separates the two axes properly. It also closes the
  roadmap's `A-SURR` (TODO 7) with evidence instead of by analogy to H11.
- **method:** for eleven per-edge shapes (sigmoid; two tanh rescalings; algebraic tails
  `q ∈ {1,2,4}`; Cauchy/arctan; two hard-clipped variants; H11's clamp; a cusp) compute on the
  connectome, with even spacing for every order: the crossover `beta*std` at which
  `F_g(best) > F_g(rocket)`; that crossover normalised by the shape's own transition width;
  the scale-free alignment ratio at Rocket's operating point; the float32 gradient survival
  per edge; and the node-level zero-gradient fraction plus the gradient direction (cosine vs
  the sigmoid) at Rocket's **real converged positions**.
- **artifacts:** `experiments/outputs/q04_surrogate_tails.json`. **No writes to `results/`.**
  Privileged: reads `data/best_solution` only via `mfas.analysis.gap.load_best_solution`.
- **answer:** → `diagnosis.md` `## Q04`.
- **touches:** `diagnosis.md` § Q01 "It is not the SHAPE of the sigmoid" (gives it an artifact
  and corrects the `tanh(x/10)` arm's interpretation); `findings.md` #3 (adds the mechanism);
  roadmap `A-SURR` (closed via H37).

## Q05 — Does a ONE-SIDED (asymmetric) surrogate escape the trade-off Q01/Q04 found?
- **status:** **answered (2026-08-17)** — `experiments/diagnostics/q05_asymmetric_surrogates.py`
  → `diagnosis.md` § Q05; artifact `experiments/outputs/q05_asymmetric_surrogates.json`.
  **Answer: yes, structurally — the symmetry assumption, not the shape, was the constraint.**
  Every shape in Q01/Q04 was odd-symmetric (`g(-z) = 1 - g(z)`). A shape that is CONSTANT on the
  feedforward branch and tanh on the feedback branch (i) **does not telescope**, so it escapes
  Q01's small-scale imbalance degeneracy — at `beta*std → 0` it ranks
  `best > rocket > imbalance_sort > random`, where all 11 symmetric shapes rank the imbalance
  sort first — and (ii) reaches alignment ratio **+1.48 … +2.00** at Rocket's operating point
  (sigmoid: −0.591) with **no crossover at any scale**, *without* narrowing its core. The mirror
  shape (flat on the feedback branch) gives −1.30 … −3.72, so the DIRECTION of the asymmetry is
  what matters. Downstream: variant H38 gains **+0.37 pp on the connectome** but **regresses
  −0.67 pp on microns** → GRAPH-DEPENDENT (see `backlog.md` § H38, `log.md`).
- **why it matters:** it bounds `findings.md` #3 (which asserts that *no* continuous/gradient
  lever closes any of the gap) and Q01's "it is not the SHAPE of the sigmoid" — both are true of
  the symmetric family they tested and false as universal statements.
- **method:** nine shapes (sigmoid; `asym_flat_pos` and its mirror at T ∈ {0.5, 1, 1.4925, 3}) on
  the connectome, even spacing per order: small-scale ranking; crossover; alignment ratio at the
  operating point; the split of gradient mass between feedforward and feedback edges (100% vs 0%
  by construction, confirmed); zero-gradient node fraction and gradient cosine vs the sigmoid at
  Rocket's real converged positions.
- **artifacts:** `experiments/outputs/q05_asymmetric_surrogates.json`. **No writes to `results/`.**
  Privileged: reads `data/best_solution` only via `mfas.analysis.gap.load_best_solution`.
- **answer:** → `diagnosis.md` `## Q05`.
- **touches:** `findings.md` #3 (scope: "no continuous lever" is false — banner added);
  `diagnosis.md` § Q01 ("not the shape") and § Q04 § 3 (both re-scoped to the symmetric family).
