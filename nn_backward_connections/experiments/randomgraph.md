# Random graphs vs real brains — Track C (thesis goal #2)

**Core question.** Do real brains carry **more or less unavoidable feedback** than random graphs
matched on their size/degree structure — and which structural features explain the difference?

**Metric.** `unavoidable_feedback = total_weight − best_feedforward`, i.e. the weight that *cannot*
be made feedforward under any linear order. Lower best_feedforward ⇒ more unavoidable feedback.
Because best_feedforward is **estimated** (MFAS is NP-hard), we report it as a **bounded** quantity
(estimator + lower bound), never a single tuned number — see the estimator contract below.

This file **owns the live status** of each Track-C question (the roadmap only points here) and
holds the young track's **findings section** (bottom). Governance: `experiments/PROTOCOL.md`
§ "Track C — Random graphs vs brains".

---

## Estimator contract (READ before building a run)

best_feedforward must be estimated the *same way* for the real graph and every null model, or the
comparison is confounded by estimator/structure bias — which is exactly the headline claim. So:

1. **Own runner, not `eval/run_variant.py`.** The frozen runner only accepts registered datasets
   (`--dataset choices=list(io.DATASETS)`), and `H35.run` keys its budget on `g.name`
   (`_MAX_SWEEPS.get(g.name, 40)`, `_EPOCHS.get(g.name, …)`) — a generated graph silently gets
   defaults. Track C needs a thin runner that calls the pipeline on an in-memory `GraphData`,
   re-scores with the frozen `mfas.metrics`, and writes provenance JSON itself.
2. **Size-scaled, family-invariant budget.** Do **not** inherit H35's connectome-tuned constants.
   Fix an epoch/sweep policy as a function of graph size and hold it constant across ER / config /
   SBM / rewiring so no family is advantaged. Log the actual epochs/sweeps per graph.
3. **Bias control = greedy-FAS lower bound.** Report a cheap `greedy_fas_order` feedforward score
   **beside** the H35 estimate for every graph. `unavoidable_feedback ∈ [total − H35, total − greedy]`.
   Trends must hold for **both** bounds, or the conclusion is an estimator artefact, not structure.
4. **Frozen oracle for all scoring.** Every ordering (real or null) is scored by `mfas.metrics`
   (int64/float64). Provenance JSON → `results/randomgraph/*.json` with: `git_commit`, `config_hash`,
   `seed`, `generator` + params, `n_nodes`, `n_edges`, `estimator_budget`, `h35_pct`, `greedy_pct`.
5. **Leakage:** none — random graphs have **no** reference solution, so `data/best_solution` and the
   `gap.py` firewall do not apply here. The only risk is estimator bias (handled by #3).

**Null models (answer-free generators, `src/mfas/randomgraph/`):**
- Erdős–Rényi (match n, edge count / density).
- Configuration model (match the exact in/out degree sequence).
- Stochastic block model (match community structure — the *structured* null, the interesting one).
- Degree-preserving rewiring (randomise wiring, keep every node's degree).

> These live in `src/mfas/randomgraph/`, **distinct** from the answer-carrying synthetic generators
> in the privileged `mfas.analysis.gap` (`make_synthetic_graph`, `make_hard_synthetic_graph`): those
> plant a known near-optimal order (for diagnostics); Track C's must **not** — a random null has no
> planted answer.

**Verdict rule.** Report mean ± std over ≥K graph realizations per model at fixed seeds; a
real-vs-null difference counts only if it exceeds realization noise (report a CI). State K.

---

## Entry template (copy for a new question)

```
## G0x — <one-line question>
- **status:** open | in-progress | answered
- **null models:** <which of ER / config / SBM / rewiring, and what is matched>
- **method:** <generator params, K realizations, estimator budget policy>
- **artifacts:** results/randomgraph/*.json + plots
- **finding:** <link to the Findings § entry once answered>
```

---

## G01 — Does the fly connectome have more/less unavoidable feedback than matched random graphs?
- **status:** open
- **null models:** ER (match n + edges), configuration model (match in/out degree sequence),
  SBM (match communities), degree-preserving rewiring — the last two are the informative
  "structure-matched" nulls; ER is the naïve floor.
- **method:** for the connectome and K realizations of each null, estimate best_feedforward with
  the Track-C runner (size-scaled budget) **and** the greedy-FAS lower bound; compare
  `unavoidable_feedback` distributions. Then attribute any gap to a structural feature (degree
  heterogeneity, reciprocity, block structure).
- **artifacts:** `results/randomgraph/*.json` + a distribution plot (real vs each null).
- **finding:** → Findings § (to be written).

## G02 — Divide-and-conquer via SBM / spectral communities (cross-links A-CLU)
- **status:** open
- **note:** shares the clustering machinery with Track-A's `A-CLU`; here the goal is *analysis*
  (where does unavoidable feedback concentrate — within blocks or between?), not optimization.
  Mind the H33 spectral-cost warning (full-graph eigensolve 304–507 s, no AMG).

---

## Findings — Track C
*(empty — populated only when a G-question is answered under this track's verdict rule; a headline
result also earns a one-line pointer from `findings.md`.)*
