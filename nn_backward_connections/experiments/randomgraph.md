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
- **status:** **answered (2026-09-30)** — `biology/studies/nullmodels.md` (689 logged runs,
  independent review 2026-09-30; FINDINGS.md A1). Nested ladder chance → G(n,m) →
  configuration (two weight variants) → dc-SBM on the 17 lanes → real: the fly is outside all
  76 null draws at every rung (decomposition: density +19.8 pp, degrees+strengths +10.2,
  regional diagram +1.9, sub-annotation targeting +1.8), yet carries a reciprocal 2-cycle core
  (6.72% of W) 18–930× larger than any null's. Mouse meso: the opposite direction — real below
  98/100 strength-matched nulls, ~80% of its FB forced by bidirectional area pairs.
- **deviations from the menu below, stated in the study §2:** ER realized as G(n,m)
  weights-on-slot; "configuration model" and "degree-preserving rewiring" collapse into one rung
  (a global target-column permutation IS a uniform draw from that ensemble); SBM uses the
  17 annotation lanes, not inferred communities. Runs live under `biology/results/nullmodels/`
  (own runner per the estimator contract), not `results/randomgraph/`.
- **finding:** → Findings § below.

## G03 — Null-ladder v2: which axes G01 left open (weights-on-topology, reciprocity, cell types; MICrONS)
- **status:** **answered on the fly (2026-11-01)** — `biology/studies/null_ladder_v2/REPORT.md`
  (39 logged runs: 3 rungs × K=12 + three 5×-budget probes; stages A–D each independently
  reviewed, PASS; probes logged for every rung). The conditional MICrONS arm stays **parked**
  (CHARTER.md §6), so the scale-vs-species deconfound (G01 red-team note e) remains open.
- **owner doc (charter + pre-registration):** `biology/studies/null_ladder_v2/CHARTER.md`
- **headline (protocol-conditional, all three directions survive the 5× probe):**
  (1) **W0** `w_topo_shuffle` — real binary topology, weights permuted: FF 82.221 ± 0.033,
  real above all 12 draws (+1.44 pp; probe +1.25) — weight placement carries estimator-visible
  order; which weight–topology coupling is not identified at this rung.
  (2) **L2R** `dyad_fixedmut` — real reciprocal backbone + exact degrees, 2-cycle floor equal
  to the real 6.7172% of W bit-exactly in all 12 draws: real above all draws by +6.08 pp
  (probe +6.31) — order in the choice of non-reciprocal targets beyond degrees + backbone;
  the backbone move explains ~40% of the w_source→real excess-FB gap (19.91 → 15.70 vs 9.62).
  (3) **L4** `type_dcsbm` (K = 8,857 cell types, 99.09% direct coverage) — INVERSION: real
  BELOW all 12 draws (84.232 ± 0.009 vs 83.658, 62σ), probe-confirmed at 5× budget (84.756 vs
  pre-stated KEEP ≥ 84.2029 / stability ≥ 84.396; the inversion grows); on the registered
  excess-FB-over-own-floor axis the real graph still beats L4 by 4.13 pp (floors 6.72 vs
  2.02% of W) — the sub-type residual is spent on reciprocity, not hierarchy.
- **null models:** W0 `w_topo_shuffle`; L2R `dyad_fixedmut`; L4 `type_dcsbm` (added at stage B,
  registered before any L4 run existed — adjudication in `review_stageD_l4.md` §G).
- **method:** identical estimator to G01 (`nullmodels_run.run_protocol`), fly K=12 per rung,
  real comparators reused from G01's runs; 2-cycle bound per draw; G01's missing w_source
  bounds backfilled for all 76 v1 nulls in `biology/results/nullmodels_v2/bounds_cache.json`.
- **artifacts:** `biology/results/nullmodels_v2/{runs,orders}/*`, `summary_v2.json`,
  `bounds_cache.json`; figure `biology/outputs/nullmodels2_fly_ladder.png`; reviews
  `biology/studies/null_ladder_v2/review_stage{A,B_w0,C_l2r,D_l4,F_report}.md`.
- **finding:** → Findings § below; `biology/FINDINGS.md` A7.

## G02 — Divide-and-conquer via SBM / spectral communities (cross-links A-CLU)
- **status:** open
- **note:** shares the clustering machinery with Track-A's `A-CLU`; here the goal is *analysis*
  (where does unavoidable feedback concentrate — within blocks or between?), not optimization.
  Mind the H33 spectral-cost warning (full-graph eigensolve 304–507 s, no AMG).

---

## Findings — Track C

* **G01 (2026-09-30, reviewed):** brains are not generically "more feedforward than chance" —
  the answer is scale-dependent. Neuron-level fly: MORE feedforward-organized than every matched
  null at every rung (16.3% unavoidable-FB upper bound vs 18.1–30.2% for the nulls), while
  carrying a reciprocal core that provably forces 6.72% of W into feedback (18–930× any null's).
  Region-level mouse: LESS orderable than 98/100 strength-matched nulls (provable against 54/100
  via the 2-cycle bound), ~80% of its feedback forced by bidirectional area pairs. Full study +
  independent review: `biology/studies/nullmodels.md`; COSYNE-ready form: `biology/FINDINGS.md` A1.

* **G03 (2026-11-01, reviewed):** below cell-type resolution the fly's 84% is NOT generic —
  real beats every matched ensemble including real-topology/permuted-weights (+1.44 pp) and
  real-reciprocal-backbone + degrees (+6.08 pp on an identical, bit-exact 2-cycle floor). At
  type resolution the attribution closes and INVERTS: the K=8,857 type-dcSBM ensemble overshoots
  the real graph by 0.57 pp (real below all 12 draws, probe-confirmed at 5× budget), while the
  real graph keeps a 3.3× larger irreducible reciprocal core and still wins by 4.13 pp on
  excess FB over own floor. Protocol-conditional throughout. Study + five reviews:
  `biology/studies/null_ladder_v2/`; COSYNE-ready form: `biology/FINDINGS.md` A7.
