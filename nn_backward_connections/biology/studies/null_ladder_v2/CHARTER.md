# Null-ladder v2 — charter, rules, and pre-registration

**Opened:** 2026-11-01 · **operator question (Aleksei + supervisor, 2026-11-01, translated):**

> Is 84.26% feedforward a lot or a little? Is it a generic property of any graph with these
> parameters — "you can almost always find such a representation" — or is the brain more
> structured, with less randomness? Randomize the connectome while preserving ever more of its
> statistics, re-run the same optimizer on each random graph, and read off, from the differences
> between randomization levels, **which property of the real graph carries the structure**.
> At each level, also state precisely what distribution the randomization represents.

## 0. Relationship to the existing study (READ FIRST)

The operator's proposed ladder (steps 1–3) **is already implemented, run, and independently
reviewed** as G01 — `biology/studies/nullmodels.md` (2026-09-29/30, 689 logged runs, review
2026-09-30 with 4 minor corrections, all load-bearing numbers confirmed). Mapping:

| operator's step | G01 rung | fly FF% (protocol) |
|---|---|---|
| (baseline below step 1) endpoints redrawn uniformly | **L1** `gnm_null` — directed G(n,m) | 69.823 ± 0.019 |
| step 1 *as literally stated* (permute src column and tgt column; weights stay on rows) | **L2** `w_shuffled` — note: permuting a column preserves its multiset, so this **preserves every node's in/out degree**; it is already the configuration model, with weights decoupled from both endpoints | 77.220 ± 0.024 |
| step 2 (keep per-neuron in/out counts, rewire) | **L2** `w_source` (weights travel with the source stub → out-strengths also exact) | 79.981 ± 0.015 |
| step 3 (+ keep per-region in/out counts, rewire within regions) | **L3** `block_configuration_null` — microcanonical degree-corrected SBM on the 17 annotation lanes | 81.896 ± 0.018 |
| real graph, same protocol | — | **83.658 ± 0.003** |
| real graph, champion budget (H64) | — | 84.258 |

Headline already measured: the real fly is outside **all 76** null draws at every rung; the
above-chance order decomposes as density +19.8 pp, degrees+strengths +10.2 pp, regional diagram
+1.9 pp, sub-annotation targeting +1.8 pp. Simultaneously the fly carries a reciprocal (2-cycle)
core that forces 6.717% of W into feedback — 18–930× more than any of its nulls. Mouse
meso-connectome: opposite direction (real below 98/100 strength-matched nulls).

**Decision (operator confirmed 2026-11-01): G01 is the base — we do NOT re-run it — but it is
accepted CRITICALLY:** where its design can be improved or better fitted to this project's
question, we change it (see §5). v2 = this charter + the rungs G01 does not have, each chosen
because the G01 review explicitly left it open. Operator priorities: **the fly brain first**;
the cell-type rung L4 is in scope; the MICrONS neuron-level ladder and the spatial
(wiring-length-preserving) null are **parked, not dropped** (recorded in §6).

## 1. New rungs (pre-registered)

Identical estimator to G01 (`nullmodels_run.run_protocol`: greedy-FAS warm start → Rocket-ASYM,
fly 4,000 epochs → under-relaxed sift, 8 sweeps, α=0.7, k_full=2; every discrete score from the
frozen oracle on CPU). Real-graph comparators are **reused from G01's runs** — same protocol, same
budget, same machine class (MacBook/MPS), so no re-measurement is needed or taken.

### Rung W0 — `w_topo_shuffle` (weights shuffled on the REAL topology)
* **Generator:** keep every edge (src, tgt) exactly as in the real graph; permute the weight
  column across edge slots with one uniform permutation.
* **Distribution:** uniform over the m! / (multiplicities) assignments of the real weight multiset
  to the real binary digraph — the microcanonical ensemble "real topology × exchangeable weights".
* **Preserves exactly:** the full binary digraph (hence all degrees, reciprocal pair set, SCCs,
  paths, motifs), weight multiset and total. **Destroys:** all weight–edge coupling.
* **What it isolates (missing axis in G01):** G01's ladder always randomized topology and asked
  how much structure the weights carry *on top*; W0 asks the complementary question — how much of
  the 83.66% is in **where the strong synapses sit**, with the wiring diagram fully real.
* **Note:** the 2-cycle *pair set* is real but the bound Σ min(w(u→v), w(v→u)) changes with the
  permutation — computed and reported per draw.
* **K = 12**, gen seeds 6000–6011, pipeline seed 42, fly first.
* **Predictions registered up front:** (a) FF(W0) ≈ real ⇒ the order is a property of the binary
  wiring; weight placement is decoration. (b) FF(W0) markedly below real ⇒ synaptic weighting is
  itself organized to support the feedforward hierarchy (strong synapses align with the order more
  than exchangeable placement would). Either outcome is informative; (b) would be the first
  evidence that *synapse counts*, not just wiring, encode the hierarchy.

### Rung L2R — `dyad_fixedmut` (degrees + the REAL reciprocal backbone)
* **Generator:** split edges into **mutual** slots (edges whose reverse edge exists: the
  reciprocal/2-cycle subgraph) and **unidirectional** slots. Keep every mutual slot exactly real
  (both directions, both weights). Permute the target column **among unidirectional slots only**
  (weights stay on slots = travel with the source stub, the `w_source` convention). Repair, by
  pairwise target swaps within the uni class, every slot that is a self-loop, lands on a
  real-mutual dyad (either direction), or whose reverse exists among current uni slots — so
  loop-freedom and the **no-new-mutual-dyads property hold EXACTLY** (enforced, not residual;
  verified by the selftest's dyad-census equality). The reported residual is a different,
  benign one: duplicate parallel uni slots (two uni slots landing on the same ordered pair,
  score-invariant per the v1 policy; `n_duplicate_slots`).
* **Distribution:** **approximately** uniform over digraphs with (i) the real mutual subgraph
  fixed pointwise and (ii) the exact unidirectional in/out degree sequence, conditioned on
  loop-freedom and no new mutual dyads — "approximately" because swap-repair biases the
  conditional (standard for edge-swap nulls); the repaired fraction is logged per draw
  (`n_bad_initial` / `n_uni_slots`; stage-A review measured ≈0.2–0.3% on the fly, 7–10% on the
  much denser mouse).
* **Preserves exactly:** every node's total in/out degree, uni in/out degree, mutual degree, total
  out-strength; the real 2-cycle bound (6.7172% of W) by construction. **Destroys:** which
  non-reciprocal target each neuron chooses.
* **What it isolates:** G01 found the fly's reciprocal core is 61–64× larger than its L2 nulls' —
  reciprocity is the most glaring statistic the L2 ensemble misses. L2R asks: **given the real
  reciprocal backbone and all degrees, is the remaining wiring still special?** Distance
  real − L2R = order that reciprocity + degrees cannot explain; L2R − L2(w_source) = what the
  reciprocal backbone itself buys/costs in orderability.
* **K = 12**, gen seeds 7000–7011, pipeline seed 42, fly first.
* **Predictions:** (a) FF(L2R) ≈ real ⇒ the fly's entire advantage over degree-matched chance is
  its reciprocal backbone plus degrees — the "fine targeting" of G01 would collapse to reciprocity.
  (b) FF(L2R) between w_source and real ⇒ reciprocity explains part; the rest is genuine
  directional targeting. (c) **FF(L2R) > FF(real)** ⇒ the mouse-direction outcome at neuron
  level: given its own reciprocal backbone and degrees, the fly's remaining wiring would be
  *less* feedforward-orderable than chance — the most surprising registered outcome, and a
  genuine claim because real and L2R share the identical 2-cycle floor (the real−L2R contrast
  is clean by construction). Sign check: fixing the mutual subgraph *forces* ≥6.72% FB into the
  null, so FF(L2R) can sit *below* w_source on the FF axis while being the far stricter null —
  interpret jointly with the 2-cycle bound, not FF alone.
* **Cross-rung statistic (adopted from the stage-A review):** beside raw FF%, every rung also
  reports **excess feedback over its own 2-cycle floor**, `(100 − FF%) − bound_2cycle_pct` —
  the estimator-visible feedback not already forced by reciprocity — which is the quantity
  comparable across rungs whose floors differ by 60×.
* **Timing note (honesty):** outcome (c) for both W0 and L2R, and the cross-rung statistic,
  were registered after the stage-A review (2026-11-01), when 6 of 12 W0 draws had already
  completed (draw gs6000's FF was visible: below real) but **before any L2R or L4 run existed**.
* **W0, the same registered branch:** FF(W0) > FF(real) would mean the real weight placement
  *hinders* orderability relative to exchangeable placement — also registered before W0's
  ensemble statistics were computed.

### Rung L4 — `type_dcsbm` (regions = the 8,840 cell types)
* **Generator:** v1's `block_configuration_null` (imported) with **cell-type labels** instead of
  the 17 lanes: per node, `cell_type` from `data/external/flywire_neuron_annotations.tsv.gz`
  (joined on root id; 98.9% coverage), falling back to `"<cell_class> (class)"` then
  `"unannotated"` — the exact `load_types` convention of `layering/tools/exp_mirror_swap.py`
  that finding A4 already uses. Target column permuted within each (type(src) × type(tgt))
  block; weights on slot.
* **Distribution:** microcanonical degree-corrected SBM at **type resolution** — uniform over
  graphs with every node's type-resolved in/out degree and out-strength profile fixed.
* **What it isolates:** the strictest biological null — "the type-level wiring diagram is the
  brain's program" (finding A4 reads cell types as computational units). FF(L4) vs real measures
  how much order lives BELOW type resolution; FF(L4) − FF(L3) prices the information the type
  diagram adds over the 17 lanes. Check and report whether the type partition refines the lane
  partition (expected ≈ yes), so the rung slots into the nested chain.
* **K = 12**, gen seeds 8000–8011, pipeline seed 42.
* **Prediction:** FF(L4) close to real (the type diagram pins most of the order); the residual
  real − L4 bounds individual-neuron (sub-type) targeting. If instead L4 ≈ L3, the type
  resolution adds little beyond lanes — which would *weaken* A4's "computational units" reading
  and must be reported as such.

### Differential budget probe for every v2 rung (G01 §5 improvement, adopted)
For ONE pinned draw per new rung (the first gen seed), re-run the protocol at 5× epochs
(20,000) + 40 sweeps and report the gap to the real graph's own 20,000-epoch value (84.1029,
already logged by G01). Guards the "not an under-optimization artifact" claim for the new rungs
exactly as G01 §5 did for the old ones.

## 2. Rules (inherited, non-negotiable)

1. Frozen oracle only (`mfas.metrics.score_from_order`, CPU); scorer and eval harness untouched.
2. Identical protocol for real and null — epoch-counted budgets, never wall-clock.
3. Every quoted number traces to a logged run JSON under `biology/results/nullmodels_v2/runs/`
   (v1 comparators: `biology/results/nullmodels/runs/`). No estimated/hand-edited numbers.
4. Seeds pinned; generators bit-reproducible under the same seed; selftest must pass before any
   null is scored.
5. The reviewed v1 files (`nullmodels.md`, `nullmodels_generators.py`, `nullmodels_run.py`,
   `nullmodels_analyze.py`) are **imported, never edited**.
6. After every stage: an independent review (subagent) that re-derives the invariants and
   recomputes the statistics from the raw runs with its own implementation; verdict files live in
   this folder (`review_*.md`). Nothing is promoted to `biology/FINDINGS.md` unreviewed.
7. One git commit per completed stage; this folder's REPORT.md owns the running conclusions.
8. All repo content in English (operator summaries in chat may be Russian).

## 3. Stages

| stage | content | status |
|---|---|---|
| A | this charter; mapping of the operator's steps onto G01; housekeeping (`experiments/randomgraph.md` G01 status sync) | done (review pending) |
| B | W0 generator + selftest + K=12 fly runs + review | runs in progress |
| C | L2R generator + selftest + K=12 fly runs + review | generator ready |
| D | L4 type-dcSBM: labels + selftest + K=12 fly runs + review | pending |
| E | budget probes (one pinned draw per v2 rung at 5× epochs) | pending |
| F | critical G01 fit-review (§5) + extended ladder analysis (v1 ∪ v2), 2-cycle bounds, figure, REPORT.md + Russian walkthrough artifact + review | pending |

## 5. Critical re-examination of G01 (operator mandate, 2026-11-01)

Accepted as the base, but these design points are re-examined rather than inherited blindly;
each gets a verdict (keep / improve / flag) in REPORT.md:

1. **Greedy-control flip on degree-matched rungs** — the pre-registered two-bound rule failed
   there and was replaced post-hoc by the 2-cycle bound + budget probe. v2 keeps the greedy
   column but treats the budget probe as the primary anti-artifact control (§1), now extended
   to every v2 rung.
2. **Protocol-conditional direction at fly L2/L3** (no estimator-free proof is possible with
   these bounds) — v2 states every new-rung direction with the same "upper bounds under an
   identical budget" wording; W0 is the one rung where topology-level statistics (greedy score,
   SCC structure) stay REAL by construction, which gives it an extra interpretive handle.
3. **K = 12 per rung** — kept for comparability and runtime honesty; the draw σ on every
   existing rung is ≈ 0.02 pp, ~100× smaller than the gaps read off the ladder, so K is not the
   binding constraint. Revisit only if a v2 rung lands within ~0.1 pp of the real value.
4. **One pipeline seed per null** — kept; justification re-checked (real across-seed σ 0.0033 pp
   is 5× below the smallest across-draw σ). If a v2 rung lands within ~0.1 pp of real, that rung
   is re-run at 3 pipeline seeds before any conclusion.
5. **Protocol (83.66) vs champion (84.26) slack, 0.60 pp** — inherited; all v2 ladder reading is
   done at protocol level, with the champion shown as a reference line only.

## 6. Parked directions (explicitly saved, not dropped)

* **Spatial null** — preserve the edge length distribution (3D soma positions; wiring-economy
  control): does geometry alone explain the orderability? Parked by operator decision
  2026-11-01 ("сохрани, но пока не делаем").
* **MICrONS neuron-level ladder** (L1/L2/W0, K=8, budget frozen after one sizing run) — fills
  the species-vs-scale confound cell of G01's headline (review red-team note e). Parked:
  fly first. Caveat to carry when un-parked: only 249/67,534 axons proofread.

## 7. Artifacts

* scripts: `biology/analysis/nullmodels2_generators.py`, `nullmodels2_run.py`,
  `nullmodels2_analyze.py` (import v1 machinery; own CONFIG).
* runs: `biology/results/nullmodels_v2/runs/*.json`; orders: `.../orders/*.npy` (int32 ranks).
* figures: `biology/outputs/nullmodels2_*.png`.
* report: `biology/studies/null_ladder_v2/REPORT.md`; reviews: `review_stage*.md` here.
