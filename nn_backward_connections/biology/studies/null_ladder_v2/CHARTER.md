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

**Decision (operator away, proceeding per standing instructions): G01 is the base — we do NOT
re-run it.** v2 = this charter + the rungs G01 does not have, each chosen because the G01 review
explicitly left it open.

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
  (weights stay on slots = travel with the source stub, the `w_source` convention). Repair
  self-loops by pairwise target swaps within the uni class; count — and repair by the same swap
  mechanism — rewired uni edges that accidentally create a *new* mutual dyad (with another uni
  slot or with a real mutual edge), so the dyad census is preserved up to a reported residual.
* **Distribution:** uniform over digraphs with (i) the real mutual subgraph fixed pointwise and
  (ii) the exact unidirectional in/out degree sequence, conditioned on loop-freedom and (up to the
  reported residual) no new mutual dyads.
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
  directional targeting. Sign check: fixing the mutual subgraph *forces* ≥6.72% FB into the null,
  so FF(L2R) could even sit *below* w_source on the FF axis while being the far stricter null —
  interpret jointly with the 2-cycle bound, not FF alone.

### Stage E (conditional go) — the MICrONS neuron-level ladder
The G01 review flagged (red-team note e) that fly↑ vs meso-mouse↓ confounds species, scale and
measurement. MICrONS (67,534 neurons, ~10.4M edges, neuron-level mammalian cortex) fills the
missing cell: L1, L2 (both weight variants), W0; real graph at 3 pipeline seeds; K = 8 per rung
(runtime honesty on a laptop); epoch budget pre-registered after ONE sizing run, then frozen.
Caveat carried from `layering/MICRONS_FINDINGS.md`: only 249/67,534 axons proofread — the graph is
truncation-biased; conclusions are about *this reconstruction*, stated as such.

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
| A | this charter; mapping of the operator's steps onto G01; housekeeping (`experiments/randomgraph.md` G01 status sync) | in progress |
| B | W0 generator + selftest + K=12 fly runs + review | pending |
| C | L2R generator + selftest + K=12 fly runs + review | pending |
| D | extended ladder analysis (v1 ∪ v2), 2-cycle bounds for new rungs, figure, REPORT.md + review | pending |
| E | MICrONS ladder (go/no-go after one sizing run) | pending |

## 4. Artifacts

* scripts: `biology/analysis/nullmodels2_generators.py`, `nullmodels2_run.py`,
  `nullmodels2_analyze.py` (import v1 machinery; own CONFIG).
* runs: `biology/results/nullmodels_v2/runs/*.json`; orders: `.../orders/*.npy` (int32 ranks).
* figures: `biology/outputs/nullmodels2_*.png`.
* report: `biology/studies/null_ladder_v2/REPORT.md`; reviews: `review_stage*.md` here.
