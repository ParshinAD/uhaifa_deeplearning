# Independent review — null-ladder v2, Stage D (rung L4 `type_dcsbm`, fly, K=12) — THE INVERSION

**Reviewer:** independent subagent, 2026-11-01 · **scope:** the 12 Stage-D run JSONs
`biology/results/nullmodels_v2/runs/connectome__type_dcsbm__gs8000..8011__ps42__e4000.json`,
the 12 saved rank vectors in `biology/results/nullmodels_v2/orders/`, the sweep log
`biology/results/nullmodels_v2/l4_sweep.log`, the type-label construction, and the stage's
draft headline/interpretation. Stage-A/B/C ground is approved in `review_stageA.md`,
`review_stageB_w0.md`, `review_stageC_l2r.md` and **not redone**. This stage produced the
study's most surprising result — the real graph BELOW its null ensemble — so this review is
deliberately adversarial on the inversion (§G–§H).

**Instruments:** reviewer-written scripts (scratchpad, not repo artifacts): `stageD_stats.py`
(every statistic recomputed straight from the raw JSONs; json + numpy only, no `nullmodels2_*`
import), `stageD_labels.py` (reviewer-own pandas TSV join, structurally different from the
study's csv.DictReader; fallback/caveat quantification), `stageD_regen.py` (4 draws
regenerated from logged `gen_seed`; the study's `fly_type_labels` +
`block_configuration_null` imported for regeneration parity ONLY, as in stages B/C; **all
invariants verified with reviewer-own code**: full-coverage sort-based multiset checks,
scipy.sparse `min(A,Aᵀ)` int64 2-cycle bound, explicit per-node dictionary check on a
1,000-node sample, frozen-oracle re-scores via `mfas.metrics.score_from_order` on CPU).
All CPU; torch never imported; GPU untouched (budget probes running). **398 integrity checks
+ 80 regeneration checks + the label-partition equality + 12/12 distinct order hashes —
0 failures.**

---

## A. Claims table (headline as circulated → reviewer recomputation)

| claim | reviewer recomputation (raw JSONs / own code) | verdict |
|---|---|---|
| FF(L4) = 84.2318 ± 0.0086 (K=12) | mean **84.231771**, sd(ddof=1) **0.008573** | CONFIRMED |
| min 84.2187, max 84.2452 | min **84.218697** (gs8011), max **84.245238** (gs8004) | CONFIRMED |
| real BELOW all 12 draws, z = −66.9 | real 83.6582 ± 0.0033 (3 G01 seeds, recomputed); highest real seed **83.661262** < lowest draw **84.218697** by **0.5574 pp** → below all 12; z = (83.6582 − 84.2318)/0.008573 = **−66.90** | CONFIRMED |
| inversion FF(L4) − FF(real) = +0.574 pp | **+0.573564 pp** (matched-seed ps42-vs-ps42: **+0.577** — larger) | CONFIRMED |
| null 2-cycle bound 2.0181 ± 0.0062 % of W | mean **2.018079**, sd **0.006206** (abs 840,947–850,962) | CONFIRMED |
| real 2-cycle bound 6.7172 % | reviewer-own sparse implementation: **2,815,303 = 6.717154 %**; matches `bound_2cycle_real` in all 12 JSONs; floor ratio 6.7172/2.0181 = **3.33×** | CONFIRMED |
| greedy 69.462 ± 0.010 vs real 68.913 | mean **69.461805**, sd **0.009591**; real **68.913428** (identical across 3 seeds) | CONFIRMED |
| excess FB over own floor: L4 13.750 vs real 9.625 | per-draw (100 − FF) − bound: mean **13.750150 ± 0.012220**; real (100 − 83.6582) − 6.717154 = **9.624639** | CONFIRMED |
| FF(L4) within 0.03 pp of the champion-budget value | champion H64 (`autoresearch/sota.json`) pct_mean_exact **84.25817951**; 84.231771 − 84.258180 = **−0.0264 pp** | CONFIRMED (but see §H5: label it a cross-budget coincidence) |
| (supporting) FF(L4) − FF(L3) | L3 dc-SBM recomputed from 12 G01 JSONs: **81.896107 ± 0.017553**; delta **+2.3357 pp** | CONFIRMED (non-nested comparison — §E) |

## B. Integrity of the 12 runs

All of the following hold for **every** run JSON (programmatic, zero exceptions):
`epochs=4000`, `n_epochs_done=4000`, `sift_max_sweeps=8`, `n_sift_sweeps=8`,
`pipeline_seed=42`, `device="mps"`, `torch_version="2.8.0"`, `n_nodes=136,648`,
`n_edges=5,657,719`, `total_weight=41,912,141`, `kind="type_dcsbm"`, `dataset="connectome"`,
`gen_stats.seed == gen_seed`, gen seeds **distinct 8000…8011**, `gen_stats.n_blocks=456,602`
(matches the selftest), loops repaired == loops initial (1,491–1,590), every `pct_*`
consistent with its `score_*` to <1e-9, every score integer-valued, `score_final ≥
score_rocket ≥ 0` and `≥ score_greedy`, `unavoidable_fb_pct_final = 100 − pct_final`,
`bound_2cycle_pct` consistent with `bound_2cycle`, `bound_2cycle_real = 2,815,303` in all 12.
All 12 saved orders are valid int32 rank permutations of 0…136,647 and all 12 files are
pairwise distinct (sha256). Sweep log: 12 `[done]` lines whose final/greedy percentages match
the JSONs, no errors/tracebacks, only the benign torch lr_scheduler warning. One cosmetic
note: `gen_stats.variant` reads `"L3_dcsbm"` (inherited from the imported v1 function) while
`kind="type_dcsbm"` — any analysis must filter on `kind`, never on `gen_stats.variant`.
**PASS.**

## C. Spot checks on regenerated draws (the decisive check)

Four draws regenerated from logged `gen_seed` — gs8000 (first/pinned, the probe draw),
gs8004 (ensemble max), gs8007, gs8011 (ensemble min, last). In all four, `gen_stats`
reproduce **bit-identically** (`n_selfloops_initial`, `repair_rounds`, `n_duplicate_slots`,
`n_blocks`), and regeneration wall time (~1.4 s) matches the logged `t_generate_s`
(1.2–1.7 s) — no timing anomaly in generation.

(i) **Frozen-oracle re-scores** (`mfas.metrics.score_from_order`, CPU, regenerated graph +
saved rank vector):

| draw | oracle re-score | logged `score_final` | reviewer 2-cycle bound | logged |
|---|---|---|---|---|
| gs8000 | 35,307,335 | 35,307,335 | 850,962 (2.030347 %) | identical |
| gs8004 | 35,308,983 | 35,308,983 | 846,536 (2.019787 %) | identical |
| gs8007 | 35,299,621 | 35,299,621 | 846,505 (2.019713 %) | identical |
| gs8011 | 35,297,859 | 35,297,859 | 843,362 (2.012214 %) | identical |

4/4 integer scores and 4/4 bounds reproduce **exactly** (the bound with a structurally
different implementation: scipy.sparse elementwise `min(A, Aᵀ)`, int64, duplicates summed).

(ii) **Reviewer-own invariant verification, coverage stated.** FULL coverage — all 5,657,719
slots, all 136,648 nodes, all 456,602 occupied blocks — via per-slot and sort-based multiset
checks: src column untouched; weights bit-identical on their slots; per-slot target TYPE
unchanged (`type(t2) == type(tgt)` on every slot); zero self-loops; per-node in-degree
preserved; per-node **type-resolved in-profile** preserved (multiset of (tgt node, type(src))
pairs); per-node **type-resolved out-profile** preserved; per-node **type-resolved
out-strength** preserved (int64 aggregation); block edge counts AND block weight sums
preserved on **all 456,602 blocks**. Plus a structurally different second method: explicit
per-node dictionary (Counter) check on a seeded random sample of **1,000 nodes**, exhaustive
per sampled node, for both the in-profile and the out-(type, weight) profile. Sanity that the
null actually randomizes: 85.34 % of edges change target node (gs8000), and per-node total
**in-strength differs on ~124,4xx of 136,648 nodes** in each draw — in-strength is indeed NOT
preserved (see §H5 wording note). 4/4 draws, all checks pass. **PASS.**

## D. Label construction verified independently (reviewer-own join)

Reviewer-own pandas join of `data/external/flywire_neuron_annotations.tsv.gz` (139,248 rows,
0 duplicate root_ids) onto the connectome's `node_ids`, applying the documented convention
(`cell_type` → `"<cell_class> (class)"` → `"<super_class> (class)"` → `"unannotated"`):
**K = 8,857**, type list and label array **identical to the study's `fly_type_labels`
output** (exact array equality). Join coverage 136,635/136,648 nodes (99.99 %); nodes with a
**direct** cell_type: 135,402 (**99.09 %**). Fallback population: 21 `"<cell_class> (class)"`
pseudo-types (1,083 nodes), 4 `"<super_class> (class)"` pseudo-types (150 nodes),
1 `"unannotated"` (13 nodes) — **1,246 nodes = 0.91 % of all nodes**.

Charter-text corrections for REPORT.md (none affects a computed number): the section header
says "the 8,840 cell types" — measured **K = 8,857** (the commit message and selftest already
say 8,857); "98.9 % coverage" — measured 99.99 % join coverage / 99.09 % direct-cell_type
coverage; the charter's fallback sentence omits the `"<super_class> (class)"` step that the
code (and the cited `load_types` convention) actually implements (150 nodes).

## E. Refinement check — the charter's "expected ≈ yes" is CONTRADICTED (check 6)

Charter: "Check and report whether the type partition refines the lane partition (expected
≈ yes), so the rung slots into the nested chain." Measured (reviewer-own labels × the
reviewed v1 `fly_lane_labels`): **91 types spanning >1 of the 17 lanes, 43,580 nodes,
31.89 %** — exactly the selftest's numbers, independently reproduced. Decisive additional
fact: **all 91 spanning types are genuine direct cell types** (R1-6 with 7,167 nodes, KCg-m,
T2a, Tm3, T4c, …) — zero of them are fallback pseudo-types, so the non-refinement is a real
property of the FlyWire annotation vs the lane partition, not an artifact of the fallback
convention. Consequences, measured on regenerated gs8000: 34,090 edges (0.60 % of m,
**0.39 % of W**) land in a different lane than in the real graph, and lane-resolved per-node
profiles are **not** preserved (checked both directions). So the L4 ensemble is **formally
NOT nested inside L3's constraint set** — near-nested by weight (≤0.4 % of W escapes the lane
constraints), but non-nested all the same. **Required:** REPORT.md must state this as a
deviation from the registered expectation, and must present FF(L4) − FF(L3) = +2.34 pp as a
comparison of two *non-nested* ensembles (the L3a precedent), not as a clean ladder
increment "priced on top of lanes".

## F. Timing / device anomalies

t_rocket 21.25–21.60 s (max/min 1.016), t_total 58.0–64.9 s, t_generate 1.2–1.7 s (matches
reviewer regeneration ~1.4 s; the first draw's 1.74 s includes the one-time label load).
Identical to the Stage-B/C and Sept G01 null-run envelope; a silent CPU fallback is excluded
(4,000 epochs over 5.7 M edges cannot finish in 21 s on CPU; `device="mps"` logged).
Timestamps 16:39:57–16:52:37 UTC 2026-11-01, ~63 s cadence for 12 sequential runs,
immediately after the L2R sweep (ended 16:38). Same torch 2.8.0 as all comparators. No
anomaly. **PASS.**

## G. Which registered prediction fired — adjudication (the honesty question)

The charter's L4 lines, verbatim:

> **Prediction:** FF(L4) close to real (the type diagram pins most of the order); the
> residual real − L4 bounds individual-neuron (sub-type) targeting. If instead L4 ≈ L3, the
> type resolution adds little beyond lanes — which would *weaken* A4's "computational units"
> reading and must be reported as such.

**Adjudication: NEITHER registered L4 branch fired as written.**

* Branch 2 (L4 ≈ L3) clearly did not fire: FF(L4) is **+2.34 pp above L3**, 133× the L3 draw σ.
* Branch 1 ("close to real") fired **in magnitude only** (0.57 pp is small on a ladder
  spanning 69.8→83.7) — but its registered reading, "the residual real − L4 **bounds**
  individual-neuron (sub-type) targeting", presupposes a non-negative residual. The measured
  residual is **−0.574 pp**: there is no sub-type-targeting bound to read off, because the
  sign is inverted. An inversion was NOT a registered L4 branch.
* The FF(null) > FF(real) branch **was** registered in this charter — but for the sibling
  rungs: L2R outcome (c) ("the mouse-direction outcome at neuron level … *less*
  feedforward-orderable than chance") and "W0, the same registered branch". Git facts
  (verified): the **entire L4 section, the (c) branch, and the timing note were added in
  commit c77c040** (2026-11-01 16:25:38 UTC); the stage-A charter (e2a0424) had **no L4
  section at all**; the first L4 run started 16:39:57 UTC; the charter is **unchanged since
  c77c040** (empty diff to HEAD). So every relevant sentence predates every L4 run — by
  14 minutes for the L4 section itself (after the L4 selftest, which produces no FF score,
  so the prediction was still blind to L4's FF).

**No number was relabeled** — but the required wording in REPORT.md is: *neither registered
L4 branch anticipated the inversion; the "less orderable than its own null" interpretation is
imported from the (c)-branch registered (pre-any-L4-run, same document) for W0/L2R.* Writing
"the registered FF(null) > FF(real) branch fired at L4" without that qualifier would be
post-hoc relabeling and is not acceptable. The 14-minute registration-to-run gap and the
post-stage-A addition of the whole L4 section must stay visible (the charter's own timing
note already discloses the pattern honestly; extend it to say the L4 section itself is
stage-B vintage).

## H. The INVERSION red-team

Draft under review: *"type-resolved degree/strength profiles pin essentially all of the
fly's orderability (84.23 vs real 83.66 at the same budget), so the real wiring is LESS
feedforward-orderable than its type-matched ensemble; the deficit co-occurs with a 3.3×
higher reciprocal floor (6.72 vs 2.02) — at type resolution the fly pays for reciprocity
just as the mouse does at region resolution."* All numbers in it are confirmed (§A). The
attacks, in order:

### H1 (a) — estimator noise and the charter triggers: SAFE; triggers NOT applicable

Arithmetic: |FF(L4) − FF(real)| = **0.5736 pp**; the charter §5.3/§5.4 trigger is "a v2 rung
lands within **~0.1 pp** of the real value" → 0.5736 = **5.7× the trigger** → neither the
3-pipeline-seed re-run (§5.4) nor the K>12 revisit (§5.3) is required. Noise budget: draw σ
(12 gen seeds, fixed ps42) = 0.0086 pp → z = −66.9; real across-pipeline-seed σ = 0.0033 pp;
combined √(0.0086² + 0.0033²) = 0.0092 → z = −62.3. Shared-pipeline-seed caveat checked: all
12 draws use ps42, and real@ps42 = 83.6546 is the **lowest** of the three real seeds, so the
matched-seed comparison *widens* the gap (+0.5771). A pipeline-seed main effect would have to
be ~60× the entire observed real seed spread (0.0066 pp) to matter. Honest nonparametric
statement (keep beside z): all 3 real seed values sit below all 12 draws by ≥ 0.5574 pp;
with K=12, empirical one-sided p ≥ 1/13 — quote z = 66.9 as dispersion-relative distance,
never as a Gaussian tail.

### H2 (b) — differential optimizer difficulty: the ONE live threat; kill conditions, stated before the probe lands

The inversion could in principle mean "the L4 ensemble is easier for THIS optimizer at THIS
budget", not "more orderable". Evidence already in hand, both ways:

* The inversion is **not stage-specific**: it is present at greedy (+0.548), after Rocket
  pre-sift (83.1855 vs 82.676 = +0.509) and after sift (+0.574) — no single pipeline stage
  manufactures it.
* But greedy "agreement" is **uninformative**: greedy favors every structured null
  regardless of the final direction (W0 70.25, L2R 70.30, **L3 70.66** — all above real
  68.91, including L3 whose *final* is 1.76 pp BELOW real). Do not cite greedy as
  corroboration; the Stage-E budget probe is the decisive control.

**Kill conditions for the pending probe** (pinned draw gs8000 at 20,000 epochs + 40 sweeps;
comparator = the logged real 20k value **84.1029**, `connectome__real__ps42__e20000.json`).
Precedent, all from logged G01 pinned-draw pairs (4k → 20k): real 83.6546 → 84.1029
(**+0.448**); L3 dc-SBM gs5000 (same generator, coarser labels) 81.8687 → 82.4710
(**+0.602**); w_source gs2000 80.0090 → 80.3035 (+0.295); L1 gs3000 69.8130 → 70.7713
(+0.958). gs8000@4k = 84.2413; pinned 4k gap over real@20k-comparator-seed = +0.5867.
Expected FF(gs8000@20k) if precedent transfers: **84.53–85.20** (center ≈ 84.84, the
L3-like gain — note the dc-SBM family precedent gained MORE than real, which would widen
the inversion).

| probe outcome | reading | disposition |
|---|---|---|
| FF(gs8000@20k) **≤ 84.1029** | the draw lost ≥ 0.14 pp at 5× budget (vs precedent gains all ≥ +0.29): the 4k inversion was real-graph under-optimization | **KILL** the inversion headline |
| **84.1029 < FF < 84.2029** (gap < 0.1 pp, the charter's own proximity scale) | direction survives, magnitude collapses ≥ 5.9× | **DEMOTE** to "budget-conditional"; extend the probe (≥3 draws, 3 pipeline seeds at 20k) before any promotion |
| FF(gs8000@20k) **≥ 84.2029** | inversion robust to 5× budget | **KEEP** the direction; claim quantitative stability only if the 20k gap ≥ 0.29 pp (half the 4k pinned gap), i.e. FF ≥ 84.396 |

Even a KEEP leaves the direction **protocol-conditional**: both FF values are achieved lower
bounds and the 2-cycle upper bounds (real optimum ≤ 93.283 %, L4 optimum ≤ ~97.98 %) do not
separate the optima — no estimator-free proof exists at this rung. State the inversion as
"under the protocol estimator and budget" in every quotation, and promote nothing to
`biology/FINDINGS.md` before the probe is logged.

### H3 (c) — the fallback-label caveat: QUANTIFIED, reportable, NOT the driver

Reviewer-own numbers (§D): fallback pseudo-types hold **1,246 nodes (0.91 %)**; edge weight
touching a fallback node at either end: **0.40 % of W** (tgt side, where the permutation
freedom lives: 0.22 %); share of the ensemble's permutation freedom (log₂-cardinality
38,200,233 bits over all 456,602 blocks) contributed by blocks with a fallback endpoint
type: **0.13 %** (tgt-side 0.09 %). Hard adversarial bound: replacing the fallback
pseudo-types by maximally strict singleton types changes the graph family only on edges
touching those 1,246 nodes, so it can move the achievable FF by **at most 0.40 pp** — less
than the 0.574 pp inversion even in the absurd extreme where every gram of that weight flips
orientation; the realistic scale (that weight's FF swinging by a few pp) is ~0.01–0.06 pp.
Also: all 91 lane-spanning types are direct types (§E), so the fallback convention is not
what breaks nestedness either. **Required:** one sentence beside the headline reporting
0.91 % of nodes / 0.40 % of W / 0.13 % of ensemble freedom in coarse fallback types, worst
case ≤ 0.40 pp — then the caveat is closed. Also note separately: 2.19 % of m (~123.7k
slots) are duplicate parallel slots, score-invariant per the v1 policy, and 3.14 % of m
(2.84 % of W) sits in singleton blocks and is frozen exactly real.

### H4 (d) — inversion arithmetic and the cleanest consistent phrasing: VERIFIED

Exact identity (verified to all printed digits): excess-FB difference = floor difference −
FF difference: **4.1255 = 4.6991 − 0.5736**. Confirmed pieces: measured FB — L4 15.7682,
real 16.3418; floors — L4 2.0181, real 6.7172 (diff 4.6991); excess over own floor — L4
**13.750**, real **9.625** (diff 4.1255). Cleanest phrasing (two equivalent renderings;
REPORT.md should use one and may show the other in a footnote):

1. *Freed-floor accounting:* the type-matched ensemble frees 4.70 pp of W from the forced
   reciprocal floor but converts only 0.57 pp of it into extra achieved feedforward weight;
   the remaining 4.13 pp reappears as estimator-visible excess feedback.
2. *Fly-side accounting:* the real fly carries a 4.70 pp higher forced reciprocal floor than
   its type-matched ensemble yet lands only 0.57 pp below it, because net of floors its
   wiring is 4.13 pp **better**-ordered (excess FB 9.62 vs 13.75).

Note what this means for presentation: the inversion is an **FF-axis** fact; on the
pre-registered cross-rung axis (excess FB over own floor, stage-A vintage) the real graph
**still beats** its L4 ensemble by 4.13 pp. Both statistics are registered; REPORT.md must
show both, not the FF axis alone. Keep "estimator-visible" attached to every excess-FB use
(bound-gap wording, stage-B §E3), and never let "pays for reciprocity" become "the fly has
more feedback than its null" — measured FB differs by only 0.57 pp and in the *opposite*
direction of the floors.

### H5 — wording corrections to the draft reading (required)

1. **"pin essentially all of the fly's orderability" understates — the ensemble OVERSHOOTS.**
   The type profiles reproduce all of the real FF and 0.57 pp more. Say "pin all of it, and
   overshoot by 0.57 pp" (that overshoot IS the finding); "pins most of the order" (branch-1
   language) is no longer the right frame.
2. **"degree/strength profiles" — out-strength only.** The ensemble fixes type-resolved
   in/out DEGREE profiles and the type-resolved OUT-strength profile; per-node in-strength is
   free and in fact differs on ~124.4k of 136.6k nodes per draw (§C). Write "type-resolved
   in/out-degree + out-strength profiles".
3. **The mouse analogy is a co-occurrence, not a demonstrated mechanism.** Within this very
   study, when the full reciprocal backbone is held fixed (L2R), the real graph sits +6.08 pp
   ABOVE its null; the deficit appears only when sub-type structure — including the choice of
   reciprocal partners — is freed (L4). "The fly pays for reciprocity" is defensible only as
   the floor-accounting statement of §H4 (the FF deficit is more than covered by the 4.70 pp
   floor gap). Flag that the mouse inversion lives at a strength-matched L2-family rung on a
   148-node meso graph — different rung, different scale; keep "at type resolution … as the
   mouse does at region resolution" only with that disclosure.
4. **z = −66.9 is an effect size**, not a p-value; the nonparametric statement is "real below
   all 12 draws" (empirical one-sided p ≥ 1/13). Same as stages B/C.
5. **The champion proximity (−0.0264 pp) is a cross-budget, cross-graph coincidence.**
   84.2318 is a protocol-budget (4k) number on null graphs; 84.25818 is the H64
   champion-budget number on the real graph (charter §5.5: champion = reference line only).
   It may be quoted as a striking aside, never as "the ensemble reaches the champion" and
   never as a same-budget comparison. (The draft's "84.23 vs real 83.66 at the same budget"
   IS a correct same-budget comparison; keep the two comparisons clearly separated.)

## I. Required before these numbers enter REPORT.md

1. **No promotion of the inversion** (to REPORT.md conclusions or `biology/FINDINGS.md`)
   before the Stage-E gs8000 20k probe is logged; apply the kill/demote/keep table of §H2
   verbatim — the thresholds are pre-stated here precisely so they cannot move after the
   probe lands. [§H2]
2. State the prediction adjudication exactly as §G: neither registered L4 branch fired as
   written; the inversion reading imports the (c)-branch registered for W0/L2R before any
   L4 run; quote the charter's L4 lines verbatim; keep the registration timeline (L4 section
   added in c77c040, 14 min before the first run) visible. [§G]
3. State the refinement deviation: the type partition does NOT refine the 17 lanes (91
   direct types, 43,580 nodes, 31.89 %; per draw ~0.60 % of edges / 0.39 % of W cross lane
   boundaries); L4 is non-nested vs L3 (like L3a), so FF(L4) − FF(L3) = +2.34 pp is a
   non-nested comparison, not a ladder increment. [§E]
4. Report the fallback caveat with its numbers (0.91 % of nodes, 0.40 % of W, 0.13 % of
   ensemble freedom; worst-case ≤ 0.40 pp < the 0.574 pp inversion) — then closed. [§H3]
5. Present BOTH axes: FF inversion (real below all 12) and excess-FB (real 4.13 pp better
   net of floors), with one of the two §H4 phrasings; "estimator-visible" on every
   excess-FB use; "under the protocol estimator and budget" on every direction claim; greedy
   agreement labeled uninformative (greedy favors every structured null, incl. L3). [§H4, §H2]
6. Apply the §H5 wording fixes (overshoot; out-strength only; mouse-analogy disclosure;
   z as effect size; champion proximity as cross-budget coincidence).
7. Correct the charter constants when echoed in REPORT.md: K = 8,857 (not 8,840); direct
   cell_type coverage 99.09 % (not "98.9 % coverage" unqualified); the fallback chain
   includes the `"<super_class> (class)"` step. Analysis code must select Stage-D runs by
   `kind="type_dcsbm"`, not `gen_stats.variant` (which reads "L3_dcsbm"). [§B, §D]

None of these touches a number produced by the runs: every logged quantity reproduced
exactly, including 4/4 bit-exact oracle re-scores on regenerated graphs, 4/4 exact 2-cycle
bounds under a structurally different implementation, full-coverage invariant verification
(all slots, all nodes, all 456,602 blocks) plus a 1,000-node exhaustive dictionary check,
and an independent reconstruction of the complete type-label partition (K = 8,857, exact
array equality).

**Final verdict: PASS** — headline statistics confirmed in full; generator invariants hold
exactly in production; the inversion is real under the protocol (not noise, not labels, not
a single pipeline stage) but remains **probe-gated**: it must not be promoted before the
Stage-E gs8000 20k probe is logged and clears the §H2 table, and it must be reported with
the §G adjudication and both axes of §H4.
