# Central-brain-only MFAS: is the center's order self-determined?

`status: reviewed with corrections` (2026-09-30 — every number independently confirmed, most bit-exact; three interpretive wording fixes applied, see "Independent review" below) · 2026-09-29 · MacBook (Apple MPS, torch 2.8.0, python 3.9.23, env `allen`) · seed 42, 2 same-seed runs

## Question

The fly-connectome champion order (H64, 84.2582% feedforward) was optimized on the
whole brain (136,648 neurons). If we run the same champion pipeline on the induced
subgraph of `super_class == 'central'` neurons only (all edges to/from non-central
neurons removed), how much does the internal order of the central neurons change
relative to their order within the full-brain champion? Is the center's order
self-determined by its own wiring, or dictated by its periphery interfaces?

## Method

- **Data.** Full graph + pinned H64 champion order via
  `layering.io_utils.load_graph_and_champion_order("connectome")` (parity-gated
  against `autoresearch/sota.json`). Annotations from
  `layering/outputs/fly_ann/node_ann.npz`, staleness-gated by asserting
  `npz['rank'] == champion rank`.
- **Subgraph.** Nodes with `super_class == 'central'`, edges with both endpoints
  central, relabeled `0..m-1`; cached as `biology/results/central_subgraph.npz`.
- **Pipeline.** The H64 connectome champion, production functions imported (never
  copied): `H02.greedy_fas_order` → `H64._rocket_asym` (20,000 epochs, ASYM
  M=0.75 T=1.5, lr 0.05, 5 beta cycles, grad clip 1.0) → `sift_underrelaxed`
  (40 sweeps, k_full 6, α 0.7) → `alternate_scc_sift` (77 cycles, 2-sweep sift,
  min_block 32, split_fracs 0.5/0.382/0.618/0.25/0.75); best-by-oracle across
  stages, frozen oracle `src/mfas/metrics.py` on CPU. **Full champion constants —
  nothing was reduced.**
- **Control (mandatory).** The pipeline is run TWICE at seed 42. The pipeline is
  deterministically greedy-initialized (the seed is never drawn from), so the
  run1-vs-run2 spread is pure MPS device noise — the noise floor for every
  order-change claim below.
- **Comparator (b).** The champion order restricted to central nodes, ranks
  compressed to `0..m-1` (compression preserves every internal edge's
  feedforward status).

Exact re-run commands (from the repo root):

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
$PY biology/analysis/exp_central_only.py --stage subgraph
$PY biology/analysis/exp_central_only.py --stage smoke
$PY biology/analysis/exp_central_only.py --stage run --run-id 1 2>&1 | tee biology/results/central_only_run1.log
$PY biology/analysis/exp_central_only.py --stage run --run-id 2 2>&1 | tee biology/results/central_only_run2.log
$PY biology/analysis/exp_central_only.py --stage analyze
```

## Numbers

All from `biology/results/central_only.json` (runs logged in
`central_only_run{1,2}.{json,log}`; orders persisted as `.npy`).

**The induced central subgraph**

| | central | full | % of full |
|---|---|---|---|
| nodes | 32,294 | 136,648 | 23.63% |
| edges | 849,937 | 5,657,719 | 15.02% |
| total weight | 12,523,698 | 41,912,141 | 29.88% |

**Feedforward % of the central subgraph (frozen oracle, CPU)**

| order | score | FF% | gap vs (b) |
|---|---|---|---|
| (b) champion restricted to central | 10,356,669 | 82.6966% | — |
| (a) central-only run 1 | 10,449,116 | 83.4347% | **+0.7382 pp** |
| (a) central-only run 2 | 10,449,826 | 83.4404% | **+0.7438 pp** |

Stage progression (run 1 / run 2): greedy 70.2244% / 70.2244% (bit-identical,
deterministic) → pure Rocket 82.3260% / 82.3264% → sift 83.1653% / 83.1800% →
final 83.4347% / 83.4404%. Wall clock 134.6 s / 144.2 s total per run.

**Order comparisons on the central line (m = 32,294)**

| pair | Spearman | Kendall τ | median disp | IQR | frac moved >5% |
|---|---|---|---|---|---|
| run1 vs champion-restricted | 0.7440 | 0.6010 | 0.0670 | [0.025, 0.177] | 57.8% |
| run2 vs champion-restricted | 0.7405 | 0.5969 | 0.0700 | [0.027, 0.175] | 61.6% |
| **run1 vs run2 (noise floor)** | **0.9602** | **0.8867** | **0.0141** | [0.004, 0.032] | **15.4%** |
| run1 vs champion, excluding CX (n=29,446) | 0.7781 | 0.6437 | 0.0595 | [0.023, 0.140] | 55.0% |

(disp = |Δrank| / line length. The noise-floor Spearman 0.960 matches the
historical seed-to-seed order stability of full-brain Rocket runs, ≈0.96.)

**Per cell_class, run1 vs champion-restricted (classes with n ≥ 20; noise = same
statistic between run1 and run2)**

| class | n | median \|disp\| | vs own noise | median signed shift | champ→new position |
|---|---|---|---|---|---|
| CX | 2,848 | 0.276 | **13.8×** | +0.275 (→ back) | 0.51 → 0.79 |
| TuBu | 150 | 0.145 | 1.6× | −0.145 (→ front) | 0.39 → 0.24 |
| Kenyon_Cell | 5,177 | 0.076 | 4.1× | +0.052 | 0.37 → 0.44 |
| NA (unannotated) | 21,912 | 0.063 | 4.9× | −0.030 | 0.63 → 0.58 |
| MBON | 96 | 0.058 | 8.3× | +0.036 | 0.49 → 0.56 |
| LHCENT | 42 | 0.054 | 3.4× | −0.052 | 0.65 → 0.58 |
| mAL | 77 | 0.045 | 2.0× | −0.024 | 0.58 → 0.58 |
| LHLN | 514 | 0.034 | 2.7× | −0.019 | 0.27 → 0.26 |
| ALIN | 24 | 0.021 | 3.2× | −0.013 | 0.08 → 0.08 |
| ALPN | 685 | 0.018 | 1.4× | +0.006 | 0.15 → 0.17 |
| ALLN | 429 | 0.016 | 3.3× | +0.004 | 0.12 → 0.09 |
| DAN | 320 | 0.005 | 11.1× (noise ≈ 0.0005) | +0.001 | 0.54 → 0.67 (bimodal: 64% of DANs move <5%, a minority relocates far — medians of position and of displacement disagree for that reason) |

**Full-brain context (logged in `full_brain_context`)**

- Median full-champion rank fraction by super_class: ascending 0.06, sensory 0.07,
  visual_centrifugal 0.32, optic 0.46, visual_projection 0.81, **central 0.82**,
  descending 0.96, endocrine 0.99, motor 0.999.
- CX synaptic weight is almost entirely intra-central: out-weight 99.32% to
  central (0.48% to descending), in-weight 99.61% from central.
- TuBu in-weight is 65.3% from visual_projection (removed in the subgraph),
  34.7% from central; out-weight 100% to central.

Figures: `biology/outputs/central_only_rank_vs_champion_hexbin.png`,
`biology/outputs/central_only_class_displacement.png`.

## Reading (cautious)

1. **At the score level the center is nearly self-determined.** Optimizing the
   central brain alone gains only ~0.74 pp of internal feedforward weight over
   simply inheriting the full-brain champion's arrangement (83.44% vs 82.70%).
   Read at face value, being embedded in the whole brain costs the center
   ~0.74 pp of internal feedforward-ness — but the measured gap bounds the true
   cost from *neither* side: the central-only run is itself a heuristic (which
   understates the isolated optimum), while part of the 0.74 pp may be
   full-brain optimization slack rather than a genuine interface constraint
   (which would overstate the cost) — the two cannot be separated here.
   *(Review 2026-09-30: original text called the gap "a lower bound" on the
   cost of embedding; that direction claim was wrong — slack on the full-brain
   side pushes the other way.)*
2. **At the order level the periphery matters much more.** The internal
   arrangement changes far beyond device noise (Spearman 0.744 vs a 0.960 noise
   floor; 58% of neurons move >5% of the line vs 15% between same-seed runs).
   Read together with (1): the central brain's objective landscape has many
   near-equally-good internal orders, and the periphery interfaces select which
   one the whole-brain optimum uses. The reordering is broad, not a single
   block — excluding CX entirely still leaves Spearman at 0.778.
3. **The central complex is the strongest and most reproducible disagreement.**
   CX (2,848 neurons) relocates coherently from the middle of the central line
   (0.51) to its sink end (0.79), 13.8× its own noise floor — and this is NOT a
   direct interface artifact, because >99% of CX synaptic weight is
   intra-central. Central wiring alone robustly reads as "everything converges
   on the central complex" (run 1 places CX at 0.79, run 2 at 0.81, run-to-run
   noise 0.02); the whole-brain champion places CX mid-line *among central
   neurons* (0.51 of the central line; on the full-brain line CX sits at 0.83,
   like central neurons generally). *(Review 2026-09-30: "equally robustly" was
   removed — the champion is a single deterministic CUDA artifact whose own
   repeat noise is unmeasured; whole-brain-side robustness is only indirectly
   supported by the historical ≈0.96 full-brain seed stability.)* The pull forward must be indirect (via
   central partners that are themselves pinned by periphery — e.g. the small
   0.48% CX→descending output, and visual-projection input arriving late on the
   full-brain line) — which mechanism dominates is not resolved here.
4. **TuBu is the textbook direct-interface case, but weakly determined.** With
   65% of its input weight (visual projections) removed, TuBu drifts toward the
   source end — yet its displacement is only 1.6× its own noise floor: once its
   dominant input is gone, its position is simply under-constrained.
5. **The sensory-side backbone barely moves.** The antennal-lobe classes (ALPN,
   ALLN, ALIN) and lateral-horn locals keep their positions (median disp
   0.016–0.034) even though their sensory drive was removed, and the typical
   DAN keeps its position exactly (median disp 0.005; a ~36% minority does
   relocate). The antennal-lobe→lateral-horn stages of the backbone appear
   genuinely self-determined by central wiring; the mushroom-body stage in
   between is only mostly so — KC (disp 0.076, 4.1× noise) and MBON (0.058,
   8.3×) shift modestly backward, beyond noise, while preserving the
   AL → MB → LH stage sequence. *(Review 2026-09-30: original text extended
   "genuinely self-determined" to the whole olfactory→MB→LH backbone; the MB
   rows in this study's own table contradict that for the MB stage.)*

**Answer to the research question:** both, split by level. How *feedforward* the
center can be is essentially self-determined (context costs ~0.74 pp), and the
antennal-lobe/lateral-horn backbone keeps its arrangement (the mushroom-body
stage shifts modestly); but *which* of the many
near-optimal internal orders is used is substantially dictated by the periphery,
most dramatically for the central complex, whose position on the line is the one
thing the center-alone and whole-brain objectives reproducibly disagree about.

## Caveats

- **MPS non-determinism.** Same-seed repeats differ (final FF% 83.4347 vs
  83.4404, order Spearman 0.960); every order-change claim is judged against that
  measured floor, not against exact reproducibility. Discrete scores themselves
  are exact (frozen oracle, CPU, int64).
- **Single-seed by construction.** The pipeline never draws from its seed
  (greedy-FAS init), so seeds would be one sample taken repeatedly; the honest
  spread is the device-noise repeat, which is what was run. n=2 runs bound the
  noise crudely.
- **Cross-device order provenance.** The full-brain champion order is a CUDA
  artifact; the central-only orders are MPS. All FF% here come from the same
  frozen CPU oracle, so scores are comparable, but the champion *order* itself
  carries CUDA-run provenance.
- **One whole-brain comparator (added in review, 2026-09-30).** The champion is
  a single deterministic CUDA artifact; its repeat noise on the restricted
  central line is unmeasured (same-seed CUDA repeats are bit-identical, hence
  uninformative about it). Every "the whole brain places X" statement rests on
  that one order; only the central-only side has a measured noise floor. The
  historical ≈0.96 full-brain seed-to-seed Spearman (pure Rocket, MPS) supports
  the whole-brain side only indirectly.
- **Annotation coverage.** 13/136,648 nodes are unannotated (excluded from
  'central'); 21,912/32,294 central neurons (67.9%) have `cell_class = NA`, so
  the per-class story covers only ~32% of central neurons by identity. The NA
  bulk itself moves 4.9× its noise floor, so the broad reordering is not an
  annotation artifact.
- **Heuristic optima on both sides.** 83.44% is a lower bound on the central
  subgraph's true optimum and 84.26% on the full brain's; "cost of context" and
  the plateau reading inherit that.
- Weight, not neuron count, defines the objective: the central subgraph holds
  15% of edges but 30% of synaptic weight — central-central connections are
  heavier than average (another logged observation of this study).

## Provenance

- Script: `biology/analysis/exp_central_only.py` (CONFIG at top; stages
  subgraph/smoke/run/analyze).
- Results: `biology/results/central_only.json` (all numbers above),
  `central_only_run{1,2}.json`, `central_only_run{1,2}.log`,
  `central_only_run{1,2}_rank.npy`, `central_only_champ_restricted_rank.npy`,
  `central_subgraph.npz`, `central_subgraph_meta.json`.
- Figures: `biology/outputs/central_only_rank_vs_champion_hexbin.png`,
  `biology/outputs/central_only_class_displacement.png`.

## Independent review (2026-09-30)

Strict pre-COSYNE review. Every number was re-derived with independent code —
`biology/analysis/review_central_only.py` (85 checks, results in
`biology/results/review/review_central_only.json`) — never by re-running the
study's own code paths: the subgraph was rebuilt from
`layering.io_utils.load_graph_and_champion_order` + `node_ann.npz` with review
code and compared array-by-array to the cached `central_subgraph.npz`; the
champion-restricted rank was recompressed via `scipy.stats.rankdata` (different
code path than the study's double argsort); every saved rank was re-scored
through the frozen oracle (`src.mfas.metrics.score_from_order`, CPU); and the
CX/TuBu memberships and all boundary flows were re-derived from the **raw**
annotation TSV (`data/external/flywire_neuron_annotations.tsv.gz`) joined on
root IDs — i.e. independently of `node_ann.npz` as well.

### Verdicts

| # | Claim | Verdict |
|---|---|---|
| 1 | Subgraph facts: 32,294 nodes (23.63%), 849,937 edges (15.02%), weight 12,523,698 (29.88%); cached `central_subgraph.npz` src/tgt/weight/node_ids | **CONFIRMED, bit-exact** (rebuilt independently; arrays identical) |
| 2 | Central-central synapses "~2× heavier than average" | **CONFIRMED** (ratio 1.989) |
| 3 | Oracle scores: champ-restricted 10,356,669 = 82.6966%; run1 10,449,116 = 83.4347%; run2 10,449,826 = 83.4404%; gaps +0.7382/+0.7438 pp | **CONFIRMED, bit-exact** (frozen oracle, CPU; all three vectors verified as permutations; champ-restricted equals independent recompression of the pinned champion on central nodes) |
| 4 | Order stats: Spearman 0.7440/0.7405, Kendall 0.6010/0.5969, median disp 0.067/0.070, IQR [0.025,0.177], frac>5% 57.8/61.6%; noise floor 0.9602/0.8867/0.0141/15.4% | **CONFIRMED** (recomputed from saved vectors; agree to ≤5e-4) |
| 5 | CX definition and n = 2,848 | **CONFIRMED and defensible** — cell_class 'CX' is the Schlegel et al. 2024 table's own code (columnar + tangential + ring neurons + LNOa; every CX row is super_class central); raw TSV holds 2,875 CX rows, 2,848 of which are graph nodes. No relabeling by the study. |
| 6 | CX moves 27.6% of line, 13.8× own noise, 0.51 → 0.79 | **CONFIRMED** (0.2759; ratio 13.75; 0.512 → 0.788, run 2 reproduces at 0.814) |
| 7 | Boundary flow: CX out 99.32% to central, 0.48% to descending; in 99.61% from central; TuBu in 65.3% from visual_projection, out 100% to central | **CONFIRMED, bit-exact on weights** (recomputed from full graph + raw-TSV classes; CX out/in totals 1,032,064 / 1,108,535) |
| 8 | Excluding CX (n = 29,446) leaves Spearman 0.778 | **CONFIRMED** (0.77807) |
| 9 | Per-class table (TuBu 1.6×, KC 4.1×, NA 4.9×, MBON 8.3×, DAN 11.1× with 35.6% minority, AL classes 0.016–0.021, …) and full-brain super_class medians | **CONFIRMED** (all spot rows recomputed; agree to quoted precision) |
| 10 | Stage progression, wall clocks, "greedy bit-identical" | **CONFIRMED** (run JSONs + logs; greedy score 8,794,686 in both runs; run1 ≠ run2 genuinely — 32,037/32,294 slots differ, so the repeat is real, not a copied file) |
| 11 | Re-run commands | **CONFIRMED real** (script exists with stages subgraph/smoke/run/analyze; paths and interpreter correct) |
| 12 | Reading 1: "+0.74 pp = cost of full-brain embedding, a lower bound" | **CORRECTED** — sign and arithmetic right, confound (full-brain slack) was already stated, but "at least / lower bound" was directionally wrong: the heuristic central-only run biases the gap *down* while full-brain slack biases it *up*, so the gap bounds the true cost from neither side. Wording fixed in place. |
| 13 | Reading 2: "plateau + periphery selects" | **SUPPORTED with one caveat added** — the plateau is licensed (three very different orders within 0.74 pp; order change 0.744 vs noise 0.960, and broad: 0.778 without CX, NA bulk 4.9× noise). "Periphery selects" additionally assumes the whole-brain side would reproduce its internal central order; that side is a single deterministic CUDA artifact with unmeasured repeat noise (new caveat). |
| 14 | Reading 3: whole brain "equally robustly" places CX mid-line | **CORRECTED** — robustness is measured only on the central-only side; "equally robustly" removed. Also clarified: mid-line means mid-line *of the central line* (0.51); on the full-brain line CX sits at 0.83. |
| 15 | Reading 5: "olfactory→MB→LH backbone genuinely self-determined" | **CORRECTED** — true for the AL/LH stages (disp 0.016–0.034) and the typical DAN, but the study's own table shows the MB stage moves beyond noise (KC 0.076 at 4.1×, MBON 0.058 at 8.3×); claim narrowed to AL→LH, MB stated as "mostly". |
| 16 | Per-class noise normalization behind "13.8×" | **FAIR** — same statistic, same class, same-pipeline repeats; n=2 crudeness already caveated; any plausible champion-side noise (whole-line same-seed median disp 0.014) is far below the 0.276 CX signal. |
| 17 | Caveats list | **COMPLETE after one addition** (single whole-brain comparator, above); the previously listed five caveats are accurate as stated. |

### Notes for downstream summaries (FINDINGS.md B6)

- B6's "the whole brain robustly holds it mid-line" must carry the qualifier
  *among central neurons* (full-brain-line CX median is 0.83, not mid-line),
  and "robustly" is unsupported on the whole-brain side (single artifact).
- B6's "The olfactory→MB→LH backbone … keeps its arrangement" overstates the
  MB stage (KC 7.6% median displacement at 4.1× noise, MBON 5.8% at 8.3×);
  restrict to the AL/LH backbone.
- Everything else in B6 quotes this study faithfully.

Review artifacts: `biology/analysis/review_central_only.py`,
`biology/results/review/review_central_only.json` (85 checks, 0 failed).
