# Null models: do real brains carry more or less unavoidable feedback than matched random graphs?

**status: reviewed with corrections (independent review 2026-09-30, see bottom)**
**Track C question G01 (`experiments/randomgraph.md`) — first runs ever taken on it.**
Date: 2026-09-29/30 · machine: MacBook (Apple Silicon, MPS) · env `allen`
(torch 2.8.0, python 3.9.23) · scripts: `biology/analysis/nullmodels_*.py` ·
raw runs: `biology/results/nullmodels/runs/*.json` (689 records, every number
below traces to one) · summary: `biology/results/nullmodels/summary.json`.

---

## 1. Question and design

`unavoidable feedback = 100% − best feedforward %`: the edge weight that cannot
point forward under ANY linear ordering of the neurons. Do real connectomes have
more or less of it than random graphs matched on their statistics — and which
statistics?

The study is a **nested constraint ladder**. Each rung adds one restriction; the
FF% difference between adjacent rungs (identical estimator, identical epoch
budget) isolates one structural ingredient of the brain's feedforward order:

| rung | generator (exact microcanonical match) | underlying null distribution |
|---|---|---|
| chance | — | any fixed order on any graph: exactly 50% of non-loop weight is FF in expectation |
| **L1** "density" | `gnm_null`: both endpoints of every edge redrawn uniformly; weights stay on their slot | directed G(n, m) with multi-edges, loop-free |
| **L2** "+degrees" | `directed_configuration_null`: target column permuted globally. Primary weight variant **`w_source`** (weights stay on the slot = travel with the source stub → every node's exact out-strength preserved). Secondary **`w_shuffled`** (weights independently permuted → degrees only, all degree–strength coupling broken) | uniform over multigraphs with the exact in/out degree sequence |
| **L3a** "block counts" (non-nested reference) | `sbm_null`: within each (lane × lane) block both endpoints redrawn uniformly from the two lanes | stochastic block model proper on the 17 lanes |
| **L3** "+regional architecture" | `block_configuration_null`: target column permuted **within** each (lane(src) × lane(tgt)) block; weights on slot | uniform over graphs with every node's **lane-resolved** in/out degree AND out-strength profile fixed (microcanonical degree-corrected SBM) |
| top | the real graph | — |

Fly regions = the **17 biological lanes** of the layering study
(`layering/tools/fly_streams.py::lane_of`, super_class refined by cell_class,
Schlegel et al. 2024 annotations, 99.99% join; every node covered). Lanes are
node attributes, so they survive rewiring and mean the same thing on every null.

**Datasets.** Fly connectome (n = 136,648, m = 5,657,719, W = 41,912,141,
int64) gets the full ladder. Mouse meso-connectome (n = 148 regions, m = 583,
float weights) gets **L1 + L2 only**: its nodes ARE regions, so L3 degenerates
to the real graph by construction.

**Self-loops / duplicates policy** (both real graphs have zero of each —
verified in `nullmodels_generators.py --selftest`): permutation-created
self-loops are repaired by pairwise target swaps (within-block for L3), because
a self-loop is guaranteed feedback and would handicap the null (fly: 188 of
5.66 M slots for L2, 911 for L3, 39–149 for L1/L3a; mouse: 12 of 583).
Duplicate parallel slots are **kept** — every stage of the protocol (oracle,
greedy peel, Rocket loss, sift gains) is a per-edge weighted sum, so parallel
edges behave exactly like one summed edge, and keeping them preserves the
weight multiset (fly: 0.36% of slots for L2, 1.16% for L3; mouse: ~11% —
reported, score-invariant).

Generator validation (`--selftest`, both datasets): out/in-degree sequences
preserved exactly (bincount equality), weight multiset + totals exact,
out-strength exact for `w_source`, lane-profile and block count/weight equality
exact for L3/L3a, loop-freedom, bit-identical reproduction under the same seed.

## 2. Estimator (identical for the real graph and every null)

Per `experiments/randomgraph.md`'s estimator contract: own thin runner
(`nullmodels_run.py`) on in-memory `GraphData`, frozen oracle for every score.

1. **greedy-FAS** warm start (`mfas.experiments.H02.greedy_fas_order`) — also
   reported alone as the cheap-estimator control;
2. **Rocket + ASYM surrogate** (`mfas.experiments.H64._rocket_asym`), fixed
   epoch budget: **fly 4,000, mouse 2,000** epochs, seed-pinned;
3. **under-relaxed exact-gain sift** (`mfas.refine.underrelax.sift_underrelaxed`),
   max_sweeps = 8, k_full = 2, alpha = 0.7;
4. every discrete score from the frozen oracle `mfas.metrics.score_from_order`
   on CPU (int64/float64).

Budgets are **epoch-counted, never wall-clock** — another agent's optimization
runs shared the GPU during this study, and identical epoch counts make
contention harmless to fairness (it only stretches runtimes: fly runs took
57–130 s each; probe runs ~240 s; mouse runs 0.3 s / 3–4 s).

Seeds: real graphs ran pipeline seeds {42, 123, 999}; each null ran pipeline
seed 42 on its own graph seed (fly gs 1000–1019 / 2000–2019 / 3000–3011 /
4000–4011 / 5000–5011; mouse 100 seeds per variant). Justification for one
pipeline seed per null: the real fly's across-pipeline-seed σ is 0.0033 pp —
5× smaller than the smallest across-draw null σ (0.015 pp), so draw-to-draw
spread already dominates estimator noise. On CPU (mouse) the protocol is
bit-deterministic (3 real seeds identical); on MPS it is not (σ above).

**Stated deviations from the `randomgraph.md` menu:** ER is realized as G(n,m)
with weights-on-slot (weights must go somewhere on a weighted graph);
"degree-preserving rewiring" and "configuration model" collapse into one rung
here (a global target-column permutation IS a uniform draw from the
degree-preserving ensemble); SBM uses the annotation-derived 17 lanes rather
than inferred communities. K = 12–20 (fly) / 100 (mouse) per rung, stated per
table. `src/mfas/randomgraph/` was left untouched (rule); generators live
self-contained under `biology/analysis/`.

## 3. Results — fly connectome

FF% after the identical protocol (mean ± sd over K independent graph draws;
"beyond" = draws at or past the real value):

| rung | K | FF% (protocol) | unavoidable FB% (upper bd) | z of real | beyond | greedy FF% |
|---|---|---|---|---|---|---|
| L1 G(n,m) | 12 | 69.823 ± 0.019 | 30.177 | +727 | 0/12 | 64.607 ± 0.018 |
| L3a SBM (ref) | 12 | 75.992 ± 0.025 | 24.008 | +306 | 0/12 | 66.970 ± 0.015 |
| L2 config, w_shuffled (ref) | 20 | 77.220 ± 0.024 | 22.780 | +265 | 0/20 | 69.100 ± 0.025 |
| L2 config, w_source | 20 | 79.981 ± 0.015 | 20.019 | +245 | 0/20 | 72.649 ± 0.015 |
| L3 dc-SBM (17 lanes) | 12 | 81.896 ± 0.018 | 18.104 | +100 | 0/12 | 70.660 ± 0.017 |
| **real graph (protocol)** | 3 seeds | **83.658 ± 0.003** | **16.342** | — | — | 68.913 |
| real graph (champion budget) | 1 | 84.258 | 15.742 | — | — | — |

The champion number is the layering champion order re-scored here by the frozen
oracle (84.25818%; matches `autoresearch/sota.json` H64 84.2582). The real
value is **outside every null draw at every rung** (empirical p = 1/13 at
K = 12, = 1/21 at K = 20, one-sided, add-one rule), by 100–727 σ of realization noise —
n = 136k self-averages, so null distributions are extremely tight (σ ≈ 0.02 pp)
and z is not the interesting scale; the **pp gaps** are.

### Decomposition of the fly's above-chance feedforward order (~33.7 pp)

Nested chain (chance → L1 → L2`w_source` → L3 → real-protocol → champion),
figure `biology/outputs/nullmodels_fly_ladder.png`:

| ingredient | Δ FF |
|---|---|
| density alone (what G(n,m) allows the optimizer) | **+19.82 pp** |
| + in/out degree sequence & out-strengths | **+10.16 pp** |
| + 17-lane regional wiring diagram (given degrees) | **+1.92 pp** |
| + fine within/between-lane targeting (real − L3) | **+1.76 pp** |
| protocol → champion budget (estimator slack on real) | +0.60 pp |

Reference points off the chain: SBM proper (block counts without degrees)
reaches only 75.99 — **degrees explain more of the order than the lane diagram
does** (79.98 vs 75.99 from non-nested starts); and w_shuffled at 77.22 vs
w_source at 79.98 shows the **degree–strength coupling** (who sends how much)
alone is worth +2.76 pp of orderability.

### Direction (fly): the brain is MORE feedforward-organized than every matched null

At every rung the real fly beats its matched ensemble; even with node-level
lane profiles and strengths fixed (L3), real wiring buys +1.76 pp more
feedforward weight than chance. Equivalently: the fly has **less unavoidable
feedback** than degree-, strength- and region-matched random graphs
(16.3% vs 18.1% for the strictest null; upper-bound estimates on both sides
under an identical budget).

### …and simultaneously a far larger *irreducible* feedback core (reciprocity)

The **2-cycle bound** is estimator-free: every reciprocated pair {u,v} forces
min(w(u→v), w(v→u)) into feedback under any order. Summed:

| graph | 2-cycle lower bound on unavoidable FB (% of W) |
|---|---|
| real fly | **6.717** |
| L3 dc-SBM nulls | 0.378 ± 0.004 |
| L2 config nulls | 0.105–0.110 ± 0.001 |
| L1 G(n,m) nulls | 0.007 |

So ≥ **42.7%** of the champion order's feedback (6.72 of 15.74 pp) is forced by
reciprocal pairs alone — feedback the fly *provably cannot shed* — while every
null's forced core is 18–930× smaller (17.8× vs the strictest L3 null, 61–64×
vs the L2 configuration nulls, 184× vs SBM, 932× vs G(n,m)). The fly is simultaneously (a) more
feedforward-orderable than matched chance and (b) built on a reciprocal
backbone that guarantees it several times more irreducible feedback than any
degree-matched random graph carries.

### Where the fly's residual feedback deviates from block-matched chance (lane z-matrix)

Real minus L3-null feedback weight per (source lane → target lane), in null σ
(K = 12; under L3 every lane-pair's total weight is preserved exactly, so
differences are pure ordering). 88/289 pairs above the 2·10⁻⁴·W weight floor.
Figure `nullmodels_fly_lane_z.png`, matrix `lane_fb_zmatrix.npz`. Extremes:

* **Excess feedback (real > null):** mushroom body→mushroom body **z = +43**
  (0.284% of W vs 0.108% — 2.6× block-matched chance), lamina→lamina +32,
  central(other)→visual projection +25, photoreceptors→medulla +23, visual
  centrifugal→lobula (plate) +20 — i.e. the known recurrent memory circuit
  (MB), peripheral visual feedback, and the literal centrifugal feedback
  pathway carry *structurally deliberate* recurrence.
* **Feedback deficit (real < null):** central(other)→central(other) **z = −82**
  (3.68% vs 4.71%), visual projection→medulla→lobula-lanes −35…−53,
  medulla→medulla −30 — the large central mass and the optic feedforward
  pathway are much more hierarchically wired than their block statistics force.

## 4. Results — mouse meso-connectome (148 regions)

Standard arm (2,000 epochs + 8 sweeps); real protocol = 92.9529% FF (three CPU
seeds bit-identical), champion (registry, H63, 20 seeds, CUDA) = 93.1754%:

| rung | K | FF% (protocol) | z of real | beyond | empirical p | greedy FF% |
|---|---|---|---|---|---|---|
| L1 G(n,m) | 100 | 92.98 ± 0.81 | −0.04 | 56/100 | 0.56 | 86.80 ± 1.33 |
| L2 w_shuffled | 100 | 93.54 ± 0.84 | −0.70 | 22/100 | 0.23 | 86.56 ± 1.98 |
| L2 w_source | 100 | **94.41 ± 0.70** | **−2.08** | **2/100** | **0.030** | 87.42 ± 1.48 |

Robustness arm at 10× epochs + 40 sweeps (identical protocol both sides):
real 92.8382; L1 93.71 ± 0.76 (z = −1.14), w_shuffled 93.82 ± 0.79 (z = −1.24),
w_source 94.64 ± 0.69 (**z = −2.61**, 2/100 beyond). Direction unchanged and
strengthened. Figure `nullmodels_mouse_nulls.png`.

### Direction (mouse): the OPPOSITE of the fly — more unavoidable feedback than matched nulls

The real mouse sits at the *density-only* level (indistinguishable from
G(n,m), z = −0.04) while its degree-and-out-strength-matched ensemble is
**more** orderable (+1.46 pp): given the mouse's hub structure, random wiring
would allow ~1.5 pp more feedforward weight than the real wiring does. The real
region graph is below 98/100 strength-matched nulls.

**Rigorous version (no estimator on the decisive side).** Real mouse 2-cycle
bound = 5.627% of W (nulls: w_source 2.58 ± 0.61, L1 0.43 ± 0.24). For **54 of
100** w_source nulls, the null's *measured* feedback (an upper bound on its
true unavoidable FB) is below the real graph's 2-cycle *lower* bound — those
nulls are **provably** less feedback-bound than the real mouse. Zero nulls are
provably the other way (max null 2-cycle bound 4.46 < real measured FB 7.05).
So at the mesoscale the conclusion does not rest on the heuristic at all for a
majority of draws: **area-to-area cortical wiring carries feedback that
degree-matched randomness would not have**, and its source is exactly the
reciprocity of area pairs (5.6% of weight forced back by bidirectional
projections — ~80% of the mouse's entire measured feedback).

## 5. Honesty: estimator bias, both bounds, and the budget probe

* **Both sides are upper bounds.** All protocol FB numbers are heuristic upper
  bounds on true unavoidable FB under an identical epoch budget. The 2-cycle
  bound supplies the rigorous floor where cited.
* **Greedy control diverges on degree-matched rungs — stated, not hidden.**
  The contract's cheap-bound control (greedy-FAS) agrees with the headline
  direction on fly L1/L3a (real above nulls in both bounds) but **flips** on
  the degree-matched rungs of both datasets: fly real greedy 68.91 vs
  w_source-null greedy 72.65 (real below); mouse real greedy 90.13 vs null
  greedy 86.6–87.4 (real above, while the full protocol puts real *below*).
  Greedy peeling measures how *greedily discoverable* an order is, and degree
  matching restores exactly the out−in imbalance cues greedy feeds on; it is a
  weak estimator whose direction tracks structure-compatibility-with-greedy,
  not unavoidable FB (on the fly it leaves 11–15 pp on the table). The
  verdict-rule role of the second bound is instead filled by (a) the
  estimator-free 2-cycle bound (mouse: majority of nulls provably below real;
  fly: reciprocity core 18–930× null) and (b) the budget probe below.
* **Differential suboptimality probe (fly).** Same protocol at 5× epochs
  (20,000) + 40 sweeps, on the real graph and one pinned null per key rung:
  real 83.6546 → 84.1029 (+0.448); w_source gs2000 80.0090 → 80.3035 (+0.295);
  L3 gs5000 81.8687 → 82.4710 (+0.602); L1 gs3000 69.8130 → 70.7713 (+0.958).
  Gaps at the strong budget: real−L3 = 1.63 pp (was 1.76), real−w_source =
  3.80 pp (was 3.65), real−L1 = 13.33 pp (was 13.84). Budget-sensitivity of
  any gap is ≤ 0.51 pp and never sign-changing: the fly direction is not an
  artifact of under-optimizing the nulls. Real is 0.155 pp below the champion
  at the probe budget, bounding the real graph's own remaining slack.
* **Mouse strong-arm real value dropped** 92.953 → 92.838 with 10× epochs —
  the known H44 mouse effect (the gradient phase can hurt the mouse final);
  identical protocol on both sides, so the comparison is unaffected (nulls
  ROSE with budget, real fell: the z against the real became more negative).
* **Mouse champion** is quoted from `autoresearch/sota.json` (H63, CUDA); no
  local re-score artifact exists. The fly champion WAS re-scored locally by the
  frozen oracle.
* **MPS non-determinism**: fly runs are not bit-reproducible (real-seed σ
  0.0033 pp recorded); mouse (CPU) runs are.
* Lanes are an annotation join (Schlegel et al. 2024, CC-BY; 13/136,648 nodes
  unannotated → 'central (other)'), not oracle data.
* Null distributions at n = 136k are so tight (σ ≈ 0.02 pp) that z-scores of
  100–727 overstate nothing but should be read as "outside all draws by many
  times the realization noise"; the effect sizes in pp are the claim.
* K per rung: fly 20 (L2 variants) / 12 (L1, L3, L3a) — 12 chosen for runtime
  honesty (~2 min/run on a shared GPU); mouse 100 per rung, both arms.

## 6. Biological reading, per rung

* **L1 (density)**: rejecting it means the brain's orderability is not a
  side-effect of sparseness. Fly: real − L1 = +13.8 pp → decisively rejected.
  Mouse: real ≈ L1 → at the region scale, raw density explains ALL of the
  above-chance orderability (and degree structure over-explains it).
* **L2 (degrees/strengths)**: matching it asks "do hubs and broadcast weights
  explain the hierarchy?" Fly: no — +3.7 pp of order lives in *which* targets
  are chosen, beyond degree forcing. Mouse: the real wiring UNDERSHOOTS its
  degree-matched ensemble — reciprocal area pairs spend ~1.5 pp of potential
  feedforward order on bidirectional communication.
* **L3a (SBM)**: the lane-level wiring diagram alone, without degrees, builds
  the least order of any structured null — mesoscale block structure is not
  where the fly's hierarchy mainly lives.
* **L3 (dc-SBM)**: even neurons with identical lane-resolved in/out profiles
  are wired *within* those constraints to respect depth: +1.76 pp of fine
  targeting below the annotation's resolution.
* **Lane z-matrix**: the fly's irreducible feedback is not diffuse — it is
  concentrated, far above block-matched chance, in known recurrent circuits
  (mushroom body, lamina, centrifugal visual feedback), while the central brain
  and the optic feedforward chain are *more* hierarchical than chance.

## 7. Headline

**The answer to G01 is scale-dependent, and both directions are now measured:**
at neuron resolution the fly connectome is dramatically **more
feedforward-organized** than random graphs matched up to degrees, strengths and
regional architecture (16.3% vs 18.1–30.2% unavoidable-FB upper bounds;
outside all 76 nulls), yet carries a reciprocal core that **guarantees** it
6.7% feedback forever (nulls: ≤ 0.4%) — recurrence the wiring pays for and
concentrates in memory/feedback circuits. At region resolution the mouse
meso-connectome is the mirror image: **more** unavoidable feedback than 98/100
strength-matched nulls (rigorously provable against 54/100), all of it
attributable to bidirectional area pairs. Brains are not generically
"more feedforward than chance": they buy hierarchy at the neuron level and pay
for reciprocity at the area level.

## 8. Artifacts & re-run

```
# generators + validation
$PY biology/analysis/nullmodels_generators.py --selftest
# protocol runs (resumable; skip existing)
$PY biology/analysis/nullmodels_run.py --dataset mouse      --mode real
$PY biology/analysis/nullmodels_run.py --dataset mouse      --mode nulls --variant w_source      # etc.
$PY biology/analysis/nullmodels_run.py --dataset connectome --mode real
$PY biology/analysis/nullmodels_run.py --dataset connectome --mode nulls --variant L3_dcsbm      # etc.
# robustness arms
... --epochs 20000 --sift-sweeps 40
# analysis: summary.json, lane matrix, figures
$PY biology/analysis/nullmodels_analyze.py
```

Outputs: `biology/results/nullmodels/runs/*.json` (689), `orders/*.npy`
(fly rank vectors, int32), `summary.json`, `lane_fb_zmatrix.npz`;
figures `biology/outputs/nullmodels_fly_ladder.png`,
`nullmodels_mouse_nulls.png`, `nullmodels_fly_lane_z.png`.

---

## Independent review (2026-09-30)

**Reviewer instrument:** `biology/analysis/review_nullmodels.py` (results:
`biology/results/review/review_nullmodels.json`). Everything in the analysis
layer was recomputed with independent implementations: the 2-cycle bound was
re-derived with a different algorithm (lexsort pair-merge, no n-based key
encoding), every ladder statistic recomputed straight from `runs/*.json`,
lane matrices rebuilt via `np.add.at`, every spot score through the frozen
oracle `mfas.metrics.score_from_order` on CPU (`tests/test_metrics.py` green,
scorer untouched since commit 8f0e506). The only study code reused is the
generator set itself — the null graphs are not archived, so they were
regenerated from the logged `gen_seed`s; the 13/13 exact spot re-scores below
demonstrate in situ that seed + variant is sufficient to reconstruct the exact
graphs the protocol ran on (a positive logging finding, but note the L3/L3a
regeneration additionally depends on `layering/tools/fly_streams.py::lane_of`
+ `layering/outputs/fly_ann/node_ann.npz` staying fixed — pin those with the
study).

### Per-claim verdicts

| # | claim (study text) | reviewer recomputation | verdict |
|---|---|---|---|
| 1 | fly real 2-cycle bound 6.717% of W | **6.7172%** (own implementation) | CONFIRMED |
| 1 | ≥ 42.7% of champion FB irreducible | 6.7172 / 15.7418 = **0.4267** (champion re-scored 35,314,407 = 84.25818%) | CONFIRMED |
| 1 | fly null bounds 0.007–0.38% | L1 0.0072±0.0003, L3a 0.0365, L2 0.1049/0.1096±0.001, L3 0.3783±0.0043 — all 76 regenerated | CONFIRMED |
| 1 | fly forced core "18–1000×" (§3) / "18–60×" (§5) null | exact ratios **17.8× (L3), 61.3–64.0× (L2), 184× (L3a), 932× (L1)** | CORRECTED → **18–930×** (both places; §5's 18–60× covered only part of the ladder and still under-stated its own top) |
| 1 | mouse real 2-cycle bound 5.63% | **5.6269%** | CONFIRMED |
| 1 | 54/100 w_source nulls provably less FB-bound than real | **54/100** (measured null FB < 5.6269); w_shuffled 17/100, L1 5/100 also match summary.json | CONFIRMED |
| 1 | zero nulls provable the other way | reverse proof requires null 2-cycle **lower** bound > real **measured** FB (7.0471%); max null bound 4.4644% → **0/300** across all mouse variants — logic and count both check out | CONFIRMED |
| 1 | ~80% of mouse FB forced by bidirectional pairs | 5.6269 / 7.0471 = **79.85%** | CONFIRMED |
| 2 | fly ladder means (69.823/75.992/77.220/79.981/81.896 ± 0.02, K 12/12/20/20/12; real 83.658±0.003, 3 seeds) | identical to 4 d.p. from raw runs; z = +727.3/+305.8/+265.4/+244.7/+100.4; beyond = 0 everywhere ("outside all **76** nulls" ✓) | CONFIRMED |
| 2 | decomposition +19.82/+10.16/+1.92/+1.76/+0.60, sums to champion 84.258 | 19.823 + 10.158 + 1.915 + 1.762 + 0.600 + 50 = 84.258 | CONFIRMED |
| 2 | w_source − w_shuffled = +2.76 pp | 2.760 | CONFIRMED |
| 2 | mouse table (L1 92.98±0.81 z −0.04 p 0.56; w_shuffled 93.54±0.84 z −0.70 p 0.23; w_source 94.41±0.70 z −2.08 p 0.030, 2/100) | identical; add-one convention verified (56+1)/101 etc.; real below 98/100 w_source nulls ✓ | CONFIRMED |
| 2 | mouse strong arm (real 92.8382; 93.71±0.76 z −1.14 / 93.82±0.79 z −1.24 / 94.64±0.69 z −2.61, 2/100) | identical; mouse real runs bit-identical across the 3 CPU seeds at both budgets (one distinct score each) | CONFIRMED |
| 2 | "empirical p < 1/13 (K=12), < 1/21 (K=20)" | add-one rule **attains** 1/13 and 1/21 | CORRECTED → "= 1/13 / = 1/21" |
| 3 | generator selftest | passes; loop/dup numbers match the quoted 188/911, 0.36%/1.16%, mouse 12 & ~11% | CONFIRMED |
| 3 | reviewer's own fresh-draw checks | all rungs pass on both datasets, incl. **negative controls the selftest lacks**: w_shuffled really breaks out-strength; L1/L3a really break degrees | CONFIRMED |
| 3 | duplicate slots score-invariant | numerically verified: L3 gs5003 order scores **34,329,470** on both the parallel-slot and the collapsed graph (66,156 dup slots) | CONFIRMED |
| 4 | saved orders ↔ logged scores | **13/13 exact integer matches** (3 fly real seeds, real e20000, one null per rung incl. gs2000/gs5000/gs3000 probe graphs at both budgets), each null regenerated from its logged seed | CONFIRMED |
| 5 | lane z-matrix hotspots | MB→MB z **+43.2**, real 0.284%W vs null 0.108%W = **2.62×**; lamina +32.3; central→VP +24.6; photoreceptors→medulla +23.0; vis-centrifugal→lobula(plate) +20.4; central→central **−81.9** (3.681 vs 4.714%W); medulla→medulla −30.1; optic negatives −34.7…−52.8; 88/289 pairs above floor | CONFIRMED |
| 5 | L3 preserves lane-pair totals | regenerated gs5000: per-pair total-weight matrix **exactly equal** to real (and to the npz `w_pair`); real_fb matrix equals reviewer recomputation | CONFIRMED |
| 5 | probe numbers (§5) | 83.6546→84.1029 (+0.448), 80.0090→80.3035, 81.8687→82.4710, 69.8130→70.7713; gaps 1.632/3.799/13.332; real 84.2582−84.1029 = 0.155 below champion; sweeps=40 logged | CONFIRMED (one gap moved 0.51 pp → "≤ 0.5" corrected to ≤ 0.51) |
| — | 689 runs / seed ranges / runtimes | 689 files; gs 1000–1019/2000–2019/3000–3011/4000–4011/5000–5011, mouse ×100 each; fly 56–130 s (study said 57–130 — one w_source run took 55.8 s), probe 239–300 s, mouse 0.3 s / 3–9 s | CONFIRMED (trivial runtime rounding) |

### Red-team notes (design)

* **(a) Budget probe.** Supports "not an under-optimization artifact" for the
  budgets tested: all four graphs improve, gaps move ≤ 0.51 pp, none flips.
  Caveats it cannot remove: one pinned null per rung (defensible — draw σ is
  ~0.02 pp), and 5× is not ∞; the claim should stay phrased, as it is, in
  terms of upper bounds under identical budget.
* **(b) Greedy-control flip — a contract deviation, honestly disclosed.**
  `experiments/randomgraph.md` pre-registered "trends must hold for BOTH
  bounds, or the conclusion is an estimator artefact." On the degree-matched
  rungs the greedy bound flips, so by the letter of that rule those rungs
  fail the pre-registered test; the study replaces the failed control
  post-hoc with the 2-cycle bound + budget probe and explains the flip
  (degree matching restores exactly the out−in imbalance cues greedy peels
  on; greedy leaves 11–15 pp on the table). The explanation is coherent and
  the substitution is stronger than the original control **for the mouse**
  (54/100 provable, estimator-free). For the **fly at L2/L3** the direction
  remains protocol-conditional: no estimator-free proof exists or is possible
  with these bounds (real measured FB 16.3% ≫ any null's 2-cycle bound
  ≤ 0.39%). The headline survives because it is worded as upper bounds under
  an identical budget — keep that wording in the abstract.
* **(c) Ladder order vs budget.** Between 4k and 20k epochs every rung
  improves and the chain order is unchanged; the smallest inter-rung gap
  (real−L3, 1.76→1.63 pp) is ~80× the null draw σ. The largest budget gain
  accrues to the worst-optimized rung (L1, +0.96), which compresses but
  cannot reorder a monotone chain at these separations. One nuance: the
  +0.60 pp "protocol → champion" step is not pure epochs — the champion
  pipeline (H64) adds refinement stages; the probe decomposes it as ~0.45
  budget + 0.155 residual (already stated in §5).
* **(d) Mouse "sits exactly at G(n,m)".** With null σ 0.81 (K=100) the
  honest statement is "statistically indistinguishable" (|real−mean| =
  0.03 pp, z = −0.04); "exactly" (also used in FINDINGS A1) is rhetorical —
  the data cannot certify exactness, only non-distinguishability.
* **(e) Headline / COSYNE sentence.** "Scale-dependent" is confounded: fly
  vs mouse differ simultaneously in species, level of description, and
  measurement (synapse counts vs projection weights); the design cannot
  attribute the flip to scale alone. The FINDINGS caveat gestures at this
  but asserts the comparison "compares level of description, not species" —
  too strong; it *confounds* them. Also "brains are not generically 'more
  feedforward than chance'" reads wrong for plain chance (mouse 92.95% ≫
  50%); it is true only with "chance" = matched random graphs. Recommended
  abstract phrasing: "more feedforward-organized than matched random
  graphs", and "the direction reverses between a neuron-level insect
  connectome and a region-level mammalian one (scale and species
  confounded)".
* **(minor) Lane z-matrix uses the seed-42 real order only.** Reviewer
  checked all three real seeds: MB→MB z = 43.2/43.4/42.9 — extremes stable;
  no correction needed.
* **(minor) Self-loop repair direction.** Repair helps the null on both
  datasets; for the fly this is conservative (headline: real above nulls),
  for the mouse it points *with* the claim — but it is the correct ensemble
  choice (the real graphs are loop-free, and a loop is guaranteed FB the
  real graph structurally cannot have), so no correction.

### Corrections applied to this document

1. §3: "18–1000× smaller" → "18–930× smaller" + exact per-rung ratios.
2. §5: "reciprocity core 18–60× null" → "18–930× null".
3. §3: "empirical p < 1/13 … < 1/21" → "= 1/13 … = 1/21 (add-one rule)".
4. §5: budget-sensitivity "≤ 0.5 pp" → "≤ 0.51 pp".
5. Header: status → **reviewed with corrections**.

FINDINGS.md A1 needs the mirror of #1–2 ("an 18–60× excess" → "an 18–930×
excess") and two wording softenings (the "3×" degrees-vs-lanes factor holds
only as the conditional-increment ratio (L3−L3a = 5.90 pp vs L3−L2 = 1.92 pp
= 3.1×) — state the basis or use the nested 5.3×; "sits exactly at" →
"statistically indistinguishable from") — reported to the maintainer, not
edited here (review scope is this file only).

**Verdict: every load-bearing number CONFIRMED (2-cycle bounds, ladder
statistics, decomposition, mouse provability counts, lane hotspots, probe,
generators, 13/13 oracle re-scores); 4 in-text corrections applied, none
touching a headline quantity. Status: reviewed with corrections.**
