# Biological findings — ranked for the scientific community (COSYNE lens)

*Master list. Ranking criterion: expected interest to the computational-
neuroscience community (novelty × strength of evidence × generality), with the
COSYNE abstract in mind. Status legend: `reviewed` = independently re-checked
inside this repo (every 2026-09-30 study passed a strict independent review:
own-code recomputation, oracle re-scores, adversarial red-team; per-study
verdict tables live in each studies/*.md). No unreviewed claims remain.*

**The instrument, in one sentence.** We order entire connectomes on a line to
maximize feedforward edge weight (MFAS; Rocket-family optimizer, frozen exact
scorer), then read the order biologically — the optimizer knows nothing about
anatomy, cell types, or transmitters. Datasets: FlyWire whole-brain fly
connectome (136,648 neurons / 5.66 M edges, best order 84.26 % FF), MICrONS
cortical mm³ (67,534 / 10.4 M, 83.24 %), Allen-style mouse meso-connectome
(148 regions / 583 edges, 93.18 %).

---

## Tier A — abstract-carrying claims

### A1. Brains vs matched random graphs: the answer is scale-dependent — hierarchy is bought at the neuron level, reciprocity is paid at the area level. — status: reviewed with corrections (studies/nullmodels.md, 2026-09-30; G01 answered, 689 logged runs, all load-bearing numbers independently reconfirmed)
**Fly (neurons):** the brain has LESS unavoidable feedback than every matched
null — outside all 76 null draws at every rung of a nested constraint ladder
(identical epoch-budgeted estimator on real and null graphs). The ladder
decomposes the ~33.7 pp of above-chance feedforward order:
chance 50 % → G(n,m) 69.8 % → +degrees 79.98 % → +17-lane regional wiring
diagram 81.90 % → real 83.66 % (protocol) → 84.26 % (champion). Contributions:
**density +19.8 pp, degree/strength sequence +10.2 pp, regional diagram
+1.9 pp, fine sub-annotation targeting +1.8 pp** (estimator slack 0.6).
Given the other constraint, degrees add 5.9 pp where the lane diagram adds
1.9 pp (≈ 3×; as nested marginals, 10.2 vs 1.9 pp).
Robust to a 5× optimization-budget probe (gaps stay large and same-signed).
**…yet the fly carries a provably irreducible feedback core:** the
estimator-free 2-cycle bound forces 6.72 % of total weight to point backward
under ANY ordering (matched nulls: 0.007–0.38 % — an 18–930× excess, 17.8×
vs the strictest L3 null up to 932× vs G(n,m)), i.e.
≥ 42.7 % of the champion's feedback is reciprocity no hierarchy can shed.
The z-matrix vs block-nulls localizes this excess recurrence exactly where
physiology expects it: mushroom body↔MB (z = +43, 2.6× chance),
lamina↔lamina (+32), visual-centrifugal→lobula (+20) — while the central
mass (z = −82) and the optic chain are far MORE hierarchical than chance.
**Mouse (regions):** the mirror image — real 92.95 % FF is statistically
indistinguishable from the density-only level (z = −0.04 vs G(n,m)) but BELOW 98/100 strength-matched
config nulls (94.41 ± 0.70, z = −2.08, p = 0.030; z = −2.61 at 10× budget);
estimator-free version: the real 2-cycle bound exceeds the *measured* FB of
54/100 nulls. ~80 % of the mouse's feedback is forced by bidirectional area
pairs.
Caveats: protocol numbers are heuristic upper bounds on both sides (rigor
supplied by the 2-cycle bounds and the budget probe); a greedy lower-bound
control flips direction on degree-matched rungs — tracked and explained in
the study; mouse is meso-scale, so "scale-dependent" compares level of
description, not species.
→ Extended — and partially INVERTED at cell-type resolution — by A7
(null-ladder v2, 2026-11-01).

### A2. Three brains, three feedback architectures. — status: reviewed (per-dataset), synthesis unreviewed
Under the same instrument, minimal feedback organizes completely differently:
- **Fly:** short local chains — 16–19 % of weight lies within ±1.7 % of the
  order; feedback is hubless (top node carries 0.84 % of FB, top 100 carry 7 %).
- **Mouse meso:** one hub — a single node touches 40.5 % of all feedback weight.
- **Cortex (MICrONS):** diffuse and long-range — only 7.7 % of FB is local
  (±1.7 %), top receiver takes 0.13 %, top 1 % of nodes take 19.8 %.
Provenance: FLY_FINDINGS §2/§9, MICRONS_FINDINGS §4, RESEARCH_NOTES (mouse).
Caveat: mouse is meso-scale (regions, not neurons) — scale confound stated.

### A3. The chemistry of feedback flips between fly brain and mammalian cortex — and dopamine is the most backward-biased transmitter in the fly. — status: reviewed with corrections (studies/label_nulls.md, 2026-09-30; DA row corrected in review)
- **Fly:** feedback is disproportionately inhibitory/modulatory. Backward
  share by transmitter: ACh 11.3 % < GABA 20.9 % < Glu 22.2 % < 5-HT 28.8 %
  < **DA 51.7 %** (brain-wide 15.7 %; DA = the 682 true-DA predictions —
  5,172 of the 5,854 raw "DA" neurons are Kenyon cells, a known classifier
  error caught in review). ACh carries the bulk of forward weight; GABA+Glu
  are overrepresented in feedback (the exact FF/FB composition split needs
  re-deriving after the annotation-cache fix — see caveat).
- **MICrONS cortex:** the reverse — feedback is excitatory-sourced (78 % of FB
  weight) and *lands on inhibition*: interneurons receive 41 % of their input
  backward vs 11 % for excitatory cells; E output is 26.6 % backward vs I 5.2 %.
Reading: in both systems, descending/backward signals preferentially engage
inhibition — the fly sends inhibition backward, the cortex sends excitation
backward onto inhibitory targets. Control, not echo.
Significance (2,000 label permutations, stratified within super_class to
control composition — studies/label_nulls.md): ACh depletion z = −33.4,
GABA z = +15.4, Glu z = +17.2, all at the empirical floor (p ≈ 0.001);
serotonin z = +9.5 and *more* anomalous after composition matching
(stratified expectation 11.9 % vs observed 28.8 %; survives the stricter
cell_class strata at z ≈ +11.4); dopamine, after removing the KC
contamination, is the **most feedback-enriched transmitter**: 51.7 %
backward (n = 682), z = +14.8 vs its composition-matched null and z = +4.4
under the stricter super_class×cell_class null (both at floor) — the
earlier 20.1 % / z +4.0 was dilution by 5,172 forward-dominated KCs.
**Refinement: octopamine is NOT enriched** (16.7 %, p = 0.17) — the honest
claim is "GABA/Glu elevated, serotonin extreme; not all neuromodulators".
MICrONS E-vs-I backward-share gap 21.41 pp, z = +37.8 under the within-layer
identity shuffle.
Caveats: fly NTs are Eckstein-2024 predictions; the KC→ACh override was NOT
applied in the annotation cache these tests ran on (uncommitted prep edit;
cache = raw predictions) — corrected per-transmitter rows live in
studies/label_nulls.md § Independent review, and the fly_ann cache (plus the
FLY_FINDINGS chemistry tables that inherit the contaminated DA row) must be
regenerated before COSYNE numbers freeze; MICrONS E/I is anatomical;
truncation plausibly inflates the MICrONS
*magnitudes* (complete I axons, truncated E axons) though significance stands
— direction analyzed in the study files.

### A4. The optimizer, blind to biology, re-derives the sensory→motor axis — and cell types come out as computational units. — status: reviewed (axis: prior review; contiguity statistic: independent review 2026-09-30)
- Fly lane medians ascend photoreceptors 7.6 k → lamina 24.9 k → medulla
  46.2 k → T4/T5 stage 69.6 k → lobula 76.5 k → mushroom body 102 k → visual
  projection 110.6 k → central complex 113.1 k → descending 130.8 k → motor
  136.5 k; the very last node of 136,648 is a neck motor neuron. Flow-label
  test: ρ = 0.263, perm. p ≤ 5·10⁻⁴.
- MICrONS: rank follows cortical depth pia→white matter (exc. median ranks
  L2/3 18.5 k → L4 26.9 k → L5 36.4 k → L6 46.9 k → WM 51.4 k; ρ = +0.356,
  p ≤ 5·10⁻⁴) — and the hierarchy is radial, not areal (rank vs V1→RL/AL axis
  ρ = +0.05).
- Cell types occupy contiguous rank blocks — now a population-level result
  (studies/label_nulls.md): over all 437 types with ≥ 20 members, median
  IQR/n = 0.0441 vs null 0.4886 ± 0.0026 (z = −172.7); **92.9 % of types are
  individually tighter than their own null's 1st percentile**. The order
  treats a cell type as one unit, type by type, not just on average.
  (Informative exceptions: mostly JO-* afferents split by the truncated-sensory
  honesty layer.)
Caveats: fly ρ modest because 87 % of neurons are "intrinsic"; MICrONS is not
the textbook L4-first circuit (thalamus outside the graph) — stated as a
feature, not an error.

### A5. Hemispheres are computationally interchangeable — the mirrored brain is a genuine twin optimum, and the measurable wiring asymmetry is zero. — status: reviewed (swap: prior review; re-tune: independent review 2026-09-30, all oracle re-scores bit-exact; twin-optimum contrast strengthened by the reviewer's displacement-profile reweighting, +15.2 → +27.7 pp)
Swapping 65,452 mirror pairs (95.8 % of all fly neurons, rank-matched within
cell type) costs only 2.21 pp of the 34.2 pp separating the champion from
chance — the mirrored brain keeps 93.5 % of the above-chance structure, with
no detectable extra price for crossing the midline. No per-type "left first"
rule exists (left earlier in 56.6 % of 267 testable types).
Re-optimization closes the question in the strong direction: applying the
champion's own refinement stages to the swapped order recovers **103 % of the
deficit** (82.05 → 84.33 %, vs the re-tuned champion's 84.26 %), so the
irrecoverable left/right asymmetry under this objective is **0 pp** — the
entire −2.21 pp was fine-tuning loss, not asymmetry (comparator scale: two
equally legitimate control endpoints differ by 0.06 pp). And it is a true
*twin* optimum, not a relaxation back: 56.7 % of mirror pairs STAY mirrored
after re-tuning while the same-side control's pairs mostly revert (58.5 %),
and the re-tuned mirror order matches the champion's coarse architecture
(Spearman 0.966) while keeping the exchanged hemisphere assignment.
Caveats: single seed-42 pairing; rank-matched, not morphological, twinning;
local refinement (a global re-run could differ).
*Algorithmic by-product, explicitly NOT a champion promotion:* the re-tuned
mirror order scores 84.3281 % (35,343,705) — +0.070 pp above the sitting H64
champion; flagged as a candidate variant ("structured-swap kick + stages
3+4") for the proper campaign gate ladder.

### A6. Interneurons sort ahead of their excitatory partners — for an arithmetic reason. — status: reviewed (lead: prior review; identity-shuffle test: independent review 2026-09-30)
In every cortical layer L1–L6 the MICrONS inhibitory lane leads its
excitatory partner by 6.7–12.9 k ranks (e.g. L4 inh 14.0 k vs exc 26.9 k;
WM excluded at n_inh = 18, n.s.). The never-before-run
E/I identity shuffle (2,000 perms, within-layer) puts the lead at the
empirical floor in every layer L2/3–L6 (z = +9.7…+24.4; mean lead 8,005
ranks, z = +16.8; quote L2/3–L6 or the mean, not L1 where n = 451 gives only
p = 0.004). Mechanism: I→E weight is
~3.6× heavier than E→I and 97.3 % of I→E weight is forward, vs 48.4 % for
E→I — an optimal order puts inhibition early to keep its heavy output forward.
Caveat: the I-weight share is inflated by reconstruction sampling (inhibitory
axons complete, excitatory truncated); the direction of this bias is analyzed
in the study file.

### A7. The cell-type wiring diagram is sufficient for the fly's entire feedforward order — the type-matched ensemble overshoots the real brain, which spends its sub-type wiring on reciprocity instead. — status: reviewed (studies/null_ladder_v2/REPORT.md, 2026-11-01; charter-pre-registered, stages A–D independently reviewed, L4 probe cleared its pre-stated KEEP threshold)
Null-ladder v2 (G03) closes the attribution A1 left open, on the fly only
(operator-parked elsewhere). With the full real wiring fixed and only synaptic
weights permuted (W0), the identical epoch-budgeted protocol loses 1.44 pp
(real outside all 12 draws; +1.25 pp at the 5× budget probe) — weight
placement carries estimator-visible order, though which weight–topology
coupling carries it is not identified at that rung. Fixing the real reciprocal
backbone plus exact degrees (L2R — the one null whose 2-cycle floor equals the
real 6.72 % of W bit-exactly, so no floor-netting can dissolve the contrast)
leaves the real graph +6.08 pp above all 12 draws (+6.31 at 5×): order in the
choice of non-reciprocal targets beyond degrees + backbone. But a
degree-corrected SBM at cell-type resolution (L4, K = 8,857 FlyWire types,
99.09 % direct coverage) INVERTS the ladder: FF(L4) = 84.23 ± 0.01 vs real
83.66 — the real brain below all 12 draws (62σ; empirical one-sided
p ≥ 1/13), probe-confirmed at 5× budget (84.756 against the pre-stated
KEEP ≥ 84.2029 / stability ≥ 84.396; the inversion grows to −0.65 pp).
Two-axis reading: the type ensemble frees 4.70 pp of forced reciprocal floor
(2.02 vs 6.72 % of W) yet converts only 0.57 pp into feedforward weight, and
on the registered excess-FB-over-own-floor axis the real graph still beats
every ensemble including L4 (9.62 vs 13.75 pp) — relative to its own
type-level diagram, the fly's sub-type wiring is net feedback-spending, buying
a 3.3× larger irreducible reciprocal core at the price of feedforward
orderability.
Caveats: every direction is protocol-conditional (achieved lower bounds under
an identical epoch budget; the greedy control flips on every structured null
and is direction-uninformative; 2-cycle upper bounds do not separate the
optima); neither registered L4 branch anticipated the inversion — the reading
imports the FF(null) > FF(real) branch registered, before any L4 run, for the
sibling rungs W0/L2R; L4 is non-nested vs L3 (91 direct cell types, 31.9 % of
neurons, span the 17 lanes), so its +2.34 pp over L3 is a non-nested
comparison; the resemblance to the mouse meso inversion (A1) is a
co-occurrence, not an established shared mechanism — with the fly's backbone
held fixed (L2R) the real graph sits far ABOVE its null; fly-only by charter
(mouse/MICrONS arms parked, scale confound open); W0's 1.44 pp is a separate
axis, not an additive ladder increment.
Provenance: biology/studies/null_ladder_v2/ (charter, REPORT, 5 reviews);
biology/results/nullmodels_v2/ (39 runs + bounds_cache).

---

## Tier B — strong supporting claims

### B1. Globally feedforward, locally recurrent — and the "one giant loop" paradox resolved. — status: reviewed
≥84.26 % of fly synaptic weight can point one way, yet 92.8 % of neurons sit
in one giant strongly-connected component (spanning ranks 7,264–134,103,
holding 1,173,079 of 1,173,245 FB edges). Both are true at once: recurrence is
everywhere but *thin*. Reconciles "the fly brain is not predominantly
feedforward" (Lin et al. 2024) with an 84 % feedforward spine. The ≤15.74 %
feedback ceiling is an upper bound from a heuristic (public contest best
84.61 % tightens it to ≤15.39 %), not an optimality proof — always stated.

### B2. Where feedback lives: the full pathway budget. — status: reviewed (headline: prior review; budget: independent review 2026-09-30, lane matrix re-aggregated from the edge list to 0 ulp)
Visual-projection→medulla connections run 38 % feedforward (i.e. mostly
*against* the global order) — descending re-entry into the visual system.
Visual centrifugal cells sit *early* (median 43.5 k, before medulla) with
99 %-forward outputs: the optimizer "legalizes" an anatomical feedback pathway
by moving its source early — suggesting two kinds of feedback: broadcast
(placeable forward) vs truly recurrent (unavoidable cost).
The budget (lane-matrix mining, FF column re-sums to the champion's 84.2582 %
exactly): **57.5 % of all backward weight never leaves the optic lobes**
(36.6 % between visual lanes + 21.0 % within one); the named down-hierarchy
retro-pairs alone carry 24.0 % (worst: VPN→visual-centrifugal 24.7 % FF;
heaviest: lobula-plate→Tm at 46.0 % FF carrying 1.64 % of all weight;
medulla→lamina 57.8 % at 1.23 %). The central brain contributes ~33 % almost
entirely as *within*-lane recurrence (between central lanes: only 2.7 %).
Twelve "one-way streets" (≥ 97 % FF) are the sensory ascents (photoreceptors→
lamina 99.6 %, AL→MB 99.5 %) and the Tm→VPN trunk (98.7 % at 4.60 % of all
weight — the heaviest one-way trunk; three within-optic pairs carry more
weight but below 97 % FF). Descending→central at 57.2 % FF is motor re-entry — the wiring one
would draw for efference copy.

### B3. Pipelines of very different depth converge on the center; the architecture is streams × depth, not a layer cake. — status: reviewed
Olfaction reaches the mushroom body in ~2 stages where vision takes ~5;
42.5 % of forward weight never leaves its own stream; stream-skeleton edges
run 91–100 % FF. A 17-stream × 16-bin grid traps only 14.4 % of weight
off-pattern vs 25.3 % for the best 10-layer cake. The fly is not
slice-layered; the mouse meso-graph is (3.4 % at L = 10).

### B4. Robustness: every qualitative conclusion survives a substantially degraded order; attenuation only ever fakes randomness, not structure. — status: reviewed
At 80 % FF (vs champion 84.26 %) every headline number is indistinguishable;
at 70 % all *shapes* hold while magnitudes compress toward chance. The bias
direction is safe: a suboptimal order can only understate structure. (Also:
coarse structure alone holds ~70 % of weight forward — most of the score
lives in the coarse order.) Methodological backbone for everything above.

### B6. The central brain is a plateau of near-equal orders — the periphery selects which one, and the central complex is the pivot. — status: reviewed with corrections (studies/central_only.md, 2026-09-30)
Amputating everything but the central brain (32,294 neurons, 29.9 % of synaptic
weight; central-central synapses ~2× heavier than average) costs the center
almost nothing in achievable feedforwardness: re-optimizing the isolated
center reaches 83.43 % FF vs 82.70 % for the whole-brain champion restricted
to the same neurons (+0.74 pp). But *which* near-optimal order is used changes
substantially: Spearman 0.74 vs a measured same-seed noise floor of 0.96;
58 % of central neurons move > 5 % of the line. The sharpest disagreement is
the **central complex** (n = 2,848): central wiring alone robustly sends it to
the sink end ("everything converges on the CX"), the whole-brain champion
holds it mid-line among central neurons (0.51 of the central line; on the
full-brain line CX sits at 0.83) — and this is *not* a direct interface
effect, since > 99 %
of CX in- and out-weight is intra-central (0.48 % of output to descending
neurons); the pull is indirect, through central partners pinned by the
periphery. The antennal-lobe/lateral-horn backbone, by contrast, keeps its
arrangement with all sensory drive removed; the mushroom-body stage shifts
modestly (KC 7.6 %, MBON 5.8 % median displacement, 4–8× noise) while keeping
the AL→MB→LH sequence. Caveats: heuristic optima on both sides; the +0.74 pp
gap bounds the true embedding cost from neither side (central-only heuristic
biases it down, full-brain slack up); single whole-brain comparator (one
CUDA artifact, repeat noise unmeasured on that side); 67.9 % of central
neurons unannotated at cell_class level.

### B5. The order audits its own dataset — at both ends. — status: reviewed (tail: prior review; head-end result: independent review 2026-09-30, exact)
Zero-output "sinks" pile up at the end of the order; in the fly they are
mostly reconstruction artifacts (2,447 pure sinks) *except* motor/endocrine
neurons, where sinking is the biology; in MICrONS 96.5 % of zero-output
fragments land in the last 10 % of the order. New: the MICrONS "unclassified"
lane (4,202 objects, suspiciously early at median rank 18.1 k) decomposes
into merge bodies (ordinary ranks), input-starved singletons, and 473
zero-*input* orphan-output objects at median rank 309 — the extreme FRONT.
Both ends of the order are QC zones: output-less debris sinks to the tail,
input-less debris floats to the head. The same mechanism that finds motor
neurons finds segmentation errors — a built-in QC readout.

---

### B7. The brain's hubs are inhibitory normalizers, not relays — and the feedback superlatives land on the famous giants. — status: reviewed with corrections (studies/untapped.md §A, 2026-09-30)
11 of the fly's top-12 degree hubs are GABAergic local giants. #1–2 by degree
AND #1–2 feedback *receivers*: the two CT1 cells (one per hemisphere, giant
amacrine, ME>LO, deg 16,784 / 16,581, mid-order at 42–43 %). #1–2 feedback
*senders*: the two APL cells (mushroom-body giant interneuron, at 74–77 % of
the order), together carrying 1.07 % of ALL backward weight in the brain —
the optimizer re-derives "APL is the brain's biggest feedback device" without
being told what APL is (CT1: Meier & Borst 2019; APL: Lin et al. 2014 /
Amin et al. 2020). (Degree-hub identity is order-independent; the
optimizer's own contribution is the feedback sender/receiver superlatives.)
Early-order pure-FB receivers are the broadcast loop-closers (cL21 visual
centrifugal, antennal-lobe LNs). Connects to A3: the hub layer is
normalization machinery, and it is inhibitory.

### B8. The E/I feedback asymmetry survives at full reconstruction quality; the depth trend cannot be tested there. — status: reviewed with corrections (studies/untapped.md §D–E1, 2026-09-30)
On the 245-neuron proofread-axon MICrONS subgraph (truncation-free but
composition-biased: 40 % inhibitory, 71.6 % of weight I→I), the E/I direction
asymmetry replicates cleanly: I→E is 88.4 % FF vs E→I 53.1 % (full graph:
97.3 / 48.4) — E→I remains the one half-backward pair, so the "pay each loop
on its lighter side" mechanism stands clear of truncation bias. The cortical
depth trend does NOT replicate there — but the champion order restricted to
the same 245 nodes loses it too (ρ = −0.05, n.s.), so this is composition,
not counter-evidence. Also: E→E (80.6 % FF) and I→I (78.1 %) sit near the
graph-wide 83.2 % baseline (−2.6 / −5.1 pp) — the entire E/I feedback story
is *cross-class*.

## Tier C — context, curiosities, open threads

- **C1. Two eyes, two towers:** optic lanes split ~99/1 by hemisphere on the
  wiring's spectral coordinate; central lanes mix to ~50/50; only 2.95 % of
  within-optic weight crosses the midline; the mushroom body's split hides on
  the *second* spectral coordinate. (reviewed)
- **C2. Mouse meso: no reclaimable feedback** — all 124 FB edges have an FF
  return path (independently re-confirmed 124/124 in the 2026-09-30 review);
  feedback is long-range and convergent onto one hub. The
  id→region mapping was investigated and HONESTLY CLOSED as not establishable
  from available sources (Oh 2014 and Calabrese refuted by density/edge-count
  fingerprint; exact float fingerprint + per-node FB ledger precomputed for
  the day a mapping arrives — studies/untapped.md §C). Hub 92 stays unnamed.
- **C3. The measured "amount of feedback in a brain" depends on the ordering
  instrument.** The standard continuous method (trophic levels) loses 6.48 pp
  on the fly (77.78 % vs 84.26 %, ρ = 0.76) — i.e. it reports a net ~2.7 M
  extra synapses as feedback (gross: 3.8 M champion-forward synapses pointed
  backward, 1.1 M the reverse), an excess equal to 41 % of the champion's
  15.74 % feedback budget; mouse loses 3.2 pp. A methodological point for any
  cross-study comparison of "feedforwardness". (reviewed with corrections,
  studies/untapped.md §E3, 2026-09-30)
- **C4. Central-brain autonomy** — answered; promoted to B6.
- **C5. Who are the hubs?** — answered (CT1 / APL); promoted to B7.
- **C6. Octopamine breaks the "neuromodulators go backward" story** —
  confirmed: 16.7 % backward, p = 0.17 vs its composition-matched null
  (n = 160 neurons); the refined claim (serotonin extreme, OA at baseline)
  is now folded into A3.

---

## COSYNE abstract skeleton (draft, updates as studies land)

Title direction: *"Maximally feedforward orderings of whole connectomes:
a comparative anatomy of feedback in three brains."*
Narrative: instrument (A4 validation) → how much feedback is unavoidable and
is it more or less than in **matched random graphs** (A1; say "matched
nulls", not "chance", and flag that fly-vs-mouse compares level of
description as well as species; fly L2/L3 direction is protocol-conditional
— quote as "upper bounds under identical budget", rigor from the 2-cycle
bounds) → what feedback is made of and where it lives (A3, A2, B2) →
invariances and robustness (A5, B4).

---

*Sources: layering/FLY_FINDINGS.md, layering/MICRONS_FINDINGS.md,
layering/QA_ORDER_AND_ROBUSTNESS.md, layering/RESEARCH_NOTES.md, the review
files beside them, and biology/studies/* as they land. Every number above
traces to a logged artifact; see the source docs for exact file paths.*
