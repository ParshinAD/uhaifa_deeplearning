# Label-permutation nulls for three descriptive headline findings

`status: reviewed with corrections` (independent review 2026-09-30, see bottom;
one correction: the T1 "dopamine" row is KC-contaminated — the KC→ACh override
this study claims was applied is NOT in the annotation cache it ran on)

| | |
|---|---|
| question | Do the three descriptive headline findings (fly transmitter–feedback enrichment, MICrONS early inhibition / E-dominated feedback, fly cell-type contiguity) survive a proper permutation null, so they can be quoted with significance at COSYNE? |
| script | `biology/analysis/exp_label_nulls.py` (CPU only; the champion orders are FIXED — no optimization anywhere) |
| command | `$PY biology/analysis/exp_label_nulls.py` (repo root; ~70 s total) |
| results | `biology/results/label_nulls_fly.json`, `biology/results/label_nulls_microns.json` (+ raw null draws in `label_nulls_{fly,microns}_nulls.npz`) |
| figures | `biology/outputs/label_nulls_t1_fly_nt.png`, `label_nulls_t2_microns_ei.png`, `label_nulls_t3_fly_contiguity.png` |
| seed / perms | seed 42 (child streams per test/null, pinned in CONFIG); T1/T2: 2,000 permutations per null, T3: 1,000 draws |
| p convention | two-sided empirical with add-one on both tails: `p = min(1, 2·min((1+#{null≤obs})/(N+1), (1+#{null≥obs})/(N+1)))`. Floors: 2/2001 ≈ **0.001** (T1, T2), 2/1001 ≈ **0.002** (T3). "p = floor" below means zero null draws beyond the observed value. |
| orders | fly H64 (84.2582 % FF), MICrONS H42 (83.2409 % FF), loaded parity-gated via `layering.io_utils` (frozen oracle re-score). FF/backward classification is the oracle rule `rank[src] < rank[tgt]` applied as direct rank comparison. |

All label shuffles are performed on the FIXED champion order: the null asks
"given this order and this graph, are the labels exchangeable?" — it never
re-optimizes.

---

## T1 — Fly: transmitter-feedback enrichment (now significant, with one honest exception)

**Statistic.** Per-transmitter backward weight share: weight of out-edges of
neurons with predicted transmitter *t* that go backward (`rank[src] > rank[tgt]`),
divided by the total out-weight of *t*-neurons. Brain-wide backward share: 15.742 %.

**Reproduced gate.** The observed shares equal
`layering/outputs/fly_streams/report_streams.json → nt_direction.fb_share_within_nt`
to < 1e-9 relative (asserted at runtime): ACh 11.3599, GABA 20.8966, Glu 22.2492,
DA 20.1066, 5-HT 28.8256, OA 16.7236; brain-wide = 100 − champion FF = 15.7418.

**Nulls.** (a) *global*: permute `top_nt` across all 136,648 neurons;
(b) *stratified*: permute `top_nt` within each `super_class` stratum. The
stratified null is the one that matters: it controls for composition (the optic
lobes are ACh-heavy AND early in the order), so any surviving effect is not a
regional artifact. 2,000 permutations each, seed 42.

| transmitter (n neurons) | observed % | global null mean±sd | z | p | stratified null mean±sd | z | p |
|---|---|---|---|---|---|---|---|
| ACh (85,086) | 11.360 | 15.740 ± 0.132 | −33.2 | floor | 15.891 ± 0.135 | −33.4 | **floor** |
| GABA (19,014) | 20.897 | 15.752 ± 0.418 | +12.3 | floor | 14.733 ± 0.401 | +15.4 | **floor** |
| Glu (24,540) | 22.249 | 15.745 ± 0.356 | +18.3 | floor | 15.937 ± 0.367 | +17.2 | **floor** |
| DA (5,854) | 20.107 | 15.707 ± 0.802 | +5.5 | floor | 16.904 ± 0.792 | +4.0 | **floor** |
| 5-HT (1,891) | 28.826 | 15.684 ± 1.360 | +9.7 | floor | 11.938 ± 1.777 | +9.5 | **floor** |
| OA (160) | 16.724 | 15.597 ± 3.722 | +0.3 | 0.598 | 10.250 ± 4.385 | +1.5 | **0.171 (n.s.)** |
| NA (103) | 40.541 | 15.662 ± 4.527 | +5.5 | 0.004 | 15.345 ± 4.416 | +5.7 | 0.006 |

(floor = 2/2001 ≈ 0.001; NA = unpredicted transmitter, 103 neurons, reported for completeness only.)

**Reading.** Every headline transmitter effect survives the
composition-matched null at the p floor: ACh is strongly *depleted* in feedback
(z ≈ −33), GABA and glutamate are strongly *enriched* (z +15.4 / +17.2),
dopamine is moderately but significantly enriched (z +4.0 — stratification
roughly halves its z from the global null's +5.5, i.e. part of the raw DA
elevation was composition), and serotonin is extreme (z +9.5; its
composition-matched expectation is only 11.9 %, making the observed 28.8 %
even more anomalous than the raw comparison suggested). **Octopamine is the
honest exception**: 16.7 % backward is statistically indistinguishable from
its null under both shuffles (p = 0.60 global, p = 0.17 stratified, n = 160).
The refined claim is therefore: *inhibitory/classical-modulatory transmitters
(GABA, Glu, DA) are feedback-enriched and serotonin is extreme, but NOT all
neuromodulators are — octopamine sits at its composition-matched baseline.*

**Caveats.** `top_nt` is a prediction (Eckstein et al. 2024), not ground
truth; Kenyon cells were overridden to ACh in the `fly_ann` cache (documented
choice, kept here — the ACh row therefore includes the KC population under
that override). The null tests
label exchangeability on the fixed order; it does not test graph rewiring.
Small-n transmitters (OA 160, NA 103) have wide nulls — the OA null spans
~2–30 %, so only a large effect could have been detected there.

---

## T2 — MICrONS: E/I identity shuffle (the §8.2 test, now run; everything survives)

**Statistics.** (a) per-layer inhibitory lead = median champion rank of E in
layer L minus median rank of I in layer L (positive = inhibitory earlier),
plus the mean lead over tested layers; (b) E-output vs I-output backward
weight share and the difference E−I.

**Reproduced gate.** Observed values equal
`layering/outputs/microns_ann/diagnostics.json` to < 1e-9 relative (asserted):
E backward share 26.6222 %, I 5.2102 %; leads L1 6,640 / L2/3 6,936 /
L4 12,935 / L5 6,672 / L6 6,843.5 ranks (the "6.7–12.9k ranks ahead" finding).

**Nulls.** Shuffle E/I identity among the 63,074 E/I-labeled neurons (55,855 E,
7,219 I; the 4,460 unlabeled stay unlabeled) — (a) globally, and (b) stratified
within cortical-layer band (preserves each band's E/I count exactly). 2,000
permutations each, seed 42. Layers tested where both groups have ≥ 20 members:
L1–L6. **WM excluded** from statistic (a): only 18 inhibitory members (its
observed lead is −3,927, n far below floor); WM neurons still participate in
statistic (b) and in the stratified shuffle.

| statistic | observed | global null mean±sd | z | p | stratified null mean±sd | z | p |
|---|---|---|---|---|---|---|---|
| lead L1 (280 E / 171 I) | 6,640 | −753 ± 3,947 | +1.9 | 0.003 | −92 ± 2,093 | +3.2 | **0.004** |
| lead L2/3 (10,381 / 1,435) | 6,936 | −34 ± 606 | +11.5 | floor | −15 ± 577 | +12.1 | **floor** |
| lead L4 (18,551 / 1,982) | 12,935 | +8 ± 490 | +26.4 | floor | +5 ± 530 | +24.4 | **floor** |
| lead L5 (11,457 / 2,640) | 6,672 | +13 ± 599 | +11.1 | floor | +2 ± 477 | +14.0 | **floor** |
| lead L6 (14,686 / 973) | 6,843.5 | +136 ± 556 | +12.1 | floor | +210 ± 684 | +9.7 | **floor** |
| mean lead over L1–L6 | 8,005.3 | −126 ± 818 | +9.9 | floor | +22 ± 476 | +16.8 | **floor** |
| E backward share (%) | 26.622 | 16.818 ± 0.068 | +145.1 | floor | 16.889 ± 0.070 | +139.0 | **floor** |
| I backward share (%) | 5.210 | 16.828 ± 0.523 | −22.2 | floor | 16.325 ± 0.481 | −23.1 | **floor** |
| difference E−I (pp) | 21.412 | −0.010 ± 0.590 | +36.3 | floor | +0.564 ± 0.551 | +37.8 | **floor** |

(floor = 2/2001 ≈ 0.001.)

**Reading.** The never-run test proposed in `MICRONS_FINDINGS.md` §8 (item 2)
now exists, and E/I identity is decisively non-exchangeable: shuffling it
destroys both the early-inhibition lead (per-layer z +9.7…+24.4, mean-lead
z +16.8) and the direction-of-feedback split (E−I difference z +37.8). Both
headline claims got **stronger** — they are now permutation-significant at the
floor under the composition-matched (within-layer) null. The only weak spot is
L1 taken alone (z +3.2, p = 0.004, only 451 labeled neurons); the layer-wise
claim should be quoted from L2/3–L6 or from the mean lead.

**Caveats — truncation bias, direction stated explicitly.** In MICrONS,
inhibitory axons are locally complete while excitatory axons are heavily
truncated (distal arbors lost). This biases the observed MAGNITUDES in the
direction of the finding, in both statistics: (i) for the lead (a), complete
I axons carry relatively more measured out-weight (7,219 I neurons carry
6.42 M out-synapses vs 55,855 E neurons carrying 7.60 M), and the optimizer
places heavy senders early — so part of the I lead may be reconstruction
artifact, and the true lead is plausibly SMALLER than 8.0k ranks; (ii) for
the backward share (b), the surviving E out-edges are the proximal branches —
which include the light return arms of local reciprocal E↔I motifs — while
the lost distal E branches are plausibly more feedforward-like, so the true E
backward share (and hence the 21.4 pp E−I gap) is plausibly smaller too. The
permutation p-values themselves remain valid as statements about THIS
reconstruction (the null is conditional on the measured graph); the caveat
attaches to the effect sizes, not to the significance. The 249 proofread-axon
neurons (`data/processed/microns_proofread.npz`) remain the built-in control
for magnitude. E/I labels are from the soma-based Allen classifier
(`allen_soma_ei_class_model_v1`), not morphology ground truth; layer bands
are our density-landmark calibration below a fitted pia proxy (display
annotation, not ground truth).

---

## T3 — Fly: cell-type contiguity across ALL 437 types (spot-check → population test)

**Statistic.** Per-type champion-rank IQR normalized by n = 136,648 (the
repo's `IQR/n` convention from `exp_degraded_order.py`), summarized as the
median over the 437 types with ≥ 20 members. Types read from
`data/external/flywire_neuron_annotations.tsv.gz` with the exact
`exp_mirror_swap.load_types` fallback (cell_type → `<class> (class)` →
`unannotated`; `unannotated` excluded, 13 nodes).

**Reproduced gate.** With the repo's earlier ≥ 100-member convention (153
types) the observed median IQR/n is **0.04913**, matching the QA anchor 0.049
(`QA_ORDER_AND_ROBUSTNESS.md`; random-order reference there: 0.497).

**Null.** Permute type labels across neurons — implemented as a joint random
permutation of the rank vector, which is exactly equivalent (each type of
size k receives a uniform random k-subset of ranks, jointly without
replacement) and yields the correct joint distribution for the
median-over-types statistic. 1,000 draws, seed 42.

| statistic | observed | null mean±sd | z | p |
|---|---|---|---|---|
| median IQR/n over 437 types | **0.0441** | 0.4886 ± 0.0026 | **−172.7** | **floor (2/1001 ≈ 0.002)** |

Per-type distribution: **406/437 types (92.9 %) are tighter than their own
null 1st percentile**, and 400/437 (91.5 %) are tighter than ALL 1,000 of
their null draws (per-type one-sided p at the 1/1001 floor). The largest
types are among the tightest (R1-6 n=7,167 IQR/n 0.057; KCg-m n=2,189 0.023;
Tm3 n=1,756 0.051).

**Reading.** The contiguity finding was previously a spot-check plus a single
median; it is now a population-level result: cell types occupy rank blocks
~11× tighter than label exchangeability allows, and this holds type-by-type
for 93 % of all types, not just for a favorable median. The claim got
**stronger and sharper** — and the 31 exceptions are informative rather than
damaging: 14/31 are Johnston's-organ auditory/mechanosensory afferent types
(JO-*, IQR/n 0.68–0.89 — consistent with the known truncated-sensory-fragment
honesty layer splitting these populations across the order), 3 are coarse
`(class)` fallback aggregates (e.g. `visual (class)` 0.94), and several PAM
dopaminergic subtypes sit just above their null 1st percentile with small n.

**Caveats.** The null tests exchangeability of type labels on the fixed
order; it does not test whether the graph could be rewired to break
contiguity. Type labels are hierarchical annotations (Schlegel et al. 2024);
the 8 fallback `(class)` aggregates are not real cell types (3 of the 31
exceptions). With 1,000 draws the per-type p floor is ~0.001, so "tighter
than all draws" is the strongest per-type statement available at this budget.

---

## Summary — what changed for the COSYNE claims

| finding | before | after | verdict |
|---|---|---|---|
| fly: GABA/Glu/5-HT feedback-enriched, ACh depleted | descriptive shares | all at p floor under composition-matched null; 5-HT even more extreme vs its stratified expectation (11.9 % expected, 28.8 % observed) | **stronger** |
| fly: dopamine enriched (20.1 %) | descriptive | significant (p floor) but z drops 5.5 → 4.0 under stratification — partly composition | **held, refined** |
| fly: "modulators are feedback-enriched" (general form) | implied | octopamine is at its composition-matched baseline (p = 0.17, n = 160) — claim must be stated per-transmitter, not for "neuromodulators" | **refined/weakened** |
| MICrONS: interneurons lead by 6.7–12.9k ranks per layer | descriptive | per-layer p at floor for L2/3–L6, mean-lead z +16.8; L1 alone only p = 0.004 | **stronger** (quote L2/3–L6) |
| MICrONS: E output 26.6 % backward vs I 5.2 % | descriptive | E−I gap z +37.8, p floor under within-layer shuffle; magnitude likely inflated by E-axon truncation (direction stated) | **stronger, magnitude caveated** |
| fly: cell types form contiguous rank blocks | spot-checked median 0.049 | population test: z −173, 92.9 % of 437 types beat their own null 1st percentile; exceptions concentrated in JO-* afferents and class aggregates | **much stronger** |

Every number above traces to `biology/results/label_nulls_fly.json` /
`label_nulls_microns.json` (summaries, gates, exceedance counts) and
`label_nulls_{fly,microns}_nulls.npz` (raw null draws); the run is
reproducible with the pinned seed via the command at the top.

---

## Independent review (2026-09-30)

Strict independent pass. All checks re-derived with code written for the
review — no function of `exp_label_nulls.py` reused (shared code: only the
parity-gated champion loader `layering.io_utils` and numpy). Review artifacts:
`biology/analysis/review_label_nulls.py`,
`biology/analysis/review_label_nulls_kc.py`;
results `biology/results/review/review_label_nulls{,_kc}.json` (+
`review_label_nulls_draws.npz`). Independent nulls: **K = 300 permutations,
seed 20260930** (deliberately ≠ 42), own stratified-permutation
implementation; review p floor is therefore 2/301 ≈ 0.0066 ("floor" below).

### Per-claim verdicts

| claim | verdict |
|---|---|
| T1 ACh depleted (z ≈ −33, floor) | **CONFIRMED** (review z −32.4; survives finer super×cell strata at z −25.5; with the KC correction applied it strengthens to −36.9) |
| T1 GABA / Glu enriched (z +15.4 / +17.2, floor) | **CONFIRMED** (review z +15.5 / +17.6; survive super_class×cell_class strata at z +9.0 / +12.3) |
| T1 serotonin extreme (z +9.5, stratified expectation 11.9 %) | **CONFIRMED** (review z +8.71, null 12.01 ± 1.93; survives and *strengthens* under super×cell strata: z +11.7; unaffected by the KC issue) |
| T1 octopamine n.s. (p 0.17) | **CONFIRMED** (n.s. under every null tried: p 0.31 super_class / 0.60 cell_class / 0.20 crossed; verdict identical) |
| T1 dopamine "enriched at 20.1 %, z 5.5 → 4.0, partly composition" | **CORRECTED** — the row is not dopamine; see below |
| T1 caveat "KCs were overridden to ACh in the fly_ann cache; the ACh row includes KCs" | **CORRECTED** — factually false; the cache carries the RAW predictions (override absent), so the ACh row *excludes* KCs and the DA row *contains* them |
| T2 gates (E 26.622 / I 5.210; leads 6,640 / 6,936 / 12,935 / 6,672 / 6,843.5) | **CONFIRMED** by per-edge groupby + per-layer medians, exact |
| T2 early-inhibition lead destroyed by E/I shuffle (mean z +16.8; L2/3–L6 at floor; L1 weak) | **CONFIRMED** (review mean-lead z +16.75; per-layer z +11.7 / +24.5 / +14.0 / +9.9; L1 z +3.4 — agree it is the weak spot) |
| T2 E−I backward gap 21.41 pp, z +37.8 | **CONFIRMED** (review z +39.0) |
| T3 median IQR/n 0.0441 vs null ≈ 0.4886, z ≈ −173; 406/437 tighter than own null pct1; 400/437 below all draws | **CONFIRMED** (review honest label-permutation null: mean 0.48832 ± 0.00239, obs at floor; 407/437 and 399/437 — MC-consistent) |
| p convention (two-sided add-one) | **CONFIRMED** — formula correct; all 34 JSON stats recompute exactly from the stored npz draws (0 mismatches) |
| determinism / reproducibility | **CONFIRMED** — a fresh full run (same seed, scratch output dir) reproduces both JSONs identically (modulo provenance/seconds) and all 13 npz arrays **bit-for-bit**; 34 s wall on the MacBook, consistent with the claimed ~70 s; all three figures exist and match run timestamps |

### Gate re-derivation (own code, per-edge route)

Fly per-NT backward shares recomputed by per-edge pandas groupby on
`nt[src]` × oracle rule `rank[src] > rank[tgt]`: ACh 11.360, GABA 20.897,
Glu 22.249, DA 20.107, 5-HT 28.826, OA 16.724, brain-wide 15.742 — all equal
to the study's (and to `report_streams.json`) to the displayed precision;
champion FF 84.25818 (fly) / 83.24085 (MICrONS) parity-gated on load.
MICrONS E 26.622 / I 5.210 and all five leads reproduced exactly; WM lead
confirmed −3,927 with 500 E / 18 I. T3 gates: 437 types ≥ 20 members, median
0.04411; 153 types ≥ 100, median 0.04913 (QA anchor 0.049).

### Independent-null comparison (study: 2,000 perms, seed 42 vs review: 300 perms, seed 20260930)

| statistic (stratified null) | study null mean ± sd | study z / p | review null mean ± sd | review z / p | consistent? |
|---|---|---|---|---|---|
| fly 5-HT share | 11.94 ± 1.78 | +9.50 / floor | 12.01 ± 1.93 | +8.71 / floor | yes |
| fly OA share | 10.25 ± 4.39 | +1.48 / 0.171 | 11.11 ± 5.05 | +1.11 / 0.312 | yes (both n.s.) |
| fly ACh share | 15.89 ± 0.135 | −33.4 / floor | 15.89 ± 0.140 | −32.4 / floor | yes |
| fly GABA share | 14.73 ± 0.401 | +15.4 / floor | 14.75 ± 0.398 | +15.5 / floor | yes |
| fly Glu share | 15.94 ± 0.367 | +17.2 / floor | 15.93 ± 0.360 | +17.6 / floor | yes |
| fly DA share (uncorrected labels) | 16.90 ± 0.792 | +4.04 / floor | 16.89 ± 0.790 | +4.07 / floor | yes (but see correction) |
| MICrONS mean I-lead | +22 ± 476 | +16.78 / floor | +14 ± 477 | +16.75 / floor | yes |
| MICrONS lead L1 | −92 ± 2,093 | +3.22 / 0.004 | −121 ± 2,007 | +3.37 / floor(300) | yes |
| MICrONS E−I diff (pp) | 0.564 ± 0.551 | +37.8 / floor | 0.580 ± 0.534 | +39.0 / floor | yes |
| T3 median IQR/n | 0.48863 ± 0.00257 | −172.7 / floor | 0.48832 ± 0.00239 | −186.0 / floor | yes (z is 1/sd-sensitive; magnitude consistent) |

### CORRECTION — the T1 "dopamine" row is 88 % Kenyon cells

Evidence chain (review_label_nulls.py, `redteam_kc_override`):
`layering/outputs/fly_ann/node_ann.npz` `top_nt` is **bit-identical** to the
raw `top_nt` column of the Schlegel TSV for all 136,648 nodes (0 differing);
the `nt_overrides` block in `layering/tools/fly_ann_prep.py` is an
**uncommitted working-tree edit**, and the cache predates it (no `top_nt_raw`
key in the npz, no `nt_corrections` in the meta). So the study — and
`report_streams.json`, whose values it gates against — ran on **raw Eckstein
predictions**. Of the 5,854 "dopamine" neurons, **5,172 are Kenyon cells**
(cell_class Kenyon_Cell: 5,177 = 5,172 DA + 4 ACh + 1 GABA predicted), which
Eckstein et al. flag as a systematic classifier error (KCs are cholinergic).

Corrected numbers (KC → ACh applied, review nulls K = 300,
`review_label_nulls_kc.json`):

| group | n | backward share | z vs within-super_class null | z vs super×cell null |
|---|---|---|---|---|
| KC alone | 5,177 | **7.54 %** (forward-dominated) | — | — |
| DA as published (KC-contaminated) | 5,854 | 20.11 % | +4.0 (study) / +4.1 (review) | **+1.8, p 0.08 — n.s.** |
| **DA corrected (true DA predictions)** | **682** | **51.70 %** | **+14.8, floor** | **+4.4, floor** |
| ACh corrected (incl. KCs) | 90,259 | 11.30 % | −36.9, floor | −27.1, floor |

Consequences:
1. The published DA row **understates** dopamine: mixing in 5,172
   forward-dominated KCs diluted 51.7 % down to 20.1 %. Corrected, dopamine is
   the **most** feedback-enriched transmitter (ACh 11.3 < GABA 20.9 < Glu 22.2
   < 5-HT 28.8 < **DA 51.7**), not a "moderate" case.
2. The study's reading "z halves 5.5 → 4.0 under stratification — partly
   composition" is an artifact of the contamination: the *uncorrected* DA row
   even goes n.s. under cell_class-level stratification (z +1.5, p 0.23,
   because the KC stratum then absorbs it), while the *corrected* DA row
   survives every control including the strictest crossed strata (z +4.4).
3. The T1 caveat is inverted: the ACh row as published EXcludes KCs; the
   corrected ACh row (with KCs, 90,259 neurons) barely moves
   (11.36 → 11.30 %) and gets stronger (z −36.9). GABA/Glu/5-HT/OA rows are
   essentially untouched (GABA loses 1 neuron).
4. Downstream: `report_streams.json` / FLY_FINDINGS chemistry tables carry the
   same contaminated DA row; the fly_ann cache should be regenerated with the
   (now-present) override and `top_nt_raw` before COSYNE numbers are frozen.

### Robustness findings (red-team)

- **(3a) Is super_class stratification enough?** Not quite, but the headline
  survives where it matters. Under cell_class and super_class×cell_class
  stratification (finer composition control; cell_class NA = 31,664 neurons
  remains one free-mixing stratum): 5-HT *strengthens* (z +11.4 / +11.7 —
  its enrichment is not a cell-class composition artifact; note 5-HT-labeled
  neurons are 66 % sensory, spread over many classes, so the null retains
  power), ACh/GABA/Glu hold (z −25 / +9 / +12), OA stays n.s. Only the
  (contaminated) DA row falls — see correction above. Caution: for
  transmitters that nearly coincide with a single cell_class (true DA ≈ DAN),
  cell_class stratification is close to conditioning on the effect itself;
  the crossed-strata z +4.4 for corrected DA is therefore a conservative
  lower bound.
- **(3c) T3 equivalence.** The claimed "joint rank permutation ≡ label
  permutation" is exactly right: both induce a uniform random assignment of
  the rank multiset to the fixed type partition (disjoint k-subsets drawn
  jointly without replacement), and every IQR/n statistic is a function of
  that assignment only. Verified empirically: the review null permutes the
  **type-label vector** (argsort-grouped), not the rank vector, and lands on
  the same null (0.48832 ± 0.00239 vs 0.48863 ± 0.00257).
- **(3e) WM exclusion (T2).** Including WM in the lead statistic does not
  rescue the null: observed 7-layer mean lead 6,016.6 ranks, z +6.97, still
  at floor. WM's own negative lead (−3,927) is itself far from significant
  (z −0.97, p 0.35, n_inh = 18) — the exclusion is honest bookkeeping, not a
  result-shaping choice. Recommend stating "cortical layers L1–L6" rather
  than "every lane" when quoting the per-layer claim, since WM's point
  estimate has the opposite sign.
- **Multiple comparisons** (minor): everything quoted at the floor survives
  any correction over the ~19 tests trivially (0.001 × 19 < 0.05); the two
  non-floor survivors (L1 p 0.004, NA p 0.004–0.006) would survive Bonferroni
  within their families but are correctly flagged as weak/completeness-only.

### Status

`reviewed with corrections`. T2 and T3 are confirmed as stated. T1 is
confirmed for ACh/GABA/Glu/5-HT/OA, **corrected for dopamine** (the row is
KC-dominated; true-DA enrichment is far stronger than published) and for the
override caveat (the override was NOT applied). FINDINGS.md A3 must be
amended accordingly (A4/A6 quote this study faithfully; A6 should say
"L1–L6" instead of "every inhibitory lane" since WM's n.s. point estimate is
negative).
