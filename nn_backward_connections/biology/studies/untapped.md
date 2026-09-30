# Untapped results — mining computed-but-never-interpreted numbers

`status: reviewed with corrections` (independent review 2026-09-30, see bottom)

*Five items (A–E) from the inventory of results that were computed but never turned
into biology, plus one blocking gap (C). Every number below comes from a logged run
of a `biology/analysis/exp_untapped_*.py` script (re-runnable via `$PY`, the `allen`
interpreter) writing JSON to `biology/results/untapped/`; orders are the pinned,
parity-gated champions (fly H64 84.2582 %, mouse H63 93.1754 %, MICrONS H42
83.2409 %) and all scores go through the frozen oracle `src/mfas/metrics.py`.*

---

## A. Who the fly's hubs are — CT1 and APL, the textbook giant local inhibitors

**Computed.** `exp_untapped_a_fly_hubs.py` → `untapped_A_fly_hubs.json`. Graph
index → FlyWire `root_id` via `GraphData.node_ids`, joined to
`data/external/flywire_neuron_annotations.tsv.gz`; degree = incident edge count;
feedback (FB) = edges with `rank[src] > rank[tgt]` under H64 (total FB weight
6,597,734 = 15.742 % of all weight).

**Top of the degree table** (full 12-row tables in the JSON):

| rank | idx | cell type | class | nt | side | degree | π-rank (%) | FB out | FB in |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 78384 | **CT1** | optic ME>LO | GABA | left | 16,784 | 43.2 | 14,180 | 28,321 |
| 2 | 89593 | **CT1** | optic ME>LO | GABA | right | 16,581 | 41.9 | 15,194 | 26,302 |
| 3–4 | 72989, 81640 | Li33 (mALC2) | optic LO | GABA | r/l | 8,711 / 8,483 | ~70 | ~9k | ~7k |
| 5–6 | 86212, 92640 | Am1 | optic LOP>ME.LO | GABA | r/l | 7,041 / 6,443 | ~70 | ~12k | ~1k |
| 8–9 | 17582, 65939 | **APL** | central MBIN | GABA | r/l | 5,606 / 5,579 | 74–77 | 34,585 / 36,283 | 13,337 / 19,086 |

11 of the top 12 degree hubs are GABAergic (the exception: MeMe_e13,
acetylcholine). **Top FB senders**: the two APLs (#1–2, together 1.07 % of ALL
backward weight), then PVLP011, the two CT1s, Am1, Li30, Li32, AVLP080, Sm42.
**Top FB receivers**: the two CT1s (#1–2), APL, PVLP011, Sm42, LPi12 — plus,
tellingly, *early-order* pure receivers: cL21 visual-centrifugal cells (π-rank
25–31 %, FB in ≈ 8k each, FB out ≈ 0) and lLN2F_b antennal-lobe local neurons
(π-rank 12 %).

**Reading (2–3 sentences).** The fly brain's biggest node is a local inhibitory
giant, and it sits mid-order: CT1 — the one-per-hemisphere giant GABAergic
amacrine cell that visits every column of the medulla and lobula and operates as
~1,400 quasi-independent subunits ([Meier & Borst 2019](https://www.cell.com/current-biology/fulltext/S0960-9822(19)30398-7)) —
is the #1 and #2 node by degree AND the #1 and #2 feedback receiver, parked at
41–43 % of the order where its massive reciprocal columnar wiring splits its
traffic. The #1 and #2 feedback *senders* are the two APLs, the mushroom body's
one-per-hemisphere giant GABAergic interneuron whose defining job is literally
feedback inhibition of Kenyon cells for sparse coding
([Lin et al. 2014; Amin et al. 2020](https://elifesciences.org/articles/56954)) — the optimizer,
knowing nothing of biology, re-derives "APL = the brain's largest feedback
device" from wiring alone. More broadly, essentially the whole hub layer of the
graph is GABAergic local giants, i.e. the highest-degree objects in the fly brain
are not relay stations but normalizers, and the order handles them by placing
them mid-stream (CT1) or late with backward-counted output (APL).

**Caveats.** Degree here counts graph edges (synapse-partner pairs), not synapses;
annotation is the community consensus table (one row per root_id, no version
pinned in-repo); "FB sender/receiver" is order-relative bookkeeping, not a causal
claim.

---

## B. Lane-matrix mining — where the fly's 15.7 % backward weight actually lives

**Computed.** `exp_untapped_b_lane_mining.py` → `untapped_B_lane_mining.json`,
figure `biology/outputs/untapped_B_fb_budget.png`. Semantics of
`layering/outputs/fly_streams/lane_matrix.json` verified: `W_ff_pct` sums to the
champion FF% to 1e-9; pair FF share = `W_ff_pct/W_pct`, pair FB = `W_pct−W_ff_pct`.
Thresholds: pair weight ≥ 0.1 % of total; counter-hierarchy < 65 % FF; one-way
≥ 97 % FF.

**Counter-hierarchy pairs (13)** — the pairs the optimal order largely refuses to
orient, sorted by FF share:

| src → tgt | FF % | weight (% of total) | share of ALL FB |
|---|---|---|---|
| visual projection → visual centrifugal | 24.7 | 0.22 | 1.1 % |
| visual projection → medulla | 37.8 | 0.50 | 2.0 % |
| central (other) → ascending (VNC) | 42.3 | 0.14 | 0.5 % |
| visual projection → medulla→lobula | 44.9 | 0.32 | 1.1 % |
| visual projection → lobula (plate) | 45.9 | 0.50 | 1.7 % |
| **lobula (plate) → medulla→lobula** | **46.0** | **1.64** | **5.6 %** |
| central (other) → visual projection | 52.6 | 0.65 | 1.9 % |
| medulla→lobula → lamina | 53.2 | 0.21 | 0.6 % |
| central (other) → visual centrifugal | 54.1 | 0.51 | 1.5 % |
| descending → central (other) | 57.2 | 0.60 | 1.6 % |
| **medulla → lamina** | **57.8** | **1.23** | **3.3 %** |
| lobula (plate) → medulla | 59.7 | 0.36 | 0.9 % |
| central (other) → olfaction: AL·LH | 62.0 | 0.31 | 0.8 % |

**One-way streets (12)** — lamina → medulla→lobula 99.7 %, photoreceptors →
lamina 99.6 %, AL·LH → mushroom body 99.5 %, VPN → descending 99.4 %, ascending →
descending 99.4 %, ascending → central 99.2 %, central → motor 98.9 %, visual
centrifugal → medulla 98.8 %, **medulla→lobula → VPN 98.7 % (4.60 % of all
weight — the heaviest one-way trunk; the heaviest inter-lane bundle of any kind
is medulla → medulla→lobula at 6.46 % / 84.8 % FF)**, sensory → descending 97.9 %,
lobula (plate) → VPN 97.8 %, central → descending 97.0 %.

**The feedback budget** (grouping the 17 lanes into visual / central /
periphery-motor families; figure):

| where | share of ALL backward weight |
|---|---|
| between visual lanes (retro-pathways) | **36.6 %** |
| within one central lane (recurrence) | 30.2 % |
| within one visual lane (recurrence) | 21.0 % |
| visual ↔ central | 4.9 % |
| involving periphery-motor | 4.2 % |
| between central lanes | 2.7 % |
| within / between periphery-motor lanes | 0.5 % |

The anatomically *descending* visual pairs alone (deeper stage → earlier stage:
Tm→medulla, LOP→Tm, medulla→lamina, VPN→everything upstream, …) carry **24.0 %
of all backward weight**.

**Reading.** Feedback in the fly is a visual-system phenomenon: 57.5 % of all
backward weight never leaves the optic lobes (36.6 % between visual stages +
21.0 % within a stage), and a quarter of the total is literally retro-projection
down the visual hierarchy — VPN axons reaching back into medulla/lobula, the
lobula-plate → Tm return, and the medulla → lamina centrifugal path, each of
which is a known anatomical feedback system, here weighted for the first time
against the whole brain's budget. The central brain contributes its 33 % almost
entirely as *within-lane recurrence* (loops inside a region), not as
between-region back-projections — between central lanes is only 2.7 %, and the
center↔periphery axes are nearly clean one-way streets (ascending in at 99.2 %,
descending out at 97.0 %). The one central counter-current that does exist is
the motor re-entry: descending → central at 57.2 % FF — copies of motor commands
flowing back into the brain, the wiring one would draw for efference copy.

**Caveats.** Lane pairs aggregate over the bio17 lane partition — a coarse,
display-level grouping (e.g. "central (other)" is huge); FF/FB per pair depends
on the node-level champion order, so alternative near-optimal orders shift pair
FF shares by a few points (QA_ORDER_AND_ROBUSTNESS.md); "retro-pathway" family
membership of `visual centrifugal` is a naming choice — it is counted in the
visual family (its FB contribution is small either way; JSON has per-pair rows
for any regrouping).

---

## C. Mouse id → region mapping — the gap stays open (honest close)

**What was sought.** A row → region-name mapping for `data/table_mouse.txt`
(148 nodes, ids 1..149, id 37 absent, 583 float-weighted edges), without which
the mouse findings — the hub node 92 funneling 40.5 % of all feedback, the
one-giant-SCC structure, the long-range feedback spans (RESEARCH_NOTES.md §1,
the de-facto findings M1–M3) — cannot be read biologically.

**Provenance trail (everything checked, in order).**

1. `data/README.md` — says only "researcher-provided mouse connectivity table".
2. `git log --follow data/table_mouse.txt` — added in the repo's *first* commit
   (`7a77547`, 2026-06-21, "main reproduce rocket algorithm") together with the
   FlyWire file; no source in any message that touches it.
3. `experiments/randomgraph.md` — no provenance (zero relevant mentions).
4. **Bader et al. 2025** (`s13278-025-01491-2 (2).pdf`, full text extracted) —
   the word "mouse" does not appear; the paper's only dataset is the FlyWire
   graph (confirmed against the [published article](https://link.springer.com/article/10.1007/s13278-025-01491-2)).
   The dataset did **not** come from the reproduced paper.
5. Vahidi 2025 (`2506.13799v1.pdf`) — likewise fly-only.
6. `university_documents/` (thesis proposal .md/.docx, RECUR_new.docx, Cosyne
   abstract, Hebrew progress notes) — mention FlyWire, MICrONS, planned
   C. elegans; never a 148-node mouse table.
7. `layering/RESEARCH_NOTES.md` already concluded (2026-09-06): *"Until the
   researcher supplies the id → region mapping, none of the drawings can be read
   by a neuroscientist, and the hub (id 92) cannot be interpreted."*
8. Web candidates, tested against the data fingerprint:
   - **Oh et al. 2014** (Allen mesoscale, the prompt's guess): 213 regions and a
     dense (~35–50 %) network — *refuted* by node count and by our density
     (583/21,756 = 2.7 %).
   - **Calabrese et al. 2015** (Duke CIVM dMRI atlas, 296 ROIs = "148 per
     hemisphere"): node-count coincidence only — tractography networks are dense
     and near-symmetric; ours is sparse and directionally asymmetric (only 166
     of 583 edges reciprocated, reciprocal weights unequal).
   - No public mouse connectome with 148/149 regions matching the fingerprint
     was found.

**Verdict: the mapping cannot be established with confidence, and a wrong
mapping is worse than none — node 92 stays unnamed.** The ids are clearly row
numbers of an ordered 149-entry region list (near-contiguous, one silent
dropout), i.e. the mapping exists on the researcher's side as a plain list.

**What would settle it** (any one suffices):
- the original file/notebook that produced `table_mouse.txt` (the 149-row region
  list, with whatever row 37 was);
- an email from the supervisor naming the parcellation;
- any candidate weight matrix: identification is then *mechanical*, because the
  file's exact floats are a fingerprint — heaviest edges 51→53
  `0.613959345844929`, 105→123 `0.594344938173106`, 133→135 `0.28193253167543`
  (all recorded in the JSON; floats survive copy-paste, names do not).

**What was computed anyway** (`exp_untapped_c_mouse_provenance.py` →
`untapped_C_mouse_fingerprint.json`): the full per-node feedback ledger under the
pinned H63 order, keyed by *original* id, so the biological re-read of M1–M3 is
one join away the day the list arrives. Highlights: hub **92** (in-deg 95,
out-deg 56 — it hears from two-thirds of all regions) sits at rank 55/148 and receives
**40.5 %** of all FB; the FB *senders* are a late-rank block **102 (12.7 %),
103 (12.2 %), 129 (8.5 %), 135 (5.6 %)** at ranks 119–128; secondary receivers
**104 (18.9 %)** and **146 (10.0 %)**. Consecutive-id clustering (102/103/104;
133/135) suggests the source list groups related structures — another hint the
list is an ordered anatomical parcellation.

---

## D. MICrONS proofread-axon control (§8.3) — run; the E/I asymmetry survives, the depth trend does not

**Computed.** `exp_untapped_d_proofread.py` → `untapped_D_proofread_control.json`.
The 245-node both-endpoints-proofread subgraph (m = 2,424, total weight 5,466)
ordered directly: greedy-FAS init + the champion stage-3 sift
(`sift_underrelaxed`, k_full = 6, α = 0.7, ≤ 40 sweeps) + 16 seeded
random-permutation restarts, best-by-frozen-oracle (0.3 s CPU). Compared against
the pinned H42 full-graph champion order *restricted* to the same 245 nodes.

| quantity | value |
|---|---|
| greedy init | 68.92 % |
| **direct optimization, best of 17 runs** | **75.58 %** (restarts 74.84–75.58) |
| full-graph champion restricted to the 245 | 67.75 % |
| full-graph champion (whole graph) | 83.24 % |
| Spearman ρ (direct vs restricted order) | **0.323** |

Biology inside the control (E n = 142, I n = 97, NA n = 6; rank statistics,
permutation / Mann-Whitney p-values, 20,000 perms):

| effect | full graph | direct proofread order | champion-restricted |
|---|---|---|---|
| depth trend, excitatory (ρ rank vs depth) | **+0.356** (p ≤ 5e-4) | **−0.168** (p = 0.045) | −0.052 (p = 0.54) |
| I before E (median rank) | every layer band, huge | 114 vs 128 (p = 0.21, n.s.) | 65 vs 150 (p < 1e-4) |
| I→E FF share | 97.3 % | **88.4 %** | — |
| E→I FF share | 48.4 % | **53.1 %** | — |
| E→E / I→I FF share | 80.6 / 78.1 % | 89.4 / 75.1 % | — |

**Reading.** The control first exposes *itself*: the proofread set is not a
miniature cortex — it is 40 % inhibitory (vs 10.7 % graph-wide), shallow (median
depth 247 µm vs 381), and **71.6 % of its weight is I→I** — the footprint of the
minnie65 interneuron-proofreading campaign. On such a sample the depth hierarchy
has no right to replicate, and it doesn't (it even flips sign, weakly); crucially
the *champion order restricted to the same nodes* also loses the trend (ρ −0.05,
n.s.), which says the loss is composition, not evidence that the full-graph
depth trend was a truncation artifact. What **does** survive at full
reconstruction quality is the directional E/I asymmetry — I→E stays
forward-aligned (88 %) while E→I stays the one half-backward pair (53 %) — i.e.
the "pay the loop on the light side" mechanism of MICRONS_FINDINGS §2–3
replicates inside a subgraph with *no* axon-truncation bias at all. The
I-before-E *median* lead shrinks to a non-significant 14 ranks, consistent with
that effect being partly powered by the full graph's inflated I→E weight share
(33.9 % there, 12.9 % here).

**Caveats (small-n honesty).** n = 245 and non-random (proofreading targeted
interneurons); 60 SCCs with a giant of 186, so ~24 % of nodes are order-flexible
periphery; the direct optimum 75.58 % is a lower bound (no exact solver run);
all claims are direction-of-effect, not magnitudes.

---

## E. Quick adds

**E1 — within-class recurrence (MICrONS).** From the gallery's
`ei_pair_budget` (never interpreted): **E→E is 80.6 % FF and I→I 78.1 % FF** —
both near the graph-wide 83.2 % (−2.6 / −5.1 pp, small next to the ±49 pp
cross-class split), while the cross-class pairs split to
the extremes (I→E 97.3 %, E→I 48.4 %). The E/I story is therefore entirely a
*cross-class* phenomenon: the optimizer does not find excitatory chains more
one-directional than inhibitory ones; what it finds is that the E↔I loop has a
heavy side and a light side. The proofread control (item D) keeps the essential
pattern — E→I is the lone half-backward pair (53.1 %) with I→E strongly forward
(88.4 %), and E→E > I→I within-class (89.4 / 75.1 %) — though not the full
graph's exact ranking: I→E, #1 there, is edged out by E→E in the control.
(`untapped_E_quick_adds.json`, source
`layering/outputs/microns_gallery/report_gallery.json`.)

**E2 — the "unclassified" lane is reconstruction debris, not interneuron
arithmetic.** The 4,202 unclassified-E/I neurons sort early (median rank 18.1k
of 67.5k) — but not for the interneuron reason: their aggregate out/in weight
ratio is **1.18**, nowhere near the inhibitory 2.70. Decomposition (computed
fresh): **42.8 % are multi-soma merge bodies** (median rank 29.1k — ordinary),
**473 (11.3 %) are zero-input orphan-output objects sitting at median rank 309 —
the extreme front of the order**, and the remaining 1,930 single-soma cells are
input-starved (median in-weight 28 vs 168 population-wide, median rank 13.0k).
So the lane's earliness is *missing inputs* — the exact mirror of the
zero-output fragments that pile at the back (MICRONS_FINDINGS §6). Both ends of
the order are quality-control zones: the optimizer sweeps output-less debris to
the tail and input-less debris to the head, and abstains (via the E/I model's
silence on merge monsters) in between.

**E3 — the trophic-order gap.** Trophic levels (MacKay-Johnson-Jones, one
Laplacian solve) reach **77.78 %** FF on the fly vs the champion's **84.26 %**
(−6.48 pp, Spearman ρ = 0.76); on the mouse map **89.98 %** vs **93.18 %**
(−3.20 pp, ρ = 0.74). Two honest conclusions. First, optimal order ≠ trophic
depth: on the fly the shortfall equals **41 % of the champion's entire 15.74 %
feedback budget** — a hierarchy inferred by the standard continuous method
would report a **net** ~2.7 M extra synapses as feedback (gross: it points
3.8 M champion-forward synapses backward while 1.1 M flip the other way), so
"how much feedback does a brain have" depends materially on the instrument. Second, the
gap *doubles* on the fly relative to the tiny mouse map, which is exactly the
graph whose feedback is concentrated in stage-recurrence and retro-pathways
(item B): trophic averaging pays most where loops are local and heavy, and an
optimizing order can lean on its discrete freedom precisely there. (Numbers from
the logged artifacts `layering/outputs/gallery_fly/report.json` and
`layering/outputs/diagnostics_mouse.txt`.)

---

*Scripts: `biology/analysis/exp_untapped_{a_fly_hubs,b_lane_mining,c_mouse_provenance,d_proofread,e_quick_adds}.py`.
Results: `biology/results/untapped/untapped_{A,B,C,D,E}_*.json`. Figure:
`biology/outputs/untapped_B_fb_budget.png`. External sources: Meier & Borst 2019
Curr Biol (CT1); Lin et al. 2014 Nat Neurosci / Amin et al. 2020 eLife (APL);
Bader et al. 2025 SNAM 15:68 (fly-only benchmark).*

---

## Independent review (2026-09-30)

*Reviewer: independent pass with fresh aggregation code —
`biology/analysis/review_untapped.py` → `biology/results/review/review_untapped.json`
(48 checks, 0 fail). Nothing was taken from the study's own result JSONs: hubs
re-joined from the graph + annotation TSV, the 17×17 lane matrix re-aggregated
from the edge list, the proofread control re-optimized with a from-scratch
sifting optimizer, trophic levels re-solved (own CG / lstsq), the mouse file
re-parsed raw, git provenance re-queried. All scores through the frozen oracle;
orders through the parity-gated champion loader.*

### Verdicts

| item | verdict | key re-computed evidence |
|---|---|---|
| §A hubs | **confirmed** (1 prose-order slip fixed) | degree = in+out edge count, stated in study. #1–2 degree AND #1–2 FB receivers = CT1 (deg 16,784/16,581, ranks 43.2/41.9 %, GABA, ME>LO); 11/12 top-degree hubs GABAergic (exception MeMe_e13, ACh); #1–2 FB senders = APL at 74.2/76.6 %, together 1.074 % of ALL FB (denominator = total FB weight 6,597,734 = 15.7418 %, verified); all per-row spot values exact; cL21 (25–31 %, FB-in≫out) and lLN2F_b (12 %) present among top receivers. |
| §B budget | **confirmed** (1 superlative corrected) | Lane matrix re-aggregated from edges reproduces `lane_matrix.json` to 0 ulp and re-sums to the champion 84.2581795094 % exactly. Budget families (own code): 36.56 / 30.15 / 20.95 / 4.92 / 4.19 / 2.70 / 0.53 — all match; never-leaves-optic = 36.56+20.95 = **57.51 %** (the rounded components 36.6+21.0 = 57.6 are why the sum looks off by 0.1); central total 32.85 %; 13 counter-hierarchy + 12 one-way pairs under the stated thresholds; named pairs VPN→VC 24.68 / LOP→Tm 46.02 @1.64 / medulla→lamina 57.84 @1.23 / descending→central 57.23 / PR→lamina 99.60 / AL→MB 99.46 / Tm→VPN 98.75 @4.60; retro-named 23.97 %. No double counting: the 8 categories partition all 17×17 pairs and sum to 100.00 %. Sensitivity: regrouping visual-centrifugal into the central family moves between-visual 36.6→34.0 % and never-leaves-optic 57.5→54.8 % — the study's "small either way" caveat is honest. |
| §C mouse closure | **confirmed — "honestly closed" is the right verdict** (2 small fixes) | Raw-file re-parse: 148 nodes, ids 1..149 missing exactly {37}, 583 edges, density 2.68 %, 166 reciprocated, heaviest 51→53 = `0.613959345844929`, 105→123 = `0.594344938173106`, 133→135 = `0.28193253167543` — fingerprint exact. `data/README.md` wording and first-commit provenance (7a77547, 2026-06-21) verified; Bader PDF: "mouse" absent from every decompressible text stream (weak-negative check; no PDF extractor in env). Hub-92 ledger re-derived: in-deg 95 / out-deg 56, rank 55, receives 40.50 % of FB, sends 0; sender block 102/103/129/135 = 12.68/12.18/8.51/5.57 % at ranks 119–128. Bonus: the C2 claim "all FB edges have an FF return path" re-tested — **124/124 confirmed**. Refusing to guess a mapping while precomputing the join is the correct scientific posture. |
| §D proofread control | **confirmed; reading licensed** (see red-team) | Deterministic legs exact: champion-restricted 67.7461 %, greedy 68.917 %, study pipeline reproduces 75.5763 % / Spearman 0.3231; composition 142E/97I/6NA (39.6 % I), 71.6 % weight I→I, depth 247 vs 381 µm; 60 SCCs, giant 186; 249 proofread axons graph-wide. Stats under the reproduced order match to quoted precision (ρ_E −0.168 p≈0.046; I 114 vs E 127.5, p 0.21; I→E 88.4 / E→I 53.1). Champion-restricted: ρ_E −0.052, p 0.54; I 65 vs E 150.5. |
| §E1 | **confirmed with corrections** | Full-graph pair FF re-computed from the edge list: E→E 80.63 / I→I 78.09 / I→E 97.30 / E→I 48.41. Two phrasing errors fixed: (i) "reproduces the pattern ordering (E→E most forward…)" was wrong — in the FULL graph the most-forward pair is I→E (97.3), and in the control E→I (53.1), not I→I, is the actual minimum; the control preserves the cross-class split and the within-class ordering, not the exact 4-pair ranking (I→E and E→E swap at the top). (ii) I→I at 78.1 is 5.1 pp below the graph-wide 83.2 — "near", not "essentially at". |
| §E2 | **confirmed** | Own aggregation: n = 4,202, median rank 18,076.5; out/in 1.18 vs inhibitory 2.70; merge bodies 42.8 % at 29,105; 473 zero-input orphans at 309; 1,930 remaining with median in-weight 28 vs 168 population-wide at 12,967. All exact. |
| §E3 | **confirmed with one precision fix** | Trophic levels re-solved from scratch: fly 77.7774 % (ρ 0.7602), mouse 89.977 % (ρ 0.736) — gaps 6.481 / 3.20 pp reproduced. Arithmetic: 6.4808 / 15.7418 = **41.2 %** — correct, denominator = the champion's own FB budget. But the "~2.7 M synapses misclassified" is a **NET** figure: gross, the trophic order points **3.82 M** champion-forward synapses backward while **1.10 M** flip the other way (net 2.72 M). Text now says net and quotes the gross split. |

### Red-team notes

1. **§D — is "composition, not truncation-evidence" licensed? Yes, on three
   legs.** (i) The champion order — which carries the depth trend globally at
   ρ = +0.356 — loses it when restricted to the same 245 nodes (ρ_E −0.05,
   n.s.), so this sample cannot express the trend under *any* order.
   (ii) The control's E cells are strongly depth-range-restricted (IQR 168–366 µm
   vs 254–577 µm for full-graph E; median 215 vs 379), which mechanically
   attenuates any rank–depth correlation. (iii) The review's own independent
   optimizer (from-scratch sift + segment-reversal kicks) found a **better**
   direct order, **75.80 %** vs the study's 75.58 % (the study correctly labels
   its number a lower bound) — and under that better order the nominally
   significant sign flip vanishes (ρ_E −0.03, p 0.75) and the I-before-E medians
   actually reverse (I 137 vs E 111.5, n.s.), while I→E ≫ E→I persists
   (89.5 vs 51.7). So: the depth flip and the 14-rank I-lead are order-noise at
   fixed score, exactly as the study treats them (n.s. / "weakly"), and the one
   claim promoted to FINDINGS — the cross-class E/I asymmetry — is stable across
   optima. Do not ever quote −0.168 (p 0.045) as a real reversal.
2. **Denominators** are used consistently: §A/§B "share of ALL backward weight"
   always divides by 6,597,734 (15.7418 % of total); pair FF shares divide by
   pair weight; §E3's 41 % divides by the champion FB budget (now stated). The
   only denominator trap found was §E3's net-vs-gross (fixed).
3. **"Single biggest trunk in the brain" was false** for Tm→VPN (4.60 %):
   medulla→ME>LO (6.46 %), ME>LO→LOP (5.04 %) and ME>LO→medulla (4.76 %) are
   heavier inter-lane bundles. It is the heaviest *one-way* (≥97 % FF) trunk.
   Corrected in §B.
4. **Small fixes applied to §A/§C:** FB-sender prose order (Li30, Li32, AVLP080
   — not Li30, AVLP080, Li32); mouse sender-block ranks 119–128 (not 119–133;
   133 is id 53, the 5th sender); "hears from nearly every region" softened to
   "two-thirds of all regions" (95 of 147).
5. **Unverifiable-locally, flagged not fixed:** Oh 2014's "213 regions, ~35–50 %
   density" and Calabrese 2015's parameters are literature claims (the local
   refutation — node count + density mismatch — holds regardless); the Bader
   PDF fly-only check is weak-negative (stream decompression, not proper text
   extraction). §D p-values were re-checked analytically (they match the study's
   permutation values to 3 decimals); E median 127.5 is printed as 128 in the
   study table (rounding, noted).

*Status: → `reviewed with corrections`. Corrections applied above in §A (sender
order), §B (one-way trunk superlative), §C (ranks 119–128, hub in-degree
phrasing), §E1 (pattern-ordering sentence, "essentially at" → "near"), §E3
(net vs gross, 41 % denominator). No headline conclusion of the study was
overturned; every number promoted to FINDINGS.md survived independent
recomputation.*
