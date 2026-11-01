# Independent review — null-ladder v2, Stage F (the two deliverables)

**Reviewer:** independent subagent, 2026-11-01 · **scope:** (1) `REPORT.md` (this folder),
(2) the Russian supervisor walkthrough page (scratchpad
`null_ladder_walkthrough.html`, the published artifact's source). Experiments are NOT re-run;
the four prior verdicts (`review_stageA.md`, `review_stageB_w0.md`, `review_stageC_l2r.md`,
`review_stageD_l4.md`, all PASS) are taken as ground. Every number in both deliverables was
re-traced to `biology/results/nullmodels_v2/summary_v2.json`, the run JSONs (incl. the three
`*__e20000.json` probes and `biology/results/nullmodels/runs/connectome__real__ps42__e20000.json`),
`bounds_cache.json`, or `biology/studies/nullmodels.md`; all derived columns recomputed from
the raw JSONs with reviewer-own arithmetic (json+numpy only).

---

## A. REPORT.md — number trace and derived-column recomputation

Recomputed from `summary_v2.json`: excess FB over own floor = `(100 − ff_mean) − bound_mean`:

| rung | report | recomputed | verdict |
|---|---|---|---|
| L1 | 30.17 | 30.1699 | CONFIRMED |
| L3a | 23.97 | 23.9713 | CONFIRMED |
| w_shuffled | 22.68 | 22.6747 | CONFIRMED |
| w_source | 19.91 | 19.9097 | CONFIRMED |
| L3 dc-SBM | 17.73 | 17.7255 | CONFIRMED |
| L2R | 15.70 | 15.7021 | CONFIRMED |
| W0 | 13.32 | 13.3191 | CONFIRMED |
| L4 | 13.75 | 13.7502 | CONFIRMED |
| real | 9.62 | 9.6246 | CONFIRMED |

All FF means/sds, K values, floors, n_beyond (0 everywhere, 12/12 at L4), greedy values
(70.2485 / 70.2963 / 69.4618 vs real 68.9134), z-values (+44.15, +520.3, −66.9; combined −62.3),
real 83.658207 ± 0.003347 (K=3), champion 84.25817951, real floor 6.7171539 — all match
`summary_v2.json` exactly. §3 G01 rows (+19.8/+10.2/+1.9/+1.8) trace to `nullmodels.md` §3
(confirmed by stage-A review). Header "39 runs" = 36 + 3 probes ✓ (runs dir holds exactly 39).
Figure `biology/outputs/nullmodels2_fly_ladder.png` exists.

**Probe gaps recomputed** (vs `connectome__real__ps42__e20000.json`, pct_final 84.102917):

| rung | pinned@20k | gap | report says | verdict |
|---|---|---|---|---|
| W0 gs6000 | 82.848357 | **+1.2546** | "+1.26 pp" | **DOES NOT REPRODUCE — must read +1.25** (double-rounding: 84.103 − 82.848 was rounded first) |
| L2R gs7000 | 77.796159 | +6.3068 | +6.31 | CONFIRMED |
| L4 gs8000 | 84.756250 | −0.6533 | −0.653; measured 84.7562 | CONFIRMED |

Supporting probe claims confirmed: 82.277 → 82.848 (gs6000 4k = 82.276723); −0.583 at 4k
(84.241306 − 83.658207 = 0.5831); 20k null over champion +0.50 (84.7562 − 84.2582 = 0.4980);
§4.1 "movements ≤ 0.23 pp, no sign change" (0.18 / 0.23 / 0.08); §4.3 "0.57 pp ≥ 17× any draw
sd" (0.5736/0.0325 = 17.6); §4.4 "5.7× the 0.1 pp trigger" (0.574/0.1).

**The single non-reproducing number in REPORT.md is "+1.26 pp" (§2.1).**

## B. Required-wording items from the four verdicts — presence in REPORT.md

| mandate | present? |
|---|---|
| B §F1: 9.62 not 9.63 | YES (9.62 everywhere; "9.63" absent by grep) |
| B §F2: protocol-conditional + greedy flip beside the W0 headline | YES (§2.1) |
| B §F3: "estimator-visible" bound-gap wording on every excess-FB use | YES (§1 definition + 3 uses; never "less feedback") |
| B §F4: W0 coupling non-attribution | YES (§2.1 "which coupling … is not identified at this rung") |
| B §F5: balanced/bidirectional concentration | YES (§2.1 "balanced bidirectional concentration") |
| C §I1: w_source bounds logged as artifacts | YES — `bounds_cache.json` exists, committed (705094f), 76 entries covering ALL 20 w_source gen seeds 2000–2019 (+20 w_shuffled, 12 L1, 12 L3a, 12 L3); spot value gs2000 = 0.10788521 matches both summary_v2.json and the stage-C regeneration |
| C §I2: both halves of the (b) adjudication | YES (§2.2 literal FF-axis (b) false + registered sign-check + excess-FB ordering 19.91 > 15.70 > 9.62; timing disclosure carried in §1 "adopted in the stage-A review, registered before any L2R/L4 run") |
| C §I3: no "pure"; L2R protocol-conditional + greedy flip beside headline | "pure" ABSENT ✓; residual correctly scoped (§2.2). **PARTIAL:** "protocol-conditional" and the greedy flip are not in §2.2 itself (greedy flip only in §4.1; §4.2 claims "kept everywhere") — fix 2 below |
| C §I4: backbone phrasing "(with the constraints needed to keep it exact)" | YES (§2.2) |
| C §I5: v1-vs-v2 ensemble sentence (incidental reciprocity ≈0.108%, no-new-mutual exact, 0.287% repaired, "approximately uniform") | YES (§2.2 Ensemble note) |
| D §I1: kill-table adjudication, measured 84.7562 vs pre-stated 84.1029 / 84.2029 / 84.396 | YES (§2.3, thresholds quoted, KEEP-with-stability) |
| D §H5.1: "pins all and overshoots" | YES (§2.3 heading + body; §3) |
| D §H5.2: out-strength only (no in-strength claim) | YES (§2.3; §2.2 adds "uni in-strengths are not preserved") |
| D §H5.3: mouse analogy as co-occurrence + scale disclosure | YES (§2.3 "co-occurrence, not an established shared mechanism"; "mouse meso … strength-matched") |
| D §I3: non-nested L4-vs-L3 deviation (91 types / 43,580 / 31.89% / 0.60% of m / 0.39% of W) | YES (§2.3 honesty ii) |
| D §I2: imported-prediction-branch honesty | PARTIAL — the import statement is there (§2.3 honesty i), but the mandated **verbatim charter quote and the registration timeline (L4 section added in c77c040, 14 min before the first run)** are absent — fix 3 below |
| D §I4: fallback caveat numbers | PARTIAL — 0.91% of nodes / 0.40% of W present; the mandated **0.13% of ensemble freedom and worst-case ≤ 0.40 pp** are deferred to the review — fix 4 below |
| D §H5.5: champion proximity labeled cross-budget coincidence | YES (§2.3 "+ §4.5") |
| D §I7: K = 8,857; 99.09% direct coverage; no "8,840"/"98.9" echo | YES ("8,840" and "98.9" absent by grep) |
| D §B note: `gen_stats.variant` mislabel ("L3_dcsbm"), filter by `kind` | YES (§2.3 honesty v) |

## C. The Russian walkthrough page — number trace and calibration vs REPORT.md

Every number checked against `summary_v2.json` / the probe JSONs / stage reviews. CONFIRMED:
84,26 champion; 136 648 / 5,66 млн; 689 G01 runs; 36 + 3 v2 runs; real 83,66 / 83,658 ± 0,003
(K=3) / floor 6,72; rung cards 69,82 (floor 0,007) / 77,22 & 79,98 (0,10–0,11) / 81,90 (0,38) /
77,58 (floor = real bit-exact; 18% связей = 18.38% of m, 25% веса = 24.86% of W) / 82,22 (4,46)
/ 84,23 (2,02; 8 857 types; 99,1% coverage); decomposition +19,8/+10,2/+1,9/+1,8/+1,44/+6,08/
"всё + 0,57"; find-2 19,9 → 15,7 (мозг 9,6), ~40%; find-3 84,23 ± 0,01 vs 83,66, below all 12,
62σ, 4,7 pp freed floor, 0,57 converted, 4,1 returned (real still wins on that axis), втрое
(3.33×) backbone; probe thresholds 84,10 / 84,40, measured 84,76, "обгоняет чемпионский"
(+0.498); honesty 91 types / 31,9% / +2,34 non-nested; JS `rungs` array **strictly ascending
by FF** (69.823 → 84.232) ✓; real row ff 83.658 with champion 84,258 in the note, chart lines
labeled "мозг 83,66" / "чемпион 84,26" ✓.

Calibration failures (Russian stronger than the English record) and number errors:

1. **(number) Find-1 probe "разрыв остаётся +1,26 pp"** → must be **+1,25** (same error as
   REPORT §2.1; exact +1.2546).
2. **(overclaim) Find 1:** "Сильные синапсы в мозге выровнены вдоль иерархии — столько же
   порядка, сколько даёт вся региональная схема." This asserts the specific order-alignment
   coupling that stage-B §E2/§F4 explicitly barred ("which coupling carries it is not
   identified"), and "столько же" misstates 1,44 vs the regional +1,9 (also a different,
   non-additive axis — the page's own table says "отдельная ось"). The sentence directly
   contradicts the caveat that follows it.
3. **(banned word) Find 2:** "эти 6 пунктов — **чистый** выбор односторонних адресатов…" —
   "чистый" = the "pure" stage-C §I3 ordered dropped; the residual's scope (regional/type
   organization + weight–target coupling bundled; L4 is the decomposition) is also missing.
4. **(number) Honesty, seeds bullet:** "σ сида 0,003 pp — в 10+ раз меньше межграфового
   разброса" — only true for W0 (0.0325/0.0033 ≈ 10×); L2R is 3.5×, L4 is 2.6×. Must read
   "в 2,6–10 раз".
5. **(minor accuracy) Honesty, greedy bullet:** "на всех степенных ступенях" — the greedy flip
   also occurs on W0 (real topology, not a degree rung), L2R and L4; say "на всех
   структурированных нулях, включая три новые ступени".
6. **(minor) "4/4 выборочных ре-скора каждой ступени"** → "каждой новой ступени" (the 4/4
   oracle re-scores are a v2-stage fact; G01's review had its own, differently-counted checks).

Everything else carries REPORT.md's calibration faithfully: protocol-conditionality is stated
as a standing caveat with the ×5 probes as the control; the L4 inversion is given both axes
with the brain winning on excess-FB; the imported-branch honesty, the non-nested L4−L3, and
the mouse co-occurrence caveat are all present in the "Оговорки" section; the bit-exact L2R
floor and "никакая арифметика полов не растворит" match stage-C's framing.

## D. Consistency

* **CHARTER.md §3 stages table vs reality: CONSISTENT.** A/B/C/D done with the right verdicts
  (478 = 398 + 80 checks ✓); E "done — all three directions survive" matches the three logged
  e20000 probes; F "final review in progress" is this review. Cosmetic only: the charter's
  section numbering skips §4 (…§3 → §5); its L4 section still says "8,840"/"98.9%" — correctly
  left as-registered, corrected wherever echoed in REPORT.md.
* **`experiments/randomgraph.md` G03 still says "in-progress (opened 2026-11-01)"** — stale.
  Proposed replacement (do not mark fully answered: the MICrONS arm is parked):

```
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
- **finding:** → Findings § below; FINDINGS.md entry proposed in `review_stageF_report.md` §F.
```

  Optional matching bullet for the "Findings — Track C" list:

```
* **G03 (2026-11-01, reviewed):** below cell-type resolution the fly's 84% is NOT generic —
  real beats every matched ensemble including real-topology/permuted-weights (+1.44 pp) and
  real-reciprocal-backbone + degrees (+6.08 pp on an identical, bit-exact 2-cycle floor). At
  type resolution the attribution closes and INVERTS: the K=8,857 type-dcSBM ensemble overshoots
  the real graph by 0.57 pp (real below all 12 draws, probe-confirmed at 5× budget), while the
  real graph keeps a 3.3× larger irreducible reciprocal core and still wins by 4.13 pp on
  excess FB over own floor. Protocol-conditional throughout. Study + four reviews:
  `biology/studies/null_ladder_v2/`.
```

## E. Text fixes required before the deliverables are final

**REPORT.md (4 required, 3 recommended):**

1. **(required, number)** §2.1: "keeps the gap at **+1.26 pp**" → "**+1.25 pp**"
   (84.102917 − 82.848357 = 1.2546).
2. **(required, stage-C §I3)** §2.2, first sentence: after "real above all draws by **+6.08 pp**"
   insert "(protocol-conditional; the greedy bound again flips, L2R 70.30 vs real 68.91 — §4.1)".
3. **(required, stage-D §I2)** §2.3 honesty item (i): append — *"The charter's two L4 branches,
   verbatim: 'FF(L4) close to real (the type diagram pins most of the order); the residual
   real − L4 bounds individual-neuron (sub-type) targeting' and 'if instead L4 ≈ L3 … must be
   reported as such.' The L4 section itself is stage-B vintage: added in commit c77c040,
   14 minutes before the first L4 run (after the selftest, which produces no FF score) —
   timeline verified in `review_stageD_l4.md` §G."*
4. **(required, stage-D §I4)** §2.3, fallback parenthesis: extend to "… 0.91% of nodes, 0.40%
   of W, **0.13% of the ensemble's permutation freedom; adversarial worst case ≤ 0.40 pp,
   below the 0.574 pp inversion** — quantified in the stage-D review".
5. (recommended) §1: "strictly monotone" → "monotone": on the excess-FB axis the two final
   rungs land out of order (W0 13.32 < L4 13.75); the "13.3–13.8" band already concedes this —
   say it explicitly ("…with W0 and L4 in a 13.3–13.8 pp band whose internal order reverses").
6. (recommended) §2.3 honesty (iii): "empirical p = 1/13" → "empirical one-sided p ≥ 1/13"
   (the reviews' mandated form).
7. (recommended, stage-B §E note) header or §1: one-line era note — "real comparators are
   Sept-2026 G01 runs; v2 nulls Nov-2026, same machine class, same torch 2.8.0."

**Russian page (4 required, 2 minor):**

1. **(required, number)** Find-1 probe line: "+1,26 pp" → "+1,25 pp".
2. **(required, overclaim)** Find 1: delete "Сильные синапсы в мозге выровнены вдоль иерархии —
   столько же порядка, сколько даёт вся региональная схема." Replace with: "Перестановка
   разрушает сразу все связи весов с топологией; вклад сравним по величине с региональной
   схемой (+1,9 pp), но это отдельная ось, и какая именно связь весов его несёт, эта ступень
   не различает." (Then drop the now-duplicated "Какая именно связь…" sentence.)
3. **(required, banned word)** Find 2: "эти 6 пунктов — чистый выбор односторонних адресатов
   сверх степеней и взаимного каркаса" → "эти 6 пунктов — порядок в выборе односторонних
   адресатов сверх степеней и взаимного каркаса (сюда входят региональная и типовая
   организация и связь весов с адресатом; дальше эта ступень не разлагает — разложение даёт
   L4)".
4. **(required, number)** Seeds bullet: "в 10+ раз меньше межграфового разброса" → "в 2,6–10
   раз меньше межграфового разброса".
5. (minor) Greedy bullet: "на всех степенных ступенях" → "на всех структурированных нулях,
   включая все три новые ступени".
6. (minor) "4/4 выборочных ре-скора каждой ступени" → "каждой новой ступени".

## F. Proposed `biology/FINDINGS.md` entry (Tier A; verbatim draft)

Placement: **Tier A**, new entry **A7** (sibling of A1, which it extends and partially
inverts; cross-reference both ways).

```
### A7. The cell-type wiring diagram is sufficient for the fly's entire feedforward
order — the type-matched ensemble overshoots the real brain, which spends its sub-type
wiring on reciprocity instead. — status: reviewed (studies/null_ladder_v2/REPORT.md,
2026-11-01; charter-pre-registered, stages A–D independently reviewed, L4 probe cleared
its pre-stated KEEP threshold)
Null-ladder v2 (G03) closes the attribution A1 left open, on the fly only (operator-parked
elsewhere). With the full real wiring fixed and only synaptic weights permuted (W0), the
identical epoch-budgeted protocol loses 1.44 pp (real outside all 12 draws; +1.25 pp at the
5× budget probe) — weight placement carries estimator-visible order, though which
weight–topology coupling carries it is not identified at that rung. Fixing the real
reciprocal backbone plus exact degrees (L2R — the one null whose 2-cycle floor equals the
real 6.72 % of W bit-exactly, so no floor-netting can dissolve the contrast) leaves the real
graph +6.08 pp above all 12 draws (+6.31 at 5×): order in the choice of non-reciprocal
targets beyond degrees + backbone. But a degree-corrected SBM at cell-type resolution (L4,
K = 8,857 FlyWire types, 99.09 % direct coverage) INVERTS the ladder: FF(L4) = 84.23 ± 0.01
vs real 83.66 — the real brain below all 12 draws (62σ; empirical one-sided p ≥ 1/13),
probe-confirmed at 5× budget (84.756 against the pre-stated KEEP ≥ 84.2029 / stability
≥ 84.396; the inversion grows to −0.65 pp). Two-axis reading: the type ensemble frees
4.70 pp of forced reciprocal floor (2.02 vs 6.72 % of W) yet converts only 0.57 pp into
feedforward weight, and on the registered excess-FB-over-own-floor axis the real graph still
beats every ensemble including L4 (9.62 vs 13.75 pp) — relative to its own type-level
diagram, the fly's sub-type wiring is net feedback-spending, buying a 3.3× larger
irreducible reciprocal core at the price of feedforward orderability.
Caveats: every direction is protocol-conditional (achieved lower bounds under an identical
epoch budget; the greedy control flips on every structured null and is
direction-uninformative; 2-cycle upper bounds do not separate the optima); neither
registered L4 branch anticipated the inversion — the reading imports the FF(null) > FF(real)
branch registered, before any L4 run, for the sibling rungs W0/L2R; L4 is non-nested vs L3
(91 direct cell types, 31.9 % of neurons, span the 17 lanes), so its +2.34 pp over L3 is a
non-nested comparison; the resemblance to the mouse meso inversion (A1) is a co-occurrence,
not an established shared mechanism — with the fly's backbone held fixed (L2R) the real
graph sits far ABOVE its null; fly-only by charter (mouse/MICrONS arms parked, scale
confound open); W0's 1.44 pp is a separate axis, not an additive ladder increment.
Provenance: biology/studies/null_ladder_v2/ (charter, REPORT, 5 reviews);
biology/results/nullmodels_v2/ (39 runs + bounds_cache).
```

**Does anything block promotion?** No. The three gates the stage reviews imposed are all
cleared: (i) stage-E probes logged for every rung, all three directions survive, and the L4
probe (84.7562) clears the pre-stated KEEP-with-stability threshold (≥ 84.396); (ii) the
w_source bounds are now logged artifacts (`bounds_cache.json`, committed, all 20 seeds);
(iii) every stage has an independent PASS. Condition: apply REPORT fix 1 (+1.26 → +1.25)
before or together with the promotion so the FINDINGS entry inherits the corrected probe
number, and apply fixes 2–4 so REPORT.md fully carries the mandated wording it is promoted
on. Charter hard-invariant 2 ("both datasets") is deviated from by explicit, documented
operator decision (CHARTER §0/§6) and the entry states the fly-only scope.

## G. Verdicts

| deliverable | verdict |
|---|---|
| `REPORT.md` | **PASS with corrections** — every statistic traces and recomputes exactly except one (+1.26 → +1.25); 4 required text fixes (§E), none touching a run-produced number |
| Russian walkthrough page | **PASS with corrections** — all chart data, rung numbers, decomposition, probe thresholds and honesty section trace exactly; 4 required fixes (one number ×2 deliverables, one overclaimed sentence, one banned "чистый", one seed-ratio claim) |

Neither correction set touches a logged quantity: the single non-reproducing number in each
deliverable is the same derived rounding (+1.26 for +1.2546), and all remaining fixes are the
prior verdicts' mandated wording.
