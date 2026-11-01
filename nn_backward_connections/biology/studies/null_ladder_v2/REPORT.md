# Null-ladder v2 — report

**Status: complete; every stage independently reviewed (A: PASS-with-corrections, B/C/D: PASS).**
Date: 2026-11-01 · machine: MacBook (Apple Silicon, MPS) · env `allen` (torch 2.8.0, py 3.9.23) ·
charter + pre-registration: `CHARTER.md` (this folder) · reviews: `review_stage*.md` (this folder) ·
runs: `biology/results/nullmodels_v2/runs/*.json` (39: 3×12 rungs + 3 budget probes) ·
summary: `biology/results/nullmodels_v2/summary_v2.json` · v1 comparators (reused, never re-run):
`biology/results/nullmodels/runs/*.json` (G01, `biology/studies/nullmodels.md`).

## 0. The question (operator + supervisor, 2026-11-01)

Is 84.26% feedforward a lot? Property of any graph with these parameters, or structure? The
original 3-step ladder proposed for this question had already been run as **G01**; this study
accepts G01 as the base (critically, per §4) and adds the three rungs it did not have. Identical
estimator everywhere (greedy-FAS warm start → Rocket-ASYM 4,000 epochs → under-relaxed sift,
8 sweeps; every discrete score from the frozen oracle on CPU); real-graph comparators reused from
G01's logged runs.

## 1. The extended ladder (fly connectome; mean ± sd over K draws)

| rung | matched exactly | K | FF% (protocol) | beyond | 2-cycle floor %W | excess FB over own floor |
|---|---|---|---|---|---|---|
| L1 G(n,m) | density | 12 | 69.823 ± 0.019 | 0/12 | 0.0072 | 30.17 |
| L3a SBM | 17-lane block counts, no degrees | 12 | 75.992 ± 0.025 | 0/12 | 0.0365 | 23.97 |
| L2 config `w_shuffled` | in/out degrees | 20 | 77.220 ± 0.024 | 0/20 | 0.1049 | 22.68 |
| L2 config `w_source` | + out-strengths | 20 | 79.981 ± 0.015 | 0/20 | 0.1096 | 19.91 |
| L3 dc-SBM | + 17-lane profiles | 12 | 81.896 ± 0.018 | 0/12 | 0.3783 | 17.73 |
| **L2R** (v2) | degrees + **real reciprocal backbone** | 12 | **77.581 ± 0.012** | 0/12 | **6.717154 (= real, bit-exact)** | 15.70 |
| **W0** (v2) | **real binary topology**, weights permuted | 12 | **82.221 ± 0.033** | 0/12 | 4.4596 ± 0.0028 | 13.32 |
| **L4** (v2) | type-resolved profiles (K = 8,857 cell types) | 12 | **84.232 ± 0.009** | **12/12** | 2.0181 ± 0.0062 | 13.75 |
| **real graph** | — | 3 seeds | **83.658 ± 0.003** | — | 6.7172 | **9.62** |
| real (champion budget, H64) | — | 1 | 84.258 | — | 6.7172 | — |

"Beyond" = draws at or past the real value. "Excess FB over own floor" =
`(100 − FF%) − own 2-cycle floor` — the **estimator-visible** feedback not already forced by
reciprocity (a gap between an FF upper bound and a floor lower bound, not measured feedback);
adopted as the cross-rung statistic in the stage-A review, registered before any L2R/L4 run.
Figure: `biology/outputs/nullmodels2_fly_ladder.png`.

**On the excess-FB axis the ladder is strictly monotone toward the real graph** (30.17 → … →
13.3–13.8 → 9.62): every added real statistic removes estimator-visible feedback, and the real
graph beats every ensemble on it — including L4.

## 2. The three new results

### 2.1 W0 — synaptic weight PLACEMENT carries order (+1.44 pp, protocol-conditional)

With the full real wiring diagram fixed and only the weight assignment exchanged, the identical
protocol loses **1.44 pp** (82.221 ± 0.033 vs 83.658; real above all 12 draws; effect = 44×
draw sd). The 5× budget probe (pinned gs6000: 82.277 → 82.848 at 20k epochs/40 sweeps, vs real
84.103 at the same budget) keeps the gap at **+1.26 pp** — not an under-optimization artifact.
Caveats (stage-B review): the direction is **protocol-conditional** (the greedy bound points the
other way: 70.25 vs real 68.91 — greedy favors every structured null and is direction-uninformative
here, as in G01); W0 destroys *all* weight–topology coupling at once, so **which** coupling
(degree/strength, reciprocal balance, order alignment) carries the 1.44 pp is not identified at
this rung. Also: real weights sit in *balanced bidirectional* concentration on reciprocal pairs —
the exchangeable floor drops to 4.46% of W vs the real 6.72%.

### 2.2 L2R — the reciprocal backbone + degrees leave 6.08 pp unexplained (identical-floor contrast)

Fixing the real reciprocal subgraph pointwise (24.86% of W) plus the exact uni/total degree
sequence and out-strengths: FF = 77.581 ± 0.012, real above all draws by **+6.08 pp**. This is
the one ladder contrast where real and null share an identical 2-cycle floor **by construction**
(6.717154% of W, bit-exact in all 12 production draws), so no floor-netting argument can dissolve
the gap: given its own reciprocity and degrees, the fly's choice of **non-reciprocal targets** is
what the +6.08 pp buys — order in the choice of non-reciprocal targets beyond degrees + backbone,
not identified further at this rung (L4 is the decomposition; uni in-strengths are not preserved).
Probe: gap +6.08 → **+6.31 pp** at 5× budget. Prediction adjudication (stage-C review): the
literal FF-axis reading of registered branch (b) ("between w_source and real") did **not** hold
(77.58 < 79.98); the charter's pre-registered sign-check ("L2R can sit below w_source on the FF
axis — interpret jointly with the 2-cycle bound") applies, and on the registered excess-FB axis
L2R sits strictly between w_source and real (19.91 > 15.70 > 9.62). Fixing the backbone (with the
constraints needed to keep it exact) moves excess FB 19.91 → 15.70, i.e. explains ~40% of the
w_source-to-real excess gap. Ensemble note: v1's `w_source` rewires mutual edges and carries
≈0.108% incidental reciprocity; L2R is a v2 ensemble with no-new-mutual-dyads enforced exactly
(repaired fraction 0.287% of uni slots; "approximately uniform" per `CHARTER.md`).

### 2.3 L4 — the cell-type wiring diagram pins ALL of the order and OVERSHOOTS: an inversion

The strictest biological null: every node keeps its type-resolved in/out degree and out-strength
profile over K = 8,857 cell types (direct `cell_type` coverage 99.09%; 1,246 fallback-labeled
nodes = 0.91% of nodes, 0.40% of W — quantified in the stage-D review as NOT the driver of what
follows). Result: **FF(L4) = 84.232 ± 0.009 — the real graph is BELOW all 12 draws** by
0.574 pp (62σ combined; all three real seeds below all twelve draws). The type-matched ensemble
does not merely reach the real level — it overshoots it, landing 0.026 pp from the champion
value (a cross-budget coincidence, not a meaningful identity).

**The probe decides it** (kill table pre-stated in `review_stageD_l4.md` §probe before the probe
ran): KILL required FF(gs8000@20k) ≤ 84.1029; DEMOTE < 84.2029; KEEP ≥ 84.2029, with
quantitative stability at ≥ 84.396. Measured: **84.7562** — KEEP with quantitative stability;
the inversion *grows* at 5× budget (−0.583 → **−0.653 pp**), and the 20k-epoch null exceeds the
real graph's all-time champion by 0.50 pp.

**Both axes, one phrasing (stage-D review):** the type ensemble frees 4.70 pp of forced
reciprocal floor (2.02 vs 6.72% of W) but converts only 0.57 pp of it into feedforward weight;
the other 4.13 pp reappear as estimator-visible excess FB (13.75 vs real 9.62 — on the
registered cross-rung axis the REAL graph still wins by 4.13 pp). So: **relative to its own
type-level wiring diagram, the fly's sub-type wiring is net feedback-spending** — it buys
reciprocal (2-cycle) structure at the price of feedforward orderability. The resemblance to the
mouse meso result (real below strength-matched nulls, G01) is a **co-occurrence, not an
established shared mechanism**: L2R shows that *fixing* the real backbone leaves the real graph
far ABOVE its null; the deficit appears only when sub-type structure — including the choice of
reciprocal partners — is freed.

**Honesty items (stage-D review, all adopted):** (i) neither registered L4 branch fired as
written ("residual bounds sub-type targeting" presumed a non-negative residual; "L4 ≈ L3" is off
by 133σ) — the inversion reading imports the FF(null) > FF(real) branch registered pre-L4 for
the sibling rungs W0/L2R; this import is stated, not disguised. (ii) The type partition does
**NOT** refine the 17 lanes — 91 genuine direct types (R1-6, KCg-m, T4c, …; 43,580 nodes,
31.89%) span lanes, a draw moves 0.60% of edges (0.39% of W) across lane boundaries, so
**L4-vs-L3 (+2.34 pp) is a non-nested comparison** like L3a, a stated deviation from the
charter's "expected ≈ yes". (iii) z-scores are effect sizes against realization noise, not
p-values (empirical p = 1/13 at K = 12). (iv) Direction remains protocol-conditional (identical
budget both sides; 2-cycle upper bounds cannot separate these graphs). (v) Cosmetic:
`gen_stats.variant` inside L4 JSONs reads "L3_dcsbm" (v1 generator reused); filter by
`kind == "type_dcsbm"`.

## 3. Updated answer to the operator's question

The order of the fly connectome, decomposed over everything measured so far (protocol level):

| ingredient | ΔFF | source |
|---|---|---|
| density alone | +19.8 pp | G01 |
| + in/out degrees & out-strengths | +10.2 pp | G01 |
| + 17-lane regional diagram | +1.9 pp | G01 |
| + fine targeting below lanes | +1.8 pp | G01 |
| weight placement on the real wiring (separate axis) | +1.44 pp | **W0** |
| beyond degrees + the real reciprocal backbone (same-floor contrast) | +6.08 pp | **L2R** |
| the full type-resolved diagram | **all of it + 0.57 pp overshoot** | **L4** |

84% feedforward is **not** a generic property of matched random graphs below type resolution —
the real fly beats every such ensemble (now including the real-topology W0). But at type
resolution the attribution closes: the cell-type wiring diagram is sufficient to produce the
entire measured orderability, and the real brain sits slightly *below* its own type ensemble
because its sub-type wiring additionally buys a 3.3× larger irreducible reciprocal core. The
structure "lives" in the type-level connectivity program; the sub-type residual is spent on
recurrence, not on hierarchy.

## 4. Critical re-examination of G01 (operator mandate) — verdicts

1. **Greedy control**: kept as a column, demoted as a direction control — on all three v2 rungs
   it again points opposite to the protocol (70.25/70.30/69.46 vs real 68.91). The budget probe
   is the primary anti-artifact control, now run for every v2 rung (all three directions survive
   5×, movements ≤ 0.23 pp, no sign change). **Verdict: improve (adopted).**
2. **Protocol-conditional wording**: kept everywhere, incl. both new headline directions.
   **Keep.**
3. **K = 12**: draw sd ≤ 0.033 pp on every rung; smallest inter-rung gap read (0.57 pp) is
   ≥ 17× any draw sd. **Keep.**
4. **One pipeline seed per null**: real across-seed sd 0.0033 pp; the L4 inversion is 5.7× the
   charter's 0.1 pp re-run trigger, so no 3-seed re-run was triggered (arithmetic in
   `review_stageD_l4.md`). **Keep.**
5. **Protocol vs champion slack (0.60 pp)**: all reading at protocol level; champion shown as a
   reference line only; the L4-champion proximity labeled a coincidence. **Keep.**
6. (Found during v2) G01's `w_source` 2-cycle bounds were never logged as artifacts; now logged
   for all 76 v1 nulls in `biology/results/nullmodels_v2/bounds_cache.json`, regenerated from
   logged gen seeds (stage-C requirement). **Improve (adopted).**

## 5. Re-run

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
$PY biology/analysis/nullmodels2_generators.py --selftest
$PY biology/analysis/nullmodels2_run.py --dataset connectome --variant w_topo_shuffle
$PY biology/analysis/nullmodels2_run.py --dataset connectome --variant dyad_fixedmut
$PY biology/analysis/nullmodels2_run.py --dataset connectome --variant type_dcsbm
# budget probes (one pinned draw per rung)
$PY biology/analysis/nullmodels2_run.py --dataset connectome --variant <rung> --n-nulls 1 --epochs 20000 --sift-sweeps 40
# ladder analysis + v1 bound cache + figure
$PY biology/analysis/nullmodels2_analyze.py
```

## 6. Parked (operator decision 2026-11-01)

Spatial (wiring-length-preserving) null; MICrONS neuron-level ladder (`CHARTER.md` §6). The L4
inversion sharpens the MICrONS question when un-parked: does a type- (or layer-)matched ensemble
overshoot there too?
