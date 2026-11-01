# Independent review — null-ladder v2, Stage B (rung W0 `w_topo_shuffle`, fly, K=12)

**Reviewer:** independent subagent, 2026-11-01 · **scope:** the 12 Stage-B run JSONs
`biology/results/nullmodels_v2/runs/connectome__w_topo_shuffle__gs6000..6011__ps42__e4000.json`,
the 12 saved rank vectors in `biology/results/nullmodels_v2/orders/`, the sweep log
`biology/results/nullmodels_v2/w0_sweep.log`, and the stage's draft headline/interpretation.
Stage-A ground (generator correctness, charter numbers) was approved in `review_stageA.md`
and is **not redone** here.

**Instruments:** reviewer-written scripts (scratchpad, not repo artifacts):
`stageB_stats.py` — all statistics recomputed straight from the raw JSONs (json + numpy
only; no `nullmodels2_*` import); `stageB_rescore.py` — spot re-scores with the frozen
oracle `mfas.metrics.score_from_order` on CPU after regenerating each null from its logged
`gen_seed` (importing `w_topo_shuffle_null` for regeneration parity only, as permitted),
plus a reviewer-own 2-cycle bound (scipy.sparse elementwise `min(W, Wᵀ)` on int64 — a
structurally different implementation from the module's unique/searchsorted one). All CPU;
torch never imported; run while the L2R GPU sweep was active (harmless: epoch-counted
budgets, and no new runs were produced here).

---

## A. Claims table (headline as circulated → reviewer recomputation)

| claim | reviewer recomputation (from raw JSONs) | verdict |
|---|---|---|
| FF(W0) = 82.221 ± 0.033 (K=12) | mean **82.2214**, sd(ddof=1) **0.0325**, K=12 | CONFIRMED |
| min 82.182, max 82.277 | min **82.1818** (gs6011), max **82.2767** (gs6000) | CONFIRMED |
| real = 83.658, outside all 12 draws | real mean **83.6582 ± 0.0033** (3 G01 seeds, recomputed from `connectome__real__ps{42,123,999}__e4000.json`); lowest real run 83.6546 > highest draw 82.2767 by **1.38 pp** → outside all 12 | CONFIRMED |
| z ≈ +44 | (83.6582 − 82.2214)/0.0325 = **+44.15** | CONFIRMED |
| null 2-cycle bound 4.4596 ± 0.0028 % of W | mean **4.4596**, sd **0.0028** (abs 1,866,491–1,870,839) | CONFIRMED |
| real 2-cycle bound 6.7172 % of W | reviewer-own implementation on the real graph: **2,815,303 = 6.717154 %** — matches every JSON's `bound_2cycle_real` | CONFIRMED |
| greedy on W0 = 70.249 ± 0.037 | mean **70.2485**, sd **0.0367** (min 70.2124, max 70.3357) | CONFIRMED |
| real greedy 68.913 | **68.91343**, identical across all 3 real runs (deterministic greedy) | CONFIRMED |
| excess FB over own floor: W0 ≈ 13.32 pp | per-draw (100 − FF) − bound: mean **13.3191 ± 0.0314** | CONFIRMED |
| excess FB real ≈ 9.63 pp | (100 − 83.6582) − 6.71715 = **9.6246** → rounds to **9.62**, not 9.63 | CORRECTED (rounding only; the delta is unaffected) |
| (implied) delta W0 − real ≈ 3.7 pp | **+3.694 pp** | CONFIRMED |
| FF deficit "loses 1.44 pp" | 83.6582 − 82.2214 = **1.4369 pp** | CONFIRMED |

## B. Integrity of the 12 runs

All of the following hold for **every** run JSON (programmatic check, zero exceptions):
`epochs=4000`, `n_epochs_done=4000`, `sift_max_sweeps=8`, `n_sift_sweeps=8`,
`pipeline_seed=42`, `device="mps"`, `n_nodes=136,648`, `n_edges=5,657,719`,
`total_weight=41,912,141`, `kind="w_topo_shuffle"`, `dataset="connectome"`,
`torch_version="2.8.0"` (the stage-A era-logging recommendation was implemented),
`gen_stats.seed == gen_seed`, gen seeds **distinct 6000…6011**, every `pct_*` consistent
with its `score_*` to <1e-9, every `score_final` integer-valued and ≥ the earlier stage
scores, `bound_2cycle_real = 2,815,303` in all 12. All 12 saved orders are valid int32
rank permutations of 0…136,647. Sweep log: 12 `[done]` lines, no errors/tracebacks, only
the benign torch lr_scheduler deprecation warning. **PASS.**

## C. Spot re-scores (the decisive check)

For 4 of 12 draws (gs6000, gs6004, gs6007, gs6011): null regenerated from the logged
`gen_seed` via `w_topo_shuffle_null` (topology verified bit-identical to real, weight
multiset verified preserved), saved rank vector loaded, scored by the **frozen oracle**
`mfas.metrics.score_from_order` on CPU (the .npy is the final RANK vector saved by
`nullmodels_run._finish_run`; `score_from_order` consumes ranks — semantics verified in
source before use):

| draw | oracle re-score | logged `score_final` | reviewer 2-cycle bound | logged |
|---|---|---|---|---|
| gs6000 | 34,483,936 | 34,483,936 | 1,869,086 (4.459534 %) | identical |
| gs6004 | 34,454,093 | 34,454,093 | 1,869,172 (4.459739 %) | identical |
| gs6007 | 34,467,194 | 34,467,194 | 1,868,992 (4.459309 %) | identical |
| gs6011 | 34,444,161 | 34,444,161 | 1,870,316 (4.462468 %) | identical |

All four integer scores and all four bounds reproduce **exactly**; the real-graph bound
recomputed with the reviewer's own implementation is also exact (2,815,303). **PASS.**

## D. Timing / device anomalies

t_rocket 21.6–22.0 s (max/min 1.02), t_total 59.6–65.8 s across all 12 — tight, uniform,
far under any deadline, and matching the September G01 *null*-run timings (21.0–21.8 s)
bit of environment history exactly; a silent CPU fallback is excluded (4,000 epochs over
5.7 M edges on CPU could not finish in 22 s, and `device="mps"` is logged). Note for the
record: G01's *real* runs logged t_rocket ≈ 43.5 s — a 2× real-vs-null wall-clock gap that
already existed within September itself, so it is **not** Nov-vs-Sept era drift; it is
fairness-neutral anyway because budgets are epoch-counted (4,000 epochs confirmed done in
every run). Same torch 2.8.0 both eras. No anomaly. **PASS.**

## E. Prediction adjudication + red-team of the draft interpretation

**Which registered outcome fired: (b)** — FF(W0) markedly below real (−1.44 pp, z ≈ +44,
real outside all 12 draws). (a) FF ≈ real did not fire; the late-registered (c)
FF(W0) > FF(real) did not fire. The charter's timing note (c registered when 6 W0 draws
existed and gs6000 was visible) is honest and does not taint (b), which was registered
up front before any run.

Draft reading under review: *"even with the full real binary topology, exchangeable
weight placement loses 1.44 pp, so synaptic weight placement itself carries order; the
real weights also concentrate on reciprocal pairs (floor 6.72 vs 4.46), and net of its
own floor the real graph has 3.7 pp LESS estimator-visible feedback."*

The numbers in it are all confirmed (§A). Four caveats, in decreasing order of weight:

1. **Protocol-conditional direction — and this rung's own greedy control flips.** Both
   FF values are achieved lower bounds on each graph's optimum under an identical budget;
   the 2-cycle upper bounds (real ≤ 93.28 %, W0 ≤ ~95.54 %) do not separate the optima, so
   there is no estimator-free proof that real topology+weights is *more* orderable than
   W0. The greedy estimator in fact points the **other way** (W0 70.25 > real 68.91,
   +1.34 pp) — the same flip G01 hit on its degree-matched rungs — which is direct
   evidence that the direction is estimator-dependent. The charter (§5.1–5.2) already
   demotes greedy and designates the 5×-budget probe as the primary anti-artifact
   control, but **the Stage-E probe for W0 has not run yet**. Until the pinned-draw
   (gs6000) 20,000-epoch probe confirms the gap against the real graph's logged
   20,000-epoch value (84.1029), "weight placement itself carries order" must be stated
   as "under the protocol estimator and budget", and should not be promoted to
   `biology/FINDINGS.md` as a standalone finding. The greedy flip must be reported
   alongside the headline, not dropped.
2. **"Weight placement carries order" over-localizes the cause.** W0 is *fully
   exchangeable* placement: it destroys every weight–topology coupling at once —
   weight–degree/strength coupling at nodes, the reciprocal-dyad weight balance, and any
   alignment of strong weights with a global order. The 1.44 pp deficit establishes that
   *some* weight–edge coupling matters, not that the coupling is specifically
   hierarchical/order-aligned. (A within-degree-class or strength-preserving weight
   shuffle would be the decomposing null; not pre-registered, so merely note the
   limitation.) Suggested wording: "weight placement carries order *information the
   estimator can exploit*; which coupling carries it is not identified at this rung."
3. **The excess-FB statistic is a bound gap, not a measured quantity.** (100 − FF) is an
   upper bound on minimal feedback; the 2-cycle floor is a lower bound. "Real has 3.7 pp
   less estimator-visible feedback net of floor" can reflect (i) genuinely less avoidable
   feedback, (ii) a *tighter* floor on the real graph (real unavoidable FB concentrated in
   2-cycles rather than longer cycles), or (iii) the protocol landing closer to the real
   optimum. These are not separable here. The draft's "estimator-visible" qualifier is
   exactly right — keep it verbatim in REPORT.md and resist any paraphrase to "the real
   brain has 3.7 pp less feedback". Also correct 9.63 → **9.62** (§A).
4. **"Concentrate on reciprocal pairs" — strictly, *balanced* concentration.** The floor
   is Σ min(w(u→v), w(v→u)) over the (identical) real pair set, so real 6.72 % vs null
   4.46 % means real reciprocal dyads carry higher *minima* than exchangeable placement
   gives — heavy weights sit on **both** directions of the same dyads. One heavy direction
   alone would not raise the bound. One-word fix ("concentrate **bidirectionally**" or
   "balanced-heavy"). Side observation worth one sentence in REPORT.md: exchangeable
   placement *frees* 2.26 pp of W from the 2-cycle floor, yet FF still drops 1.44 pp —
   the excess-FB statistic (13.32 vs 9.62) is precisely the right way to net these two
   movements, and it moves in the same direction as the FF headline.

Minor statistical notes (no action needed): one pipeline seed per draw is justified here
(draw σ 0.033 pp ≈ 10× the real across-seed σ 0.0033 pp; both ~40× below the 1.44 pp gap;
the charter §5.4 re-run trigger of 0.1 pp is nowhere near hit). z ≈ 44 is an effect size,
not a p-value — with K=12 the honest nonparametric statement stays "outside all 12 draws"
(empirical one-sided p ≥ 1/13); the draft already uses it. Comparator era: real
comparators are Sept-2026 runs, W0 is Nov-2026, same machine class, same torch 2.8.0,
and the stable null-run timings across eras argue against environment drift; given the
gap/σ ratio this is a non-issue, but REPORT.md should carry the one-line era note the
stage-A review asked for.

## F. Required before the numbers enter REPORT.md

1. Change real excess FB **9.63 → 9.62 pp** (delta stays +3.69 ≈ 3.7 pp). [§A]
2. State the headline direction as **protocol-conditional** and report the greedy flip
   (W0 greedy 70.25 > real 68.91) next to it; do not promote "weight placement carries
   order" beyond this study until the Stage-E 5×-budget probe on gs6000 is logged. [§E1]
3. Keep "estimator-visible" (or equivalent bound-gap wording) attached to every use of
   the excess-FB statistic; do not let it become "less feedback". [§E3]
4. Soften "weight placement itself carries order" to scope it to "some weight–topology
   coupling, not identified at this rung" (or add the one-sentence limitation). [§E2]
5. Cosmetic: "concentrate on reciprocal pairs" → balanced/bidirectional concentration. [§E4]

None of these touches a number produced by the runs; every logged quantity reproduced
exactly, including four bit-exact oracle re-scores from regenerated nulls and an
independent recomputation of every 2-cycle bound quoted.

**Final verdict: PASS** (headline statistics confirmed in full; one rounding correction;
interpretation usable in REPORT.md only with the caveats in §F).
