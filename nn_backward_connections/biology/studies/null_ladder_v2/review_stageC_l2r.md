# Independent review — null-ladder v2, Stage C (rung L2R `dyad_fixedmut`, fly, K=12)

**Reviewer:** independent subagent, 2026-11-01 · **scope:** the 12 Stage-C run JSONs
`biology/results/nullmodels_v2/runs/connectome__dyad_fixedmut__gs7000..7011__ps42__e4000.json`,
the 12 saved rank vectors in `biology/results/nullmodels_v2/orders/`, the sweep log
`biology/results/nullmodels_v2/l2r_sweep.log`, and the stage's draft headline/interpretation.
Stage-A ground (sufficiency of the repair constraints for bit-exact bound preservation,
charter numbers) was approved in `review_stageA.md` and is **not redone**; this review
verifies the PRODUCTION runs.

**Instruments:** reviewer-written scripts (scratchpad, not repo artifacts):
`stageC_stats.py` — every statistic recomputed straight from the raw JSONs (json + numpy
only; no `nullmodels2_*` import); `stageC_rescore.py` — 4 draws regenerated from logged
`gen_seed` (the study's `dyad_fixedmut_null` imported for regeneration parity ONLY, as in
stage B), saved rank vectors re-scored by the frozen oracle `mfas.metrics.score_from_order`
on CPU, and **all invariants verified with reviewer-own code**: scipy.sparse elementwise
`min(W, Wᵀ)` int64 2-cycle bound (structurally different from the module's
unique/searchsorted one), `np.isin`-based mutual-dyad classification, bincount degree and
strength checks. Plus 3 regenerated v1 `w_source` draws (v1 `directed_configuration_null`,
logged gen seeds) for their 2-cycle bounds, absent from the v1 JSONs. All CPU; torch never
imported; GPU untouched (L4 sweep running). 64 programmatic checks, 0 failures.

---

## A. Claims table (headline as circulated → reviewer recomputation)

| claim | reviewer recomputation (from raw JSONs / own code) | verdict |
|---|---|---|
| FF(L2R) = 77.5808 ± 0.0117 (K=12) | mean **77.580761**, sd(ddof=1) **0.011680**, K=12 | CONFIRMED |
| min 77.5631, max 77.6033 | min **77.563074** (gs7004), max **77.603318** (gs7006) | CONFIRMED |
| real outside all 12, z ≈ +520 | real 83.6582 ± 0.0033 (3 G01 seeds, recomputed); lowest real 83.6546 > highest draw 77.6033 by **6.051 pp** → outside all 12; z = (83.6582 − 77.5808)/0.0117 = **+520.3** | CONFIRMED |
| real − L2R = +6.08 pp | **+6.0774 pp** | CONFIRMED |
| bound_2cycle_pct = 6.717154 EXACTLY in all 12 | every JSON: `bound_2cycle` = 2,815,303.0 = `bound_2cycle_real`, `bound_2cycle_pct` = **6.7171538671813495** = `bound_2cycle_real_pct`, identical to the last digit in all 12; reviewer-own bound on the real graph = **2,815,303** | CONFIRMED |
| greedy 70.296 ± 0.011 | mean **70.296305**, sd **0.010948** (min 70.2767, max 70.3136) | CONFIRMED |
| real greedy 68.913 (flip again) | **68.913428**, identical across all 3 real seeds; flip confirmed: L2R greedy **+1.383 pp above** real | CONFIRMED |
| excess FB over own floor: L2R 15.70 pp | per-draw (100 − FF) − bound: mean **15.7021 ± 0.0117** | CONFIRMED |
| excess FB real 9.62 pp | (100 − 83.6582) − 6.717154 = **9.6246** | CONFIRMED |
| excess FB w_source ≈ 19.9 pp | reviewer-computed, see §E: 3 regenerated draws give per-draw excess **19.883 / 19.888 / 19.911** (mean 19.894); ensemble estimate 20.0193 − 0.1090 = **19.910** | CONFIRMED (with the provenance requirement in §E — the bounds are not in any logged run JSON) |
| frac_uni_repaired ≈ 0.287% mean | mean **0.2875%** (min 0.2830, max 0.2947; n_bad_initial 13,068–13,611 of 4,617,929 uni slots) | CONFIRMED |

Supporting constants, all re-verified with reviewer-own code on the real graph:
n_mutual_slots = 1,039,790 (18.38% of m), mutual_weight = 10,419,331 (24.86% of W),
real graph loop-free, real 2-cycle bound 2,815,303 = 6.717154% of W.

## B. Integrity of the 12 runs

All of the following hold for **every** run JSON (programmatic check, zero exceptions):
`epochs=4000`, `n_epochs_done=4000`, `sift_max_sweeps=8`, `n_sift_sweeps=8`,
`pipeline_seed=42`, `device="mps"`, `torch_version="2.8.0"`, `n_nodes=136,648`,
`n_edges=5,657,719`, `total_weight=41,912,141`, `kind="dyad_fixedmut"`,
`dataset="connectome"`, `gen_stats.seed == gen_seed`, gen seeds **distinct 7000…7011**,
`gen_stats.variant="dyad_fixedmut"`, `n_bad_remaining=0`,
`n_mutual_slots + n_uni_slots = 5,657,719`, `frac_uni_repaired = n_bad_initial/n_uni_slots`
exactly, every `pct_*` consistent with its `score_*` to <1e-9, every `score_final`
integer-valued and ≥ the greedy/rocket stage scores,
`unavoidable_fb_pct_final = 100 − pct_final`. All 12 saved orders are valid int32 rank
permutations of 0…136,647. Sweep log: 12 `[done]` lines whose final/greedy percentages
match the JSONs, no errors or tracebacks, only the benign torch lr_scheduler deprecation
warning. **PASS.**

## C. Spot checks on regenerated draws (the decisive check)

Four draws regenerated from logged `gen_seed` — gs7000 (first/pinned), gs7004 (ensemble
min), gs7006 (ensemble max), gs7011 (last). In all four, `gen_stats` reproduce
bit-identically (`n_bad_initial`, `n_duplicate_slots`, `repair_rounds`, `n_mutual_slots`),
confirming seed-deterministic regeneration.

(i) **Frozen-oracle re-scores** (`mfas.metrics.score_from_order`, CPU, on the regenerated
graph + the saved rank vector):

| draw | oracle re-score | logged `score_final` | match |
|---|---|---|---|
| gs7000 | 32,514,357 | 32,514,357 | exact |
| gs7004 | 32,508,345 | 32,508,345 | exact |
| gs7006 | 32,525,212 | 32,525,212 | exact |
| gs7011 | 32,512,121 | 32,512,121 | exact |

(ii) **Reviewer-own invariant verification** on each regenerated graph (independent
implementations, not the study's): 2-cycle bound **exactly 2,815,303 = the real graph's**
(sparse min(W,Wᵀ) int64); mutual ordered-pair set **identical to real** (1,039,790 keys);
mutual-slot mask identical (same slots); mutual slots bit-identical in (src, tgt, w);
weights stay on slots (w_source convention, `w2 == w` exactly); src column untouched;
per-node **total** in/out degrees preserved; per-node **uni** in/out degrees preserved;
out-strengths preserved exactly; **zero self-loops**; **zero uni slots whose reverse
exists anywhere** in the null's edge set (no new mutual dyads — direct check, not via the
module's mask). 4/4 draws, all checks pass. **PASS.**

## D. Exact-bound claim across ALL 12 JSONs

Checked field-by-field: `bound_2cycle == bound_2cycle_real == 2815303.0` and
`bound_2cycle_pct == bound_2cycle_real_pct == 6.7171538671813495` in **all 12** JSONs, to
the last digit. Combined with §C(ii) (bound re-derived with a structurally different
implementation on 4 regenerated graphs) the production guarantee — identical 2-cycle floor
for real and every L2R draw — holds **exactly**, as constructed. **PASS.**

## E. The w_source excess-FB comparator — provenance

The v1 `w_source` run JSONs (Sept 2026) carry **no** 2-cycle-bound field, and no
`bounds_cache.json` exists anywhere in the repo. The reviewer therefore regenerated 3 of
the 20 fly w_source draws (gs2000, gs2010, gs2019) from their logged gen seeds
(`gen_stats` reproduce: self-loop and duplicate counts match the Sept logs exactly) and
computed their bounds with the reviewer-own implementation:

| draw | bound | % of W | FF (logged) | excess FB |
|---|---|---|---|---|
| gs2000 | 45,217 | 0.107885 | 80.0090 | 19.883 |
| gs2010 | 46,300 | 0.110469 | 80.0016 | 19.888 |
| gs2019 | 45,578 | 0.108747 | 79.9802 | 19.911 |

So "w_source ≈ 19.9 pp" is **CONFIRMED numerically** — but as quoted in the draft it does
not yet satisfy charter rule 3 ("every quoted number traces to a logged run JSON"): the
FF part does, the bound part exists only as reviewer regenerations. **Required:** before
REPORT.md quotes 19.9, either (preferred) compute and log the bounds for all 20 fly
w_source draws to a results artifact (e.g. `biology/results/nullmodels_v2/bounds_cache.json`)
with the re-runnable command, or state explicitly that the w_source floor (≈0.108% of W)
comes from K=3 seed-exact regenerations documented in this review. The spread observed
(0.1079–0.1105% of W) is ~25× smaller than the 4.2 pp w_source→L2R excess gap, so K=3 vs
K=20 cannot change any conclusion — this is a bookkeeping requirement, not a numeric risk.

## F. Timing / device anomalies

t_rocket 21.22–21.60 s (max/min 1.018), t_total 56.5–58.3 s, t_generate 27.8–54.4 s
(co-varying with repair_rounds 3–5 — expected, the repair loop dominates generation).
Null-run t_rocket matches Stage B (21.6–22.0 s) and the Sept G01 null runs — a silent CPU
fallback is excluded (4,000 epochs over 5.7 M edges cannot finish in 22 s on CPU;
`device="mps"` logged). Timestamps 16:20–16:38 UTC on 2026-11-01, ~88 s cadence for 12
sequential runs — consistent with t_total + t_generate. Same torch 2.8.0 as the Sept
comparators. No anomaly. **PASS.**

## G. Which registered prediction fired

**Adjudication: (b), under its registered joint reading — with one disclosure required.**

* Literal (b) as worded on the FF axis — "FF(L2R) between w_source and real" — did **not**
  hold: FF(L2R) 77.58 sits **below** w_source 79.98.
* But the **same stage-A bullet** that registered (b) also registered the sign check:
  "fixing the mutual subgraph *forces* ≥6.72% FB into the null, so FF(L2R) could even sit
  *below* w_source on the FF axis while being the far stricter null — interpret jointly
  with the 2-cycle bound, not FF alone." Verified in git: that sentence is in the stage-A
  commit (e2a0424), i.e. registered **before any v2 run existed**. The cross-rung
  excess-FB statistic and outcome (c) were added post-stage-A but **before any L2R run**
  (commit c77c040; the charter's own timing note discloses this honestly).
* On the pre-specified commensurable axis (excess FB over own floor): w_source **19.91** >
  L2R **15.70** > real **9.62** — L2R sits strictly between, which is exactly (b)'s
  substance: reciprocity + degrees explain part (the w_source→L2R move, −4.21 pp of excess
  FB, ≈41% of the w_source→real span of 10.29 pp); the remaining real−L2R gap (6.08 pp,
  ≈59%) is order the null cannot produce.
* (a) FF(L2R) ≈ real did not fire (6.08 pp, real outside all 12). (c) FF(L2R) > FF(real)
  did not fire.

**No post-hoc re-labeling is happening** — the joint-reading instruction predates all v2
runs and the excess-FB axis predates all L2R runs — **provided** REPORT.md states the
adjudication exactly as above: literal FF-axis (b) false, (b) fired on the registered
joint reading, with the timing note cited. Writing "(b) fired" with no FF-axis disclaimer
would be a (mild) re-labeling and is not acceptable.

## H. Red-team of the draft interpretation

Draft under review: *"real and L2R share an identical 2-cycle floor, so the +6.08 pp is
pure directional targeting beyond reciprocity and degrees; the reciprocal backbone alone
moves the excess-FB statistic from ~19.9 (w_source) to 15.70, i.e. explains part but far
from all of the fly's advantage."*

All numbers in it are confirmed (§A, §E). The floor-identity premise is exactly right and
is the rung's real strength: real vs L2R is the one contrast in the whole ladder where the
FF gap and the excess-FB gap are the **same number by construction** (identical floors),
so no floor-netting argument can dissolve it. Caveats, in decreasing order of weight:

1. **"Pure directional targeting" — the word "pure" overclaims twice.**
   (i) *Protocol-conditional:* both FF values are achieved lower bounds under an identical
   epoch budget; there is no estimator-free proof that the real graph is more orderable
   (the 2-cycle upper bounds are identical, hence non-separating). The greedy estimator
   again points the **other way** (L2R 70.30 > real 68.91, +1.38 pp — the same flip as W0
   and G01's degree-matched rungs), direct evidence the direction is estimator-dependent
   at the weak-estimator end. The Stage-E 5×-budget probe for this rung (pinned draw
   gs7000 vs the real 20,000-epoch value 84.1029) **has not run**. Until it is logged,
   state the +6.08 pp as "under the protocol estimator and budget" and do not promote to
   `biology/FINDINGS.md`. The 6.08 pp gap is 4× larger than W0's 1.44, so the needed
   estimator asymmetry would be implausibly large — say that too, but as plausibility,
   not proof.
   (ii) *"Directional targeting" bundles every axis the null destroys:* which uni target
   each neuron picks **including its weight** (uni in-strength profiles are NOT preserved
   — only out-strengths are, the w_source convention), plus all meso-structure above
   degrees — G01 already prices the 17-lane regional diagram at ≈ +1.9 pp and
   sub-annotation structure at ≈ +1.8 pp of exactly this residual. Required wording:
   "order in the choice of non-reciprocal targets beyond degrees + the reciprocal backbone
   (including regional/type-level organization and weight–target coupling; not identified
   further at this rung)" — or drop "pure" and add that one sentence. The upcoming L4
   rung is precisely the decomposition of this residual.
2. **"The reciprocal backbone alone moves 19.9 → 15.70" — not a single-axis delta.**
   L2R differs from v1 w_source by a bundle: the mutual subgraph fixed pointwise *with its
   weights* (24.86% of W immobilized), permutation restricted to uni slots, the
   no-new-mutual constraint (absent in v1: w_source draws carry small incidental
   reciprocity, ≈0.108% of W, and rewire the real mutual edges too), and different repair
   conditioning (0.287% of uni slots vs 186 self-loops of 5.66M in v1). The excess-FB
   statistic nets out the floor difference, which is the dominant effect, but the residual
   ensemble differences are bundled into "the backbone". Required wording: "fixing the
   real reciprocal backbone (with the constraints needed to keep it exact)". Also keep the
   v1-vs-v2 ensemble note: the 19.9 comparator is a Sept-2026 v1 ensemble under v1's
   looser dyad conventions, same protocol/machine/torch (era note as in stage B).
3. **Excess-FB is a bound gap, not a measured quantity** (stage-B §E3 verbatim applies):
   (100 − FF) is an upper bound on minimal feedback, the floor a lower bound. Keep
   "estimator-visible" attached to every use; never let 15.70 vs 9.62 become "the real
   brain has 6 pp less feedback".
4. **Approximate uniformity:** swap-repair biases the conditional ensemble; the charter
   already says "approximately uniform", and the two logged residuals are both tiny —
   0.2875% of uni slots repaired, 0.241% collapsed into duplicate parallel slots
   (11,123 mean of 4.62M; score-invariant per the v1 policy) — orders of magnitude too
   small to threaten a 6.08 pp / 520σ gap. But the "approximately" qualifier and the
   per-draw repair fraction must stay in REPORT.md (stage-A correction 2, already applied
   to the charter — keep it in the report too).
5. **One pipeline seed per draw — justified here:** draw σ 0.0117 pp ≈ 3.5× the real
   across-seed σ 0.0033 pp and both are ~500× below the 6.08 pp gap; the charter §5.4
   re-run trigger (within 0.1 pp of real) is nowhere near hit.
6. **z ≈ +520 is an effect size, not a p-value.** With K=12 the honest nonparametric
   statement is "real outside all 12 draws" (empirical one-sided p ≥ 1/13). Quote z as
   dispersion-relative distance only; a Gaussian tail at 520σ is meaningless.

## I. Required before these numbers enter REPORT.md

1. Log the w_source 2-cycle bounds (all 20 draws, re-runnable command) or explicitly
   source the 19.9 comparator to this review's K=3 regenerations. [§E]
2. State the prediction adjudication exactly as in §G: literal FF-axis (b) did not hold;
   (b) fired on the registered joint reading (sign check + excess-FB axis); cite the
   charter's timing note. Never an unqualified "(b) fired". [§G]
3. Drop "pure"; scope the residual as in §H1(ii); state the +6.08 pp as
   protocol-conditional and report the greedy flip (70.296 vs 68.913) beside the
   headline; no promotion to FINDINGS.md before the Stage-E gs7000 budget probe is
   logged. [§H1]
4. "The reciprocal backbone alone" → "fixing the real reciprocal backbone (with the
   constraints needed to keep it exact)", plus the v1-vs-v2 ensemble/era sentence. [§H2]
5. Keep "estimator-visible" (bound-gap wording) on every excess-FB use; keep
   "approximately uniform" + the 0.287% repaired fraction. [§H3, §H4]

None of these touches a number produced by the runs: every logged quantity reproduced
exactly, including 4 bit-exact oracle re-scores on regenerated graphs, the exact-floor
guarantee verified to the last digit in all 12 JSONs and re-derived independently on 4
draws, and the full invariant suite (degrees, strengths, dyad census, loop-freedom)
passing under reviewer-own implementations — 64 checks, 0 failures.

**Final verdict: PASS** (headline statistics confirmed in full; production exact-bound
guarantee holds bit-exactly; interpretation usable in REPORT.md only with §I applied).
