# Mirror-swap re-optimization: none of the −2.21 pp is irrecoverable

**status: reviewed with corrections** (independent review 2026-09-30, appended below)
Date: 2026-09-30 · Machine: MacBook (Apple Silicon), all stages CPU · Interpreter: conda env `allen`
Script: `biology/analysis/exp_mirror_retune.py` · Results: `biology/results/mirror_retune.json`
Figure: `biology/outputs/mirror_retune_recovery.png`

## Question

`layering/QA_ORDER_AND_ROBUSTNESS.md` § Q3 showed that swapping every mirror-paired
left/right neuron's rank in the fly-connectome champion order (65,452 within-type
rank-matched pairs, 95.8 % of neurons) costs FF 84.26 % → 82.05 % (−2.21 pp), and
stated the open caveat: −2.21 pp is only an **upper** bound on the cost of
mirroring, because the swapped order was never re-optimized. This study closes that
caveat: **re-optimize after the swap and measure the irrecoverable residual**, which
is a lower-bound estimate of true left/right wiring asymmetry as seen by this
objective, separated from mere fine-tuning loss. Secondarily: does re-optimization
**un-swap** the pairs (relaxation back to the champion) or stay mirrored (a
symmetric twin optimum)?

## Method

Re-optimization = the champion's own discrete refinement, exact production code and
the H64 connectome constants, applied **identically** to three starts:

- **stage 3** `src/mfas/refine/underrelax.py::sift_underrelaxed`
  (`max_sweeps=40, k_full=6, alpha=0.7`),
- **stage 4** `src/mfas/refine/scc_recursive.py::alternate_scc_sift`
  (`n_cycles=77, sift_sweeps=2, k_full=2, alpha=0.7, min_block=32,
  split_fracs=(0.5, 0.382, 0.618, 0.25, 0.75)`, default `tie_break="first"`).

**No stage-2 Rocket restart** — a gradient restart would discard the swap under
study. Both stages are best-by-oracle (frozen oracle `src/mfas/metrics.py`, discrete
score on CPU), so no retuned score can regress below its start. Both stages are
deterministic given the start order (no RNG is consumed).

The three starts:

| start | what it is | why |
|---|---|---|
| (a) champion | the pinned H64 order itself | control — the champion should be ≈ a fixed point of its own refinement; what it still gains is the yardstick's zero |
| (b) mirror | Q3's mirror-swapped order | the object of study |
| (c) c1b | Q3's same-side, same-type, rank-matched control swap | fair structural control: identical pairing structure, no hemisphere crossing |

**Pairing provenance.** The mirror pairing and all controls are rebuilt bit-exactly
from `layering/tools/exp_mirror_swap.py` (seed 42): `load_types` and `swapped_rank`
are imported from it; the pairing/control construction (inline in its `main()`, not
importable) is replicated verbatim in `build_pairings()`, including the order of RNG
consumption (C1 pool → C1 subsample → C1b → C2 from one `default_rng(42)`).

**Regression gate (passed).** Before anything ran, the reproduced swapped orders
were scored through the frozen oracle and compared against
`layering/outputs/qa/exp_mirror_swap.json`. All four schemes matched **bit-exactly**
(n_pairs and FF %):

```
mirror_swap                               65,452  82.0453052017 %  OK
control_same_side_same_type_random        62,137  81.2468969314 %  OK
control_same_side_same_type_rank_matched  62,137  82.7206441208 %  OK
control_random_pairs                      65,452  50.8451834995 %  OK
```

## Numbers

Champion reference: 84.25817950936937 % = exact score **35,314,407** (parity gate in
`layering/io_utils.py`; total weight 41,912,141). *[Corrected in review: the draft
said "34,751,902", which is the Rocket-era scorer parity anchor (82.9161 %), not
this champion's exact score.]* All scores below are frozen-oracle
values from the logged run `biology/results/mirror_retune.json`; the saved rank
vectors were re-scored through the oracle after the run and reproduce the JSON
values exactly.

| start | FF before | after stage 3 | after stage 4 (= retuned) | exact score | recovered | % of deficit | vs champion | wall |
|---|---|---|---|---|---|---|---|---|
| (a) champion | 84.2582 % | 84.2587 % | **84.2621 %** | 35,316,059 | +0.0039 pp | — (zero deficit) | +0.0039 pp | 1,001 s |
| (b) mirror | 82.0453 % | 84.2372 % | **84.3281 %** | 35,343,705 | +2.2828 pp | **103.2 %** | **+0.0699 pp** | 1,217 s |
| (c) c1b | 82.7206 % | 84.2764 % | **84.3228 %** | 35,341,493 | +1.6022 pp | **104.2 %** | +0.0646 pp | 1,134 s |

Retuned rank vectors: `biology/results/retuned_{champion,mirror,c1b}_rank.npy`;
pairings: `biology/results/{mirror,c1b}_pairs.npy`.

### Headline

| quantity | value |
|---|---|
| irrecoverable mirror gap, retuned(a) − retuned(b) | **−0.0660 pp** |
| structural control gap, retuned(a) − retuned(c) | −0.0607 pp |
| mirror-specific residual, retuned(c) − retuned(b) | **−0.0053 pp** |

Both perturbed starts **over-recovered**: they end not merely at, but ~0.065–0.07 pp
*above* the champion and ~0.06–0.066 pp above the champion's own re-tuning. The
"irrecoverable residual" the study set out to measure is therefore **zero** — a
lower-bound estimate of true left/right wiring asymmetry, as seen by this objective,
of **0 pp**. The scale against which "zero" is judged: two starts of equal
legitimacy (retuned champion vs retuned C1b) land 0.061 pp apart, so endpoint
differences ≲ 0.06 pp are basin lottery, not signal; the mirror-vs-C1b difference
(0.0053 pp, mirror *ahead*) is far inside that.

### Un-swap analysis

After re-tuning the mirror start, for each of the 65,452 pairs: is the within-pair
rank relation back to the champion's (reverted) or still exchanged (stayed
mirrored)?

| quantity | mirror retune | c1b retune (control) |
|---|---|---|
| pairs reverted to champion relation | **43.3 %** | 58.5 % |
| pairs stayed swapped | **56.7 %** | 41.5 % |

Spearman (rank vectors are permutations, so this is Pearson on ranks):

| pair of orders | ρ |
|---|---|
| swapped start vs champion | 0.9119 |
| retuned mirror vs champion | 0.9660 |
| retuned mirror vs swapped start | 0.9066 |

Mean |rank shift| of the retuned mirror order: 4,696 vs the champion, 6,387 vs its
own swapped start.

**Reading.** The majority of mirror pairs (56.7 %) keep the exchanged
hemisphere relation — under the identical procedure, same-side control pairs
mostly revert (58.5 %). So the mirrored configuration is not an unstable state that
refinement relaxes back; it is retained *preferentially relative to the structural
control* while the surrounding order re-tunes, and it supports a solution slightly
better than the champion. That is direct evidence for **two symmetric near-optima**:
the champion's hemisphere assignment and its mirror image are interchangeable
solutions of essentially equal (here: measurably equal-or-better) quality. At the
same time the coarse geometry drifts back toward champion-like structure (ρ vs
champion rises 0.912 → 0.966; the retuned order is *closer* to the champion than to
its own start in mean |rank shift|) — the twin optimum shares the champion's global
lane architecture and differs in the L/R identity assignment within it.

## Biological reading (cautious)

- Q3's conclusion is **strengthened**: the −2.21 pp mirror-swap cost was entirely
  fine-tuning loss, not wiring asymmetry. After the champion's own refinement is
  re-applied, mirroring costs **nothing measurable**. Within what this objective and
  this pairing can see, the fly connectome's two hemispheres are computationally
  interchangeable for feedforward layout.
- This does **not** show the hemispheres are identical — only that whatever L/R
  wiring differences exist (and Q1/Q2 document real ones, e.g. the left-early lean)
  are invisible to the total-feedforward-weight objective at ≲ 0.06 pp resolution
  after local re-optimization with a rank-matched within-type pairing.

## Surprise worth flagging (methodological, not biological)

`retuned_mirror_rank.npy` scores **84.3281 %** (35,343,705) — **+0.0699 pp above
the registered H64 connectome champion** (84.2582 %), found incidentally by
perturb-then-retune (a basin-hopping-style kick: a large structured perturbation
followed by the champion's own local refinement escapes the champion's local
optimum). The C1b kick lands at 84.3228 %, and even re-running stages 3+4 on the
sitting champion gains +0.0039 pp (its cycle budget, not a fixed point, is what
terminated the original run). **This is an observation, not a champion promotion**:
no campaign protocol run, screen/confirm gates, or `autoresearch/audit.py` was
involved, and this study did not touch `autoresearch/`. If the campaign wants it,
"structured-swap kick + stages 3+4" is a candidate variant to run through the
proper gate ladder.

## Caveats

- **Single pairing scheme.** Within-type rank-order matching (seed 42), not
  morphological twin matching; ~4.2 % of neurons are unpaired and never swap. A
  finer (e.g. soma-position) pairing could expose asymmetry this one averages away.
- **Refinement is local.** Stages 3+4 are the champion's own local refiners; a
  global re-run (greedy + Rocket from the mirrored configuration) could land
  elsewhere. "Irrecoverable" here means "irrecoverable by the champion's own
  discrete refinement", which is the comparison that isolates fine-tuning loss.
- **One draw of each control.** C1b is a single seed-42 construction (as in Q3);
  the ~0.06 pp basin-lottery scale is read from two points, not a distribution.
- **Deterministic, single-machine numbers.** All stages are CPU numpy/scipy and
  consume no RNG given the start order; scores are exact oracle values. Wall times
  (~17–20 min per start) were measured under concurrent CPU load and are
  indicative only.
- The un-swap fractions treat each pair's *relative* order only; they say nothing
  about absolute displacement (both retuned orders move every neuron somewhat).

## Re-run

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python   # MacBook; use this machine's allen env
# full run (~57 min total on a MacBook under load; deterministic):
$PY biology/analysis/exp_mirror_retune.py
# plumbing smoke test (sift 2 sweeps, 1 alt cycle, '_smoke' outputs):
$PY biology/analysis/exp_mirror_retune.py --smoke
```

Outputs: `biology/results/mirror_retune.json`,
`biology/results/retuned_{champion,mirror,c1b}_rank.npy`,
`biology/results/{mirror,c1b}_pairs.npy`,
`biology/outputs/mirror_retune_recovery.png` (regenerated from the logged JSON
after a label-layout fix; all plotted values are the logged run's).

---

## Independent review (2026-09-30)

Reviewer: independent session, own code only. Script:
`biology/analysis/review_mirror_retune.py` · Results:
`biology/results/review/mirror_retune_review.json` (43 checks; re-run:
`$PY biology/analysis/review_mirror_retune.py`, ~30 s). All scores below were
recomputed through the frozen oracle (`src/mfas/metrics.py`, CPU) from the saved
artifacts; nothing was taken from the study's JSON except to cross-check it.

### Verdicts

| claim | verdict |
|---|---|
| Champion 84.25817950936937 % on total weight 41,912,141 (io_utils parity gate) | **confirmed bit-exact** |
| Retuned vectors are valid permutations; oracle scores: champion+retune 84.262121088016 % = 35,316,059; mirror+retune 84.32808288175973 % = **35,343,705**; C1b+retune 84.32280517475832 % = 35,341,493 | **confirmed bit-exact** (all three) |
| Swap starts rebuilt from saved pairs + champion rank: mirror 82.04530520165983 %, C1b 82.72064412075727 %; regression gate vs `layering/outputs/qa/exp_mirror_swap.json` | **confirmed bit-exact**, incl. an independent rebuild of ALL four pairings from the annotations (reviewer replication of `exp_mirror_swap.py`): saved `mirror_pairs.npy`/`c1b_pairs.npy` are array-identical to the rebuild, and the unsaved C1/C2 starts also reproduce the QA json bit-exactly (81.24689693136888 % / 50.84518349945425 %) |
| Recovery: mirror +2.2828 pp = 103.2 % of its 2.2129 deficit; C1b +1.6022 pp = 104.2 % of 1.5375 | **confirmed** (1.0315894…, 1.0420320… exactly) |
| Headline gaps: a−b = −0.0660 pp, a−c = −0.0607 pp, c−b = −0.0053 pp | **confirmed bit-exact** |
| Un-swap: mirror 56.7 % stay / 43.3 % revert; C1b 41.5 % stay / 58.5 % revert | **confirmed bit-exact**; definition checked — within-pair rank relation vs the champion's original relation, identical computation and per-scheme denominators for both schemes (fair) |
| Spearman 0.9119 / 0.9660 / 0.9066; mean \|rank shift\| 4,696 / 6,387 | **confirmed bit-exact** |
| Pairs disjoint, in-range; swapped starts are permutations; 65,452 pairs = 95.8 % of 136,648 neurons | **confirmed** |
| "Both stages deterministic, no RNG consumed" | **confirmed by code audit** (not re-run): `sift_underrelaxed` and `alternate_scc_sift` use only stable argsorts, a heap keyed on positions, and scipy `connected_components` — no RNG anywhere on the path (the only RNG in `refine/insertion.py` is in `sift_gauss_seidel_ref`, unused here). Best-by-oracle tracking in both stages also confirms "no retuned score can regress below its start". H64 constants verified against `src/mfas/experiments/H64.py` (40/6/0.7; 77/2/2/0.7/32/default fracs, `tie_break="first"` default) |
| Stage-3 intermediates (84.2587/84.2372/84.2764) and wall times | **not independently re-run** (logged-run values; internally consistent: start ≤ stage 3 ≤ stage 4 for all three) |

### Red-team

**(a) "Irrecoverable asymmetry = 0" given that both perturbed starts
over-recover.** The over-recovery does mean stages 3+4 are not run to
convergence from any start (the sitting champion itself gains +0.0039 pp, so it
was terminated by its cycle budget, as the study says — confirmed by oracle:
35,316,059 > 35,314,407). That makes every endpoint a basin-lottery draw at the
~0.06 pp scale — which is exactly how the study calibrates its zero: the honest
claim is "no *detectable* asymmetry at the ~0.06 pp resolution set by the
retuned-champion vs retuned-C1b spread", and the study's own wording ("endpoint
differences ≲ 0.06 pp are basin lottery, not signal"; "invisible … at ≲ 0.06 pp
resolution") says precisely that. The alternative reading (perturbed starts
merely escape the champion's basin) does not rescue an asymmetry: for a real
mirror cost to hide, the mirror endpoint would have to land systematically
*below* the same-structure control, and it lands 0.0053 pp *above* it. Note the
comparator spread rests on two points (already in Caveats) — 0.06 pp is an
estimate of the lottery scale, not a measured distribution. **No overclaim
found**, provided "0 pp" is always read with the stated resolution, as the study
does.

**(b) Twin-optimum contrast 56.7 % vs 41.5 %.** Same definition, same code path,
per-scheme denominators — fair as computed. Two-proportion z ≈ 54 under pair
independence (n = 65,452 vs 62,137); pairs are not independent (block moves),
but the gap is ~50× the independent-pairs SE, so it cannot be a fluctuation.
The reviewer additionally tested the one real confound the study missed:
**within-pair rank displacement** (mirror pairs are ~2× farther apart than C1b:
mean |Δrank| 6,565 vs 3,218, and far pairs could stay swapped merely because
local refinement cannot transport that far). Binned by pooled displacement
decile, the confound goes the *other way*: C1b's stay fraction falls with
displacement (0.49 → 0.11) while the mirror's stays high (0.54–0.69) in every
decile except the very smallest and largest; reweighting C1b to the mirror's
displacement profile *widens* the gap to +27.7 pp (raw +15.2 pp). The raw
numbers therefore **understate** the preferential retention; the twin-optimum
reading survives and is strengthened. (Full table in the review JSON.) One
precision note: the "twin" keeps the exchanged relation on 56.7 % of pairs, not
all — the study's own text states this correctly.

**(c) Above-champion 84.3281 %.** Confirmed through the oracle (35,343,705 =
84.32808288175973 %, +0.0699 pp over 35,314,407). The study explicitly labels it
"an observation, not a champion promotion" and routes it to the campaign gate
ladder; `biology/FINDINGS.md` A5 carries the same "explicitly NOT a champion
promotion" flag. No governance violation: `autoresearch/` untouched, scorer and
eval harness untouched.

### Corrections

1. **§ Numbers, first line** — the parenthetical "(34,751,902 + gate …)" wrongly
   implied the champion's exact score is 34,751,902; that number is the
   Rocket-era scorer parity anchor (82.9161 %). The champion's exact score is
   **35,314,407**. Fixed in place (marked). No other quoted number required
   correction; every other value in the study reproduced bit-exactly.

### Minor notes (no action required)

- `exp_mirror_retune.py --smoke` writes `mirror_pairs.npy` / `c1b_pairs.npy`
  *without* the `_smoke` suffix, silently touching the full run's artifacts
  (byte-identical content since the rebuild is deterministic, but mtimes churn).
- The H64 production driver passes a wall-clock `time_budget_s` to both stages;
  the retune omits it (pure sweep/cycle budgets). This is a deviation from the
  H64 *invocation* (constants match), applied identically to all three starts —
  arguably the right choice for determinism, but worth knowing when comparing
  against the original H64 run.

**status → reviewed with corrections.**
