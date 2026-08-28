# Divergence reasoning — cycle 21 (2026-08-28)

Written BEFORE proposing anything, as divergent mode requires. Trigger:
`cycles_since_score_move = 3` (campaign.yaml `escalation.divergent_after_k_cycles_without_score_move`).
P23 was checked first, for the fifth cycle running, and is still `awaiting-operator`
(HEAD was `4dc75ff`, no ruling on the branch), so H70/H63/H74 stay blocked and the
divergent axis is the unblocked one — exactly as cycle 20's handoff instructed.

## 1. What the campaign already knew, and why it was not enough

Three independent measurements had converged on the same uncomfortable shape:

| measurement | cycle | number |
|---|---|---|
| stage-4 absorbs a deliberate stage-3 perturbation (M14) | 20 | **87.7%** absorbed |
| champion is a strict local max along the ASYM gradient (M13/H66) | 19 | **0.000000 pp** repair headroom |
| every local move class since H42 | 14–20 | **≤ +0.018 pp**, most ≤ 0 |

Cycle 20's own handoff drew the right inference in prose — "the refinement stack is at a
joint fixed point that is robust to how it got there… that is an argument for attacking the
DECOMPOSITION or the FORMULATION" — but nothing had *measured* the formulation-level claim.

## 2. The observation that made this cycle worth spending

**Every structural statement this campaign makes about the connectome gap was measured from
a base it has since left behind by 1.34 pp.**

`experiments/outputs/localsearch_sizing.json` (2026-06-22) is the source of

- meta-rule **M4** ("prefer global-range or structurally decomposed move classes"),
- the **H22** kill (bounded-window sift, declined at the sizing gate),
- the `p25=8,290 / p50=22,580 / p90=87,497` rank-distance percentiles quoted in
  `findings.md` #3 as *the* description of where the gap lives.

All of it was computed on **H02's converged order, 82.9161%**. The champion is **H64 at
84.2582%**. Of the +1.6874 pp net gap those numbers describe, **78.9% has since been
harvested** — by precisely the global/structural move classes M4 recommended. The map was
drawn of terrain the campaign has already crossed, and nobody had ever looked at the
**+0.3565 pp** that is left.

That is not a variant of the current design. It is the question of what the current design
is standing on. Hence H78.

## 3. Re-reading Q01 / Q02 with this in hand

- **Q01** says the continuous relaxation is scale-blind at the operating point and *ranks the
  better order lower* — the surrogate is a different objective at Rocket's `β·std ≈ 148`, and
  the aligned regime is a factor 3.2 away. Consequence for divergence: no continuous lever is
  a candidate, and H38/H64's asymmetric surrogate already took the one exception (objective
  *shape*, not scale). **The continuous axis stays closed.**
- **Q02** says the near-optimal set is **degenerate and flat**: score std 0.0225 pp across
  seeds while Spearman ρ is only 0.958. Many near-equivalent orderings score nearly the same.
  This is the pointer that mattered. If the near-optimal set is flat and wide, then the
  reference solution is plausibly **not "the champion plus some moves"** but a *different
  point in that set* — and the campaign has been implicitly assuming the former for twenty
  cycles.

Q02 was measured seed-to-seed on plain Rocket. Nobody had ever asked the same question of the
pair that actually matters: **champion vs. reference**.

## 4. What was measured (H78, this cycle)

Artifacts: `experiments/outputs/proto_H78_connectome.json`,
`proto_H78b_path_connectome.json`, `proto_H78c_slot_adoption.json`.
Measures 1–2 are imported **verbatim** from `experiments/size_localsearch.py`, so the
champion-era and H02-era numbers differ only in the base order.

**(a) The residual IS shifted shorter — H78's stated prediction, confirmed.**
Gain-weight rank-distance median **22,580 → 10,430**, ratio **0.4619**, far outside the
pre-registered kill band [0.75, 1.25]. The pipeline harvested the long-range component
preferentially, as a full-range sift plus SCC-recursive block refinement should.

*But short-er is not short.* p50 = 10,430 is still 7.6% of n = 136,648, a W=1000 window holds
**18.23%** of total gain and W=5000 only **36.51%**. Bounded-window local search is still not
the answer, and **M4's directive survives** — narrowed, not overturned.

**(b) The real finding: the disagreement is 6.7× the prize.**

    gain +2.3787 pp   lose +2.0222 pp   NET +0.3565 pp

The champion and the reference disagree about the orientation of edges carrying **4.40 pp**
of weight, and **85% of that cancels**. At the H02 era the same ratio was 4.60/1.69 = 2.7:1;
it is now **6.7:1**. 53.6% of nodes carry gain stake; median node displacement is 6,051 ranks.
The two orders are structurally *far apart* while being 0.357 pp apart in score. This is
Q02's flat degenerate set, measured for the first time on the pair that matters.

**(c) There is no incremental path — three probes, one answer.**

- *Blend / path-relinking* (`(1−t)·rank_champ + t·rank_best`): a **valley 0.786 pp deep** at
  t = 0.60. The barrier is **2.2× the 0.357 pp prize**.
- *Teleport top-k by gain stake to reference rank*: loses at **k = 1** (−0.011753 pp) and
  monotonically worse to −4.74 pp at k = 30,000. Recovers only at k = n.
- *Slot-preserving adoption* (subset re-ordered internally to the reference's opinion, **zero
  collateral displacement** of any other node), three selection rules — stake, displacement,
  seeded random: **best proper subset over all rules is exactly +0.000000 pp**, a no-op.
  Every subset that changes anything, loses.

> **Honest correction to our own instrument.** `proto_H78b_path_connectome.json` records
> `barrier_confirmed: false`. That flag is wrong and its rule was mis-specified by me, not by
> the data: it counted t = 0.995 (+0.038146 pp) as an "interior" point. At t = 0.995 the blend
> is the *reference order locally re-sorted within ~200-rank windows* — it scores **0.318 pp
> below the reference**, sits on the reference's side of the valley, and cannot be constructed
> without the reference. It is not a reachable point and not a win. The barrier is real; probe
> (c), which has no such ambiguity, is the load-bearing evidence.

## 5. What this licenses, and what it does not

**Licensed.** The reference solution's advantage is **not decomposable**. It is a property of
the whole permutation. From the champion's basin, no monotone process — no move class, however
clever, that only accepts non-worsening steps — reaches it, because every partial adoption of
it loses and the natural path between them runs through a valley twice as deep as the prize.
This explains, in one mechanism, M14's 87.7% absorption, H66's 0.000000 pp headroom, H31's
null, and the ≤ +0.018 pp ceiling on every move class since H42.

**Not licensed.** Two geometric families of partial adoption were tested, not all of them. A
valley along the linear-rank-blend path does not prove no path exists. The correct claim is
evidential, not a proof: *the two natural ways of walking from here to there both fail, and no
subset of the destination is worth importing.* Also connectome-only — no reference solution
exists for microns or mouse, so this cannot be replicated there (the standing scope limit on
all gap work, `findings.md` #3).

## 6. Where that points

Not at the move class. Two directions survive this measurement, and they are the two divergent
levels CAMPAIGN.md § Escalation names:

1. **A different basin** (decomposition / construction). The champion's basin is the one
   greedy-FAS + Rocket puts us in; H48 measured a start **+5.70 pp better** that still lost
   after refinement (M7), so "better start" alone is dead — but *structurally different* start
   is not the same claim, and it has never been tested against the current stack.
2. **A method that accepts uphill moves at ~0.8 pp scale.** Everything the campaign has run is
   monotone or near-monotone. H31 (random ruin-and-recreate) is the only metaheuristic tried
   and it was *generic* and killed at ~1.9× wall for −0.0008 pp; its revival condition is
   explicitly "the destroy operator is STRUCTURE-AWARE and the rebuild is exact-gain"
   (= queue item H40). The barrier depth now gives that class a *sizing number it never had*:
   any acceptance schedule must admit ≈0.79 pp of temporary loss, which is 66× the 0.012 pp
   promotion bar and rules out timid annealing.

Both are filed as queue items this cycle (H79, H80). Neither is proposed as this cycle's
variant — the cycle spent itself measuring, which is what divergent mode is for, and the
measurement changed what the next variant should be.

## 7. Literature status

`cycles_since_literature_scan = 2`; the last full scan is `scan_cycle18.md` (2026-08-27,
30 KB) and its yield (H71/H72/H73) is not exhausted — H73 was run and killed this week, H71
and H72 remain queued. A fresh broad scan two cycles later would re-till the same ground.
What this cycle's result actually calls for is a **targeted** scan on one question — *large
neighbourhood / crossover / path-relinking methods for linear ordering that cross barriers of
known depth* — which is filed as **L02** rather than run here, because the barrier number that
makes the scan answerable was only produced at the end of this cycle.
