# Q&A: hemispheres in the order, and how robust the conclusions are

*2026-09-14. Four questions asked of the fly-streams results, answered in
order; every answer was checked by an independent reviewer before being
committed (review notes at the end of each answer). Numbers computed from the
same pipeline artifacts as everything else (`outputs/fly_ann/node_ann.npz`,
`outputs/fly_artifact/artifact_data.json`, the pinned H64 order); one-off
computations are quoted with the command in the session log, experiments have
their own scripts under `tools/`.*

---

## Q1. Why does the left hemisphere always come first, and then the right?

**Short answer: it doesn't — "left first" is a human convention in our labels
and legends, not a fact about the data.** Where you see "left … right" in
captions, legends and tables, that is just the order we (and most people)
write the two words in.

Whether the *optimizer* puts left neurons earlier is a separate, checkable
question — so we checked:

- Aggregate: the median champion rank of all left neurons is 65,770 vs 70,376
  for right — a ~4,600-rank (3.4 % of the order) difference of medians.
  Counts are almost equal (68,462 left / 67,995 right).
- But per cell type the effect dissolves: among the 267 types with ≥20
  neurons on each side, the left copy comes earlier in only **56.6 %** of
  types — barely better than a coin flip — with a median offset of just
  −430 ranks (0.3 % of the order) and a spread that straddles zero
  (IQR −3,062 … +1,921).

**Reading.** There is no "left goes first" rule inside the order. The two
hemispheres are near-interchangeable for the objective, so the optimizer has
to break the tie *somehow*; the small aggregate lean is likely tie-breaking
drift plus a composition effect of which types sit where in the order — a
leave-out test found no small set of populations that accounts for it, and
it is not a consistent per-type policy. Whether the lean is load-bearing at
all is exactly what the mirror-swap experiment (Q3) measures.

**Caveat.** "Left/right" here is the annotation's side label (soma side); the
191 midline/unlabeled neurons (157 "center", 21 "na", 13 unannotated) are
excluded from the comparison.

*Reviewed: all numeric claims recomputed and confirmed by an independent
check; two wording fixes applied (the excluded-neuron count, and the
attribution of the aggregate lean, which the reviewer's leave-out test
showed to be unsupported).*

---

## Q2. Why is one hemisphere usually on top inside a lane's band — and why are some lanes mixed?

**Short answer: the vertical order inside a band is the rank of ONE global
"graph-affinity" coordinate; where the wiring separates the hemispheres, that
coordinate gives every left neuron one sign and every right neuron the other
— so the same side lands on top in every such lane. Which side that is (here:
right) is an arbitrary sign choice, and mixed lanes are lanes whose neurons
genuinely wire across the midline.**

First, the measurement (share of "right" neurons in the top half of each
band, by the within-lane spectral rank):

| lane | % right in top half | % right in bottom half |
|---|---|---|
| lobula (plate) | 99.9 | 0.3 |
| medulla | 99.3 | 1.1 |
| medulla→lobula | 98.7 | 0.1 |
| visual projection | 97.2 | 3.1 |
| lamina | 96.3 | 1.2 |
| photoreceptors | 91.4 | 6.0 |
| mushroom body | 57.1 | 43.2 |
| central (other) | 52.0 | 47.7 |
| central complex | 49.8 | 50.2 |
| descending | 50.0 | 49.5 |

(The table is a curated subset of the 17 lanes; among the omitted ones,
visual centrifugal is 68/31 and the 205-neuron "optic (other)" is the one
lane that weakly leans the other way. And one correction to the intuition in
the question: in the current figures the top half is **right** (blue), not
left — consistently so in every major optic lane in the table.)

**Why the same side is on top everywhere.** The y-offset inside a band is not
computed per lane: it is the within-lane *rank* of one brain-wide spectral
coordinate (§4 of FLY_FINDINGS, computed from wiring alone). Left-optic
neurons connect almost exclusively to left-optic neurons, so the whole left
optic block gets coherently negative-or-positive values, and the right block
the opposite — one global sign decision that every optic lane inherits.
Hence the same side tops every band.

**Why "right on top" specifically means nothing.** An eigenvector is defined
up to sign: flipping every value's sign is the same mathematical object. Our
sign is fixed only by the random seed (42) of the power iteration — with a
different seed the pink and blue halves could trade places, and no number in
the artifact would change (the two columns of the table above would simply
swap).

**Why some lanes are mixed.** Central-brain lanes (central complex,
descending, central (other)) sit at ~50/50 in both halves: their neurons
receive from and project to both hemispheres, so graph affinity simply does
not sort them by side — which is itself the finding (§4: integration happens
in the center). The mushroom body's 57/43 is a different story, and a
subtler one: it is *not* a diluted optic-style split — Kenyon cells alone
give the same 58/42, and the 420 DANs/MBONs (7.5 % of the lane, themselves
sided left/right) cannot account for it. The **first** spectral coordinate
simply carries little hemisphere signal for the mushroom body; its
left/right split is real (only ~17 % of MB-internal weight crosses the
midline) but lives on the **second** spectral coordinate, which the cloud's
y does not use. The visible pink/blue Kenyon-cell blocks in the cloud sit
side by side **along x** — the champion order grouped left KCs (median rank
100,875) before right KCs (104,045) — not stacked in the band's vertical.

**Caveat.** "Top half" is measured on the spectral rank, i.e. ordinal
position, not distance; and the per-lane numbers above are a direct
computation, not a statistical test.

*Reviewed: the table, the rendering-direction claim (both PNG and web page)
and the sign argument were independently recomputed and confirmed — including
that only 2.95 % of within-optic-lane weight crosses the midline. The
original mushroom-body explanation was found factually wrong by the
reviewer's Kenyon-cell-only check and replaced with the corrected mechanism
above.*

---

## Q3. Swap the mirror twins in the best solution — how much does the score drop?

**Short answer: reflecting 95.8 % of the brain into its mirror image costs
2.21 points of the ~34 that separate the champion from a random order — and
no more than swapping the same neurons within their own hemisphere. The two
halves are, to a first approximation, computationally interchangeable.**

*Experiment: `tools/exp_mirror_swap.py`, results
`outputs/qa/exp_mirror_swap.json`, seed 42, scored by the frozen oracle.*

**Setup.** Within every cell type, the left members and the right members
were each sorted by champion rank and paired i-th ↔ i-th; each pair then
exchanged ranks. This pairs **65,452 couples across 8,644 types — 95.8 % of
all neurons**; unpaired neurons keep their rank. Two same-side controls with
the same pairing structure calibrate the result, and a random-pair swap is
the destruction floor.

| swap | pairs | FF after | Δ (pp) | mean rank displacement |
|---|---|---|---|---|
| **mirror (left ↔ right twin)** | 65,452 | **82.05 %** | **−2.21** | 6,565 |
| same side, same type, rank-matched | 62,137 | 82.72 % | −1.54 | 3,218 |
| same side, same type, random pair | 62,137 | 81.25 % | −3.01 | 10,992 |
| random pairs (floor) | 65,452 | 50.85 % | −33.41 | 45,442 |

**Reading.**
- Swapping 95.8 % of the brain into its mirror image costs **2.21 points of
  the 34.2 that separate the champion from chance** (chance = the ≈50.05 %
  a random order scores, the repo's Simple baseline; using this experiment's
  own random-swap floor of 50.85 % instead gives 93.4 %) — the swapped brain
  keeps **93.5 %** of the champion's above-chance orientation.
- The mirror swap's cost sits **between** the two same-side controls, about
  where its rank displacement (6.6k, between 3.2k and 11k) predicts. So we
  find **no detectable extra cost of crossing the midline**: a neuron's
  mirror twin is about as good a stand-in as a same-type neighbor on its own
  side at a comparable rank distance. Most of the 2.21 pp is the generic
  price of moving fine-tuned neurons, not asymmetry. (This reads a
  cost-vs-displacement curve off two points: the relation is strongly
  non-linear at large displacements, the controls swap ~5 % fewer pairs, and
  each control is a single seed-42 draw — a bracket, not a fit.)
- Where the loss localizes (top single entries of the lane-pair
  decomposition): the largest is **central (other) → central (other)**
  (−0.56 pp of total weight), with within-optic pairs next (medulla→medulla
  −0.20, medulla→lobula internal −0.11). Relative to how much weight each
  pair carries, the central brain and the optic lobes lose comparably — the
  "periphery symmetric, center asymmetric" hypothesis from the earlier
  discussion is *not* cleanly confirmed at this granularity.
- Back to Q1: the small left-early lean survives this test as inert — the
  order works almost as well with the hemispheres exchanged wholesale.

**Caveats.** The pairing is rank-order matching within type, not
morphological twin matching (mirrored soma positions would be finer); with
count-mismatched types the leftover neurons never move. The score after a
swap is a *feasible* value, so 2.21 pp is an upper bound on what a re-tuned
mirror order would lose — local re-optimization could claw some of it back.
And per §11.7, this drop mixes true wiring asymmetry with pairing error and
fine-tuning; it is an index, not a pure symmetry measure.

*Reviewed: every number reproduced bit-exactly through the frozen oracle by
an independent rerun (pairing, swaps, disjointness, and the full lane-delta
matrix, which sums to −2.2129 pp exactly); the chance convention behind
93.5 % was made explicit and the bracketing argument softened per review.*

---

## Q4. What if we stop early — at 80 % or 70 % feedforward? Do the conclusions change?

*Experiment: `tools/exp_degraded_order.py`, results
`outputs/qa/exp_degraded_order.json`, seed 42. Degradation model: random
2,000-rank windows of the champion order are repeatedly shuffled internally
until the oracle score falls to the target — a stated *model* of "stopped
early" (coarse structure right, local detail wrong), not a rerun of the
optimizer. A fully random order is the floor.*

| metric | champion 84.26 % | degraded 79.65 % | degraded 69.99 % | random 49.98 % |
|---|---|---|---|---|
| lane-median Spearman vs champion | 1.000 | **1.000** | **0.995** | 0.375 |
| staircase inversions (of 8 steps) | 0 | **0** | **0** | 3 |
| flow hierarchy ρ | 0.263 | **0.263** | 0.247 | 0.004 |
| ACh / GABA / Glu / 5-HT backward share | 11.4 / 20.9 / 22.2 / 28.8 | 15.6 / 25.2 / 28.2 / 32.7 | 26.2 / 33.3 / 37.6 / 39.8 | ~50 all |
| type-block tightness (IQR/n, median) | 0.049 | 0.052 | 0.131 | 0.497 |

**Short answer: the qualitative conclusions survive easily; the quantitative
values decay in one predictable direction — toward "no structure". So yes, a
suboptimal order is usable, if you treat magnitudes as attenuated.**

**Reading.**
- At 80 % *every headline conclusion is numerically indistinguishable* from
  the champion's: the lane staircase is identical (Spearman 1.000, zero
  inversions), the hierarchy test gives the same ρ, and the transmitter
  ordering ACh < GABA < Glu < serotonin is untouched.
- At 70 % the *shape* of every conclusion still holds — staircase intact,
  ρ 0.247, the same transmitter ordering — but the *numbers* have drifted
  toward the chance value of 50 %: all backward shares inflate (ACh 11→26 %),
  and type blocks are 2.7× looser. This is classic attenuation: a noisier
  order misclassifies some forward edges as backward and vice versa,
  compressing every direction-based statistic toward "random".
- The bias direction is the safe one: a worse order makes the brain look
  *less* structured, never more. A conclusion of the form "X flows backward
  more than Y" or "stage A precedes stage B" reached on a suboptimal order
  would only get *stronger* with a better one. What you may NOT quote from a
  weak order is any absolute number — above all the feedback ceiling
  (a 70 % order proves only "≤ 30 % unavoidable", a far looser bound).
- A side observation worth keeping: reaching 70 % took 345 shuffle batches
  where 80 % took 4 (~86×; the 70 % run is a fresh descent from the champion,
  re-crossing 80 % around batch 4) — with all structure
  finer than 2,000 ranks destroyed, the coarse arrangement alone still holds
  ~70 % of weight forward. Most of the score lives in the coarse order;
  most of the *last few points* live in the local detail.

**Caveats.** The degradation model shuffles locally; a real early-stopped
optimizer errs differently (typically coarse structure converges first
too, but this was not re-run here — checking an actual short Rocket run is
the obvious follow-up). "Chance = 50 %" refers to a uniformly random order
on this graph (measured 49.98 %). The lane-median Spearman is computed on
just 17 lane medians — at n = 17 the null standard deviation is ≈ 0.25, so
the random column's 0.375 is within noise of zero and should not be read as
residual lane structure. All metrics recomputed with the same code paths as
the main pipeline; one seed per order (the degradation is a single draw, not
an ensemble).

*Reviewed: an independent end-to-end re-run reproduced every table cell and
in-text number bit-identically; the metric definitions were verified to match
the main pipeline exactly (including staircase fairness under the
champion-derived lane ordering); the n = 17 caveat above was added at the
reviewer's request.*

---

*Bottom line across Q3–Q4: the representation's conclusions are robust both
to reflecting the brain (2.2 pp) and to substantial suboptimality (all
qualitative claims intact at 70 %); the numbers to guard are the absolute
ones, which always require the best available order.*
