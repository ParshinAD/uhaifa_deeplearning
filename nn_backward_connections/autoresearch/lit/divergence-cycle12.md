# Divergence reasoning — cycle 12 (2026-08-25)

Written **before** any run of this cycle, as CAMPAIGN.md § Escalation requires.

## Why divergent mode is active

`state.json.cycles_since_score_move = 3`, against
`campaign.yaml: escalation.divergent_after_k_cycles_without_score_move: 3`. The last champion
change was cycle 8 (H52, mouse only). Cycles 9 (P09, iterate), 10 (H56, kill) and 11 (H57, kill)
moved no score. `consecutive_kills = 2`, so the *secondary* trigger has not fired — the primary
one has, and it is the one that matters.

Divergent mode forbids two things: another variant of the current design, and satisfying the
requirement with an infrastructure item. Both exclusions bite this cycle, so they are applied
explicitly below rather than left implicit.

## What the last two cycles closed

M9 and M10 together close the **multi-start / relabelling family** on arithmetic, not on taste:

- M9: best-of-R over R draws gains `sigma * a_R`, and R is not free — it is
  `floor(deadline / per-run wall clock)`. On connectome the champion admits R = 2, on microns
  R = 3. So a multi-start idea needs `sigma >= min_promotion_delta_pp / a_R`.
- M10: the relabelling dispersion is ~96% **prefix** (greedy-FAS tie-breaks -> Rocket) and ~4%
  **tail** (refinement). Measured: whole-pipeline sigma 0.019124 pp (n=5) vs tail-only 0.003731 pp
  (n=8). So sharing the prefix multiplies R by ~2 while dividing sigma by ~5 — it cannot pay.

The only door M9/M10 leave open is a mechanism that **widens the prefix's sigma past 0.021269 pp
at the champion's per-arm cost**. Nothing in the queue does that, and this cycle does not try:
that is still the same design, one level down.

## The five proposed science items, judged against divergence

Priority order is H54 (p3), H58 (p3), H39 (p4), H47 (p5), H40 (p6). Priority alone does not decide
a divergent cycle; the level attacked does.

| item | level attacked | divergent? | disposition this cycle |
|---|---|---|---|
| H54 | a schedule constant inside the existing refiner | **no** | deferred, not dropped |
| H58 | node labelling / tie-break structure | **no** | deferred, not dropped |
| H39 | continuous <-> discrete hybrid | nominally yes | **dropped by novelty** (below) |
| H47 | the decomposition's terminal case | **yes** | **TAKEN** |
| H40 | destroy-repair move class | yes | deferred (medium cost, no cheap gate yet) |

**H54 — deferred.** It flips one hand-tuned constant (`k_full = 6`) to a detector inside a refiner
the champion already ships. That is the definition of a variant of the current design. It stays
`proposed` at p3 and is a good first item for the next *incremental* cycle; it is not a defensible
answer to "we have stopped making progress".

**H58 — deferred.** It is adjacent to the family M9/M10 just closed (starting basin via labelling),
and cycle 11 amended the item itself to record that the mechanism which best explains the canonical
labelling's rank — tie-break **consistency** between prefix and tail, not a special on-disk order —
**predicts that H58's 1x-cost win does not exist**. Its premise test alone costs ~1.9 h of GPU to
move a rank statistic from p = 0.4 (n = 5) to something readable at n >= 11. Spending a divergent
cycle's entire GPU budget to firm up the premise of an item whose own record predicts failure is
the wrong trade. Left `proposed`, with the amendment standing.

**H39 — dropped by novelty.** This is the one disposition that changes an item's status, so it is
argued in full. H39 shares an axis with **M1-continuous-exhausted**, whose revival condition is
explicit: *"Any new continuous-only hypothesis must explain why it escapes the scale-blindness of
the surrogate (diagnosis.md Q01)."* H39's stated escape is that Q01 showed the optimum is
"HOLDABLE at its own scale", so pinning the gradient phase above the Q01 crossover (beta*std ~ 470)
should let a short continuous phase kick the sift off its fixed point without destroying it.

Q01 as it stands today **retracts exactly that claim**. Its "What this does NOT establish" section
says, verbatim: *"Nothing about the stability of the optimum. A 'hold' measured at very large scale
is a frozen optimizer, not stability: at std ~ 5e4 over 99.99% of edges have |beta*Delta| > 37,
sigma' underflows, and every configuration holds — Rocket's own 82.92% order holds just as exactly
as the 84.61% one."* The H39 queue entry was written before that section existed; it is quoting a
superseded reading of its own primary artifact.

The vice is then closed on both sides by Q01 insight 3: below the crossover the surrogate does not
rank the better order higher (measured net −174.88 at the operating point), and above it sigma
saturates so the gradient vanishes and nothing moves. There is no scale at which the phase both
prefers the better order **and** can travel toward it. H39's own kill condition — *"If no scale
above the crossover produces a post-re-sift gain over the sift fixed point, the hybrid direction is
closed"* — is therefore already answered by a logged artifact
(`experiments/outputs/q01_surrogate_ranking.json`, `q01_drift.json`), at zero additional cost.

Status -> `dropped-by-novelty`. **Revival condition** recorded with it: a continuous phase that is
NOT a member of the family `sum_e w_e * g(Delta_e)` — Q01's shape comparison showed all four
shapes (sigmoid, slow tanh, hard clip, cusp) agree at both scales, so changing `g` moves along the
trade-off rather than escaping it. Q05's one-sided asymmetric surrogate is such a non-member and
is aligned at every scale; if Q05's degeneracy is removed, H39's question may be re-asked in that
form. This is a drop of the *stated mechanism*, not of the hybrid direction as a whole.

## The item taken: H47

**Level attacked: the decomposition.** `SccRecursiveRefiner._refine` (src/mfas/refine/scc_recursive.py:212)
opens with `if nb <= self.min_block or eidx.size == 0: return`, and `min_block = 32`. The block
refiner also preserves each SCC's internal relative order by construction (:248). So the champion's
structural stage **optimises nothing at all below 32 nodes** — not badly, but literally not at all.
H47 makes that base case exact by Held-Karp subset DP,
`f[S] = max over j in S of ( f[S minus j] + sum over i in S minus j of w(i->j) )`, giving the exact
optimum internal order of a k-node block in O(2^k k^2) instead of O(k! k).

**Why this is not covered by M4.** M4's proven half is bounded-window **single-node** local search
for W <= 100. H47 is not a rank window and not single-node: it is the terminal case of a
structurally-derived recursion, and it optimises over all k! orderings of a block jointly. M4's own
standing directive is *"prefer global-range or structurally decomposed (SCC / block) move classes"*,
which this satisfies rather than violates. A single-node sift can sit at a fixed point while a
16-node block is far from its joint optimum — that gap is precisely what the DP measures.

**Why it is cheap, which is what makes it the right divergent item.** The prototype needs **zero
GPU**. Disjoint contiguous windows compose additively: permuting the nodes inside a contiguous
position range cannot change the orientation of any edge with an endpoint outside it, so the summed
exact gain over a tiling **is** the realised delta, verifiable against the frozen oracle. The
champion's own final orders are already on disk as tracked evidence
(`experiments/evidence/20260810T105922Z-H42-connectome-s31415-confirm-f41d7e_positions.npy` and the
matching microns file), so no prefix has to be recomputed.

**Honest prior, recorded before the measurement.** M4 is the strongest argument against this item,
and the H47 entry says so itself: the champion's sift is already single-node optimal, which removes
most of what a small window can hold. The expectation is that this dies at its gate. It is taken
anyway because the gate is minutes of CPU and the number it produces — how much of the residual
0.4606 pp gap is intra-block at all — is a diagnosis the campaign does not currently own, and it
bounds a whole family of block-exactness ideas in one measurement.

**Kill condition (H47's own, unchanged):** total exact gain below 0.012 pp at every k in {8, 12, 16}.

## Literature

A scout scan was launched at the start of this cycle (divergent mode step 1) even though
`cycles_since_literature_scan = 3` does not yet force one. Its output goes to
`autoresearch/lit/scan-cycle12.md` and does not gate H47, which was chosen from the existing L01
scan (`lit/hypotheses.md` rank 3, `lit/notes-vahidi-flywire.md` §§ 3.3 and 3.5: Vahidi 2025
Algorithms 3 and 5 both exhaustively permute sub-SCCs of size <= 9, where this campaign does
nothing below 32).
