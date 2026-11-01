# Independent review — null-ladder v2, Stage A (charter + generators + G-doc sync)

**Reviewer:** independent subagent, 2026-11-01 · **scope:** `CHARTER.md`,
`biology/analysis/nullmodels2_generators.py`, the Stage-A edits to
`experiments/randomgraph.md`. Read-only on all source and v1 files; this file is the
review's only write.

**Instruments:** reviewer-written property tests (numpy, CPU only; mouse dataset +
synthetic graphs; the connectome was NOT loaded — the GPU/W0 sweep was running), script at
`/private/tmp/claude-501/-Users-abed359-IdeaProjects-university-deeplearning-thesis-nn-backward-connections/830df5a2-af10-4cbe-8b63-7cb75e47ee21/scratchpad/review_tests.py`
(scratchpad — not a repo artifact; findings reproduced in full below). All invariant
checks re-implemented independently (python-dict/fsum 2-cycle bound; set-based dyad
classification; exact int64 reference scorer); the module's own `_validate_*` helpers
were not reused.

---

## A. Numbers in CHARTER.md §0 vs `biology/studies/nullmodels.md`

| claim (charter) | source (nullmodels.md / other) | verdict |
|---|---|---|
| L1 `gnm_null` 69.823 ± 0.019 | §3 table: 69.823 ± 0.019 | CONFIRMED |
| L2 `w_shuffled` 77.220 ± 0.024 | §3: 77.220 ± 0.024 | CONFIRMED |
| L2 `w_source` 79.981 ± 0.015 | §3: 79.981 ± 0.015 | CONFIRMED |
| L3 dc-SBM 81.896 ± 0.018 | §3: 81.896 ± 0.018 | CONFIRMED |
| real (protocol) 83.658 ± 0.003 | §3: 83.658 ± 0.003 (3 seeds) | CONFIRMED |
| real (champion, H64) 84.258; operator's "84.26%" | §3: 84.258; `autoresearch/sota.json` pct_mean 84.2582 / exact 84.25817950936937 | CONFIRMED |
| "outside all **76** null draws" | K = 12+12+20+20+12 = 76 (L1, L3a, two L2 variants, L3); review row "outside all 76 nulls ✓" | CONFIRMED |
| decomposition +19.8 / +10.2 / +1.9 / +1.8 pp | §3: +19.82 / +10.16 / +1.92 / +1.76 (reviewer: 19.823/10.158/1.915/1.762) — rounding consistent | CONFIRMED |
| 2-cycle core 6.717% of W, "18–930× more than any of its nulls" | §3: 6.717 (review-exact 6.7172); corrected ratio range 18–930× (17.8–932×) | CONFIRMED |
| L2R text: real bound "6.7172% of W" | v1 review recomputation: 6.7172% | CONFIRMED |
| "reciprocal core 61–64× larger than its L2 nulls'" | v1 review exact ratios 61.3–64.0× (L2) | CONFIRMED |
| mouse: "real below 98/100 strength-matched nulls" | §4: 2/100 beyond, p = 0.030; "below 98/100" | CONFIRMED |
| "689 logged runs, review 2026-09-30 with 4 minor corrections, all load-bearing numbers confirmed" | header + review verdict ("4 in-text corrections applied, none touching a headline quantity") | CONFIRMED |
| estimator recap (greedy-FAS → Rocket-ASYM fly 4,000 epochs → sift 8 sweeps, α=0.7, k_full=2; oracle on CPU) | §2 items 1–4 verbatim | CONFIRMED |
| Stage E: MICrONS 67,534 neurons, ~10.4M edges; "249/67,534 axons proofread" | `src/mfas/io.py` expected_n=67,534, expected_m=10,436,569; `layering/MICRONS_FINDINGS.md`: "Only 249 of 67,534 neurons … have a proofread axon" | CONFIRMED |

**No numeric mismatch found anywhere in §0 or §1.**

## B. The mathematical claim behind the operator-step-1 mapping

**Claim 1 — permuting a column preserves every node's in/out degree.** Trivially true
(a permutation preserves the column's multiset; out-degrees are the src-column multiset,
in-degrees the tgt-column multiset) and verified numerically (T3a: bincount equality on a
5,000-edge random graph under independent src- and tgt-permutations). So the operator's
literal step 1 is indeed already a configuration-model draw. CONFIRMED.

**Claim 2 — both-columns vs target-only.** The two are **not the same map but ARE the
same ensemble**, with one crucial caveat the charter gets exactly right:

* Unweighted: for independent uniform σ (src) and π (tgt), re-indexing slots by j = σ(i)
  gives edges (src_j, tgt_{π∘σ⁻¹(j)}); π∘σ⁻¹ is uniform, so the edge-multiset
  distribution is identical to permuting the target column alone.
* Weighted, "weights stay on rows": the same re-indexing puts weight w_{σ⁻¹(j)} on slot j,
  and (π∘σ⁻¹, σ⁻¹) are **independent** uniform permutations
  (P(π∘σ⁻¹=a, σ⁻¹=b) = P(σ=b⁻¹, π=a∘b⁻¹) = (1/m!)²). So permuting both columns with
  weights fixed on rows ≡ permuting the target column + **independently permuting the
  weights** — i.e. exactly v1's `w_shuffled` ensemble, *not* `w_source`. The weights get
  decoupled from the sources even though they "stay on rows", because the rows' sources move.
* Empirical corroboration (T3b): on a 4-edge loop-proof graph, 400k draws per method,
  the two weighted-graph distributions agree on all 144 support points to within
  5.5·10⁻⁴ (5σ band 9.3·10⁻⁴).

**Consequence for the mapping table: the charter's assignment of operator step 1 to L2
`w_shuffled` (77.220), not `w_source` (79.981), is correct — and it is load-bearing**
(the two rungs differ by 2.76 pp). The only residual difference between the operator's
literal step and v1's implementation is the identical self-loop-repair conditioning
applied in both framings; it does not distinguish them. CONFIRMED.

## C. Code review — `biology/analysis/nullmodels2_generators.py`

### C1. `two_cycle_bound` (lines 67–87)

| aspect | finding | verdict |
|---|---|---|
| aggregation + reverse match + halving | correct: unique-key aggregation handles parallel slots; each reciprocated ordered pair contributes min(W_uv, W_vu) once per direction, halved. Brute-force parity (python dict + fsum, reviewer implementation) on 200 random loop-free graphs *with parallel slots*: exact match in all 200 (T1); 60k-edge subgraph: exact (T2b). | CONFIRMED |
| overflow / float64 exactness | int64 keys: n² ≈ 1.87·10¹⁰ for the fly < 2⁶³ ✓. float64: weights ≤ 2405 are exact; aggregated sums ≤ W = 41,912,141 ≪ 2⁵³; verified against a reviewer-written pure-int64 implementation on a synthetic at **beyond-fly scale** (m = 2.3M, heavy reciprocity, total W = 2,767,378,403, bound 241,250,078): bit-exact (T2). The docstring's exactness claim holds with large margin. For the mouse (float weights) exactness is not claimed and not needed: the decisive L2R guarantee reduces to identical pair/value multisets (see C3). | CONFIRMED |
| **latent defect: self-loops counted at half weight** | a self-loop key is its own reverse → contributes w/2, but a loop is *fully* unavoidable feedback. Verified: graph with loops of weight 7+1 → module adds 4.0 not 8.0 (T1b). Unreachable on current data (both real graphs loop-free — mouse re-verified simple+loop-free in T4a, fly validated in v1; all nulls loop-free by repair/construction), but the function is a general-purpose bound: add a guard or a docstring line. | PROBLEM (latent, zero current impact) |
| `_in_sorted` crash on empty key array | line 63: `sorted_keys[pos]` with `pos = min(·, -1)` raises IndexError when `sorted_keys` is empty and queries are not. Triggers `dyad_fixedmut_null` crash on any graph with **zero mutual edges** (verified, T6a/T6b). Impossible on fly/mouse/MICrONS; one-line guard recommended before the generators are reused on arbitrary graphs (the random-graph track makes that likely). | PROBLEM (robustness nit) |

### C2. `w_topo_shuffle_null` (W0)

Correct and near-trivially so: topology arrays returned as untouched copies, one uniform
`default_rng(seed)` permutation of the weight column; multiset/total preserved by
construction (re-verified independently on mouse, T4b). Stated distribution (uniform over
the m!/∏mult! weight assignments) is exact — no repair step exists to bias it. Seed
reproducibility is v1-convention `default_rng`. No failure mode found against its stated
guarantees. CONFIRMED.

**But its selftest is fragile:** `_validate_w0` line 200 asserts `w.sum() == w2.sum()`
**bitwise** — for float weights np.sum over a permuted array is not order-invariant.
Measured on the real mouse weight vector: **178/500 random permutations give a
bitwise-different `np.sum()`** (T4c). The selftest passed on mouse with seeds 7/8 by
luck (~64% per draw). Fail-safe direction (a spurious *failure*, never a false pass of a
real violation — the multiset assert on line 199 is the real guarantee), and MICrONS
weights are int64 so Stage E is safe, but replace with `math.fsum` equality or drop the
redundant sum assert before adding any float-weight dataset. PROBLEM (fail-safe, fix cheap).

### C3. `dyad_fixedmut_null` (L2R) — the decisive claims

**Sufficiency of the three repair constraints for bit-exact 2-cycle-bound preservation.**
Proven and verified. Argument: (i) mutual slots are returned untouched (topology and
weights), so the reciprocated-pair set of the real graph survives with identical
aggregated weights — no uni slot can land ON a mutual ordered pair (the `on_mut` check
covers *both* fwd and rev), so no uni weight contaminates a mutual key's aggregate;
(ii) a repaired uni slot has no reverse among mutual keys (`on_mut` rev side) and no
reverse among current uni keys (`new_mutual`), so uni slots contribute **zero** to the
null's bound — exactly as real uni slots contribute zero to the real bound (by definition
of `mutual_slot_mask`); (iii) duplicate uni slots (parallel edges) aggregate only with
other uni slots whose reverse still doesn't exist → still zero contribution. Hence both
bounds reduce to the identical mutual-subgraph computation over the identical sorted key
set with identical values → bit-identical float64 result (same values, same order, same
pairwise-summation tree). Verified independently on mouse at 4 seeds: reviewer fsum bound
real = null = 0.515373… **exactly**, module bound real = null exactly, zero violating uni
slots under set-based reclassification (T4). CONFIRMED.

**Stale-snapshot swap logic.** The loop cannot terminate with an accepted-but-unrechecked
violation: `bad_uni_mask()` is recomputed **from the full current state** (it rebuilds
`uni_sorted` from the live `new_tgt` internally, line 148) and the loop exits only when it
is empty, else raises after `max_repair_rounds` (honest failure, line 178–181). Within a
round, `slot_ok` checks candidates against a round-start snapshot, so (a) keys created
mid-round can be missed → a swap can *create* a violation (e.g. two self-loop slots
swapped into a fresh 2-cycle — constructed by the reviewer as a concrete scenario), but it
is caught by the end-of-round recomputation and queued for the next round; (b) stale
not-yet-removed keys can only cause false *rejections* (conservative). So staleness
affects convergence speed only, never correctness. Stress test: 300 small high-reciprocity
synthetics (~50% reciprocated, deduped, mixed densities) — every converged draw passes all
reviewer invariants; 5/300 pathologically tight graphs raise the honest non-convergence
error rather than returning a bad graph (T5; max observed repair_rounds 89,
max n_bad_initial 21). CONFIRMED (fail-safe).

**Per-node preservation list** (total in/out degree, uni in/out degree, mutual degree,
out-strength): follows from construction — src column never modified (out-degrees,
out-strengths with weights-on-slot); uni target multiset preserved by permutation+swaps
(in-degrees); mutual slots pointwise fixed and the repair guarantees every uni slot stays
uni (reverse never exists) so the slot classification — hence per-node uni/mutual split —
is invariant. All re-verified independently on mouse via sorted-array and set-based checks
(T4). The charter correctly does NOT claim in-strength preservation (w_source convention).
CONFIRMED.

**Two precision issues (charter wording, not code):**

1. **"dyad census … preserved up to a reported residual" misstates where the residual
   lives.** The code enforces "no new mutual dyads" **exactly** (converge-or-raise; the
   charter's "(up to the reported residual) no new mutual dyads" undersells its own
   implementation). The actual residual is **duplicate parallel uni slots**
   (`n_duplicate_slots`; mouse: 29–35 of 417), which reduce the count of *distinct*
   asymmetric dyads while keeping every per-node degree and the mutual census exact.
   Suggested rewording: "mutual dyad census exact; asymmetric dyads preserved as slots,
   with parallel-slot collapse reported as `n_duplicate_slots` (score-invariant, as
   established in v1)." CORRECTED (wording only; the guarantee as implemented is
   *stronger* than registered).
2. **"Uniform over digraphs with (i)–(ii) conditioned on …" is approximate, not exact.**
   Swap-repair after a uniform permutation is not an exact sampler of the conditional
   uniform distribution (same convention as v1's self-loop repair — but v1 repaired
   188/5.66M ≈ 0.003% of slots, whereas L2R on the **mouse** repairs 29–40 of 417 uni
   slots ≈ **7–10%**, four orders of magnitude more conditioning). On the fly the
   violating fraction should be far smaller (to be read off `n_bad_initial`/`n_uni_slots`
   in the Stage C run JSONs — the stats dict already logs both, good). Add one sentence
   ("approximately uniform; repair fraction reported per draw") before Stage C numbers are
   quoted, and treat the mouse arm of L2R (if ever run) with care. CORRECTED (wording).

**Cosmetic:** `n_bad_repaired` (line 188) is defined as `n_initial` — violations created
and fixed mid-repair are not counted; rename or drop (misleading stat, no numeric use).

### C4. Pre-registered prediction sets — coherence

* **L2R:** coherent, with the right joint-reading rule. Because the null's 2-cycle floor
  equals the real one **bit-exactly**, the primary contrast real − L2R compares FF% on
  equal floors — every outcome of it is interpretable. The secondary contrast
  L2R − w_source mixes two effects (floor difference ≈ 6.717 − 0.105…0.110 ≈ 6.61 pp max
  attributable to the forced core, vs. orderability of the residual wiring); the charter's
  sign-check paragraph acknowledges exactly this. Recommendation: pre-specify the
  commensurable statistic "FB in excess of own 2-cycle floor" (FB − bound, per graph) for
  cross-rung comparisons before Stage C/D analysis, so the joint reading is mechanical
  rather than narrative.
* **Gap in both prediction sets:** neither W0 (a)/(b) nor L2R (a)/(b) registers the
  outcome **FF(null) > FF(real)**. For L2R this is precisely the mouse-direction outcome
  (real uni-targeting *worse* than random given backbone+degrees) and is scientifically
  the most interesting possible result; for W0 it would mean anti-hierarchical weight
  placement. Neither outcome is *uninterpretable* — but to keep the pre-registration
  airtight, register their interpretations before the Stage B/C numbers exist. PROBLEM
  (pre-registration completeness; fix costs two sentences, zero runs).
* **Comparator-era note:** v2 nulls (Nov 2026) will be compared against G01 real-graph
  comparators measured Sept 2026 on the same machine class. Same protocol and
  epoch-counted budgets, and real-seed σ (0.0033 pp) ≪ draw σ — but the environment is
  not part of any config hash (the project's own moving-comparator lesson, P01). Log the
  torch version in every v2 run JSON and spot-check one real-graph re-run at protocol
  budget if any env component changed since 2026-09-30.

## D. `experiments/randomgraph.md` edits (commit e2a0424)

| item | finding | verdict |
|---|---|---|
| G01 status text numbers | 689 runs ✓, review date ✓, all-76-draws ✓, +19.8/+10.2/+1.9/+1.8 ✓ (rounded from 19.82/10.16/1.92/1.76), 6.72% ✓, 18–930× ✓, mouse 98/100 ✓, ~80% (79.85) ✓ — **no invented numbers** | CONFIRMED |
| deviations bullet | mirrors study §2 faithfully (G(n,m) weights-on-slot; config ≡ rewiring collapse; 17 annotation lanes; results path deviation disclosed) | CONFIRMED |
| Findings bullet | 16.3% vs 18.1–30.2% ✓, 6.72% / 18–930× ✓, 54/100 provable ✓, ~80% ✓; it reuses §7's phrases "not generically 'more feedforward than chance'" and "scale-dependent" — faithful to the reviewed study's own headline (so: a faithful summary), but v1 review red-team note (e) flagged both phrases for export; the species/scale confound caveat lives only in the G03 entry ("deconfound the fly↑/meso-mouse↓ scale-vs-species flip"). Acceptable internally; apply note (e)'s wording if this bullet ever leaves the repo. | CONFIRMED (minor wording note) |
| G03 entry vs charter | owner doc ✓, both generators + what they isolate ✓, stage-E conditional ✓, identical estimator + `run_protocol` (exists, `nullmodels_run.py:120`) ✓, K=12 ✓, comparators reused ✓, 2-cycle bound per draw ✓, artifact paths match charter §4 ✓ | CONFIRMED |

## Summary of required / recommended fixes before Stage B–C results are trusted

1. (required before Stage C *interpretation*, not before runs) Charter L2R wording: fix
   the two precision issues in C3 (residual location; "approximately uniform" + report
   `n_bad_initial/n_uni_slots` per draw).
2. (required before Stage B/C *interpretation*) Register the FF(null) > FF(real) outcome
   for both rungs (C4).
3. (recommended) `_validate_w0` float-sum assert → fsum/tolerance (safe for current
   datasets; breaks on any future float-weight dataset ~36% of the time).
4. (recommended) One-line empty-array guard in `_in_sorted`; docstring note (or guard) for
   self-loop half-counting in `two_cycle_bound`; rename `n_bad_repaired`.
5. (recommended) Log torch/env version in v2 run JSONs (comparator-era hazard).

None of these invalidates a Stage-A artifact: every quoted number is confirmed, the
mapping table's mathematics is correct (including the subtle and load-bearing
step-1 → `w_shuffled` assignment), and the decisive L2R bit-exactness guarantee is proven
sufficient and independently verified.

**Final verdict: PASS-with-corrections** (wording/pre-registration corrections only; no
numeric error, no load-bearing code defect).
