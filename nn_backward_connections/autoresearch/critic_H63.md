# Critic verdict — cycle 15, item H59, variant H63 (minimal-FAS arc reclamation)

Adjudicated 2026-08-27 against the proposed verdict **keep-partial** (promote mouse, refuse
connectome, hold microns). Read-only review: no source, `results/`, `sota.json` or `state.json`
was modified. Every number below was re-derived by this critic from the artifacts, not taken
from the cycle's summary.

Audits re-run independently (not the committed copies):

    audit.py --variant H63 --comparator champion --role confirm --comparator-role confirm \
             --gate promotion                       -> VERDICT FAIL, exit 1, 13 WARN
    ... --gate promotion --datasets mouse           -> VERDICT PASS, exit 0,  4 WARN
    ... --gate promotion --datasets microns         -> VERDICT FAIL, exit 1,  6 WARN

---

## VERDICT BY LEG

| leg | verdict |
|---|---|
| connectome | **REFUSE** |
| microns | **REFUSE as a promotion today — the cycle's HOLD is the correct word** |
| mouse | **APPROVE-WITH-CONDITIONS** (4 conditions, all bookkeeping; none touches the number) |

---

## 1. Frozen integrity — **PASS**

All four frozen files re-hashed by hand against `eval/frozen.sha256`: `src/mfas/metrics.py`,
`eval/harness.py`, `eval/aggregate.py`, `tests/test_metrics.py` — all OK.
`git log auto/campaign-v3 -- <the four frozen paths>` returns **empty**: no commit in the entire
campaign has ever touched them. `pytest tests/test_metrics.py tests/test_reclaim2.py
tests/test_audit_gates.py` -> 68 passed. Neither `H63.py` nor `reclaim2.py` is a frozen path.

## 2. Metric leakage — **PASS**. This is the check I pushed hardest on.

`leakage.dataset_keying` WARNs on 10 sites in `src/mfas/experiments/H63.py`. Every one is a
compute budget. Stating what each dict keys, as required:

| dict | line | what it is |
|---|---|---|
| `_EPOCHS` | 131 | Rocket gradient epochs — `{connectome 20000, mouse 0, microns 70000}` |
| `_MAX_SWEEPS` | 132 | stage-3 sift sweep cap |
| `_ALT_CYCLES` | 135 | stage-4 alternation cycle count |
| `_ALT_SIFT_SWEEPS` | 136 | sweeps per alternation cycle |
| `_PAIR_MAX_POPS` | 145 | stage-5 pop budget; 0 = stage absent |
| `_RECLAIM_ROUNDS` | 151 | stage-6 round count (uniformly 1 — not actually a per-dataset case) |

None reads the target metric. The oracle (`score_from_order`, line 229) is called exactly twice
in the module — lines 205-210 (best-by-oracle across whole stage outputs) and lines 234-236
(accept or discard the whole stage-6 vector). Both accept/reject a **complete candidate vector**;
no oracle value ever influences a move choice. `data/best_solution` is never opened; no reference
constant (34751902 / 84.6147 / 35463823) appears. `src/mfas/refine/reclaim2.py` and `reclaim.py`
contain **zero** occurrences of `g.name`, no dataset keying at all, and do not import `metrics` —
arc selection is purely reachability over the forward DAG. CLAUDE.md invariant 6 is intact.

**One thing I want on the record, not as a failure.** `_PAIR_MAX_POPS = {connectome: 0,
microns: 0, mouse: 100000}` is a *structural* per-dataset switch, not a dial — it turns a whole
stage off. It is justified (compose on the per-dataset champion: H42/H42/H52) and it is not an
oracle peek, but "a budget of 0" is how a structural special-case will look next time too. The
justification has to be restated every time it is used, and here it is: `sota.json` shows H42 has
no stage 5 on the primaries. Not leakage. Worth a standing note.

## 3. Novelty — **PASS**

Nearest killed axes checked in `autoresearch/killed.json`: **H09** ("free-edge / tie recovery",
revival_if "Never on these datasets") is anti-tie jitter on near-equal *positions*, killed by M5
because zero ties exist — a different object from reachability in the forward DAG. **H41**
(move class inside stage 4) and **H31** (ILS/LNS wrapper) are distinct mechanisms. **M11**
demands "a realised gain from an exact or run optimum" rather than a ceiling statistic; H63
reports a realised gain. **M8**'s redundancy requirement is satisfied structurally and
numerically — see check 6. No revival condition is needed.

**But M8 has a second clause the cycle has not addressed** — see FINDING 1 below.

## 4. Moving comparator — **PASS**, and stronger than the audit's WARN suggests

`comparator_homogeneity.connectome.variant` WARNs that H63 spans 3 commits. I resolved this by
blob hash rather than by commit id:

    src/mfas/experiments/H63.py   = f2d88df5...  at 42788d9, 77c18de, 221a2b7, 79f1fca,
                                                    89a6896, d2f94bd, b20df5b, HEAD, worktree
    src/mfas/refine/reclaim2.py   = b8d1255e...  at all of the same

Both files are **byte-identical at every commit any run was stamped with and in the working
tree**. The variant is one configuration; the `+dirty` flags and multi-commit spread come from
results/logs/state churn. This is the `H30@12 vs H30@40` failure mode and it is *not* present.

Comparator side, per leg:

- **mouse**: H52, n=20, **one commit** (`5734d23d`, 19 of them `+dirty`), **one distinct value**
  93.10282596057698, `total_grad_steps` 0 for all 20. A single configuration. Clean.
- connectome: the H42 pool does pool two distinct commits (`1865372b` n=3, `3791350d` n=2). All
  five values are identical, so there is no numerical effect — but it is the hygiene CAMPAIGN.md
  warns about and it is another reason the connectome leg is not promotable evidence.

## 5. Significance — **connectome FAIL, microns FAIL (on the gate, not the arithmetic), mouse PASS**

Re-derived, not quoted:

| dataset | H63 | comparator | delta pp | effect_size | protocol CI lo | relabel |
|---|---|---|---|---|---|---|
| connectome | 84.17175347830596 | H42 84.15409511053134 | +0.017658 | PASS (bar 0.012) | **-0.0058 FAIL** | **FAIL** |
| microns | 83.25965742667618 | H42 83.24085291200831 | +0.018805 | PASS (bar 0.002) | +0.0181 PASS | **FAIL** |
| mouse | 93.17538325903583 | H52 93.10282596057698 | +0.072557 | n/a (supporting) | n/a | n/a |

The connectome refusal is not marginal and it is not new. **H52's connectome leg was refused on
2026-08-17 at +0.020927 pp** (`experiments/log.md:5031`, and the caveat is written into
`sota.json`'s mouse row). H63's +0.017658 pp is *smaller* than the delta that was already
refused, and it now faces the additional P09 relabelling gate that did not exist in cycle 8. The
connectome leg is further from promotable than its own precedent. REFUSE, comfortably.

No cherry-picking: the audit's inventory lists exactly the `confirm_seeds` from `campaign.yaml`
(connectome/microns 42/123/999/7/31415; mouse all 20). I enumerated every `results/*H63*.json`
including `quarantine/`: exactly one seed has two attempts (microns s7) and nothing else is
duplicated, dropped or added.

**Independent oracle re-score of the mouse leg** (frozen `score_from_order`, CPU):
H63 s42 = 8.533967060075327 -> 93.17538325903583; H52 s42 = 8.527321510860956 ->
93.10282596057698; delta = +0.07255729845884673 pp; **56 of 148 positions differ**. Matches the
records exactly.

## 6. Double counting — **PASS on connectome and mouse; microns is a compound and must be labelled as one**

From `variant_attrs` in the confirm JSONs, not from prose:

| ds | `base_best_pct` (pre-stage-6) | champion | difference | `reclaim_increment_pp` |
|---|---|---|---|---|
| connectome | 84.15409511053134 | 84.15409511053134 | **0.0** | +0.017658 |
| mouse | 93.10282596057698 | 93.10282596057698 | **0.0** | +0.072557 |
| microns | 83.24154769... | 83.24085291200831 | **+0.000695** | +0.018110 |

- connectome and mouse: the pre-stage-6 base reproduces the champion **exactly**, so 100% of the
  delta is stage 6. No double counting. It also settles M8's redundancy clause numerically: the
  earlier stages' credit is unchanged by adding stage 6, i.e. redundancy fraction 0 — which is
  structural, since a terminal stage cannot compete with the classes ahead of it for the same
  ground.
- microns: the base is **+0.000695 pp ABOVE** the champion, not below as the module docstring
  predicted (H63.py:77-83). So the H62 epoch cut 80000 -> 70000 did not cost score, it gained a
  hair. The microns delta is therefore a compound: 96.3% stage 6, 3.7% epoch cut. Attributing
  **zero** to the epoch cut still leaves +0.018110 pp against a 0.002 bar, so the microns
  conclusion is robust to the decomposition — but the cycle must report +0.018110 as the stage's
  contribution and +0.018805 as the composed delta, never one as the other.

## 7. Compute fairness — **PASS on all three, and inverted in H63's favour on microns**

| ds | H63 grad steps | comparator | wall H63 | wall comparator |
|---|---|---|---|---|
| connectome | 20000 | H42 20000 (equal) | 1346-1552 s | 1226-1238 s |
| microns | **70000** | H42 80000 (**12.5% fewer**) | 2955-3398 s | 3398-3418 s |
| mouse | **0** | H52 0 (**equal**) | 0.7-0.8 s | 0.7-0.8 s |

Every run is inside `runtime.max_wall_clock_s_per_run` = 3600 s (max observed 3397.7 s). The
`compute.microns` WARN is inverted: H63 wins microns on **strictly less** compute on both bases.
That is a stronger claim than equal-compute, not a weaker one, and the cycle must say so
explicitly rather than let a WARN stand unexplained.

**The mouse leg is exactly compute-matched on both bases (0 grad steps, 0.7-0.8 s both arms).**
That is as clean as this check gets.

Honest cost on the primaries: stage 6 costs 156.9 s (connectome) / 151.2 s (microns) of wall
clock that the champion does not spend. Reported, not hidden. On mouse it costs <0.01 s.

## 8. Reproducibility — **PASS for mouse and connectome; INCOMPLETE for microns**

Every number traces to a `results/*.json` with `role=confirm`, a pinned seed, a `git_commit` and
`total_grad_steps`. `provenance.*` PASSes on all three (H63.py exists at every stamping commit).
`rescore` PASSes: 30 runs re-scored exact by the frozen oracle.

**But the evidence is not all committed** (CLAUDE.md invariant 5, "reproducible via `git
checkout` + the logged command"). Tracked confirm runs: mouse **20/20**, connectome **5/5**,
microns **3/5**. The two replacement microns runs
(`20260826T232612Z-...-s7-...`, `20260827T001529Z-...-s31415-...`), both audit JSONs, the
quarantine `_positions.npy`, the `MANIFEST.md` edit and `state.json` are all uncommitted. **The
mouse leg is fully committed and reproducible today. The microns leg is not.** This is a second,
independent reason microns cannot be promoted in this cycle.

---

## The quarantine — attacked as data selection, and it survives

**Sealing order, verified from git rather than from prose:**

    b20df5b  "cycle 15 resume: seal the degraded-run rule BEFORE re-running microns s7/s31415"
             committed 2026-08-27T02:26:06+03:00  (= 2026-08-26T23:26:06Z)
    state.json (in that commit) records the launch at 02:26:11 local
    replacement s7  run STARTED 2026-08-26T23:26:12Z  (filename = start; the internal
             `timestamp` field 2026-08-27T00:15:27Z is the write time, = start + 2954.6 s)
    replacement s31415 STARTED 2026-08-27T00:15:29Z

Six seconds. Tight, but the order is right, and the far stronger evidence is not the clock: both
replacement runs carry `git_commit = b20df5b3...+dirty`, the commit that **contains** the prereg,
while the degraded run carries `89a6896...+dirty`, at which
`git cat-file -e 89a6896:.../prereg_H63_confirm_rerun.json` **fails — the file does not exist**.
The rule was in the tree the replacements were run from and was not in the tree the excluded run
was run from.

**Is `runtime_guard.degraded` score-independent?** Yes, mechanically.
`eval/runtime_guard.py:222` sets `degraded = bool(deadline_reached or truncated)`, where
`deadline_reached` compares wall clock to the limit and `truncated` compares stage
done-vs-requested counts (lines 194-211). Grepping `score|pct|best_` across the whole file
returns only docstring prose. No score of any run can reach the flag.

**One re-run per seed?** Held. Exactly one duplicate exists in the entire H63 corpus.

**Does the rule bind in the losing direction?** Yes, and this is the finding that convinced me.
I computed the counterfactual:

| microns pool | mean | std | delta | effect_size | protocol CI lo | pool degenerate? | relabel gate |
|---|---|---|---|---|---|---|---|
| degraded EXCLUDED (as reported) | 83.25965743 | 0 | +0.018805 | PASS | +0.018061 PASS | **yes** | **ARMED -> FAIL** |
| degraded INCLUDED | 83.24791499 | 2.63e-2 | +0.007062 | PASS | +0.006318 PASS | **no** | **not armed** |

**Excluding the degraded run made the microns gate STRICTER, not weaker.** Including it would
have de-degenerated the pool and thereby *switched off* the P09 relabelling requirement, leaving
microns clearing every remaining numeric gate. (It would then have been caught by
`audit.py:860`, which FAILs any pool containing a degraded run — but the point stands.) A cycle
optimising for headlines would have kept the run and argued that line 860 is over-strict. This
one quarantined it and took the harder gate. That is the opposite of cherry-picking. The sign of
the microns result is also unchanged either way (+0.0071 vs +0.0188), so the *conclusion* does
not depend on the exclusion at all.

I note for completeness that the excluded run's score **was** known when the rule was sealed
(`state.json` at b20df5b quotes 83.20094526 and calls it below the champion). The prereg says so
itself. What was not known — and what the rule turns on — is the replacement's score. Combined
with the table above, I do not think this is exploitable.

**The mouse leg does not depend on any of this, and I verified it rather than assuming it:** all
20 mouse confirm runs were written 2026-08-26T15:39Z at commit `77c18de0+dirty`, ~4.5 h before
the degraded microns run and ~8 h before the quarantine existed. All 20 are committed. Nothing
about mouse changes if the quarantine is thrown out entirely.

---

## Is `--datasets mouse` gaming the exit code? — **No.**

Two independent reasons, both verified in code:

1. **`update_sota.py` runs exactly that command itself.** Lines 137-145: a promotion invokes
   `audit.py --variant ... --datasets <args.dataset> --gate promotion`. Auditing the mouse leg at
   `--datasets mouse` is not a narrowing the critic has to forgive; it is *the* promotion command.
   Auditing at all three and then promoting mouse would in fact be the mismatched thing to do.
2. **None of the three FAILs can bear on mouse.** `audit.py:706` gates `effect_size`, `relabel`
   and `protocol_ci` behind `if ds_role == "primary"`. Mouse takes the `else` branch (line 788+),
   which is non-inferiority only. `promotion_gate.primary.degenerate_pool_rule` in
   `campaign.yaml` is nested under `primary:`; the `supporting:` block contains only `se:` and
   `require_non_inferiority:`. The campaign's own record says the same in prose:
   `relabel_index.json`'s `H52|H44|mouse` entry notes "mouse is a SUPPORTING dataset and is gated
   by non-inferiority, not by this rule."

`update_sota.py` additionally enforces `min_promotion_delta_pp` for mouse (0.01) at lines 124-134
before it ever calls the audit; +0.072557 clears it 7x.

**What I will not let pass unstated:** `non_inferiority.mouse` PASSing means "H63 is not worse
than H52 by more than 0.26 pp". It does **not** carry a significance test that H63 is *better*.
The mouse PROTOCOL CI lower bound is **-0.0901** (negative). The affirmative case for the mouse
promotion rests on the point estimate + `min_promotion_delta_pp` + determinism (std 0 over 20
seeds, 56/148 positions moved, gain exactly equal to the reclaimed weight 0.00664554921437035 /
9.159036176272162 = +0.072557 pp), which is the campaign's designed standard for a supporting
dataset — not on a CI. Say it that way in the log; do not write "significant".

## Is the cycle-8 / H52 precedent the same shape? — **Yes, and H63's case is better on both sides**

| | cycle 8 (H52) | cycle 15 (H63) |
|---|---|---|
| connectome | REFUSED by `protocol_ci`, delta +0.020927 | REFUSED by `protocol_ci` **and** `relabel`, delta +0.017658 |
| mouse | PROMOTED, delta +0.019946 vs H44 | proposed, delta **+0.072557** vs H52 |
| microns | bit-identical, nothing to decide | real +0.018805, **blocked by `relabel`** |
| relabel gate | did not exist | in force |

H63's connectome case is *weaker* than the one already refused; its mouse case is 3.6x stronger;
and it clears a strictly stronger gate. The precedent is not being stretched. The one genuine
difference is microns, where cycle 8 had nothing to hold and cycle 15 does.

## Is holding microns right? — **Yes, and it is machine-enforced, not a judgement call**

I ran it: `audit.py ... --datasets microns --gate promotion` -> **exit 1**, FAIL on
`relabel.microns`. `update_sota.py --variant H63 --dataset microns` would therefore refuse.
microns is not "available" — the campaign's own tooling will not install it. Cost to unblock:
~9.4 h GPU for the (H63, H42, microns) relabelling study per `campaign.yaml`'s P09 note.

So: **HOLD is exactly the right word, and the cycle would be over-claiming if it said microns is
promotable-but-declined.** It is refused, on a gate the campaign imposed on itself. Add the
uncommitted 2/5 runs and it is refused twice over. The cycle is not under-claiming either — the
correct statement is "microns clears both numeric gates and is blocked on missing P09 evidence",
and that is a queue item, not a championship.

---

## FINDINGS to record in the log regardless of verdict

**FINDING 1 — M8's marginal-rate clause was not evaluated, and stage 6 fails it on both
primaries.** `killed.json` M8 says a new move class "must also clear the marginal rate of the
stage it is inserted into (stage 4 measured at 3.45e-4 pp/s on connectome, 4.33e-4 on microns)".
Computed from `variant_attrs.reclaim_time_s`:

| ds | stage-6 time | increment | rate | M8 bar | |
|---|---|---|---|---|---|
| connectome | 156.9 s | +0.017658 | **1.13e-4 pp/s** | 3.45e-4 | **3.1x below** |
| microns | 151.2 s | +0.018110 | **1.20e-4 pp/s** | 4.33e-4 | **3.6x below** |
| mouse | <0.01 s | +0.072557 | 7.3e+1 pp/s | — | clears by ~5 orders |

This is an *additional*, independent argument against the primaries that the cycle did not make,
and it leaves the mouse leg untouched. The counter-argument (stage 6 is terminal and monotone, so
it is not literally "inserted into" stage 4, and on connectome stage 4 already ran 77 cycles) is
reasonable but must be *made*, not assumed — cycle 14 argued exactly that shape for M12 and it
was recorded. Either address it or amend M8 to say it does not apply to a terminal stage.

**FINDING 2 — the degeneracy-conditional relabel gate has a perverse incentive.** As the
counterfactual table above shows, a **noisier** variant pool faces a **weaker** promotion gate,
because `degenerate_pool_rule` only arms when `std <= zero_std_tol` on both pools. Any future
variant with one flaky run gets the P09 requirement switched off for free. This cycle happened to
walk into it from the safe side. It should be filed as a P-item against `audit.py` /
`campaign.yaml` before someone walks into it from the other side.

**FINDING 3 — H63 is missing from `autoresearch/seed_class.json`.** PROTOCOL.md § Phase-7.4:
"Every log entry must record the classification of the variant it reports." The live classifier
answers `H63 -> rng, 3 seeds`, on the fail-safe path (`UNRESOLVED:
mfas.refine.pair_relocate.pair_relocate: _time.time() line 198` — `time.time()` is not an RNG, so
this is a conservative false positive). No number is affected: the screen did run 3 seeds per
dataset, and `confirm_seeds` is unchanged either way. Register it and record it.

**FINDING 4 — a 3e-14 pp discrepancy between `sota.json` and the run records.**
`sota.json` mouse `pct_mean_exact` = 93.10282596057695; every one of the 20 H52 run JSONs says
93.10282596057698. Float re-mean noise, eleven orders below anything that matters, no effect on
any gate. Noted so it is not rediscovered later as a mystery.

**FINDING 5 — the microns runtime margin is still thin and the P07 risk is live.** Clean microns
slack against the 3450 s deadline: s7 527 s, s31415 524 s, s42 404 s, s123 383 s, **s999 87 s**.
One of the five accepted runs came within 87 s of degrading. H62's epoch cut helped (the champion
sits at 3398-3418 s) but did not remove the exposure.

---

## RECOMMENDATION: **keep-partial, as proposed — promote MOUSE only**

The mouse leg is the cleanest promotion evidence this campaign has produced: exact compute match
on both bases, single-commit single-valued comparator pool, base reproducing H52 bit-for-bit at
four intermediate checkpoints (`pure` 90.12629794323948, `sift` 93.08288021668459, `alt` same,
`pair` 93.10282596057698 — all identical to H52 s42), the entire gain isolated to one terminal
monotone stage and equal to the reclaimed arc weight exactly, independently re-scored by me with
the frozen oracle, 20/20 runs committed, and untouched by the quarantine. **P14 is genuinely
fixed and this is not a comparator swap** — cycle 14's H59 mouse leg scored 92.94798739 with no
`base_best_pct` attribute at all because it sat on H42; H63's base *is* H52's output, verified
numerically stage by stage.

The connectome and microns legs are refused, and the cycle is right to refuse them.

**Conditions on the mouse promotion (all bookkeeping; none changes a number):**

1. Commit the two untracked microns confirm runs, both audit JSONs, the quarantine `.npy`, the
   `MANIFEST.md` edit and `state.json` **before** running `update_sota.py` (invariant 5).
2. Register H63 in `seed_class.json` and record the classification (`rng`) in the log entry
   (PROTOCOL § Phase-7.4).
3. Write the `sota.json` mouse row with `--caveat` covering: SUPPORTING DATASET ONLY, does not
   advance the Phase-1 mission target; the connectome leg was REFUSED at +0.017658 pp
   (`protocol_ci` -0.0058 **and** `relabel`), i.e. by a *smaller* delta than the +0.020927 pp
   already refused in cycle 8; the microns leg was REFUSED on `relabel` despite clearing both
   numeric gates; the affirmative case is a point estimate under non-inferiority, not a CI
   (mouse PROTOCOL CI lower bound is -0.0901).
4. In the log, report **+0.018110 pp** as stage 6's microns contribution and **+0.018805 pp** as
   the composed microns delta, and state that the microns comparison uses 12.5% **fewer** gradient
   steps than the champion. Do not present the connectome or microns numbers as wins anywhere.

## What would falsify this later — where the mouse finding is fragile

- **Scale.** The gain is 4x larger, relatively, on the smallest graph: mouse recovers +0.0726 pp
  from **5** arcs on 148 nodes, while connectome recovers +0.0177 pp from 2,231 arcs on 136,648
  and microns +0.0181 pp from 2,833 on 67,534. The reclaimable fraction of the FAS shrinks fast
  with graph size. Read against `diagnosis.md` § Q01 this is the exact silhouette of a
  scale-dependent effect, and **it must not be extrapolated to the connectome** in any prose. If
  a later cycle finds the reclaimable set is essentially a small-graph phenomenon, the mouse
  championship survives (it is a real, exact gain on that graph) but the *mechanism's* standing
  as a general improvement does not.
- **Base dependence.** The whole mouse gain is measured on H52's specific output order. Stage 6
  certifies *that* arc set minimal — round 2 found 0 arcs on all three datasets. If the mouse
  champion moves, H63's increment must be re-measured from the new base; it does not transfer.
- **The relabelling gate, if it ever reaches mouse.** Nothing today requires a relabelling study
  for a supporting dataset. If P09 is extended to supporting datasets, this promotion becomes
  retroactively unsupported, exactly as H36/H42 are (`campaign.yaml`: "KNOWN DEBT").
- **The 87 s microns slack (FINDING 5).** If this box or the next one is 3% slower, microns
  degrades again and the sealed rule forces `iterate`. The mouse leg is immune, but the *cycle's*
  ability to ever close the microns leg is not.
