# Handoff — 2026-08-15 night session

**Read this if you are a fresh session picking up work that a previous session started.**
`CAMPAIGN.md` is still the constitution and `state.json` is still the resume point; this file
covers only what happened in the 2026-08-15 session, which ran OUTSIDE the normal cycle loop
(an operator-driven infrastructure pass, not a `/research-cycle`).

## What is running right now, unattended

Two detached shell scripts. They are ordinary OS processes: they survive a Claude Code restart,
a session ending, and a permission prompt. **Do not relaunch them blindly — both keep a state
file and will skip completed steps, but a duplicate would contend for the one GPU and poison
`wall_clock_s`, which is a measured quantity in this project.**

| script | log | state | what it does |
|---|---|---|---|
| `dr_tmp/night_run.sh` | `dr_tmp/night_run.log` | `dr_tmp/night_run.state` | microns epoch grid -> connectome epoch grid -> full pytest -> `verify_cycle.py` |
| `dr_tmp/night_run2.sh` | `dr_tmp/night_run2.log` | `dr_tmp/night_run2.state` | waits for the first, then screens **H41** on connectome -> mouse -> microns |

Check progress with `tail`, not by relaunching. `night_run2.sh` reads a kill signal out of the
connectome run record (`variant_attrs.n_segment_moves`) and **skips the 57-minute microns leg
automatically** if the segment move class never fired.

## What landed tonight (5 commits)

| commit | what |
|---|---|
| `6cb38f3` | the 2 completed arms of the aborted 2026-08-11 P07 probe, so they are not lost |
| `91220f6` | nine driver/lifecycle defects; sweep state machine; `detach.sh`; `verify_cycle.py` |
| `a083368` | `audit.py --gate promotion`; mouse gate re-specified; queue re-prioritised; P08 filed |
| `ef049e1` | **H41** — the segment move class, source committed BEFORE its screen |
| `a5f1d91` | **P05 closed** — the microns leg, bit-identical |

### The three findings that change how you should read the record

1. **P05 is closed and it corrected the runtime picture.** H42/microns/s42 on an idle machine:
   0 of 67,534 positions differ from the stored confirm vector, guard armed and silent. It ran
   in **3258.2 s** against cycle 5's confirm pool of 3397.8-3418.0 s — so **that pool was never
   an idle measurement**. Real slack against the 3450 s deadline is 223 s (6.5%), not the 68 s
   (2.0%) `campaign.yaml` still says. P07's "~20% slower under load" now has a counter-point at
   ~4.3%; it is two load points, not one. Measured stage split: sift 87.0 s + stage 4 86.0 s =
   173 s = **5.3%**, so microns is ~94.7% Rocket. That is what the epoch grid is sizing.

2. **The auditor could not previously refuse a promotion for any scientific reason**, and
   `update_sota.py` compared against a value rounded to 4 decimals. That is not hypothetical: the
   H36 and H42 mouse confirm pools are bit-identical, and H42 took the mouse championship because
   `92.917014102 > round(92.917014102, 4)`. Both are fixed. `--gate promotion` now FAILS on
   effect size, CI, non-inferiority, empty comparator pools, and **provenance**.

3. **Provenance FAILS for both existing champions and this is real.** All 36 H36 runs cite
   `9d43b977+dirty`, where `src/mfas/experiments/H36.py` does not exist; 32 of H42's confirm runs
   cite `1865372b+dirty`, where `H42.py` does not exist. The module was committed one commit later
   each time and the runs were never re-derived. Filed as **P08**. Severity is moderate, not
   critical — the measuring apparatus provably did not change, and H42's numbers have now been
   re-derived from tracked source on all three datasets. H36 is what remains open.
   **The forward fix is free: commit the variant module BEFORE launching the screen.**

## H41 — the live scientific question

A third move class: exact-gain bounded-span **segment** moves. A right move of a segment is the
swap of two adjacent contiguous blocks, so by the same lemma stage 4 already uses, only the cross
edges flip and `delta = W(J->S) - W(S->J)` exactly. 145,386 (segment, target) pairs were
cross-checked against the frozen scorer with zero mismatches.

**Read the prototype with this correction.** On the PRODUCTION mouse path the segment class fires
**zero** times — H42's mouse order is already a fixed point of the whole bounded-span
neighbourhood. The prototype's mouse row (+0.0589 pp) came from a greedy+sift start that skipped
Rocket and sits in a different, higher basin (93.083 vs 92.902). It does not transfer. What
survives is hard6000: +0.0373 pp, and there both arms were at their own fixed points at 20 s and
at 60 s, so it is a fixed-point-to-fixed-point difference and cannot be an extra-compute artefact.

**The open question is whether the class fires at all on connectome's production order.** Its
widest window is 2048 of 136,648 positions — 1.5% of the line, against 34-64% on every proxy — so
no cheap proxy tests the regime that matters. `n_segment_moves` in the connectome run record
answers it directly. Zero there is a clean kill for the DEFAULT ladder and an immediate argument
for the extended one (the diagnosis puts median rank-distance at ~22,580, 11x beyond the top
rung), which needs the greedy disjoint-packing rule reconsidered first — do not just lengthen the
ladder.

If connectome shows a gain, **do not promote without a matched-wall-clock control** (H42 given
H41's seconds via extra `_ALT_CYCLES`). `budget_basis` is still `total_grad_steps` and stages 3-4
add zero optimizer steps, so the audit will call H41 and H42 compute-matched while H41 spends
strictly more CPU. That is the same accounting hole H36 had.

## Known-untested

- **`driver.sh` grew 225 -> 672 lines tonight and has NOT been run end to end.** Syntax is clean
  and `kill_tree`'s ordering fix was read and verified, but the watchdog path, the `taskkill`
  branch, the `--abort` flow and the live auth preflight were never exercised. **Do a short
  supervised run before trusting it unattended.**
- `--abort` now refuses to kill a pid it cannot verify as ours, matching a name regex against the
  `ps -W` COMMAND column. Nobody has observed what a live cycle's row actually says on this box.
  If abort stops working, that is the first thing to check.
- `verify_cycle.py` FAILs today on `gates_run` (no past cycle recorded it — that field is new) and
  on cycle 4's missing history entry (a real, permanent hole). Both are expected.

## The structural problem — diagnosed and FIXED, but the counter needs seeding

The campaign's escape from incrementalism — divergent mode — **had never once executed**.
`consecutive_kills` never left 0, because 4 of the first 7 cycles were infrastructure items which
by construction never increment it, and both science items passed. A campaign that produces no
kills can never trip a kill-counting trigger. `autoresearch/lit/` does not exist; L01, the
literature scan itself, sat at priority 8 and was unreachable.

Fixed 2026-08-15 in `campaign.yaml § escalation` and `research-cycle.md § 2`: the primary trigger
is now **`cycles_since_score_move >= 3`** — cycles since any champion last changed, counting every
verdict including no-ops. On the real history it fires at cycle 7. `consecutive_kills >= 3` is
kept as a second, independent trigger. Literature scan forced every 5 cycles rather than 8. And
**an infrastructure item no longer satisfies divergent mode**: if a P-item is on top while
divergence is active, take the highest-priority science item instead.

**ACTION REQUIRED:** `state.json` has no `cycles_since_score_move` field yet. Derive it from
`history` (the last champion change was cycle 5, H42) and write it, or the new trigger reads as
absent. As of cycle 7 the correct value is **2**, so one more cycle without a score move forces
divergence.

---

# WHAT TO DO NEXT — read this as the instruction, not as background

The operator's standing direction, verbatim in spirit: **get the best possible result; the
research method is open.** The process rules in `CAMPAIGN.md` are means, not ends. Treat them in
two tiers and do not confuse them:

**Tier 1 — never bend these.** They are what makes a number real, and every one of them was
written after this project got burned:
- Never fabricate or estimate a figure. Every number traces to a `results/*.json` or
  `experiments/outputs/*.json`. `autoresearch/verify_cycle.py` now checks this mechanically.
- Never let the target metric into an algorithm; the reference solution is for measurement only.
- Never edit a frozen file.
- Never claim a win inside noise, and never on one dataset.
- Commit the variant module BEFORE the runs that will be cited as its evidence.

**Tier 2 — everything else is yours to change if it buys a better result.** Gate ordering, seed
counts, how many cycles a hypothesis gets, whether to prototype at all, whether to run several
cheap probes in one cycle instead of one expensive ladder. If a rule is costing more than it
protects, change it and write down why. Two examples already on the table: the confirm stage
spends ~9.8 h of GPU re-running bit-identical deterministic variants at 5 seeds, which is 47% of
all in-cycle time the campaign has ever spent; and the campaign is generating infrastructure debt
2.5x faster than it closes it.

## The immediate agenda, in order

1. **Read the two night logs and finish the analysis.** `dr_tmp/night_run.log` carries the epoch
   grids; `dr_tmp/night_run2.log` carries the H41 screen. Do not relaunch either script.
2. **The epoch-sizing question is live and the early read was wrong.** Arms so far on microns:
   0 -> 83.1115%, 2500 -> 83.12094%, 5000 -> 83.1356%, against a champion of 83.2409% at 80,000.
   The first two arms suggested the gradient phase was ceremonial; the third shows the curve still
   climbing (+0.0147 pp for the third 2500 epochs, against +0.0094 for the second). Do not cut
   microns' epochs on a two-point read. Wait for 10,000 and 20,000 and fit the shape.
3. **H41 is the live score hypothesis.** Its decisive signal is `variant_attrs.n_segment_moves` in
   the connectome run record. If it is zero, that is a clean kill for the default ladder and an
   immediate argument for the extended one — but the extended ladder needs the greedy
   disjoint-packing rule reconsidered first, not just more rungs. If it fired and the score moved,
   you still need a matched-wall-clock control on connectome before promotion.
4. **Then run the driver** (`bash autoresearch/driver.sh`), but supervise its first cycle — see
   Known-untested above. It is the mechanism for periodic unattended research and it now has a
   rolling budget, no-op detection and a blocked-mode stop after three no-ops.

## Where the score is most likely to come from

The gap is 0.4606 pp on connectome and it is known reachable — Vahidi 2025 got there with cheap
greedy + bounded-span insertion + SCC and no MIP. Two facts worth carrying:

- **The champion's two wins are both CPU refinement, not GPU.** `src/mfas/refine/` has zero torch
  calls. On connectome, stage 3 + stage 4 are 831.7 s of a 1483.9 s run — **56% single-threaded
  NumPy on a 12-core box.** Nothing in the queue addresses that; S01 sizes only the Rocket half.
- **connectome has ~2368 s of its 3600 s cap unused**, while microns has ~342 s. Anything that
  costs connectome time is affordable today; anything that costs microns time is not, until the
  sizing question is settled.

The constants `min_block=32`, `n_cycles=77`, `sift_sweeps=2` were all tuned on the evaluation
graphs using the target metric, with no held-out graph. That is not leakage under the project's
own definition, but "we closed 34.6% of the gap" is a training-set number and should be stated as
one. A held-out graph would make the claim defensible; nobody has built one.
