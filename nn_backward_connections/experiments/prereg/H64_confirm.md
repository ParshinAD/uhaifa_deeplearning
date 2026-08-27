# H64 — CONFIRM PRE-REGISTRATION (sealed before launch)

**Cycle 16, 2026-08-27. Committed BEFORE the confirm sweep runs.** Same discipline as cycle 15,
and for the same reason: the microns leg is expected to sit on the runtime boundary, and a rule
for handling a truncated run must not be written after seeing which way it truncated.

## What the screen established (all six runs on disk, provenance commit f954ad6)

| dataset | n | pct | std | champion | delta pp | bar | wall s |
|---|---|---|---|---|---|---|---|
| connectome | 3 (42/123/999) | 84.25817950936937 | 0 | H42 84.15409511053134 | **+0.104084** | 0.012 | 1185.1 / 1186.0 / 1185.6 |
| mouse | 3 (42/123/999) | 93.17538325903584 | 0 | H63 93.17538325903584 | **0.000000** | non-inf | 0.8 / 0.7 / 0.7 |

The connectome delta is **8.67x** the minimum effect size. All three seeds are bit-identical, so
`seed_plan`'s `class=rng` was fail-safe rather than correct — H64 never draws from `seed`, because
`run()` always supplies `init_positions`. The screen ran 3 seeds anyway (`--auto-seeds` decides
mechanically, and the cycle does not hand-pick), which is a strictly better pool than the policy
required.

The seed-42 positions vector was re-scored independently with the frozen oracle:
**35,314,407 / 41,912,141 = 84.25817950936937**, matching the run record exactly.

## The runtime risk on microns, stated in advance with its numbers

`H64._EPOCHS["microns"] = 80_000` because H42 — the microns champion — runs 80,000, and P14
requires composing on the champion. Per-epoch cost on this machine, from records already on disk:

* sigmoid, microns, H63 at 70,000 epochs: **37.75 – 43.70 ms/epoch** (10 runs)
* sigmoid, connectome, H63 at 20,000 epochs: **21.11 – 24.09 ms/epoch** (8 runs)
* **ASYM, connectome, H64 at 20,000 epochs: 21.66 ms/epoch** (1185.1 − 118.3 sift − 633.7 alt
  = 433.1 s / 20,000)

So the asymmetric surrogate costs **essentially the same per epoch as the sigmoid** on this
hardware. The queue item's "~236 s of 1.6x gradient-phase overhead" is an MPS-era figure and is
**wrong on CUDA**; that is recorded here because it was an input to the cost estimate.

Projecting to microns at 80,000 epochs: stage 2 ≈ **3020 – 3496 s**, plus stages 3+4 ≈ 314 – 339 s
(H63 microns records), giving a total of **≈ 3334 – 3835 s** against a runtime guard deadline of
**3450 s** and a hard cap of **3600 s**. **Some or all microns seeds are expected to come back
`degraded`.** That is queue item **P07** biting exactly where P07 predicted it would.

## The rule for a truncated run — SEALED, adopted verbatim from cycle 15

1. Exclusion is decided by `runtime_guard.degraded`, which is set from the wall clock alone and
   cannot see the score.
2. A degraded run is **quarantined, not deleted**. It stays on disk and is named in the log.
3. **One** re-run of the **same** seed is permitted (box load is a known confounder: cycle 15's
   microns s7 re-run was 592.9 s faster on byte-identical code).
4. If the replacement is also degraded, the verdict is **iterate, not keep**.

## What a systematic microns truncation MEANS — decided now, not later

If microns degrades on a re-run too, that is **not** a fact about H64's algorithm; H64's microns
configuration is the *champion's* configuration, unchanged. It is the P07 finding: the shipped
microns configuration does not fit the runtime invariant on this machine. In that case:

* the connectome result stands on its own evidence and is reported as such;
* **the cycle does NOT cut microns' epoch budget to make it fit.** That would be a different
  variant (it is literally H62's move) and it would need its own ladder. Changing a constant to
  rescue a promotion is the thing `campaign.yaml` exists to prevent;
* whether a connectome-only championship is available while microns cannot be measured is the
  SAME open operator question already filed as **P15** (cycle 15 filed it with the legs
  reversed). The cycle does not decide it unilaterally.

## Pass criteria (unchanged from `campaign.yaml`, restated so they cannot drift)

* connectome: Welch 95% CI lower bound > 0 **and** PROTOCOL CI lower bound > 0 (sigma floored at
  `baseline_sigma_pp` = 0.0189) **and** delta > `min_promotion_delta_pp` = 0.012.
  At n=5 with a floored sigma the protocol SE is 0.0189 * sqrt(2/5) = 0.011954, so the CI demands
  delta > 1.96 * 0.011954 = **0.023431 pp**. H64's screen delta is **+0.104084 pp**, i.e. 4.44x
  that threshold — so unlike H52, H59 and H63, this delta is expected to clear the criterion that
  refused all three of them. Stated in advance.
* microns: same two criteria at `min_promotion_delta_pp` = 0.002, **and no degraded runs**.
* mouse: non-inferiority only. H64's mouse leg is bit-identical to H63 by construction, so the
  delta is exactly 0 and the test is passed trivially and **vacuously** — it exercises none of
  the changed code. This is stated in the module docstring and pinned by
  `tests/test_experiment_H64.py::test_mouse_leg_is_bit_identical_to_the_mouse_champion`.
* The **P09 relabel gate** applies to any degenerate-pool primary promotion. H64's pools are
  degenerate (std = 0), so it applies here, and no relabelling study is registered. This is
  the gate that refused H63's microns leg on a delta that cleared both numeric bars. It is named
  here so that the outcome is not a surprise: **the mechanical audit decides it, not the cycle.**

## What would make this result wrong, listed before the confirm runs

1. **A moving comparator.** The champion pool is 5 H42 connectome runs at `role=confirm`, all
   bit-identical at 84.15409511053134. H64's confirm will be compared at the same role.
2. **Extra compute.** H64 runs the SAME 20,000 epochs, the SAME 40 sift sweeps and the SAME 77
   alternation cycles as H42, and finishes in **1185.6 s mean against the champion's 1231.7 s**
   — it is cheaper, so no accounting can attribute the gain to compute.
3. **Oracle leakage.** `_asym_surrogate` reads positions and beta only. Best-by-oracle tracking is
   the baseline's own pre-existing behaviour and is unchanged.
4. **A stage-3/4 change smuggled in.** Pinned by `tests/test_experiment_H64.py`, which asserts
   every constant equals the champion's, per dataset.
5. **The mechanism being something other than the shape.** The one thing the screen already shows
   that needs explaining is that ASYM's PURE order is much WORSE (83.2379 vs the champion's
   stage-3 input) and its post-sift order much BETTER (84.1176 vs 83.9135). If that reverses at
   confirm, the mechanism story is wrong even if the number holds.

---

# ADDENDUM, sealed 2026-08-27 12:25Z — BEFORE the control run, AFTER the confirm

## What actually happened on microns

| seed | pct | wall s | degraded |
|---|---|---|---|
| 42 | 83.24619687456759 | 3419.6 | no |
| 123 | 83.24619687456759 | 3442.4 | no |
| 999 | 83.06280740365429 | 3501.1 | **yes** |
| 7 | 82.72980646089619 | 3509.0 | **yes** |
| 31415 | 82.56430000000000 (see audit) | >3450 | **yes** |

3 of 5 truncated. The walls are **monotone increasing in start time** across a 6.5 h continuous
GPU session, while the five connectome runs that ran FIRST (05:47-07:27) were flat at
1182.4-1185.4 s. That pattern is the signature of a machine that slows under sustained load, not
of a variant that is slow.

## Why the sealed re-run is NOT the test being run, and what is instead

The sealed rule (clause 3) permits ONE re-run of the same seed. Its purpose is to ask *"was this
run unlucky?"* With **3 of 5** degraded and a monotone drift, the unlucky-single-run hypothesis is
already largely excluded, so that test has little left to resolve — and re-running H64 now, on a
box warm from 6.5 h of continuous GPU, is confounded by exactly the accumulated thermal state it
would be trying to control for. A degraded result would therefore be uninformative.

The question that IS open, and that P07 asks in its own method step 1, is **attribution**: is
this the variant or the machine? The decisive control for that is to run the **CHAMPION, H42, on
microns, right now, on the same warm box**:

* if **H42 also exceeds the guard** → the degradation is the MACHINE. H64's microns leg is
  unmeasurable today for the same reason the champion's would be, and this is P07, confirmed on
  the champion itself rather than inferred from a variant;
* if **H42 runs clean at ~3400 s while H64 degrades** → the asymmetric surrogate genuinely costs
  more on microns, and that is a fact about H64 that must be reported against it.

This is a **deviation from the sealed clause 3**, taken deliberately and recorded before the run:
a strictly more informative test at the same cost, chosen because the hypothesis clause 3 tests is
already nearly excluded by 3/5. **The H64 s999 re-run remains owed** and is queued for the next
cycle; this addendum does not discharge it.

Run at `--role verify`, NOT `--role confirm`, so it can never be pooled into the champion's
comparator evidence.

## The verdict is already constrained, whatever this control returns

A full promotion is NOT reachable this cycle and that is stated before the control runs:
`campaign.yaml` requires **both** primaries, and microns additionally needs a relabelling study
(~9.4 h) that no budget here can buy. Even connectome alone fails `relabel.connectome`, whose
study costs ~3.3 h (5 labellings x 2 arms x 1185 s) against ~2 h of remaining cycle budget
(`budget.max_cycle_wall_clock_h` = 10 h; this cycle began ~04:31Z).

So this cycle ends in **iterate**, and the control run only determines *what the next cycle should
do* — buy the connectome relabelling study, or escalate the microns runtime to the operator.
