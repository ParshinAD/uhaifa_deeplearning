# Divergent mode, cycle 22 (2026-08-28) — why H79, and why not the other five things

**Trigger.** `cycles_since_score_move = 4` ≥ `campaign.yaml`
`escalation.divergent_after_k_cycles_without_score_move` (3). Second trigger
`consecutive_kills = 2` is not yet armed. `cycles_since_literature_scan = 3` (fires at 5), so
no broad scan this cycle — and cycle 21's handoff (d) says so explicitly, because
`scan_cycle18.md`'s yield is not exhausted (H71, H72 still queued).

This file is written **before** the item was worked, as divergent mode requires. It is short
because cycle 21 did the divergent *thinking*: M15 is the reasoning, and its operative directive
names the two surviving levels. This file records the **selection**, which is the part cycle 21
could not do for itself.

## P23 was checked first, and it is still the operator's

For the **sixth** cycle running. HEAD at preflight was `894492e`; no ruling commit exists on the
branch and `queue.json` still carries `status: awaiting-operator`. So H70 (a CONFIRMED microns
result, blocked only on `relabel.microns`), H63's microns leg, H74 and every future microns
variant stay blocked. Nothing this cycle can do changes that, and self-authorising an exemption
for a comparator arm that would knowingly exceed the 3600 s runtime invariant is exactly the
class of change `relabel_gate.py`'s docstring reserves for the operator. Left alone.

## The choice: H79 over H80, and over the whole M15-shaped rest of the queue

M15's directive is that an item whose mechanism is a **new or extended monotone move class on the
existing basin** must now argue against M15 before it is scheduled. That disqualifies, for this
cycle, H61 (k-node re-insertion), H71 (subset-bipartition interval repair), H72 (SCC cut
placement), H77 (more over-transport) and H67 (handoff snapshot) — all of them are move-class or
tie-break work on the champion's basin, all of them are predicted by M15 to land at or under the
+0.018 pp ceiling that every move class since H42 has hit, and none of them carries an argument
against M15 today. They are not killed; they are **not this cycle's item**.

That leaves the two levels M15 licenses:

* **H79 — a structurally different basin.** Rung 1 is a **free CPU pre-gate** (pairwise Kendall
  tau between constructions, against the champion's own relabelling ball as the unit of "same
  construction"). It can die for nothing, and if it does it kills the whole axis cheaply. That
  property is why it goes first.
* **H80 — barrier-crossing search sized to the 0.786 pp valley.** Strictly the more interesting
  of the two, and the harder one. Its own item states the honest difficulty: 0.786 pp is the depth
  of **one path** between two specific orders, not the landscape's barrier height, and a schedule
  admitting that much loss inside 3600 s may simply random-walk. It also needs a structure-aware
  destroy operator built from scratch (H40, never implemented). There is no free rung: rung 1 is
  ~1 h CPU on the hard synthetic and rung 2 is a full connectome run.

Ordering by **cost of being wrong** puts H79 first. If H79's rung 1 fails, the basin axis closes
for ~20 min of CPU and H80 becomes next cycle's item with one alternative eliminated. If H79's
rung 1 passes, its rung 2 measures a quantity — the refined-score spread across genuinely
different basins — that **H80 needs anyway**: a search that accepts uphill moves is only worth
sizing if there is more than one basin worth reaching.

## What H79 must not be allowed to become

Three earlier kills sit on this axis and each constrains the design rather than forbidding it:

1. **H48** (`starting basin`) killed ratio-greedy as a drop-in warm start — it was **+5.70387 pp
   better** as an init, still **+0.03456 pp** ahead after stage 3, and **−0.01382 pp** behind
   after the full pipeline. Its revival condition is "never again as a drop-in warm start for the
   current stack". So `ratio_greedy` is **measured but blocked as an arm**, by name, in
   `H79.BLOCKED_AS_ARM`. It appears in rung 1 only as a geometric reference point with a known
   post-pipeline outcome.
2. **H56/H57** (`restarts / multi-start`) killed harvesting dispersion by relabelling. M10 states
   the one surviving door in its own words: *"a mechanism that WIDENS the PREFIX's sigma at the
   champion's per-arm cost"* — and H56's revival condition (b) gives the bar, **0.021269 pp**. A
   different **construction** is precisely a prefix-level mechanism, so H79 is admissible under a
   condition the kill index itself wrote, not under one invented here.
3. **M6/M7** say better inits do not help and expensive ones lose. H79 contradicts neither: the
   arms are not quality proposals (two of three start within 1.2 pp of greedy-FAS) and all three
   cost under 20 s.

## The trap this item has to avoid, stated before the numbers

Reporting the **max** over arms would be a multi-start claim, and M9/M10 already govern those:
best-of-R over R draws gains `sigma * a_R`, R is not free (it is
`floor(deadline / per-run wall clock)`, i.e. **2** on connectome), and the canonical labelling
already sits at z ≈ +1.03 so random best-of-R is a *regression*. The deliverable is therefore the
**SPREAD**, and the kill condition is stated against the measured nuisance sigma (0.019124 pp)
rather than against zero.

And the honest limit, from M6/M2/H48: **distinct inits need not imply distinct basins.** Rung 1
can only ever refuse. Only rung 2 can say whether the stack erases the difference — which is the
one thing worth spending GPU on here.
