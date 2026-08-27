# H64 — PROTOTYPE PRE-REGISTRATION (sealed before launch)

**Cycle 16, 2026-08-27. Committed BEFORE the prototype runs** (P11; the H57 precedent).
Item: `autoresearch/queue.json` H64, priority 0, science, source "ideator + critic, cycle 15
(operator directive: challenge the gradient-axis kills)".

## What is actually being tested

H38's one-sided ASYMMETRIC surrogate is the only continuous mechanism that ever moved this
metric (+0.36675 pp PURE Rocket on connectome, 3/3 seeds). The measurement has three defects,
each verified in the artifacts and quoted in the queue item:

* **(a) comparator** — it was measured against `baseline_passthrough` (82.89580), not the
  champion (84.15409511053134);
* **(b) basin** — `src/mfas/experiments/H38.py:113,132-136` builds `RocketConfig` with
  `init_mode='random'` and `run()` takes no `init_positions`, so the entire gain lives in the
  RANDOM-INIT basin, while the champion's stage 2 starts from greedy-FAS;
* **(c) device** — all 18 H38* records are Apple MPS, so none of it is re-scorable here.

`killed.json` meta-rule **M1**, as amended 2026-08-26, names this cycle's job in its own words:
"Whether a shape gain survives composition with the refinement stack is OPEN and is what H64
measures." M1's burden for an asymmetric shape is (a) pass `surrogate_gate.py`, (b) show no
crossover in `beta*std`, (c) be measured COMPOSED with the champion stack.

## Rungs, and what each one can kill

### Rung 0 — zero GPU, seconds. M1 burden (a) + (b), re-derived on this machine.
`cheap_gate(_asym_surrogate)` and a `beta*std` crossover sweep. Q05 established both on MPS;
this re-derives them here so the artifact exists at the campaign device_tag.
**KILL IF**: the sigmoid-reparametrisation residual says ASYM is the sigmoid at another beta,
or `collapse_check` finds `g(0) == sup g`, or a crossover appears in `beta*std ∈ [1e-2, 1e6]`.

### Rung 1 — zero GPU, seconds. The REACH diagnostic, connectome-native.
From the stored champion positions, compute the fraction of BACKWARD WEIGHT lying inside ASYM's
live gradient support as a function of rank distance, against the same figure for the sigmoid.
ASYM's `tanh((z-M)/T)` is numerically dead for `(z-M)/T < -4`; at the operating point
`beta*Delta = 0.003753` per rank, so its reach is ~1,400 ranks, while `diagnosis.md` Step 2 puts
the median rank distance of the actual disagreement at 22,580 (p75 54,432).
**This rung KILLS NOTHING.** It is recorded because H65 and H66 are both downstream of it and
the queue item asks for it explicitly. A short reach is not a falsifier: a short-range repair
operator that hands a better order to stage 3 is exactly what composition is supposed to test.

### Rung 2 — the PROXY prototype. mouse + hard synthetic. Minutes, CPU/GPU-light.
Six arms per proxy, one seed each on mouse (deterministic pipeline) and the synthetic:

| arm | init | surrogate | stages | what it isolates |
|---|---|---|---|---|
| A0 | random | sigmoid | 1-2 (pure) | H38's original control condition |
| A1 | random | ASYM    | 1-2 (pure) | reproduces H38's setting on this device |
| B0 | greedy-FAS | sigmoid | 1-2 (pure) | champion stage 1+2 |
| B1 | greedy-FAS | ASYM    | 1-2 (pure) | **defect (b): the BASIN** |
| C0 | greedy-FAS | sigmoid | 1-4 (full H42) | the champion pipeline |
| C1 | greedy-FAS | ASYM    | 1-4 (full H42) | **defect (c): COMPOSITION** |

Reported: `(A1-A0)` the random-init pure gain, `(B1-B0)` the warm-start pure gain,
`(C1-C0)` the composed gain, and the realised pure->final transfer.

**PRE-REGISTERED KILL RULE (rung 2).** Kill H64 at the prototype iff, on BOTH proxies,
`(B1-B0) <= 0` **AND** `(C1-C0) <= 0`. That is: the shape gain neither survives the warm start
nor survives composition, anywhere it was cheap to look. Otherwise **proceed to the screen** —
arm A costs ~23 min of GPU, so the bar for spending it is deliberately low.

Stated in advance so it cannot be moved afterwards: mouse is `n=148` with
`baseline_sigma_pp = 0.2624` and the campaign's own SMALL-GRAPH ARTIFACT class exists for
exactly this dataset. A POSITIVE proxy result is therefore **not** evidence that H64 works — it
only buys the 23 minutes. Only a NEGATIVE-on-both result is being treated as decisive, and it is
decisive only in the weak sense that it removes the cheapest reason to look further.

### Rung 3 — the SCREEN (only if rung 2 does not kill).
`src/mfas/experiments/H64.py` = H42 with `torch.sigmoid` replaced by `_asym_surrogate` in stage 2
and NOTHING else changed. **The module is committed BEFORE the sweep launches** (the provenance
gate that currently FAILs for H36 and H42). Datasets per `campaign.yaml`: connectome + mouse at
the screen (`microns.in_screen = false`, operator decision 2026-08-26), seeds by `--auto-seeds`.
Pass iff `Delta > 0.012 pp` on connectome and mouse non-inferior.

## The pre-registered prior, stated before any number is seen

The queue item puts arm A at connectome **-0.10 to +0.12 pp, midpoint +0.03**, and its own
`counter_evidence_stated_before_running` section argues AGAINST the item at ~1 in 3. The measured
pure->final transfer coefficient is 10.3% / 28.0% / **-4.9%** (`proto_S01_connectome.json`): past
the operating point a strictly better pure order finishes WORSE. Applying that band to +0.36675
gives +0.04..+0.10 pp; the top-of-curve ratio gives <= 0. A-MBAND is the sharper precedent — it
repaired the surrogate's RANKING and then lost on all 12 arms, monotone in lambda.

**So the expected outcome of this cycle is a KILL, and a kill is the useful result**: it converts
M1's open clause into a closed one and licenses the amendment the queue item already drafts —
"pure-Rocket quality does not compose".

## What this prototype cannot do

It cannot substitute for arm A. H64 is an INNER-stage change, so M12 applies at full force
(measured 3.76x overstatement on H60) and M12's cycle-14 narrowing exempts only TERMINAL
appends. The proxy arms are a cost filter, not a prediction of the connectome number.
