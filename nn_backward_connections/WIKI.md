# What this project is, in short

*(Repo content is English by project rule. A Russian copy for the lab wiki lives outside the repo.)*

## The task

Take the fly brain connectome — **136,648 neurons, 5.7 million connections** — and line all the
neurons up in a single row. A connection counts as **forward** if it points from an earlier neuron
to a later one. Goal: arrange the row so that as much connection weight as possible points forward.

This is the Maximum Feedforward Arc Set problem. It is NP-hard, so nobody computes the true optimum;
everybody reports how close they got.

Two reasons we care:
1. **Algorithmic** — improve *Rocket*, the method from Bader et al. (2025), which we reproduced.
2. **Neuroscientific** — the weight that *cannot* be made forward is the brain's unavoidable
   feedback. Do real brains carry more or less of it than random networks of the same shape?
   (This second track is scaffolded but not yet run.)

## Where we stand

| | forward weight |
|---|---|
| trivial baseline (random order) | 50.05% |
| simple greedy heuristic | 72.04% |
| **Rocket, reproduced from the paper** | **82.92%** |
| + our improvements | **83.91%** |
| best known solution for this graph | 84.61% |

We closed **0.98 of the 1.69 percentage points** between the reproduced baseline and the best known
solution — with no commercial solver and no extra optimizer steps. The paper's own way of going
further needs a 20-day run of a commercial MIP solver, which we do not have.

## The rules we work under

These exist because it is very easy to fool yourself in this kind of work.

1. **The scorer is frozen.** One piece of code computes the metric. It is read-only, hash-checked,
   and no experiment may touch it. If it changes, everything measured before is void.
2. **Every number comes from a logged run.** No estimates, no numbers typed by hand. Each figure in
   every document points to a result file and a command that regenerates it.
3. **A gain inside the noise is not a gain.** Everything runs on at least 3 random seeds, reported as
   mean ± spread. We know each dataset's noise floor (fly connectome 0.019, mouse cortex 0.0006,
   mouse 0.26 percentage points) and a result must clear it.
4. **Three datasets, always.** Two large connectomes from different species plus a small one. A
   result on one graph is not a result.
5. **The algorithm never sees the target.** The score is computed only afterwards, by the frozen
   scorer, on a finished answer. The known best solution is used for measurement only and is
   read through a single gated interface.
6. **One experiment, one commit.** Any result can be reproduced by checking out that commit and
   running the logged command.

## How an idea is tested

Ideas climb a ladder of increasing cost, and most die on the first rung.

```
idea  →  is it different from what's already dead?     free
      →  prototype on a small graph                    minutes
      →  screen: 3 seeds × 3 datasets                  ~1 hour
      →  confirm: more seeds, confidence interval      hours
      →  red team: someone tries to break it           minutes
      →  verdict written down, with a revival condition
```

The red-team step is not a formality. It once demolished one of our own published answers and we
rewrote it — see below.

Roughly **25 hypotheses** have gone through this. **Three** survived.

## What we learned

**1. Where you start matters; how you descend does not.**
Eight different ways of changing the optimizer — learning rate, schedule, noise, a different
optimizer entirely — all landed back on the same plateau. The only thing that helped was giving it a
better starting order.

**2. The real gain came from abandoning the smooth objective.**
Rocket optimizes a *smoothed* version of the score, because you cannot take a derivative of a
counting problem. After Rocket converges, we run a cheap discrete cleanup that repeatedly moves each
neuron to its exactly-best position. That single step is worth +0.85 points, more than everything
else combined, and costs zero optimizer steps.

**3. We found out why — and it is a measurement, not a story.**
The smoothed objective judges a connection by *how far apart* its two neurons are, not just by which
comes first. Two neurons standing close together look identical to it, so it awards them half credit
regardless of order. On this graph its resolution is about **290 neurons out of 136,648**: it simply
cannot see the ordering at any finer detail. The good solution earns much of its advantage from
exactly those fine-grained wins — so at the scale the optimizer actually works at, **it rates the
better solution as worse than the one it found itself.** That is the whole reason gradient methods
plateau here, and the reason the discrete cleanup works: it counts a one-position win exactly like a
twenty-thousand-position win, as the true metric does.

**4. Fixing that does not help — which was the surprise.**
We built a modified objective that provably prefers the better solution. Training with it came out
**worse**, by up to 2.4 points, and we measured why: the fix crowds out the signal that spreads
neurons apart, and the resolution ends up worse than before. So the limit is not what the objective
*prefers* — it is what gradient descent can *do*. That closes the "just design a better objective"
line of attack with an experiment instead of an argument.

## The part worth copying

Most of this project is negative results, and they are treated as the product, not as failure. Each
kill is written down with the evidence and with the condition under which the idea may come back.
That index is what stops the work from going in circles, and it is why the surviving three results
are trustworthy.

The most useful single day was the one where an adversarial review broke our own explanation of why
starting from the best solution loses score. The old answer was well-written, internally consistent,
and wrong. It survived for weeks because nobody had tried to break it. Now it is a rule.

## Where things are

| what | where |
|---|---|
| the rules, in full | `experiments/PROTOCOL.md` |
| conclusions, ranked | `experiments/findings.md` |
| the narrative record of every experiment | `experiments/log.md` |
| open ideas | `experiments/backlog.md`, `experiments/roadmap.md` |
| mechanism explanations | `experiments/diagnosis.md` |
| raw results | `results/*.json` |
