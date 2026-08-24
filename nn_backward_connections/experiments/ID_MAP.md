# Variant ID map — the authoritative namespace

**Read this before allocating a new hypothesis ID, and before trusting any citation of `H36`–`H39`
written between 2026-08-09 and 2026-08-18.**

For ten days the project ran on two branches in parallel (`auto/campaign` on Windows/CUDA,
`phase6-global-discrete` on macOS/MPS). Both allocated IDs from the same `H##` space without
coordinating, so a few numbers came to mean two different things. The branches were unified into
`main` on 2026-08-25; this file is the record of what collided and how it was resolved.

## The allocation rule, from now on

**One namespace, one allocator.** `autoresearch/queue.json` → `next_free_id` is the only source of
truth for the next `H##`, and `next_free_protocol_id` for the next `P##`. Take the value, use it,
increment it in the same commit that files the hypothesis. Never reuse a retired ID.

The `A-*` labels (`A_INIT`, `A-SUB`, `A-ALT`, `A-VIOL`, `A-MBAND`, `A-SURR`) are a **separate**
namespace used on the phase6 track for ideas that had not yet earned a number. They never collide
with `H##`, and an `A-*` idea gets an `H##` only when it is implemented as a variant module.

## Resolution principle

Where two hypotheses shared an ID, **the side with committed code and logged runs kept it**, and
the side with neither was renamed. This is not a judgement about which idea was better: it is
because `results/*-<ID>-*.json` filenames embed the ID, and renaming a variant that has result
records would break the evidence trail that the whole protocol rests on.

## The collisions

| ID | kept by | renamed to | why |
|---|---|---|---|
| `H37` | phase6 — algebraic-tail surrogate (`H37`, `H37B`) | campaign's → **`H54`** | phase6 has `src/mfas/experiments/H37.py`, `H37B.py` and 6 logged runs; the campaign's entry was `proposed` with no code and no runs. |
| `H38` | phase6 — one-sided (asymmetric) surrogate (`H38`, `H38B/C/D`) | campaign's → **`H55`** | phase6 has four committed modules and 18 logged runs; the campaign's entry was killed at the prototype gate, with its evidence in `experiments/outputs/`, never in `results/`. |

**What each ID now means, unambiguously:**

- **`H37` / `H37B`** — algebraic-tail surrogate. *Killed*: the surrogate TAIL axis is closed at
  equal core width. `src/mfas/experiments/H37{,B}.py`, `results/*-H37{,B}-*.json`.
- **`H38` / `H38B` / `H38C` / `H38D`** — one-sided (asymmetric) surrogate: flat above a margin,
  tanh below, so only violated edges receive gradient. *Graph-dependent*: **+0.3668 pp** on the fly
  connectome, **−0.6689 pp** on MICrONS. The largest gradient-side gain the project has produced,
  and the reason the A-VIOL idea exists. `src/mfas/experiments/H38*.py`, `results/*-H38*-*.json`.
- **`H54`** (was the campaign's `H37`) — cycle-triggered under-relaxation: switch to `α<1` only
  once a period-2 limit cycle is detected, instead of unconditionally after `k_full=6` sweeps.
  Still *proposed*.
- **`H55`** (was the campaign's `H38`) — Gauss-Seidel / block-sequential exact-gain sift.
  *Killed* at the prototype gate 2026-08-16: block-sequential loses to under-relaxed Jacobi at
  every block count and under both damping schedules.

## Not collisions, though they look like ones

- **`H36`** is **continuity, not a clash.** The phase6 backlog entry
  *"H36 — Collective discrete refinement: paired backward-edge relocation + block-SCC
  condensation"* is the **proposal**; the campaign's `H36` *"SCC-decomposed recursive bounded-span
  insertion"* is the block-SCC half of that same proposal, **implemented**. All 36
  `results/*-H36-*.json` are the campaign's. The other half of the proposal — paired backward-edge
  relocation — was implemented later, as `H45` → `H51` → `H52`.
- **`H39` vs `A-ALT`** — the same idea (alternate the discrete and gradient phases) under two
  labels, one per track. `H39` is its number; `A-ALT` is what the phase6 documents call it.

## What was deliberately *not* changed

`experiments/log.md`, `findings.md`, `diagnosis.md` and `backlog.md` are the historical record and
are append-only. **Their existing `H37`/`H38` citations were left exactly as written**, because
editing them would misrepresent what was known when each entry was made. Disambiguate them by
track: inside `log.md`, everything under *"Track 1 — Phase-7 autonomous campaign"* uses the
campaign's meaning (now `H54`/`H55`); everything under *"Track 2 — Phase-6 global-discrete &
surrogate track"* uses the surviving meaning above.

Only `autoresearch/queue.json` was rewritten, because it is live machine-readable **state** that
the driver and the ideator read to decide what to run next — not narrative. The renamed entries
carry `former_id` and a `collision_note`.
