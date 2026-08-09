---
description: Run ONE autonomous research cycle of the Phase-7 campaign (plan → gate ladder → decide → commit).
---

You are running **one complete research cycle** of the autonomous MFAS campaign, unattended.
Nobody will answer questions during this cycle — every decision is yours, and every decision must
be defensible from the artifacts you leave behind. A fresh context will run the next cycle and
will know only what you write to disk.

Work in `nn_backward_connections/` (the campaign root). All paths below are relative to it.

## 0. Preflight — abort the cycle if any of this fails

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python
git status --porcelain | head -20          # expect a clean-ish tree; NEVER proceed with frozen-file edits
git rev-parse --abbrev-ref HEAD            # MUST be auto/campaign
PYTHONPATH=src $PY -m pytest tests/ -q     # MUST be green (25 passed)
```

If the branch is not `auto/campaign`, or a frozen file is modified, or tests fail: write the
reason into `autoresearch/state.json` (`"mode": "blocked"`), print it, and stop. Do not "fix" a
frozen file. Ever.

## 1. Orient (read, in this order)

1. `autoresearch/CAMPAIGN.md` — the constitution. Non-negotiable.
2. `autoresearch/state.json` — phase, cycle number, `consecutive_kills`, `current_item`.
3. `autoresearch/sota.json` — the champion you must beat, per dataset.
4. `autoresearch/queue.json` — candidate hypotheses.
5. `autoresearch/killed.json` — what is already dead, and under what condition it may return.
6. `autoresearch/campaign.yaml` — thresholds, seeds, budgets, gate ladder.

If `state.json.current_item` is non-null, the previous cycle died mid-flight: **resume that item**
from the last stage that has artifacts on disk rather than starting a new one.

## 2. Choose the mode

- `consecutive_kills >= 3` → **divergent mode** (see CAMPAIGN.md § Escalation). Do not propose a
  variant of the current design. Scout the literature, re-read `experiments/diagnosis.md` Q01/Q02,
  and attack a different level (move class / decomposition / formulation). Write your reasoning to
  `autoresearch/lit/` *before* proposing anything.
- `cycles_since_literature_scan >= 8` → run a **literature scan** first (scout subagent).
- fewer than 3 viable queue items → **ideate** first (ideator subagent), deduping against
  `killed.json`.
- otherwise → **incremental mode**: take the top-priority `status: proposed` item.

## 3. Run the gate ladder — never skip a rung

**Novelty.** Does the item share an axis with anything in `killed.json`? If so, it must satisfy a
stated revival condition; name it in the log. If it does not, drop the item (status
`dropped-by-novelty`) and pick the next one. This costs nothing and saves hours.

**Prototype.** Cheap proxies only — mouse, `gap.make_hard_synthetic_graph`, and/or an SCC
subgraph. Minutes of CPU, no large-graph compute. Write the outcome to
`experiments/outputs/proto_<id>.json`. If the mechanism shows no signal, kill it here: this rung
is what made Phase 6 cheap.

**Screen.** Use the `implementer` subagent. Isolated module `src/mfas/experiments/<id>.py`,
3 seeds (42/123/999) × 3 datasets (connectome, microns, mouse), `--role implement`, compared to
the **champion** from `sota.json`:

```bash
for DS in connectome microns mouse; do for S in 42 123 999; do
  $PY -m eval.run_variant --exp <id> --dataset $DS --seed $S --out results/ --role implement --device auto
done; done
```

Pass iff `Δ > screen_delta_pp` on **both primaries** (connectome, microns) and mouse is
non-inferior. Run these **sequentially** — one MPS device; parallel runs contend and poison
`wall_clock_s`.

**Confirm.** Only if the screen passed. Use the `verifier` subagent (read-only on source, generates
its own runs at `--role confirm`, confirm seeds from `campaign.yaml`). Pass iff the Welch 95% CI
lower bound > 0 on both primaries and mouse is non-inferior.

**Critic.** Only if confirm passed. Use the `critic` subagent, and run the mechanical audit
yourself as well — the critic adjudicates, the audit computes:

```bash
$PY autoresearch/audit.py --variant <id> --comparator champion \
    --role confirm --comparator-role confirm --out autoresearch/audit_<id>.json
```

A non-zero exit is a hard stop for promotion.

## 4. Decide, record, commit

Verdict is one of **keep / kill / iterate**, per the ladder. Then:

- Append a full cycle entry to `experiments/log.md` in the PROTOCOL format (hypothesis, per-stage
  numbers with seeds and file ids, critic verdict, decision, re-runnable commands).
- **kill** → add an entry to `autoresearch/killed.json` with `why_dead` **and a revival condition**;
  set the queue item's status; increment `consecutive_kills`.
- **keep** → update `autoresearch/sota.json` via `$PY autoresearch/update_sota.py --variant <id>
  --dataset <ds> ...`, add a ranked section to `experiments/findings.md`, reset `consecutive_kills`
  to 0.
- **iterate** → keep the item, record what changes next time, and count it toward
  `consecutive_kills` if it did not improve.
- Regenerate the dashboard: `$PY autoresearch/dashboard.py`.
- Update `autoresearch/state.json`: `cycle += 1`, `current_item: null`, streaks, outcome, history.
- Commit everything as **one** commit:
  `git add -A && git commit -m "phase7 <id>: <verdict> — <one-line result>"`
  Never `git push`. Never merge into another branch.

## 5. Stop conditions (write the reason into `state.json` and exit)

- The Phase-1 exit criteria in `campaign.yaml` are met → set `"mode": "blocked"` with a note that
  the human must review before Phase 2 starts. This is a success, not a failure.
- The audit fails in a way you cannot explain except as a harness bug.
- The frozen manifest fails.
- The queue is empty and ideation produced nothing that survives the novelty gate.

## Standing rules

- One hypothesis per cycle. Depth over breadth.
- Most cycles end in **kill**. That is the expected outcome and a good one — provided the kill is
  written down with its revival condition.
- Never report a number you did not produce; never keep a number the audit contradicts.
- Sequential heavy runs only. One MPS device.
- Keep every run under the `runtime.max_wall_clock_s_per_run` budget (3600 s). A variant that
  cannot answer in an hour is not a usable algorithm — kill it or make it cheaper.
- Scratch goes to `dr_tmp/` (gitignored). Anything that matters gets promoted to its track home.
- If you are unsure whether something counts as a win: it does not. Write down why it was
  ambiguous and what measurement would settle it.
