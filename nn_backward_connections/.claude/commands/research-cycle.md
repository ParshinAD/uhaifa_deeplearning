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
PY=/c/ProgramData/anaconda3/envs/allen/python.exe
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

If a prototype will take more than a few minutes, it gets the same treatment as a sweep: launch
it **detached** (`setsid python dr_tmp/proto_<id>.py … </dev/null > dr_tmp/proto_<id>.out 2>&1 &`)
and then **poll** it — never end your turn waiting. Record it in `state.json.current_item_note`
(what, where the output lands, when it started) before you do anything else, so a cycle that dies
anyway is resumed rather than duplicated. Cycle #4 skipped that step and nearly had two copies of
the same 75-minute CPU prototype racing each other.

**Screen.** Use the `implementer` subagent. Isolated module `src/mfas/experiments/<id>.py`,
3 datasets (connectome, microns, mouse), `--role implement`, compared to the **champion** from
`sota.json`. The **seed count is per variant** (P02, `PROTOCOL.md § Phase-7.4`): a variant that
never draws from `seed` screens at 1 seed on the primaries, everything else at 3, and mouse
always at 3 as the tripwire. `--auto-seeds` decides it — do not hand-pick seeds:

```bash
$PY autoresearch/seed_plan.py --variant <id> --role implement          # the plan, and why
bash autoresearch/sweep.sh --exp <id> --role implement --auto-seeds    # detached; returns at once
while ! bash autoresearch/waitfor.sh; do :; done                       # poll until it prints DONE
```

Record `class=deterministic|rng` and the seeds actually run in the log entry. If the three mouse
runs disagree for a variant classified deterministic, the classification is falsified: the
primary numbers are **void** and it must be re-screened at 3 seeds.

**Never run a sweep in the foreground, and never end your turn while one is in flight.** A
microns run is ~3240 s on this machine — longer than a single Bash call may last. Cycle #1
(2026-08-09) learned this the expensive way: it started microns in the foreground-ish, stopped its
turn to avoid GPU contention, the headless session exited, the child died with it, and ~50 minutes
of GPU time produced no result. `sweep.sh` detaches the runs so they survive even if your session
does, and `waitfor.sh` blocks in bounded slices (exit 10 = still running, call again) so you keep
your turn. If you ever come back to a cycle and find `autoresearch/.sweep/log` already complete,
those results are real — use them rather than re-running.

Pass iff `Δ > screen_delta_pp` on **both primaries** (connectome, microns) and mouse is
non-inferior. The sweep runs **sequentially** by construction — one GPU device; parallel runs
contend and poison `wall_clock_s`.

Note `screen_delta_pp` is a **minimum effect size**, not 2σ: the champion pipelines are
deterministic here (σ = 0), so a Welch CI on these seeds is degenerate. See `experiments/log.md`
P01 and the `significance.<ds>.degenerate` audit check.

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
- Sequential heavy runs only. One GPU device (RTX 4060).
- **Never end your turn while any job is running — this applies to EVERY rung, not just the
  screen.** There is no "I'll resume when it reports": your turn ending ends the session, and a
  new cycle starts from files. This has now cost two cycles. Cycle #1 lost a ~50 min microns run
  outright. Cycle #4 (2026-08-10) reached the H42 prototype rung, launched five arms of ~75 min
  CPU, wrote "I'm waiting on the H42 prototype", and stopped — it survived only because the job
  happened to be detached, and it left `current_item` unset so the next cycle nearly started a
  duplicate against it.
  - Anything longer than a few minutes — sweeps AND prototypes AND ad-hoc scripts — must be
    launched **detached** (`autoresearch/sweep.sh`, or `setsid`/`nohup … </dev/null &` for a
    one-off) so it outlives the session, and then **polled in a loop** (`autoresearch/waitfor.sh`,
    or repeated bounded `sleep`+check calls) so you keep your turn.
  - Before launching, check whether the job is **already running** — a duplicate CPU-bound
    prototype contends with the original and corrupts both timings.
- **Set `current_item` the moment you pick an item, not at the end.** It is the only thing that
  tells the next cycle to resume rather than restart. If you launch a long job, record in
  `current_item_note` what was launched, where its output lands, and when it started.
- **Finish the bookkeeping even if the science is unfinished.** Before you stop for any reason,
  update `state.json` / `queue.json`, append what you learned to `experiments/log.md`, and COMMIT.
  An interrupted cycle that committed its partial result is resumable; one that did not leaves the
  next cycle a dirty tree and no record of what was already established.
- Keep every run under the `runtime.max_wall_clock_s_per_run` budget (3600 s). A variant that
  cannot answer in an hour is not a usable algorithm — kill it or make it cheaper.
- Scratch goes to `dr_tmp/` (gitignored). Anything that matters gets promoted to its track home.
- If you are unsure whether something counts as a win: it does not. Write down why it was
  ambiguous and what measurement would settle it.
