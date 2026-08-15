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
it **detached**, and then **poll** it — never end your turn waiting.

```bash
bash autoresearch/detach.sh dr_tmp/proto_<id>.out $PY dr_tmp/proto_<id>.py --arm a
tail -5 dr_tmp/proto_<id>.out       # poll in a loop until it finishes; keep your turn
```

Use `detach.sh` rather than writing the launch by hand: on this box `setsid` **does not exist**
and `nohup <python.exe> &` does **not** outlive the session — only an MSYS `bash -c` wrapper that
keeps python as its child does (measured 2026-08-15; this file recommended the non-existent
`setsid` until then). `detach.sh` is that wrapper and nothing else; do not "simplify" it away.

Record the job in `state.json.current_item_note` (what, where the output lands, when it started)
before you do anything else, so a cycle that dies anyway is resumed rather than duplicated. Cycle
#4 skipped that step and nearly had two copies of the same 75-minute CPU prototype racing each
other.

**Screen.** Use the `implementer` subagent. Isolated module `src/mfas/experiments/<id>.py`,
3 datasets (connectome, microns, mouse), `--role implement`, compared to the **champion** from
`sota.json`. The **seed count is per variant** (P02, `PROTOCOL.md § Phase-7.4`): a variant that
never draws from `seed` screens at 1 seed on the primaries, everything else at 3, and mouse
always at 3 as the tripwire. `--auto-seeds` decides it — do not hand-pick seeds:

```bash
$PY autoresearch/seed_plan.py --variant <id> --role implement          # the plan, and why
bash autoresearch/sweep.sh --exp <id> --role implement --auto-seeds    # detached; returns at once
while :; do                                    # poll — branch on the exit code, never `while !`
  bash autoresearch/waitfor.sh; rc=$?
  case $rc in
    0)     echo "sweep finished"; break ;;     # DONE (rc= inside the output may still be non-zero)
    10)    continue ;;                         # still running -> poll again, do NOT end your turn
    20|21) echo "sweep aborted/crashed (rc=$rc) — READ its report"; break ;;
    2)     echo "nothing was launched"; break ;;
    *)     echo "unexpected waitfor rc=$rc"; break ;;
  esac
done
```

`waitfor.sh` exit codes: **0** done, **10** still running (call again), **2** nothing launched,
**20** aborted from outside, **21** crashed. 20 and 21 are terminal — a `while ! waitfor` loop
spins on them forever, which is why the loop above branches. On 20/21 the runs it lists as
COMPLETED wrote **real** `results/*.json`: reuse them and re-launch only the ones it lists as
NOT DONE. The next `sweep.sh` reclaims a crashed/aborted `.sweep/` automatically — never `rm`
anything under `autoresearch/.sweep/` by hand.

Record `class=deterministic|rng` and the seeds actually run in the log entry. If the three mouse
runs disagree for a variant classified deterministic, the classification is falsified: the
primary numbers are **void** and it must be re-screened at 3 seeds.

**Never run a sweep in the foreground, and never end your turn while one is in flight.** A
microns run is ~3240 s on this machine — longer than a single Bash call may last. `sweep.sh`
detaches the runs so they survive even if your session does, and `waitfor.sh` blocks in bounded
slices (exit 10 = still running, call again) so you keep your turn. If you come back to a cycle
and `waitfor.sh` reports runs as COMPLETED, those results are real — use them rather than
re-running. See § Lessons that changed the rules for what this cost.

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
    --role confirm --comparator-role confirm --gate promotion \
    --out autoresearch/audit_<id>.json
```

A non-zero exit is a hard stop for promotion.

**Use `--gate promotion` here.** The default `--gate report` reports significance as INFO and
degeneracy as WARN and therefore exits 0 on evidence that cannot support a champion change; until
2026-08-15 that was the ONLY mode and `screen_delta_pp` was evaluated by no code at all. Strict
mode enforces the `promotion_gate` block in `campaign.yaml`: per-dataset minimum effect size,
`protocol_ci_lower > 0` on both primaries, mouse non-inferiority, a non-empty comparator pool, and
**provenance** — the backing runs must come from a commit that contains the variant's own module.

That provenance gate currently FAILS for both H36 and H42: all 36 H36 runs cite
`9d43b977+dirty`, where `src/mfas/experiments/H36.py` does not exist, and 32 of H42's confirm runs
cite `1865372b+dirty`, where `H42.py` does not exist. The module was committed one commit later in
the same cycle each time, and the runs were never re-derived. **So: commit the variant module
BEFORE you launch the screen, not after the confirm.** Otherwise the numbers you promote are not
reproducible by checkout, which is CLAUDE.md invariant 5 and CAMPAIGN.md rule 3.

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
- **Record which gates you actually ran.** The new history entry MUST carry a `gates_run` list:

  ```json
  {"cycle": 8, "item": "H41", "outcome": "keep",
   "gates_run": ["novelty", "prototype", "screen", "confirm", "critic"]}
  ```

  A `keep` requires all five. If you skipped one, say so by leaving it out — do not list a gate you
  did not run. This exists because cycle #5 skipped the critic rung for TIME reasons (confirm ran
  to 16:14 against a 17:08 driver cap) and that survived only because the cycle volunteered it in
  prose; nothing mechanical recorded it, and `audit.py` exited 0 with the rung unrun while
  `update_sota.py` promoted the champion. `autoresearch/verify_cycle.py` now FAILs a `keep` whose
  `gates_run` does not cover the ladder.
- **Verify your own bookkeeping before you commit**, and fix what it finds:

  ```bash
  $PY autoresearch/verify_cycle.py            # exit 0 required
  ```

  It re-derives the record from artifacts: every score figure in your `log.md` section must appear
  in a `results/*.json` or `experiments/outputs/*.json` (CAMPAIGN.md rule 3, previously enforced by
  nothing), the history must be contiguous, and each champion in `sota.json` must have backing
  runs. A figure that is legitimately foreign — a prior machine, the published reference — goes in
  `autoresearch/known_figures.json` **with a reason**, never left bare. If a number you want to
  quote lives only in gitignored `dr_tmp/`, promote the artifact instead of quoting it: cycle #3
  quoted four prototype figures that exist nowhere tracked, and they are unverifiable today.
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
  new cycle starts from files. (What this has cost is in § Lessons that changed the rules.)
  - Anything longer than a few minutes — sweeps AND prototypes AND ad-hoc scripts — must be
    launched **detached** (`autoresearch/sweep.sh` for sweeps, `autoresearch/detach.sh` for
    anything else) so it outlives the session, and then **polled in a loop**
    (`autoresearch/waitfor.sh`, or repeated bounded `tail`/`kill -0` checks) so you keep your turn.
  - Before launching, check whether the job is **already running** — a duplicate CPU-bound
    prototype contends with the original and corrupts both timings. For sweeps, `sweep.sh` checks
    for you: it refuses while a runner is genuinely alive, and reclaims a dead one by itself.
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

### Lessons that changed the rules

Post-mortems live **here**, with their numbers, so the rules above stay short and so a rule can be
retired when its lesson stops applying. Every figure below is traceable to `results/*.json`,
`autoresearch/logs/` or `autoresearch/.sweep/`; if you find one that is not, correct it in place
and say so in `experiments/log.md` — a document that misquotes its own evidence is worth less than
no document.

- **Cycle #1 (2026-08-09) — ending a turn kills the job.** It ran H35/connectome/verify to
  completion (533 s, finished 21:06:15; that result was kept and used), started the microns run at
  ~21:06:30, then ended its turn. The headless session exited at 21:11:01 and took the child with
  it — **~271 s into microns**, which the next cycle then had to redo: about **524 s** of
  duplicated GPU time. *(Corrected 2026-08-15: this document claimed "~50 minutes lost outright"
  until then. The cycle log's "microns has ~50 minutes left" meant time REMAINING, and was
  misread. The rule stands — it is independently justified by cycles #4 and #7 below — but the
  cost was ~9 minutes, not ~50.)* → produced `sweep.sh` + `waitfor.sh`.
- **Cycle #4 (2026-08-10) — bookkeeping is part of the job.** At the H42 prototype rung it
  launched five arms of ~75 min CPU, wrote "I'm waiting on the H42 prototype", and stopped. It
  survived only because the job happened to be detached, and it left `current_item` unset, so the
  next cycle nearly started a duplicate racing the original. → produced the `current_item` /
  `current_item_note` rules.
- **2026-08-11 — a killed sweep must not read as "never launched".** An operator killed an
  H42/microns verify run; the old `.sweep/running`+`.sweep/done` flag pair left NEITHER file, and
  `waitfor.sh` reported "nothing launched" — indistinguishable from an unstarted sweep, so a
  resuming cycle could have redone or skipped hours of GPU work. → produced the single atomic
  `.sweep/status` file, the derived `crashed`/`aborted` states, automatic reclaim, and exit codes
  20/21 (2026-08-15).
- **2026-08-15 — the detach recipe named a binary that does not exist.** This file told cycles to
  use `setsid`, which is not installed under Git Bash; and `nohup <python.exe> &`, the obvious
  fallback, does not survive a session exit either. Only an MSYS `bash -c` wrapper does. →
  produced `autoresearch/detach.sh`.
