# Windows launch runbook — autonomous MFAS campaign

**Purpose.** Everything that could be re-verified analytically before the port has been. This
file records what was checked here (macOS, read-only-for-results), and gives the exact,
ordered steps + go/no-go gates to run **on the Windows/RTX 4060 box** where the campaign
actually executes. Branch: `generator/phase0`. Only the oracle is frozen; nothing below edits it.

Convention: `$PY = /c/ProgramData/anaconda3/envs/allen/python.exe` (Git Bash), run from the
`nn_backward_connections/` checkout.

---

## A. What was verified analytically on this machine (no Windows needed)

| # | check | result |
|---|---|---|
| 1 | Frozen oracle untouched vs `main` | `git diff main -- src/mfas/metrics.py tests/test_metrics.py eval/harness.py` is **empty** |
| 2 | Oracle unit test | `test_metrics.py` **8/8**; parity anchor 34,751,902 / 41,912,141 = 82.916074% |
| 3 | Full test suite (this branch) | **385 passed, 1 xfailed** (the xfail is the CUDA-champion-retention debt, closed by step D3 below) |
| 4 | Hardcoded POSIX paths in new code | **none** — `device_policy.py`, `reversal_subsumption.py`, `watch.py`, `pin_champion.py`, `residual_1opt.py` all use `Path(__file__)`-relative resolution |
| 5 | Driver Windows-readiness | already Windows-native: `taskkill //T //F //PID`, `ps -W`, interpreter default `/c/ProgramData/...`, avoids GNU-only `date -d`. It ran all of Phase-7 on Windows. |
| 6 | Metric-leakage in new code | clean — `pin_champion`/`residual_1opt` use the frozen oracle only to score an order that **already exists**; the reference is read solely through the `mfas.analysis.gap` firewall |
| 7 | campaign.yaml is CUDA-configured | `python: /c/.../python.exe`, `device_tag: "NVIDIA GeForce RTX 4060 Laptop GPU"`, `python_version: 3.9.25` — so `device_policy` classifies runs as `cuda` and gating is allowed |
| 8 | Frozen-guard SCRIPT logic | pipe-tested: blocks frozen files, `sota.json`, `best_solution`, `results/*.json`, and out-of-sandbox writes, for **both** posix and `D:\...` backslash paths (`canon()` folds `\`→`/`, `D:`→`/d/`) |
| 9 | Campaign self-gate | `verify_cycle.py --all` → **VERDICT: PASS** (0 FAIL, 10 WARN) after the launch fixes |
| 10 | `--force` self-promotion hole | closed — human-only via `MFAS_HUMAN_OVERRIDE`, which the driver never sets |

## B. The five launch blockers, and their state

| # | blocker | state | closed by |
|---|---|---|---|
| 1 | campaign self-gate red + `--force` bypass | **DONE** | commit `8c378d3` |
| 2 | nothing watches a dead run | **DONE** (code) — install per C below | commit `48067e6` |
| 3 | frozen-guard hook dead off one path | **DONE** — verify firing per D1 | commit `2cebfe7` |
| 4 | champion orderings never left the RTX box | **tool ready** — run D3 on Windows | `tools/pin_champion.py` |
| 5 | MPS device noise vs CUDA gating | **DONE** — `device_policy` refuses cross-device deltas | commits `ceaee70`, `13e93f7` |

Only #4 and the *verification* of #2 and #3 require the Windows machine. Everything else is done.

---

## C. Port steps (Windows/RTX box, in order)

```bash
# C1. Get the branch
git fetch && git checkout generator/phase0
git rev-parse --show-toplevel     # sanity: the nn_backward_connections checkout

# C2. Sanity that the environment matches campaign.yaml
$PY -c "import torch,sys;print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"
#   expect: 3.9.25  2.8.0+cu128  True

# C3. Oracle integrity — MUST pass before anything runs
$PY -m pytest tests/test_metrics.py -q          # expect 8 passed
```

---

## D. Go / No-Go gates (each MUST pass; a failure halts the launch)

### D1 — the frozen guard actually FIRES under Claude Code (blocker 3)
The script is proven correct; this proves Claude Code invokes it at the new `$CLAUDE_PROJECT_DIR`
path on Windows. Start Claude Code fresh in the checkout (so it reloads settings), then:

```bash
export MFAS_HOOK_LOG="$PWD/autoresearch/logs/hook_probe.log"   # forensic trace
```
Ask Claude (or via a probe) to **Write one character to `src/mfas/metrics.py`**.
- **GO**: the edit is **denied** with the "FROZEN file" message, AND `hook_probe.log` contains
  a line for `metrics.py` (proof the hook ran).
- **NO-GO**: the edit succeeds, or the log is empty → the hook is not wired. Do **not** launch.
  Fallback: in `.claude/settings.json` set the command back to the absolute path of *this*
  checkout's `.claude/hooks/protect-frozen-files.sh`, restart, retest.

### D2 — the watchdog delivers (blocker 2)
```bash
$PY autoresearch/watch.py --test          # GO: prints a nonce and "ALERT file carries the nonce: True"
$PY autoresearch/watch.py --once ; echo $?  # with driver down: prints NO_DRIVER, exit 5
```
Optional real channel: `export MFAS_ALERT_WEBHOOK=<your Telegram/Slack/ntfy URL>` before the run,
and the watcher POSTs every alert there.

### D3 — champions are retained for cross-machine evaluation (blocker 4)
Run **on the RTX box**, where the champion orderings actually exist:
```bash
$PY tools/pin_champion.py results/<H42-connectome-s42 record>.json
$PY tools/pin_champion.py results/<champion-microns record>.json
$PY tools/pin_champion.py results/<champion-mouse   record>.json
git add results/champions/ && git commit -m "pin CUDA champions for cross-machine evaluation"
```
- **GO**: each prints `pinned ... pct=... device=cuda`; the tool re-scores through the frozen
  oracle and refuses on any mismatch. After this, `pytest tests/test_device_policy.py` turns the
  xfail into an XPASS — then remove the `@pytest.mark.xfail` marker on
  `test_cuda_champion_ordering_is_retained_in_the_repo`.
- (Find the exact record filenames with `ls results/ | grep -E 'H42-connectome-s42.*implement'`.)

### D4 — the campaign self-gate is green on this machine
```bash
$PY autoresearch/verify_cycle.py --all     # GO: VERDICT: PASS
```

---

## E. Launch

```bash
# foreground driver (writes autoresearch/logs/driver.log; touch autoresearch/STOP to end after the cycle)
nohup bash autoresearch/driver.sh > autoresearch/logs/nohup.out 2>&1 &

# the watchdog beside it — install under Task Scheduler for a real supervisor, or loop by hand:
$PY autoresearch/watch.py --watch --every-min 5
```

**Supervisor (recommended).** A Task Scheduler entry running `watch.py --once` every 5 min,
configured to email/notify on a non-zero exit code, is a complete out-of-band supervisor: exit
`3` stale, `4` blocked, `5` no-driver, `6` no-progress. Without it, `--watch` in a terminal is
the minimum.

**Stop conditions the driver already honors:** `touch autoresearch/STOP` (graceful), the 24h
rolling wall-clock budget in `campaign.yaml`, `max_cycles`, and disk floors. On a blocking
problem the driver sets `state.json mode=blocked` and exits — which the watchdog reports.

---

## F. What the campaign will do, and the honest caveats

- The queue has **8 `proposed`** items at priority 0; the driver runs the `/research-cycle` ladder
  (novelty → prototype → screen → confirm → critic) per `CAMPAIGN.md` and commits one cycle each.
- **No champion is promoted without a passing `audit.py --gate promotion`**, and `--force` is now
  human-only. A green campaign cannot crown itself.
- **New cycles must record `gates_run`** (mandatory from cycle 9); the self-gate FAILs otherwise.
- Everything measured on this CUDA box is comparable; the **MacBook is for evaluating results
  only** — `device_policy` refuses to compare an MPS number against a CUDA champion (offset
  0.0257 pp = 2.1× the gate).

**Open, not blocking launch:** the ARC quality layer (theorist, generator, honest `config_hash`
with device in the group key) is designed (`GENERATOR_NEXT_STEPS.md`) but not built; it is folded
into the *running* campaign later, one component at a time. Launch does not wait on it.
