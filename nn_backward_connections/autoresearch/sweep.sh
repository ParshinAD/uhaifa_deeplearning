#!/usr/bin/env bash
# Launch a DETACHED evaluation sweep and return immediately.
#
# Why this exists. A single microns run takes ~3240 s here, and an agent's Bash call is capped far
# below that, so a cycle cannot run one in the foreground. Cycle #1 (2026-08-09) hit the
# consequence: it started a microns run, ended its turn while waiting, and the headless session
# exited at 21:11:01, killing the child ~271 s into that run. The next cycle had to redo it —
# ~524 s of duplicated GPU time, not the "~50 minutes" this comment claimed until 2026-08-15
# (that figure was a misreading of a cycle-log line saying microns had ~50 minutes LEFT; see
# .claude/commands/research-cycle.md § Lessons that changed the rules). A small bill, but the
# failure mode is unbounded: every cycle would have repeated it forever.
#
# So: detach the sweep so it OUTLIVES the session that started it. If the cycle survives, it polls
# with waitfor.sh; if the cycle dies, the runs keep going and their results/*.json land anyway, and
# the next cycle finds them instead of redoing them.
#
# HOW the detach works, and why it is shaped EXACTLY like this (measured on this box 2026-08-15):
#   * `setsid` DOES NOT EXIST here (Git Bash / MSYS): `command -v setsid` finds nothing, so the
#     setsid branch that used to live at the bottom of this file was always dead code. Removed.
#   * `nohup <python.exe> &` does NOT survive the launching session's exit — nohup does not detach
#     a NATIVE Windows process, and neither does putting a bash launcher in front of it.
#   * `nohup bash -c '...' &` DOES survive. That is the only reason this file works: the detached
#     unit is an MSYS bash that owns python as its CHILD. Load-bearing — do not "simplify" the
#     wrapper away. `autoresearch/detach.sh` packages the same pattern for one-off jobs.
#
# Usage:
#   bash autoresearch/sweep.sh --exp H36 --role implement --auto-seeds   # seed count per P02 policy
#   bash autoresearch/sweep.sh --exp H36 --role implement                # flat 3 datasets x 3 seeds
#   bash autoresearch/sweep.sh --exp H36 --role implement --seeds 42     # 1 seed, everywhere
#   bash autoresearch/sweep.sh --exp H36 --datasets connectome,mouse --seeds "42 123 999"
# then poll:
#   bash autoresearch/waitfor.sh            # repeat until it prints DONE
#
# --auto-seeds (P02, PROTOCOL.md § Phase-7.4) asks autoresearch/seed_plan.py how many seeds this
# particular variant needs PER DATASET: a variant that never draws from `seed` screens at 1 seed
# on the primaries (mouse always keeps 3, as the tripwire). It is per-dataset, which is why it
# cannot be expressed with the flat --seeds. The classifier is fail-safe — anything it cannot
# resolve comes back "rng" and keeps 3 seeds — and `--role confirm` is never touched by it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

EXP=""; ROLE="implement"; DATASETS="connectome,microns,mouse"; SEEDS=""; AUTO=0
while [ $# -gt 0 ]; do
  case "$1" in
    --exp)         EXP="$2"; shift 2 ;;
    --role)        ROLE="$2"; shift 2 ;;
    --datasets)    DATASETS="$2"; shift 2 ;;
    --seeds)       SEEDS="$2"; shift 2 ;;
    --auto-seeds)  AUTO=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -z "$EXP" ] && { echo "--exp is required" >&2; exit 2; }

# Strict allowlists (added 2026-08-15, defect D4). These values used to be interpolated into a
# `bash -c "... EXP='$EXP' ..."` string, so a value containing a single quote broke out of the
# quoting and became shell code. They now travel in the ENVIRONMENT instead (see the launch at the
# bottom), and are validated here as well: no legitimate campaign value needs anything else, and a
# value that does is far more likely to be a mistake than an intention.
case "$EXP"      in *[!A-Za-z0-9_-]*)  echo "REFUSED: --exp must match [A-Za-z0-9_-]+ (got: $EXP)" >&2; exit 2 ;; esac
case "$ROLE"     in ''|*[!A-Za-z0-9_-]*) echo "REFUSED: --role must match [A-Za-z0-9_-]+ (got: $ROLE)" >&2; exit 2 ;; esac
case "$DATASETS" in ''|*[!A-Za-z0-9_,-]*) echo "REFUSED: --datasets must be comma-separated [A-Za-z0-9_-]+ (got: $DATASETS)" >&2; exit 2 ;; esac
case "$SEEDS"    in *[!0-9\ ,]*) echo "REFUSED: --seeds must be digits separated by spaces or commas (got: $SEEDS)" >&2; exit 2 ;; esac

AR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$AR/.sweep"
mkdir -p "$STATE"

# ── .sweep status (SHARED BLOCK — keep in sync with the copy in waitfor.sh) ───────────────────
# Rewritten 2026-08-15 (defect D1). The old layout was two flag files, `.sweep/running` and
# `.sweep/done`, and it had two holes:
#   * a sweep killed from OUTSIDE left NEITHER file. That happened on 2026-08-11, when an operator
#     killed an H42/microns verify run mid-flight; waitfor.sh then printed "nothing launched", so a
#     killed sweep and an unstarted one were INDISTINGUISHABLE and a resuming cycle could silently
#     redo — or skip — hours of GPU work;
#   * a runner that died between writing `running` and writing `done` left `running` forever, and
#     every later sweep refused until a human deleted it: a silent, permanent campaign stop.
# Now there is ONE status file, written to a temp name in the same directory and mv'd into place,
# carrying state / launcher pid / runner pid / timestamps / exp / role / plan. `state=running` is
# only believed while the recorded runner pid is still ALIVE; otherwise the state is DERIVED as
# `crashed`, and the next sweep reclaims it by itself, loudly. No path here needs a human `rm`.
# A `.sweep/` left over in the old layout (bare `running`/`done`, or a log with neither) is still
# understood on read; those files are never written again.

# Liveness of a pid WE recorded. The pids in the status file are MSYS/POSIX pids (the detached unit
# is an MSYS bash — see the header), so plain `kill -0` is correct here and portable to macOS and
# Linux; no `ps -W`/`taskkill` translation is needed or attempted. `ps -p` is a fallback for hosts
# where signalling is restricted. Returns 0 alive, 1 dead, 2 unknown (no usable pid recorded).
# Accepted caveat: a RECYCLED pid reads as "alive", which errs toward waiting on a finished sweep
# rather than declaring a live one crashed — the safe direction.
sweep_pid_alive() {
  case "${1:-}" in ''|*[!0-9]*) return 2 ;; esac
  kill -0 "$1" 2>/dev/null && return 0
  ps -p "$1" >/dev/null 2>&1 && return 0
  return 1
}

# mtime in epoch seconds: GNU stat, then BSD stat, then give up (empty = "unknown", never guessed).
sweep_mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || true; }

# Read the sweep state into ST_*. ST_state is the DERIVED state and is always one of:
#   none | running | done | aborted | crashed
sweep_status_read() {
  ST_state="none"; ST_launcher_pid=""; ST_runner_pid=""; ST_exp=""; ST_role=""
  ST_started=""; ST_ended=""; ST_rc=""; ST_plan=""; ST_note=""; ST_source="none"
  local line key val alive mt age
  if [ -f "$STATE/status" ]; then
    ST_source="status"
    while IFS= read -r line || [ -n "$line" ]; do
      key="${line%%=*}"; val="${line#*=}"
      [ "$key" = "$line" ] && continue
      case "$key" in
        state)        ST_state="$val" ;;
        launcher_pid) ST_launcher_pid="$val" ;;
        runner_pid)   ST_runner_pid="$val" ;;
        exp)          ST_exp="$val" ;;
        role)         ST_role="$val" ;;
        started)      ST_started="$val" ;;
        ended)        ST_ended="$val" ;;
        rc)           ST_rc="$val" ;;
        plan)         ST_plan="$val" ;;
        note)         ST_note="$val" ;;
      esac
    done < "$STATE/status"
  elif [ -f "$STATE/done" ]; then
    ST_source="legacy"; ST_state="done"
    ST_rc="$(cat "$STATE/done" 2>/dev/null)"
    ST_note="read from the pre-2026-08-15 .sweep/done flag"
  elif [ -f "$STATE/running" ]; then
    ST_source="legacy"; ST_state="running"
    ST_started="$(cat "$STATE/running" 2>/dev/null)"
    ST_note="pre-2026-08-15 .sweep/running flag: no pid was recorded, so liveness CANNOT be checked"
  elif [ -f "$STATE/log" ] || [ -f "$STATE/plan" ]; then
    # No status and no legacy flag, but a sweep plainly ran here. This is exactly the state this
    # tree was left in on 2026-08-11, and the state that used to read as "nothing launched".
    ST_source="orphan"
    ST_exp="$(awk '$1=="===" && $4=="/" {print $3; exit}' "$STATE/log" 2>/dev/null)"
    if grep -q 'SWEEP DONE' "$STATE/log" 2>/dev/null; then
      ST_state="done"
      ST_rc="$(sed -n 's/.*SWEEP DONE rc=\([0-9][0-9]*\).*/\1/p' "$STATE/log" 2>/dev/null | tail -1)"
      ST_note="reconstructed from .sweep/log: the sweep finished, but no status file was left"
    elif grep -q 'OPERATOR ABORT' "$STATE/log" 2>/dev/null; then
      ST_state="aborted"
      ST_note="reconstructed from .sweep/log: it carries an OPERATOR ABORT line"
    else
      ST_state="crashed"
      ST_note="reconstructed from .sweep/: a log/plan is present but no status — the sweep was killed from outside"
    fi
  fi

  # The recovery rule: `running` is a CLAIM, and the runner pid is what makes it falsifiable.
  if [ "$ST_state" = "running" ]; then
    sweep_pid_alive "$ST_runner_pid"; alive=$?
    if [ "$alive" -eq 1 ]; then
      if grep -q 'SWEEP DONE' "$STATE/log" 2>/dev/null; then
        ST_state="done"
        ST_rc="$(sed -n 's/.*SWEEP DONE rc=\([0-9][0-9]*\).*/\1/p' "$STATE/log" 2>/dev/null | tail -1)"
        ST_note="runner pid $ST_runner_pid is gone but the log says SWEEP DONE — its final status write was lost"
      else
        ST_state="crashed"
        ST_note="runner pid $ST_runner_pid is gone and the log has no SWEEP DONE line"
      fi
    elif [ "$alive" -eq 2 ] && [ "$ST_source" = "status" ]; then
      # Our own format with no runner pid: either sweep.sh is a few milliseconds into launching and
      # has not written the pid back yet, or the launcher died inside that window. Age decides.
      mt="$(sweep_mtime "$STATE/status")"
      case "$mt" in ''|*[!0-9]*) age=0 ;; *) age=$(( $(date +%s) - mt )) ;; esac
      if [ "$age" -gt 60 ]; then
        ST_state="crashed"
        ST_note="no runner pid was ever recorded and the status is ${age}s old — the launcher died before it could detach the runner"
      fi
    fi
    # A legacy `running` keeps state=running: with no pid there is nothing to falsify it with.
    # Only pre-2026-08-15 directories can reach that, and the note above says so.
  fi
}
# ── end shared block ──────────────────────────────────────────────────────────────────────────

# Write the status ATOMICALLY: temp name in the SAME directory, then mv. rename(2) is atomic, so a
# concurrent waitfor.sh sees either the whole old file or the whole new one, never half of either.
#   $1 = state, $2 = rc (may be empty), $3 = note (may be empty)
sweep_status_write() {
  local tmp="$STATE/status.tmp.$$" ended="" ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  case "$1" in done|aborted|crashed) ended="$ts" ;; esac
  {
    echo "state=$1"
    echo "launcher_pid=${LAUNCHER_PID:-}"
    echo "runner_pid=${RUNNER_PID:-}"
    echo "exp=${EXP:-}"
    echo "role=${ROLE:-}"
    echo "started=${STARTED:-}"
    echo "ended=$ended"
    echo "rc=${2:-}"
    echo "note=${3:-}"
    echo "plan=${PLAN1:-}"
  } > "$tmp" || return 1
  if ! mv -f "$tmp" "$STATE/status" 2>/dev/null; then
    # Windows can refuse a rename while a reader still holds the target open. Retry once, then fall
    # back to a non-atomic copy: a momentarily torn status file is recoverable, whereas losing the
    # state entirely is the exact bug this file exists to prevent.
    sleep 1
    mv -f "$tmp" "$STATE/status" 2>/dev/null || { cp -f "$tmp" "$STATE/status" 2>/dev/null; rm -f "$tmp"; }
  fi
}

sweep_status_read
case "$ST_state" in
  running)
    echo "REFUSED: a sweep is already in flight — exp=${ST_exp:-?} role=${ST_role:-?}, started ${ST_started:-?},"
    echo "  runner pid ${ST_runner_pid:-unknown}. Poll it:  bash autoresearch/waitfor.sh"
    if [ "$ST_source" = "legacy" ]; then
      echo "  NOTE: $ST_note"
      echo "  If you are certain nothing is running:  rm autoresearch/.sweep/running"
      echo "  (only a pre-2026-08-15 .sweep can reach this branch; the current format self-recovers)"
    fi
    exit 1 ;;
  crashed|aborted)
    # Reclaimed AUTOMATICALLY and loudly — never a human `rm`. See the D1 note above.
    STAMP="$(date '+%Y%m%dT%H%M%S')"
    echo "RECLAIMING a previous sweep in state '$ST_state' — exp=${ST_exp:-?} role=${ST_role:-?} started=${ST_started:-?}"
    [ -n "$ST_note" ] && echo "  why: $ST_note"
    for f in log plan; do
      [ -f "$STATE/$f" ] || continue
      mv -f "$STATE/$f" "$STATE/$f.$ST_state.$STAMP" 2>/dev/null &&
        echo "  kept for forensics: autoresearch/.sweep/$f.$ST_state.$STAMP"
    done
    echo "  ANY results/*.json those runs already wrote are REAL. Reuse them; re-run only what is missing."
    rm -f "$STATE/status" "$STATE/running" "$STATE/done"
    ;;
  done|none)
    # Retire the legacy flags if this .sweep predates 2026-08-15. They are never written again.
    rm -f "$STATE/running" "$STATE/done" ;;
esac

PY="${MFAS_PY:-/c/ProgramData/anaconda3/envs/allen/python.exe}"
LOG="$STATE/log"
PLAN="$STATE/plan"

# Build the plan: one "<dataset> <seed> [<seed> ...]" line per dataset. Both branches produce the
# same format, so runner() never has to know which one ran.
if [ "$AUTO" -eq 1 ]; then
  if ! "$PY" "$AR/seed_plan.py" --variant "$EXP" --role "$ROLE" --datasets "$DATASETS" > "$PLAN"; then
    echo "REFUSED: seed_plan.py failed for $EXP; re-run without --auto-seeds to force 3 seeds" >&2
    rm -f "$PLAN"; exit 2
  fi
elif [ -n "$SEEDS" ]; then
  : > "$PLAN"
  for DS in ${DATASETS//,/ }; do echo "$DS $SEEDS" >> "$PLAN"; done
else
  # No --seeds and no --auto-seeds: take the seed list for this ROLE from campaign.yaml, per
  # dataset. Fixed 2026-08-10 (queue item P06). The old default was a hardcoded "42 123 999" for
  # every role, so `--role confirm` silently ran a 3-seed confirm where campaign.yaml asks for 5
  # on the primaries and 20 on mouse — a quietly weakened promotion gate. Cycle #5 only reached
  # 5/5/20 because it noticed and ran the remainder by hand. Refuse rather than guess.
  if ! "$PY" - "$ROLE" "$DATASETS" > "$PLAN" <<'PYEOF'
import sys, yaml
role, datasets = sys.argv[1], sys.argv[2].split(",")
# sweep.sh cd's to the campaign root before running this.
cfg = yaml.safe_load(open("autoresearch/campaign.yaml"))
key = "confirm_seeds" if role == "confirm" else "screen_seeds"
for ds in [d.strip() for d in datasets if d.strip()]:
    entry = (cfg.get("datasets") or {}).get(ds) or {}
    seeds = entry.get(key)
    if not seeds:
        sys.stderr.write(f"campaign.yaml datasets.{ds}.{key} is missing\n")
        raise SystemExit(1)
    print(ds, " ".join(str(s) for s in seeds))
PYEOF
  then
    echo "REFUSED: could not read the $ROLE seed list from campaign.yaml. Pass --seeds explicitly" >&2
    echo "  if you really mean to override the protocol." >&2
    rm -f "$PLAN"; exit 2
  fi
fi

: > "$LOG"
STARTED="$(date '+%Y-%m-%d %H:%M:%S')"
LAUNCHER_PID=$$
RUNNER_PID=""                                   # filled in below, once the runner exists
PLAN1="$(tr '\n' ';' < "$PLAN" | sed 's/;*$//')"   # the plan, flattened onto the status file's one line
sweep_status_write running "" "launching"

runner() {
  local rc_all=0
  RUNNER_PID="$$"    # this bash IS the detached unit; the launcher records the same pid below
  # One line per dataset; datasets run in the given order so cheap ones fail fast on a broken
  # variant. Seeds differ per dataset under --auto-seeds (P02), hence reading the plan file.
  while read -r DS SEEDLIST; do
    [ -z "$DS" ] && continue
    for S in $SEEDLIST; do
      echo "=== $(date '+%H:%M:%S')  $EXP / $DS / seed $S ===" >> "$LOG"
      PYTHONPATH=src "$PY" -m eval.run_variant --exp "$EXP" --dataset "$DS" --seed "$S" \
          --out results/ --role "$ROLE" --device auto >> "$LOG" 2>&1
      local rc=$?
      echo "    rc=$rc" >> "$LOG"
      [ $rc -ne 0 ] && rc_all=$rc
    done
  done < "$PLAN"
  echo "SWEEP DONE rc=$rc_all $(date '+%H:%M:%S')" >> "$LOG"
  sweep_status_write done "$rc_all" ""
}

# Launch. The detached unit MUST be an MSYS `bash -c` (see the header: a bare `nohup python.exe &`
# does not outlive this session on Windows, and `setsid` does not exist here at all). `runner` and
# `sweep_status_write` are shipped into it with `declare -f`; every VALUE it needs travels in the
# ENVIRONMENT and is never interpolated into the -c string — that interpolation was defect D4,
# where an --exp containing a single quote turned into shell code.
export EXP ROLE PY LOG PLAN STATE STARTED LAUNCHER_PID PLAN1
nohup bash -c "$(declare -f sweep_status_write); $(declare -f runner); runner" </dev/null >/dev/null 2>&1 &
RUNNER_PID=$!
disown 2>/dev/null

# Record the runner pid: this is what makes `state=running` falsifiable later. Guarded on the state
# still being `running`, so a plan that empties out in milliseconds cannot have its `done` clobbered
# by this write. (Residual, accepted: a sub-millisecond window between the grep and the mv. Losing
# it would misreport a finished sweep as crashed — loudly, and with its results intact on disk.)
if grep -q '^state=running' "$STATE/status" 2>/dev/null; then
  sweep_status_write running "" ""
fi

echo "LAUNCHED  exp=$EXP role=$ROLE  runner pid $RUNNER_PID  (plan: autoresearch/.sweep/plan)"
sed 's/^/  /' "$PLAN"
echo "  log:    autoresearch/.sweep/log"
echo "  status: autoresearch/.sweep/status"
echo "  poll: bash autoresearch/waitfor.sh   (repeat until DONE — do NOT end your turn instead)"
