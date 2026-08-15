#!/usr/bin/env bash
# Block on the detached sweep for a bounded interval, then report and exit.
#
# Designed to be called REPEATEDLY from a cycle: each call blocks for up to --for seconds (default
# 540, comfortably inside an agent Bash timeout) and then returns, so the agent stays in its turn
# instead of ending it while a run is in flight.
#
# EXIT CODES — a resuming cycle branches on these (see .claude/commands/research-cycle.md):
#    0  DONE     the sweep ran to completion. rc= is the worst rc of its runs; 0 is not implied.
#   10  RUNNING  still in flight -> CALL AGAIN. Do not end your turn.
#    2  NONE     no sweep was ever launched in this .sweep/
#   20  ABORTED  a sweep was deliberately killed from outside (an OPERATOR ABORT line in the log)
#   21  CRASHED  a sweep was launched, but its runner is gone and it never recorded an outcome
#
# 20 and 21 are TERMINAL: calling again cannot change them. In both cases the runs listed below as
# completed WROTE REAL results/*.json — reuse them; only the runs listed as not done still need
# running. Added 2026-08-15 (defect D2): before that, both states came back as exit 2 "nothing
# launched", indistinguishable from "never started", so a resuming cycle could silently redo — or
# skip — hours of GPU work. The 2026-08-11 operator abort left the tree in exactly that state.
#
# CALLERS: `while ! bash autoresearch/waitfor.sh; do :; done` only terminates on exit 0, so it
# spins on 2/20/21. Use the `case` form in research-cycle.md. Until every caller is converted, this
# script sleeps a few seconds before returning a terminal non-zero code, so an old-style loop
# degrades to a slow, loud drip instead of pegging a core for the rest of the cycle timeout.
#
# Usage:  bash autoresearch/waitfor.sh [--for 540]
set -uo pipefail
AR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$AR/.sweep"
BUDGET=540
[ "${1:-}" = "--for" ] && BUDGET="${2:-540}"
# Backoff before returning a TERMINAL non-zero code (see the caller note above). Overridable so a
# test — or an operator reading the report by hand — does not sit through it.
BACKOFF="${WAITFOR_TERMINAL_BACKOFF_S:-5}"

# ── .sweep status (SHARED BLOCK — keep in sync with the copy in sweep.sh) ─────────────────────
# Rewritten 2026-08-15 (defect D1). The old layout was two flag files, `.sweep/running` and
# `.sweep/done`, and it had two holes:
#   * a sweep killed from OUTSIDE left NEITHER file. That happened on 2026-08-11, when an operator
#     killed an H42/microns verify run mid-flight; this script then printed "nothing launched", so a
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
# is an MSYS bash — see sweep.sh's header), so plain `kill -0` is correct here and portable to macOS
# and Linux; no `ps -W`/`taskkill` translation is needed or attempted. `ps -p` is a fallback for
# hosts where signalling is restricted. Returns 0 alive, 1 dead, 2 unknown (no usable pid recorded).
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

# One "<dataset> <seed>|<rc>|<pct>" line per run the log shows STARTED, in log order. The format is
# fixed by sweep.sh's runner(): a "=== HH:MM:SS  <exp> / <ds> / seed <s> ===" header, then the run's
# output, then an indented "rc=<n>". rc="-" means the run started and never reported — i.e. it was
# interrupted mid-run. The OPERATOR ABORT line also begins with "===" but has no " / ", so the
# header pattern below (which requires the two slash fields) does not match it.
sweep_run_table() {
  awk '
    $1=="===" && $4=="/" && $6=="/" && $7=="seed" {
      if (cur != "") printf "%s|%s|%s\n", cur, rc, pct
      cur = $5 " " $8; rc="-"; pct="-"; next
    }
    /DONE  score=/ { for (i=1; i<=NF; i++) if ($i ~ /^pct=/) pct=substr($i, 5) }
    /^[ \t]*rc=[0-9]+[ \t]*$/ { line=$0; sub(/^[ \t]*rc=/, "", line); sub(/[ \t]*$/, "", line); rc=line }
    END { if (cur != "") printf "%s|%s|%s\n", cur, rc, pct }
  ' "$STATE/log" 2>/dev/null
}

# The report a resuming cycle actually needs: what is REAL on disk, and what is still owed.
# Before 2026-08-15 this was a bare `tail -20` of the log, which said neither.
report_runs() {
  local table entry ds seeds s key rc pct
  local n_done=0 n_todo=0 done_lines="" todo_lines=""
  table="$(sweep_run_table)"
  if [ ! -f "$STATE/plan" ]; then
    if [ -n "$table" ]; then
      echo "  (no autoresearch/.sweep/plan on disk — cannot say what was PLANNED. Runs the log shows:)"
      printf '%s\n' "$table" | sed 's/^/    /'
    else
      echo "  (no autoresearch/.sweep/plan and no runs in the log — nothing can be said about coverage)"
    fi
    return 0
  fi
  while read -r ds seeds; do
    [ -z "$ds" ] && continue
    for s in $seeds; do
      key="$ds $s"
      entry="$(printf '%s\n' "$table" | grep -F -m1 "$key|")"
      if [ -z "$entry" ]; then
        todo_lines="$todo_lines    $key   (never started)
"
        n_todo=$((n_todo + 1))
      else
        rc="$(printf '%s' "$entry" | cut -d'|' -f2)"
        pct="$(printf '%s' "$entry" | cut -d'|' -f3)"
        if [ "$rc" = "-" ]; then
          todo_lines="$todo_lines    $key   (started, interrupted before it reported — no result file)
"
          n_todo=$((n_todo + 1))
        else
          done_lines="$done_lines    $key   rc=$rc   pct=$pct
"
          n_done=$((n_done + 1))
        fi
      fi
    done
  done < "$STATE/plan"

  if [ "$n_done" -gt 0 ]; then
    echo "  COMPLETED ($n_done): these runs wrote results/*.json. Those numbers are REAL —"
    echo "  a resuming cycle must REUSE them and must NOT re-run them:"
    printf '%s' "$done_lines"
  else
    echo "  COMPLETED (0): no run of this sweep finished."
  fi
  if [ "$n_todo" -gt 0 ]; then
    echo "  NOT DONE ($n_todo): planned, but no result on disk. Only these still need running:"
    printf '%s' "$todo_lines"
  fi
}

started=$SECONDS
while : ; do
  sweep_status_read
  case "$ST_state" in
    none)
      echo "NO SWEEP: nothing launched. Start one with autoresearch/sweep.sh --exp <id> --role <role>."
      sleep "$BACKOFF"; exit 2 ;;
    done)
      echo "DONE rc=${ST_rc:-?}  exp=${ST_exp:-?} role=${ST_role:-?}  started ${ST_started:-?}  ended ${ST_ended:-?}"
      [ -n "$ST_note" ] && echo "  note: $ST_note"
      report_runs
      grep -E "DONE  score|^SWEEP DONE" "$STATE/log" 2>/dev/null | tail -20 | sed 's/^/  /'
      exit 0 ;;
    aborted|crashed)
      # tr, not ${x^^}: that expansion needs bash 4 and this file still has to parse under the
      # bash 3.2 that ships as /bin/bash on macOS.
      echo "SWEEP $(printf '%s' "$ST_state" | tr 'a-z' 'A-Z') — TERMINAL, calling this again cannot change it."
      echo "  exp=${ST_exp:-?} role=${ST_role:-?}  started ${ST_started:-?}"
      [ -n "$ST_note" ] && echo "  why: $ST_note"
      report_runs
      echo "  The next autoresearch/sweep.sh reclaims this state automatically — no rm, no human."
      echo "  Do NOT end your turn here: re-launch ONLY the runs listed as NOT DONE, then poll again."
      sleep "$BACKOFF"
      [ "$ST_state" = "aborted" ] && exit 20
      exit 21 ;;
    running)
      : ;;
    *)
      # An unknown state means the status file was hand-edited or truncated. Say so rather than
      # silently treating it as "running" and blocking a cycle for the whole budget.
      echo "UNKNOWN SWEEP STATE '$ST_state' in autoresearch/.sweep/status — treating it as crashed."
      report_runs
      sleep "$BACKOFF"; exit 21 ;;
  esac
  [ $((SECONDS - started)) -ge "$BUDGET" ] && break
  sleep 15
done

elapsed_total=""
[ -n "$ST_started" ] && elapsed_total="(sweep started $ST_started)"
echo "STILL RUNNING after ${BUDGET}s of this call $elapsed_total"
echo "  exp=${ST_exp:-?} role=${ST_role:-?}  runner pid ${ST_runner_pid:-unknown}"
# `grep -c` prints 0 AND exits 1 when there is no match, so the old `|| echo 0` printed "0 0".
n_so_far="$(grep -c 'DONE  score' "$STATE/log" 2>/dev/null | head -1)"
echo "  completed so far: ${n_so_far:-0} run(s)"
tail -1 "$STATE/log" 2>/dev/null | sed 's/^/  last: /'
echo "  -> call this again. Do NOT end your turn: a cycle that stops here loses the whole sweep."
exit 10
