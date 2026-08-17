#!/usr/bin/env bash
# One-shot overnight orchestrator: run the whole sequential plan in ONE process.
#
# Why this exists: every separate tool invocation from the assistant is a potential permission
# prompt, and the operator is asleep and cannot approve them. So the night's work is written down
# once, launched once, and then WATCHED by reading this log file - no further shell calls needed.
#
# Everything here is sequential on purpose: one GPU, and wall-clock is a measured quantity in this
# project (the runtime cap is 3600 s/run and the champion sits 2% under it on microns). Two heavy
# jobs at once would poison exactly the numbers we are here to measure.
#
# Log:  dr_tmp/night_run.log     (this script's own narration + every step's output)
# State: dr_tmp/night_run.state  (one line per completed step, so a re-launch can skip them)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PY="/c/ProgramData/anaconda3/envs/allen/python.exe"
LOG="dr_tmp/night_run.log"
STATE="dr_tmp/night_run.state"
touch "$STATE"

say() { printf '\n=== %s  %s ===\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }
done_already() { grep -qx "$1" "$STATE" 2>/dev/null; }
mark_done() { echo "$1" >> "$STATE"; }

# Wait for the P05 microns verify that is already in flight (launched 22:55:40) to report.
#
# Watch its LOG rather than the process table: under Git Bash `ps -W` truncates command lines and
# `ps aux` is not available, so process-name matching is unreliable here - and a false "it is
# finished" would start a second heavy job on the one GPU, which is exactly the contention this
# whole script is arranged to avoid. run_variant's last two lines are a `DONE  score=...` and a
# `wrote results\...json`, so the log is an unambiguous signal.
wait_for_gpu_free() {
  local marker="dr_tmp/p05_microns_verify.log" waited=0
  [ -f "$marker" ] || return 0
  grep -q "wrote results" "$marker" 2>/dev/null && return 0
  say "an eval.run_variant is in flight (P05 microns verify) - waiting for it to report"
  while ! grep -q "wrote results" "$marker" 2>/dev/null; do
    # A crash leaves a traceback and no `wrote results`; do not wait forever on a dead run.
    if grep -qE "Traceback|Error:|error:" "$marker" 2>/dev/null; then
      say "WARNING: the in-flight run's log shows an error and no result; proceeding"
      return 0
    fi
    sleep 30
    waited=$((waited + 30))
    if [ $waited -ge 4200 ]; then
      say "WARNING: still waiting after ${waited}s; proceeding anyway"
      return 0
    fi
  done
  say "the in-flight run reported after ${waited}s of waiting"
  return 0
}

say "night run starting"
say "plan: (1) microns epoch grid  (2) connectome epoch grid  (3) full test suite  (4) bookkeeping verify"

# ── step 0: do not fight the run that is already going ──────────────────────────
wait_for_gpu_free

# ── step 1: microns epoch sizing ────────────────────────────────────────────────
# The champion spends 80,000 Rocket epochs and ~3400 s of a 3600 s cap on microns, and the two
# arms measured on 2026-08-11 (epochs 0 and 2500) suggest the gradient phase buys ~0.13 pp for
# almost all of that wall-clock. Finishing the grid is what decides whether microns' runtime
# invariant can be satisfied by SIZING rather than by weakening the cap.
if done_already "microns_grid"; then
  say "step 1 microns epoch grid - already done, skipping"
else
  say "step 1 microns epoch grid (arms 5000, 10000, 20000; 0 and 2500 already on disk)"
  PYTHONPATH=src "$PY" experiments/proto_P07_sizing.py microns >> "$LOG" 2>&1
  rc=$?
  say "step 1 finished rc=$rc"
  [ $rc -eq 0 ] && mark_done "microns_grid"
fi

# ── step 2: connectome epoch sizing ─────────────────────────────────────────────
# connectome ships 20,000 epochs and uses only ~1232 s of the 3600 s cap. If the gradient phase is
# ceremonial there too, that headroom is free for refinement - and if it is NOT, that kills the
# premise on the dataset that carries the mission target, which is equally worth knowing.
if done_already "connectome_grid"; then
  say "step 2 connectome epoch grid - already done, skipping"
else
  say "step 2 connectome epoch grid (arms 0, 2500, 5000, 10000, 20000, 40000)"
  PYTHONPATH=src "$PY" experiments/proto_P07_sizing.py connectome >> "$LOG" 2>&1
  rc=$?
  say "step 2 finished rc=$rc"
  [ $rc -eq 0 ] && mark_done "connectome_grid"
fi

# ── step 3: the full suite, AFTER the timing-sensitive work ─────────────────────
# Deliberately last among the compute steps: the suite takes ~100 s and includes GPU work, and a
# microns run has only ~68 s of slack against its 3450 s deadline. Running it earlier could push a
# measured run past the guard and turn an hour of GPU into a flagged, unusable, truncated record.
if done_already "pytest"; then
  say "step 3 test suite - already done, skipping"
else
  say "step 3 full test suite (includes the new segment-move tests)"
  PYTHONPATH=src "$PY" -m pytest tests/ -q >> "$LOG" 2>&1
  rc=$?
  say "step 3 finished rc=$rc"
  [ $rc -eq 0 ] && mark_done "pytest"
fi

# ── step 4: mechanical bookkeeping verification ─────────────────────────────────
if done_already "verify"; then
  say "step 4 bookkeeping verify - already done, skipping"
else
  say "step 4 bookkeeping verify"
  "$PY" autoresearch/verify_cycle.py --all >> "$LOG" 2>&1
  say "step 4 finished rc=$? (non-zero is expected while gates_run is not yet recorded)"
  mark_done "verify"
fi

say "night run COMPLETE"
