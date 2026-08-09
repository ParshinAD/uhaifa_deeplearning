#!/usr/bin/env bash
# Unattended campaign driver — runs research cycles back to back, one at a time.
#
#   start:   nohup bash autoresearch/driver.sh > /dev/null 2>&1 &
#   watch:   tail -f autoresearch/logs/driver.log
#   stop:    touch autoresearch/STOP        (finishes the current cycle, then exits)
#   kill:    bash autoresearch/driver.sh --abort
#
# Design notes
# - ONE cycle at a time (lock dir). The machine has a single Apple MPS device; concurrent
#   training runs would contend and would corrupt wall-clock measurements.
# - The lock is the only concurrency control that matters: cycles run heavy training.
# - Daily wall-clock budget from campaign.yaml, tracked in autoresearch/.budget.
# - caffeinate keeps the Mac awake for overnight campaigns (idle sleep only, not lid-close).
# - Every cycle is a fresh `claude -p` invocation: state lives in files, not in a context window.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Every cycle must start in the campaign root: that is where .claude/commands, CLAUDE.md,
# results/ and the conda-relative paths are resolved from.
cd "$ROOT" || exit 1
AR="$ROOT/autoresearch"
LOCK="$AR/.lock"
STOP="$AR/STOP"
BUDGET_FILE="$AR/.budget"
LOGDIR="$AR/logs"
DRIVER_LOG="$LOGDIR/driver.log"

mkdir -p "$LOGDIR"

# ── config ────────────────────────────────────────────────────────────────────
PY="${MFAS_PY:-/opt/homebrew/Caskroom/miniforge/base/envs/allen/bin/python}"
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude-personal}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CYCLE_PROMPT="${CYCLE_PROMPT:-/research-cycle}"
CYCLE_TIMEOUT_S="${CYCLE_TIMEOUT_S:-21600}"      # 6 h, matches campaign.yaml max_cycle_wall_clock_h
PAUSE_S="${PAUSE_S:-60}"                          # breather between cycles
MAX_DAILY_H="${MAX_DAILY_H:-12}"                  # campaign.yaml budget.max_wall_clock_h_per_day
MIN_FREE_GB="${MIN_FREE_GB:-50}"
MODEL_ARG=""
[ -n "${CAMPAIGN_MODEL:-}" ] && MODEL_ARG="--model ${CAMPAIGN_MODEL}"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$DRIVER_LOG"; }

# ── --abort: stop now, release the lock ───────────────────────────────────────
if [ "${1:-}" = "--abort" ]; then
  touch "$STOP"
  # Kill the running cycle first, then the driver itself: killing the driver alone would
  # orphan a claude process that is mid-training-run.
  for f in cycle_pid pid; do
    [ -f "$LOCK/$f" ] || continue
    pid="$(cat "$LOCK/$f")"
    log "ABORT: terminating $f=$pid"
    kill -TERM "$pid" 2>/dev/null || true
    sleep 2
    kill -KILL "$pid" 2>/dev/null || true
  done
  rm -rf "$LOCK"
  log "ABORT: lock released, STOP file set"
  exit 0
fi

# ── single-driver guard ───────────────────────────────────────────────────────
if ! mkdir "$LOCK" 2>/dev/null; then
  log "REFUSING TO START: lock $LOCK exists (another driver or cycle is running)."
  log "  If you are sure nothing is running:  rm -rf '$LOCK'"
  exit 1
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"; log "driver exited, lock released"' EXIT

rm -f "$STOP"
log "=============================================================="
log "campaign driver starting"
log "  root         : $ROOT"
log "  python       : $PY"
log "  claude config: $CLAUDE_CONFIG_DIR"
log "  cycle prompt : $CYCLE_PROMPT"
log "  timeout      : ${CYCLE_TIMEOUT_S}s   daily budget: ${MAX_DAILY_H}h"
log "=============================================================="

# ── daily budget helpers ──────────────────────────────────────────────────────
budget_today() {
  local today; today="$(date '+%F')"
  if [ -f "$BUDGET_FILE" ]; then
    local d h; read -r d h < "$BUDGET_FILE"
    if [ "$d" = "$today" ]; then echo "$h"; return; fi
  fi
  echo "0"
}
budget_add() {
  local today secs prev new
  today="$(date '+%F')"; secs="$1"; prev="$(budget_today)"
  new="$(awk -v p="$prev" -v s="$secs" 'BEGIN{printf "%.3f", p + s/3600.0}')"
  printf '%s %s\n' "$today" "$new" > "$BUDGET_FILE"
  echo "$new"
}

# ── run a command with a timeout, portably (macOS has no coreutils timeout) ───
run_with_timeout() {
  local secs="$1"; shift
  "$@" &
  local cmd_pid=$!
  echo "$cmd_pid" > "$LOCK/cycle_pid"
  ( sleep "$secs"; kill -TERM "$cmd_pid" 2>/dev/null; sleep 10; kill -KILL "$cmd_pid" 2>/dev/null ) &
  local watchdog=$!
  wait "$cmd_pid"; local rc=$?
  kill -TERM "$watchdog" 2>/dev/null
  rm -f "$LOCK/cycle_pid"
  return $rc
}

# Keep the machine awake for overnight campaigns. macOS: caffeinate. Linux/WSL: systemd-inhibit
# when available. Neither is required — the loop just runs unprotected without them.
CAFF=""
if command -v caffeinate >/dev/null 2>&1; then
  CAFF="caffeinate -i"
elif command -v systemd-inhibit >/dev/null 2>&1; then
  CAFF="systemd-inhibit --what=idle --why=mfas-campaign"
fi

# ── preflight: headless auth ──────────────────────────────────────────────────
# Headless runs authenticate from .claude/settings.local.json (CLAUDE_CODE_OAUTH_TOKEN), which is
# gitignored and therefore absent from a fresh worktree. Fail loudly now rather than burning the
# daily budget on a crash loop of "Not logged in".
if [ ! -f "$ROOT/.claude/settings.local.json" ]; then
  log "HALT: $ROOT/.claude/settings.local.json is missing — headless claude will not be logged in."
  log "  Copy it from the original checkout (it is gitignored and holds CLAUDE_CODE_OAUTH_TOKEN)."
  exit 1
fi
if ! "$CLAUDE_BIN" -p "reply with OK" --dangerously-skip-permissions < /dev/null 2>&1 \
     | grep -qi "ok"; then
  log "HALT: headless auth check failed (expected a reply). Refresh CLAUDE_CODE_OAUTH_TOKEN in"
  log "  $ROOT/.claude/settings.local.json, then restart the driver."
  exit 1
fi
log "preflight: headless auth OK"

# ── main loop ─────────────────────────────────────────────────────────────────
cycle=0
while true; do
  if [ -f "$STOP" ]; then
    log "STOP file present -> exiting cleanly"; break
  fi

  # POSIX `df -k` works on both macOS (BSD) and Linux/WSL; `df -g` is BSD-only.
  free_gb=$(df -k "$ROOT" | awk 'NR==2 {print int($4/1048576)}')
  if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
    log "HALT: only ${free_gb}GB free (< ${MIN_FREE_GB}GB). Stopping to protect the machine."
    break
  fi

  spent="$(budget_today)"
  if awk -v s="$spent" -v m="$MAX_DAILY_H" 'BEGIN{exit !(s >= m)}'; then
    log "daily budget reached (${spent}h / ${MAX_DAILY_H}h) — sleeping until tomorrow"
    sleep 1800
    continue
  fi

  cycle=$((cycle + 1))
  stamp="$(date '+%Y%m%dT%H%M%S')"
  cycle_log="$LOGDIR/cycle-${stamp}.log"
  log "--- cycle #$cycle starting (today: ${spent}h/${MAX_DAILY_H}h) -> $cycle_log"

  start_s=$(date +%s)
  # shellcheck disable=SC2086
  # stdin from /dev/null: headless claude waits ~3s for piped input otherwise, and a detached
  # nohup'd driver has no terminal to read from.
  run_with_timeout "$CYCLE_TIMEOUT_S" \
    $CAFF "$CLAUDE_BIN" -p "$CYCLE_PROMPT" \
      --dangerously-skip-permissions \
      $MODEL_ARG \
      < /dev/null > "$cycle_log" 2>&1
  rc=$?
  end_s=$(date +%s)
  dur=$((end_s - start_s))
  total="$(budget_add "$dur")"

  if [ $rc -eq 0 ]; then
    log "--- cycle #$cycle finished OK in ${dur}s (today: ${total}h)"
  else
    log "--- cycle #$cycle EXITED rc=$rc after ${dur}s (today: ${total}h) — see $cycle_log"
    tail -20 "$cycle_log" | sed 's/^/      | /' | tee -a "$DRIVER_LOG" >/dev/null
  fi

  # A cycle that dies in seconds is a broken setup, not a research result: back off hard
  # rather than burning the daily budget on a crash loop.
  if [ "$dur" -lt 60 ]; then
    log "cycle was suspiciously short (${dur}s) — backing off 10 min"
    sleep 600
  else
    sleep "$PAUSE_S"
  fi
done

log "driver loop ended after $cycle cycle(s)"
