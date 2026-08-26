#!/usr/bin/env bash
# Unattended campaign driver — runs research cycles back to back, one at a time.
#
#   start:   nohup bash autoresearch/driver.sh > /dev/null 2>&1 &
#   watch:   tail -f autoresearch/logs/driver.log
#   stop:    touch autoresearch/STOP        (finishes the current cycle, then exits)
#   kill:    bash autoresearch/driver.sh --abort
#
# Design notes
# - ONE cycle at a time (lock dir). The machine has a single GPU; concurrent training runs
#   would contend and would corrupt wall-clock measurements.
# - The lock is the only concurrency control that matters: cycles run heavy training.
# - Every campaign-level knob comes from campaign.yaml (which declares itself the only place
#   they live); env vars override; a missing key falls back to a hardcoded default WITH a warning.
# - Wall-clock budget is a ROLLING 24 h WINDOW, not a calendar day, tracked as an append-only
#   interval ledger in autoresearch/.budget. See "budget" below for why the calendar-day version
#   was wrong.
# - caffeinate / systemd-inhibit / keepawake.ps1 keep the machine awake for overnight campaigns.
# - Every cycle is a fresh `claude -p` invocation: state lives in files, not in a context window.
# - A cycle counts as done only if it made PROGRESS (a commit or a state.json history entry).
#   The exit code alone is not evidence — see "no-op detection" below.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Every cycle must start in the campaign root: that is where .claude/commands, CLAUDE.md,
# results/ and the conda-relative paths are resolved from.
cd "$ROOT" || exit 1
AR="$ROOT/autoresearch"
LOCK="$AR/.lock"
STOP="$AR/STOP"
BUDGET_FILE="$AR/.budget"
STATE_FILE="$AR/state.json"
CAMPAIGN_YAML="$AR/campaign.yaml"
LOGDIR="$AR/logs"
DRIVER_LOG="$LOGDIR/driver.log"

mkdir -p "$LOGDIR"

# ── config ────────────────────────────────────────────────────────────────────
PY="${MFAS_PY:-/c/ProgramData/anaconda3/envs/allen/python.exe}"
# CLAUDE_CONFIG_DIR is left at the CLI default here: on this box `claude /login` wrote the
# credentials to ~/.claude, and forcing a non-existent config dir makes every headless cycle
# start out "Not logged in". Export it before starting the driver if you keep a separate profile.
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CYCLE_PROMPT="${CYCLE_PROMPT:-/research-cycle}"
PAUSE_S="${PAUSE_S:-60}"                          # breather between cycles
BUDGET_WINDOW_S=86400                             # rolling window: 24 h, not "today"
AUTH_SENTINEL="CAMPAIGN_AUTH_OK"
AUTH_TIMEOUT_S="${AUTH_TIMEOUT_S:-300}"
MODEL_ARG=""
[ -n "${CAMPAIGN_MODEL:-}" ] && MODEL_ARG="--model ${CAMPAIGN_MODEL}"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$DRIVER_LOG"; }

# Same as log(), but stdout goes to STDERR so it is safe to call from inside a $(...) that is
# capturing a value. cfg() below needs exactly this.
logv() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$DRIVER_LOG" >&2; }

hms() { awk -v s="${1:-0}" 'BEGIN{printf "%dh%02dm", int(s/3600), int((s%3600)/60)}'; }

# ── campaign.yaml knobs ───────────────────────────────────────────────────────
# campaign.yaml says it is "the ONLY place campaign-level knobs live", but until 2026-08-15 the
# driver read none of it: the cycle timeout, the daily cap and the free-disk floor were hardcoded
# copies free to drift from the file, and budget.max_cycles / budget.results_dir_max_gb were
# implemented NOWHERE — declared limits that nothing enforced. One python call reads the block;
# yaml.safe_load is the same mechanism sweep.sh already uses.
CFG_KV=""
if [ -f "$CAMPAIGN_YAML" ]; then
  CFG_KV="$("$PY" - "$CAMPAIGN_YAML" <<'PYEOF' 2>/dev/null
import sys, yaml
try:
    b = (yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}).get("budget") or {}
except Exception:
    b = {}
for k in ("max_wall_clock_h_per_day", "max_cycle_wall_clock_h", "max_cycles",
          "results_dir_max_gb", "min_free_disk_gb"):
    v = b.get(k)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        print("%s=%s" % (k, v))
PYEOF
)"
fi

# cfg <campaign.yaml budget key> <hardcoded fallback> <env override or "">
# Precedence: environment > campaign.yaml > hardcoded default (the last one warns, so a typo in
# the yaml degrades to "the old behaviour, loudly" instead of "no limit at all").
cfg() {
  local key="$1" fallback="$2" override="${3:-}" v
  if [ -n "$override" ]; then printf '%s' "$override"; return 0; fi
  v="$(printf '%s\n' "$CFG_KV" | awk -F= -v k="$key" '$1==k {print $2; exit}')"
  if [ -z "$v" ]; then
    logv "WARNING: campaign.yaml budget.$key missing or unreadable -> using default $fallback"
    v="$fallback"
  fi
  printf '%s' "$v"
}

MAX_DAILY_H="$(cfg max_wall_clock_h_per_day 12 "${MAX_DAILY_H:-}")"
MIN_FREE_GB="$(cfg min_free_disk_gb 50 "${MIN_FREE_GB:-}")"
MAX_CYCLES="$(cfg max_cycles 200 "${MAX_CYCLES:-}")"
RESULTS_MAX_GB="$(cfg results_dir_max_gb 20 "${RESULTS_MAX_GB:-}")"
CYCLE_H="$(cfg max_cycle_wall_clock_h 10 "")"
CYCLE_TIMEOUT_S="${CYCLE_TIMEOUT_S:-$(awk -v h="$CYCLE_H" 'BEGIN{printf "%d", h*3600}')}"

# Normalise to integers: the yaml may legitimately carry 12.0 or 20.5, and `[ x -lt y ]` would
# choke on a decimal point.
MAX_DAILY_S="$(awk -v h="$MAX_DAILY_H" 'BEGIN{printf "%d", h*3600}')"
MIN_FREE_GB="$(awk -v v="$MIN_FREE_GB" 'BEGIN{printf "%d", v}')"
MAX_CYCLES="$(awk -v v="$MAX_CYCLES"  'BEGIN{printf "%d", v}')"
CYCLE_TIMEOUT_S="$(awk -v v="$CYCLE_TIMEOUT_S" 'BEGIN{printf "%d", v}')"

# ── process control ───────────────────────────────────────────────────────────
# Kill a background job and everything under it.
#
# On macOS/Linux `kill` is enough. Under Git Bash it is NOT: a native Windows child (claude is a
# node process; so is powershell) survives a signal sent to its MSYS job — verified on this box,
# where a backgrounded powershell outlived `kill -TERM` and kept running. The watchdog would then
# believe it had killed a wedged cycle, `wait` would return, and the next cycle would start
# alongside the orphan — two concurrent cycles on one GPU, which is exactly what the lock exists
# to prevent. So translate the MSYS pid to a Windows pid via `ps -W` (column 4 is the WINPID) and
# take down the tree.
#
# ORDER MATTERS, and it used to be wrong (fired once, 2026-08). The old body sent `kill -TERM`
# BEFORE resolving the winpid. When the TERM did land, the process was already off `ps -W` by the
# time we looked, `winpid` came back empty, the `[ -n "$winpid" ]` guard was false and the
# taskkill — the only thing that reaches native Windows grandchildren — never ran. Resolve the
# winpid FIRST, while the process is still alive.
#
# Returns 0 if the pid is gone afterwards, 1 if it survived. The caller must be able to tell a
# successful kill from a failed one: --abort decides whether to release the lock on this.
kill_tree() {
  local pid="${1:-}" winpid=""
  [ -z "$pid" ] && return 0
  kill -0 "$pid" 2>/dev/null || return 0    # already gone — not a failure

  if command -v taskkill >/dev/null 2>&1; then
    winpid="$(ps -W 2>/dev/null | awk -v p="$pid" '$1==p {print $4; exit}')"
  fi

  kill -TERM "$pid" 2>/dev/null

  # Unconditional when a winpid was resolved: the tree must come down whether or not the TERM was
  # enough, and taskkill against an already-dead pid is a harmless no-op.
  [ -n "$winpid" ] && taskkill //T //F //PID "$winpid" >/dev/null 2>&1

  sleep 1
  kill -KILL "$pid" 2>/dev/null
  sleep 1

  if kill -0 "$pid" 2>/dev/null; then
    log "kill_tree: pid $pid (winpid ${winpid:-unresolved}) SURVIVED TERM + taskkill + KILL"
    return 1
  fi
  return 0
}

# Record a pid together with the epoch second at which we spawned it. pid_is_ours() below needs
# the stamp to tell our process from a recycled pid.
write_pid() {  # write_pid <name> <pid>
  printf '%s\n' "$2" > "$LOCK/$1"
  date +%s > "$LOCK/$1.started"
}

# Is <pid> alive AND plausibly the process we recorded?
#
# --abort used to `cat .lock/cycle_pid` and taskkill //T it with no checks at all. Windows
# recycles pids aggressively, so a stale lock (driver crashed without cleaning up) turned --abort
# into a force-kill of the process TREE of whatever unrelated program had since inherited that
# number. Three checks, in increasing strength:
#   1. liveness    — `kill -0`. All any platform is guaranteed to give us.
#   2. identity    — the row in `ps -W` must look like the kind of process we started.
#   3. start time  — `ps` prints STIME as HH:MM:SS with no date, so compare seconds-of-day against
#                    the epoch we stamped, trying today and yesterday and keeping the closer
#                    match (10 h cycles cross midnight routinely). Unparseable -> skip 3 and say
#                    so; 1+2 is the floor, not the ceiling.
# On macOS/Linux there is no `ps -W`, so only check 1 runs — that is what the platform allows.
pid_is_ours() {  # pid_is_ours <pid> <stamp_epoch|""> <name_regex>
  local pid="$1" stamp="${2:-}" want="$3" row stime hh mm ss now sod_now midnight c0 c1 d0 d1 best
  if ! kill -0 "$pid" 2>/dev/null; then
    log "  pid $pid is not alive"
    return 1
  fi
  command -v taskkill >/dev/null 2>&1 || return 0     # non-Windows: liveness is all we have

  row="$(ps -W 2>/dev/null | awk -v p="$pid" '$1==p {print; exit}')"
  if [ -z "$row" ]; then
    log "  pid $pid has no row in \`ps -W\` — refusing to taskkill an unidentifiable pid"
    return 1
  fi
  # Here-strings, not `printf | grep -q`: grep -q exits on the first match, and under
  # `set -o pipefail` a SIGPIPE'd producer would make the pipeline's status non-zero and this
  # check reject a process that in fact matched.
  if ! grep -qiE "$want" <<< "$row"; then
    log "  pid $pid does not look like /$want/ — refusing. ps says: $row"
    return 1
  fi

  stime="$(awk '{print $7}' <<< "$row")"
  if [ -z "$stamp" ] || ! grep -qE '^[0-9]{2}:[0-9]{2}:[0-9]{2}$' <<< "$stime"; then
    log "  pid $pid: start time not verifiable (stime='${stime:-?}') — accepting on liveness+name"
    return 0
  fi

  now="$(date +%s)"
  # Seconds-of-day for "now", without GNU-only `date -d`. 10# forces base 10: `date +%H` prints
  # 08/09 and bash would otherwise read those as invalid octal.
  hh="$(date '+%H')"; mm="$(date '+%M')"; ss="$(date '+%S')"
  sod_now=$((10#$hh * 3600 + 10#$mm * 60 + 10#$ss))
  midnight=$((now - sod_now))
  hh="${stime%%:*}"; ss="${stime##*:}"; mm="${stime#*:}"; mm="${mm%%:*}"
  c0=$((midnight + 10#$hh * 3600 + 10#$mm * 60 + 10#$ss))   # started today
  c1=$((c0 - 86400))                                        # ... or yesterday
  d0=$((c0 - stamp)); [ "$d0" -lt 0 ] && d0=$((-d0))
  d1=$((c1 - stamp)); [ "$d1" -lt 0 ] && d1=$((-d1))
  best="$d0"; [ "$d1" -lt "$best" ] && best="$d1"
  if [ "$best" -gt 300 ]; then
    log "  pid $pid started ${best}s away from the pid file's stamp — that is a RECYCLED pid, refusing"
    return 1
  fi
  return 0
}

# ── --abort: stop now, release the lock ───────────────────────────────────────
if [ "${1:-}" = "--abort" ]; then
  touch "$STOP"
  abort_rc=0
  # Kill the running cycle first, then the driver, then the keepawake helper. Killing the driver
  # first would orphan a claude process that is mid-training-run; and keepawake must be handled
  # here explicitly because --abort force-kills the driver, so the EXIT trap that used to be the
  # only thing releasing it never runs (a powershell from pid 19844 was still resident afterwards).
  for f in cycle_pid pid keepawake_pid; do
    [ -f "$LOCK/$f" ] || continue
    pid="$(tr -dc '0-9' < "$LOCK/$f")"
    if [ -z "$pid" ]; then
      log "ABORT: $LOCK/$f holds no pid — ignoring"
      continue
    fi
    stamp=""
    [ -f "$LOCK/$f.started" ] && stamp="$(tr -dc '0-9' < "$LOCK/$f.started")"
    case "$f" in
      keepawake_pid) want='powershell|pwsh' ;;
      pid)           want='bash|sh|driver' ;;
      *)             want='claude|node|bash|caffeinate|systemd-inhibit' ;;
    esac
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ABORT: $f=$pid is already dead — nothing to kill"
      continue
    fi
    if ! pid_is_ours "$pid" "$stamp" "$want"; then
      log "ABORT: REFUSING to kill $f=$pid — it is not recognisably the process we recorded."
      log "  Windows recycles pids; a tree-kill on a stale lock takes down an unrelated program."
      abort_rc=1
      continue
    fi
    log "ABORT: terminating $f=$pid (process tree)"
    if ! kill_tree "$pid"; then
      log "ABORT: kill of $f=$pid FAILED"
      abort_rc=1
    fi
  done
  if [ "$abort_rc" -eq 0 ]; then
    rm -rf "$LOCK"
    log "ABORT: lock released, STOP file set"
  else
    log "ABORT: INCOMPLETE — lock $LOCK deliberately LEFT IN PLACE. Something is still running or"
    log "  could not be identified. Inspect it, then remove by hand:  rm -rf '$LOCK'"
  fi
  exit "$abort_rc"
fi

# ── single-driver guard ───────────────────────────────────────────────────────
if ! mkdir "$LOCK" 2>/dev/null; then
  log "REFUSING TO START: lock $LOCK exists (another driver or cycle is running)."
  log "  If you are sure nothing is running:  rm -rf '$LOCK'"
  exit 1
fi
write_pid pid $$
trap 'rm -rf "$LOCK"; kill_tree "${KEEPAWAKE_PID:-}"; log "driver exited, lock released"' EXIT

rm -f "$STOP"
log "=============================================================="
log "campaign driver starting"
log "  root         : $ROOT"
log "  python       : $PY"
log "  claude config: ${CLAUDE_CONFIG_DIR:-<CLI default>}"
log "  cycle prompt : $CYCLE_PROMPT"
if [ "$MAX_DAILY_S" -gt 0 ]; then
  log "  cycle timeout: ${CYCLE_TIMEOUT_S}s     budget: ${MAX_DAILY_H}h per rolling 24 h"
else
  log "  cycle timeout: ${CYCLE_TIMEOUT_S}s     budget: DAILY CAP DISABLED (runs continuously)"
  log "     stop with: touch autoresearch/STOP   (honoured between cycles and inside any wait)"
  log "     remaining backstops: max_cycles=${MAX_CYCLES}, ${CYCLE_H}h per cycle, disk floors"
fi
log "  backstops    : max_cycles=${MAX_CYCLES}  results/<=${RESULTS_MAX_GB}GB  free>=${MIN_FREE_GB}GB"
log "=============================================================="

# ── rolling-window budget ledger ──────────────────────────────────────────────
# WHY a rolling window and not a calendar day. The old ledger was one line, `<date> <hours>`:
# budget_today() returned 0 whenever the stored date was not today's, and budget_add() wrote
# `today + secs`. A cycle spanning midnight therefore WIPED the previous day's total and billed
# its entire duration to the day it ENDED. Verified: cycle #3 ran 22:17:45 (2026-08-09) ->
# 06:56:55 (2026-08-10), 31150 s; .budget went `2026-08-09 1.462` -> `2026-08-10 8.653`, so
# ~7.2 h of 08-09's work was charged to 08-10 and 08-09's own 1.462 h vanished.
#
# The overlap of a set of (start, end) intervals with [now-24h, now] has no such seam. Format:
#
#     cycle <start_epoch> <end_epoch>
#
# Append-only. Any line that is not exactly that (including the legacy `<date> <hours>` line, and
# anything half-written by a crash) is IGNORED rather than fatal — this file is gitignored runtime
# state, not an input we can validate up front.
#
# Known limitation, stated so nobody is surprised: an interval is appended when the cycle ENDS, so
# a cycle killed by a machine crash is not billed at all. The lock is what protects against
# re-entry in that case, not the ledger.
budget_record() {  # budget_record <start_epoch> <end_epoch>
  printf 'cycle %s %s\n' "$1" "$2" >> "$BUDGET_FILE"
}

# Seconds of recorded cycle wall-clock lying inside [now-24h, now].
budget_window_s() {  # budget_window_s <now_epoch>
  [ -f "$BUDGET_FILE" ] || { printf '0'; return 0; }
  awk -v now="$1" -v win="$BUDGET_WINDOW_S" '
    $1=="cycle" && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ {
      s=$2+0; e=$3+0
      if (e < s) next
      lo = now-win; if (s > lo) lo = s
      hi = now;     if (e < hi) hi = e
      if (hi > lo) t += hi-lo
    }
    END { printf "%d", t+0 }
  ' "$BUDGET_FILE"
}

# Projected cost of the cycle we are about to launch: median of the last 3 recorded durations,
# 4 h if we have fewer than 3. Used to refuse a cycle that would breach the window, instead of
# admitting it and discovering the breach afterwards — cycle #5 started at 8.815h/12h and left the
# day at 18.030h precisely because the budget was checked only BEFORE a cycle and never during.
budget_projected_s() {
  [ -f "$BUDGET_FILE" ] || { printf '14400'; return 0; }
  awk '
    $1=="cycle" && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ && $3 >= $2 { d[++n] = $3-$2 }
    END {
      if (n < 3) { printf "14400"; exit }
      a=d[n-2]; b=d[n-1]; c=d[n]
      m = (a>b) ? ((b>c) ? b : ((a>c) ? c : a)) : ((a>c) ? a : ((b>c) ? c : b))
      printf "%d", m
    }
  ' "$BUDGET_FILE"
}

# How long until the rolling window has shed <deficit> seconds of spend?
#
# As the window's left edge L=now-24h advances by dt, we lose exactly the recorded time lying in
# [L, L+dt]. So walk the clipped intervals oldest-first, accumulate, and stop at the point where
# the accumulated amount reaches the deficit. Intervals are appended in chronological order (one
# cycle at a time, under the lock), so file order is time order.
#
# This replaces `sleep 1800; continue`, which on 2026-08-10 idled the driver for exactly 8h00m01s
# in 16 identical iterations because it never asked when the budget would actually free up.
budget_wait_s() {  # budget_wait_s <now_epoch> <deficit_s>
  [ -f "$BUDGET_FILE" ] || { printf '%s' "$BUDGET_WINDOW_S"; return 0; }
  awk -v now="$1" -v need="$2" -v win="$BUDGET_WINDOW_S" '
    $1=="cycle" && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ {
      s=$2+0; e=$3+0
      if (e < s) next
      lo = now-win; if (s < lo) s = lo
      if (e > now) e = now
      if (e <= s) next
      n++; S[n]=s; E[n]=e
    }
    END {
      acc = 0
      for (i=1; i<=n; i++) {
        d = E[i]-S[i]
        if (acc + d >= need) { printf "%d", (S[i] + (need-acc)) - (now-win); exit }
        acc += d
      }
      printf "%d", win    # not enough recorded to shed `need` — wait the whole window out
    }
  ' "$BUDGET_FILE"
}

# Sleep, but wake as soon as a STOP is requested. A single long unattended sleep used to make
# `touch STOP` take up to half an hour to be noticed.
sleep_interruptible() {
  local left="${1:-0}" slice
  while [ "$left" -gt 0 ]; do
    [ -f "$STOP" ] && return 0
    slice=60; [ "$left" -lt 60 ] && slice="$left"
    sleep "$slice"
    left=$((left - slice))
  done
  return 0
}

# ── campaign state helpers ────────────────────────────────────────────────────
# Prints "<campaign cycle> <history length>". Unknown cycle is "?", unknown history is -1 — and
# -1 compares equal to -1, so an unparseable state.json reads as "history did not grow", which is
# the conservative answer for the no-op test below.
state_read() {
  "$PY" - "$STATE_FILE" 2>/dev/null <<'PYEOF' || printf '? -1\n'
import sys, json
c, n = "?", -1
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    if isinstance(d.get("cycle"), int):
        c = d["cycle"]
    h = d.get("history")
    if isinstance(h, list):
        n = len(h)
except Exception:
    pass
print(c, n)
PYEOF
}

# Park the campaign: state.json gets "mode": "blocked" plus the reason, and the driver exits.
# Refuses to touch an unparseable state.json rather than replacing it with a stub — losing the
# campaign's state file would be far worse than not recording the block.
mark_blocked() {  # mark_blocked <reason>
  "$PY" - "$STATE_FILE" "$1" <<'PYEOF'
import sys, json, os, datetime
path, reason = sys.argv[1], sys.argv[2]
try:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
except Exception as exc:
    # ASCII only: this goes to a Windows console whose codepage may not be UTF-8.
    sys.stderr.write("state.json unreadable (%s) - NOT overwriting it\n" % exc)
    raise SystemExit(1)
d["mode"] = "blocked"
d["blocked_reason"] = reason
d["blocked_at"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(d, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
os.replace(tmp, path)
PYEOF
}

# ── run a command with a timeout, portably (macOS has no coreutils timeout) ───
run_with_timeout() {
  local secs="$1"; shift
  "$@" &
  local cmd_pid=$!
  write_pid cycle_pid "$cmd_pid"
  # The watchdog used to be `( sleep "$secs"; kill_tree "$cmd_pid" )`. If the TERM that cancels it
  # after `wait` were ever lost, it would wake up hours later and tree-kill whatever MSYS pid had
  # since been recycled into that number. Poll in short slices instead and exit the moment this
  # cycle is over: run_with_timeout removes $LOCK/cycle_pid on the way out, so a missing file or a
  # different pid in it means this watchdog has nothing left to guard.
  (
    local waited=0
    while [ "$waited" -lt "$secs" ]; do
      sleep 15
      waited=$((waited + 15))
      [ "$(cat "$LOCK/cycle_pid" 2>/dev/null)" = "$cmd_pid" ] || exit 0
      kill -0 "$cmd_pid" 2>/dev/null || exit 0
    done
    log "WATCHDOG: pid $cmd_pid exceeded ${secs}s — killing the process tree"
    kill_tree "$cmd_pid"
  ) &
  local watchdog=$!
  wait "$cmd_pid"; local rc=$?
  rm -f "$LOCK/cycle_pid" "$LOCK/cycle_pid.started"
  kill -TERM "$watchdog" 2>/dev/null
  return $rc
}

# Keep the machine awake for overnight campaigns. macOS: caffeinate. Linux/WSL: systemd-inhibit
# when available. Neither is required — the loop just runs unprotected without them.
CAFF=""
KEEPAWAKE_PID=""
if command -v caffeinate >/dev/null 2>&1; then
  CAFF="caffeinate -i"
elif command -v systemd-inhibit >/dev/null 2>&1; then
  CAFF="systemd-inhibit --what=idle --why=mfas-campaign"
elif command -v powershell >/dev/null 2>&1 && [ -f "$AR/keepawake.ps1" ]; then
  # Windows: neither of the above exists. A separate process holds SetThreadExecutionState for
  # as long as the driver runs; the trap below releases it. Not a CAFF prefix, because the
  # request must span the whole campaign rather than a single cycle.
  powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$AR/keepawake.ps1")" \
      >> "$LOGDIR/keepawake.log" 2>&1 &
  KEEPAWAKE_PID=$!
  # The EXIT trap is not enough: --abort force-kills the driver, so the trap never runs and this
  # powershell leaks (one from pid 19844 outlived its driver by days). Record it where --abort
  # can find — and verify — it.
  write_pid keepawake_pid "$KEEPAWAKE_PID"
fi

# ── preflight: headless auth ──────────────────────────────────────────────────
# Headless runs need credentials. On the MacBook those came from .claude/settings.local.json
# (CLAUDE_CODE_OAUTH_TOKEN); on this box they come from an interactive `claude /login`, which
# writes to the CLI's own config dir. Either is fine — so the check that matters is the live one:
# does a headless invocation actually answer? Fail loudly now rather than burning the budget on a
# crash loop of "Not logged in".
#
# The old check piped the reply into `grep -qi "ok"`, which passes on ANY message containing that
# substring — token, broken, revoked, cookie. "Invalid API token" sailed straight through. Ask for
# a sentinel that cannot appear by accident and match it case-sensitively as a fixed string. The
# call is also bounded by the timeout machinery now: this runs before the first cycle, so a hung
# `claude -p` here had no timeout of any kind and could wedge the driver indefinitely.
AUTH_LOG="$LOGDIR/preflight-auth.log"
run_with_timeout "$AUTH_TIMEOUT_S" \
  "$CLAUDE_BIN" -p "reply with exactly the word $AUTH_SENTINEL and nothing else" \
    --dangerously-skip-permissions \
    < /dev/null > "$AUTH_LOG" 2>&1
auth_rc=$?
if [ $auth_rc -ne 0 ] || ! grep -qF "$AUTH_SENTINEL" "$AUTH_LOG"; then
  log "HALT: headless auth check failed (rc=$auth_rc, no $AUTH_SENTINEL in the reply)."
  log "  Reply was: $(head -c 300 "$AUTH_LOG" | tr '\n' ' ')"
  log "  Run \`claude\` interactively and \`/login\` (or refresh CLAUDE_CODE_OAUTH_TOKEN in"
  log "  $ROOT/.claude/settings.local.json), then restart the driver."
  exit 1
fi
log "preflight: headless auth OK ($AUTH_SENTINEL received)"

# ── main loop ─────────────────────────────────────────────────────────────────
cycle=0            # driver-local sequence number (resets whenever the driver restarts)
noop_streak=0      # consecutive cycles that changed nothing
noop_total=0       # no-ops this driver has burned — budget spent that left no trace in history
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

  # campaign.yaml budget.results_dir_max_gb, previously declared but enforced nowhere. `du -sk`
  # is POSIX and gives kilobytes, which compare without unit parsing (`du -sh` does not).
  results_kb=$(du -sk "$ROOT/results" 2>/dev/null | awk '{print $1+0; exit}')
  results_gb=$(awk -v k="${results_kb:-0}" 'BEGIN{printf "%.2f", k/1048576}')
  if awk -v g="$results_gb" -v m="$RESULTS_MAX_GB" 'BEGIN{exit !(g >= m)}'; then
    log "HALT: results/ is ${results_gb}GB (>= ${RESULTS_MAX_GB}GB, campaign.yaml"
    log "  budget.results_dir_max_gb). Prune or archive it before restarting the driver."
    break
  fi

  # Campaign state, read BEFORE the cycle: the campaign cycle number for the log (driver-local
  # numbering alone made driver.log #1..#7 map to campaign cycles {—,2,3,—,5,6,7}, which cost real
  # time to untangle) and the history length for the no-op test after the cycle.
  state_before="$(state_read)"
  camp_cycle="${state_before%% *}"; [ -z "$camp_cycle" ] && camp_cycle="?"
  hist_before="${state_before##* }"; [ -z "$hist_before" ] && hist_before="-1"

  # campaign.yaml budget.max_cycles — a hard backstop on TOTAL research cycles, also previously
  # unimplemented. history/state.cycle account for every cycle that left a trace; noop_total adds
  # the ones this driver burned that did not (they cost budget just the same).
  cycles_used="$hist_before"
  [ "$camp_cycle" != "?" ] && [ "$camp_cycle" -gt "$cycles_used" ] 2>/dev/null && cycles_used="$camp_cycle"
  [ "$cycles_used" -lt 0 ] && cycles_used=0
  cycles_used=$((cycles_used + noop_total))
  if [ "$MAX_CYCLES" -gt 0 ] && [ "$cycles_used" -ge "$MAX_CYCLES" ]; then
    log "HALT: campaign cycle backstop reached ($cycles_used / $MAX_CYCLES, campaign.yaml"
    log "  budget.max_cycles). Raise it deliberately or end the campaign."
    break
  fi

  # ── rolling 24 h budget gate ────────────────────────────────────────────────
  # MAX_DAILY_S == 0 means the cap is DISABLED (campaign.yaml max_wall_clock_h_per_day: 0),
  # same 0-means-off idiom budget.max_cycles already uses. The ledger keeps accruing either
  # way, so `spend_s` stays honest in the logs and turning the cap back on needs no reset.
  now_s=$(date +%s)
  spend_s="$(budget_window_s "$now_s")"
  proj_s="$(budget_projected_s)"
  if [ "$MAX_DAILY_S" -gt 0 ] && [ "$proj_s" -gt "$MAX_DAILY_S" ]; then
    log "NOTE: projected cycle cost $(hms "$proj_s") exceeds the whole ${MAX_DAILY_H}h window;"
    log "  clamping the projection to the cap — the timeout below cuts the cycle short instead of"
    log "  deadlocking the driver forever."
    proj_s="$MAX_DAILY_S"
  fi
  if [ "$MAX_DAILY_S" -gt 0 ] && [ $((spend_s + proj_s)) -gt "$MAX_DAILY_S" ]; then
    deficit=$((spend_s + proj_s - MAX_DAILY_S))
    wait_s="$(budget_wait_s "$now_s" "$deficit")"
    [ "$wait_s" -lt 60 ] && wait_s=60
    resume="$(date -d "@$((now_s + wait_s))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || printf 'in %ss' "$wait_s")"
    log "budget window full: $(hms "$spend_s") spent in the last 24 h + $(hms "$proj_s") projected"
    log "  > cap ${MAX_DAILY_H}h. Waiting $(hms "$wait_s") for the window to free up (until $resume)."
    sleep_interruptible "$wait_s"
    continue
  fi
  # Cap the launched cycle at what the window can actually afford, so a cycle can never blow the
  # budget it was admitted under.
  eff_timeout_s="$CYCLE_TIMEOUT_S"
  if [ "$MAX_DAILY_S" -gt 0 ]; then
    remain_s=$((MAX_DAILY_S - spend_s))
    if [ "$remain_s" -lt "$eff_timeout_s" ]; then
      eff_timeout_s="$remain_s"
    fi
  fi

  cycle=$((cycle + 1))
  stamp="$(date '+%Y%m%dT%H%M%S')"
  cycle_log="$LOGDIR/cycle-${stamp}.log"
  head_before="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
  log "--- driver cycle #$cycle (campaign cycle $camp_cycle, history=$hist_before) starting"
  if [ "$MAX_DAILY_S" -gt 0 ]; then
    log "      budget $(hms "$spend_s")/${MAX_DAILY_H}h in window, timeout ${eff_timeout_s}s -> $cycle_log"
  else
    log "      spent $(hms "$spend_s") in the last 24 h (cap off), timeout ${eff_timeout_s}s -> $cycle_log"
  fi
  [ "$hist_before" = "-1" ] && log "      NOTE: state.json history unreadable — the no-op test below is weakened"

  start_s=$(date +%s)
  # shellcheck disable=SC2086
  # stdin from /dev/null: headless claude waits ~3s for piped input otherwise, and a detached
  # nohup'd driver has no terminal to read from.
  run_with_timeout "$eff_timeout_s" \
    $CAFF "$CLAUDE_BIN" -p "$CYCLE_PROMPT" \
      --dangerously-skip-permissions \
      $MODEL_ARG \
      < /dev/null > "$cycle_log" 2>&1
  rc=$?
  end_s=$(date +%s)
  dur=$((end_s - start_s))
  budget_record "$start_s" "$end_s"
  window_s="$(budget_window_s "$end_s")"

  state_after="$(state_read)"
  camp_cycle_after="${state_after%% *}"; [ -z "$camp_cycle_after" ] && camp_cycle_after="?"
  hist_after="${state_after##* }"; [ -z "$hist_after" ] && hist_after="-1"
  head_after="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"

  if [ $rc -eq 0 ]; then
    log "--- driver cycle #$cycle / campaign cycle $camp_cycle_after finished rc=0 in ${dur}s"
    log "      window now $(hms "$window_s")/${MAX_DAILY_H}h"
  else
    log "--- driver cycle #$cycle / campaign cycle $camp_cycle_after EXITED rc=$rc after ${dur}s"
    log "      window now $(hms "$window_s")/${MAX_DAILY_H}h — see $cycle_log"
    tail -20 "$cycle_log" | sed 's/^/      | /' | tee -a "$DRIVER_LOG" >/dev/null
  fi

  # ── no-op detection ─────────────────────────────────────────────────────────
  # The exit code is NOT evidence that a cycle did research. Driver cycles #1 and #4 both logged
  # "finished OK", produced no commit and no state.json history entry, and burned ~33 min of
  # budget between them in silence; the "suspiciously short" guard below has a 60 s threshold, so
  # a 583 s no-op sailed through it. The only trustworthy signals a cycle leaves are a moved git
  # HEAD and a longer state.json history — require at least one of them.
  if [ "$head_after" = "$head_before" ] && [ "$hist_after" = "$hist_before" ]; then
    noop_streak=$((noop_streak + 1))
    noop_total=$((noop_total + 1))
    log "!!! NO-OP CYCLE (#$noop_streak in a row): HEAD still $head_before and history still"
    log "!!!   $hist_before after ${dur}s. The cycle produced no commit and no history entry."
    [ "$hist_before" = "-1" ] && log "!!!   (state.json was unreadable, so 'history unchanged' is an assumption, not a fact)"
    log "!!!   Tail of $cycle_log:"
    tail -20 "$cycle_log" | sed 's/^/      | /' | tee -a "$DRIVER_LOG" >/dev/null
    if [ "$noop_streak" -ge 3 ]; then
      reason="driver halted after $noop_streak consecutive no-op cycles (no git commit, no state.json history entry); last cycle log: $cycle_log"
      if mark_blocked "$reason" 2>>"$DRIVER_LOG"; then
        log "HALT: 3 consecutive no-op cycles — state.json mode set to \"blocked\". A human must"
        log "  look at $cycle_log before the campaign resumes."
      else
        log "HALT: 3 consecutive no-op cycles — could NOT write mode=blocked into state.json"
        log "  (see the error above). Set it by hand before restarting."
      fi
      break
    fi
    # Escalating backoff: something systemic is wrong (auth, a wedged prompt, a crash loop) and
    # retrying immediately just spends budget faster.
    backoff=600
    [ "$noop_streak" -ge 2 ] && backoff=1800
    log "backing off ${backoff}s before retrying"
    sleep_interruptible "$backoff"
    continue
  fi

  noop_streak=0

  # A cycle that dies in seconds but DID commit something is odd rather than broken; still back
  # off rather than risking a fast loop.
  if [ "$dur" -lt 60 ]; then
    log "cycle was suspiciously short (${dur}s) — backing off 10 min"
    sleep_interruptible 600
  else
    sleep_interruptible "$PAUSE_S"
  fi
done

log "driver loop ended after $cycle cycle(s), $noop_total of them no-ops"
