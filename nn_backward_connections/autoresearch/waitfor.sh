#!/usr/bin/env bash
# Block on the detached sweep for a bounded interval, then report and exit.
#
# Designed to be called REPEATEDLY from a cycle: each call blocks for up to --for seconds (default
# 540, comfortably inside an agent Bash timeout) and then returns, so the agent stays in its turn
# instead of ending it while a run is in flight. Exit codes let a loop branch:
#   0  sweep finished (prints DONE and the per-run outcomes)
#   10 still running (prints elapsed + the last progress line) -> call again
#   2  no sweep has been launched
#
# Usage:  bash autoresearch/waitfor.sh [--for 540]
set -uo pipefail
AR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$AR/.sweep"
BUDGET=540
[ "${1:-}" = "--for" ] && BUDGET="${2:-540}"

if [ ! -f "$STATE/running" ] && [ ! -f "$STATE/done" ]; then
  echo "NO SWEEP: nothing launched. Start one with autoresearch/sweep.sh --exp <id> --role <role>."
  exit 2
fi

started=$SECONDS
while [ $((SECONDS - started)) -lt "$BUDGET" ]; do
  if [ -f "$STATE/done" ]; then
    rc="$(cat "$STATE/done" 2>/dev/null || echo '?')"
    echo "DONE rc=$rc"
    grep -E "DONE  score|rc=[0-9]+$" "$STATE/log" 2>/dev/null | tail -20
    exit 0
  fi
  sleep 15
done

elapsed_total=""
[ -f "$STATE/running" ] && elapsed_total="(sweep started $(cat "$STATE/running"))"
echo "STILL RUNNING after ${BUDGET}s of this call $elapsed_total"
echo "  completed so far: $(grep -c 'DONE  score' "$STATE/log" 2>/dev/null || echo 0) run(s)"
tail -1 "$STATE/log" 2>/dev/null | sed 's/^/  last: /'
echo "  -> call this again. Do NOT end your turn: a cycle that stops here loses the whole sweep."
exit 10
