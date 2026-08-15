#!/usr/bin/env bash
# Launch ONE long job so it OUTLIVES the session that started it, and return immediately.
# For sweeps use autoresearch/sweep.sh; this is the same detach pattern for everything else
# (prototypes, ad-hoc scripts) so a cycle cannot get it wrong.
#
#   bash autoresearch/detach.sh dr_tmp/proto_H43.out "$PY" dr_tmp/proto_H43.py --arm a
#   tail -5 dr_tmp/proto_H43.out     # poll in a loop; NEVER end your turn while it runs
#
# WHY THE SHAPE IS EXACTLY THIS — all three measured on this box, 2026-08-15. Do not "simplify" it:
#   * `setsid` DOES NOT EXIST here (Git Bash / MSYS). Any recipe naming it is a recipe that fails.
#   * `nohup <python.exe> &` does NOT survive this session's exit: nohup does not detach a NATIVE
#     Windows process, and neither does putting a launcher bash in front of it that then exits.
#   * `nohup bash -c '...' &` DOES survive, as long as that MSYS bash STAYS ALIVE owning the native
#     process as its CHILD. Hence bash -c '"$@"' and not exec: exec would replace the very bash
#     that makes this work. This is also the only reason sweep.sh survives, undocumented until now.
#   * </dev/null (a headless session's stdin disappears), output to a FILE (a pipe dies with the
#     session), `disown` (nothing left in the job table for a signal to reach).
set -uo pipefail
[ $# -lt 2 ] && { echo "usage: detach.sh <logfile> <command> [args...]" >&2; exit 2; }
LOGF="$1"; shift
mkdir -p "$(dirname "$LOGF")" 2>/dev/null

# The command travels as POSITIONAL PARAMETERS, never interpolated into the -c string: a value
# containing a quote would otherwise become shell code (the same defect fixed in sweep.sh, D4).
nohup bash -c '"$@"' _ "$@" </dev/null >>"$LOGF" 2>&1 &
PID=$!
disown 2>/dev/null

printf 'DETACHED pid=%s  log=%s  started %s\n' "$PID" "$LOGF" "$(date '+%Y-%m-%d %H:%M:%S')"
echo "  cmd:  $*"
echo "  poll: tail -5 '$LOGF'   (repeat until it finishes — do NOT end your turn instead)"
echo "  live: kill -0 $PID 2>/dev/null && echo alive || echo gone"
echo "  record what you launched, where its output lands and when, in state.json.current_item_note"
