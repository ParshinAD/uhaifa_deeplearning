#!/usr/bin/env bash
# PreToolUse guardrail: BLOCK any Edit/Write to a FROZEN file.
#
# The frozen oracle/harness/tests define the ground-truth metric. An autonomous run must
# never silently "improve" the metric by editing them. This hook denies such edits.
#
# Input : JSON on stdin (PreToolUse), with .tool_input.file_path
# Block : exit 2 + reason on stderr (Claude Code convention)
# Allow : exit 0
set -euo pipefail

JQ="${MFAS_JQ:-/usr/bin/jq}"
command -v "$JQ" >/dev/null 2>&1 || JQ="jq"

INPUT="$(cat)"
FILE_PATH="$(printf '%s' "$INPUT" | "$JQ" -r '.tool_input.file_path // empty' 2>/dev/null || true)"

# No file path (non-file tool) -> nothing to protect.
[ -z "$FILE_PATH" ] && exit 0

# Frozen files, as repo-relative suffixes. Matching on suffix handles both absolute and
# relative paths the tool might present.
FROZEN=(
  "src/mfas/metrics.py"
  "eval/harness.py"
  "eval/aggregate.py"
  "tests/test_metrics.py"
)

# Normalize: strip a leading "./"
norm="${FILE_PATH#./}"

for f in "${FROZEN[@]}"; do
  # match exact path or any path ending in "/<frozen>"
  if [ "$norm" = "$f" ] || [[ "$norm" == */"$f" ]]; then
    echo "BLOCKED: '$FILE_PATH' is a FROZEN file (oracle/harness/tests). Edits are forbidden — it defines the ground-truth metric. See CLAUDE.md / experiments/PROTOCOL.md." >&2
    exit 2
  fi
done

exit 0
