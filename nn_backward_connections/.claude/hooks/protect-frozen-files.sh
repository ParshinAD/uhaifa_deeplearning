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
  # Integrity reference + its checker — protecting these stops a tampered oracle from
  # being "re-blessed" by silently rewriting the manifest or disabling the gate.
  "eval/frozen.sha256"
  "eval/frozen_guard.py"
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

# ── Track B/C hardening (leakage boundary + results namespace) ──
# The near-optimal reference must never be edited — mutating it would silently corrupt every
# gap/diagnostic measurement. It is read-only and only via mfas.analysis.gap.
if [[ "$norm" == *data/best_solution* ]]; then
  echo "BLOCKED: '$FILE_PATH' is the near-optimal reference (data/best_solution) — read-only, and only via mfas.analysis.gap. See experiments/PROTOCOL.md § 'Track B — Diagnostics'." >&2
  exit 2
fi

# results/*.json records are produced ONLY by the frozen runner (eval/run_variant.py), never
# hand-authored: hand-editing a metric is fabrication (CLAUDE.md invariant #3), and it is also how
# a diagnostic could leak the target into results/. The runner writes via Python I/O (not the
# Edit/Write tool), so it is unaffected. (Covers results/randomgraph/*.json too.)
if [[ "$norm" == results/*.json ]] || [[ "$norm" == */results/*.json ]]; then
  echo "BLOCKED: '$FILE_PATH' — results/*.json is runner-produced only (eval/run_variant.py). Hand-editing a metric is forbidden (no fabricated numbers). See experiments/PROTOCOL.md." >&2
  exit 2
fi

exit 0
