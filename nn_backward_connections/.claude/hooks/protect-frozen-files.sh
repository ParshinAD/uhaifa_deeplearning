#!/usr/bin/env bash
# PreToolUse guardrail: BLOCK any Edit/Write to a FROZEN file.
#
# The frozen oracle/harness/tests define the ground-truth metric. An autonomous run must
# never silently "improve" the metric by editing them. This hook denies such edits.
#
# Input : JSON on stdin (PreToolUse), with .tool_input.file_path
# Block : exit 2 + reason on stderr (Claude Code convention)
# Allow : exit 0
#
# PORT NOTE (Windows): every pattern below is written with '/' separators, and the sandbox
# check tested for a leading '/'. Claude Code on native Windows hands over
# 'D:\repo\src\mfas\metrics.py', so on the unported hook NOTHING matched and the guard
# silently allowed every edit — verified by a probe Edit against src/mfas/metrics.py.
# `canon()` below folds '\' -> '/' and maps 'D:/x' to the Git-Bash spelling '/d/x'.
set -euo pipefail

JQ="${MFAS_JQ:-/usr/bin/jq}"
command -v "$JQ" >/dev/null 2>&1 || JQ="jq"

INPUT="$(cat)"
FILE_PATH="$(printf '%s' "$INPUT" | "$JQ" -r '.tool_input.file_path // empty' 2>/dev/null || true)"

# No file path (non-file tool) -> nothing to protect.
[ -z "$FILE_PATH" ] && exit 0

# Optional forensic trace (set MFAS_HOOK_LOG to a writable path) — proves the hook fired.
if [ -n "${MFAS_HOOK_LOG:-}" ]; then
  printf '%s\n' "$FILE_PATH" >> "$MFAS_HOOK_LOG" 2>/dev/null || true
fi

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

# Normalize separators, strip a leading "./", and give a Windows drive path its Git-Bash
# spelling so both spellings of the same file compare equal.
canon() {
  local p="${1//\\//}"
  p="${p#./}"
  case "$p" in
    [A-Za-z]:/*) p="/$(printf '%s' "${p%%:*}" | tr '[:upper:]' '[:lower:]')/${p#*:/}" ;;
  esac
  printf '%s' "$p"
}

norm="$(canon "$FILE_PATH")"

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

# ── Phase-7 hardening (autonomous campaign) ──
# The champion registry decides what every future variant is compared against. It is written
# ONLY by autoresearch/update_sota.py, which refuses to promote without a passing audit. Letting
# an agent hand-edit it would let a campaign crown itself.
if [[ "$norm" == autoresearch/sota.json ]] || [[ "$norm" == */autoresearch/sota.json ]]; then
  echo "BLOCKED: '$FILE_PATH' — the champion registry is written only by autoresearch/update_sota.py (which requires a passing audit). See autoresearch/CAMPAIGN.md." >&2
  exit 2
fi

# Sandbox containment: the campaign runs in one checkout and must never write outside it.
# The original repository is read-only reference material. Temp dirs stay allowed so ordinary
# tooling keeps working. Both sides are canonicalized, so 'D:\...' and '/d/...' compare equal.
# CLAUDE_PROJECT_DIR is not guaranteed to reach the hook process on every platform, and a
# missing root would silently disable containment — so fall back to this script's own location
# (.claude/hooks/ -> campaign root), which is always correct.
SANDBOX_ROOT="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$SANDBOX_ROOT" ]; then
  SANDBOX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [ -n "$SANDBOX_ROOT" ]; then
  root="$(canon "$SANDBOX_ROOT")"
  case "$norm" in
    /*|[A-Za-z]:/*)
      case "$norm" in
        "$root"/*|/tmp/*|/private/tmp/*|/var/folders/*|/private/var/folders/*|/c/users/*/appdata/local/temp/*)
          ;;
        *)
          echo "BLOCKED: '$FILE_PATH' is OUTSIDE the campaign sandbox ($SANDBOX_ROOT). The autonomous campaign may not write to the original repository or anywhere else on disk. See autoresearch/CAMPAIGN.md § 'The five things this campaign must never do'." >&2
          exit 2
          ;;
      esac
      ;;
  esac
fi

exit 0
