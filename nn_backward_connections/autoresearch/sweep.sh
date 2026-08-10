#!/usr/bin/env bash
# Launch a DETACHED evaluation sweep and return immediately.
#
# Why this exists. A single microns run takes ~3240 s here, and an agent's Bash call is capped far
# below that, so a cycle cannot run one in the foreground. Cycle #1 (2026-08-09) hit the
# consequence: it started a microns run, ended its turn while waiting, the headless session
# exited, the child died with it, and ~50 minutes of GPU work produced nothing. Every cycle would
# have repeated that forever.
#
# So: `nohup`/`setsid` the sweep so it OUTLIVES the session that started it. If the cycle survives,
# it polls with waitfor.sh; if the cycle dies, the runs keep going and their results/*.json land
# anyway, and the next cycle finds them instead of redoing them.
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

AR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$AR/.sweep"
mkdir -p "$STATE"

if [ -f "$STATE/running" ]; then
  echo "REFUSED: a sweep is already in flight (see $STATE/log). Poll it with waitfor.sh, or"
  echo "  remove $STATE/running if you are certain nothing is running."
  exit 1
fi

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
rm -f "$STATE/done"
date '+%Y-%m-%d %H:%M:%S' > "$STATE/running"

runner() {
  local rc_all=0
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
  echo "$rc_all" > "$STATE/done"
  rm -f "$STATE/running"
}

# setsid where available so the sweep is not in the session's process group at all.
if command -v setsid >/dev/null 2>&1; then
  setsid bash -c "$(declare -f runner); EXP='$EXP' ROLE='$ROLE' PY='$PY' LOG='$LOG' PLAN='$PLAN' STATE='$STATE' runner" </dev/null >/dev/null 2>&1 &
else
  nohup bash -c "$(declare -f runner); EXP='$EXP' ROLE='$ROLE' PY='$PY' LOG='$LOG' PLAN='$PLAN' STATE='$STATE' runner" </dev/null >/dev/null 2>&1 &
fi
disown 2>/dev/null

echo "LAUNCHED  exp=$EXP role=$ROLE  (plan: autoresearch/.sweep/plan)"
sed 's/^/  /' "$PLAN"
echo "  log:  autoresearch/.sweep/log"
echo "  poll: bash autoresearch/waitfor.sh   (repeat until DONE — do NOT end your turn instead)"
