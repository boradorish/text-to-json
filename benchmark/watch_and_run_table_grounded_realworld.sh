#!/usr/bin/env bash
# Wait for GPU 0 to be genuinely idle, then launch the reproducible full run.
# This never stops or competes with an existing GPU process.
set -euo pipefail

ROOT="/mnt/ddn/prod-runs/interns/sunghee/text-to-json"
LOG_DIR="$ROOT/outputs/table_grounded_realworld_all"
LOCK="$LOG_DIR/launch.lock"
mkdir -p "$LOG_DIR"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another table-grounded launcher already holds $LOCK; exiting."
  exit 0
fi

while true; do
  USED=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
  if [ "${USED:-999999}" -lt 1024 ]; then
    break
  fi
  echo "$(date -u +%FT%TZ) GPU 0 busy (${USED} MiB); waiting 30s"
  sleep 30
done

echo "$(date -u +%FT%TZ) GPU 0 is idle; launching table-grounded full run"
cd "$ROOT"
exec /root/work/sunghee/venv/bin/python benchmark/run_table_grounded_realworld.py --gpu 0
