#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs"
OUT_DIR="./output"
mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "[${TS}] Starting run" 
python3 main.py --config config/config.yaml  2>&1 
RC=${PIPESTATUS[0]}

if [[ $RC -ne 0 ]]; then
  echo "[${TS}] Run failed with exit code $RC" 
  exit $RC
fi

echo "[${TS}] Run completed successfully" 


