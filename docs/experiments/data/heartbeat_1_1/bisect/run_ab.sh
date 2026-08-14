#!/bin/bash
# Sequential A/B: OLD (Exp 48 commit) then NEW (1.1 candidate). ONE harness at a
# time — two concurrent harnesses clobber each other's workdir and contend for
# the LLM server (learned 2026-08-11).
set -u
PY=~/Scripts/Maxim/.venv/bin/python
D=~/Scripts/Maxim-heartbeat-1.1/heartbeat_data/bisect
OLD=~/Scripts/Maxim-exp48-bisect        # 45bd1789
NEW=~/Scripts/Maxim-heartbeat-1.1       # f05c63aa

echo "[$(date +%H:%M)] ARM A: OLD 45bd1789" >> "$D/ab.log"
cd "$OLD" && PYTHONPATH="$OLD/src" "$PY" scripts/benchmark_cradle_mother.py \
  --arms taught --trials 2 --seed-base 42 --model mistral-7b --timeout-s 7200 \
  --out "$D/old_45bd1789.jsonl" --workdir "$D/old_runs" >> "$D/ab.log" 2>&1

echo "[$(date +%H:%M)] ARM B: NEW f05c63aa" >> "$D/ab.log"
cd "$NEW" && PYTHONPATH="$NEW/src" "$PY" scripts/benchmark_cradle_mother.py \
  --arms taught --trials 2 --seed-base 42 --model mistral-7b --timeout-s 7200 \
  --out "$D/new_f05c63aa.jsonl" --workdir "$D/new_runs" >> "$D/ab.log" 2>&1

echo "[$(date +%H:%M)] AB COMPLETE" >> "$D/ab.log"
"$PY" "$D/nomove_rate.py" "$D/old_runs" "$D/new_runs" >> "$D/ab.log" 2>&1
