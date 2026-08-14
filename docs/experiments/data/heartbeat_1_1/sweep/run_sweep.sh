#!/bin/bash
# Explore-weight sweep, arm-controlled per the concurrent session's discipline:
# taught AND no_feed at each weight, so a change in taught can be attributed.
# weight 1.5 is already measured (n=12 both arms, heartbeat run).
# ONE harness at a time (S8). Durable workdir (S4). stdin closed.
set -u
PY=~/Scripts/Maxim/.venv/bin/python
W=~/Scripts/Maxim-heartbeat-1.1
D=$W/heartbeat_data/sweep

# wait for any in-flight harness (the ew0.0 smoke) to exit
while pgrep -f "python.*benchmark_cradle_mother.py" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] smoke clear — starting ew0.4 sweep (taught + no_feed, 6 seeds each)" >> $D/sweep.log

cd $W && PYTHONPATH="$W/src" $PY scripts/benchmark_cradle_mother.py \
  --arms taught,no_feed --trials 6 --seed-base 42 --model mistral-7b \
  --explore-weight 0.4 --timeout-s 7200 \
  --out $D/ew0.4.jsonl --workdir $D/ew0.4_runs < /dev/null >> $D/sweep.log 2>&1

echo "[$(date +%H:%M)] SWEEP COMPLETE" >> $D/sweep.log
$PY $D/../bisect/nomove_rate.py $D/ew0.4_runs >> $D/sweep.log 2>&1
cd $W && PYTHONPATH="$W/src" $PY scripts/analyze_cradle_mother.py --in $D/ew0.4.jsonl --trials 6 >> $D/sweep.log 2>&1
